from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from langsmith_integration import get_chat_model


embedding_model = None
cross_encoder = None


def _ensure_models() -> None:
    global embedding_model, cross_encoder
    if embedding_model is not None and cross_encoder is not None:
        return
    from sentence_transformers import CrossEncoder, SentenceTransformer

    embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, "r", encoding="utf-8-sig") as file:
        content = file.read()
    return [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]


def embed_chunk(chunk: str) -> List[float]:
    _ensure_models()
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()


def save_embeddings(chunks: List[str], embeddings: List[List[float]], chromadb_collection, source_id: str) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{source_id}:{i}"],
            metadatas=[{"source": source_id, "chunk_id": str(i)}],
        )


def retrieve(query: str, top_k: int, chromadb_collection) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return (results.get("documents") or [[]])[0]


def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    _ensure_models()
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)
    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda item: item[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks[:top_k]]


def generate(query: str, chunks: List[str]) -> str:
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: {query}

相关片段:
{"\n\n".join(chunks)}

请基于上述内容作答，不要编造信息。"""

    llm = get_chat_model()
    if not llm:
        return f"[LLM unavailable]\n\n{prompt}"

    response = llm.invoke(prompt)
    return getattr(response, "content", None) or str(response)


def _collect_docs(docs_dir: Path) -> List[Path]:
    files: List[Path] = []
    for suffix in ("*.md", "*.txt"):
        files.extend(sorted(docs_dir.rglob(suffix)))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-style RAG KB program (adapted from VideoCode notebook).")
    parser.add_argument("--docs-dir", default="data/knowledge_docs", help="Directory containing .md/.txt source docs.")
    parser.add_argument("--query", required=True, help="Question for retrieval.")
    parser.add_argument("--db-dir", default="data/reference_rag_db", help="Chroma DB directory.")
    parser.add_argument("--collection", default="reference_demo", help="Chroma collection name.")
    parser.add_argument("--retrieve-k", type=int, default=8, help="Top-K retrieval candidates.")
    parser.add_argument("--rerank-k", type=int, default=3, help="Top-K chunks after rerank.")
    parser.add_argument("--skip-generate", action="store_true", help="Only run retrieval + rerank.")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    try:
        import chromadb
    except Exception as exc:
        raise RuntimeError(
            "chromadb is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    chromadb_client = chromadb.PersistentClient(path=str(Path(args.db_dir)))
    try:
        chromadb_client.delete_collection(name=args.collection)
    except Exception:
        pass
    chromadb_collection = chromadb_client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: List[str] = []
    for doc_path in _collect_docs(docs_dir):
        chunks = split_into_chunks(str(doc_path))
        if not chunks:
            continue
        embeddings = [embed_chunk(chunk) for chunk in chunks]
        save_embeddings(chunks, embeddings, chromadb_collection, source_id=doc_path.stem)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError(f"No chunks found in {docs_dir}. Add .md/.txt content first.")

    retrieved_chunks = retrieve(args.query, args.retrieve_k, chromadb_collection)
    reranked_chunks = rerank(args.query, retrieved_chunks, args.rerank_k)

    print("\n[Retrieved]")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"[{i}] {chunk}\n")

    print("\n[Reranked]")
    for i, chunk in enumerate(reranked_chunks):
        print(f"[{i}] {chunk}\n")

    if not args.skip_generate:
        print("\n[Answer]")
        answer = generate(args.query, reranked_chunks)
        print(answer)


if __name__ == "__main__":
    main()
