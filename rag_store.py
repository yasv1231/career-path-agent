from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    HAS_CHROMADB = True
except Exception:
    chromadb = None
    HAS_CHROMADB = False

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    CrossEncoder = None
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False


TOKEN_RE = re.compile(r"[a-z0-9]+")


def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, "r", encoding="utf-8-sig") as file:
        content = file.read()
    return [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _expand_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        source_id = str(doc.get("id", "unknown"))
        base_meta = {
            "source_id": source_id,
            "title": str(doc.get("title", "")),
            "source_title": str(doc.get("title", "")),
            "career": str(doc.get("career", "")),
            "type": str(doc.get("type", "")),
            "resource_type": str(doc.get("resource_type", "")),
            "url": str(doc.get("url", "")),
        }

        raw_chunks = doc.get("chunks")
        chunks: List[Dict[str, Any]] = []
        if isinstance(raw_chunks, list) and raw_chunks:
            for item in raw_chunks:
                if isinstance(item, str):
                    chunks.append({"text": item})
                elif isinstance(item, dict):
                    chunks.append(item)
        else:
            text = str(doc.get("text", ""))
            for chunk in [part.strip() for part in text.split("\n\n") if part.strip()]:
                chunks.append({"text": chunk})

        total = max(len(chunks), 1)
        for idx, chunk in enumerate(chunks, start=1):
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            chunk_id = str(chunk.get("chunk_id", f"{source_id}_c{idx}"))
            expanded.append(
                {
                    **base_meta,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "chunk_total": total,
                    "text": text,
                    "vector_id": f"{source_id}::{chunk_id}",
                }
            )
    return expanded


class RagStore:
    def __init__(
        self,
        documents: List[Dict[str, Any]],
        db_dir: Optional[Path] = None,
        collection_name: str = "career_knowledge",
        embedding_model_name: str = "shibing624/text2vec-base-chinese",
        reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        enable_rerank: bool = False,
    ) -> None:
        self._documents = _expand_documents(documents)
        self._doc_vectors: List[Dict[str, float]] = []
        self._doc_norms: List[float] = []
        self._idf: Dict[str, float] = {}

        self._db_dir = Path(db_dir) if db_dir else Path(__file__).resolve().parent / "data" / "rag_chroma"
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model_name
        self._reranker_model_name = reranker_model_name
        self._enable_rerank = enable_rerank

        self._embedding_model: Any = None
        self._cross_encoder: Any = None
        self._client: Any = None
        self._collection: Any = None

        self._vector_enabled = HAS_CHROMADB and HAS_SENTENCE_TRANSFORMERS and bool(self._documents)

        if self._vector_enabled:
            self._init_vector_store()
        else:
            self._build_lexical_index()

    @classmethod
    def from_path(
        cls,
        path: Path,
        db_dir: Optional[Path] = None,
        collection_name: str = "career_knowledge",
        embedding_model_name: str = "shibing624/text2vec-base-chinese",
        reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        enable_rerank: bool = False,
    ) -> "RagStore":
        if not path.exists():
            return cls(
                [],
                db_dir=db_dir,
                collection_name=collection_name,
                embedding_model_name=embedding_model_name,
                reranker_model_name=reranker_model_name,
                enable_rerank=enable_rerank,
            )

        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            data = []
        return cls(
            data,
            db_dir=db_dir,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            reranker_model_name=reranker_model_name,
            enable_rerank=enable_rerank,
        )

    def embed_chunk(self, chunk: str) -> List[float]:
        if not self._embedding_model:
            raise RuntimeError("Embedding model is not initialized")
        embedding = self._embedding_model.encode(chunk, normalize_embeddings=True)
        return embedding.tolist()

    def _manifest_path(self) -> Path:
        return self._db_dir / f"{self._collection_name}.manifest.json"

    def _fingerprint(self) -> str:
        payload = [
            {
                "id": doc.get("vector_id"),
                "text": doc.get("text"),
                "career": doc.get("career"),
                "type": doc.get("type"),
                "resource_type": doc.get("resource_type"),
                "url": doc.get("url"),
            }
            for doc in self._documents
        ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _load_manifest(self) -> Dict[str, Any]:
        path = self._manifest_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _init_vector_store(self) -> None:
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_model = SentenceTransformer(self._embedding_model_name)
        if self._enable_rerank:
            try:
                self._cross_encoder = CrossEncoder(self._reranker_model_name)
            except Exception:
                self._cross_encoder = None

        self._client = chromadb.PersistentClient(path=str(self._db_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        current_count = 0
        try:
            current_count = int(self._collection.count())
        except Exception:
            current_count = 0

        manifest = self._load_manifest()
        fingerprint = self._fingerprint()
        needs_rebuild = (
            manifest.get("fingerprint") != fingerprint
            or manifest.get("count") != len(self._documents)
            or current_count != len(self._documents)
        )

        if not needs_rebuild:
            return

        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids: List[str] = []
        docs: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for doc in self._documents:
            ids.append(str(doc.get("vector_id")))
            docs.append(str(doc.get("text", "")))
            metadatas.append(
                {
                    "source_id": str(doc.get("source_id", "")),
                    "source_title": str(doc.get("source_title", "")),
                    "title": str(doc.get("title", "")),
                    "career": str(doc.get("career", "")),
                    "type": str(doc.get("type", "")),
                    "resource_type": str(doc.get("resource_type", "")),
                    "url": str(doc.get("url", "")),
                    "chunk_id": str(doc.get("chunk_id", "")),
                    "chunk_index": int(doc.get("chunk_index", 0)),
                    "chunk_total": int(doc.get("chunk_total", 0)),
                }
            )

        batch_size = 64
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            batch_embeddings = [self.embed_chunk(text) for text in batch_docs]
            self._collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )

        self._save_manifest({"fingerprint": fingerprint, "count": len(self._documents)})

    def _build_lexical_index(self) -> None:
        doc_tokens: List[List[str]] = []
        df: Dict[str, int] = {}

        for doc in self._documents:
            text = " ".join(
                str(doc.get(field, ""))
                for field in ("title", "text", "career", "type", "resource_type")
            )
            tokens = _tokenize(text)
            doc_tokens.append(tokens)
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        n_docs = max(len(self._documents), 1)
        self._idf = {
            token: math.log((1 + n_docs) / (1 + freq)) + 1.0 for token, freq in df.items()
        }

        for tokens in doc_tokens:
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            vec: Dict[str, float] = {}
            for token, count in tf.items():
                vec[token] = (count / max(len(tokens), 1)) * self._idf.get(token, 0.0)
            norm = math.sqrt(sum(v * v for v in vec.values()))
            self._doc_vectors.append(vec)
            self._doc_norms.append(norm if norm > 0 else 1.0)

    def _match_filters(self, doc: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if expected is None:
                continue
            actual = doc.get(key)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            else:
                if str(actual).lower() != str(expected).lower():
                    return False
        return True

    def _build_chroma_where(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
        where: Dict[str, Any] = {}
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                where[key] = {"$in": [str(v) for v in value]}
            else:
                where[key] = str(value)
        return where or None

    def _vector_search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if not self._collection:
            return []

        query_embedding = self.embed_chunk(query)
        n_candidates = max(top_k * 3, top_k)
        where = self._build_chroma_where(filters)
        payload = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            where=where,
        )

        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        ids = (payload.get("ids") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]

        candidates: List[Dict[str, Any]] = []
        for idx, text in enumerate(documents):
            metadata = dict((metadatas[idx] or {})) if idx < len(metadatas) else {}
            doc_id = str(ids[idx]) if idx < len(ids) else ""
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            score = max(0.0, 1.0 - distance)

            doc = {
                "source_id": str(metadata.get("source_id", "")),
                "source_title": str(metadata.get("source_title", metadata.get("title", ""))),
                "title": str(metadata.get("title", "")),
                "career": str(metadata.get("career", "")),
                "type": str(metadata.get("type", "")),
                "resource_type": str(metadata.get("resource_type", "")),
                "url": str(metadata.get("url", "")),
                "chunk_id": str(metadata.get("chunk_id", doc_id)),
                "text": str(text),
            }
            if score < min_score:
                continue
            if not self._match_filters(doc, filters):
                continue
            candidates.append({"score": score, "doc": doc})

        if self._enable_rerank and self._cross_encoder and candidates:
            pairs = [(query, item["doc"]["text"]) for item in candidates]
            rerank_scores = self._cross_encoder.predict(pairs)
            for item, rerank_score in zip(candidates, rerank_scores):
                item["score"] = float(rerank_score)

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:top_k]

    def _lexical_search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if not self._documents:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        qvec: Dict[str, float] = {}
        for token, count in tf.items():
            qvec[token] = (count / len(tokens)) * self._idf.get(token, 0.0)
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0

        scored: List[tuple[float, Dict[str, Any]]] = []
        for doc, dvec, dnorm in zip(self._documents, self._doc_vectors, self._doc_norms):
            if not self._match_filters(doc, filters):
                continue
            score = 0.0
            for token, qval in qvec.items():
                score += qval * dvec.get(token, 0.0)
            score /= (dnorm * qnorm)
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: List[Dict[str, Any]] = []
        for score, doc in scored[:top_k]:
            if score <= min_score:
                continue
            results.append({"score": score, "doc": doc})
        return results

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if self._vector_enabled:
            return self._vector_search(query, top_k=top_k, filters=filters, min_score=min_score)
        return self._lexical_search(query, top_k=top_k, filters=filters, min_score=min_score)
