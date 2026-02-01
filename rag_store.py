import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_RE = re.compile(r"[^.!?;]+[.!?;]?")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _chunk_text(text: str, max_chars: int = 260, overlap: int = 40) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    sentences = [s.strip() for s in SENTENCE_RE.findall(raw) if s.strip()]
    if not sentences:
        return [raw]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        add_len = len(sentence) + (1 if current else 0)
        if current and current_len + add_len > max_chars:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
            if overlap > 0 and chunk:
                tail = chunk[-overlap:]
                current = [tail]
                current_len = len(tail)
            else:
                current = []
                current_len = 0
        current.append(sentence)
        current_len += add_len

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _expand_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        source_id = str(doc.get("id", "unknown"))
        base_meta = {
            "source_id": source_id,
            "title": doc.get("title", ""),
            "source_title": doc.get("title", ""),
            "career": doc.get("career", ""),
            "type": doc.get("type", ""),
            "resource_type": doc.get("resource_type", ""),
            "url": doc.get("url", ""),
        }

        raw_chunks = doc.get("chunks")
        if isinstance(raw_chunks, list) and raw_chunks:
            chunks = []
            for item in raw_chunks:
                if isinstance(item, str):
                    chunks.append({"text": item})
                elif isinstance(item, dict):
                    chunks.append(item)
        else:
            chunks = [{"text": txt} for txt in _chunk_text(str(doc.get("text", "")))]

        total = max(len(chunks), 1)
        for idx, chunk in enumerate(chunks, start=1):
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            expanded.append(
                {
                    **base_meta,
                    "chunk_id": str(chunk.get("chunk_id", f"{source_id}_c{idx}")),
                    "chunk_index": idx,
                    "chunk_total": total,
                    "text": text,
                }
            )
    return expanded


class RagStore:
    def __init__(self, documents: List[Dict[str, Any]]) -> None:
        self._documents = _expand_documents(documents)
        self._doc_vectors: List[Dict[str, float]] = []
        self._doc_norms: List[float] = []
        self._idf: Dict[str, float] = {}
        self._build_index()

    @classmethod
    def from_path(cls, path: Path) -> "RagStore":
        if not path.exists():
            return cls([])

        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            return cls([])
        return cls(data)

    def _build_index(self) -> None:
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
                vec[token] = (count / len(tokens)) * self._idf.get(token, 0.0)
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

    def search(
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
