from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_resources import load_rag_resources
from rag_store import RagStore


_RAG_RESOURCES = load_rag_resources()
CAREER_ALIASES: Dict[str, List[str]] = _RAG_RESOURCES.get("career_aliases", {})


def _confidence_label(score: float, thresholds: Dict[str, float]) -> str:
    low = float(thresholds.get("low", 0.08))
    mid = float(thresholds.get("mid", 0.14))
    if score >= mid:
        return "high"
    if score >= low:
        return "medium"
    return "low"


def _normalize_career(career: str) -> str:
    text = (career or "").lower().strip()
    if not text:
        return "unknown"

    for key, aliases in CAREER_ALIASES.items():
        for alias in aliases:
            if text == alias:
                return key
    for key, aliases in CAREER_ALIASES.items():
        for alias in aliases:
            if alias in text:
                return key
    return text


def _format_evidence(doc: Dict[str, str]) -> str:
    source_id = doc.get("source_id", "source")
    chunk_id = doc.get("chunk_id", "chunk")
    title = doc.get("source_title") or doc.get("title") or "source"
    return f"[{source_id}:{chunk_id}] {title}"


class RagAgent:
    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        resources_path: Optional[Path] = None,
    ) -> None:
        self._resources = load_rag_resources(resources_path)
        self._thresholds = self._resources.get("confidence_thresholds", {})
        self._sections = self._resources.get("sections", [])
        self._section_map = {section.get("name"): section for section in self._sections if section.get("name")}

        vector_cfg = self._resources.get("vector_store", {})

        if corpus_path is None:
            corpus_path = Path(__file__).resolve().parent / "data" / "rag_corpus.json"

        db_dir = Path(vector_cfg.get("db_dir", "data/rag_chroma"))
        if not db_dir.is_absolute():
            db_dir = Path(__file__).resolve().parent / db_dir

        self._store = RagStore.from_path(
            corpus_path,
            db_dir=db_dir,
            collection_name=str(vector_cfg.get("collection_name", "career_knowledge")),
            embedding_model_name=str(
                vector_cfg.get("embedding_model", "shibing624/text2vec-base-chinese")
            ),
            reranker_model_name=str(
                vector_cfg.get("reranker_model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
            ),
            enable_rerank=bool(vector_cfg.get("enable_rerank", False)),
        )

    def get_context_format(self) -> Dict[str, str]:
        context_cfg = self._resources.get("context_format", {})
        return {
            "context_prefix": str(context_cfg.get("context_prefix", "RAG_CONTEXT:")),
            "citations_label": str(context_cfg.get("citations_label", "Citations")),
        }

    def _retrieve(
        self,
        career: str,
        section: Dict[str, Any],
        rag_filters: Dict[str, Any] | None = None,
        query_hints: List[str] | None = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], float]:
        career_key = _normalize_career(career)
        doc_type = str(section.get("doc_type", "")).strip()
        query_hint = str(section.get("query_hint", "")).strip()
        top_k = int(section.get("top_k", 3))
        hint_text = " ".join(str(item).strip() for item in (query_hints or []) if str(item).strip())
        query = f"{career_key} {query_hint} {hint_text}".strip()

        filters = {"career": career_key}
        if doc_type:
            filters["type"] = doc_type
        for key, value in (rag_filters or {}).items():
            # Keep doc_type fixed by section config.
            if key == "type":
                continue
            if key == "resource_type" and doc_type != "resource":
                continue
            filters[key] = value

        results = self._store.search(query, top_k=top_k, filters=filters)
        if not results:
            return ([], [], 0.0)

        item_text_key = str(section.get("item_text_key", "text"))
        items: List[Dict[str, Any]] = []
        evidence: List[Dict[str, str]] = []
        top_score = float(results[0]["score"])

        for result in results:
            doc = result["doc"]
            title = str(doc.get("title", "Knowledge"))
            text = str(doc.get("text", ""))
            item: Dict[str, Any] = {"title": title, item_text_key: text}
            for extra_key in ("url", "resource_type"):
                value = str(doc.get(extra_key, "")).strip()
                if value:
                    item[extra_key] = value
            items.append(item)
            evidence.append(
                {
                    "ref": _format_evidence(doc),
                    "score": f"{result['score']:.3f}",
                }
            )

        return items, evidence, top_score

    def _retrieve_by_name(
        self,
        career: str,
        section_name: str,
        rag_filters: Dict[str, Any] | None = None,
        query_hints: List[str] | None = None,
    ) -> Dict[str, Any]:
        section = self._section_map.get(section_name)
        if not section:
            return {
                "status": "no_data",
                "confidence": "low",
                "items": [],
                "citations": [],
                "action": "clarify",
                "refusal": "insufficient_evidence",
                "clarify": "No retrieval section configured.",
            }

        items, evidence, top_score = self._retrieve(
            career,
            section,
            rag_filters=rag_filters or {},
            query_hints=query_hints or [],
        )
        confidence = _confidence_label(top_score, self._thresholds)
        if not items:
            status = "no_data"
        elif confidence == "low":
            status = "low_confidence"
        else:
            status = "ok"

        return {
            "status": status,
            "confidence": confidence,
            "items": items,
            "citations": evidence,
            "action": "clarify" if status != "ok" else "answer",
            "refusal": "insufficient_evidence" if status != "ok" else "",
            "clarify": str(section.get("clarify_message", "")) if status != "ok" else "",
        }

    def retrieve_sections(
        self,
        career: str,
        rag_filters: Dict[str, Any] | None = None,
        query_hints: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for section in self._sections:
            section_name = str(section.get("name", ""))
            if not section_name:
                continue
            result = self._retrieve_by_name(
                career,
                section_name,
                rag_filters=rag_filters or {},
                query_hints=query_hints or [],
            )
            payload.append(
                {
                    "name": section_name,
                    "label": str(section.get("label", section_name)),
                    "item_text_key": str(section.get("item_text_key", "text")),
                    "result": result,
                }
            )
        return payload

    def job_confirmation(
        self,
        career: str,
        rag_filters: Dict[str, Any] | None = None,
        query_hints: List[str] | None = None,
    ) -> Dict[str, Any]:
        return self._retrieve_by_name(
            career,
            "job_profile",
            rag_filters=rag_filters or {},
            query_hints=query_hints or [],
        )

    def competency_model(
        self,
        career: str,
        rag_filters: Dict[str, Any] | None = None,
        query_hints: List[str] | None = None,
    ) -> Dict[str, Any]:
        return self._retrieve_by_name(
            career,
            "competency",
            rag_filters=rag_filters or {},
            query_hints=query_hints or [],
        )

    def resource_retrieval(
        self,
        career: str,
        top_k: int = 6,
        rag_filters: Dict[str, Any] | None = None,
        query_hints: List[str] | None = None,
    ) -> Dict[str, Any]:
        # top_k is kept for backwards compatibility; runtime config controls default retrieval size.
        _ = top_k
        return self._retrieve_by_name(
            career,
            "resource",
            rag_filters=rag_filters or {},
            query_hints=query_hints or [],
        )
