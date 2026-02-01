from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_store import RagStore


CAREER_ALIASES = {
    "data analyst": [
        "data analyst",
        "da",
        "business analyst",
        "bi analyst",
        "analytics analyst",
        "reporting analyst",
    ],
    "data scientist": [
        "data scientist",
        "ds",
        "applied scientist",
        "research scientist",
    ],
    "ml engineer": [
        "ml engineer",
        "machine learning engineer",
        "ml ops",
        "mlops engineer",
        "ml",
        "machine learning",
    ],
    "software engineer": [
        "software engineer",
        "software developer",
        "backend engineer",
        "frontend engineer",
        "full stack engineer",
        "developer",
        "swe",
    ],
}

LOW_CONF_THRESHOLD = 0.08
MID_CONF_THRESHOLD = 0.14


def _confidence_label(score: float) -> str:
    if score >= MID_CONF_THRESHOLD:
        return "high"
    if score >= LOW_CONF_THRESHOLD:
        return "medium"
    return "low"


def _normalize_career(career: str) -> str:
    text = (career or "").lower()
    for key, aliases in CAREER_ALIASES.items():
        for alias in aliases:
            if text == alias:
                return key
    for key, aliases in CAREER_ALIASES.items():
        for alias in aliases:
            if alias in text:
                return key
    return text.strip() or "unknown"


def _format_evidence(doc: Dict[str, str]) -> str:
    source_id = doc.get("source_id", "source")
    chunk_id = doc.get("chunk_id", "chunk")
    title = doc.get("source_title") or doc.get("title") or "source"
    return f"[{source_id}:{chunk_id}] {title}"


class RagAgent:
    """
    RagAgent:
    - Retrieves job confirmation, competency model, and course/project resources.
    """

    def __init__(self, corpus_path: Optional[Path] = None) -> None:
        if corpus_path is None:
            corpus_path = Path(__file__).resolve().parent / "data" / "rag_corpus.json"
        self._store = RagStore.from_path(corpus_path)

    def _retrieve(
        self,
        career: str,
        doc_type: str,
        query_hint: str,
        top_k: int = 3,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], float]:
        career_key = _normalize_career(career)
        query = f"{career_key} {query_hint}".strip()
        results = self._store.search(
            query,
            top_k=top_k,
            filters={"type": doc_type, "career": career_key},
        )

        if not results:
            return ([], [], 0.0)

        items: List[Dict[str, str]] = []
        evidence: List[Dict[str, str]] = []
        top_score = results[0]["score"]
        for result in results:
            doc = result["doc"]
            title = doc.get("title", "Role summary")
            text = doc.get("text", "")
            items.append({"title": title, "text": text})
            evidence.append(
                {
                    "ref": _format_evidence(doc),
                    "score": f"{result['score']:.3f}",
                }
            )
        return items, evidence, float(top_score)

    def job_confirmation(self, career: str) -> Dict[str, Any]:
        career_key = _normalize_career(career)
        items, evidence, top_score = self._retrieve(
            career_key,
            "job_profile",
            "role responsibilities overview",
            top_k=2,
        )
        confidence = _confidence_label(top_score)
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
            "clarify": (
                "Please confirm the exact role title or provide more profile details."
                if status != "ok"
                else ""
            ),
        }

    def competency_model(self, career: str) -> Dict[str, Any]:
        career_key = _normalize_career(career)
        items, evidence, top_score = self._retrieve(
            career_key,
            "competency",
            "competency model skills behaviors",
            top_k=2,
        )
        confidence = _confidence_label(top_score)
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
            "clarify": (
                "Low confidence. Confirm the target role or choose a close alternative."
                if status != "ok"
                else ""
            ),
        }

    def resource_retrieval(self, career: str, top_k: int = 6) -> Dict[str, Any]:
        career_key = _normalize_career(career)
        query = f"{career_key} courses projects resources"
        results = self._store.search(query, top_k=top_k, filters={"type": "resource", "career": career_key})
        resources: List[Dict[str, str]] = []
        top_score = results[0]["score"] if results else 0.0
        for result in results:
            doc = result["doc"]
            confidence = _confidence_label(result["score"])
            resources.append(
                {
                    "title": str(doc.get("title", "Resource")),
                    "notes": str(doc.get("text", "")),
                    "url": str(doc.get("url", "")),
                    "resource_type": str(doc.get("resource_type", "resource")),
                    "confidence": confidence,
                    "citations": [
                        {"ref": _format_evidence(doc), "score": f"{result['score']:.3f}"}
                    ],
                }
            )
        status = "ok"
        if not resources:
            status = "no_data"
        elif _confidence_label(top_score) == "low":
            status = "low_confidence"
        return {
            "status": status,
            "confidence": _confidence_label(top_score),
            "items": resources,
            "action": "clarify" if status != "ok" else "answer",
            "refusal": "insufficient_evidence" if status != "ok" else "",
            "clarify": (
                "Low confidence. Share preferred domain or platform to narrow resources."
                if status != "ok"
                else ""
            ),
        }
