from __future__ import annotations

from typing import Any, Dict, List

from rag_agent import RagAgent, CAREER_ALIASES, _normalize_career

try:
    from langchain_core.messages import SystemMessage, BaseMessage
    HAS_LANGCHAIN_MESSAGES = True
except Exception:
    SystemMessage = None
    BaseMessage = None
    HAS_LANGCHAIN_MESSAGES = False

from .messages import serialize_messages_openai


def _infer_career_from_text(text: str) -> str:
    lowered = (text or "").lower()
    for career, aliases in CAREER_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                return career
    return _normalize_career(lowered)


def build_rag_context(career: str, rag_agent: RagAgent) -> Dict[str, Any]:
    job = rag_agent.job_confirmation(career)
    competency = rag_agent.competency_model(career)
    resources = rag_agent.resource_retrieval(career)

    lines: List[str] = []
    citations: List[str] = []

    if job.get("items"):
        lines.append("Job Profile:")
        for item in job.get("items", []):
            lines.append(f"- {item.get('title','')}: {item.get('text','')}")
        citations.extend([c.get("ref", "") for c in job.get("citations", []) if c.get("ref")])

    if competency.get("items"):
        lines.append("Competency Model:")
        for item in competency.get("items", []):
            lines.append(f"- {item.get('title','')}: {item.get('text','')}")
        citations.extend([c.get("ref", "") for c in competency.get("citations", []) if c.get("ref")])

    if resources.get("items"):
        lines.append("Resources:")
        for item in resources.get("items", []):
            lines.append(f"- {item.get('title','')}: {item.get('notes','')}")
        for item in resources.get("items", []):
            for citation in item.get("citations", []):
                ref = citation.get("ref", "")
                if ref:
                    citations.append(ref)

    context = "\n".join(lines).strip()
    return {"context": context, "citations": citations}


def maybe_attach_rag_context(messages: List[Any], route: str) -> Dict[str, Any]:
    if route != "rag":
        return {"messages": messages, "rag_context": ""}

    text = "\n".join(
        item.get("content", "") for item in serialize_messages_openai(messages) if item.get("role") == "user"
    )
    career = _infer_career_from_text(text)
    rag_agent = RagAgent()
    payload = build_rag_context(career, rag_agent)

    if not payload["context"]:
        return {"messages": messages, "rag_context": ""}

    rag_text = payload["context"]
    if payload["citations"]:
        rag_text = f"{rag_text}\nCitations: {', '.join(payload['citations'])}"

    if HAS_LANGCHAIN_MESSAGES and SystemMessage:
        messages = [SystemMessage(content=f"RAG_CONTEXT:\n{rag_text}")] + list(messages)
    else:
        messages = [{"role": "system", "content": f"RAG_CONTEXT:\n{rag_text}"}] + list(messages)

    return {"messages": messages, "rag_context": rag_text}