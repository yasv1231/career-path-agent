from __future__ import annotations

from functools import lru_cache
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


@lru_cache(maxsize=1)
def _get_rag_agent() -> RagAgent:
    return RagAgent()


def build_rag_context(
    career: str,
    rag_agent: RagAgent,
    rag_filters: Dict[str, Any] | None = None,
    query_hints: List[str] | None = None,
) -> Dict[str, Any]:
    lines: List[str] = []
    citations: List[str] = []

    for section in rag_agent.retrieve_sections(
        career,
        rag_filters=rag_filters or {},
        query_hints=query_hints or [],
    ):
        result = section.get("result", {})
        items = result.get("items", [])
        if not items:
            continue

        label = section.get("label", section.get("name", "Knowledge"))
        item_text_key = section.get("item_text_key", "text")
        lines.append(f"{label}:")
        for item in items:
            text = item.get(item_text_key, "")
            suffix = f" ({item.get('url')})" if item.get("url") else ""
            lines.append(f"- {item.get('title', '')}: {text}{suffix}")

        for citation in result.get("citations", []):
            ref = citation.get("ref", "")
            if ref:
                citations.append(ref)

    context = "\n".join(lines).strip()
    return {"context": context, "citations": citations}


def maybe_attach_rag_context(
    messages: List[Any],
    route: str,
    rag_filters: Dict[str, Any] | None = None,
    query_hints: List[str] | None = None,
) -> Dict[str, Any]:
    if route != "rag":
        return {"messages": messages, "rag_context": ""}

    text = "\n".join(
        item.get("content", "") for item in serialize_messages_openai(messages) if item.get("role") == "user"
    )
    career = _infer_career_from_text(text)
    rag_agent = _get_rag_agent()
    payload = build_rag_context(
        career,
        rag_agent,
        rag_filters=rag_filters or {},
        query_hints=query_hints or [],
    )

    if not payload["context"]:
        return {"messages": messages, "rag_context": ""}

    context_format = rag_agent.get_context_format()
    rag_text = payload["context"]
    if payload["citations"]:
        citations_label = context_format.get("citations_label", "Citations")
        rag_text = f"{rag_text}\n{citations_label}: {', '.join(payload['citations'])}"

    context_prefix = context_format.get("context_prefix", "RAG_CONTEXT:")

    if HAS_LANGCHAIN_MESSAGES and SystemMessage:
        messages = [SystemMessage(content=f"{context_prefix}\n{rag_text}")] + list(messages)
    else:
        messages = [{"role": "system", "content": f"{context_prefix}\n{rag_text}"}] + list(messages)

    return {"messages": messages, "rag_context": rag_text}
