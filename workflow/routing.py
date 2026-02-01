from __future__ import annotations

from typing import Any, Dict, List

try:
    from langchain_core.messages import BaseMessage
    HAS_LANGCHAIN_MESSAGES = True
except Exception:
    BaseMessage = None
    HAS_LANGCHAIN_MESSAGES = False

RAG_KEYWORDS = [
    "according to",
    "cite",
    "citation",
    "source",
    "docs",
    "documentation",
    "policy",
    "latest",
    "current",
    "today",
]


def _last_user_text(messages: List[Any]) -> str:
    for msg in reversed(messages):
        if HAS_LANGCHAIN_MESSAGES and BaseMessage and isinstance(msg, BaseMessage):
            if getattr(msg, "type", "") == "human":
                return str(getattr(msg, "content", ""))
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def decide_route(messages: List[Any]) -> Dict[str, Any]:
    text = _last_user_text(messages).lower()
    reasons: List[str] = []
    for keyword in RAG_KEYWORDS:
        if keyword in text:
            reasons.append(f"keyword:{keyword}")

    if reasons:
        return {"route": "rag", "confidence": 0.7, "reasons": reasons}

    # Default to direct unless explicit grounding is needed.
    return {"route": "direct", "confidence": 0.6, "reasons": ["no_rag_triggers"]}


def route_to_state(messages: List[Any]) -> Dict[str, Any]:
    decision = decide_route(messages)
    return {
        "route": decision.get("route", "direct"),
        "route_reasons": list(decision.get("reasons", [])),
    }
