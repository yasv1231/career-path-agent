from __future__ import annotations

from typing import Any, Dict, List

try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
    HAS_LANGCHAIN_MESSAGES = True
except Exception:
    SystemMessage = None
    HumanMessage = None
    AIMessage = None
    BaseMessage = None
    HAS_LANGCHAIN_MESSAGES = False


def _openai_role_from_type(msg_type: str | None) -> str:
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"
    if msg_type == "system":
        return "system"
    return "assistant"


def openai_from_lc_message(message: Any) -> Dict[str, str]:
    msg_type = getattr(message, "type", None)
    role = _openai_role_from_type(msg_type)
    content = getattr(message, "content", "")
    return {"role": role, "content": str(content)}


def lc_from_openai_message(message: Dict[str, Any]) -> Any:
    role = str(message.get("role", "user")).lower()
    content = str(message.get("content", ""))
    if not HAS_LANGCHAIN_MESSAGES:
        return {"role": role, "content": content}
    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def is_openai_message(obj: Any) -> bool:
    return isinstance(obj, dict) and "role" in obj and "content" in obj


def normalize_messages(messages: Any) -> List[Any]:
    if messages is None:
        return []
    if not isinstance(messages, list):
        return []
    normalized: List[Any] = []
    for msg in messages:
        if HAS_LANGCHAIN_MESSAGES and BaseMessage and isinstance(msg, BaseMessage):
            normalized.append(msg)
            continue
        if is_openai_message(msg):
            normalized.append(lc_from_openai_message(msg))
            continue
    return normalized


def serialize_messages_openai(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []
    out: List[Dict[str, str]] = []
    for msg in messages:
        if HAS_LANGCHAIN_MESSAGES and BaseMessage and isinstance(msg, BaseMessage):
            out.append(openai_from_lc_message(msg))
        elif is_openai_message(msg):
            out.append({"role": str(msg["role"]), "content": str(msg["content"])})
    return out