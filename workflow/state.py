from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict, Any


class ConversationState(TypedDict, total=False):
    profile: Dict[str, Any]
    has_last_profile: bool
    reuse_last_profile: bool
    fields_to_update: List[str]
    career: str
    career_source: str
    career_reasoning: str
    courses: List[str]
    roadmap: str
    evaluation: str
    free_courses: List[Tuple[str, str]]
    job_confirmation: Dict[str, Any]
    competency_model: Dict[str, Any]
    rag_resources: Dict[str, Any]
    action: str
    llm_enabled: bool


class ChatConversationState(TypedDict, total=False):
    messages: List[Any]
    last_answer: str
    extracted: Dict[str, Any]
    route: str
    route_reasons: List[str]
    rag_context: str
    evaluation: Dict[str, Any]
    llm_enabled: bool
    enable_memory_maintain: bool
    max_messages: int


REQUIRED_FIELDS = ["education", "favorites", "skills", "interest", "hours_per_week"]

EXTRACT_SCHEMA_VERSION = "v1"
EXTRACT_ALLOWED_INTENTS = {"ask", "request", "inform", "confirm", "refine", "other"}
EXTRACT_SCHEMA_V1 = {
    "type": "object",
    "required": ["schema_version", "task", "intent", "entities", "constraints", "actions", "confidence"],
    "properties": {
        "schema_version": {"type": "string", "enum": [EXTRACT_SCHEMA_VERSION]},
        "task": {"type": "string"},
        "intent": {"type": "string", "enum": sorted(EXTRACT_ALLOWED_INTENTS)},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "text"],
                "properties": {
                    "type": {"type": "string"},
                    "text": {"type": "string"},
                    "normalized": {"type": "string"},
                },
            },
        },
        "constraints": {"type": "array", "items": {"type": "string"}},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action", "target"],
                "properties": {
                    "action": {"type": "string"},
                    "target": {"type": "string"},
                    "params": {"type": "object"},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}
