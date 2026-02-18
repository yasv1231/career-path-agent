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
    memory: Dict[str, Any]
    profile_update: Dict[str, Any]
    goal_constraints: Dict[str, Any]
    goal_constraints_update: Dict[str, Any]
    strategy_tags: List[str]
    rag_filters: Dict[str, Any]
    missing_fields: List[str]
    next_question: str
    asked_questions: List[str]
    attempts: Dict[str, int]
    conversation_complete: bool
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

PROFILE_SCHEMA_VERSION = "profile_v1"
PROFILE_REQUIRED_FIELDS = [
    "education",
    "skills",
    "interests",
    "hours_per_week",
    "experience_level",
    "location",
    "timeline_weeks",
]
PROFILE_SCHEMA_V1 = {
    "type": "object",
    "required": [
        "schema_version",
        "education",
        "skills",
        "interests",
        "target_role",
        "hours_per_week",
        "experience_level",
        "constraints",
        "location",
        "timeline_weeks",
        "industry",
        "goals",
    ],
    "properties": {
        "schema_version": {"type": "string", "enum": [PROFILE_SCHEMA_VERSION]},
        "education": {"type": ["string", "null"]},
        "skills": {"type": "array", "items": {"type": "string"}},
        "interests": {"type": "array", "items": {"type": "string"}},
        "target_role": {"type": ["string", "null"]},
        "hours_per_week": {"type": ["number", "null"]},
        "experience_level": {"type": ["string", "null"]},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "location": {"type": ["string", "null"]},
        "timeline_weeks": {"type": ["number", "null"]},
        "industry": {"type": ["string", "null"]},
        "goals": {"type": "array", "items": {"type": "string"}},
    },
}
