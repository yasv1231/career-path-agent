from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from auth_utils import hash_password, verify_password
from db_store import DBStore
from workflow.extractor import default_extracted, default_profile
from workflow.goal_constraints import default_goal_constraints
from workflow.graph_chat import build_message_graph
from workflow.messages import serialize_messages_openai
from workflow.state import PROFILE_REQUIRED_FIELDS

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _load_env_file() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    if load_dotenv:
        load_dotenv(env_path, override=False)

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key.lower().startswith("$env:"):
            key = key[5:]
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env_file()

app = FastAPI(title="career-path-agent API", version="2.0.0")

frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
allowed_origins = [origin.strip() for origin in frontend_origin.split(",") if origin.strip()]
if not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_message_graph(enable_memory=True).compile()

_db_path = os.getenv("APP_DB_PATH", "data/app.db")
db = DBStore(_db_path)

_store_lock = threading.Lock()
_sessions: dict[str, dict[str, Any]] = {}


class UserResponse(BaseModel):
    id: str
    email: str
    name: str


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=8)
    name: str = Field(min_length=1, max_length=64)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=1)


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class SessionCreateResponse(BaseModel):
    session_id: str
    assistant_message: str = ""
    messages: list[dict[str, str]] = Field(default_factory=list)
    question_stage: str = ""
    question_phase_complete: bool = False
    conversation_complete: bool = False
    missing_fields: list[str] = Field(default_factory=list)
    profile: dict[str, Any] = Field(default_factory=dict)
    goal_constraints: dict[str, Any] = Field(default_factory=dict)
    collected_points: list[dict[str, str]] = Field(default_factory=list)
    final_plan: dict[str, Any] = Field(default_factory=dict)


class ChatMessageRequest(BaseModel):
    content: str = Field(min_length=1)


class ChatMessageResponse(SessionCreateResponse):
    pass


class ProfileResponse(BaseModel):
    user: UserResponse
    has_plan: bool = False
    conversation_complete: bool = False
    session_id: str | None = None
    profile: dict[str, Any] = Field(default_factory=dict)
    goal_constraints: dict[str, Any] = Field(default_factory=dict)
    strategy_tags: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    collected_points: list[dict[str, str]] = Field(default_factory=list)
    planning_history: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: int | None = None


def _is_valid_email(email: str) -> bool:
    value = email.strip()
    return "@" in value and "." in value.split("@")[-1]


def _parse_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    return parts[1].strip()


def _sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    return str(value)


def _serialize_state(state: dict[str, Any]) -> str:
    snapshot = dict(state)
    snapshot["messages"] = serialize_messages_openai(state.get("messages", []))
    safe = _sanitize_for_json(snapshot)
    return json.dumps(safe, ensure_ascii=False)


def _build_initial_state() -> dict[str, Any]:
    return {
        "messages": [],
        "max_messages": 12,
        "enable_memory_maintain": True,
        "user_state": {
            "profile": default_profile(),
            "goal_constraints": default_goal_constraints(),
            "strategy_tags": [],
            "rag_filters": {},
            "extracted": {},
            "turn_index": 0,
        },
        "progress_state": {
            "question_stage": "collecting",
            "asked_questions": [],
            "attempts": {},
            "required_slots": list(PROFILE_REQUIRED_FIELDS),
            "optional_slots": ["compensation_floor", "work_mode"],
            "pending_slots": list(PROFILE_REQUIRED_FIELDS),
            "current_slot": PROFILE_REQUIRED_FIELDS[0] if PROFILE_REQUIRED_FIELDS else "",
            "current_slot_retries": 0,
            "max_slot_retries": 2,
            "awaiting_confirmation": False,
            "candidate_slot": "",
            "candidate_value": None,
            "candidate_confidence": 0.0,
            "candidate_source": "",
            "question_phase_complete": False,
            "plan_ready": False,
            "last_node": "",
        },
        "memory": default_profile(),
        "goal_constraints": default_goal_constraints(),
        "asked_questions": [],
        "attempts": {},
        "question_stage": "collecting",
        "question_phase_complete": False,
        "conversation_complete": False,
    }


def _last_assistant_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def _build_collected_points(state: dict[str, Any]) -> list[dict[str, str]]:
    points: list[dict[str, str]] = []
    profile = state.get("memory")
    if not isinstance(profile, dict):
        profile = {}

    skills = profile.get("skills") if isinstance(profile.get("skills"), list) else []
    for item in skills[:6]:
        text = str(item).strip()
        if text:
            points.append({"category": "Skill", "text": text})

    for key in ("constraints", "goals"):
        values = profile.get(key) if isinstance(profile.get(key), list) else []
        for item in values[:6]:
            text = str(item).strip()
            if text:
                points.append({"category": "Value", "text": text})

    goal_constraints = state.get("goal_constraints")
    hard_constraints: list[dict[str, Any]] = []
    if isinstance(goal_constraints, dict):
        raw_hard = goal_constraints.get("hard_constraints")
        if isinstance(raw_hard, list):
            hard_constraints = [item for item in raw_hard if isinstance(item, dict)]

    for item in hard_constraints[:6]:
        name = str(item.get("name", "")).strip()
        value = str(item.get("value", "")).strip()
        unit = str(item.get("unit", "")).strip()
        parts = [part for part in (name, value, unit) if part]
        if parts:
            points.append({"category": "Constraint", "text": " ".join(parts)})

    dedup: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for point in points:
        key = (point["category"], point["text"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(point)
    return dedup[:12]


def _snapshot(session_id: str, state: dict[str, Any]) -> ChatMessageResponse:
    messages = serialize_messages_openai(state.get("messages", []))
    profile = state.get("memory")
    if not isinstance(profile, dict):
        profile = default_profile()
    goal_constraints = state.get("goal_constraints")
    if not isinstance(goal_constraints, dict):
        goal_constraints = default_goal_constraints()
    final_plan = state.get("final_plan")
    if not isinstance(final_plan, dict):
        final_plan = {}

    return ChatMessageResponse(
        session_id=session_id,
        assistant_message=_last_assistant_message(messages),
        messages=messages,
        question_stage=str(state.get("question_stage", "")),
        question_phase_complete=bool(state.get("question_phase_complete", False)),
        conversation_complete=bool(state.get("conversation_complete", False)),
        missing_fields=list(state.get("missing_fields", [])),
        profile=profile,
        goal_constraints=goal_constraints,
        collected_points=_build_collected_points(state),
        final_plan=final_plan,
    )


def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = _parse_bearer_token(authorization)
    user = db.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user["token"] = token
    return user


def _load_session_for_user(session_id: str, user_id: str) -> dict[str, Any] | None:
    with _store_lock:
        cached = _sessions.get(session_id)
        if isinstance(cached, dict):
            return cached

    state = db.get_chat_session_state(session_id, user_id)
    if not isinstance(state, dict):
        return None

    with _store_lock:
        _sessions[session_id] = state
    return state


def _build_profile_response(user: dict[str, Any]) -> ProfileResponse:
    sessions = db.list_chat_sessions(str(user["id"]), limit=20)

    planning_history: list[dict[str, Any]] = []
    for item in sessions:
        state = item.get("state", {})
        if not isinstance(state, dict):
            state = {}
        profile = state.get("memory")
        if not isinstance(profile, dict):
            profile = {}

        raw_message = str(item.get("last_assistant_message", "")).strip()
        preview = raw_message.replace("\n", " ")
        if len(preview) > 140:
            preview = f"{preview[:140].rstrip()}..."

        planning_history.append(
            {
                "session_id": str(item.get("id", "")),
                "target_role": str(profile.get("target_role", "")).strip(),
                "question_stage": str(state.get("question_stage", "")).strip(),
                "question_phase_complete": bool(state.get("question_phase_complete", False)),
                "conversation_complete": bool(state.get("conversation_complete", False)),
                "created_at": int(item.get("created_at", 0) or 0),
                "updated_at": int(item.get("updated_at", 0) or 0),
                "preview": preview,
            }
        )

    latest = db.get_latest_chat_session(str(user["id"]))
    if not latest:
        return ProfileResponse(
            user=UserResponse(id=str(user["id"]), email=str(user["email"]), name=str(user["name"])),
            has_plan=False,
            conversation_complete=False,
            profile=default_profile(),
            goal_constraints=default_goal_constraints(),
            strategy_tags=[],
            missing_fields=list(PROFILE_REQUIRED_FIELDS),
            collected_points=[],
            planning_history=planning_history,
            updated_at=None,
        )

    state = latest.get("state", {})
    if not isinstance(state, dict):
        state = {}
    profile = state.get("memory")
    if not isinstance(profile, dict):
        profile = default_profile()
    goal_constraints = state.get("goal_constraints")
    if not isinstance(goal_constraints, dict):
        goal_constraints = default_goal_constraints()
    missing_fields = state.get("missing_fields")
    if not isinstance(missing_fields, list):
        missing_fields = []
    strategy_tags = state.get("strategy_tags")
    if not isinstance(strategy_tags, list):
        strategy_tags = []

    conversation_complete = bool(state.get("conversation_complete", False))
    has_plan = conversation_complete or bool(state.get("question_phase_complete", False))
    return ProfileResponse(
        user=UserResponse(id=str(user["id"]), email=str(user["email"]), name=str(user["name"])),
        has_plan=has_plan,
        conversation_complete=conversation_complete,
        session_id=str(latest.get("id")),
        profile=profile,
        goal_constraints=goal_constraints,
        strategy_tags=[str(item) for item in strategy_tags if str(item).strip()],
        missing_fields=[str(item) for item in missing_fields if str(item).strip()],
        collected_points=_build_collected_points(state),
        planning_history=planning_history,
        updated_at=int(latest.get("updated_at")) if latest.get("updated_at") else None,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest) -> AuthResponse:
    email = payload.email.strip().lower()
    if not _is_valid_email(email):
        raise HTTPException(status_code=422, detail="Invalid email")

    if db.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Email already exists")

    salt, pwd_hash = hash_password(payload.password)
    user = db.create_user(email=email, name=payload.name.strip(), password_salt=salt, password_hash=pwd_hash)
    token = db.create_token(user_id=user["id"])
    return AuthResponse(token=token, user=UserResponse(id=user["id"], email=user["email"], name=user["name"]))


@app.post("/api/v1/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    email = payload.email.strip().lower()
    user = db.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(payload.password, str(user["password_salt"]), str(user["password_hash"])):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = db.create_token(user_id=str(user["id"]))
    return AuthResponse(
        token=token,
        user=UserResponse(id=str(user["id"]), email=str(user["email"]), name=str(user["name"])),
    )


@app.post("/api/v1/auth/logout")
def logout(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, str]:
    token = str(user.get("token", ""))
    if token:
        db.revoke_token(token)
    return {"status": "ok"}


@app.get("/api/v1/auth/me", response_model=UserResponse)
def me(user: dict[str, Any] = Depends(get_current_user)) -> UserResponse:
    return UserResponse(id=str(user["id"]), email=str(user["email"]), name=str(user["name"]))


@app.get("/api/v1/profile/me", response_model=ProfileResponse)
def profile_me(user: dict[str, Any] = Depends(get_current_user)) -> ProfileResponse:
    return _build_profile_response(user)


@app.post("/api/v1/chat/sessions", response_model=SessionCreateResponse)
def create_session(user: dict[str, Any] = Depends(get_current_user)) -> SessionCreateResponse:
    state = _build_initial_state()
    state = graph.invoke(state)
    state_json = _serialize_state(state)

    session_id = db.create_chat_session(user_id=str(user["id"]), state_json=state_json)
    messages = serialize_messages_openai(state.get("messages", []))
    db.append_chat_messages(session_id, messages)

    with _store_lock:
        _sessions[session_id] = state

    return _snapshot(session_id, state)


@app.post("/api/v1/chat/sessions/{session_id}/messages", response_model=ChatMessageResponse)
def chat_with_session(
    session_id: str,
    payload: ChatMessageRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> ChatMessageResponse:
    state = _load_session_for_user(session_id=session_id, user_id=str(user["id"]))
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Message content is required")

    before_messages = serialize_messages_openai(state.get("messages", []))

    messages = list(state.get("messages", []))
    messages.append({"role": "user", "content": content})
    state["messages"] = messages
    state = graph.invoke(state)

    after_messages = serialize_messages_openai(state.get("messages", []))
    delta = after_messages[len(before_messages) :]

    db.update_chat_session_state(session_id=session_id, user_id=str(user["id"]), state_json=_serialize_state(state))
    db.append_chat_messages(session_id=session_id, messages=delta)

    with _store_lock:
        _sessions[session_id] = state

    return _snapshot(session_id, state)


@app.get("/api/v1/chat/sessions/{session_id}", response_model=ChatMessageResponse)
def get_session(session_id: str, user: dict[str, Any] = Depends(get_current_user)) -> ChatMessageResponse:
    state = _load_session_for_user(session_id=session_id, user_id=str(user["id"]))
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _snapshot(session_id, state)


@app.get("/api/v1/chat/sessions/{session_id}/result")
def get_result(session_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    state = _load_session_for_user(session_id=session_id, user_id=str(user["id"]))
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    extracted = state.get("extracted", default_extracted(confidence=0.2))
    evaluation = state.get("evaluation", {})
    return {
        "session_id": session_id,
        "conversation_complete": bool(state.get("conversation_complete", False)),
        "question_phase_complete": bool(state.get("question_phase_complete", False)),
        "question_stage": state.get("question_stage", ""),
        "memory": state.get("memory", {}),
        "goal_constraints": state.get("goal_constraints", {}),
        "strategy_tags": state.get("strategy_tags", []),
        "rag_filters": state.get("rag_filters", {}),
        "missing_fields": state.get("missing_fields", []),
        "extracted": extracted,
        "evaluation": evaluation,
    }
