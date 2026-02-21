import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

try:
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN_OPENAI = True
except Exception:
    ChatOpenAI = None
    HAS_LANGCHAIN_OPENAI = False


def configure_langsmith(project: str | None = None) -> dict[str, Any]:
    # Backward-compatible interface retained for existing callers.
    project_name = project or "career-path-agent"
    return {
        "has_api_key": False,
        "project": project_name,
        "tracing_enabled": False,
        "provider": "stdout",
    }


def get_chat_model() -> Any | None:
    if not HAS_LANGCHAIN_OPENAI:
        print("[LLM] langchain_openai not available; skipping LLM node.")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[LLM] OPENAI_API_KEY not set; skipping LLM node.")
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    temp_raw = os.getenv("OPENAI_TEMPERATURE", "0")
    try:
        temperature = float(temp_raw)
    except ValueError:
        temperature = 0.0

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    try:
        chat = ChatOpenAI(**kwargs)
        print(f"[LLM] Chat model ready: model={model} temperature={temperature}")
        return chat
    except Exception as e:
        print(f"[LLM] Failed to initialize chat model: {e}")
        return None


class StdoutLogger:
    def __init__(self, project: str | None = None):
        self.project = project or "career-path-agent"
        self.enabled = True
        self.run_id = None

    def _safe_uuid_str(self, value: Any) -> str | None:
        if value is None:
            return None
        try:
            return str(uuid.UUID(str(value)))
        except Exception:
            return None

    def start_run(self, name: str) -> str:
        self.run_id = str(uuid.uuid4())
        print(f"[Logger] run_started: {self.run_id} name={name} project={self.project}")
        return self.run_id

    def log_event(self, body: str, metadata: dict | None = None) -> None:
        metadata = metadata or {}
        event = {
            "run_id": self._safe_uuid_str(self.run_id),
            "timestamp": time.time(),
            "event": body,
            "metadata": metadata,
        }
        print(f"[LOG] {event}")

    def end_run(self, status: str = "completed") -> None:
        end_time = datetime.now(timezone.utc).isoformat()
        print(f"[Logger] run_ended: {self.run_id} status={status} end_time={end_time}")


# Backward-compatible alias for external imports.
LangSmithLogger = StdoutLogger


def get_default_logger():
    return StdoutLogger()
