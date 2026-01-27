import os
import uuid
import time
from datetime import datetime, timezone
from typing import Any

try:
    from langsmith import Client
    HAS_LANGSMITH = True
except Exception:
    Client = None
    HAS_LANGSMITH = False

try:
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN_OPENAI = True
except Exception:
    ChatOpenAI = None
    HAS_LANGCHAIN_OPENAI = False


def configure_langsmith(project: str | None = None) -> dict[str, Any]:
    api_key = os.getenv("LANGSMITH_API_KEY")
    project_name = project or os.getenv("LANGSMITH_PROJECT") or "career-path-agent"

    status: dict[str, Any] = {
        "has_api_key": bool(api_key),
        "project": project_name,
        "tracing_enabled": False,
    }

    if not api_key:
        return status

    if not os.getenv("LANGSMITH_TRACING"):
        os.environ["LANGSMITH_TRACING"] = "true"

    os.environ["LANGSMITH_PROJECT"] = project_name
    status["tracing_enabled"] = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    return status


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


class LangSmithLogger:
    def __init__(self, api_key: str | None = None, project: str | None = None):
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        self.project = project or os.getenv("LANGSMITH_PROJECT") or "career-path-agent"
        self.enabled = HAS_LANGSMITH and bool(self.api_key)
        self.client = None
        self.run_id = None

        if self.enabled:
            try:
                self.client = Client(api_key=self.api_key)
            except Exception as e:
                print(f"[LangSmith] Client init failed: {e}")
                self.enabled = False

    def start_run(self, name: str) -> str:
        self.run_id = str(uuid.uuid4())
        if self.enabled:
            try:
                run = self.client.create_run(
                    name=name,
                    project_name=self.project,
                    inputs={},
                    run_type="chain",
                )
                self.run_id = str(getattr(run, "id", self.run_id))
            except Exception as e:
                print(f"[LangSmith] start_run failed: {e}")
                self.enabled = False

        print(f"[LangSmith] run_started: {self.run_id} name={name}")
        return self.run_id

    def log_event(self, body: str, metadata: dict | None = None) -> None:
        metadata = metadata or {}
        ts = time.time()
        if self.enabled:
            try:
                if hasattr(self.client, "log_event"):
                    self.client.log_event(self.run_id, body=body, metadata=metadata, timestamp=ts)
                elif hasattr(self.client, "create_event"):
                    self.client.create_event(run_id=self.run_id, body=body, metadata=metadata)
            except Exception:
                # Degrade to stdout but keep the run alive.
                self.enabled = False
                print(f"[LangSmith-LOG] {body} | {metadata}")
        else:
            print(f"[LangSmith-LOG] {body} | {metadata}")

    def end_run(self, status: str = "completed") -> None:
        if self.enabled:
            try:
                end_time = datetime.now(timezone.utc)
                if hasattr(self.client, "update_run"):
                    self.client.update_run(run_id=self.run_id, outputs={}, end_time=end_time)
                elif hasattr(self.client, "finish_run"):
                    self.client.finish_run(self.run_id, status=status)
            except Exception as e:
                print(f"[LangSmith] end_run failed: {e}")
                self.enabled = False

        print(f"[LangSmith] run_ended: {self.run_id} status={status}")


def get_default_logger():
    return LangSmithLogger()
