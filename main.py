# main.py
# Multi-Agent Career Path Recommendation System (Console Version)
# This version uses a LangGraph workflow for multi-turn orchestration.
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _load_env_file() -> None:
    """Best-effort .env loading for local runs.

    Supports both standard KEY=VALUE lines and PowerShell-style
    "$env:KEY=VALUE" lines.
    """
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    # First, try standard dotenv parsing if available.
    if load_dotenv:
        load_dotenv(env_path, override=False)

    # Then, handle PowerShell-style lines explicitly.
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key.lower().startswith("$env:"):
            key = key[5:]

        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env_file()

from langgraph_workflow import run_conversation


def main() -> None:
    run_conversation()


if __name__ == "__main__":
    main()
