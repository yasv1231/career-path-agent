# main.py
# Multi-Agent Career Path Recommendation System (Console Version)
# Message flow runner (chat -> extract -> memory).
from pathlib import Path
import os

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

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from workflow.runner import run_message_flow


def main() -> None:
    run_message_flow(enable_memory=True)


if __name__ == "__main__":
    main()
