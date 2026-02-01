from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def get_mcp_servers(config_path: Path | None = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "mcp.config.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data.get("mcpServers", {})