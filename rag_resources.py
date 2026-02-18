from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_RAG_RESOURCES: Dict[str, Any] = {
    "career_aliases": {},
    "confidence_thresholds": {
        "low": 0.08,
        "mid": 0.14,
    },
    "vector_store": {
        "db_dir": "data/rag_chroma",
        "collection_name": "career_knowledge",
        "embedding_model": "shibing624/text2vec-base-chinese",
        "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "enable_rerank": False,
    },
    "context_format": {
        "context_prefix": "RAG_CONTEXT:",
        "citations_label": "Citations",
    },
    "sections": [],
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_resources_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "rag_resources.json"


def load_rag_resources(path: Path | None = None) -> Dict[str, Any]:
    resource_path = path or default_resources_path()
    defaults = json.loads(json.dumps(DEFAULT_RAG_RESOURCES, ensure_ascii=False))

    if not resource_path.exists():
        return defaults

    try:
        data = json.loads(resource_path.read_text(encoding="utf-8"))
    except Exception:
        return defaults

    if not isinstance(data, dict):
        return defaults

    merged = _deep_merge(defaults, data)
    if not isinstance(merged.get("sections"), list):
        merged["sections"] = defaults.get("sections", [])
    return merged
