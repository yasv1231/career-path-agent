from __future__ import annotations

import json
from typing import Any, Dict, List

from .state import EXTRACT_ALLOWED_INTENTS, EXTRACT_SCHEMA_VERSION, EXTRACT_SCHEMA_V1


def extract_json_block(text: str) -> str | None:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return None


def try_parse_json(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        block = extract_json_block(text)
        if not block:
            return None
        try:
            return json.loads(block)
        except Exception:
            return None


def default_extracted(confidence: float = 0.2) -> Dict[str, Any]:
    return {
        "schema_version": EXTRACT_SCHEMA_VERSION,
        "task": "",
        "intent": "other",
        "entities": [],
        "constraints": [],
        "actions": [],
        "confidence": confidence,
    }


def coerce_extracted_v1(data: Any, confidence: float = 0.4) -> Dict[str, Any]:
    result = default_extracted(confidence=confidence)
    if not isinstance(data, dict):
        return result
    if isinstance(data.get("schema_version"), str):
        result["schema_version"] = data.get("schema_version")
    if isinstance(data.get("task"), str):
        result["task"] = data.get("task")
    if isinstance(data.get("intent"), str):
        result["intent"] = data.get("intent")
    if isinstance(data.get("entities"), list):
        result["entities"] = data.get("entities")
    if isinstance(data.get("constraints"), list):
        result["constraints"] = data.get("constraints")
    if isinstance(data.get("actions"), list):
        result["actions"] = data.get("actions")
    if isinstance(data.get("confidence"), (int, float)):
        result["confidence"] = float(data.get("confidence"))
    return result


def validate_extracted_v1(data: Any) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if not isinstance(data, dict):
        return [{"path": "$.", "rule": "type", "message": "expected object", "expected": "object", "found": type(data).__name__}]

    if data.get("schema_version") != EXTRACT_SCHEMA_VERSION:
        errors.append(
            {
                "path": "$.schema_version",
                "rule": "enum",
                "message": "schema_version must be v1",
                "expected": EXTRACT_SCHEMA_VERSION,
                "found": str(data.get("schema_version")),
            }
        )

    if not isinstance(data.get("task"), str):
        errors.append(
            {
                "path": "$.task",
                "rule": "type",
                "message": "task must be string",
                "expected": "string",
                "found": type(data.get("task")).__name__,
            }
        )

    intent = data.get("intent")
    if intent not in EXTRACT_ALLOWED_INTENTS:
        errors.append(
            {
                "path": "$.intent",
                "rule": "enum",
                "message": "intent must be a supported value",
                "expected": "|".join(sorted(EXTRACT_ALLOWED_INTENTS)),
                "found": str(intent),
            }
        )

    if not isinstance(data.get("entities"), list):
        errors.append(
            {
                "path": "$.entities",
                "rule": "type",
                "message": "entities must be array",
                "expected": "array",
                "found": type(data.get("entities")).__name__,
            }
        )
    else:
        for idx, entity in enumerate(data.get("entities", [])):
            if not isinstance(entity, dict):
                errors.append(
                    {
                        "path": f"$.entities[{idx}]",
                        "rule": "type",
                        "message": "entity must be object",
                        "expected": "object",
                        "found": type(entity).__name__,
                    }
                )
                continue
            if not isinstance(entity.get("type"), str):
                errors.append(
                    {
                        "path": f"$.entities[{idx}].type",
                        "rule": "type",
                        "message": "entity.type must be string",
                        "expected": "string",
                        "found": type(entity.get("type")).__name__,
                    }
                )
            if not isinstance(entity.get("text"), str):
                errors.append(
                    {
                        "path": f"$.entities[{idx}].text",
                        "rule": "type",
                        "message": "entity.text must be string",
                        "expected": "string",
                        "found": type(entity.get("text")).__name__,
                    }
                )

    if not isinstance(data.get("constraints"), list):
        errors.append(
            {
                "path": "$.constraints",
                "rule": "type",
                "message": "constraints must be array",
                "expected": "array",
                "found": type(data.get("constraints")).__name__,
            }
        )
    else:
        for idx, item in enumerate(data.get("constraints", [])):
            if not isinstance(item, str):
                errors.append(
                    {
                        "path": f"$.constraints[{idx}]",
                        "rule": "type",
                        "message": "constraint must be string",
                        "expected": "string",
                        "found": type(item).__name__,
                    }
                )

    if not isinstance(data.get("actions"), list):
        errors.append(
            {
                "path": "$.actions",
                "rule": "type",
                "message": "actions must be array",
                "expected": "array",
                "found": type(data.get("actions")).__name__,
            }
        )
    else:
        for idx, item in enumerate(data.get("actions", [])):
            if not isinstance(item, dict):
                errors.append(
                    {
                        "path": f"$.actions[{idx}]",
                        "rule": "type",
                        "message": "action must be object",
                        "expected": "object",
                        "found": type(item).__name__,
                    }
                )
                continue
            if not isinstance(item.get("action"), str):
                errors.append(
                    {
                        "path": f"$.actions[{idx}].action",
                        "rule": "type",
                        "message": "action.action must be string",
                        "expected": "string",
                        "found": type(item.get("action")).__name__,
                    }
                )
            if not isinstance(item.get("target"), str):
                errors.append(
                    {
                        "path": f"$.actions[{idx}].target",
                        "rule": "type",
                        "message": "action.target must be string",
                        "expected": "string",
                        "found": type(item.get("target")).__name__,
                    }
                )
            if "params" in item and not isinstance(item.get("params"), dict):
                errors.append(
                    {
                        "path": f"$.actions[{idx}].params",
                        "rule": "type",
                        "message": "action.params must be object",
                        "expected": "object",
                        "found": type(item.get("params")).__name__,
                    }
                )

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        errors.append(
            {
                "path": "$.confidence",
                "rule": "range",
                "message": "confidence must be between 0 and 1",
                "expected": "0..1",
                "found": str(confidence),
            }
        )

    return errors


def retry_prompt(base_system: str, last_answer: str, errors: List[Dict[str, str]]) -> str:
    schema_json = json.dumps(EXTRACT_SCHEMA_V1, ensure_ascii=True)
    error_hint = json.dumps(errors[:5], ensure_ascii=True)
    return (
        f"{base_system}\n\nInput text:\n{last_answer}\n\nSchema:\n{schema_json}\n\n"
        f"Validation errors:\n{error_hint}\n\nFix the JSON."
    )