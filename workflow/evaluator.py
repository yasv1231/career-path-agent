from __future__ import annotations

from typing import Any, Dict, List

from .extractor import validate_extracted_v1


def evaluate_extracted(extracted: Dict[str, Any]) -> Dict[str, Any]:
    errors = validate_extracted_v1(extracted)
    valid = len(errors) == 0
    notes: List[str] = []

    if not extracted.get("task"):
        notes.append("task missing or empty")
    if not extracted.get("intent"):
        notes.append("intent missing")
    if extracted.get("confidence", 0) < 0.5:
        notes.append("low confidence")

    score = 1.0
    if not valid:
        score -= 0.5
    if notes:
        score -= 0.2
    if score < 0:
        score = 0.0

    return {
        "valid": valid,
        "errors": errors,
        "notes": notes,
        "score": round(score, 2),
    }


def evaluation_feedback_message(evaluation: Dict[str, Any]) -> str:
    if evaluation.get("valid") and not evaluation.get("notes"):
        return "Evaluation: extracted schema valid; no issues detected."

    parts: List[str] = ["Evaluation: issues found."]
    if not evaluation.get("valid"):
        parts.append("Schema errors detected.")
    if evaluation.get("notes"):
        parts.append("Notes: " + "; ".join(evaluation.get("notes", [])))
    return " ".join(parts)


def evaluate_profile(profile: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
    missing: List[str] = []
    for field in required_fields:
        value = profile.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, list) and not value:
            missing.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field)
            continue

    notes: List[str] = []
    if missing:
        notes.append("missing_fields")

    score = 1.0
    if missing:
        score -= min(0.6, 0.08 * len(missing))
    if score < 0:
        score = 0.0

    return {
        "valid": len(missing) == 0,
        "missing_fields": missing,
        "notes": notes,
        "score": round(score, 2),
    }
