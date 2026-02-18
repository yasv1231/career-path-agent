from __future__ import annotations

import re
from typing import Any, Dict, List


GOAL_CONSTRAINTS_SCHEMA_VERSION = "goal_constraints_v1"


def default_goal_constraints() -> Dict[str, Any]:
    return {
        "schema_version": GOAL_CONSTRAINTS_SCHEMA_VERSION,
        "time_dimension": {
            "target_horizon_weeks": None,
            "weekly_hours": None,
            "milestones": [],
        },
        "goals": [],
        "acceptable_cost": {
            "max_budget_cny": None,
            "max_delay_weeks": None,
            "max_failure_trials": None,
            "intensity_preference": None,
        },
        "hard_constraints": [],
        "preferences": {
            "resource_types": [],
            "locations": [],
            "languages": [],
            "delivery_modes": [],
        },
        "query_hints": [],
    }


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    number = _coerce_float(value)
    if number is None:
        return None
    return int(round(number))


def _coerce_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[,\n;|/]+", value)]
        return [part for part in parts if part]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _normalize_goal(goal: Dict[str, Any], fallback_rank: int) -> Dict[str, Any]:
    description = str(goal.get("description", "")).strip()
    metric = str(goal.get("metric", "")).strip()
    target_value = str(goal.get("target_value", "")).strip()
    deadline_weeks = _coerce_int(goal.get("deadline_weeks"))
    weight = _coerce_float(goal.get("weight"))
    rank = _coerce_int(goal.get("priority_rank"))
    if rank is None:
        rank = fallback_rank
    if weight is None:
        # Rank-based default weight.
        weight = max(0.1, 1.0 - (rank - 1) * 0.2)
    return {
        "description": description,
        "metric": metric,
        "target_value": target_value,
        "deadline_weeks": deadline_weeks,
        "priority_rank": rank,
        "weight": float(max(0.0, min(1.0, weight))),
    }


def _normalize_constraint(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": str(item.get("name", "")).strip(),
        "operator": str(item.get("operator", "")).strip() or "must",
        "value": str(item.get("value", "")).strip(),
        "unit": str(item.get("unit", "")).strip(),
        "priority": str(item.get("priority", "")).strip() or "hard",
    }


def coerce_goal_constraints_v1(data: Any) -> Dict[str, Any]:
    result = default_goal_constraints()
    if not isinstance(data, dict):
        return result

    if isinstance(data.get("schema_version"), str):
        result["schema_version"] = data["schema_version"]

    time_dimension = data.get("time_dimension", {})
    if isinstance(time_dimension, dict):
        result["time_dimension"]["target_horizon_weeks"] = _coerce_int(
            time_dimension.get("target_horizon_weeks")
        )
        result["time_dimension"]["weekly_hours"] = _coerce_int(
            time_dimension.get("weekly_hours")
        )
        result["time_dimension"]["milestones"] = _coerce_text_list(
            time_dimension.get("milestones")
        )

    goals = data.get("goals", [])
    if isinstance(goals, list):
        normalized: List[Dict[str, Any]] = []
        for index, item in enumerate(goals, start=1):
            if not isinstance(item, dict):
                continue
            normalized_goal = _normalize_goal(item, fallback_rank=index)
            if normalized_goal["description"]:
                normalized.append(normalized_goal)
        normalized.sort(key=lambda item: (item["priority_rank"], -item["weight"]))
        result["goals"] = normalized

    acceptable_cost = data.get("acceptable_cost", {})
    if isinstance(acceptable_cost, dict):
        result["acceptable_cost"]["max_budget_cny"] = _coerce_float(
            acceptable_cost.get("max_budget_cny")
        )
        result["acceptable_cost"]["max_delay_weeks"] = _coerce_int(
            acceptable_cost.get("max_delay_weeks")
        )
        result["acceptable_cost"]["max_failure_trials"] = _coerce_int(
            acceptable_cost.get("max_failure_trials")
        )
        intensity = acceptable_cost.get("intensity_preference")
        if isinstance(intensity, str) and intensity.strip():
            result["acceptable_cost"]["intensity_preference"] = intensity.strip().lower()

    hard_constraints = data.get("hard_constraints", [])
    if isinstance(hard_constraints, list):
        normalized_hard: List[Dict[str, Any]] = []
        for item in hard_constraints:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_constraint(item)
            if normalized["name"] or normalized["value"]:
                normalized_hard.append(normalized)
        result["hard_constraints"] = normalized_hard

    preferences = data.get("preferences", {})
    if isinstance(preferences, dict):
        for key in ("resource_types", "locations", "languages", "delivery_modes"):
            result["preferences"][key] = _coerce_text_list(preferences.get(key))

    result["query_hints"] = _coerce_text_list(data.get("query_hints"))
    return result


def merge_goal_constraints(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    merged = coerce_goal_constraints_v1(base)
    update = coerce_goal_constraints_v1(delta)

    for key in ("target_horizon_weeks", "weekly_hours"):
        value = update["time_dimension"].get(key)
        if value is not None:
            merged["time_dimension"][key] = value
    if update["time_dimension"].get("milestones"):
        existing = merged["time_dimension"].get("milestones", [])
        merged["time_dimension"]["milestones"] = list(
            dict.fromkeys(existing + update["time_dimension"]["milestones"])
        )

    if update.get("goals"):
        merged["goals"] = update["goals"]

    for key in ("max_budget_cny", "max_delay_weeks", "max_failure_trials", "intensity_preference"):
        value = update["acceptable_cost"].get(key)
        if value not in (None, ""):
            merged["acceptable_cost"][key] = value

    if update.get("hard_constraints"):
        merged["hard_constraints"] = list(
            dict.fromkeys(
                tuple(sorted(item.items())) for item in merged["hard_constraints"] + update["hard_constraints"]
            )
        )
        merged["hard_constraints"] = [dict(item) for item in merged["hard_constraints"]]

    for key in ("resource_types", "locations", "languages", "delivery_modes"):
        if update["preferences"].get(key):
            existing = merged["preferences"].get(key, [])
            merged["preferences"][key] = list(dict.fromkeys(existing + update["preferences"][key]))

    if update.get("query_hints"):
        merged["query_hints"] = list(dict.fromkeys(merged.get("query_hints", []) + update["query_hints"]))

    return merged


def _to_weeks(value: int, unit: str) -> int:
    normalized = unit.lower().strip()
    if normalized in {"week", "weeks", "w", "周"}:
        return value
    if normalized in {"month", "months", "m", "月", "个月"}:
        return int(round(value * 4.345))
    if normalized in {"year", "years", "y", "年"}:
        return int(round(value * 52))
    return value


def _infer_operator(sentence: str) -> str:
    lowered = sentence.lower()
    if "at least" in lowered or "不少于" in sentence or "至少" in sentence:
        return ">="
    if "at most" in lowered or "不超过" in sentence or "至多" in sentence:
        return "<="
    if "不能" in sentence or "must not" in lowered or "cannot" in lowered or "can't" in lowered:
        return "not"
    return "must"


def _extract_goals(text: str) -> List[Dict[str, Any]]:
    goals: List[Dict[str, Any]] = []
    lines = [line.strip() for line in re.split(r"[\n。！？!?]", text) if line.strip()]
    rank = 1
    for line in lines:
        lowered = line.lower()
        if any(keyword in lowered for keyword in ("goal", "target", "希望", "目标", "想要", "达成", "完成", "拿到")):
            goals.append(
                {
                    "description": line,
                    "metric": "",
                    "target_value": "",
                    "deadline_weeks": None,
                    "priority_rank": rank,
                    "weight": max(0.1, 1.0 - (rank - 1) * 0.2),
                }
            )
            rank += 1
        if len(goals) >= 5:
            break
    return goals


def extract_goal_constraints_from_text(text: str) -> Dict[str, Any]:
    payload = default_goal_constraints()
    raw = (text or "").strip()
    if not raw:
        return payload

    lowered = raw.lower()

    # Time horizon
    horizon_candidates: List[int] = []
    for match in re.finditer(r"(\d+)\s*(week|weeks|w|周|month|months|月|个月|year|years|年)", lowered):
        value = int(match.group(1))
        unit = match.group(2)
        weeks = _to_weeks(value, unit)
        horizon_candidates.append(weeks)
    if horizon_candidates:
        payload["time_dimension"]["target_horizon_weeks"] = min(horizon_candidates)

    # Weekly hours
    hours_match = re.search(
        r"(\d+)\s*(小时|h|hour|hours)\s*(/week|per week|每周|每星期|每周可投入)?",
        lowered,
    )
    if hours_match:
        payload["time_dimension"]["weekly_hours"] = int(hours_match.group(1))

    # Budget
    budget_match_cny = re.search(r"(预算|cost|budget|费用)[^\d]{0,8}(\d{2,6})", lowered)
    if budget_match_cny:
        payload["acceptable_cost"]["max_budget_cny"] = float(budget_match_cny.group(2))

    # Delay tolerance
    delay_match = re.search(r"(\d+)\s*(week|weeks|周).{0,8}(delay|延期|延后)", lowered)
    if delay_match:
        payload["acceptable_cost"]["max_delay_weeks"] = _to_weeks(
            int(delay_match.group(1)),
            delay_match.group(2),
        )

    # Intensity preference
    if any(word in lowered for word in ("轻量", "low intensity", "轻松", "不卷")):
        payload["acceptable_cost"]["intensity_preference"] = "low"
    elif any(word in lowered for word in ("高强度", "intensive", "all-in", "冲刺")):
        payload["acceptable_cost"]["intensity_preference"] = "high"

    # Resource type preferences
    if any(word in lowered for word in ("course", "课程", "教程")):
        payload["preferences"]["resource_types"].append("course")
    if any(word in lowered for word in ("project", "项目", "实战", "portfolio")):
        payload["preferences"]["resource_types"].append("project")

    # Delivery mode preferences
    if any(word in lowered for word in ("remote", "online", "线上", "远程")):
        payload["preferences"]["delivery_modes"].append("online")
    if any(word in lowered for word in ("onsite", "线下", "面授")):
        payload["preferences"]["delivery_modes"].append("offline")

    # Hard constraints from modal language.
    for sentence in re.split(r"[。！？!?\n]", raw):
        line = sentence.strip()
        if not line:
            continue
        lowered_line = line.lower()
        if any(
            marker in lowered_line
            for marker in ("must", "must not", "cannot", "can't", "only", "at least", "at most")
        ) or any(marker in line for marker in ("必须", "不能", "只接受", "至少", "不超过")):
            payload["hard_constraints"].append(
                {
                    "name": line[:40],
                    "operator": _infer_operator(line),
                    "value": line,
                    "unit": "",
                    "priority": "hard",
                }
            )

    payload["goals"] = _extract_goals(raw)

    query_hints: List[str] = []
    for goal in payload["goals"][:3]:
        if goal["description"]:
            query_hints.append(goal["description"])
    for item in payload["hard_constraints"][:2]:
        if item["value"]:
            query_hints.append(item["value"])
    payload["query_hints"] = list(dict.fromkeys(query_hints))

    payload["preferences"]["resource_types"] = list(dict.fromkeys(payload["preferences"]["resource_types"]))
    payload["preferences"]["delivery_modes"] = list(dict.fromkeys(payload["preferences"]["delivery_modes"]))
    return payload


def derive_strategy_tags(goal_constraints: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    time_dimension = goal_constraints.get("time_dimension", {})
    acceptable_cost = goal_constraints.get("acceptable_cost", {})
    preferences = goal_constraints.get("preferences", {})

    horizon = _coerce_int(time_dimension.get("target_horizon_weeks"))
    weekly_hours = _coerce_int(time_dimension.get("weekly_hours"))
    budget = _coerce_float(acceptable_cost.get("max_budget_cny"))
    hard_constraints = goal_constraints.get("hard_constraints", [])

    if horizon is not None:
        if horizon <= 12:
            tags.append("short_horizon")
        elif horizon <= 24:
            tags.append("mid_horizon")
        else:
            tags.append("long_horizon")

    if weekly_hours is not None:
        if weekly_hours < 6:
            tags.append("low_bandwidth")
        elif weekly_hours > 15:
            tags.append("high_bandwidth")
        else:
            tags.append("mid_bandwidth")

    if budget is not None:
        if budget < 500:
            tags.append("low_budget")
        elif budget > 5000:
            tags.append("high_budget")
        else:
            tags.append("mid_budget")

    if hard_constraints:
        tags.append("constraint_driven")

    for resource_type in preferences.get("resource_types", []):
        tags.append(f"prefer_{str(resource_type).lower()}")

    for mode in preferences.get("delivery_modes", []):
        tags.append(f"prefer_{str(mode).lower()}")

    return list(dict.fromkeys(tags))


def build_rag_filters(goal_constraints: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    preferences = goal_constraints.get("preferences", {})
    resource_types = preferences.get("resource_types", [])
    if isinstance(resource_types, list) and resource_types:
        filters["resource_type"] = [str(item).lower() for item in resource_types if str(item).strip()]
    return filters
