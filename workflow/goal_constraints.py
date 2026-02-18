from __future__ import annotations

import re
from typing import Any, Dict, List


GOAL_CONSTRAINTS_SCHEMA_VERSION = "goal_constraints_v1"

EN_NUMBER_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

ZH_NUMBER_MAP = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

ZH_UNIT_MAP = {"十": 10, "百": 100, "千": 1000, "万": 10000}

NUMBER_TOKEN = (
    r"(?:\d+(?:\.\d+)?|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"[一二三四五六七八九十百千万两俩]+)"
)
TIME_UNIT_PATTERN = r"(?:years?|yrs?|yr|months?|mos?|month|weeks?|wks?|week|days?|day|年|个月|月|周|星期|天)"
TIME_RE = re.compile(
    rf"(?P<a>{NUMBER_TOKEN})(?:\s*(?:-|to|or|/|~|～|至|到)\s*(?P<b>{NUMBER_TOKEN}))?\s*(?P<unit>{TIME_UNIT_PATTERN})",
    re.IGNORECASE,
)
WITHIN_RE_EN = re.compile(
    rf"(?:within|in)\s+(?P<a>{NUMBER_TOKEN})(?:\s*(?:-|to|or|/|~|～)\s*(?P<b>{NUMBER_TOKEN}))?\s*(?P<unit>{TIME_UNIT_PATTERN})",
    re.IGNORECASE,
)
WITHIN_RE_ZH = re.compile(
    rf"(?P<a>{NUMBER_TOKEN})(?:\s*(?:-|到|至|~|～)\s*(?P<b>{NUMBER_TOKEN}))?\s*(?P<unit>年|个月|月|周|星期|天)\s*(?:内|之内|以内)"
)
MONTHLY_INCOME_RE = re.compile(
    r"(?:rmb|cny|¥|￥)?\s*(\d[\d,]*)\s*(?:/month|per month|monthly|每月|月入|月薪)",
    re.IGNORECASE,
)


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
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
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
        weight = max(0.1, 1.0 - (rank - 1) * 0.15)
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
        result["time_dimension"]["weekly_hours"] = _coerce_int(time_dimension.get("weekly_hours"))
        result["time_dimension"]["milestones"] = _coerce_text_list(time_dimension.get("milestones"))

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
        result["acceptable_cost"]["max_budget_cny"] = _coerce_float(acceptable_cost.get("max_budget_cny"))
        result["acceptable_cost"]["max_delay_weeks"] = _coerce_int(acceptable_cost.get("max_delay_weeks"))
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
        packed = [tuple(sorted(item.items())) for item in merged["hard_constraints"] + update["hard_constraints"]]
        merged["hard_constraints"] = [dict(item) for item in dict.fromkeys(packed)]

    for key in ("resource_types", "locations", "languages", "delivery_modes"):
        if update["preferences"].get(key):
            existing = merged["preferences"].get(key, [])
            merged["preferences"][key] = list(dict.fromkeys(existing + update["preferences"][key]))

    if update.get("query_hints"):
        merged["query_hints"] = list(dict.fromkeys(merged.get("query_hints", []) + update["query_hints"]))

    return merged


def _split_sentences(text: str) -> List[str]:
    parts = [part.strip() for part in re.split(r"[。！？!?;.\n]+", text) if part.strip()]
    return parts


def _parse_en_number(text: str) -> int | None:
    parts = [part for part in text.lower().replace("-", " ").split() if part]
    if not parts:
        return None
    total = 0
    current = 0
    for part in parts:
        if part == "hundred":
            current = max(current, 1) * 100
            continue
        value = EN_NUMBER_MAP.get(part)
        if value is None:
            return None
        current += value
    total += current
    return total if total > 0 else None


def _parse_zh_number(text: str) -> int | None:
    token = text.strip()
    if not token:
        return None

    if token in ZH_NUMBER_MAP:
        return ZH_NUMBER_MAP[token]

    total = 0
    current = 0
    for char in token:
        if char in ZH_NUMBER_MAP:
            current = ZH_NUMBER_MAP[char]
            continue
        unit = ZH_UNIT_MAP.get(char)
        if unit is None:
            return None
        if current == 0:
            current = 1
        total += current * unit
        current = 0
    return total + current if (total + current) > 0 else None


def _parse_number_token(token: str) -> float | None:
    cleaned = token.strip().lower().replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except Exception:
        pass
    en_num = _parse_en_number(cleaned)
    if en_num is not None:
        return float(en_num)
    zh_num = _parse_zh_number(token)
    if zh_num is not None:
        return float(zh_num)
    return None


def _to_weeks(value: float, unit: str) -> int:
    normalized = unit.lower().strip()
    if normalized in {"year", "years", "yr", "yrs", "年"}:
        return int(round(value * 52))
    if normalized in {"month", "months", "mo", "mos", "月", "个月"}:
        return int(round(value * 4.345))
    if normalized in {"week", "weeks", "wk", "wks", "w", "周", "星期"}:
        return int(round(value))
    if normalized in {"day", "days", "d", "天"}:
        return int(round(value / 7))
    return int(round(value))


def _extract_horizon_weeks(raw: str) -> int | None:
    lowered = raw.lower()
    for regex in (WITHIN_RE_EN, WITHIN_RE_ZH):
        match = regex.search(lowered)
        if not match:
            continue
        first = _parse_number_token(match.group("a"))
        second = _parse_number_token(match.group("b") or "")
        if first is None:
            continue
        value = max(first, second or first)
        return _to_weeks(value, match.group("unit"))

    values: List[int] = []
    for match in TIME_RE.finditer(lowered):
        first = _parse_number_token(match.group("a"))
        second = _parse_number_token(match.group("b") or "")
        if first is None:
            continue
        value = max(first, second or first)
        values.append(_to_weeks(value, match.group("unit")))
    if not values:
        return None
    return max(values)


def _extract_weekly_hours(raw: str) -> int | None:
    patterns = [
        re.compile(rf"(?P<n>{NUMBER_TOKEN})\s*(?:hours?|hrs?|h)\s*(?:per|a)?\s*week", re.IGNORECASE),
        re.compile(rf"(?:per|a)\s*week\s*(?P<n>{NUMBER_TOKEN})\s*(?:hours?|hrs?|h)", re.IGNORECASE),
        re.compile(rf"每周(?:可投入|能投入|投入|学习)?\s*(?P<n>{NUMBER_TOKEN})\s*(?:小时|h)"),
    ]
    for pattern in patterns:
        match = pattern.search(raw)
        if not match:
            continue
        value = _parse_number_token(match.group("n"))
        if value is not None:
            return int(round(value))
    return None


def _extract_budget(raw: str) -> float | None:
    patterns = [
        re.compile(
            r"(?:budget|cost|spend|tuition|预算|费用|花费|成本)[^\d]{0,12}(?:rmb|cny|¥|￥)?\s*(\d[\d,]*)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:under|below|less than|up to|不超过|最多)\s*(?:rmb|cny|¥|￥)?\s*(\d[\d,]*)\s*(?:for courses|for learning|学习|课程)?",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(raw)
        if not match:
            continue
        value = _coerce_float(match.group(1))
        if value is not None:
            return value
    return None


def _extract_delay_weeks(raw: str) -> int | None:
    pattern = re.compile(
        rf"(?P<n>{NUMBER_TOKEN})\s*(?P<unit>{TIME_UNIT_PATTERN}).{{0,12}}(?:delay|postpone|延期|延后)",
        re.IGNORECASE,
    )
    match = pattern.search(raw)
    if not match:
        return None
    value = _parse_number_token(match.group("n"))
    if value is None:
        return None
    return _to_weeks(value, match.group("unit"))


def _infer_intensity(raw: str) -> str | None:
    lowered = raw.lower()
    low_markers = [
        "do not want overtime",
        "don't want overtime",
        "prefer not to work long-term overtime",
        "not overly exhausting",
        "work-life balance",
        "cannot accept frequent changes",
        "不想长期加班",
        "不想太累",
        "不接受频繁变动",
        "稳定",
        "不卷",
    ]
    high_markers = [
        "high-intensity",
        "high intensity",
        "high-pressure",
        "high pressure",
        "long hours",
        "996",
        "all-in",
        "rapid growth",
        "愿意高强度",
        "高强度",
        "愿意加班",
        "冲刺",
    ]
    low_score = sum(1 for marker in low_markers if marker in lowered)
    high_score = sum(1 for marker in high_markers if marker in lowered)
    if low_score > high_score and low_score > 0:
        return "low"
    if high_score > low_score and high_score > 0:
        return "high"
    return None


def _extract_locations(raw: str) -> List[str]:
    city_aliases = {
        "beijing": ["beijing", "北京"],
        "shanghai": ["shanghai", "上海"],
        "shenzhen": ["shenzhen", "深圳"],
        "guangzhou": ["guangzhou", "广州"],
    }
    lowered = raw.lower()
    found: List[str] = []
    for city, aliases in city_aliases.items():
        if any(alias in lowered for alias in aliases):
            found.append(city)
    if "first-tier city" in lowered or "一线城市" in raw:
        found.append("first-tier city")
    return list(dict.fromkeys(found))


def _extract_resource_types(raw: str) -> List[str]:
    lowered = raw.lower()
    found: List[str] = []
    if any(word in lowered for word in ("course", "courses", "tutorial", "certification", "课程", "教程")):
        found.append("course")
    if any(word in lowered for word in ("project", "portfolio", "case study", "项目", "实战")):
        found.append("project")
    return list(dict.fromkeys(found))


def _extract_delivery_modes(raw: str) -> List[str]:
    lowered = raw.lower()
    found: List[str] = []
    if any(word in lowered for word in ("remote", "online", "virtual", "远程", "线上")):
        found.append("online")
    if any(word in lowered for word in ("onsite", "offline", "in person", "线下", "现场")):
        found.append("offline")
    return list(dict.fromkeys(found))


def _extract_languages(raw: str) -> List[str]:
    lowered = raw.lower()
    found: List[str] = []
    if "english" in lowered or "英语" in raw:
        found.append("english")
    if "chinese" in lowered or "中文" in raw:
        found.append("chinese")
    return found


def _infer_operator(sentence: str) -> str:
    lowered = sentence.lower()
    if any(marker in lowered for marker in ("at least", "不少于", "至少")):
        return ">="
    if any(marker in lowered for marker in ("at most", "no more than", "不超过", "至多")):
        return "<="
    if any(
        marker in lowered
        for marker in (
            "must not",
            "cannot",
            "can't",
            "do not want",
            "don't want",
            "prefer not to",
            "cannot accept",
            "不想",
            "不能",
            "不接受",
            "不考虑",
            "不要",
            "不希望",
        )
    ):
        return "not"
    return "must"


def _extract_hard_constraints(raw: str) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = []
    markers = (
        "must",
        "must not",
        "cannot",
        "can't",
        "only",
        "at least",
        "at most",
        "need to",
        "require",
        "do not want",
        "don't want",
        "prefer not to",
        "cannot accept",
        "cannot be",
        "不想",
        "不能",
        "必须",
        "只接受",
        "至少",
        "至多",
        "不超过",
        "不接受",
        "不考虑",
        "最好不要",
    )
    for sentence in _split_sentences(raw):
        lowered = sentence.lower()
        if any(marker in lowered for marker in markers):
            constraints.append(
                {
                    "name": sentence[:64],
                    "operator": _infer_operator(sentence),
                    "value": sentence,
                    "unit": "",
                    "priority": "hard",
                }
            )
    packed = [tuple(sorted(item.items())) for item in constraints]
    return [dict(item) for item in dict.fromkeys(packed)]


def _extract_goal_deadline_weeks(sentence: str) -> int | None:
    values: List[int] = []
    for match in TIME_RE.finditer(sentence.lower()):
        first = _parse_number_token(match.group("a"))
        second = _parse_number_token(match.group("b") or "")
        if first is None:
            continue
        values.append(_to_weeks(max(first, second or first), match.group("unit")))
    if not values:
        return None
    return max(values)


def _extract_goals(raw: str) -> List[Dict[str, Any]]:
    goals: List[Dict[str, Any]] = []
    goal_markers = (
        "goal",
        "target",
        "aim",
        "hope",
        "want",
        "would like",
        "plan to",
        "ideally",
        "希望",
        "目标",
        "想要",
        "打算",
        "希望能",
    )
    rank = 1
    for sentence in _split_sentences(raw):
        lowered = sentence.lower()
        if not any(marker in lowered for marker in goal_markers):
            continue
        if len(sentence) < 8:
            continue

        metric = ""
        target_value = ""
        income_match = MONTHLY_INCOME_RE.search(sentence)
        if income_match:
            metric = "income_monthly_cny"
            target_value = income_match.group(1).replace(",", "")

        goals.append(
            {
                "description": sentence,
                "metric": metric,
                "target_value": target_value,
                "deadline_weeks": _extract_goal_deadline_weeks(sentence),
                "priority_rank": rank,
                "weight": max(0.1, 1.0 - (rank - 1) * 0.15),
            }
        )
        rank += 1
        if len(goals) >= 8:
            break
    return goals


def _trim_hint(text: str, max_len: int = 200) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def extract_goal_constraints_from_text(text: str) -> Dict[str, Any]:
    payload = default_goal_constraints()
    raw = (text or "").strip()
    if not raw:
        return payload

    horizon_weeks = _extract_horizon_weeks(raw)
    if horizon_weeks is not None:
        payload["time_dimension"]["target_horizon_weeks"] = horizon_weeks

    weekly_hours = _extract_weekly_hours(raw)
    if weekly_hours is not None:
        payload["time_dimension"]["weekly_hours"] = weekly_hours

    budget = _extract_budget(raw)
    if budget is not None:
        payload["acceptable_cost"]["max_budget_cny"] = budget

    delay = _extract_delay_weeks(raw)
    if delay is not None:
        payload["acceptable_cost"]["max_delay_weeks"] = delay

    intensity = _infer_intensity(raw)
    if intensity:
        payload["acceptable_cost"]["intensity_preference"] = intensity

    payload["hard_constraints"] = _extract_hard_constraints(raw)
    payload["goals"] = _extract_goals(raw)

    payload["preferences"]["resource_types"] = _extract_resource_types(raw)
    payload["preferences"]["locations"] = _extract_locations(raw)
    payload["preferences"]["languages"] = _extract_languages(raw)
    payload["preferences"]["delivery_modes"] = _extract_delivery_modes(raw)

    query_hints: List[str] = []
    for goal in payload["goals"][:4]:
        if goal["description"]:
            query_hints.append(_trim_hint(goal["description"]))
    for item in payload["hard_constraints"][:3]:
        if item["value"]:
            query_hints.append(_trim_hint(item["value"]))
    payload["query_hints"] = list(dict.fromkeys([item for item in query_hints if item]))
    return payload


def derive_strategy_tags(goal_constraints: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    time_dimension = goal_constraints.get("time_dimension", {})
    acceptable_cost = goal_constraints.get("acceptable_cost", {})
    preferences = goal_constraints.get("preferences", {})
    goals = goal_constraints.get("goals", [])

    horizon = _coerce_int(time_dimension.get("target_horizon_weeks"))
    weekly_hours = _coerce_int(time_dimension.get("weekly_hours"))
    budget = _coerce_float(acceptable_cost.get("max_budget_cny"))
    intensity = str(acceptable_cost.get("intensity_preference", "")).strip().lower()
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
        if budget < 1000:
            tags.append("low_budget")
        elif budget > 10000:
            tags.append("high_budget")
        else:
            tags.append("mid_budget")

    if intensity == "low":
        tags.append("low_intensity")
    elif intensity == "high":
        tags.append("high_intensity")

    if hard_constraints:
        tags.append("constraint_driven")

    goal_text = " ".join(str(item.get("description", "")) for item in goals).lower()
    if any(keyword in goal_text for keyword in ("stable", "stability", "稳定")):
        tags.append("stability_first")
    if any(keyword in goal_text for keyword in ("high income", "rapid growth", "高收入", "成长速度")):
        tags.append("growth_first")
    if any(keyword in goal_text for keyword in ("ai", "machine learning", "算法", "人工智能")):
        tags.append("ai_oriented")

    for resource_type in preferences.get("resource_types", []):
        tags.append(f"prefer_{str(resource_type).lower()}")
    for mode in preferences.get("delivery_modes", []):
        tags.append(f"prefer_{str(mode).lower()}")
    if preferences.get("locations"):
        tags.append("location_bound")

    return list(dict.fromkeys(tags))


def build_rag_filters(goal_constraints: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    preferences = goal_constraints.get("preferences", {})
    resource_types = preferences.get("resource_types", [])
    if isinstance(resource_types, list) and resource_types:
        filters["resource_type"] = [str(item).lower() for item in resource_types if str(item).strip()]
    return filters
