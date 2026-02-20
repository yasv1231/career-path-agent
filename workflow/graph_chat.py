from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from langsmith_integration import configure_langsmith, get_chat_model, get_default_logger
from .evaluator import evaluate_profile
from .extractor import default_profile, merge_profile, try_parse_json
from .goal_constraints import (
    build_rag_filters,
    default_goal_constraints,
    derive_strategy_tags,
    extract_goal_constraints_from_text,
    merge_goal_constraints,
)
from .mcp import get_mcp_servers
from .messages import is_openai_message, normalize_messages, serialize_messages_openai
from .rag import maybe_attach_rag_context
from .routing import route_to_state
from .state import ChatConversationState, PROFILE_REQUIRED_FIELDS, ProgressState, UserState

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

    HAS_LANGCHAIN_MESSAGES = True
except Exception:
    AIMessage = None
    BaseMessage = None
    HumanMessage = None
    SystemMessage = None
    HAS_LANGCHAIN_MESSAGES = False

logger = get_default_logger()

QUESTION_MAP: Dict[str, str] = {
    "location": "Which city are you targeting first?",
    "target_role": "Which exact target role title do you want to prioritize first?",
    "timeline_weeks": "In how many weeks do you want to complete phase-1 transition?",
    "hours_per_week": "How many hours per week can you invest consistently?",
    "compensation_floor": "What is your minimum acceptable monthly compensation (number only, for example 10000)?",
    "work_mode": "What is your preferred work mode: remote, hybrid, or on-site?",
}

SLOT_LABEL_MAP: Dict[str, str] = {
    "location": "target city",
    "target_role": "target role",
    "timeline_weeks": "transition timeline (weeks)",
    "hours_per_week": "weekly available hours",
    "compensation_floor": "minimum compensation floor",
    "work_mode": "preferred work mode",
}

LIST_SLOTS: set[str] = set()
NUMBER_SLOTS = {"hours_per_week", "timeline_weeks", "compensation_floor"}
REQUIRED_SLOT_ORDER = list(PROFILE_REQUIRED_FIELDS)
OPTIONAL_SLOT_ORDER = ["compensation_floor", "work_mode"]
FOLLOWUP_TARGET_SLOTS = REQUIRED_SLOT_ORDER + OPTIONAL_SLOT_ORDER
FOLLOWUP_REASON_CODES = {
    "high_impact_gap",
    "high_uncertainty",
    "execution_risk",
    "data_quality_risk",
    "already_sufficient",
}
FOLLOWUP_QUESTION_TEMPLATES: Dict[str, str] = {
    "compensation_floor_focus": "What is your minimum acceptable monthly compensation for this role?",
    "work_mode_focus": "Which work mode do you prefer most for this transition: remote, hybrid, or on-site?",
}
OPTIONAL_SLOT_MIN_ITEMS: Dict[str, int] = {}
WORK_MODE_ENUM = {"remote", "hybrid", "on-site"}
CITY_ALLOWLIST = {
    "guangzhou",
    "shenzhen",
    "beijing",
    "shanghai",
    "hangzhou",
    "chengdu",
    "wuhan",
    "nanjing",
    "xian",
    "new york",
    "san francisco",
    "london",
    "singapore",
    "tokyo",
    "seoul",
}
STRUCTURED_REQUIRED_SLOTS = list(REQUIRED_SLOT_ORDER)
STRUCTURED_OPTIONAL_SLOTS = list(OPTIONAL_SLOT_ORDER)
STRUCTURED_SLOTS = set(STRUCTURED_REQUIRED_SLOTS + STRUCTURED_OPTIONAL_SLOTS)

YES_WORDS = {"yes", "y", "correct", "right", "sure", "ok", "okay"}
NO_WORDS = {"no", "n", "wrong", "not", "nope"}
UNCERTAIN_WORDS = {"not sure", "unsure", "unknown", "n/a", "idk", "none", "no idea"}


def build_message_graph(enable_memory: bool = True):
    langsmith_status = configure_langsmith("message-flow")
    llm = get_chat_model()

    def _default_user_state() -> UserState:
        return {
            "profile": default_profile(),
            "goal_constraints": default_goal_constraints(),
            "strategy_tags": [],
            "rag_filters": {},
            "extracted": {},
            "turn_index": 0,
        }

    def _default_progress_state() -> ProgressState:
        required_slots = list(REQUIRED_SLOT_ORDER)
        return {
            "question_stage": "collecting",
            "asked_questions": [],
            "attempts": {},
            "required_slots": required_slots,
            "optional_slots": list(OPTIONAL_SLOT_ORDER),
            "pending_slots": required_slots,
            "current_slot": required_slots[0] if required_slots else "",
            "current_slot_retries": 0,
            "max_slot_retries": 2,
            "awaiting_confirmation": False,
            "candidate_slot": "",
            "candidate_value": None,
            "candidate_confidence": 0.0,
            "candidate_source": "",
            "question_phase_complete": False,
            "plan_ready": False,
            "required_slots_closed": False,
            "non_basic_questions_asked": 0,
            "non_basic_questions_limit": 6,
            "followup_budget_remaining": 6.0,
            "followup_min_ig": 0.35,
            "followup_candidate": {},
            "followup_signatures_asked": [],
            "followup_should_stop": False,
            "followup_stop_reason": "",
            "followup_last_ig": 0.0,
            "last_node": "",
        }

    def _get_user_state(state: ChatConversationState) -> UserState:
        user_state = state.get("user_state")
        if isinstance(user_state, dict):
            merged = _default_user_state()
            merged.update(user_state)
            return merged
        return {
            "profile": state.get("memory") or default_profile(),
            "goal_constraints": state.get("goal_constraints") or default_goal_constraints(),
            "strategy_tags": list(state.get("strategy_tags", [])),
            "rag_filters": dict(state.get("rag_filters", {})),
            "extracted": dict(state.get("extracted", {})),
            "turn_index": 0,
        }

    def _get_progress_state(state: ChatConversationState) -> ProgressState:
        progress_state = state.get("progress_state")
        if isinstance(progress_state, dict):
            merged = _default_progress_state()
            merged.update(progress_state)
            if not merged.get("required_slots"):
                merged["required_slots"] = list(REQUIRED_SLOT_ORDER)
            if not merged.get("optional_slots"):
                merged["optional_slots"] = list(OPTIONAL_SLOT_ORDER)
            return merged
        merged = _default_progress_state()
        merged["asked_questions"] = list(state.get("asked_questions", []))
        merged["attempts"] = dict(state.get("attempts", {}))
        merged["question_stage"] = str(state.get("question_stage", "collecting"))
        merged["question_phase_complete"] = bool(state.get("question_phase_complete", False))
        return merged

    def _is_empty(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, list):
            return len(value) == 0
        return False

    def _is_slot_complete(slot: str, value: Any) -> bool:
        if slot in LIST_SLOTS:
            items = value if isinstance(value, list) else _split_to_list(str(value or ""))
            min_items = OPTIONAL_SLOT_MIN_ITEMS.get(slot, 1)
            return len([item for item in items if str(item).strip()]) >= min_items
        if slot in {"hours_per_week", "timeline_weeks", "compensation_floor"}:
            try:
                return float(value) > 0
            except Exception:
                return False
        if slot == "work_mode":
            return str(value or "").strip().lower() in WORK_MODE_ENUM
        return not _is_empty(value)

    def _last_user_content(messages: List[Any]) -> str:
        for msg in reversed(messages):
            if HAS_LANGCHAIN_MESSAGES and BaseMessage and isinstance(msg, BaseMessage):
                if getattr(msg, "type", "") == "human":
                    return str(getattr(msg, "content", ""))
            if is_openai_message(msg) and str(msg.get("role")).lower() == "user":
                return str(msg.get("content", ""))
        return ""

    def _append_assistant(messages: List[Any], content: str) -> None:
        if HAS_LANGCHAIN_MESSAGES and AIMessage:
            messages.append(AIMessage(content=content))
        else:
            messages.append({"role": "assistant", "content": content})

    def _invoke_llm(messages: List[Any]) -> str:
        if not llm:
            return ""
        try:
            if HAS_LANGCHAIN_MESSAGES:
                response = llm.invoke(messages)
            else:
                prompt = "\n".join(
                    f"{item.get('role', 'user')}: {item.get('content', '')}" for item in serialize_messages_openai(messages)
                )
                response = llm.invoke(prompt)
            return getattr(response, "content", None) or str(response)
        except Exception as exc:
            try:
                logger.log_event("llm_chat_error", {"error": str(exc)})
            except Exception:
                pass
            return ""

    def _missing_fields(profile: Dict[str, Any], required_slots: List[str]) -> List[str]:
        return [field for field in required_slots if _is_empty(profile.get(field))]

    def _sync_pending_slots(profile: Dict[str, Any], required_slots: List[str], pending_slots: List[str]) -> List[str]:
        pending: List[str] = []
        for slot in pending_slots:
            if slot in required_slots and _is_empty(profile.get(slot)):
                pending.append(slot)
        for slot in required_slots:
            if slot not in pending and _is_empty(profile.get(slot)):
                pending.append(slot)
        return pending

    def _all_pending_exhausted(pending_slots: List[str], attempts: Dict[str, int], max_slot_retries: int) -> bool:
        return bool(pending_slots) and all(int(attempts.get(slot, 0)) >= max_slot_retries for slot in pending_slots)

    def _split_to_list(raw: str) -> List[str]:
        normalized = raw
        # Protect thousands separators like "10,000" from naive comma splitting.
        protected = re.sub(r"(?<=\d),(?=\d)", "__NUM_COMMA__", normalized)
        parts = [part.strip().replace("__NUM_COMMA__", ",") for part in re.split(r"[,/\n;|]+", protected) if part.strip()]
        unique: List[str] = []
        for part in parts:
            if part not in unique:
                unique.append(part)
        return unique

    def _parse_compensation_floor(text: str) -> float | None:
        raw = (text or "").strip().lower().replace(",", "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", raw)
        if match:
            return float(match.group(1)) * 1000.0
        match = re.search(r"(\d+(?:\.\d+)?)", raw)
        if match:
            return float(match.group(1))
        return None

    def _normalize_work_mode(text: str) -> str | None:
        raw = (text or "").strip().lower()
        if not raw:
            return None
        if "hybrid" in raw:
            return "hybrid"
        if "remote" in raw:
            return "remote"
        if "on-site" in raw or "onsite" in raw or "office" in raw:
            return "on-site"
        return None

    def _normalize_city(value: Any) -> str | None:
        raw = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        if not raw:
            return None
        for city in CITY_ALLOWLIST:
            if city == raw or re.search(rf"\b{re.escape(city)}\b", raw):
                return city.title()
        return None

    def _append_summary_note(memory: Dict[str, Any], text: str) -> Dict[str, Any]:
        note = re.sub(r"\s+", " ", str(text or "")).strip()
        if not note:
            return memory
        existing = list(memory.get("summary_notes", []))
        if note not in existing:
            existing.append(note)
        return merge_profile(memory, {"summary_notes": existing[-20:]})

    def _append_unconfirmed(memory: Dict[str, Any], slot: str, raw_text: str, reason: str) -> Dict[str, Any]:
        record = f"{slot}: {reason} | {re.sub(r'\\s+', ' ', str(raw_text or '')).strip()}"
        existing = list(memory.get("unconfirmed_structured", []))
        if record not in existing:
            existing.append(record)
        memory = merge_profile(memory, {"unconfirmed_structured": existing[-20:]})
        return _append_summary_note(memory, f"Unconfirmed {slot}: {raw_text}")

    def _validate_structured_slot(slot: str, value: Any) -> tuple[bool, Any, str]:
        if slot == "location":
            city = _normalize_city(value)
            return (True, city, "") if city else (False, None, "city_not_recognized")
        if slot == "target_role":
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            lowered = text.lower()
            if len(text) >= 2 and not re.search(r"\b(week|weeks|hour|hours|per week)\b", lowered):
                return True, text, ""
            return False, None, "target_role_invalid"
        if slot == "hours_per_week":
            try:
                number = float(value)
            except Exception:
                return False, None, "hours_not_numeric"
            if 1 <= number <= 112:
                return True, number, ""
            return False, None, "hours_out_of_range"
        if slot == "timeline_weeks":
            try:
                number = float(value)
            except Exception:
                return False, None, "timeline_not_numeric"
            if 1 <= number <= 260:
                return True, number, ""
            return False, None, "timeline_out_of_range"
        if slot == "compensation_floor":
            parsed = _parse_compensation_floor(str(value))
            if parsed is not None and parsed > 0:
                return True, parsed, ""
            return False, None, "compensation_not_numeric"
        if slot == "work_mode":
            mode = _normalize_work_mode(str(value))
            if mode in WORK_MODE_ENUM:
                return True, mode, ""
            return False, None, "work_mode_not_in_enum"
        return False, None, "slot_not_structured"

    def _try_commit_slot(memory: Dict[str, Any], slot: str, value: Any, user_text: str) -> tuple[Dict[str, Any], bool, str]:
        if slot not in STRUCTURED_SLOTS:
            # Soft signals are summary-first and not structured.
            return (_append_summary_note(memory, user_text), False, "soft_signal_summary_only")
        ok, normalized, reason = _validate_structured_slot(slot, value)
        if ok:
            return (merge_profile(memory, {slot: normalized}), True, "")
        return (_append_unconfirmed(memory, slot, user_text or str(value), reason), False, reason)

    def _extract_hours_per_week(text: str) -> float | None:
        patterns = [
            re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\s*(?:per|a)?\s*week", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    pass
        if re.fullmatch(r"\d+(?:\.\d+)?", text.strip()):
            return float(text.strip())
        return None

    def _extract_timeline_weeks(text: str) -> float | None:
        lowered = text.lower().strip()
        week_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:weeks?|wks?|week)", lowered, re.IGNORECASE)
        if week_match:
            return float(week_match.group(1))
        month_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:months?|month)", lowered, re.IGNORECASE)
        if month_match:
            return float(month_match.group(1)) * 4.345
        year_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|year)", lowered, re.IGNORECASE)
        if year_match:
            return float(year_match.group(1)) * 52.0
        if re.fullmatch(r"\d+(?:\.\d+)?", lowered):
            return float(lowered)
        return None

    def _normalize_slot_value(slot: str, value: Any) -> Any:
        if value is None:
            return [] if slot in LIST_SLOTS else None
        if slot in LIST_SLOTS:
            if isinstance(value, list):
                items = [str(item).strip() for item in value if str(item).strip()]
            else:
                items = _split_to_list(str(value))
            return items
        if slot in NUMBER_SLOTS:
            if slot == "compensation_floor":
                parsed = _parse_compensation_floor(str(value))
                if parsed is not None:
                    return parsed
            try:
                return float(value)
            except Exception:
                return None
        if slot == "work_mode":
            return _normalize_work_mode(str(value)) or re.sub(r"\s+", " ", str(value)).strip().lower()
        return re.sub(r"\s+", " ", str(value)).strip()

    def _has_uncertain_signal(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in UNCERTAIN_WORDS)

    def _extract_rule_slot(slot: str, text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {"slot": slot, "value": None, "confidence": 0.0, "source": "none"}
        if _has_uncertain_signal(raw):
            return {"slot": slot, "value": None, "confidence": 0.2, "source": "rule"}

        if slot == "hours_per_week":
            value = _extract_hours_per_week(raw)
            return {"slot": slot, "value": value, "confidence": 0.98 if value is not None else 0.2, "source": "rule"}

        if slot == "timeline_weeks":
            value = _extract_timeline_weeks(raw)
            return {"slot": slot, "value": value, "confidence": 0.98 if value is not None else 0.2, "source": "rule"}

        if slot == "compensation_floor":
            value = _parse_compensation_floor(raw)
            return {"slot": slot, "value": value, "confidence": 0.95 if value is not None else 0.2, "source": "rule"}

        if slot == "work_mode":
            value = _normalize_work_mode(raw)
            return {"slot": slot, "value": value, "confidence": 0.95 if value is not None else 0.2, "source": "rule"}

        if slot in LIST_SLOTS:
            values = _split_to_list(raw)
            return {"slot": slot, "value": values if values else None, "confidence": 0.95 if values else 0.2, "source": "rule"}

        if slot == "location":
            city = _normalize_city(raw)
            if city:
                return {"slot": slot, "value": city, "confidence": 0.95, "source": "rule"}
            return {"slot": slot, "value": None, "confidence": 0.3, "source": "rule"}

        if slot == "target_role":
            if 2 <= len(raw) <= 80:
                return {"slot": slot, "value": raw, "confidence": 0.93, "source": "rule"}
            return {"slot": slot, "value": None, "confidence": 0.3, "source": "rule"}

        if len(raw) <= 100:
            return {"slot": slot, "value": raw, "confidence": 0.75, "source": "rule"}
        return {"slot": slot, "value": None, "confidence": 0.3, "source": "rule"}

    def _extract_llm_slot(slot: str, text: str) -> Dict[str, Any]:
        if not llm:
            return {"slot": slot, "value": None, "confidence": 0.0, "source": "none"}

        instruction = (
            "Extract one slot from user text and return strict JSON only: "
            '{"slot":"<slot>","value":<string|number|array|null>,"confidence":<0-1>}'
        )
        prompt = (
            f"slot={slot}\n"
            f"user_text={text}\n"
            "Rules:\n"
            "1) Keep only value for this slot.\n"
            "2) If unclear, set value=null and confidence <= 0.4.\n"
            "3) confidence must be numeric between 0 and 1."
        )
        try:
            if HAS_LANGCHAIN_MESSAGES and SystemMessage and HumanMessage:
                response = llm.invoke([SystemMessage(content=instruction), HumanMessage(content=prompt)])
            else:
                response = llm.invoke(instruction + "\n\n" + prompt)
            raw = getattr(response, "content", None) or str(response)
            data = try_parse_json(raw)
        except Exception as exc:
            try:
                logger.log_event("llm_slot_extract_error", {"slot": slot, "error": str(exc)})
            except Exception:
                pass
            data = None

        if not isinstance(data, dict):
            return {"slot": slot, "value": None, "confidence": 0.0, "source": "none"}

        value = _normalize_slot_value(slot, data.get("value"))
        confidence = data.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        confidence = max(0.0, min(1.0, float(confidence)))
        return {"slot": slot, "value": value, "confidence": confidence, "source": "llm"}

    def _classify_confidence(source: str, confidence: float) -> str:
        if source == "rule" and confidence > 0:
            return "high"
        if confidence >= 0.9:
            return "high"
        if confidence >= 0.6:
            return "medium"
        return "low"

    def _extract_slot_with_confidence(slot: str, text: str) -> Dict[str, Any]:
        rule_result = _extract_rule_slot(slot, text)
        rule_value = _normalize_slot_value(slot, rule_result.get("value"))
        rule_conf = float(rule_result.get("confidence", 0.0) or 0.0)
        if not _is_empty(rule_value) and rule_conf >= 0.9:
            return {"slot": slot, "value": rule_value, "confidence": rule_conf, "source": "rule", "status": "high"}

        llm_result = _extract_llm_slot(slot, text)
        llm_value = _normalize_slot_value(slot, llm_result.get("value"))
        llm_conf = float(llm_result.get("confidence", 0.0) or 0.0)
        llm_status = _classify_confidence("llm", llm_conf)
        if not _is_empty(llm_value):
            return {"slot": slot, "value": llm_value, "confidence": llm_conf, "source": "llm", "status": llm_status}

        return {
            "slot": slot,
            "value": rule_value,
            "confidence": rule_conf if not _is_empty(rule_value) else max(rule_conf, llm_conf),
            "source": "rule" if not _is_empty(rule_value) else "none",
            "status": _classify_confidence("rule", rule_conf) if not _is_empty(rule_value) else "low",
        }

    def _parse_confirmation(text: str) -> str:
        normalized = re.sub(r"[^\w]+", " ", (text or "").lower()).strip()
        if not normalized:
            return "unknown"
        tokens = normalized.split()
        if any(token in YES_WORDS for token in tokens) or normalized in YES_WORDS:
            return "yes"
        if any(token in NO_WORDS for token in tokens) or normalized in NO_WORDS:
            return "no"
        return "unknown"

    def _build_slot_question(slot: str, retries: int) -> str:
        base = QUESTION_MAP.get(slot, f"Please provide your {slot}.")
        if retries <= 0:
            return base
        return base + "\nPlease answer in one short sentence or comma-separated list so I can capture it precisely."

    def _build_confirmation_question(slot: str, value: Any, confidence: float) -> str:
        shown = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
        label = SLOT_LABEL_MAP.get(slot, slot)
        return f"I extracted your {label} as: {shown}. Please confirm yes/no (confidence {confidence:.2f})."

    def _clamp01(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except Exception:
            number = default
        return max(0.0, min(1.0, number))

    def _normalize_question_text(text: Any) -> str:
        raw = re.sub(r"\s+", " ", str(text or "")).strip()
        return raw[:300]

    def _collect_user_answers(messages: List[Any]) -> List[str]:
        answers: List[str] = []
        for msg in serialize_messages_openai(messages):
            if msg.get("role") == "user":
                content = str(msg.get("content", "")).strip()
                if content:
                    answers.append(content)
        return answers

    def _profile_uncertainty_hints(profile: Dict[str, Any]) -> List[str]:
        hints: List[str] = []
        hours_value = profile.get("hours_per_week")
        try:
            hours = float(hours_value) if hours_value is not None else None
        except Exception:
            hours = None
        if hours is not None and hours > 100:
            hints.append("hours_per_week seems unusually high; likely parsing ambiguity (e.g., 15-20 vs 1520).")

        weeks_value = profile.get("timeline_weeks")
        try:
            weeks = float(weeks_value) if weeks_value is not None else None
        except Exception:
            weeks = None
        if weeks is not None and weeks > 78:
            hints.append("timeline_weeks appears long for phase-1 and may need confirmation.")

        location = str(profile.get("location") or "")
        if location.startswith("[") and location.endswith("]"):
            hints.append("location looks serialized as list text and may need normalization.")
        return hints

    def _build_locked_slot_candidate(slot: str, memory: Dict[str, Any]) -> Dict[str, Any]:
        del memory
        template_by_slot = {"compensation_floor": "compensation_floor_focus", "work_mode": "work_mode_focus"}
        template_id = template_by_slot.get(slot, "")
        if not template_id:
            return {}
        signature = f"{slot}:completion_lock"
        question = _render_followup_question(
            {"id": f"locked_{slot}", "signature": signature, "target_slot": slot, "template_id": template_id, "params": {}}
        )
        return {
            "question": question,
            "target_slot": slot,
            "impact": 0.92,
            "uncertainty": 0.62,
            "cost": 0.25,
            "ig": 1.29,
            "reason": "high_impact_gap",
            "signature": signature,
            "template_id": template_id,
            "candidate_id": f"locked_{slot}",
        }

    def _build_followup_candidate_bank(memory: Dict[str, Any], progress_state: ProgressState) -> List[Dict[str, Any]]:
        asked_signatures = {
            str(item).strip().lower() for item in list(progress_state.get("followup_signatures_asked", [])) if str(item).strip()
        }
        candidates: List[Dict[str, Any]] = []

        def add_candidate(
            cid: str,
            signature: str,
            target_slot: str,
            template_id: str,
            description: str,
            rule_priority: float,
            params: Dict[str, Any] | None = None,
        ) -> None:
            if signature.lower() in asked_signatures:
                return
            if target_slot not in FOLLOWUP_TARGET_SLOTS:
                return
            candidates.append(
                {
                    "id": cid,
                    "signature": signature,
                    "target_slot": target_slot,
                    "template_id": template_id,
                    "description": description,
                    "rule_priority": _clamp01(rule_priority, default=0.5),
                    "params": dict(params or {}),
                }
            )

        if _is_empty(memory.get("compensation_floor")):
            add_candidate(
                "compensation_floor_focus",
                "compensation_floor:minimum_monthly",
                "compensation_floor",
                "compensation_floor_focus",
                "Capture compensation floor as a validated numeric constraint.",
                0.90,
            )
        if _is_empty(memory.get("work_mode")):
            add_candidate(
                "work_mode_focus",
                "work_mode:enum_choice",
                "work_mode",
                "work_mode_focus",
                "Capture preferred work mode as a validated enum.",
                0.82,
            )

        candidates.sort(key=lambda item: float(item.get("rule_priority", 0.0)), reverse=True)
        return candidates

    def _build_followup_decision_prompt(
        messages: List[Any],
        memory: Dict[str, Any],
        goal_constraints: Dict[str, Any],
        strategy_tags: List[str],
        progress_state: ProgressState,
        candidate_bank: List[Dict[str, Any]],
    ) -> List[Any]:
        memory_json = json.dumps(memory, ensure_ascii=False)
        goal_constraints_json = json.dumps(goal_constraints, ensure_ascii=False)
        strategy_tag_text = ", ".join(strategy_tags) if strategy_tags else "none"
        user_answers = _collect_user_answers(messages)
        answer_text = "\n".join(f"- {item}" for item in user_answers[-20:]) or "- none"
        candidate_view = [
            {
                "id": item.get("id"),
                "signature": item.get("signature"),
                "target_slot": item.get("target_slot"),
                "description": item.get("description"),
                "rule_priority": item.get("rule_priority"),
            }
            for item in candidate_bank
        ]
        candidate_json = json.dumps(candidate_view, ensure_ascii=False)
        non_basic_asked = int(progress_state.get("non_basic_questions_asked", 0) or 0)
        non_basic_limit = int(progress_state.get("non_basic_questions_limit", 6) or 6)
        budget_remaining = float(progress_state.get("followup_budget_remaining", float(non_basic_limit - non_basic_asked)) or 0.0)

        system_text = (
            "You are a lightweight decision engine for interview planning. "
            "Select one candidate id only. Never generate question text. Return strict JSON."
        )
        human_text = (
            f"Profile JSON:\n{memory_json}\n\n"
            f"Goal/constraints JSON:\n{goal_constraints_json}\n\n"
            f"Strategy tags: {strategy_tag_text}\n"
            f"Latest user answers:\n{answer_text}\n\n"
            f"Candidate bank JSON:\n{candidate_json}\n\n"
            f"Budget status: asked={non_basic_asked}/{non_basic_limit}, remaining={budget_remaining:.2f}\n\n"
            "Objective: maximize IG = impact + uncertainty - cost under budget and avoid redundant coverage.\n"
            "Output JSON schema:\n"
            '{"selected_id":"<candidate id or STOP>",'
            '"impact":<0-1>,'
            '"uncertainty":<0-1>,'
            '"cost":<0-1>,'
            '"reason_code":"<one of high_impact_gap|high_uncertainty|execution_risk|data_quality_risk|already_sufficient>",'
            '"confidence":<0-1>}'
        )
        if HAS_LANGCHAIN_MESSAGES and SystemMessage and HumanMessage:
            return [SystemMessage(content=system_text), HumanMessage(content=human_text)]
        return [{"role": "system", "content": system_text}, {"role": "user", "content": human_text}]

    def _render_followup_question(candidate: Dict[str, Any]) -> str:
        template_id = str(candidate.get("template_id", "")).strip()
        template = FOLLOWUP_QUESTION_TEMPLATES.get(template_id, "")
        params = dict(candidate.get("params", {}) or {})
        if not template:
            return ""
        try:
            return _normalize_question_text(template.format(**params))
        except Exception:
            return _normalize_question_text(template)

    def _generate_followup_candidate(
        messages: List[Any],
        memory: Dict[str, Any],
        goal_constraints: Dict[str, Any],
        strategy_tags: List[str],
        rag_context: str,
        progress_state: ProgressState,
    ) -> Dict[str, Any]:
        del rag_context
        asked_set = {_normalize_question_text(item).lower() for item in list(progress_state.get("asked_questions", []))}
        candidate_bank = _build_followup_candidate_bank(memory, progress_state)
        if not candidate_bank:
            return {
                "question": "",
                "target_slot": "",
                "impact": 0.0,
                "uncertainty": 0.0,
                "cost": 1.0,
                "ig": -1.0,
                "reason": "already_sufficient",
                "signature": "",
                "template_id": "",
                "candidate_id": "",
            }

        default_selected = candidate_bank[0]
        selected_id = str(default_selected.get("id", ""))
        impact = float(default_selected.get("rule_priority", 0.5) or 0.5)
        uncertainty = 0.55
        cost = 0.35
        reason_code = "high_impact_gap"

        if llm:
            prompt = _build_followup_decision_prompt(
                messages,
                memory,
                goal_constraints,
                strategy_tags,
                progress_state,
                candidate_bank,
            )
            try:
                if HAS_LANGCHAIN_MESSAGES and SystemMessage and HumanMessage:
                    response = llm.invoke(prompt)
                else:
                    response = llm.invoke(prompt[0]["content"] + "\n\n" + prompt[1]["content"])
                raw = getattr(response, "content", None) or str(response)
                data = try_parse_json(raw)
            except Exception as exc:
                try:
                    logger.log_event("llm_followup_decision_error", {"error": str(exc)})
                except Exception:
                    pass
                data = None

            if isinstance(data, dict):
                selected_id = str(data.get("selected_id", selected_id)).strip()
                impact = _clamp01(data.get("impact"), default=impact)
                uncertainty = _clamp01(data.get("uncertainty"), default=uncertainty)
                cost = _clamp01(data.get("cost"), default=cost)
                reason_code = str(data.get("reason_code", reason_code)).strip().lower() or reason_code

        if reason_code not in FOLLOWUP_REASON_CODES:
            reason_code = "high_uncertainty"

        if selected_id.upper() == "STOP":
            return {
                "question": "",
                "target_slot": "",
                "impact": impact,
                "uncertainty": uncertainty,
                "cost": cost,
                "ig": impact + uncertainty - cost,
                "reason": reason_code if reason_code else "already_sufficient",
                "signature": "",
                "template_id": "",
                "candidate_id": "STOP",
            }

        selected = next((item for item in candidate_bank if str(item.get("id")) == selected_id), default_selected)
        question = _render_followup_question(selected)
        signature = str(selected.get("signature", "")).strip()

        if not question or question.lower() in asked_set:
            return {
                "question": "",
                "target_slot": str(selected.get("target_slot", "")),
                "impact": impact,
                "uncertainty": uncertainty,
                "cost": cost,
                "ig": impact + uncertainty - cost,
                "reason": "already_sufficient",
                "signature": signature,
                "template_id": str(selected.get("template_id", "")),
                "candidate_id": str(selected.get("id", "")),
            }

        return {
            "question": question,
            "target_slot": str(selected.get("target_slot", "")),
            "impact": impact,
            "uncertainty": uncertainty,
            "cost": cost,
            "ig": impact + uncertainty - cost,
            "reason": reason_code,
            "signature": signature,
            "template_id": str(selected.get("template_id", "")),
            "candidate_id": str(selected.get("id", "")),
        }

    def _evaluate_followup_policy(
        messages: List[Any],
        user_state: UserState,
        progress_state: ProgressState,
        pending_slots: List[str],
        rag_context: str,
    ) -> Dict[str, Any]:
        required_closed = len(pending_slots) == 0
        asked = int(progress_state.get("non_basic_questions_asked", 0) or 0)
        limit = int(progress_state.get("non_basic_questions_limit", 6) or 6)
        budget_remaining = float(progress_state.get("followup_budget_remaining", float(limit - asked)) or 0.0)
        min_ig = float(progress_state.get("followup_min_ig", 0.35) or 0.35)

        if not required_closed:
            return {
                "required_slots_closed": False,
                "followup_should_stop": True,
                "followup_stop_reason": "required_slots_not_closed",
                "followup_candidate": {},
                "followup_last_ig": 0.0,
            }
        if asked >= limit:
            return {
                "required_slots_closed": True,
                "followup_should_stop": True,
                "followup_stop_reason": "followup_limit_reached",
                "followup_candidate": {},
                "followup_last_ig": 0.0,
            }
        if budget_remaining <= 0:
            return {
                "required_slots_closed": True,
                "followup_should_stop": True,
                "followup_stop_reason": "followup_budget_exhausted",
                "followup_candidate": {},
                "followup_last_ig": 0.0,
            }

        memory = user_state.get("profile") or default_profile()
        current_slot = str(progress_state.get("current_slot", "")).strip()
        attempts = dict(progress_state.get("attempts", {}))
        if current_slot in OPTIONAL_SLOT_ORDER and not _is_slot_complete(current_slot, memory.get(current_slot)):
            # Rule A: one round focuses on one slot until completion.
            # Safety fallback: if the previous attempt was still unclear/none, skip this slot and continue.
            if int(attempts.get(current_slot, 0) or 0) < 1:
                locked_candidate = _build_locked_slot_candidate(current_slot, memory)
                if locked_candidate.get("question"):
                    return {
                        "required_slots_closed": True,
                        "followup_should_stop": False,
                        "followup_stop_reason": "",
                        "followup_candidate": locked_candidate,
                        "followup_last_ig": float(locked_candidate.get("ig", 1.29) or 1.29),
                    }

        unresolved_optional = [slot for slot in OPTIONAL_SLOT_ORDER if not _is_slot_complete(slot, memory.get(slot))]
        if not unresolved_optional:
            return {
                "required_slots_closed": True,
                "followup_should_stop": True,
                "followup_stop_reason": "optional_slots_sufficiently_covered",
                "followup_candidate": {},
                "followup_last_ig": 0.0,
            }
        goal_constraints = user_state.get("goal_constraints") or default_goal_constraints()
        strategy_tags = user_state.get("strategy_tags") or []
        candidate = _generate_followup_candidate(messages, memory, goal_constraints, strategy_tags, rag_context, progress_state)
        ig = float(candidate.get("ig", 0.0) or 0.0)
        question = _normalize_question_text(candidate.get("question", ""))
        if not question:
            return {
                "required_slots_closed": True,
                "followup_should_stop": True,
                "followup_stop_reason": str(candidate.get("reason") or "empty_followup_question"),
                "followup_candidate": candidate,
                "followup_last_ig": ig,
            }
        if ig < min_ig:
            return {
                "required_slots_closed": True,
                "followup_should_stop": True,
                "followup_stop_reason": f"low_ig<{min_ig:.2f}",
                "followup_candidate": candidate,
                "followup_last_ig": ig,
            }
        return {
            "required_slots_closed": True,
            "followup_should_stop": False,
            "followup_stop_reason": "",
            "followup_candidate": candidate,
            "followup_last_ig": ig,
        }

    def _build_closure_note(memory: Dict[str, Any], progress_state: ProgressState) -> str:
        reason = str(progress_state.get("followup_stop_reason", "") or "stopping_criteria_met")
        role = str(memory.get("target_role") or "target role")
        city = str(memory.get("location") or "target city")
        summary_notes = [str(item).strip() for item in (memory.get("summary_notes") or []) if str(item).strip()]
        unconfirmed = [str(item).strip() for item in (memory.get("unconfirmed_structured") or []) if str(item).strip()]
        summary_hint = (
            f"Summary memory captured {len(summary_notes)} user signals and will guide recommendations."
            if summary_notes
            else "Summary signals are captured and will guide recommendations."
        )
        unconfirmed_hint = (
            f" Unconfirmed structured items will remain summary-only: {', '.join(unconfirmed[-2:])}."
            if unconfirmed
            else ""
        )

        return (
            f"Structured profile is now anchored on role={role} and city={city}. "
            f"{summary_hint}{unconfirmed_hint}\n\n"
            "If any part of this judgment is off, you can directly challenge it. "
            "I will proceed with the final plan using the current assumptions.\n"
            f"[stopping_reason={reason}]"
        )

    def _plan_prompt(
        memory: Dict[str, Any],
        goal_constraints: Dict[str, Any],
        strategy_tags: List[str],
        messages: List[Any],
        last_user: str,
        rag_context: str,
    ) -> List[Any]:
        structured_profile = {
            "location": memory.get("location"),
            "target_role": memory.get("target_role"),
            "timeline_weeks": memory.get("timeline_weeks"),
            "hours_per_week": memory.get("hours_per_week"),
            "compensation_floor": memory.get("compensation_floor"),
            "work_mode": memory.get("work_mode"),
        }
        memory_json = json.dumps(structured_profile, ensure_ascii=False)
        goal_constraints_json = json.dumps(goal_constraints, ensure_ascii=False)
        strategy_tag_text = ", ".join(strategy_tags) if strategy_tags else "none"
        user_answers = _collect_user_answers(messages)
        user_answers_text = "\n".join(f"- {item}" for item in user_answers) if user_answers else "- none"
        summary_notes = [str(item).strip() for item in (memory.get("summary_notes") or []) if str(item).strip()]
        unconfirmed = [str(item).strip() for item in (memory.get("unconfirmed_structured") or []) if str(item).strip()]
        summary_text = "\n".join(f"- {item}" for item in summary_notes[-20:]) if summary_notes else "- none"
        unconfirmed_text = "\n".join(f"- {item}" for item in unconfirmed[-20:]) if unconfirmed else "- none"

        system_text = "You are a career planning expert. Produce one final, concrete, actionable plan. Do not ask follow-up questions in this step."
        human_text = (
            f"Latest user message:\n{last_user}\n\n"
            f"All user answers in this conversation:\n{user_answers_text}\n\n"
            f"Validated structured profile (JSON):\n{memory_json}\n\n"
            f"Summary memory notes:\n{summary_text}\n\n"
            f"Unconfirmed structured info (do not treat as confirmed facts):\n{unconfirmed_text}\n\n"
            f"Goal and constraints (JSON):\n{goal_constraints_json}\n\n"
            f"Strategy tags:\n{strategy_tag_text}\n\n"
            f"RAG context:\n{rag_context or 'none'}\n\n"
            "Output sections:\n"
            "1) Recommended target direction (1-2 options)\n"
            "2) Key capability gaps\n"
            "3) 8-week action plan\n"
            "4) Resources/projects with rationale\n"
            "5) Validation checklist (measurable)\n"
            "6) Risks and assumptions\n"
        )
        if HAS_LANGCHAIN_MESSAGES and SystemMessage and HumanMessage:
            return [SystemMessage(content=system_text), HumanMessage(content=human_text)]
        return [{"role": "system", "content": system_text}, {"role": "user", "content": human_text}]

    def routing_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        return route_to_state(messages)

    def rag_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        # Question stage: completely disable RAG injection into messages/context.
        # RAG is fetched only in plan_node right before final plan generation.
        return {"messages": messages, "rag_context": ""}

    def profile_extract_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        last_user = _last_user_content(messages)
        if not last_user:
            return {"slot_extract": {}, "slot_confirmation": {}, "goal_constraints_update": {}}

        progress_state = _get_progress_state(state)
        goal_constraints_update = extract_goal_constraints_from_text(last_user)

        if bool(progress_state.get("awaiting_confirmation", False)):
            decision = _parse_confirmation(last_user)
            return {
                "slot_extract": {},
                "slot_confirmation": {"decision": decision},
                "goal_constraints_update": goal_constraints_update,
            }

        pending_slots = list(progress_state.get("pending_slots", []))
        current_slot = str(progress_state.get("current_slot", "")).strip()
        if not current_slot and pending_slots:
            current_slot = pending_slots[0]
        if not current_slot:
            return {"slot_extract": {}, "slot_confirmation": {}, "goal_constraints_update": goal_constraints_update}

        slot_extract = _extract_slot_with_confidence(current_slot, last_user)
        return {"slot_extract": slot_extract, "slot_confirmation": {}, "goal_constraints_update": goal_constraints_update}

    def memory_update_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        last_user = _last_user_content(messages)
        user_state = _get_user_state(state)
        progress_state = _get_progress_state(state)

        memory = user_state.get("profile") or default_profile()
        required_slots = list(progress_state.get("required_slots", REQUIRED_SLOT_ORDER))
        pending_slots = _sync_pending_slots(memory, required_slots, list(progress_state.get("pending_slots", required_slots)))
        attempts = dict(progress_state.get("attempts", {}))
        max_slot_retries = int(progress_state.get("max_slot_retries", 2) or 2)

        current_slot = str(progress_state.get("current_slot", "")).strip()
        if not current_slot and pending_slots:
            current_slot = pending_slots[0]
        current_slot_retries = int(progress_state.get("current_slot_retries", 0) or 0)

        awaiting_confirmation = bool(progress_state.get("awaiting_confirmation", False))
        candidate_slot = str(progress_state.get("candidate_slot", "")).strip()
        candidate_value = progress_state.get("candidate_value")
        candidate_confidence = float(progress_state.get("candidate_confidence", 0.0) or 0.0)
        candidate_source = str(progress_state.get("candidate_source", "")).strip()

        slot_confirmation = state.get("slot_confirmation") or {}
        slot_extract = state.get("slot_extract") or {}
        committed = False

        if awaiting_confirmation:
            decision = str(slot_confirmation.get("decision", "unknown")).strip().lower()
            if decision == "yes" and candidate_slot and not _is_empty(candidate_value):
                memory, committed_now, reject_reason = _try_commit_slot(memory, candidate_slot, candidate_value, last_user)
                if committed_now:
                    pending_slots = _sync_pending_slots(memory, required_slots, pending_slots)
                    current_slot = pending_slots[0] if pending_slots else ""
                    current_slot_retries = 0
                    committed = True
                else:
                    attempts[candidate_slot] = int(attempts.get(candidate_slot, 0)) + 1
                    current_slot = candidate_slot
                    current_slot_retries = 0
                    if reject_reason:
                        logger.log_event("structured_validation_reject", {"slot": candidate_slot, "reason": reject_reason})
                awaiting_confirmation = False
                candidate_slot = ""
                candidate_value = None
                candidate_confidence = 0.0
                candidate_source = ""
            elif decision == "no":
                target_slot = candidate_slot or current_slot
                if target_slot:
                    attempts[target_slot] = int(attempts.get(target_slot, 0)) + 1
                current_slot_retries += 1
                awaiting_confirmation = False
                candidate_slot = ""
                candidate_value = None
                candidate_confidence = 0.0
                candidate_source = ""
        else:
            slot = str(slot_extract.get("slot", "")).strip()
            status = str(slot_extract.get("status", "low")).strip().lower()
            confidence = float(slot_extract.get("confidence", 0.0) or 0.0)
            source = str(slot_extract.get("source", "")).strip().lower()
            value = slot_extract.get("value")
            in_targeted_phase = bool(progress_state.get("question_phase_complete", False))

            if not slot:
                if last_user:
                    memory = _append_summary_note(memory, last_user)
            elif status == "high" and not _is_empty(value):
                memory, committed_now, reject_reason = _try_commit_slot(memory, slot, value, last_user)
                if committed_now:
                    pending_slots = _sync_pending_slots(memory, required_slots, pending_slots)
                    if in_targeted_phase and slot in OPTIONAL_SLOT_ORDER and not _is_slot_complete(slot, memory.get(slot)):
                        current_slot = slot
                    else:
                        current_slot = pending_slots[0] if pending_slots else ""
                    current_slot_retries = 0
                    committed = True
                else:
                    attempts[slot] = int(attempts.get(slot, 0)) + 1
                    current_slot = "" if in_targeted_phase and slot in OPTIONAL_SLOT_ORDER else slot
                    current_slot_retries = 0
                    if reject_reason:
                        logger.log_event("structured_validation_reject", {"slot": slot, "reason": reject_reason})
            elif status == "medium" and not _is_empty(value):
                if in_targeted_phase and slot in OPTIONAL_SLOT_ORDER:
                    memory, committed_now, reject_reason = _try_commit_slot(memory, slot, value, last_user)
                    if committed_now:
                        pending_slots = _sync_pending_slots(memory, required_slots, pending_slots)
                        if not _is_slot_complete(slot, memory.get(slot)):
                            current_slot = slot
                        else:
                            current_slot = pending_slots[0] if pending_slots else ""
                        current_slot_retries = 0
                        committed = True
                    else:
                        attempts[slot] = int(attempts.get(slot, 0)) + 1
                        current_slot = ""
                        current_slot_retries = 0
                        if reject_reason:
                            logger.log_event("structured_validation_reject", {"slot": slot, "reason": reject_reason})
                else:
                    awaiting_confirmation = True
                    candidate_slot = slot
                    candidate_value = value
                    candidate_confidence = confidence
                    candidate_source = source
            else:
                attempts[slot] = int(attempts.get(slot, 0)) + 1
                if in_targeted_phase and slot in OPTIONAL_SLOT_ORDER:
                    # If answer is unclear/none in targeted phase, avoid repeated re-ask and continue.
                    current_slot = ""
                    current_slot_retries = 0
                else:
                    current_slot = slot
                    current_slot_retries += 1
                    if current_slot_retries >= max_slot_retries and slot in pending_slots and len(pending_slots) > 1:
                        pending_slots = [item for item in pending_slots if item != slot] + [slot]
                        current_slot = pending_slots[0]
                        current_slot_retries = 0

        goal_constraints = user_state.get("goal_constraints") or default_goal_constraints()
        goal_constraints_update = state.get("goal_constraints_update") or {}
        merged_goal_constraints = merge_goal_constraints(goal_constraints, goal_constraints_update)
        strategy_tags = derive_strategy_tags(merged_goal_constraints)
        rag_filters = build_rag_filters(merged_goal_constraints)
        if last_user:
            memory = _append_summary_note(memory, last_user)

        pending_slots = _sync_pending_slots(memory, required_slots, pending_slots)
        missing_fields = _missing_fields(memory, required_slots)
        in_targeted_phase = bool(progress_state.get("question_phase_complete", False))
        if current_slot and current_slot not in pending_slots:
            if not (in_targeted_phase and current_slot in OPTIONAL_SLOT_ORDER):
                current_slot = pending_slots[0] if pending_slots else ""
                current_slot_retries = 0
        if not current_slot and pending_slots:
            current_slot = pending_slots[0]
        required_slots_closed = len(pending_slots) == 0
        question_phase_complete = bool(progress_state.get("question_phase_complete", False)) or required_slots_closed

        next_turn_index = int(user_state.get("turn_index", 0) or 0)
        if committed or goal_constraints_update:
            next_turn_index += 1

        next_user_state: UserState = {
            **user_state,
            "profile": memory,
            "goal_constraints": merged_goal_constraints,
            "strategy_tags": strategy_tags,
            "rag_filters": rag_filters,
            "turn_index": next_turn_index,
        }
        next_progress_state: ProgressState = {
            **progress_state,
            "question_stage": "confirming" if awaiting_confirmation else ("targeted" if question_phase_complete else "collecting"),
            "required_slots": required_slots,
            "pending_slots": pending_slots,
            "current_slot": current_slot,
            "current_slot_retries": current_slot_retries,
            "max_slot_retries": max_slot_retries,
            "attempts": attempts,
            "awaiting_confirmation": awaiting_confirmation,
            "candidate_slot": candidate_slot,
            "candidate_value": candidate_value,
            "candidate_confidence": candidate_confidence,
            "candidate_source": candidate_source,
            "question_phase_complete": question_phase_complete,
            "required_slots_closed": required_slots_closed,
            "plan_ready": False,
            "last_node": "memory_update",
        }

        return {
            "user_state": next_user_state,
            "progress_state": next_progress_state,
            "memory": memory,
            "goal_constraints": merged_goal_constraints,
            "strategy_tags": strategy_tags,
            "rag_filters": rag_filters,
            "missing_fields": missing_fields,
            "attempts": attempts,
            "question_stage": next_progress_state["question_stage"],
            "question_phase_complete": bool(question_phase_complete),
        }

    def evaluation_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        rag_context = str(state.get("rag_context", "") or "")
        user_state = _get_user_state(state)
        progress_state = _get_progress_state(state)
        profile = user_state.get("profile") or default_profile()
        required_slots = list(progress_state.get("required_slots", REQUIRED_SLOT_ORDER))
        evaluation = evaluate_profile(profile, required_slots)

        pending_slots = _sync_pending_slots(profile, required_slots, list(progress_state.get("pending_slots", required_slots)))
        current_slot = str(progress_state.get("current_slot", "")).strip()
        if not current_slot and pending_slots:
            current_slot = pending_slots[0]
        if current_slot and current_slot not in pending_slots:
            current_slot = pending_slots[0] if pending_slots else ""

        required_slots_closed = len(pending_slots) == 0
        awaiting_confirmation = bool(progress_state.get("awaiting_confirmation", False))
        question_phase_complete = bool(progress_state.get("question_phase_complete", False)) or required_slots_closed
        followup_policy = {
            "required_slots_closed": required_slots_closed,
            "followup_should_stop": False,
            "followup_stop_reason": "",
            "followup_candidate": {},
            "followup_last_ig": 0.0,
        }
        if question_phase_complete and required_slots_closed and not awaiting_confirmation:
            followup_policy = _evaluate_followup_policy(messages, user_state, progress_state, pending_slots, rag_context)

        next_progress_state: ProgressState = {
            **progress_state,
            "pending_slots": pending_slots,
            "current_slot": current_slot,
            "question_phase_complete": question_phase_complete,
            "required_slots_closed": bool(followup_policy.get("required_slots_closed", required_slots_closed)),
            "followup_should_stop": bool(followup_policy.get("followup_should_stop", False)),
            "followup_stop_reason": str(followup_policy.get("followup_stop_reason", "")),
            "followup_candidate": dict(followup_policy.get("followup_candidate", {})),
            "followup_last_ig": float(followup_policy.get("followup_last_ig", 0.0) or 0.0),
            "question_stage": "confirming" if awaiting_confirmation else ("targeted" if question_phase_complete else "collecting"),
            "plan_ready": bool(question_phase_complete and required_slots_closed and followup_policy.get("followup_should_stop", False)),
            "last_node": "evaluate",
        }
        return {
            "progress_state": next_progress_state,
            "evaluation": evaluation,
            "missing_fields": list(evaluation.get("missing_fields", [])),
            "conversation_complete": False,
        }

    def decide_next_step(state: ChatConversationState) -> str:
        progress_state = _get_progress_state(state)
        pending_slots = list(progress_state.get("pending_slots", []))
        attempts = dict(progress_state.get("attempts", {}))
        max_slot_retries = int(progress_state.get("max_slot_retries", 2) or 2)
        if bool(progress_state.get("awaiting_confirmation", False)):
            return "ask"

        required_slots_closed = bool(progress_state.get("required_slots_closed", len(pending_slots) == 0))
        question_phase_complete = bool(progress_state.get("question_phase_complete", False)) or required_slots_closed

        # Hard condition: required profile slots must be closed before planning.
        if not required_slots_closed or not question_phase_complete:
            if _all_pending_exhausted(pending_slots, attempts, max_slot_retries):
                return "plan"
            return "ask"

        if bool(progress_state.get("followup_should_stop", False)):
            return "plan"

        candidate = progress_state.get("followup_candidate") or {}
        if not _normalize_question_text(candidate.get("question", "")):
            return "plan"

        return "ask_followup"

    def decide_node(state: ChatConversationState) -> str:
        return decide_next_step(state)

    def ask_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        progress_state = _get_progress_state(state)

        asked_questions = list(progress_state.get("asked_questions", []))
        pending_slots = list(progress_state.get("pending_slots", []))
        current_slot = str(progress_state.get("current_slot", "")).strip()
        attempts = dict(progress_state.get("attempts", {}))
        awaiting_confirmation = bool(progress_state.get("awaiting_confirmation", False))
        candidate_slot = str(progress_state.get("candidate_slot", "")).strip()
        candidate_value = progress_state.get("candidate_value")
        candidate_confidence = float(progress_state.get("candidate_confidence", 0.0) or 0.0)

        question = ""
        if awaiting_confirmation and candidate_slot:
            question = _build_confirmation_question(candidate_slot, candidate_value, candidate_confidence)
        else:
            if not current_slot and pending_slots:
                current_slot = pending_slots[0]
            if current_slot:
                question = _build_slot_question(current_slot, int(attempts.get(current_slot, 0)))

        if question:
            _append_assistant(messages, question)
            asked_questions.append(question)

        next_progress_state: ProgressState = {
            **progress_state,
            "asked_questions": asked_questions,
            "question_stage": "confirming" if awaiting_confirmation else "collecting",
            "current_slot": current_slot,
            "last_node": "ask",
        }
        return {
            "messages": messages,
            "next_question": question,
            "progress_state": next_progress_state,
            "asked_questions": asked_questions,
            "attempts": attempts,
            "question_stage": next_progress_state["question_stage"],
            "question_phase_complete": bool(progress_state.get("question_phase_complete", False)),
            "conversation_complete": False,
        }

    def ask_followup_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        progress_state = _get_progress_state(state)

        asked_questions = list(progress_state.get("asked_questions", []))
        signatures_asked = [str(item).strip() for item in list(progress_state.get("followup_signatures_asked", [])) if str(item).strip()]
        candidate = dict(progress_state.get("followup_candidate", {}) or {})
        question = _normalize_question_text(candidate.get("question", ""))
        target_slot = str(candidate.get("target_slot", "")).strip()
        signature = str(candidate.get("signature", "")).strip()
        if target_slot not in FOLLOWUP_TARGET_SLOTS:
            target_slot = ""

        asked = int(progress_state.get("non_basic_questions_asked", 0) or 0)
        budget_remaining = float(progress_state.get("followup_budget_remaining", 0.0) or 0.0)
        cost = _clamp01(candidate.get("cost"), default=1.0)
        budget_cost = max(0.5, cost)

        if question:
            _append_assistant(messages, question)
            asked_questions.append(question)
            asked += 1
            budget_remaining = max(0.0, budget_remaining - budget_cost)
            if signature and signature not in signatures_asked:
                signatures_asked.append(signature)

        next_progress_state: ProgressState = {
            **progress_state,
            "asked_questions": asked_questions,
            "question_stage": "targeted",
            "current_slot": target_slot,
            "current_slot_retries": 0,
            "non_basic_questions_asked": asked,
            "followup_budget_remaining": budget_remaining,
            "followup_candidate": {},
            "followup_signatures_asked": signatures_asked,
            "followup_should_stop": False,
            "followup_stop_reason": "",
            "last_node": "ask_followup",
        }
        return {
            "messages": messages,
            "next_question": question,
            "progress_state": next_progress_state,
            "asked_questions": asked_questions,
            "question_stage": next_progress_state["question_stage"],
            "question_phase_complete": bool(next_progress_state.get("question_phase_complete", False)),
            "conversation_complete": False,
        }

    def question_exit_node(state: ChatConversationState) -> ChatConversationState:
        progress_state = _get_progress_state(state)
        next_progress_state: ProgressState = {
            **progress_state,
            "question_stage": "done",
            "question_phase_complete": True,
            "last_node": "question_exit",
        }
        return {"progress_state": next_progress_state, "question_stage": "done", "question_phase_complete": True}

    def plan_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        user_state = _get_user_state(state)
        progress_state = _get_progress_state(state)
        last_user = _last_user_content(messages)
        memory = user_state.get("profile") or default_profile()
        goal_constraints = user_state.get("goal_constraints") or default_goal_constraints()
        strategy_tags = user_state.get("strategy_tags") or []
        rag_context = state.get("rag_context", "")
        if not rag_context:
            rag_filters = user_state.get("rag_filters") or build_rag_filters(goal_constraints)
            query_hints = list(goal_constraints.get("query_hints", []))
            rag_payload = maybe_attach_rag_context(
                messages,
                "rag",
                rag_filters=rag_filters,
                query_hints=query_hints,
            )
            rag_context = rag_payload.get("rag_context", "") or ""

        prompt = _plan_prompt(memory, goal_constraints, strategy_tags, messages, last_user, rag_context)
        text = _invoke_llm(prompt)
        if not text:
            text = "Sorry, the model is currently unavailable and cannot generate a final plan."
        closure_note = _build_closure_note(memory, progress_state)
        text = f"{closure_note}\n\n{text}"

        _append_assistant(messages, text)
        next_progress_state: ProgressState = {**progress_state, "plan_ready": True, "last_node": "plan"}
        return {"messages": messages, "last_answer": text, "progress_state": next_progress_state, "conversation_complete": True}

    def memory_maintain_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        max_messages = int(state.get("max_messages", 20))
        if len(messages) <= max_messages:
            return {"messages": messages}

        summary_text = ""
        if llm:
            summary_system = "Summarize the conversation briefly for memory. Output 3 bullets max."
            summary_human = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in serialize_messages_openai(messages)
            )
            try:
                if HAS_LANGCHAIN_MESSAGES and SystemMessage and HumanMessage:
                    response = llm.invoke([SystemMessage(content=summary_system), HumanMessage(content=summary_human)])
                else:
                    response = llm.invoke(summary_system + "\n\n" + summary_human)
                summary_text = getattr(response, "content", None) or str(response)
            except Exception:
                summary_text = ""

        trimmed = messages[-max_messages:]
        if summary_text:
            if HAS_LANGCHAIN_MESSAGES and SystemMessage:
                trimmed = [SystemMessage(content=f"Summary: {summary_text}")] + trimmed[1:]
            else:
                trimmed = [{"role": "system", "content": f"Summary: {summary_text}"}] + trimmed[1:]
        return {"messages": trimmed}

    def wrap_node(name: str, fn):
        def _wrapped(state: ChatConversationState):
            try:
                logger.log_event(f"node_started:{name}", {"state_keys": list(state.keys())})
            except Exception:
                pass
            try:
                res = fn(state)
                try:
                    logger.log_event(f"node_finished:{name}", {"result_keys": list(res.keys()) if isinstance(res, dict) else None})
                except Exception:
                    pass
                return res
            except Exception as exc:
                try:
                    logger.log_event(f"node_error:{name}", {"error": str(exc)})
                except Exception:
                    pass
                raise

        return _wrapped

    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        from _local_langgraph import END, StateGraph

    graph = StateGraph(ChatConversationState)
    graph.add_node("route", wrap_node("route", routing_node))
    graph.add_node("rag", wrap_node("rag", rag_node))
    graph.add_node("profile_extract", wrap_node("profile_extract", profile_extract_node))
    graph.add_node("memory_update", wrap_node("memory_update", memory_update_node))
    graph.add_node("evaluate", wrap_node("evaluate", evaluation_node))
    graph.add_node("ask", wrap_node("ask", ask_node))
    graph.add_node("ask_followup", wrap_node("ask_followup", ask_followup_node))
    graph.add_node("question_exit", wrap_node("question_exit", question_exit_node))
    graph.add_node("plan", wrap_node("plan", plan_node))

    if enable_memory:
        graph.add_node("memory_maintain", wrap_node("memory_maintain", memory_maintain_node))

    graph.set_entry_point("route")
    graph.add_edge("route", "rag")
    graph.add_edge("rag", "profile_extract")
    graph.add_edge("profile_extract", "memory_update")
    graph.add_edge("memory_update", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        decide_node,
        {"ask": "ask", "ask_followup": "ask_followup", "question_exit": "question_exit", "plan": "plan"},
    )
    graph.add_edge("question_exit", "plan")

    if enable_memory:
        graph.add_edge("ask", "memory_maintain")
        graph.add_edge("ask_followup", "memory_maintain")
        graph.add_edge("plan", "memory_maintain")
        graph.add_edge("memory_maintain", END)
    else:
        graph.add_edge("ask", END)
        graph.add_edge("ask_followup", END)
        graph.add_edge("plan", END)

    try:
        logger.start_run("message_flow")
        logger.log_event("langsmith_config", langsmith_status)
        logger.log_event("llm_config", {"enabled": bool(llm), "model": os.getenv("OPENAI_MODEL")})
        logger.log_event("mcp_config", {"servers": list(get_mcp_servers().keys())})
    except Exception:
        pass

    return graph


def decide_next_from_profile(profile: Dict[str, Any], attempts: Dict[str, int] | None = None) -> str:
    attempts = attempts or {}
    missing: List[str] = []
    for field in PROFILE_REQUIRED_FIELDS:
        value = profile.get(field)
        if value is None:
            missing.append(field)
        elif isinstance(value, list) and not value:
            missing.append(field)
        elif isinstance(value, str) and not value.strip():
            missing.append(field)
    askable = any(attempts.get(field, 0) < 2 for field in missing)
    return "ask" if missing and askable else "plan"
