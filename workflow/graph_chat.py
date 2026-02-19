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
    "education": "What is your education background and major?",
    "skills": "What are your current core skills (for example: Python, SQL, Excel)?",
    "interests": "Which business domains or topics are you most interested in?",
    "hours_per_week": "How many hours per week can you invest consistently?",
    "experience_level": "What is your current experience level (entry/junior/mid/senior)?",
    "location": "Which location or city are you targeting?",
    "timeline_weeks": "In how many weeks do you want to complete phase-1 transition?",
    "target_role": "Which target role do you want to pursue first?",
    "industry": "Do you have a preferred industry? (optional)",
    "constraints": "Any key constraints (budget/time/language/location)?",
    "goals": "What measurable goal do you want to achieve first?",
}

SLOT_LABEL_MAP: Dict[str, str] = {
    "education": "education background",
    "skills": "core skills",
    "interests": "interest domains",
    "hours_per_week": "weekly available hours",
    "experience_level": "experience level",
    "location": "target location",
    "timeline_weeks": "transition timeline (weeks)",
    "target_role": "target role",
    "industry": "preferred industry",
    "constraints": "constraints",
    "goals": "primary goals",
}

LIST_SLOTS = {"skills", "interests", "constraints", "goals"}
NUMBER_SLOTS = {"hours_per_week", "timeline_weeks"}
REQUIRED_SLOT_ORDER = list(PROFILE_REQUIRED_FIELDS)
OPTIONAL_SLOT_ORDER = ["target_role", "constraints", "goals", "industry"]

YES_WORDS = {"yes", "y", "correct", "right", "sure", "ok", "okay", "对", "是", "没错", "正确", "好", "可以"}
NO_WORDS = {"no", "n", "wrong", "not", "nope", "不", "不是", "不对", "错", "否"}
UNCERTAIN_WORDS = {"not sure", "unsure", "unknown", "n/a", "idk", "不知道", "不确定", "暂时没有", "随便"}


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
        normalized = raw.replace("，", ",").replace("；", ",").replace("、", ",")
        parts = [part.strip() for part in re.split(r"[,/\n;|]+", normalized) if part.strip()]
        unique: List[str] = []
        for part in parts:
            if part not in unique:
                unique.append(part)
        return unique

    def _extract_hours_per_week(text: str) -> float | None:
        patterns = [
            re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\s*(?:per|a)?\s*week", re.IGNORECASE),
            re.compile(r"每周(?:可投入|投入|学习)?\s*(\d+(?:\.\d+)?)\s*(?:小时|h)"),
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
        week_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:weeks?|wks?|week|周|星期)", lowered, re.IGNORECASE)
        if week_match:
            return float(week_match.group(1))
        month_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:months?|month|个月|月)", lowered, re.IGNORECASE)
        if month_match:
            return float(month_match.group(1)) * 4.345
        year_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|year|年)", lowered, re.IGNORECASE)
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
            try:
                return float(value)
            except Exception:
                return None
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

        lowered = raw.lower()
        if slot == "hours_per_week":
            value = _extract_hours_per_week(raw)
            return {"slot": slot, "value": value, "confidence": 0.98 if value is not None else 0.2, "source": "rule"}

        if slot == "timeline_weeks":
            value = _extract_timeline_weeks(raw)
            return {"slot": slot, "value": value, "confidence": 0.98 if value is not None else 0.2, "source": "rule"}

        if slot == "experience_level":
            level_map = {
                "entry": ["entry", "new grad", "beginner", "入门", "应届"],
                "junior": ["junior", "初级", "1-2 years", "1 year", "2 years"],
                "mid": ["mid", "intermediate", "中级", "3 years", "4 years", "5 years"],
                "senior": ["senior", "lead", "expert", "高级", "资深", "10 years"],
            }
            for level, aliases in level_map.items():
                if any(alias in lowered for alias in aliases):
                    return {"slot": slot, "value": level, "confidence": 0.96, "source": "rule"}
            if len(raw.split()) <= 4:
                return {"slot": slot, "value": raw, "confidence": 0.75, "source": "rule"}
            return {"slot": slot, "value": None, "confidence": 0.25, "source": "rule"}

        if slot in LIST_SLOTS:
            values = _split_to_list(raw)
            return {"slot": slot, "value": values if values else None, "confidence": 0.95 if values else 0.2, "source": "rule"}

        if slot == "location":
            if len(raw) <= 40:
                return {"slot": slot, "value": raw, "confidence": 0.9, "source": "rule"}
            return {"slot": slot, "value": None, "confidence": 0.3, "source": "rule"}

        if slot == "education":
            markers = ["bachelor", "master", "phd", "degree", "本科", "硕士", "博士", "大专", "学历"]
            if any(marker in lowered for marker in markers):
                return {"slot": slot, "value": raw, "confidence": 0.94, "source": "rule"}
            if len(raw) <= 80:
                return {"slot": slot, "value": raw, "confidence": 0.72, "source": "rule"}
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
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", (text or "").lower()).strip()
        if not normalized:
            return "unknown"
        tokens = normalized.split()
        if any(token in YES_WORDS for token in tokens) or normalized in YES_WORDS:
            return "yes"
        if any(token in NO_WORDS for token in tokens) or normalized in NO_WORDS:
            return "no"
        if any(marker in normalized for marker in ("是的", "对的", "没问题")):
            return "yes"
        if any(marker in normalized for marker in ("不是", "不对", "不准确")):
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

    def _plan_prompt(memory: Dict[str, Any], goal_constraints: Dict[str, Any], strategy_tags: List[str], last_user: str, rag_context: str) -> List[Any]:
        memory_json = json.dumps(memory, ensure_ascii=False)
        goal_constraints_json = json.dumps(goal_constraints, ensure_ascii=False)
        strategy_tag_text = ", ".join(strategy_tags) if strategy_tags else "none"

        system_text = "You are a career planning expert. Produce one final, concrete, actionable plan. Do not ask follow-up questions in this step."
        human_text = (
            f"Latest user message:\n{last_user}\n\n"
            f"User profile (JSON):\n{memory_json}\n\n"
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
        route = state.get("route", "direct")
        user_state = _get_user_state(state)
        memory = user_state.get("profile") or {}
        force_rag = bool(memory.get("target_role"))

        goal_constraints = user_state.get("goal_constraints") or default_goal_constraints()
        rag_filters = user_state.get("rag_filters") or build_rag_filters(goal_constraints)
        query_hints = list(goal_constraints.get("query_hints", []))
        return maybe_attach_rag_context(
            messages,
            "rag" if (route == "rag" or force_rag) else route,
            rag_filters=rag_filters,
            query_hints=query_hints,
        )

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
                memory = merge_profile(memory, {candidate_slot: candidate_value})
                pending_slots = _sync_pending_slots(memory, required_slots, pending_slots)
                current_slot = pending_slots[0] if pending_slots else ""
                current_slot_retries = 0
                awaiting_confirmation = False
                candidate_slot = ""
                candidate_value = None
                candidate_confidence = 0.0
                candidate_source = ""
                committed = True
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

            if not slot:
                pass
            elif status == "high" and not _is_empty(value):
                memory = merge_profile(memory, {slot: value})
                pending_slots = _sync_pending_slots(memory, required_slots, pending_slots)
                current_slot = pending_slots[0] if pending_slots else ""
                current_slot_retries = 0
                committed = True
            elif status == "medium" and not _is_empty(value):
                awaiting_confirmation = True
                candidate_slot = slot
                candidate_value = value
                candidate_confidence = confidence
                candidate_source = source
            else:
                attempts[slot] = int(attempts.get(slot, 0)) + 1
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

        pending_slots = _sync_pending_slots(memory, required_slots, pending_slots)
        missing_fields = _missing_fields(memory, required_slots)
        if current_slot and current_slot not in pending_slots:
            current_slot = pending_slots[0] if pending_slots else ""
            current_slot_retries = 0
        if not current_slot and pending_slots:
            current_slot = pending_slots[0]

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
            "question_stage": "confirming" if awaiting_confirmation else "collecting",
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
            "plan_ready": len(missing_fields) == 0 and bool(progress_state.get("question_phase_complete", False)),
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
            "question_phase_complete": bool(next_progress_state.get("question_phase_complete", False)),
        }

    def evaluation_node(state: ChatConversationState) -> ChatConversationState:
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

        next_progress_state: ProgressState = {
            **progress_state,
            "pending_slots": pending_slots,
            "current_slot": current_slot,
            "plan_ready": len(pending_slots) == 0 and bool(progress_state.get("question_phase_complete", False)),
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

        if bool(progress_state.get("question_phase_complete", False)):
            return "plan"
        if bool(progress_state.get("awaiting_confirmation", False)):
            return "ask"
        if not pending_slots:
            return "question_exit"
        if _all_pending_exhausted(pending_slots, attempts, max_slot_retries):
            return "question_exit"
        return "ask"

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
            "question_phase_complete": False,
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

        prompt = _plan_prompt(memory, goal_constraints, strategy_tags, last_user, rag_context)
        text = _invoke_llm(prompt)
        if not text:
            text = "Sorry, the model is currently unavailable and cannot generate a final plan."

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
    graph.add_node("question_exit", wrap_node("question_exit", question_exit_node))
    graph.add_node("plan", wrap_node("plan", plan_node))

    if enable_memory:
        graph.add_node("memory_maintain", wrap_node("memory_maintain", memory_maintain_node))

    graph.set_entry_point("route")
    graph.add_edge("route", "rag")
    graph.add_edge("rag", "profile_extract")
    graph.add_edge("profile_extract", "memory_update")
    graph.add_edge("memory_update", "evaluate")
    graph.add_conditional_edges("evaluate", decide_node, {"ask": "ask", "question_exit": "question_exit", "plan": "plan"})
    graph.add_edge("question_exit", "plan")

    if enable_memory:
        graph.add_edge("ask", "memory_maintain")
        graph.add_edge("plan", "memory_maintain")
        graph.add_edge("memory_maintain", END)
    else:
        graph.add_edge("ask", END)
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
