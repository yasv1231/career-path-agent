from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from langsmith_integration import configure_langsmith, get_chat_model, get_default_logger
from .evaluator import evaluate_profile
from .extractor import coerce_profile_v1, default_profile, merge_profile, try_parse_json
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
from .state import ChatConversationState, PROFILE_REQUIRED_FIELDS, PROFILE_SCHEMA_V1

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

BASIC_QUESTION_FIELDS: List[str] = [
    "education",
    "skills",
    "interests",
    "experience_level",
    "location",
    "hours_per_week",
    "timeline_weeks",
    "target_role",
]


def build_message_graph(enable_memory: bool = True):
    langsmith_status = configure_langsmith("message-flow")
    llm = get_chat_model()

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
                    f"{item.get('role', 'user')}: {item.get('content', '')}"
                    for item in serialize_messages_openai(messages)
                )
                response = llm.invoke(prompt)
            return getattr(response, "content", None) or str(response)
        except Exception as exc:
            try:
                logger.log_event("llm_chat_error", {"error": str(exc)})
            except Exception:
                pass
            return ""

    def _prune_profile(update: Dict[str, Any]) -> Dict[str, Any]:
        pruned: Dict[str, Any] = {}
        for key, value in update.items():
            if key == "schema_version":
                continue
            if isinstance(value, list) and value:
                pruned[key] = value
            elif isinstance(value, str) and value.strip():
                pruned[key] = value.strip()
            elif isinstance(value, (int, float)) and value is not None:
                pruned[key] = value
            elif value is not None and value != "":
                pruned[key] = value
        return pruned

    def _missing_fields(profile: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        for field in PROFILE_REQUIRED_FIELDS:
            value = profile.get(field)
            if value is None:
                missing.append(field)
            elif isinstance(value, list) and not value:
                missing.append(field)
            elif isinstance(value, str) and not value.strip():
                missing.append(field)
        return missing

    def _missing_basic_fields(profile: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        for field in BASIC_QUESTION_FIELDS:
            value = profile.get(field)
            if value is None:
                missing.append(field)
            elif isinstance(value, list) and not value:
                missing.append(field)
            elif isinstance(value, str) and not value.strip():
                missing.append(field)
        return missing

    def _has_askable_missing(missing: List[str], attempts: Dict[str, int]) -> bool:
        return any(attempts.get(field, 0) < 2 for field in missing)

    def _is_empty(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, list):
            return len(value) == 0
        return False

    def _extract_competency_gaps(rag_context: str, skills: List[str]) -> List[str]:
        if not rag_context:
            return []
        skill_set = {str(skill).strip().lower() for skill in skills if str(skill).strip()}
        if not skill_set:
            return []

        competency_terms: List[str] = []
        in_competency = False
        for raw_line in rag_context.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("competency model"):
                in_competency = True
                continue
            if in_competency and line.endswith(":") and not line.startswith("-"):
                break
            if in_competency and line.startswith("-"):
                text = line[1:].strip()
                if ":" in text:
                    text = text.split(":", 1)[1].strip()
                parts = re.split(r",|;|/|\band\b", text)
                for part in parts:
                    token = re.sub(r"\s+", " ", part.strip().lower())
                    if 3 <= len(token) <= 48:
                        competency_terms.append(token)

        gaps: List[str] = []
        seen = set()
        for token in competency_terms:
            if token in seen:
                continue
            seen.add(token)
            if not any(token in skill or skill in token for skill in skill_set):
                gaps.append(token)
        return gaps[:5]

    def _build_basic_questions(
        missing: List[str],
        asked_questions: List[str],
        attempts: Dict[str, int],
    ) -> Tuple[str, List[str], Dict[str, int]]:
        questions: List[str] = []
        asked_now: List[str] = []
        attempts_next = dict(attempts)
        for field in missing:
            if attempts_next.get(field, 0) >= 2:
                continue
            question = QUESTION_MAP.get(field)
            if question and question not in asked_questions:
                questions.append(question)
                asked_now.append(question)
                attempts_next[field] = attempts_next.get(field, 0) + 1
            if len(questions) >= 2:
                break

        if not questions:
            return "", [], attempts_next
        formatted = "\n".join(f"{idx + 1}. {question}" for idx, question in enumerate(questions))
        return "Before planning, please clarify:\n" + formatted, asked_now, attempts_next

    def _build_deep_questions(
        profile: Dict[str, Any],
        attempts: Dict[str, int],
        asked_questions: List[str],
        rag_context: str,
    ) -> Tuple[str, List[str], Dict[str, int]]:
        prompts: List[Tuple[str, str, int]] = []
        if _is_empty(profile.get("target_role")):
            prompts.append(
                (
                    "deep:target_role",
                    "Confirm your target role (1-2 options).",
                    2,
                )
            )
        if _is_empty(profile.get("goals")):
            prompts.append(
                (
                    "deep:goals",
                    "Provide one measurable goal (for example: complete 2 portfolio projects in 8 weeks).",
                    2,
                )
            )
        if _is_empty(profile.get("constraints")):
            prompts.append(
                (
                    "deep:constraints",
                    "List your top constraints (time/budget/location/language), ordered by priority.",
                    2,
                )
            )

        gaps = _extract_competency_gaps(rag_context, profile.get("skills", []))
        if gaps:
            prompts.append(
                (
                    "deep:competency_gap",
                    f"Which gap should be prioritized first: {' / '.join(gaps[:3])} ?",
                    1,
                )
            )

        picked: List[str] = []
        asked_now: List[str] = []
        attempts_next = dict(attempts)
        for key, question, limit in prompts:
            if attempts_next.get(key, 0) >= limit:
                continue
            if question in asked_questions:
                continue
            picked.append(question)
            asked_now.append(question)
            attempts_next[key] = attempts_next.get(key, 0) + 1
            if len(picked) >= 2:
                break

        if not picked:
            return "", [], attempts_next
        return (
            "Before the final one-shot plan, I need two more details:\n"
            + "\n".join(f"{idx + 1}. {question}" for idx, question in enumerate(picked)),
            asked_now,
            attempts_next,
        )

    def _has_targeted_question(
        profile: Dict[str, Any],
        attempts: Dict[str, int],
        asked_questions: List[str],
        rag_context: str,
    ) -> bool:
        question, _, _ = _build_deep_questions(profile, attempts, asked_questions, rag_context)
        return bool(question.strip())

    def _needs_deep_dive(profile: Dict[str, Any], attempts: Dict[str, int], rag_context: str) -> bool:
        deep_fields = ("target_role", "goals", "constraints")
        for field in deep_fields:
            if _is_empty(profile.get(field)) and attempts.get(f"deep:{field}", 0) < 2:
                return True
        gaps = _extract_competency_gaps(rag_context, profile.get("skills", []))
        if gaps and attempts.get("deep:competency_gap", 0) < 1:
            return True
        return False

    def _plan_prompt(
        memory: Dict[str, Any],
        goal_constraints: Dict[str, Any],
        strategy_tags: List[str],
        last_user: str,
        rag_context: str,
    ) -> List[Any]:
        memory_json = json.dumps(memory, ensure_ascii=False)
        goal_constraints_json = json.dumps(goal_constraints, ensure_ascii=False)
        strategy_tag_text = ", ".join(strategy_tags) if strategy_tags else "none"

        system_text = (
            "You are a career planning expert. Produce one final, concrete, actionable plan. "
            "Do not ask follow-up questions in this step."
        )
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
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": human_text},
        ]

    def routing_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        return route_to_state(messages)

    def rag_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        route = state.get("route", "direct")
        memory = state.get("memory") or {}
        force_rag = bool(memory.get("target_role"))

        goal_constraints = state.get("goal_constraints") or default_goal_constraints()
        rag_filters = state.get("rag_filters") or build_rag_filters(goal_constraints)
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
            return {"profile_update": {}, "goal_constraints_update": {}}

        goal_constraints_update = extract_goal_constraints_from_text(last_user)
        if not llm:
            return {"profile_update": {}, "goal_constraints_update": goal_constraints_update}

        schema_json = json.dumps(PROFILE_SCHEMA_V1, ensure_ascii=True)
        base_system = (
            "You are a strict JSON extraction engine. "
            "Output JSON only with schema_version=profile_v1. "
            "Unknown fields should be null or empty array."
        )
        base_prompt = f"User text:\n{last_user}\n\nSchema:\n{schema_json}"
        attempts = 0
        last_data: Any = None
        while attempts < 2:
            attempts += 1
            try:
                if HAS_LANGCHAIN_MESSAGES and SystemMessage and HumanMessage:
                    response = llm.invoke(
                        [
                            SystemMessage(content=base_system),
                            HumanMessage(content=base_prompt),
                        ]
                    )
                else:
                    response = llm.invoke(base_system + "\n\n" + base_prompt)
                text = getattr(response, "content", None) or str(response)
            except Exception as exc:
                try:
                    logger.log_event("llm_profile_extract_error", {"error": str(exc)})
                except Exception:
                    pass
                text = ""

            data = try_parse_json(text)
            last_data = data
            if isinstance(data, dict):
                break

        update = coerce_profile_v1(last_data)
        return {
            "profile_update": _prune_profile(update),
            "goal_constraints_update": goal_constraints_update,
        }

    def memory_update_node(state: ChatConversationState) -> ChatConversationState:
        memory = state.get("memory") or default_profile()
        update = state.get("profile_update") or {}
        merged_profile = merge_profile(memory, update)

        goal_constraints = state.get("goal_constraints") or default_goal_constraints()
        goal_constraints_update = state.get("goal_constraints_update") or {}
        merged_goal_constraints = merge_goal_constraints(goal_constraints, goal_constraints_update)
        strategy_tags = derive_strategy_tags(merged_goal_constraints)
        rag_filters = build_rag_filters(merged_goal_constraints)

        return {
            "memory": merged_profile,
            "goal_constraints": merged_goal_constraints,
            "strategy_tags": strategy_tags,
            "rag_filters": rag_filters,
            "missing_fields": _missing_fields(merged_profile),
        }

    def evaluation_node(state: ChatConversationState) -> ChatConversationState:
        memory = state.get("memory") or default_profile()
        evaluation = evaluate_profile(memory, PROFILE_REQUIRED_FIELDS)
        missing = evaluation.get("missing_fields", [])
        return {
            "evaluation": evaluation,
            "missing_fields": missing,
            "conversation_complete": evaluation.get("valid", False),
        }

    def decide_next_step(state: ChatConversationState) -> str:
        profile = state.get("memory") or default_profile()
        attempts = state.get("attempts") or {}
        asked_questions = state.get("asked_questions") or []
        rag_context = state.get("rag_context", "")
        stage = str(state.get("question_stage", "basic")).strip().lower()
        targeted_rounds = int(state.get("targeted_rounds", 0) or 0)
        max_targeted_rounds = int(state.get("max_targeted_rounds", 2) or 2)

        if bool(state.get("question_phase_complete")):
            return "plan"

        missing_basic = _missing_basic_fields(profile)
        if stage == "basic":
            if missing_basic and _has_askable_missing(missing_basic, attempts):
                return "ask"
            if targeted_rounds < max_targeted_rounds and _has_targeted_question(
                profile,
                attempts,
                asked_questions,
                rag_context,
            ):
                return "ask"
            return "question_exit"

        if stage == "targeted":
            if targeted_rounds >= max_targeted_rounds:
                return "question_exit"
            if _has_targeted_question(profile, attempts, asked_questions, rag_context):
                return "ask"
            return "question_exit"

        return "question_exit"

    def decide_node(state: ChatConversationState) -> str:
        return decide_next_step(state)

    def ask_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        asked_questions = state.get("asked_questions", [])
        attempts = state.get("attempts", {})
        memory = state.get("memory") or default_profile()
        rag_context = state.get("rag_context", "")
        stage = str(state.get("question_stage", "basic")).strip().lower()
        targeted_rounds = int(state.get("targeted_rounds", 0) or 0)
        max_targeted_rounds = int(state.get("max_targeted_rounds", 2) or 2)

        question = ""
        asked_now: List[str] = []
        attempts_next = dict(attempts)
        next_stage = stage if stage in {"basic", "targeted"} else "basic"

        if next_stage == "basic":
            missing_basic = _missing_basic_fields(memory)
            if missing_basic and _has_askable_missing(missing_basic, attempts_next):
                question, asked_now, attempts_next = _build_basic_questions(
                    missing_basic,
                    asked_questions,
                    attempts_next,
                )
                if not question:
                    next_stage = "targeted"
            else:
                next_stage = "targeted"

        if not question and next_stage == "targeted" and targeted_rounds < max_targeted_rounds:
            question, asked_now, attempts_next = _build_deep_questions(
                memory,
                attempts_next,
                asked_questions,
                rag_context,
            )
            if question:
                targeted_rounds += 1

        if question:
            _append_assistant(messages, question)
        return {
            "messages": messages,
            "next_question": question,
            "asked_questions": asked_questions + asked_now,
            "attempts": attempts_next,
            "question_stage": next_stage,
            "targeted_rounds": targeted_rounds,
            "max_targeted_rounds": max_targeted_rounds,
            "question_phase_complete": False,
            "conversation_complete": False,
        }

    def question_exit_node(state: ChatConversationState) -> ChatConversationState:
        return {
            "question_stage": "done",
            "question_phase_complete": True,
        }

    def plan_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        last_user = _last_user_content(messages)
        memory = state.get("memory") or default_profile()
        goal_constraints = state.get("goal_constraints") or default_goal_constraints()
        strategy_tags = state.get("strategy_tags") or []
        rag_context = state.get("rag_context", "")

        prompt = _plan_prompt(memory, goal_constraints, strategy_tags, last_user, rag_context)
        text = _invoke_llm(prompt)
        if not text:
            text = "Sorry, the model is currently unavailable and cannot generate a final plan."

        _append_assistant(messages, text)
        return {
            "messages": messages,
            "last_answer": text,
            "conversation_complete": True,
        }

    def memory_maintain_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        max_messages = int(state.get("max_messages", 20))
        if len(messages) <= max_messages:
            return {"messages": messages}

        summary_text = ""
        if llm:
            summary_system = "Summarize the conversation briefly for memory. Output 3 bullets max."
            summary_human = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in serialize_messages_openai(messages)
            )
            try:
                if HAS_LANGCHAIN_MESSAGES and SystemMessage and HumanMessage:
                    response = llm.invoke(
                        [
                            SystemMessage(content=summary_system),
                            HumanMessage(content=summary_human),
                        ]
                    )
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
                    logger.log_event(
                        f"node_finished:{name}",
                        {"result_keys": list(res.keys()) if isinstance(res, dict) else None},
                    )
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

    graph.add_conditional_edges(
        "evaluate",
        decide_node,
        {
            "ask": "ask",
            "question_exit": "question_exit",
            "plan": "plan",
        },
    )
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
