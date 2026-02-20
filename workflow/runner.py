from __future__ import annotations

import json

from langsmith_integration import get_default_logger
from .graph_chat import build_message_graph
from .messages import serialize_messages_openai
from .extractor import default_extracted, default_profile
from .goal_constraints import default_goal_constraints
from .state import PROFILE_REQUIRED_FIELDS

logger = get_default_logger()


def run_message_flow(enable_memory: bool = True) -> None:
    print("Message Flow Runner (multi-turn: memory + follow-up)\n")
    graph = build_message_graph(enable_memory=enable_memory).compile()
    state = {
        "messages": [],
        "max_messages": 12,
        "enable_memory_maintain": enable_memory,
        "user_state": {
            "profile": default_profile(),
            "goal_constraints": default_goal_constraints(),
            "strategy_tags": [],
            "rag_filters": {},
            "extracted": {},
            "turn_index": 0,
        },
        "progress_state": {
            "question_stage": "collecting",
            "asked_questions": [],
            "attempts": {},
            "required_slots": list(PROFILE_REQUIRED_FIELDS),
            "optional_slots": ["compensation_floor", "work_mode"],
            "pending_slots": list(PROFILE_REQUIRED_FIELDS),
            "current_slot": PROFILE_REQUIRED_FIELDS[0] if PROFILE_REQUIRED_FIELDS else "",
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
        },
        "memory": default_profile(),
        "goal_constraints": default_goal_constraints(),
        "asked_questions": [],
        "attempts": {},
        "question_stage": "collecting",
        "question_phase_complete": False,
    }

    # Start by asking basic questions before receiving user content.
    state = graph.invoke(state)
    messages_bootstrap = serialize_messages_openai(state.get("messages", []))
    first_assistant = ""
    for msg in reversed(messages_bootstrap):
        if msg.get("role") == "assistant":
            first_assistant = msg.get("content", "")
            break
    if first_assistant:
        print(f"Assistant: {first_assistant}\n")

    while True:
        user_text = input("User: ").strip()
        if not user_text:
            continue
        messages = list(state.get("messages", []))
        messages.append({"role": "user", "content": user_text})
        state["messages"] = messages

        state = graph.invoke(state)

        messages_out = serialize_messages_openai(state.get("messages", []))
        last_assistant = ""
        for msg in reversed(messages_out):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break

        if last_assistant:
            print(f"Assistant: {last_assistant}\n")

        if state.get("conversation_complete"):
            extracted = state.get("extracted", default_extracted(confidence=0.2))
            evaluation = state.get("evaluation", {})
            output = {
                "messages": messages_out,
                "user_state": state.get("user_state", {}),
                "progress_state": state.get("progress_state", {}),
                "memory": state.get("memory", {}),
                "goal_constraints": state.get("goal_constraints", {}),
                "strategy_tags": state.get("strategy_tags", []),
                "rag_filters": state.get("rag_filters", {}),
                "missing_fields": state.get("missing_fields", []),
                "question_stage": state.get("question_stage", ""),
                "question_phase_complete": bool(state.get("question_phase_complete", False)),
                "extracted": extracted,
                "evaluation": evaluation,
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
            break

    try:
        logger.end_run("completed")
    except Exception:
        pass
