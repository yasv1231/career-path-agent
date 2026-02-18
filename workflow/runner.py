from __future__ import annotations

import json

from langsmith_integration import get_default_logger
from .graph_chat import build_message_graph
from .messages import serialize_messages_openai
from .extractor import default_extracted, default_profile
from .goal_constraints import default_goal_constraints

logger = get_default_logger()


def run_message_flow(enable_memory: bool = True) -> None:
    print("Message Flow Runner (multi-turn: memory + follow-up)\n")
    graph = build_message_graph(enable_memory=enable_memory).compile()
    state = {
        "messages": [],
        "enable_memory_maintain": enable_memory,
        "memory": default_profile(),
        "goal_constraints": default_goal_constraints(),
        "asked_questions": [],
        "attempts": {},
    }

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
                "memory": state.get("memory", {}),
                "goal_constraints": state.get("goal_constraints", {}),
                "strategy_tags": state.get("strategy_tags", []),
                "rag_filters": state.get("rag_filters", {}),
                "missing_fields": state.get("missing_fields", []),
                "extracted": extracted,
                "evaluation": evaluation,
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
            break

    try:
        logger.end_run("completed")
    except Exception:
        pass
