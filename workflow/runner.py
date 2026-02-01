from __future__ import annotations

import json

from langsmith_integration import get_default_logger
from .graph_chat import build_message_graph
from .graph_career import build_graph
from .messages import serialize_messages_openai
from .extractor import default_extracted

logger = get_default_logger()


def run_career_conversation() -> None:
    print("Welcome to the AI Career Path Agent (LangGraph workflow)!\n")
    graph = build_graph().compile()
    try:
        graph.invoke({})
    finally:
        try:
            logger.end_run("completed")
        except Exception:
            pass


def run_message_flow(enable_memory: bool = True) -> None:
    print("Message Flow Runner (chat -> extract -> memory)\n")
    user_text = input("User: ").strip()
    initial_messages = [{"role": "user", "content": user_text}]
    graph = build_message_graph(enable_memory=enable_memory).compile()
    try:
        result = graph.invoke({"messages": initial_messages, "enable_memory_maintain": enable_memory})
    finally:
        try:
            logger.end_run("completed")
        except Exception:
            pass

    messages_out = serialize_messages_openai(result.get("messages", []))
    extracted = result.get("extracted", default_extracted(confidence=0.2))
    evaluation = result.get("evaluation", {})
    output = {"messages": messages_out, "extracted": extracted, "evaluation": evaluation}
    print(json.dumps(output, ensure_ascii=False, indent=2))
