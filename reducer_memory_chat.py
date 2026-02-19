from __future__ import annotations

import json
import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


load_dotenv()


class State(TypedDict, total=False):
    # Reducer memory: every node return {"messages": [...]} will be appended.
    messages: Annotated[List[BaseMessage], operator.add]
    extracts: Annotated[List[str], operator.add]


def _build_llm() -> ChatOpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_API_BASE", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing")
    return ChatOpenAI(
        model="qwen-plus",
        api_key=api_key,
        base_url=base_url or None,
        temperature=0,
    )


def chat_with_model(state: State) -> State:
    llm = _build_llm()
    history = state.get("messages", [])
    if not history:
        return {}
    response = llm.invoke(history)
    return {"messages": [response]}


def convert_messages(state: State) -> State:
    llm = _build_llm()
    history = state.get("messages", [])
    if not history:
        return {}

    latest_ai = None
    for msg in reversed(history):
        if isinstance(msg, AIMessage):
            latest_ai = msg
            break
    if latest_ai is None:
        return {}

    extraction_prompt = (
        "You are a data extraction specialist. "
        "Extract key information from the text and output compact JSON only."
    )
    prompt = [
        SystemMessage(content=extraction_prompt),
        HumanMessage(content=str(latest_ai.content)),
    ]
    extracted = llm.invoke(prompt)
    return {
        "messages": [AIMessage(content=f"[EXTRACTED]\n{extracted.content}")],
        "extracts": [str(extracted.content)],
    }


def build_graph():
    builder = StateGraph(State)
    builder.add_node("chat_with_model", chat_with_model)
    builder.add_node("convert_messages", convert_messages)
    builder.set_entry_point("chat_with_model")
    builder.add_edge("chat_with_model", "convert_messages")
    builder.add_edge("convert_messages", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()

    # Keep one shared state object across turns => true multi-turn memory.
    state: State = {"messages": [], "extracts": []}

    print("Memory chat started. Type 'exit' to stop.")
    while True:
        user_text = input("User: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        # Add only new user message; reducer appends node outputs automatically.
        state = graph.invoke({**state, "messages": [HumanMessage(content=user_text)]})

        latest_assistant = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                latest_assistant = str(msg.content)
                break
        print(f"Assistant: {latest_assistant}\n")

    # Optional: print final memory snapshot
    snapshot = {
        "turn_messages": len(state.get("messages", [])),
        "extract_count": len(state.get("extracts", [])),
        "latest_extract": (state.get("extracts") or [""])[-1],
    }
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
