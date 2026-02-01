from __future__ import annotations

import json
from typing import Any, List

from langsmith_integration import get_default_logger, configure_langsmith, get_chat_model
from .state import ChatConversationState, EXTRACT_SCHEMA_V1
from .messages import (
    normalize_messages,
    serialize_messages_openai,
    is_openai_message,
)
from .routing import route_to_state
from .rag import maybe_attach_rag_context
from .evaluator import evaluate_extracted, evaluation_feedback_message
from .mcp import get_mcp_servers
from .extractor import (
    default_extracted,
    try_parse_json,
    validate_extracted_v1,
    coerce_extracted_v1,
    retry_prompt,
)

try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
    HAS_LANGCHAIN_MESSAGES = True
except Exception:
    SystemMessage = None
    HumanMessage = None
    AIMessage = None
    BaseMessage = None
    HAS_LANGCHAIN_MESSAGES = False

logger = get_default_logger()


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

    def _invoke_llm_with_messages(messages: List[Any]) -> str:
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
        except Exception as e:
            try:
                logger.log_event("llm_chat_error", {"error": str(e)})
            except Exception:
                pass
            return ""

    def chat_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        if not messages:
            return {"messages": messages, "last_answer": "", "llm_enabled": bool(llm)}

        text = _invoke_llm_with_messages(messages)
        if not text:
            last_user = _last_user_content(messages)
            text = f"(LLM disabled) {last_user}".strip()

        if HAS_LANGCHAIN_MESSAGES and AIMessage:
            messages.append(AIMessage(content=text))
        else:
            messages.append({"role": "assistant", "content": text})

        return {"messages": messages, "last_answer": text, "llm_enabled": bool(llm)}

    def routing_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        return route_to_state(messages)

    def rag_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        route = state.get("route", "direct")
        return maybe_attach_rag_context(messages, route)

    def extract_node(state: ChatConversationState) -> ChatConversationState:
        last_answer = state.get("last_answer", "")
        if not last_answer:
            return {"extracted": default_extracted(confidence=0.2)}

        if not llm:
            extracted = default_extracted(confidence=0.2)
            extracted["task"] = last_answer[:120]
            extracted["intent"] = "inform"
            return {"extracted": extracted}

        schema_json = json.dumps(EXTRACT_SCHEMA_V1, ensure_ascii=True)
        base_system = (
            "You are a strict JSON extraction engine. "
            "Output JSON only with schema_version=v1. "
            "No markdown, no commentary."
        )
        base_prompt = f"Input text:\n{last_answer}\n\nSchema:\n{schema_json}"

        attempts = 0
        last_data: Any = None
        while attempts < 3:
            attempts += 1
            try:
                if HAS_LANGCHAIN_MESSAGES:
                    response = llm.invoke(
                        [
                            SystemMessage(content=base_system),
                            HumanMessage(content=base_prompt),
                        ]
                    )
                else:
                    response = llm.invoke(base_system + "\n\n" + base_prompt)
                text = getattr(response, "content", None) or str(response)
            except Exception as e:
                try:
                    logger.log_event("llm_extract_error", {"error": str(e)})
                except Exception:
                    pass
                text = ""

            data = try_parse_json(text)
            last_data = data
            errors = validate_extracted_v1(data)
            if not errors:
                return {"extracted": data}

            base_prompt = retry_prompt(base_system, last_answer, errors)

        coerced = coerce_extracted_v1(last_data, confidence=0.4)
        if validate_extracted_v1(coerced):
            coerced = default_extracted(confidence=0.3)
        return {"extracted": coerced}

    def evaluation_node(state: ChatConversationState) -> ChatConversationState:
        extracted = state.get("extracted", default_extracted(confidence=0.2))
        evaluation = evaluate_extracted(extracted)
        feedback = evaluation_feedback_message(evaluation)

        messages = normalize_messages(state.get("messages", []))
        if HAS_LANGCHAIN_MESSAGES and AIMessage:
            messages.append(AIMessage(content=feedback))
        else:
            messages.append({"role": "assistant", "content": feedback})

        return {"messages": messages, "evaluation": evaluation}

    def memory_maintain_node(state: ChatConversationState) -> ChatConversationState:
        messages = normalize_messages(state.get("messages", []))
        max_messages = int(state.get("max_messages", 20))
        if len(messages) <= max_messages:
            return {"messages": messages}

        summary_text = ""
        if llm:
            summary_system = "Summarize the conversation briefly for memory. Output 3 bullets max."
            summary_human = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in serialize_messages_openai(messages)
            )
            try:
                if HAS_LANGCHAIN_MESSAGES:
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
        def _wrapped(state: ChatConversationState) -> ChatConversationState:
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
            except Exception as e:
                try:
                    logger.log_event(f"node_error:{name}", {"error": str(e)})
                except Exception:
                    pass
                raise

        return _wrapped

    try:
        from langgraph.graph import StateGraph, END
    except Exception:
        from _local_langgraph import StateGraph, END

    graph = StateGraph(ChatConversationState)
    graph.add_node("route", wrap_node("route", routing_node))
    graph.add_node("rag", wrap_node("rag", rag_node))
    graph.add_node("chat", wrap_node("chat", chat_node))
    graph.add_node("extract", wrap_node("extract", extract_node))
    graph.add_node("evaluate", wrap_node("evaluate", evaluation_node))
    graph.set_entry_point("route")
    graph.add_edge("route", "rag")
    graph.add_edge("rag", "chat")
    graph.add_edge("chat", "extract")
    graph.add_edge("extract", "evaluate")

    if enable_memory:
        graph.add_node("memory_maintain", wrap_node("memory_maintain", memory_maintain_node))
        graph.add_edge("evaluate", "memory_maintain")
        graph.add_edge("memory_maintain", END)
    else:
        graph.add_edge("evaluate", END)

    try:
        logger.start_run("message_flow")
        logger.log_event("langsmith_config", langsmith_status)
        logger.log_event("llm_config", {"enabled": bool(llm), "model": "gpt-4o-mini"})
        logger.log_event("mcp_config", {"servers": list(get_mcp_servers().keys())})
    except Exception:
        pass

    return graph
