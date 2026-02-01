from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict, Any
import sys
import os
import json

from langsmith_integration import get_default_logger, configure_langsmith, get_chat_model
from tool_agent import ToolAgent
from career_agent import CareerAgent
from course_agent import CourseAgent
from roadmap_agent import RoadmapAgent
from evaluator_agent import EvaluatorAgent
from memory_manager import MemoryManager
from rag_agent import RagAgent
from .state import ConversationState, REQUIRED_FIELDS

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    HAS_LANGCHAIN_MESSAGES = True
except Exception:
    SystemMessage = None
    HumanMessage = None
    HAS_LANGCHAIN_MESSAGES = False

logger = get_default_logger()


def _parse_list(value: str) -> List[str]:
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def _safe_print(*args: object, sep: str = " ", end: str = "\n") -> None:
    encoding = sys.stdout.encoding or "utf-8"
    text = sep.join(str(arg) for arg in args) + end
    try:
        text.encode(encoding)
        sys.stdout.write(text)
    except UnicodeEncodeError:
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe_text)


def _extract_career_from_text(text: str) -> tuple[str, str]:
    if not text:
        return "", ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", text

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            career = str(data.get("career", "")).strip()
            reasoning = str(data.get("reasoning", "")).strip()
            if career:
                return career, reasoning or text
    except Exception:
        pass

    for ln in lines:
        lower = ln.lower()
        if lower.startswith("career:"):
            career = ln.split(":", 1)[1].strip(" -\t")
            reasoning = text
            return career, reasoning

    career = lines[0].lstrip("-? ").strip()
    reasoning = text
    return career, reasoning


def build_graph():
    langsmith_status = configure_langsmith("career-path-agent")
    llm = get_chat_model()

    memory = MemoryManager()
    career_agent = CareerAgent()
    course_agent = CourseAgent()
    roadmap_agent = RoadmapAgent()
    evaluator_agent = EvaluatorAgent()
    tool_agent = ToolAgent()
    rag_agent = RagAgent()

    def load_memory(state: ConversationState) -> ConversationState:
        last_profile = memory.load_last_profile()
        return {
            "profile": last_profile or {},
            "has_last_profile": bool(last_profile),
            "llm_enabled": bool(llm),
        }

    def decide_reuse(state: ConversationState) -> ConversationState:
        if not state.get("has_last_profile"):
            return {"reuse_last_profile": False}

        profile = state.get("profile", {})
        _safe_print("Welcome back! I found your last saved profile.\n")
        _safe_print(f"Last education: {profile.get('education', 'N/A')}")
        _safe_print(f"Last interest : {profile.get('interest', 'N/A')}")
        use_last = input("\nDo you want to reuse this profile? (y/n): ").strip().lower()
        return {
            "reuse_last_profile": use_last == "y",
            "profile": profile if use_last == "y" else {},
        }

    def collect_profile(state: ConversationState) -> ConversationState:
        profile = dict(state.get("profile") or {})
        fields_to_update = [f.strip() for f in state.get("fields_to_update", []) if f.strip()]

        if fields_to_update:
            target_fields = fields_to_update
        else:
            target_fields = [f for f in REQUIRED_FIELDS if not profile.get(f)]

        if not profile and not target_fields:
            target_fields = REQUIRED_FIELDS

        for field in target_fields:
            if field == "education":
                value = input("1) What is your current education/branch? (e.g., B.Tech CSE, AI&DS): ")
                profile["education"] = value.strip()
            elif field == "favorites":
                value = input("2) What are your favorite subjects or topics? (comma separated): ")
                profile["favorites"] = _parse_list(value)
            elif field == "skills":
                value = input("3) What technical skills do you know? (e.g., Python, SQL, Excel): ")
                profile["skills"] = _parse_list(value)
            elif field == "interest":
                value = input("4) What career are you interested in? (e.g., Data Analyst, Data Scientist, ML Engineer, Not sure): ")
                profile["interest"] = value.strip().lower()
            elif field == "hours_per_week":
                value = input("5) How many hours per week can you spend on learning? (number): ")
                profile["hours_per_week"] = value.strip()

        return {"profile": profile, "fields_to_update": []}

    def save_memory(state: ConversationState) -> ConversationState:
        profile = state.get("profile", {})
        if profile:
            memory.save_profile(profile)
        return {}

    def run_llm_career_agent(state: ConversationState) -> ConversationState:
        profile = state.get("profile", {})
        if not llm or not profile:
            return {"llm_enabled": bool(llm)}

        profile_json = json.dumps(profile, ensure_ascii=True)

        system_prompt = (
            "You are a careful career advisor. "
            "Given a student profile JSON, recommend exactly ONE target career. "
            "Prefer common, actionable roles like Data Analyst, Data Scientist, ML Engineer, "
            "Software Engineer, Product Analyst, or similar. "
            "Respond as JSON with keys: career, reasoning."
        )
        human_prompt = f"Student profile JSON:\n{profile_json}"

        try:
            if HAS_LANGCHAIN_MESSAGES:
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ])
            else:
                response = llm.invoke(system_prompt + "\n\n" + human_prompt)
        except Exception as e:
            try:
                logger.log_event("llm_career_agent_error", {"error": str(e)})
            except Exception:
                pass
            return {"llm_enabled": True}

        text = getattr(response, "content", None) or str(response)
        career, reasoning = _extract_career_from_text(text)
        if not career:
            return {"llm_enabled": True}

        return {
            "career": career,
            "career_source": "llm",
            "career_reasoning": reasoning,
            "llm_enabled": True,
        }

    def run_career_agent(state: ConversationState) -> ConversationState:
        if state.get("career") and state.get("career_source") == "llm":
            return {}

        profile = state.get("profile", {})
        career = career_agent.infer_career(profile)
        return {"career": career, "career_source": "rules"}

    def run_course_agent(state: ConversationState) -> ConversationState:
        career = state.get("career", "")
        return {"courses": course_agent.get_courses(career)}

    def run_rag_job_confirmation(state: ConversationState) -> ConversationState:
        career = state.get("career", "")
        return {"job_confirmation": rag_agent.job_confirmation(career)}

    def run_rag_competency_model(state: ConversationState) -> ConversationState:
        career = state.get("career", "")
        return {"competency_model": rag_agent.competency_model(career)}

    def run_rag_resource_retrieval(state: ConversationState) -> ConversationState:
        career = state.get("career", "")
        return {"rag_resources": rag_agent.resource_retrieval(career)}

    def run_tool_agent(state: ConversationState) -> ConversationState:
        career = state.get("career", "")
        return {"free_courses": tool_agent.get_free_courses(career)}

    def run_roadmap_agent(state: ConversationState) -> ConversationState:
        career = state.get("career", "")
        hours = state.get("profile", {}).get("hours_per_week", "")
        return {"roadmap": roadmap_agent.build_roadmap(career, hours)}

    def run_evaluator_agent(state: ConversationState) -> ConversationState:
        profile = state.get("profile", {})
        career = state.get("career", "")
        return {"evaluation": evaluator_agent.evaluate(profile, career)}

    def render_output(state: ConversationState) -> ConversationState:
        career = state.get("career", "")
        courses = state.get("courses", [])
        roadmap = state.get("roadmap", "")
        evaluation = state.get("evaluation", "")
        free_courses = state.get("free_courses", [])
        job_confirmation = state.get("job_confirmation", {})
        competency_model = state.get("competency_model", {})
        rag_resources = state.get("rag_resources", {})
        career_source = state.get("career_source", "unknown")
        career_reasoning = state.get("career_reasoning", "")

        _safe_print("\nRecommended Career Path for you:", career, "\n")
        _safe_print(f"Career source: {career_source}")

        if career_reasoning:
            _safe_print("Reasoning (from LLM):")
            _safe_print(career_reasoning)

        if job_confirmation:
            _safe_print("\nJob Confirmation (RAG):")
            items = job_confirmation.get("items", [])
            confidence = job_confirmation.get("confidence")
            if confidence:
                _safe_print(f"Confidence: {confidence}")
            status = job_confirmation.get("status")
            if status:
                _safe_print(f"Status: {status}")
            action = job_confirmation.get("action")
            if action:
                _safe_print(f"Action: {action}")
            refusal = job_confirmation.get("refusal")
            if refusal:
                _safe_print(f"Refusal: {refusal}")
            clarify = job_confirmation.get("clarify")
            if clarify:
                _safe_print(f"Clarify: {clarify}")
            if items:
                for item in items:
                    _safe_print(f"  - {item.get('title', '')}: {item.get('text', '')}")
            evidence = job_confirmation.get("citations", [])
            if evidence:
                _safe_print("Citations:")
                for item in evidence:
                    _safe_print(f"  - {item.get('ref', '')} (score={item.get('score', '')})")

        if competency_model:
            _safe_print("\nCompetency Model (RAG):")
            items = competency_model.get("items", [])
            confidence = competency_model.get("confidence")
            if confidence:
                _safe_print(f"Confidence: {confidence}")
            status = competency_model.get("status")
            if status:
                _safe_print(f"Status: {status}")
            action = competency_model.get("action")
            if action:
                _safe_print(f"Action: {action}")
            refusal = competency_model.get("refusal")
            if refusal:
                _safe_print(f"Refusal: {refusal}")
            clarify = competency_model.get("clarify")
            if clarify:
                _safe_print(f"Clarify: {clarify}")
            if items:
                for item in items:
                    _safe_print(f"  - {item.get('title', '')}: {item.get('text', '')}")
            evidence = competency_model.get("citations", [])
            if evidence:
                _safe_print("Citations:")
                for item in evidence:
                    _safe_print(f"  - {item.get('ref', '')} (score={item.get('score', '')})")

        if rag_resources:
            _safe_print("\nCourse / Project Resources (RAG):")
            items = rag_resources.get("items", [])
            confidence = rag_resources.get("confidence")
            if confidence:
                _safe_print(f"Confidence: {confidence}")
            status = rag_resources.get("status")
            if status:
                _safe_print(f"Status: {status}")
            action = rag_resources.get("action")
            if action:
                _safe_print(f"Action: {action}")
            refusal = rag_resources.get("refusal")
            if refusal:
                _safe_print(f"Refusal: {refusal}")
            clarify = rag_resources.get("clarify")
            if clarify:
                _safe_print(f"Clarify: {clarify}")
            for item in items:
                title = item.get("title", "Resource")
                resource_type = item.get("resource_type", "resource")
                notes = item.get("notes", "")
                url = item.get("url", "")
                confidence = item.get("confidence", "")
                citations = item.get("citations", [])
                line = f"  - [{resource_type}] {title}"
                if notes:
                    line = f"{line} - {notes}"
                if confidence:
                    line = f"{line} (confidence={confidence})"
                if url:
                    line = f"{line} -> {url}"
                _safe_print(line)
                if citations:
                    for citation in citations:
                        _safe_print(f"      citation: {citation.get('ref', '')} (score={citation.get('score', '')})")

        _safe_print("Suggested Courses / Topics to Learn:")
        for course in courses:
            _safe_print(f"  - {course}")

        _safe_print("\nFREE Courses you can start:")
        for name, url in free_courses:
            _safe_print(f"  - {name} -> {url}")

        _safe_print(roadmap)

        _safe_print("Evaluation & Guidance:")
        _safe_print(evaluation)

        _safe_print("\nNext Step: Start with the suggested courses and follow the roadmap step by step.")
        _safe_print("You can run this agent again later - it will remember your profile and adjust.\n")
        return {}

    def follow_up(state: ConversationState) -> ConversationState:
        _safe_print("\nWhat would you like to do next?")
        _safe_print("  1) Update profile")
        _safe_print("  2) Rerun recommendations")
        _safe_print("  3) Show current profile")
        _safe_print("  4) Exit")
        choice = input("Choose 1-4: ").strip()

        if choice == "1":
            fields = input("Enter fields to update (education,favorites,skills,interest,hours) or 'all': ").strip().lower()
            if fields == "all" or not fields:
                fields_to_update = REQUIRED_FIELDS
            else:
                fields_to_update = []
                for raw in fields.split(","):
                    item = raw.strip()
                    if item == "hours":
                        fields_to_update.append("hours_per_week")
                    elif item in REQUIRED_FIELDS:
                        fields_to_update.append(item)

            if not fields_to_update:
                fields_to_update = REQUIRED_FIELDS

            return {"action": "update", "fields_to_update": fields_to_update}
        if choice == "2":
            return {"action": "rerun"}
        if choice == "3":
            return {"action": "show_profile"}

        return {"action": "exit"}

    def show_profile(state: ConversationState) -> ConversationState:
        profile = state.get("profile", {})
        _safe_print("\nCurrent profile:")
        for key in REQUIRED_FIELDS:
            _safe_print(f"  - {key}: {profile.get(key, 'N/A')}")
        return {}

    def wrap_node(name: str, fn):
        def _wrapped(state: ConversationState) -> ConversationState:
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

    graph = StateGraph(ConversationState)
    graph.add_node("load_memory", wrap_node("load_memory", load_memory))
    graph.add_node("decide_reuse", wrap_node("decide_reuse", decide_reuse))
    graph.add_node("collect_profile", wrap_node("collect_profile", collect_profile))
    graph.add_node("save_memory", wrap_node("save_memory", save_memory))
    graph.add_node("llm_career_agent", wrap_node("llm_career_agent", run_llm_career_agent))
    graph.add_node("career_agent", wrap_node("career_agent", run_career_agent))
    graph.add_node("rag_job_confirmation", wrap_node("rag_job_confirmation", run_rag_job_confirmation))
    graph.add_node("rag_competency_model", wrap_node("rag_competency_model", run_rag_competency_model))
    graph.add_node("rag_resource_retrieval", wrap_node("rag_resource_retrieval", run_rag_resource_retrieval))
    graph.add_node("course_agent", wrap_node("course_agent", run_course_agent))
    graph.add_node("tool_agent", wrap_node("tool_agent", run_tool_agent))
    graph.add_node("roadmap_agent", wrap_node("roadmap_agent", run_roadmap_agent))
    graph.add_node("evaluator_agent", wrap_node("evaluator_agent", run_evaluator_agent))
    graph.add_node("render_output", wrap_node("render_output", render_output))
    graph.add_node("follow_up", wrap_node("follow_up", follow_up))
    graph.add_node("show_profile", wrap_node("show_profile", show_profile))

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "decide_reuse")
    graph.add_edge("decide_reuse", "collect_profile")
    graph.add_edge("collect_profile", "save_memory")
    graph.add_edge("save_memory", "llm_career_agent")
    graph.add_edge("llm_career_agent", "career_agent")
    graph.add_edge("career_agent", "rag_job_confirmation")
    graph.add_edge("rag_job_confirmation", "rag_competency_model")
    graph.add_edge("rag_competency_model", "rag_resource_retrieval")
    graph.add_edge("rag_resource_retrieval", "course_agent")
    graph.add_edge("course_agent", "tool_agent")
    graph.add_edge("tool_agent", "roadmap_agent")
    graph.add_edge("roadmap_agent", "evaluator_agent")
    graph.add_edge("evaluator_agent", "render_output")
    graph.add_edge("render_output", "follow_up")

    def should_continue(state: ConversationState) -> str:
        action = state.get("action")
        if action == "update":
            return "collect_profile"
        if action == "rerun":
            return "llm_career_agent"
        if action == "show_profile":
            return "show_profile"
        return END

    graph.add_conditional_edges("follow_up", should_continue)
    graph.add_edge("show_profile", "follow_up")

    try:
        logger.start_run("conversation_run")
        logger.log_event("langsmith_config", langsmith_status)
        logger.log_event("llm_config", {"enabled": bool(llm), "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")})
    except Exception:
        pass

    return graph