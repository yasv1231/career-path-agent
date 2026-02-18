from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_agent import RagAgent
from workflow.goal_constraints import (
    build_rag_filters,
    default_goal_constraints,
    derive_strategy_tags,
    extract_goal_constraints_from_text,
    merge_goal_constraints,
)


DEFAULT_SAMPLE_ANSWERS = """
I hope to complete a role transition in 12 weeks and start applying in week 13.
I can invest 10 hours per week. My learning budget should stay below RMB 1500.
My primary goal is to build two portfolio projects and then secure interviews.
Hard constraints: I cannot relocate and prefer online-first learning.
"""

SAMPLE_HEADER_RE = re.compile(r"^\s*(?:user\s*response\s*sample|用户回答样本)\s*(\d+)\s*$", re.IGNORECASE)


def _load_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _split_samples(raw: str) -> List[Tuple[str, str]]:
    if not raw.strip():
        return []

    samples: List[Tuple[str, str]] = []
    current_name: str | None = None
    current_lines: List[str] = []

    for line in raw.splitlines():
        match = SAMPLE_HEADER_RE.match(line)
        if match:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    samples.append((current_name or "sample_1", body))
            current_name = f"sample_{match.group(1)}"
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            samples.append((current_name or "sample_1", body))

    if len(samples) <= 1:
        return [("sample_1", raw.strip())]
    return samples


def _validate_structure(payload: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    required_keys = {
        "schema_version",
        "time_dimension",
        "goals",
        "acceptable_cost",
        "hard_constraints",
        "preferences",
        "query_hints",
    }
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        issues.append(f"missing keys: {missing}")

    goals = payload.get("goals", [])
    if not isinstance(goals, list):
        issues.append("goals must be array")
    else:
        last_rank = 0
        for idx, goal in enumerate(goals):
            if not isinstance(goal, dict):
                issues.append(f"goals[{idx}] must be object")
                continue
            rank = int(goal.get("priority_rank") or 0)
            if rank < last_rank:
                issues.append("goals are not sorted by priority_rank")
            last_rank = rank
    return issues


def _run_single(agent: RagAgent, answer_text: str, career: str) -> Dict[str, Any]:
    goal_constraints = merge_goal_constraints(
        default_goal_constraints(),
        extract_goal_constraints_from_text(answer_text),
    )
    strategy_tags = derive_strategy_tags(goal_constraints)
    rag_filters = build_rag_filters(goal_constraints)
    query_hints = goal_constraints.get("query_hints", [])
    issues = _validate_structure(goal_constraints)
    sections = agent.retrieve_sections(career, rag_filters=rag_filters, query_hints=query_hints)
    return {
        "goal_constraints": goal_constraints,
        "strategy_tags": strategy_tags,
        "rag_filters": rag_filters,
        "query_hints": query_hints,
        "issues": issues,
        "sections": sections,
    }


def _print_detailed(result: Dict[str, Any], questions: List[str], question_file: Path) -> None:
    issues = result["issues"]
    if issues:
        print("[STRUCTURE] FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("[STRUCTURE] PASS")
    print(json.dumps(result["goal_constraints"], ensure_ascii=False, indent=2))
    print(f"\n[STRATEGY TAGS] {result['strategy_tags']}")
    print(f"[RAG FILTERS] {result['rag_filters']}")
    print(f"[QUERY HINTS] {result['query_hints']}")
    if questions:
        print(f"\n[QUESTION BANK] loaded {len(questions)} lines from {question_file}")
    print("\n[RETRIEVAL] PASS")
    for section in result["sections"]:
        name = section.get("name")
        section_result = section.get("result", {})
        items = section_result.get("items", [])
        status = section_result.get("status")
        print(f"- {name}: status={status}, items={len(items)}")
        for item in items[:2]:
            print(f"  * {item.get('title', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval-only validation for Goal+Constraint structured capture module."
    )
    parser.add_argument("--question-file", default="base_question.txt", help="Question bank file path.")
    parser.add_argument("--answers-file", default="", help="Optional file with user answers text.")
    parser.add_argument("--career", default="data analyst", help="Target career for retrieval validation.")
    parser.add_argument("--sample-id", type=int, default=0, help="Run one sample only (1-based). 0 means all.")
    parser.add_argument("--max-samples", type=int, default=0, help="Cap number of samples in batch mode.")
    parser.add_argument("--verbose-json", action="store_true", help="Print full JSON in batch mode.")
    args = parser.parse_args()

    question_file = Path(args.question_file)
    answers_file = Path(args.answers_file) if args.answers_file else None
    questions = _load_lines(question_file)

    if answers_file and answers_file.exists():
        answers_text = answers_file.read_text(encoding="utf-8-sig")
    else:
        answers_text = DEFAULT_SAMPLE_ANSWERS.strip()

    samples = _split_samples(answers_text)
    if args.sample_id > 0:
        target = [item for item in samples if item[0] == f"sample_{args.sample_id}"]
        if not target:
            raise SystemExit(f"sample_{args.sample_id} not found in answers file")
        samples = target

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    agent = RagAgent()

    if len(samples) == 1 and not args.verbose_json:
        _, body = samples[0]
        result = _run_single(agent, body, args.career)
        _print_detailed(result, questions, question_file)
        return

    print(f"[BATCH] samples={len(samples)}")
    if questions:
        print(f"[QUESTION BANK] loaded {len(questions)} lines from {question_file}")

    structure_pass = 0
    retrieval_pass = 0
    for index, (sample_name, body) in enumerate(samples, start=1):
        result = _run_single(agent, body, args.career)
        issues = result["issues"]
        section_statuses = [section.get("result", {}).get("status", "") for section in result["sections"]]
        section_items = [len(section.get("result", {}).get("items", [])) for section in result["sections"]]
        has_any_items = any(count > 0 for count in section_items)

        if not issues:
            structure_pass += 1
        if has_any_items:
            retrieval_pass += 1

        print(
            f"- {sample_name} (#{index}): structure={'PASS' if not issues else 'FAIL'}, "
            f"goals={len(result['goal_constraints'].get('goals', []))}, "
            f"hard_constraints={len(result['goal_constraints'].get('hard_constraints', []))}, "
            f"statuses={section_statuses}, items={section_items}, tags={result['strategy_tags']}"
        )

        if args.verbose_json:
            print(json.dumps(result["goal_constraints"], ensure_ascii=False, indent=2))

    print(
        f"\n[BATCH SUMMARY] structure_pass={structure_pass}/{len(samples)}, "
        f"retrieval_has_items={retrieval_pass}/{len(samples)}"
    )


if __name__ == "__main__":
    main()
