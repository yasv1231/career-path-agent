from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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
我希望在12周内完成转向数据分析师的准备，并在第13周开始投递岗位。
每周我可以投入10小时，预算不超过1500元，最好以在线学习为主。
我的首要目标是完成2个可展示的数据分析项目；第二目标是系统学习SQL和可视化。
硬约束：不能脱产，不能搬家，只接受远程或本地机会。
我更偏好课程+项目结合，不希望只看理论。
"""


def _load_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval-only validation for Goal+Constraint structured capture module."
    )
    parser.add_argument("--question-file", default="base_question.txt", help="Question bank file path.")
    parser.add_argument("--answers-file", default="", help="Optional file with user answers text.")
    parser.add_argument("--career", default="data analyst", help="Target career for retrieval validation.")
    args = parser.parse_args()

    question_file = Path(args.question_file)
    answers_file = Path(args.answers_file) if args.answers_file else None

    questions = _load_lines(question_file)
    if answers_file and answers_file.exists():
        answers_text = answers_file.read_text(encoding="utf-8-sig")
    else:
        answers_text = DEFAULT_SAMPLE_ANSWERS.strip()

    goal_constraints = merge_goal_constraints(
        default_goal_constraints(),
        extract_goal_constraints_from_text(answers_text),
    )
    strategy_tags = derive_strategy_tags(goal_constraints)
    rag_filters = build_rag_filters(goal_constraints)
    query_hints = goal_constraints.get("query_hints", [])

    issues = _validate_structure(goal_constraints)
    if issues:
        print("[STRUCTURE] FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("[STRUCTURE] PASS")
    print(json.dumps(goal_constraints, ensure_ascii=False, indent=2))
    print(f"\n[STRATEGY TAGS] {strategy_tags}")
    print(f"[RAG FILTERS] {rag_filters}")
    print(f"[QUERY HINTS] {query_hints}")

    if questions:
        print(f"\n[QUESTION BANK] loaded {len(questions)} lines from {question_file}")

    agent = RagAgent()
    sections = agent.retrieve_sections(
        args.career,
        rag_filters=rag_filters,
        query_hints=query_hints,
    )

    print("\n[RETRIEVAL] PASS")
    for section in sections:
        name = section.get("name")
        result = section.get("result", {})
        items = result.get("items", [])
        status = result.get("status")
        print(f"- {name}: status={status}, items={len(items)}")
        for item in items[:2]:
            print(f"  * {item.get('title', '')}")


if __name__ == "__main__":
    main()
