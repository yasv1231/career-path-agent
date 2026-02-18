# Repository Guidelines

## Project Structure & Module Organization
The repository is Python-first, with `main.py` as the CLI entrypoint. Core orchestration is under `workflow/` (`graph_chat.py`, routing, extraction, evaluation, and state models). Retrieval helpers are in `rag_agent.py` and `rag_store.py`, with corpus data in `data/rag_corpus.json`.

## Build, Test, and Development Commands
Use these commands from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
python scripts/build_rag_corpus.py --docs-dir data/knowledge_docs --output data/rag_corpus.json
python reference_rag_kb.py --docs-dir data/knowledge_docs --query "What are the core skills of a data analyst?"
```

- `python main.py`: runs the CLI message-flow agent.
- `python scripts/build_rag_corpus.py ...`: converts raw text docs to `data/rag_corpus.json`.
- `python reference_rag_kb.py ...`: runs the reference-style RAG retrieval/rerank pipeline.

## Coding Style & Naming Conventions
Follow standard Python conventions: 4-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for test classes. Keep functions focused and prefer explicit state keys in workflow code. Add type hints for new/updated public functions. Group imports as stdlib, third-party, then local modules.

## Testing Guidelines
This repository is currently runtime-focused and does not keep a committed test suite. If you add tests, prefer built-in `unittest`, name files `tests/test_<feature>.py`, and keep tests deterministic (no live API calls).

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects (for example, `Add workflow single-entry test`, `Update README.md`). Keep messages concise and scoped to one change. For pull requests, include:
- what changed and why,
- touched modules (for example `workflow/routing.py`),
- test commands run and outcomes,
- sample terminal output only when conversational behavior changes.

## Security & Configuration Tips
Keep secrets in `.env` and never commit API keys.
