# Career Path Agent (Minimal Runtime)

This repository now keeps only the core runtime for the single-entry conversation workflow.

## What It Does

- Accepts user messages in CLI.
- Extracts profile fields from conversation context.
- Routes conversation to ask follow-up questions or generate a plan.
- Optionally enriches output with local RAG content from `data/rag_corpus.json`.
- Uses a reference-style RAG pipeline (chunk -> embed -> Chroma retrieve -> rerank).

## Key Files

- `main.py`: program entrypoint.
- `workflow/`: main workflow pipeline (`graph_chat.py`, `routing.py`, `extractor.py`, etc.).
- `rag_agent.py` and `rag_store.py`: RAG retrieval helpers.
- `data/rag_resources.json`: replaceable RAG config (aliases, query hints, thresholds, sections).
- `data/rag_corpus.json`: replaceable knowledge corpus.
- `reference_rag_kb.py`: reference-style RAG program adapted from the linked VideoCode notebook.
- `scripts/build_rag_corpus.py`: convert raw `.md/.txt` docs into `data/rag_corpus.json`.
- `langsmith_integration.py`: tracing and model client setup.
- `mcp.config.json`: optional MCP server config.
- `AGENTS.md`: repository contribution guidelines.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Build RAG Corpus From Text Docs

1. Put raw docs into `data/knowledge_docs/` (`.md` or `.txt`).
2. Front matter is optional; the builder can infer `career/type/title` from free-form text.
3. Build corpus:

```powershell
python scripts/build_rag_corpus.py --docs-dir data/knowledge_docs --output data/rag_corpus.json
```

Optional semantic refinement via LLM:

```powershell
python scripts/build_rag_corpus.py --docs-dir data/knowledge_docs --output data/rag_corpus.json --use-llm-semantic
```

4. Run the app. RAG index is auto-built in `data/rag_chroma/` on first retrieval.

## Run The Reference-Style RAG Program

```powershell
python reference_rag_kb.py --docs-dir data/knowledge_docs --query "What are the core skills of a data analyst?"
```

## Configuration

Set credentials in `.env` if you need remote model or tracing:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL` (optional)
- `LANGSMITH_API_KEY` (optional)

Do not commit real secrets.
