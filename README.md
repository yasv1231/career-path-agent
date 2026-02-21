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
- `langsmith_integration.py`: runtime logger + model client setup (no external tracing dependency).
- `mcp.config.json`: optional MCP server config.
- `AGENTS.md`: repository contribution guidelines.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Run API Server

```powershell
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Key endpoints:
- `POST /api/v1/auth/register`: register + return token.
- `POST /api/v1/auth/login`: login + return token.
- `GET /api/v1/auth/me`: get current user (`Authorization: Bearer <token>`).
- `POST /api/v1/chat/sessions`: create a conversation session and get first assistant question.
- `POST /api/v1/chat/sessions/{session_id}/messages`: send user message and get updated state.
- `GET /api/health`: health check.

Env:
- `APP_DB_PATH` (optional): sqlite path, default `data/app.db`.

## Run With Docker Compose (Backend + career-UI)

```powershell
docker compose up --build
```

After startup:
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

## Bind Domain + Reverse Proxy (`career.yasv.xyz`)

1. DNS bind (at your domain provider):
- Add an `A` record: `career.yasv.xyz -> <your_server_public_ip>`.
- Wait for DNS propagation.

2. Start project services on server:

```bash
docker compose up -d --build
```

3. Install and enable Nginx config:

```bash
sudo cp deploy/nginx/career.yasv.xyz.conf /etc/nginx/sites-available/career.yasv.xyz.conf
sudo ln -sf /etc/nginx/sites-available/career.yasv.xyz.conf /etc/nginx/sites-enabled/career.yasv.xyz.conf
sudo nginx -t
sudo systemctl reload nginx
```

4. Enable HTTPS (Let's Encrypt):

```bash
sudo apt-get update
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d career.yasv.xyz
```

5. Verify:

```bash
curl -I https://career.yasv.xyz
curl -I https://career.yasv.xyz/api/health
```

If your backend CORS checks origin, set in `.env` then restart:

```bash
FRONTEND_ORIGIN=https://career.yasv.xyz
```

## Build RAG Corpus From Text Docs

1. Put raw docs into `data/knowledge_docs/` (`.md` or `.txt`).
2. Front matter is optional; the builder can infer `career/type/title` from free-form text.
3. Build corpus:

```powershell
python scripts/build_rag_corpus.py --docs-dir data/knowledge_docs --output data/rag_corpus.json
```

To append new docs while keeping existing core corpus:

```powershell
python scripts/build_rag_corpus.py --docs-dir data/knowledge_docs --output data/rag_corpus.json --merge-existing
```

Notes:
- Files containing repeated headers like `User Response Sample N` are auto-split into `profile_case` docs.
- `profile_case` docs include an auto-generated structured summary chunk for stronger semantic retrieval.

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

Set credentials in `.env` if you need remote model access:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL` (optional)

Do not commit real secrets.
