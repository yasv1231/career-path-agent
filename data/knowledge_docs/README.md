# Knowledge Docs Template

Put your `.md` / `.txt` source files in this folder, then build `data/rag_corpus.json`:

```powershell
python scripts/build_rag_corpus.py --docs-dir data/knowledge_docs --output data/rag_corpus.json
```

Front matter is optional. If omitted, the builder will do semantic auto-inference for:
- `career`
- `type` (`job_profile` / `competency` / `resource` / `profile_case`)
- `title`
- `resource_type` (`course` / `project`)

Optional front matter (if you want strict control):

```md
---
id: job_profile_data_analyst
career: data analyst
type: job_profile
title: Data Analyst role summary
resource_type:
url:
---
Data Analysts turn raw data into reports and dashboards.

They define metrics, clean datasets, and run SQL queries.
```

Fields:
- `career`: used as retrieval filter (`data analyst`, `data scientist`, etc.).
- `type`: `job_profile`, `competency`, `resource`, or `profile_case`.
- `resource_type`: optional (for `resource` docs), e.g. `course` / `project`.
- `url`: optional source link.

Chunking:
- Supports long free-form articles.
- Automatically splits by paragraphs/sentences with overlap.

Batch user profile files:
- If a file contains repeated headers like `User Response Sample 1`, `User Response Sample 2`, ...,
  the builder auto-splits each sample into an independent `profile_case` document.
- Each `profile_case` will get an extra structured summary chunk (goals/constraints/tags) for better retrieval recall.

Merge mode (recommended):

```powershell
python scripts/build_rag_corpus.py --docs-dir data/knowledge_docs --output data/rag_corpus.json --merge-existing
```

- `--merge-existing` appends new docs by `id` and keeps existing core job/competency/resource docs.

Optional semantic refinement with LLM:

```powershell
python scripts/build_rag_corpus.py --docs-dir data/knowledge_docs --output data/rag_corpus.json --use-llm-semantic
```
