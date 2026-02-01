---
name: routing-skill
description: Decide whether to use RAG or not; use when selecting retrieval vs. direct reasoning, especially when external docs may be needed or freshness/grounding is required.
---

# Routing Skill

## Goal
Choose between RAG (retrieve external or local docs) and direct response, with a short justification and a routing signal.

## Node Context
- Typical placement: before `chat_node`
- Output is a routing decision that drives whether retrieval runs

## Decision Criteria
Use RAG when:
- The task requires factual grounding in project files or knowledge base.
- The user asks for citations, quotes, or "according to the docs".
- The information is likely stale or version-specific.
- The answer depends on large context not in the conversation.

Avoid RAG when:
- The task is purely creative or procedural with no external facts.
- The user provides all needed data in the prompt.
- The response is a simple transformation (rewrite, summarize, translate).

## Routing Output
Return JSON only:

```
{
  "route": "rag|direct",
  "confidence": 0.0,
  "reasons": ["..."]
}
```

## Tie-Breakers
- If user asks for "latest", "current", or "policy", prefer RAG.
- If retrieval cost is high and benefit low, prefer direct.
- If unsure, choose RAG with lower confidence.