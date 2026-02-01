---
name: memory-skill
description: Manage conversational memory: decide what to store, prune, and summarize messages; use when the user asks for memory, long chat needs condensation, or you must maintain a compact memory state.
---

# Memory Skill

## Goal
Maintain a compact, accurate memory state from `messages` by extracting durable facts, pruning noise, and writing summaries that preserve intent and constraints.

## Node Context
- Node: `memory_maintain_node` (optional)
- Input: `messages`
- Output: updated `messages` (compressed) plus optional memory summary payload

## Workflow
1. Collect inputs: `messages`, existing memory store, and any explicit memory instructions.
2. Classify each candidate into: durable fact, preference, task state, ephemeral, or sensitive.
3. Update memory:
   - Add durable facts and stable preferences.
   - Update task state (current objective, open questions, deadlines).
   - Remove or downgrade ephemeral details.
4. Prune:
   - Drop chit-chat, one-off info, and redundant duplicates.
   - Keep only last known value for mutable facts.
5. Summarize:
   - Write a short summary covering: user goals, constraints, decisions, and open items.
6. Validate: ensure memory is concise, non-contradictory, and safe to store.

## Classification Rules
- Durable fact: stable identity, long-term goals, skill level, tools/environment.
- Preference: format, language, style, ordering preferences.
- Task state: current project, files touched, pending steps, known blockers.
- Ephemeral: jokes, transient emotions, one-off dates unless explicitly marked to remember.
- Sensitive: secrets, credentials, personal data not required for the task.

## Pruning Rules
- Remove any sensitive data unless explicitly required and permitted.
- Keep at most one canonical value per attribute (e.g., preferred language).
- If a fact conflicts with a newer fact, keep the newer and mark the old as superseded.
- Prefer concise bullet statements over long prose.

## Summary Template
Use this template when producing a memory summary:

```
User Profile:
- ...
Preferences:
- ...
Current Task:
- ...
Open Questions:
- ...
Recent Decisions:
- ...
```

## Memory Update Format
When emitting memory updates, use JSON with these fields:

```
{
  "add": ["..."],
  "update": ["..."],
  "remove": ["..."],
  "summary": "..."
}
```

## Edge Cases
- If the user says "don't remember this," ensure it is removed and not summarized.
- If instructions conflict, prefer the most recent explicit instruction.
- If the memory budget is tight, keep task state and constraints, then preferences, then durable facts.