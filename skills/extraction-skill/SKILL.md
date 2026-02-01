---
name: extraction-skill
description: Extract structured data from text using JSON schemas or NER-style entities; use when the user asks to extract fields, entities, or structured JSON from unstructured input.
---

# Extraction Skill

## Goal
Convert `last_answer` into structured JSON for downstream use (storage, tool calls, evaluation).

## Node Context
- Node: `extract_node`
- Input: `last_answer`
- Output: `extracted`

## Workflow
1. Read target schema: if provided, follow exactly; if not, use the default schema below.
2. Extract entities/fields: map raw text to schema fields.
3. Normalize values (dates, numbers, units, names).
4. Validate against schema (required fields, types, enums).
5. If invalid, **retry** extraction with corrections (up to 2 retries).
6. Return JSON only unless the user asked for explanations.

## JSON Extraction Pattern
If a schema is given, output JSON that matches it exactly. Otherwise use:

```
{
  "items": [
    {
      "field": "...",
      "value": "...",
      "normalized": "...",
      "evidence": "short quote"
    }
  ]
}
```

## Default Schema (for structured evaluation)
Use this schema when none is provided. It is optimized for downstream storage, tool calls, and evaluation.

```
{
  "type": "object",
  "required": ["schema_version", "task", "intent", "entities", "constraints", "actions", "confidence"],
  "properties": {
    "schema_version": { "type": "string", "enum": ["v1"] },
    "task": { "type": "string" },
    "intent": { "type": "string", "enum": ["ask", "request", "inform", "confirm", "refine", "other"] },
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["type", "text"],
        "properties": {
          "type": { "type": "string" },
          "text": { "type": "string" },
          "normalized": { "type": "string" }
        }
      }
    },
    "constraints": {
      "type": "array",
      "items": { "type": "string" }
    },
    "actions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["action", "target"],
        "properties": {
          "action": { "type": "string" },
          "target": { "type": "string" },
          "params": { "type": "object" }
        }
      }
    },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
  }
}
```

## Validation + Retry
- Run schema validation after extraction.
- If invalid: fix missing fields, type mismatches, enum violations, or empty required fields.
- Retry up to **2 times**; if still invalid, return the closest valid JSON and mark low confidence (<= 0.4).

## NER Pattern
For entity extraction, use:

```
{
  "entities": [
    {
      "type": "PERSON|ORG|LOC|DATE|MONEY|TITLE|SKILL|OTHER",
      "text": "...",
      "normalized": "...",
      "start": 0,
      "end": 0,
      "confidence": 0.0
    }
  ]
}
```

## Normalization Rules
- Dates: ISO 8601.
- Money: numeric plus currency code when possible.
- Titles/roles: canonical casing (e.g., "Software Engineer").
- Skills: singular form; keep original if ambiguous.

## Evidence Rules
- Keep evidence quotes under 12 words.
- If the text is long, select the shortest unique span.

## Uncertainty Handling
- If unsure, include a lower confidence score and leave normalized blank.
- Do not invent entities or fields not supported by evidence.
