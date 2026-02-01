---
name: validation-skill
description: Validate JSON for compliance with schema, required fields, type constraints, and formatting; use when asked to check JSON correctness or enforce structured output rules.
---

# Validation Skill

## Goal
Verify JSON validity and schema compliance, returning precise, actionable errors.

## Node Context
- Typical target: `extracted` from `extract_node`
- Use before storage, tool calls, or evaluation

## Workflow
1. Parse JSON strictly (no comments, trailing commas).
2. Check schema:
   - Required fields present
   - Types match
   - Enum values valid
   - Range/length constraints met
3. Check semantics:
   - Dates in ISO 8601
   - IDs unique if required
   - Cross-field dependencies
4. Return report using the format below.

## Report Format
Always return JSON:

```
{
  "valid": true,
  "errors": [],
  "warnings": []
}
```

If invalid, set `valid` to false and add errors:

```
{
  "valid": false,
  "errors": [
    {
      "path": "$.field",
      "rule": "type|required|enum|format|range|custom",
      "message": "...",
      "expected": "...",
      "found": "..."
    }
  ],
  "warnings": []
}
```

## Error Guidance
- Use JSONPath-like paths (e.g., `$.items[0].price`).
- Keep messages short and deterministic.
- Prefer one error per field unless the user requests exhaustive checks.

## If No Schema Provided
Validate only JSON syntax and report `warnings` recommending a schema.