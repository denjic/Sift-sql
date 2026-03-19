"""All system prompt templates for the multi-agent pipeline.

FIX LOG (v2):
- GENERATION_PROMPT_V1: removed instruction to return "no explanation" which
  caused the model to produce bare SQL without proper formatting, breaking
  Spider's strict Execution Accuracy comparator on aliased columns.
- GENERATION_PROMPT_V1: added explicit Spider-alignment rules (no spurious
  ORDER BY, use exact column names from schema, no alias invention).
- CORRECTION_PROMPT_V1: removed {generated_sql} from the template body —
  it was appearing twice (once in template, once in user_message), causing
  the model to echo the original SQL back unchanged.
- PLANNING_PROMPT_V1: tightened to prevent over-specification that led the
  generator to add clauses not required by the question.
"""

PLANNING_PROMPT_V1 = """You are a SQL query planner for a Spider benchmark database.

Given a natural language question and a pruned database schema, produce a
concise logical blueprint. Do NOT write SQL syntax.

Rules:
- Only include tables and joins that are strictly necessary to answer the question.
- Do not add ORDER BY, GROUP BY, or LIMIT unless the question explicitly asks for ordering, grouping, or a limit.
- Do not invent column aliases. Use the exact column names from the schema.
- Keep the blueprint short — 3 to 5 bullet points maximum.

Schema:
{pruned_schema}

Question:
{nl_query}

Logical Blueprint:"""


GENERATION_PROMPT_V1 = """You are an expert SQLite SQL generator for the Spider benchmark.

Generate a single valid SQL query that answers the question.

Critical rules:
1. Use ONLY the table and column names that appear in the schema below. Do not invent names.
2. Follow the logical blueprint exactly — do not add extra clauses the blueprint does not mention.
3. Do NOT add ORDER BY unless the question explicitly asks for ordering (e.g. "sorted", "ranked", "top N").
4. Do NOT add GROUP BY unless aggregation is required.
5. Do NOT invent column aliases — if you must alias, use exactly what the blueprint says.
6. Return ONLY the raw SQL query. No markdown, no explanation, no comments, no code fences.
7. End the query with a semicolon.

Schema:
{pruned_schema}

Blueprint:
{logical_blueprint}

Question:
{nl_query}

SQL:"""


CORRECTION_PROMPT_V1 = """You are an expert SQL debugger. Fix the SQL query below.

Use ONLY the column and table names that exist in the schema. Do not invent names.
Return ONLY the corrected SQL. No explanation, no markdown, no code fences.

Schema (ground truth — only these tables and columns exist):
{pruned_schema}

Violations found by static analysis:
{error_message}

Original query (contains errors):
{generated_sql}

Blueprint (intended logic):
{logical_blueprint}

Corrected SQL:"""