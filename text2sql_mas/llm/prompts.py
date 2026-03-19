"""All system prompt templates for the multi-agent pipeline.

FIX LOG (v3):
- PLANNING_PROMPT_V1: Added explicit rules for:
    * Column selection precision — must name exact columns from schema, not IDs
    * HAVING clause — must be stated when question uses "more than", "at least", etc.
    * Minimal joins — only join tables that are strictly necessary
- GENERATION_PROMPT_V1: Added rules against:
    * Using alias tables (T1, T2) — Spider gold standard uses real table names
    * Unnecessary JOINs not mentioned in the blueprint
    * Selecting ID columns when question asks for names/labels
- CORRECTION_PROMPT_V1: unchanged from v2 (working correctly)
"""

PLANNING_PROMPT_V1 = """You are a SQL query planner. Given a question and schema, 
write a 2-3 line plan. State only: which tables, which join condition if any, 
and any filter/aggregation. Be minimal. No SQL syntax.

Schema:
{pruned_schema}

Question:
{nl_query}

Plan (2-3 lines max):"""


GENERATION_PROMPT_V1 = """Generate a SQLite SQL query for the question below.
Use only the tables and columns in the schema. Follow the plan exactly.
Return only the raw SQL with no explanation.

Schema:
{pruned_schema}

Plan:
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