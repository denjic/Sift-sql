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

PLANNING_PROMPT_V1 = """You are a SQL query planner for the Spider benchmark database.

Given a natural language question and a pruned schema, produce a concise logical
blueprint. Do NOT write SQL syntax.

Strict rules:
1. REQUIRED TABLES: List only the tables strictly needed. Do not include tables
   just because they are in the schema — only include them if the question
   requires data from them.
2. JOIN CONDITIONS: Only specify joins if data from multiple tables is needed.
   State the exact columns: e.g. "JOIN singer ON singer_in_concert.Singer_ID = singer.Singer_ID"
3. SELECT COLUMNS: Name the exact columns to select using their schema names.
   If the question asks for a name or label, select the Name/text column, NOT the ID column.
4. FILTER CONDITIONS: State exact WHERE conditions if the question has filters.
5. AGGREGATION: State COUNT/AVG/SUM/MIN/MAX only if the question asks for it.
6. GROUP BY: Only if aggregating over groups.
7. HAVING: If the question says "more than N", "at least N", "greater than N" after
   an aggregation, you MUST include a HAVING clause. Do not use WHERE for post-aggregation filters.
8. ORDER BY / LIMIT: Only if the question explicitly says "order", "rank", "top N", "most", "least".
9. Do NOT add ORDER BY just because there is a GROUP BY.

Schema:
{pruned_schema}

Question:
{nl_query}

Logical Blueprint (bullet points only, no SQL):"""


GENERATION_PROMPT_V1 = """You are an expert SQLite SQL generator for the Spider benchmark.

Generate a single valid SQL query that answers the question exactly.

Critical rules:
1. Use ONLY the table and column names that appear in the schema. Do not invent names.
2. Follow the logical blueprint exactly — do not add tables, joins, or clauses not mentioned.
3. Use real table names (e.g. singer, concert) — do NOT use aliases like T1, T2, T3.
4. Do NOT add ORDER BY unless the blueprint explicitly mentions ordering.
5. Do NOT add unnecessary JOINs. If the blueprint says single table, use single table.
6. If the question asks for a name or description, select the text/name column, NOT the ID column.
7. HAVING goes after GROUP BY for post-aggregation filters, not WHERE.
8. Return ONLY the raw SQL query. No markdown, no explanation, no code fences.
9. End the query with a semicolon.

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