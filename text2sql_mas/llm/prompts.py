
PLANNING_PROMPT_V1 = """You are a SQL query planner. Given a question and schema,
output a concise blueprint (4-6 lines max). Use these exact fields:

- Strategy: one of SIMPLE_SELECT | JOIN | SUBQUERY | INTERSECT | UNION | EXCEPT
- Tables: comma-separated list of tables needed
- Joins: join conditions, or NONE
- Aggregation: e.g. COUNT, SUM, AVG, or NONE
- Special: ORDER BY, LIMIT, HAVING, GROUP BY details, or NONE

STRICT OUTPUT FORMATTING RULES:
1. RETURN EXACTLY 5 LINES matching the bullet points above.
2. DO NOT write any explanations, notes, or "Alternative Strategies".
3. DO NOT write any SQL code.
4. If you write anything other than the 5 bullet points, you will break the pipeline.

CRITICAL RULES — read carefully:
1. NO EXTERNAL KNOWLEDGE: Never assume facts (e.g. 'Fiats are from Italy'). If a category is mentioned (e.g. 'Fiat'), find the table/column containing that category (e.g. Model='fiat') and plan to JOIN it.
2. AMBIGUITY: 'Larger than any' means 'at least one' → use `> (SELECT MIN(...))`. 'Larger than all' → use `> (SELECT MAX(...))`.
3. GROUP BY: Only group by the requested dimension (e.g. 'for each citizenship' -> `GROUP BY Citizenship`). NEVER include Primary Keys in the GROUP BY unless explicitly asked for per-record stats.
4. JOIN MINIMIZATION: Do not join a table if its information (like a code or ID) is already available in a table you've already selected.
5. EXCEPT / INTERSECT / UNION: These are SET OPERATIONS combining two SELECT statements. Use EXCEPT only for "rows in A but not in B". For "whose X is not Y", use SIMPLE_SELECT with WHERE != or NOT IN.

Schema:
{pruned_schema}

Question:
{nl_query}

Logical Blueprint:"""


GENERATION_PROMPT_V1 = """Generate a single SQLite SQL query. Return ONLY raw SQL — no explanation, no markdown, no code fences.

Rules:
1. STRICT SELECTION: Select ONLY the exact columns asked for. Extra columns cause execution mismatches.
2. NO EXTERNAL KNOWLEDGE: Do not hardcode values like 'Italy' for 'Fiat'. Filter on the column values mentioned (e.g. Model='fiat').
3. STRICT GROUP BY: Group ONLY by the column(s) requested for aggregation. Do NOT include Primary Keys or other IDs in the GROUP BY unless they are the grouping dimension.
4. JOIN MINIMIZATION: Do not join tables if the required column is already present as a Foreign Key in an already joined table.
5. NO CAST IN SELECT: NEVER wrap a column in CAST() in the SELECT clause. Only use CAST(col AS NUMERIC) inside ORDER BY or WHERE for text-formatted numbers.
6. ALIASING: Use short aliases (e.g., T1, T2) for all tables.
7. BLUEPRINT OVERRIDE: If the blueprint strategy is logically incorrect (e.g. uses EXCEPT for a simple filter), override it with correct SQL.

Schema:
{pruned_schema}

Blueprint:
{logical_blueprint}

Question:
{nl_query}

SQL:"""


CORRECTION_PROMPT_V1 = """You are an expert SQL debugger. Fix the SQL query below.

Use ONLY the column and table names that exist in the schema. Do not invent names or values.
Return ONLY the corrected SQL. No explanation, no markdown, no code fences.

FIX RULES:
1. NO EXTERNAL KNOWLEDGE: Do not hardcode values (e.g. 'Italy'). Filter on the columns mentioned in the question.
2. STRICT GROUP BY: Fix the GROUP BY clause to only include the requested dimensions. Remove redundant Primary Keys or IDs that are not the target of the grouping.
3. JOIN MINIMIZATION: Remove unnecessary joins if the data is available in the current tables.

Schema (ground truth — only these tables and columns exist):
{pruned_schema}

Violations found by static analysis:
{error_message}

Original query (contains errors):
{generated_sql}

Blueprint (intended logic):
{logical_blueprint}

Corrected SQL:"""