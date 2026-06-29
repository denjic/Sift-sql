"""
Multi-Agent Text-to-SQL System — Streamlit Frontend
Covers: Interactive demo + Spider benchmark evaluation dashboard
"""

import streamlit as st
import sqlite3
import os
import json
import time
import re
import difflib
from pathlib import Path
from typing import Optional
from datetime import datetime
import pickle
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Text-to-SQL",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Dependency imports with friendly error messages --------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import sqlglot
    from sqlglot import exp as sqlexp
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K_DEFAULT = 5
MAX_RETRIES = 3
SPIDER_DB_PATH = "./spider/database"
SPIDER_DEV_JSON = "./spider/dev.json"
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_INTERVAL = 5  
LOGS_DIR = "./logs"


#  CHECKPOINT SYSTEM

def init_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(checkpoint_data: dict, timestamp: str = None):
    """
    Save checkpoint to both session state and disk.

    Args:
        checkpoint_data: Dictionary containing evaluation state
        timestamp: Optional timestamp string, auto-generated if None
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to session state (primary)
    st.session_state["checkpoint"] = checkpoint_data
    st.session_state["checkpoint_timestamp"] = timestamp

    # Save to disk (backup)
    init_checkpoint_dir()
    filepath = os.path.join(
        CHECKPOINT_DIR, f"checkpoint_eval_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    return filepath


def load_checkpoint_from_session():
    """Load checkpoint from session state."""
    if "checkpoint" in st.session_state:
        return st.session_state["checkpoint"]
    return None


def load_latest_checkpoint_from_disk():
    """Find and load the most recent checkpoint file from disk."""
    init_checkpoint_dir()
    checkpoint_files = sorted(Path(CHECKPOINT_DIR).glob(
        "checkpoint_eval_*.json"), reverse=True)

    if checkpoint_files:
        latest = checkpoint_files[0]
        with open(latest, 'r') as f:
            data = json.load(f)
        # Also load into session state
        st.session_state["checkpoint"] = data
        st.session_state["checkpoint_timestamp"] = latest.stem.replace(
            "checkpoint_eval_", "")
        return data
    return None


def clear_checkpoint():
    """Clear checkpoint from session state and optionally delete from disk."""
    if "checkpoint" in st.session_state:
        del st.session_state["checkpoint"]
    if "checkpoint_timestamp" in st.session_state:
        timestamp = st.session_state["checkpoint_timestamp"]
        del st.session_state["checkpoint_timestamp"]

        # Delete the checkpoint file
        filepath = os.path.join(
            CHECKPOINT_DIR, f"checkpoint_eval_{timestamp}.json")
        if os.path.exists(filepath):
            os.remove(filepath)


def is_api_limit_error(exception: Exception) -> bool:
    """
    Detect if an exception is due to API rate limit or quota exhaustion.

    Args:
        exception: The caught exception

    Returns:
        True if it's an API limit error, False otherwise
    """
    error_str = str(exception).lower()
    error_type = type(exception).__name__.lower()

    # Common Groq API limit error patterns
    limit_indicators = [
        "rate_limit",
        "rate limit",
        "quota",
        "too many requests",
        "429",
        "limit exceeded",
        "insufficient_quota"
    ]

    return any(indicator in error_str or indicator in error_type for indicator in limit_indicators)


#  TRACE LOGGING SYSTEM


class LogManager:
    """Manages trace logging with batched writes to disk."""

    def __init__(self, eval_timestamp: str):
        self.eval_timestamp = eval_timestamp
        self.log_root = os.path.join(LOGS_DIR, f"eval_{eval_timestamp}")
        self.traces_buffer = {}  
        self.errors_buffer = {}  

        # Create directory structure
        os.makedirs(self.log_root, exist_ok=True)

    def add_trace(self, config: str, trace_entry: dict):
        """Add a trace entry to the buffer."""
        if config not in self.traces_buffer:
            self.traces_buffer[config] = []
            self.errors_buffer[config] = []

        self.traces_buffer[config].append(trace_entry)

        # If this example failed, also add to errors buffer
        if not trace_entry.get("execution_match", False):
            self.errors_buffer[config].append(trace_entry)

    def flush_to_disk(self, config: str = None):
        """Write buffered traces to disk (JSONL format)."""
        configs_to_flush = [config] if config else list(
            self.traces_buffer.keys())

        for cfg in configs_to_flush:
            if cfg not in self.traces_buffer:
                continue

            cfg_dir = os.path.join(self.log_root, cfg)
            os.makedirs(cfg_dir, exist_ok=True)

            # Write traces
            traces_file = os.path.join(cfg_dir, "traces.jsonl")
            with open(traces_file, 'a') as f:
                for trace in self.traces_buffer[cfg]:
                    f.write(json.dumps(trace) + '\n')

            # Write errors
            if self.errors_buffer[cfg]:
                errors_file = os.path.join(cfg_dir, "errors_only.jsonl")
                with open(errors_file, 'a') as f:
                    for error in self.errors_buffer[cfg]:
                        f.write(json.dumps(error) + '\n')

            # Clear buffers after writing
            self.traces_buffer[cfg] = []
            self.errors_buffer[cfg] = []

    def export_summary_csv(self, results_store: dict) -> str:
        """Generate summary CSV file."""
        csv_path = os.path.join(self.log_root, "summary.csv")

        with open(csv_path, 'w') as f:
            # Header
            f.write(
                "configuration,ex_percent,correct,total,avg_latency_ms,avg_corrections\n")

            # Data rows
            for cfg, metrics in results_store.items():
                f.write(f"{cfg},{metrics['EX (%)']},{metrics['Correct']},{metrics['Total']},"
                        f"{metrics['Avg Latency (ms)']},{metrics['Avg Corrections']}\n")

        return csv_path

    def export_detailed_excel(self, results_store: dict) -> str:
        """Generate multi-sheet Excel workbook."""
        try:
            import pandas as pd
            from io import BytesIO

            excel_path = os.path.join(self.log_root, "summary.xlsx")

            # Sheet 1: Evaluation Summary
            summary_data = []
            for cfg, metrics in results_store.items():
                summary_data.append({
                    "Configuration": cfg,
                    "Execution Accuracy (%)": metrics["EX (%)"],
                    "Correct": metrics["Correct"],
                    "Total": metrics["Total"],
                    "Avg Latency (ms)": metrics["Avg Latency (ms)"],
                    "Avg Corrections": metrics["Avg Corrections"]
                })
            df_summary = pd.DataFrame(summary_data)

            # Sheet 2: All Examples (read from traces.jsonl)
            all_examples = []
            for cfg in results_store.keys():
                traces_file = os.path.join(self.log_root, cfg, "traces.jsonl")
                if os.path.exists(traces_file):
                    with open(traces_file, 'r') as f:
                        for line in f:
                            trace = json.loads(line)
                            all_examples.append({
                                "Config": cfg,
                                "Example ID": trace["example_id"],
                                # Truncate for display
                                "Question": trace["question"][:100],
                                "Final SQL": trace["final_sql"][:150],
                                "Golden SQL": trace.get("golden_sql", "")[:150],
                                "Correct": trace["execution_match"],
                                "Latency (ms)": trace["total_latency_ms"],
                                "Corrections": trace.get("correction_attempts", 0)
                            })
            df_all = pd.DataFrame(
                all_examples) if all_examples else pd.DataFrame()

            # Sheet 3: Errors Only
            errors = []
            for cfg in results_store.keys():
                errors_file = os.path.join(
                    self.log_root, cfg, "errors_only.jsonl")
                if os.path.exists(errors_file):
                    with open(errors_file, 'r') as f:
                        for line in f:
                            trace = json.loads(line)
                            error_info = trace.get("error_category", "unknown")
                            errors.append({
                                "Config": cfg,
                                "Example ID": trace["example_id"],
                                "Question": trace["question"][:100],
                                "Error Type": error_info,
                                "Failed Phase": trace.get("failed_at_phase", "N/A"),
                                "Final SQL": trace["final_sql"][:150],
                                "Golden SQL": trace.get("golden_sql", "")[:150]
                            })
            df_errors = pd.DataFrame(errors) if errors else pd.DataFrame()

            # Sheet 4: Phase Latency Breakdown
            phase_latencies = []
            for cfg in results_store.keys():
                traces_file = os.path.join(self.log_root, cfg, "traces.jsonl")
                if os.path.exists(traces_file):
                    with open(traces_file, 'r') as f:
                        for line in f:
                            trace = json.loads(line)
                            for phase in trace.get("phases", []):
                                if "latency_ms" in phase:
                                    phase_latencies.append({
                                        "Config": cfg,
                                        "Example ID": trace["example_id"],
                                        "Phase": phase["name"],
                                        "Latency (ms)": phase.get("latency_ms", 0)
                                    })
            df_latency = pd.DataFrame(
                phase_latencies) if phase_latencies else pd.DataFrame()

            # Write to Excel
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_summary.to_excel(
                    writer, sheet_name='Evaluation Summary', index=False)
                if not df_all.empty:
                    df_all.to_excel(
                        writer, sheet_name='All Examples (Detailed)', index=False)
                if not df_errors.empty:
                    df_errors.to_excel(
                        writer, sheet_name='Errors Only', index=False)
                if not df_latency.empty:
                    df_latency.to_excel(
                        writer, sheet_name='Phase Latency Breakdown', index=False)

            return excel_path

        except ImportError:
            # Fallback if pandas/openpyxl not available
            return None


def create_trace_entry(example_id: int, config: str, question: str,
                       trace: dict, golden_sql: str, execution_match: bool,
                       db_id: str) -> dict:
    """
    Create a structured trace entry with all relevant information.

    Args:
        example_id: Index of the example in the evaluation
        config: Configuration name (baseline, model_a, etc.)
        question: Natural language question
        trace: Pipeline trace from run_pipeline()
        golden_sql: Ground truth SQL from Spider dataset
        execution_match: Whether results match golden SQL
        db_id: Database identifier

    Returns:
        Complete trace entry dictionary
    """
    entry = {
        "example_id": example_id,
        "config": config,
        "db_id": db_id,
        "question": question,
        "golden_sql": golden_sql,
        "final_sql": trace["final_sql"],
        "execution_match": execution_match,
        "total_latency_ms": trace["elapsed_ms"],
        "correction_attempts": trace.get("correction_attempts", 0),
        "phases": []
    }

    # Extract phase details with API prompts/responses where applicable
    for phase_data in trace.get("phases", []):
        phase_entry = {
            "name": phase_data["name"],
            "latency_ms": phase_data.get("latency_ms", 0)
        }

        # Phase 1: Schema Linking
        if "Phase 1" in phase_data["name"]:
            phase_entry["selected_tables"] = phase_data.get(
                "selected_tables", [])
            phase_entry["token_reduction"] = phase_data.get(
                "token_reduction", "0%")
            phase_entry["pruned_schema"] = phase_data.get("pruned_schema", "")

        # Phase 2: CoT Planning (has API call)
        elif "Phase 2" in phase_data["name"]:
            phase_entry["blueprint"] = phase_data.get("blueprint", "")
            if "api_prompt" in phase_data:
                phase_entry["api_prompt"] = phase_data["api_prompt"]
            if "api_response" in phase_data:
                phase_entry["api_response"] = phase_data["api_response"]

        # Phase 3: SQL Generation (has API call)
        elif "Phase 3" in phase_data["name"]:
            phase_entry["generated_sql"] = phase_data.get("sql", "")
            if "api_prompt" in phase_data:
                phase_entry["api_prompt"] = phase_data["api_prompt"]
            if "api_response" in phase_data:
                phase_entry["api_response"] = phase_data["api_response"]

        # Phase 4: Validation + Correction
        elif "Phase 4" in phase_data["name"]:
            phase_entry["validation_history"] = phase_data.get(
                "validation_history", [])
            phase_entry["correction_attempts"] = phase_data.get(
                "correction_attempts", 0)
            phase_entry["final_valid"] = phase_data.get("final_valid")

        entry["phases"].append(phase_entry)

    # Add error categorization for failed examples
    if not execution_match:
        entry["error_category"] = categorize_error(trace)
        entry["failed_at_phase"] = identify_failure_phase(trace)

    return entry


def categorize_error(trace: dict) -> str:
    """Categorize the type of error that occurred."""
    # Check Phase 4 validation
    for phase in trace.get("phases", []):
        if "Phase 4" in phase.get("name", ""):
            if not phase.get("final_valid", True):
                # Check validation history for error types
                val_hist = phase.get("validation_history", [])
                if val_hist and val_hist[-1].get("errors"):
                    error_types = [e["type"] for e in val_hist[-1]["errors"]]
                    if "NONEXISTENT_TABLE" in error_types:
                        return "schema_hallucination_table"
                    if "NONEXISTENT_COLUMN" in error_types:
                        return "schema_hallucination_column"
                    return "ast_validation_failed"

    # If validation passed but execution failed, it's a logic error
    return "logic_error"


def identify_failure_phase(trace: dict) -> str:
    """Identify which phase the error occurred in."""
    for phase in trace.get("phases", []):
        if "Phase 4" in phase.get("name", ""):
            if not phase.get("final_valid", True):
                return "Phase 4 - AST Guard"

    # If AST validation passed but execution failed
    return "Phase 3 - SQL Generation (logic error)"


# CORE PIPELINE FUNCTIONS

# Schema loading 


@st.cache_data(show_spinner=False)
def load_schema_from_sqlite(db_path: str) -> dict:
    """Introspect SQLite database to extract full schema with FK references."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    schema = {}
    for tbl in tables:
        cursor.execute(f"PRAGMA table_info('{tbl}')")
        cols = [{"name": r[1], "type": r[2], "is_pk": bool(
            r[5])} for r in cursor.fetchall()]
        cursor.execute(f"PRAGMA foreign_key_list('{tbl}')")
        for fk in cursor.fetchall():
            for col in cols:
                if col["name"] == fk[3]:
                    col["fk_ref"] = f"{fk[2]}.{fk[4]}"
        schema[tbl] = {"columns": cols}
    conn.close()
    return schema


def get_raw_ddl_schema(db_path: str) -> str:
    """Extract raw CREATE TABLE statements from SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'")
    ddls = [row[0] for row in cursor.fetchall()]
    conn.close()
    return "\n\n".join(ddls)


def serialize_table(tbl_name: str, cols: list) -> str:
    parts = []
    for c in cols:
        p = f"{c['name']} ({c['type']}"
        if c.get("is_pk"):
            p += ", PK"
        if c.get("fk_ref"):
            p += f", FK->{c['fk_ref']}"
        p += ")"
        parts.append(p)
    return f"{tbl_name}: {', '.join(parts)}"


def format_pruned_schema(selected_tables: list, schema: dict) -> str:
    lines = []
    for tbl in selected_tables:
        lines.append(f"TABLE: {tbl}")
        for c in schema[tbl]["columns"]:
            s = f"  - {c['name']} ({c['type']})"
            if c.get("is_pk"):
                s += " [PK]"
            if c.get("fk_ref"):
                s += f" [FK -> {c['fk_ref']}]"
            lines.append(s)
        lines.append("")
    return "\n".join(lines)


# Phase 1: Semantic Schema Linking
_EMBEDDING_MODEL = None


def load_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def embed_schema(_model, db_path: str) -> dict:
    """Cache embeddings per database path."""
    schema = load_schema_from_sqlite(db_path)
    texts = [serialize_table(t, d["columns"]) for t, d in schema.items()]
    names = list(schema.keys())
    vecs = _model.encode(texts, normalize_embeddings=True)
    return {n: vecs[i] for i, n in enumerate(names)}


def retrieve_top_k(nl_query: str, schema_embs: dict, model, k: int = 5) -> list:
    qvec = model.encode([nl_query], normalize_embeddings=True)[0]
    scores = {t: float(np.dot(qvec, e)) for t, e in schema_embs.items()}
    return [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def fk_closure_smart(selected: list, schema: dict, nl_query: str, hops: int = 1) -> list:
    closed, frontier = set(selected), set(selected)
    for _ in range(hops):
        new_f = set()
        for tbl in frontier:
            if tbl not in schema:
                continue
            for col in schema[tbl]["columns"]:
                if col.get("fk_ref"):
                    ref = col["fk_ref"].split(".")[0]
                    if ref in schema and ref not in closed:
                        closed.add(ref)
                        new_f.add(ref)
        frontier = new_f
        if not frontier:
            break
    return list(closed)


# LLM call
def call_groq(system_prompt: str, user_msg: str, api_key: str) -> str:
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=1024,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_msg}
        ]
    )
    raw = resp.choices[0].message.content.strip()
    # Strip chain-of-thought <think> blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw.strip()


def clean_sql(raw: str) -> str:
    """Robustly extract the first executable SQL statement from a raw LLM response.

    Handles:
    - Markdown code fences (```sql ... ``` or ``` ... ```)
    - Inline SQL comments at the end (-- Alternative approach ...)
    - Multi-line comments after the first statement
    - Any prose text that the model appended after the query
    """
    if not raw:
        return ""

    # 1. Strip markdown code fences
    raw = re.sub(r"^```(?:sql)?\s*\n?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw.strip())

    # 2. Split on comment lines that start with "-- " and look like prose
    lines = raw.split("\n")
    sql_lines = []
    prose_comment_pattern = re.compile(
        r"^--\s*(alternative|note:|explanation|approach|or |you can|another|option)",
        re.IGNORECASE
    )
    for line in lines:
        if prose_comment_pattern.match(line.strip()):
            break  # Everything from here onwards is explanatory prose - discard
        sql_lines.append(line)
    raw = "\n".join(sql_lines).strip()

    # 3. If multiple statements are present, keep only the first SELECT statement
    #    (split on semicolons, take the first non-empty fragment)
    statements = [s.strip() for s in raw.split(";") if s.strip()]
    if statements:
        raw = statements[0]  # Only the first complete statement

    return raw.strip()


# Phase 2: CoT Query Planning
PLAN_SYS = """You are a SQL query planner. Given a question and a schema,
output a concise logical blueprint using EXACTLY these 5 fields:

- Strategy: one of SIMPLE_SELECT | JOIN | SUBQUERY | INTERSECT | UNION | EXCEPT
- Tables: comma-separated list of tables needed
- Joins: join conditions, or NONE
- Aggregation: e.g. COUNT, SUM, AVG, or NONE
- Special: ORDER BY, LIMIT, HAVING, GROUP BY details, or NONE

STRICT OUTPUT FORMATTING RULES:
1. RETURN EXACTLY 5 LINES matching the 5 bullet points above. No extra lines.
2. DO NOT write any explanations, notes, or "Alternative Strategies".
3. DO NOT write any SQL code.
4. If you write anything other than the 5 bullet points, you will break the pipeline.

CRITICAL RULES — read carefully:
1. NO EXTERNAL KNOWLEDGE: Never assume facts (e.g. 'Fiats are from Italy'). Filter on the column value mentioned in the question.
2. AMBIGUITY: 'Larger than any' -> `> (SELECT MIN(...))`. 'Larger than all' -> `> (SELECT MAX(...))`.
3. GROUP BY: Only group by the requested dimension. NEVER add Primary Keys to GROUP BY unless explicitly asked.
4. JOIN MINIMIZATION: Do NOT join a table if its data is already available via a FK column in a table you already selected. DO NOT use JOIN as a substitute for INTERSECT or EXCEPT.
5. SET OPERATIONS - use this guide:
   - Question has "both X and Y" or "in A and also in B" -> Strategy: INTERSECT
   - Question has "in A but not in B" or "A that are not in B" -> Strategy: EXCEPT
   - Question has "X or Y" across separate row-sets -> Strategy: UNION
   - Question has "not speak X", "not from X", "whose X is not Y" -> Strategy: SIMPLE_SELECT with WHERE NOT IN / !=
   NEVER replace INTERSECT or EXCEPT with a JOIN.
6. COUNT DISTINCT: For "how many different X" or "how many unique X", use COUNT(DISTINCT col), NOT a GROUP BY self-join.

FEW-SHOT EXAMPLES:
Q: "Find the states where both owners and professionals live."
-> Strategy: INTERSECT
-> Tables: Owners, Professionals
-> Joins: NONE
-> Aggregation: NONE
-> Special: NONE

Q: "Give the names of nations that speak both English and French."
-> Strategy: INTERSECT
-> Tables: country, countrylanguage
-> Joins: country.Code = countrylanguage.CountryCode
-> Aggregation: NONE
-> Special: NONE

Q: "What are the country codes for countries that do not speak English?"
-> Strategy: EXCEPT
-> Tables: countrylanguage
-> Joins: NONE
-> Aggregation: NONE
-> Special: NONE

Q: "How many different forms of governments are there in Africa?"
-> Strategy: SIMPLE_SELECT
-> Tables: country
-> Joins: NONE
-> Aggregation: COUNT(DISTINCT GovernmentForm)
-> Special: WHERE Continent = 'Africa'"""


def phase2_plan(nl_query: str, pruned_schema_str: str, api_key: str) -> tuple[str, str]:
    """Returns (prompt, response) tuple for logging."""
    prompt = f"Schema:\n{pruned_schema_str}\n\nQuestion:\n{nl_query}\n\nLogical Blueprint:"
    response = call_groq(PLAN_SYS, prompt, api_key)
    return prompt, response


# Phase 3: SQL Generation
# model_c: expert rules + CoT blueprint section in prompt
GEN_SYS = """Generate a single SQLite SQL query. Return ONLY raw SQL - no explanation, no markdown, no code fences.

CRITICAL RULES:
1. STRICT SELECTION: Select ONLY the exact columns asked for. Extra columns cause execution mismatches.
2. NO EXTERNAL KNOWLEDGE: Do not hardcode values. Filter on columns mentioned (e.g. Model='fiat' not 'Italy').
3. STRICT GROUP BY: Group ONLY by the requested dimension. Remove redundant PKs/IDs.
4. JOIN MINIMIZATION: Do NOT join tables that are not in the Blueprint's Tables list. Do NOT use JOIN as a substitute for INTERSECT or EXCEPT - they produce different results.
5. NO CAST IN SELECT: Only use CAST() inside ORDER BY or WHERE, never in SELECT clause.
6. ALIASING: Use short aliases (T1, T2) for all tables.
7. COLUMN QUALIFICATION: Always qualify ambiguous columns with table alias.
8. SET OPERATIONS: When the Blueprint says Strategy: INTERSECT, EXCEPT, or UNION - you MUST use those SQL keywords, NOT a JOIN with GROUP BY.
9. COUNT DISTINCT: When the Blueprint says COUNT(DISTINCT ...) - use exactly that, do NOT rewrite as a GROUP BY with self-join.

FEW-SHOT EXAMPLES (INTERSECT / EXCEPT / COUNT DISTINCT):
Q: "Find the states where both owners and professionals live."
Blueprint: Strategy: INTERSECT, Tables: Owners, Professionals
A: SELECT state FROM Owners INTERSECT SELECT state FROM Professionals

Q: "Give the names of nations that speak both English and French."
Blueprint: Strategy: INTERSECT, Tables: country, countrylanguage
A: SELECT T1.Name FROM country T1 JOIN countrylanguage T2 ON T1.Code=T2.CountryCode WHERE T2.Language='English' INTERSECT SELECT T1.Name FROM country T1 JOIN countrylanguage T2 ON T1.Code=T2.CountryCode WHERE T2.Language='French'

Q: "What are the country codes for countries that do not speak English?"
Blueprint: Strategy: EXCEPT, Tables: countrylanguage
A: SELECT CountryCode FROM countrylanguage EXCEPT SELECT CountryCode FROM countrylanguage WHERE Language='English'

Q: "How many different forms of governments are there in Africa?"
Blueprint: Strategy: SIMPLE_SELECT, Aggregation: COUNT(DISTINCT GovernmentForm)
A: SELECT COUNT(DISTINCT GovernmentForm) FROM country WHERE Continent='Africa'

Q: "How many employees are in each department?"
Blueprint: Strategy: JOIN, Tables: Employee, Department
A: SELECT T2.name, COUNT(*) FROM Employee T1 JOIN Department T2 ON T1.dept_id=T2.id GROUP BY T2.id"""

# model_a / model_b: same expert rules but NO blueprint section in prompt
GEN_SYS_NO_PLAN = """You are a SQLite Expert. You are given a database Schema and an English question. Write a valid SQL query.
RULES:
1. STRICT SELECTION: Select ONLY the exact columns asked for in the question.
2. NO CAST IN SELECT: NEVER wrap a column in CAST() in the SELECT clause or inside aggregates. Only use CAST(col AS NUMERIC) inside ORDER BY or WHERE.
3. ALIASING: Use short aliases (e.g., T1, T2) for all tables.
4. JOIN LIMIT: Only join absolutely necessary tables. If an entity can be fetched from a direct Foreign Key column (e.g., size_code), do NOT join the referenced table.
5. NO MARKDOWN: Return only the raw SQL string.
6. PRIMARY KEY GROUPING: Prefer grouping by primary keys (IDs) rather than name columns.

Example for TEXT columns: SELECT Model FROM T1 ORDER BY CAST(Horsepower AS NUMERIC) DESC;"""

# baseline: truly zero-shot - minimal system prompt, no expert SQL rules
BASELINE_SYS = """You are a SQL assistant. Translate the user's English question into a single valid SQLite SELECT statement.
Return ONLY the raw SQL. No explanations, no markdown, no code fences."""


def phase3_generate(nl_query: str, pruned_schema_str: str, blueprint: str, api_key: str) -> tuple[str, str]:
    """
    model_c path: expert rules (GEN_SYS) + pruned schema + CoT blueprint.
    The Blueprint section is always present and non-empty for model_c.
    """
    prompt = f"Schema:\n{pruned_schema_str}\n\nBlueprint:\n{blueprint}\n\nQuestion:\n{nl_query}\n\nSQL:"
    raw = call_groq(GEN_SYS, prompt, api_key)
    sql = clean_sql(raw)
    return prompt, sql


def phase3_generate_no_plan(nl_query: str, pruned_schema_str: str, api_key: str) -> tuple[str, str]:
    """
    model_a / model_b path: expert rules (GEN_SYS_NO_PLAN) + pruned schema + question.
    Blueprint section is OMITTED entirely - no empty header pollution.
    """
    prompt = f"Schema:\n{pruned_schema_str}\n\nQuestion:\n{nl_query}\n\nSQL:"
    raw = call_groq(GEN_SYS_NO_PLAN, prompt, api_key)
    sql = clean_sql(raw)
    return prompt, sql


def phase3_generate_baseline(nl_query: str, raw_schema_str: str, api_key: str) -> tuple[str, str]:
    """
    baseline path: zero-shot, minimal BASELINE_SYS, full unformatted schema, no blueprint or expert rules.
    """
    prompt = f"Database schema:\n{raw_schema_str}\n\nQuestion: {nl_query}\n\nSQL:"
    raw = call_groq(BASELINE_SYS, prompt, api_key)
    sql = clean_sql(raw)
    return prompt, sql


# ---- Phase 4A: AST Schema Guard --------------------------------------------------------------------------------------------------
def ast_validate(sql: str, schema: dict) -> tuple[bool, list[dict]]:
    errors = []
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception as e:
        return False, [{"level": 1, "type": "SYNTAX_ERROR", "detail": str(e), "suggestion": None}]

    valid_tables = {t.lower() for t in schema}
    valid_columns = {}
    for tbl, data in schema.items():
        for col in data["columns"]:
            valid_columns[(tbl.lower(), col["name"].lower())] = True
            valid_columns[(None, col["name"].lower())] = True

    # Build a map of alias -> actual table name
    aliases = {}
    for t_node in ast.find_all(sqlexp.Table):
        tname = t_node.name.lower()
        if t_node.alias:
            aliases[t_node.alias.lower()] = tname

        if tname and tname not in valid_tables:
            sugg = difflib.get_close_matches(
                tname, valid_tables, n=1, cutoff=0.4)
            errors.append({"level": 2, "type": "NONEXISTENT_TABLE",
                           "detail": f"Table '{tname}' not in schema.",
                           "suggestion": sugg[0] if sugg else None})

    valid_aliases = {a.alias.lower()
                     for a in ast.find_all(sqlexp.Alias) if a.alias}

    valid_cte_table_aliases = set()
    for subq in ast.find_all(sqlexp.Subquery):
        if subq.alias:
            valid_cte_table_aliases.add(subq.alias.lower())
    for cte in ast.find_all(sqlexp.CTE):
        if cte.alias:
            valid_cte_table_aliases.add(cte.alias.lower())

    for c_node in ast.find_all(sqlexp.Column):
        cname = c_node.name.lower()
        if cname in valid_aliases:
            continue

        raw_tname = c_node.table.lower() if c_node.table else None

        if raw_tname:
            # Resolve alias to actual table name if applicable
            actual_tname = aliases.get(raw_tname, raw_tname)

            # Subqueries and CTEs create virtual tables that bypass schema validation
            if actual_tname in valid_cte_table_aliases or raw_tname in valid_cte_table_aliases:
                continue

            if (actual_tname, cname) not in valid_columns:
                tbl_cols = [c["name"].lower() for c in schema.get(
                    actual_tname, {}).get("columns", [])]
                sugg = difflib.get_close_matches(
                    cname, tbl_cols, n=1, cutoff=0.4)
                errors.append({"level": 2, "type": "NONEXISTENT_COLUMN",
                               "detail": f"Column '{cname}' not in table '{actual_tname}'.",
                               "suggestion": f"{raw_tname}.{sugg[0]}" if sugg else None})
        else:
            # Unqualified column
            if (None, cname) not in valid_columns:
                # Gather all valid columns across all tables for suggestion
                all_cols = []
                for data in schema.values():
                    all_cols.extend([c["name"].lower() for c in data.get("columns", [])])
                sugg = difflib.get_close_matches(cname, all_cols, n=1, cutoff=0.4)
                errors.append({"level": 2, "type": "NONEXISTENT_COLUMN",
                               "detail": f"Column '{cname}' not found in any table.",
                               "suggestion": sugg[0] if sugg else None})

    return len(errors) == 0, errors


def format_errors(errors: list) -> str:
    lines = ["Schema violations detected:\n"]
    for i, e in enumerate(errors, 1):
        line = f"{i}. [{e['type']}] {e['detail']}"
        if e.get("suggestion"):
            line += f"\n   -> Use '{e['suggestion']}' instead."
        lines.append(line)
    return "\n".join(lines)


# Phase 4B: Self-Correction 
CORR_SYS = """You are an expert SQL debugger. Fix the SQL query below.

Use ONLY the column and table names that exist in the schema. Do not invent names or values.
Return ONLY the corrected SQL. No explanation, no markdown, no code fences.

FIX RULES:
1. NO EXTERNAL KNOWLEDGE: Do not hardcode values (e.g. 'Italy'). Filter on the columns mentioned in the question.
2. STRICT GROUP BY: Fix the GROUP BY clause to only include the requested dimensions. Remove redundant Primary Keys or IDs that are not the target of the grouping.
3. JOIN MINIMIZATION: Remove unnecessary joins if the data is available in the current tables.
4. SET OPERATIONS: If the Blueprint says Strategy: INTERSECT, EXCEPT, or UNION — and the query is using JOIN instead — rewrite it using the correct set operation keyword. A JOIN produces different results from INTERSECT/EXCEPT.
5. COUNT DISTINCT: If the Blueprint says COUNT(DISTINCT ...) — use exactly that. Do not rewrite as a GROUP BY with extra joins.

Schema (ground truth — only these tables and columns exist):
{pruned_schema}

Violations found by static analysis:
{error_message}

Original query (contains errors):
{generated_sql}

Blueprint (intended logic):
{logical_blueprint}

Corrected SQL:"""

EXEC_CORR_SYS = """You are a SQL Debugger. Fix the SQL query that failed execution against SQLite.
FIX STRATEGY:
1. Read the SQLite error message carefully — it tells you exactly what is wrong.
2. If 'ambiguous column name' → qualify the column with its table alias.
3. If 'no such column' or 'no such table' → check the schema for the correct name.
4. If the error is about aggregate misuse → restructure GROUP BY / HAVING.
5. If the query logic seems wrong for the question, rewrite using the simplest correct approach.
6. Return ONLY the corrected SQL. No explanations."""


def phase4b_correct(sql: str, errors: list, schema_str: str, blueprint: str, api_key: str) -> str:
    """Correct SQL based on AST validation errors (schema hallucinations)."""
    user = (f"Original SQL:\n{sql}\n\n"
            f"Errors:\n{format_errors(errors)}\n\n"
            f"Schema:\n{schema_str}\n\n"
            f"Blueprint:\n{blueprint}\n\nCorrected SQL:")
    raw = call_groq(CORR_SYS, user, api_key)
    return clean_sql(raw)


def phase4_execute_validate(sql: str, db_path: str) -> tuple[bool, str]:
    """Dry-run SQL against SQLite to catch runtime errors that AST validation misses."""
    if not sql or not sql.strip():
        return False, "Empty SQL query"
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        cur.fetchmany(5)  # Fetch a few rows to verify full execution
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        if conn:
            conn.close()


def phase4b_correct_exec(sql: str, exec_error: str, schema_str: str, blueprint: str,
                         nl_query: str, api_key: str) -> str:
    """Correct SQL based on execution error from SQLite runtime."""
    user = (f"Original SQL:\n{sql}\n\n"
            f"SQLite Error:\n{exec_error}\n\n"
            f"Schema:\n{schema_str}\n\n"
            f"Original Question:\n{nl_query}\n\nCorrected SQL:")
    raw = call_groq(EXEC_CORR_SYS, user, api_key)
    return clean_sql(raw)



#  LANGGRAPH NODES (Phase 4 Validation Loop)

def ast_guard_node(state: dict) -> dict:
    sql = state["sql"]
    attempt = state["attempt"]
    hist = list(state.get("validation_history", []))
    
    # Step A: AST structural check
    is_valid, ast_errors = ast_validate(sql, state["pruned"])
    
    if not is_valid:
        hist.append({"attempt": attempt, "valid": False, "errors": ast_errors, "error_source": "ast"})
        return {**state, "validation_history": hist, "is_valid": False, "ast_errors": ast_errors, "error_source": "ast"}
    
    # Step B: Execution dry-run
    exec_ok, exec_err = phase4_execute_validate(sql, state["db_path"])
    if exec_ok:
        hist.append({"attempt": attempt, "valid": True, "errors": [], "error_source": None})
        return {**state, "validation_history": hist, "is_valid": True, "error_source": None}
    else:
        hist.append({"attempt": attempt, "valid": False, "errors": [{"type": "EXECUTION_ERROR", "detail": exec_err}], "error_source": "execution"})
        return {**state, "validation_history": hist, "is_valid": False, "exec_err": exec_err, "error_source": "execution"}


def self_correction_node(state: dict) -> dict:
    sql = state["sql"]
    if state.get("error_source") == "ast":
        new_sql = phase4b_correct(sql, state["ast_errors"], state["schema_str"], state["blueprint"], state["api_key"])
    else:
        new_sql = phase4b_correct_exec(sql, state["exec_err"], state["schema_str"], state["blueprint"], state["nl_query"], state["api_key"])
        
    return {
        **state,
        "prev_sql": sql,
        "sql": new_sql,
        "correction_attempts": state.get("correction_attempts", 0) + 1,
        "attempt": state["attempt"] + 1
    }


def route_after_guard(state: dict) -> str:
    if state.get("is_valid"):
        return END
    
    if state["attempt"] >= state["max_retries"]:
        return END
        
    if state.get("prev_sql") and state["sql"] == state["prev_sql"]:
        return END
        
    return "correct"


def compile_correction_graph():
    workflow = StateGraph(dict)
    
    workflow.add_node("guard", ast_guard_node)
    workflow.add_node("correct", self_correction_node)
    
    workflow.set_entry_point("guard")
    workflow.add_conditional_edges("guard", route_after_guard, {
        END: END,
        "correct": "correct"
    })
    workflow.add_edge("correct", "guard")
    
    return workflow.compile()


# Full pipeline
def run_pipeline(nl_query: str, db_path: str, api_key: str,
                 top_k: int = 5, max_retries: int = 3,
                 config: str = "model_c") -> dict:
    """
    Runs the full 4-phase pipeline and returns a trace dict.
    config: "baseline" | "model_a" | "model_b" | "model_c"
    """
    t0 = time.time()
    schema = load_schema_from_sqlite(db_path)
    trace = {"config": config, "query": nl_query, "phases": []}

    # ---- Phase 1 ----
    t1 = time.time()
    if config in ("model_a", "model_b", "model_c"):
        model = load_embedding_model()
        embs = embed_schema(model, db_path)
        top_tables = retrieve_top_k(nl_query, embs, model, top_k)
        top_tables = fk_closure_smart(top_tables, schema, nl_query, hops=1)
        pruned = {t: schema[t] for t in top_tables if t in schema}
        schema_str = format_pruned_schema(list(pruned.keys()), pruned)
        trace["phases"].append({
            "name": "Phase 1 - Schema Linking",
            "selected_tables": top_tables,
            "token_reduction": f"{100 - int(len(pruned)/max(len(schema), 1)*100)}%",
            "pruned_schema": schema_str,
            "latency_ms": int((time.time() - t1) * 1000)
        })
    else:
        pruned = schema
        schema_str = get_raw_ddl_schema(db_path)
        trace["phases"].append({
            "name": "Phase 1 - Schema Linking",
            "selected_tables": list(schema.keys()),
            "token_reduction": "0% (baseline - full unformatted DDL schema)",
            "pruned_schema": schema_str,
            "latency_ms": int((time.time() - t1) * 1000)
        })

    # ---- Phase 2 ----
    t2 = time.time()
    if config == "model_c":
        prompt, blueprint = phase2_plan(nl_query, schema_str, api_key)
        trace["phases"].append({
            "name": "Phase 2 - CoT Planning",
            "blueprint": blueprint,
            "api_prompt": prompt,
            "api_response": blueprint,
            "latency_ms": int((time.time() - t2) * 1000)
        })
    else:
        blueprint = ""  # No CoT plan for non-model_c configs
        trace["phases"].append({
            "name": "Phase 2 - CoT Planning",
            "blueprint": "(Skipped - not in this config)",
            "latency_ms": 0
        })

    # ---- Phase 3 ----
    t3 = time.time()
    if config == "baseline":
        # Zero-shot: minimal BASELINE_SYS, full unformatted schema, no blueprint, no expert rules
        prompt, sql = phase3_generate_baseline(nl_query, schema_str, api_key)
    elif config == "model_c":
        # Full agentic: expert GEN_SYS + pruned schema + CoT blueprint
        prompt, sql = phase3_generate(nl_query, schema_str, blueprint, api_key)
    else:
        # model_a / model_b: expert GEN_SYS_NO_PLAN + pruned schema, Blueprint section OMITTED
        prompt, sql = phase3_generate_no_plan(nl_query, schema_str, api_key)
    trace["phases"].append({
        "name": "Phase 3 - SQL Generation",
        "sql": sql,
        "api_prompt": prompt,
        "api_response": sql,
        "latency_ms": int((time.time() - t3) * 1000)
    })

    # ---- Phase 4 ----
    t4 = time.time()
    correction_attempts = 0
    validation_history = []

    if config in ("model_b", "model_c"):
        if LANGGRAPH_AVAILABLE:
            graph = compile_correction_graph()
            initial_state = {
                "sql": sql,
                "pruned": pruned,
                "schema_str": schema_str,
                "blueprint": blueprint,
                "api_key": api_key,
                "nl_query": nl_query,
                "db_path": db_path,
                "attempt": 0,
                "max_retries": max_retries,
                "validation_history": [],
                "correction_attempts": 0,
                "prev_sql": "",
                "is_valid": False,
                "ast_errors": [],
                "exec_err": "",
                "error_source": ""
            }
            
            final_state = graph.invoke(initial_state)
            
            sql = final_state["sql"]
            validation_history = final_state["validation_history"]
            correction_attempts = final_state.get("correction_attempts", 0)
        else:
            # Fallback if langgraph is not installed
            for attempt in range(max_retries + 1):
                # Step A: AST structural check (fast, no DB hit)
                is_valid, ast_errors = ast_validate(sql, pruned)
                if not is_valid:
                    validation_history.append({"attempt": attempt, "valid": False,
                                               "errors": ast_errors, "error_source": "ast"})
                    if attempt < max_retries:
                        prev_sql = sql
                        sql = phase4b_correct(
                            sql, ast_errors, schema_str, blueprint, api_key)
                        correction_attempts += 1
                        if sql == prev_sql:
                            break
                    continue

                # Step B: Execution dry-run (catches runtime errors AST misses)
                exec_ok, exec_err = phase4_execute_validate(sql, db_path)
                if exec_ok:
                    validation_history.append({"attempt": attempt, "valid": True,
                                               "errors": [], "error_source": None})
                    break
                else:
                    validation_history.append({
                        "attempt": attempt, "valid": False,
                        "errors": [{"type": "EXECUTION_ERROR", "detail": exec_err}],
                        "error_source": "execution"
                    })
                    if attempt < max_retries:
                        prev_sql = sql
                        sql = phase4b_correct_exec(sql, exec_err, schema_str, blueprint,
                                                   nl_query, api_key)
                        correction_attempts += 1
                        if sql == prev_sql:
                            break

        trace["phases"].append({
            "name": "Phase 4 - Validation + Self-Correction",
            "validation_history": validation_history,
            "correction_attempts": correction_attempts,
            "final_valid": validation_history[-1]["valid"] if validation_history else None,
            "latency_ms": int((time.time() - t4) * 1000)
        })
    else:
        trace["phases"].append({
            "name": "Phase 4 - Validation + Self-Correction",
            "validation_history": [],
            "correction_attempts": 0,
            "final_valid": None,
            "latency_ms": 0
        })

    trace["final_sql"] = sql
    trace["elapsed_ms"] = int((time.time() - t0) * 1000)
    trace["correction_attempts"] = correction_attempts
    return trace


#  EVALUATION HELPERS

def normalize_row(row):
    try:
        return tuple(sorted(row))
    except TypeError:
        return tuple(sorted(str(x) if x is not None else "" for x in row))


def execute_sql(sql: str, db_path: str):
    if not sql or not sql.strip():
        return None, "empty_sql"
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        normalized_rows = [normalize_row(r) for r in rows]
        return frozenset(normalized_rows), None
    except Exception as e:
        return None, str(e)


def execution_accuracy(pred_sql, gold_sql, db_path):
    if not pred_sql or not pred_sql.strip():
        return False
    pred_rows, pred_err = execute_sql(pred_sql, db_path)
    gold_rows, gold_err = execute_sql(gold_sql, db_path)
    if pred_err or gold_err:
        return False
    return pred_rows == gold_rows


#  STREAMLIT UI

# Custom CSS 

st.markdown("""
<style>
.phase-box   { background:#f8f9fa; border-left:4px solid #4CAF50;
               padding:12px 16px; border-radius:6px; margin:8px 0; }
.phase-fail  { border-left-color:#f44336; }
.phase-warn  { border-left-color:#FF9800; }
.metric-card { background:#f0f7ff; border:1px solid #90caf9;
               border-radius:8px; padding:12px; text-align:center; }
.sql-box     { background:#1e1e1e; color:#d4d4d4; padding:14px;
               border-radius:6px; font-family:monospace; font-size:13px;
               white-space:pre-wrap; }
.badge-pass  { background:#e8f5e9; color:#2e7d32; padding:2px 10px;
               border-radius:12px; font-size:12px; font-weight:600; }
.badge-fail  { background:#ffebee; color:#c62828; padding:2px 10px;
               border-radius:12px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("Configuration")

    api_key = st.text_input("Groq API Key", type="password",
                            help="Get a free key at console.groq.com")

    st.divider()
    st.subheader("Pipeline Settings")
    top_k = st.slider("Schema top-k tables", 3, 10, TOP_K_DEFAULT)
    max_retries = st.slider("Max correction retries", 1, 5, MAX_RETRIES)

    st.divider()
    st.subheader("Dataset")
    spider_path = st.text_input("Spider database folder", SPIDER_DB_PATH)
    spider_dev = st.text_input("Spider dev.json path",   SPIDER_DEV_JSON)


    # Dependency checks
    st.divider()
    st.subheader("Dependency Check")
    st.write("sentence-transformers",
             "OK" if ST_AVAILABLE else "MISSING - run: pip install sentence-transformers")
    st.write("sqlglot", "OK" if SQLGLOT_AVAILABLE else "MISSING - run: pip install sqlglot")
    st.write("groq", "OK" if GROQ_AVAILABLE else "MISSING - run: pip install groq")


# ---- Tab layout ----------------------------------------------------------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Interactive Demo", "Benchmark Evaluation", "Results Dashboard"])



# TAB 1 - INTERACTIVE DEMO

with tab1:
    st.header("Interactive Query Demo")
    st.caption(
        "Type a natural language question and watch the 4-phase pipeline process it in real time.")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Database selector
        db_options = []
        if os.path.exists(spider_path):
            db_options = [d for d in os.listdir(spider_path)
                          if os.path.isdir(os.path.join(spider_path, d))]

        if db_options:
            selected_db = st.selectbox(
                "Select Spider database", sorted(db_options))
            db_file = os.path.join(
                spider_path, selected_db, f"{selected_db}.sqlite")
        else:
            db_file = st.text_input(
                "SQLite database path", placeholder="./my_database.sqlite")

        nl_query = st.text_area(
            "Natural Language Question",
            placeholder='e.g. "How many singers are there?"',
            height=80
        )

        config_choice = st.radio(
            "Pipeline configuration",
            ["model_c - Full Agentic (all 4 phases)",
             "model_b - Linking + Guard (no planner)",
             "model_a - Linking only",
             "baseline - Zero-Shot (no agents)"],
            horizontal=True
        )
        config = config_choice.split(" - ")[0]

    with col2:
        st.markdown("### Quick Test Queries")
        quick = {
            "concert_singer": [
                "How many singers are there?",
                "What are the names of singers from USA?",
                "List all concerts with more than 1000 attendees"
            ],
            "car_1": [
                "How many car models are there?",
                "What is the most common car maker country?",
            ]
        }
        if db_options and selected_db in quick:
            for q in quick[selected_db]:
                if st.button(q, key=q):
                    nl_query = q

    run_btn = st.button("▶ Run Pipeline", type="primary",
                        disabled=not (api_key and nl_query and db_file and os.path.exists(str(db_file))))

    if not api_key:
        st.info("Add your Groq API key in the sidebar to run the pipeline.")

    if run_btn:
        st.divider()
        st.subheader("Pipeline Execution Trace")

        # Progress display
        progress = st.progress(0)
        status_ph = st.empty()

        with st.spinner("Running pipeline..."):
            try:
                # Phase status placeholders
                ph1 = st.empty()
                ph2 = st.empty()
                ph3 = st.empty()
                ph4 = st.empty()

                status_ph.info("Phase 1: Semantic Schema Linking...")
                progress.progress(10)
                trace = run_pipeline(nl_query, db_file, api_key,
                                     top_k=top_k, max_retries=max_retries, config=config)
                progress.progress(100)
                status_ph.success(f"Completed in {trace['elapsed_ms']}ms")

                # ---- Display phase cards ----
                for phase in trace["phases"]:
                    pname = phase["name"]

                    if "Phase 1" in pname:
                        with st.expander(f"{pname}", expanded=True):
                            c1, c2 = st.columns(2)
                            c1.metric("Tables selected", len(
                                phase["selected_tables"]))
                            c2.metric("Token reduction",
                                      phase["token_reduction"])
                            st.write("**Selected tables:**",
                                     ", ".join(phase["selected_tables"]))
                            schema_title = "View full raw schema" if config == "baseline" else "View pruned schema"
                            with st.expander(schema_title):
                                st.code(phase["pruned_schema"], language="sql")

                    elif "Phase 2" in pname:
                        skipped = "(Skipped" in phase["blueprint"]
                        with st.expander(f"{pname}", expanded=not skipped):
                            if skipped:
                                st.caption(
                                    "CoT Planning skipped in this configuration.")
                            else:
                                st.markdown(phase["blueprint"])

                    elif "Phase 3" in pname:
                        with st.expander(f"{pname}", expanded=True):
                            st.markdown('<div class="sql-box">' + phase["sql"] + '</div>',
                                        unsafe_allow_html=True)

                    elif "Phase 4" in pname:
                        hist = phase["validation_history"]
                        if not hist:
                            with st.expander("Phase 4 - Validation + Self-Correction"):
                                st.caption(
                                    "Validation skipped in this configuration.")
                        else:
                            final_valid = hist[-1]["valid"]
                            with st.expander(f"{pname}", expanded=True):
                                for h in hist:
                                    if h["valid"]:
                                        st.markdown(f'<span class="badge-pass">Attempt {h["attempt"]+1}: PASSED</span>',
                                                    unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<span class="badge-fail">Attempt {h["attempt"]+1}: FAILED</span>',
                                                    unsafe_allow_html=True)
                                        for e in h["errors"]:
                                            st.warning(f"**{e['type']}** - {e['detail']}" +
                                                       (f"\n-> Suggestion: `{e['suggestion']}`" if e.get("suggestion") else ""))
                                if phase["correction_attempts"] > 0:
                                    st.info(
                                        f"Self-Correction invoked {phase['correction_attempts']} time(s)")

                # ---- Final SQL + execution ----
                st.divider()
                st.subheader("Final SQL Output")
                st.code(trace["final_sql"], language="sql")

                if os.path.exists(str(db_file)):
                    import pandas as pd
                    try:
                        conn = sqlite3.connect(db_file)
                        cur = conn.cursor()
                        cur.execute(trace["final_sql"])
                        raw_cols = [d[0] for d in cur.description]
                        all_rows = cur.fetchall()
                        conn.close()

                        st.success(
                            f"Query executed successfully - {len(all_rows)} rows returned")

                        if all_rows:
                            # Fix duplicate column names from SELECT * queries
                            seen = {}
                            deduped_cols = []
                            for c in raw_cols:
                                if c in seen:
                                    seen[c] += 1
                                    deduped_cols.append(f"{c}_{seen[c]}")
                                else:
                                    seen[c] = 0
                                    deduped_cols.append(c)
                            df = pd.DataFrame(all_rows, columns=deduped_cols)
                            st.dataframe(df, width='stretch')
                    except Exception as exec_err:
                        st.error(f"Execution error: {exec_err}")

                # ── Latency breakdown ──
                st.caption(f"Total pipeline time: {trace['elapsed_ms']}ms | "
                           f"Corrections: {trace['correction_attempts']} | "
                           f"Config: {config}")

            except Exception as e:
                progress.progress(0)
                status_ph.error(f"Pipeline error: {str(e)}")
                st.exception(e)


#  TAB 2 - BENCHMARK EVALUATION (WITH CHECKPOINT SUPPORT)
with tab2:
    st.header("Spider Benchmark Evaluation")
    st.caption(
        "Run ablation study across all 4 configurations. Results reproduce Table I from the paper.")

    # Download Logs So Far button (always visible)
    if "log_manager" in st.session_state and "evaluation_running" in st.session_state:
        log_mgr = st.session_state["log_manager"]

        # Create temporary exports
        temp_results = st.session_state.get(
            "checkpoint", {}).get("results_store", {})
        if temp_results:
            try:
                csv_data = []
                for cfg, metrics in temp_results.items():
                    csv_data.append(f"{cfg},{metrics['EX (%)']},{metrics['Correct']},{metrics['Total']},"
                                    f"{metrics['Avg Latency (ms)']},{metrics['Avg Corrections']}")

                csv_content = "configuration,ex_percent,correct,total,avg_latency_ms,avg_corrections\n"
                csv_content += "\n".join(csv_data)

                st.download_button(
                    "Download Logs So Far",
                    csv_content,
                    "partial_results.csv",
                    "text/csv",
                    help="Download results and logs from completed configurations"
                )
            except:
                pass

    # ---- Check for existing checkpoint on app load ----
    checkpoint_banner_placeholder = st.empty()
    existing_checkpoint = load_checkpoint_from_session(
    ) or load_latest_checkpoint_from_disk()

    if existing_checkpoint and "evaluation_paused" in st.session_state and st.session_state["evaluation_paused"]:
        with checkpoint_banner_placeholder.container():
            st.warning(
                "**API Limit Reached!** Evaluation paused. Replace your API key and click 'Resume from Checkpoint' to continue.")
    elif existing_checkpoint and not ("evaluation_running" in st.session_state and st.session_state["evaluation_running"]):
        timestamp = st.session_state.get("checkpoint_timestamp", "unknown")
        with checkpoint_banner_placeholder.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(
                    f"Found incomplete evaluation from checkpoint `{timestamp}`. You can resume it or start fresh.")
            with col2:
                if st.button("Discard Checkpoint"):
                    clear_checkpoint()
                    st.rerun()

    if not os.path.exists(spider_dev):
        st.warning(f"Spider dev.json not found at `{spider_dev}`. "
                   "Download Spider from https://yale-lily.github.io/spider")
    else:
        with open(spider_dev) as f:
            dev_data = json.load(f)
        st.success(f"Spider dev set loaded: {len(dev_data)} examples")

    n_samples = st.slider("Number of examples to evaluate", 10, 200, 50,
                          help="Start with 50 for a quick run (~15 min on free Groq tier)")
    configs_to_run = st.multiselect(
        "Configurations to evaluate",
        ["baseline", "model_a", "model_b", "model_c"],
        default=["baseline", "model_c"],
        help="Run all 4 for the full ablation. Start with baseline + model_c for the key result."
    )

    # Buttons with checkpoint logic
    button_col1, button_col2 = st.columns(2)

    with button_col1:
        # Disable Start button if checkpoint exists and evaluation is paused
        start_disabled = (
            not (api_key and os.path.exists(spider_dev) and configs_to_run) or
            ("evaluation_paused" in st.session_state and st.session_state["evaluation_paused"])
        )
        run_eval = st.button("▶ Start Evaluation",
                             type="primary", disabled=start_disabled)

    with button_col2:
        # Resume button only enabled if checkpoint exists
        resume_enabled = existing_checkpoint is not None and api_key
        resume_eval = st.button(
            "Resume from Checkpoint", type="secondary", disabled=not resume_enabled)

    # Helper function for running evaluation 
    def estimate_difficulty(sample):
        q = sample.get('query', '').lower()
        score = 0
        if 'join' in q:
            score += 1
        if q.count('join') > 1:
            score += 1
        if 'group by' in q:
            score += 1
        if 'having' in q:
            score += 1
        if 'intersect' in q or 'except' in q or 'union' in q:
            score += 2
        if 'select' in q[q.find('from'):]:
            score += 2  # nested subquery
        if q.count('select') > 1:
            score += 1
        return score

    def prepare_samples(dev_data, n_samples, seed=42):
        """Return a stratified sample sorted by difficulty tier.

        Sampling strategy (fixed, backend-only — no UI controls):
            easy       25 %  — simple SELECT / WHERE
            medium     35 %  — JOINs, GROUP BY, basic aggregates
            hard       25 %  — subqueries, HAVING, multi-join
            extra hard 15 %  — INTERSECT / EXCEPT / UNION, nested subqueries

        Priority order for tier detection:
          1. Spider native ``difficulty`` field ("easy" / "medium" / "hard" / "extra")
          2. ``estimate_difficulty`` heuristic score if native field is absent

        Examples are returned in difficulty order (easy → extra hard) so the
        evaluation log naturally reflects increasing query complexity.
        """
        import random
        random.seed(seed)

        # Fixed stratification ratios
        _SPLIT = {"easy": 0.25, "medium": 0.35, "hard": 0.25, "extra": 0.15}
        _TIER_ORDER = ["easy", "medium", "hard", "extra"]

        # ── 1. Assign each example to a tier 
        def _get_tier(ex):
            native = ex.get("difficulty", "").strip().lower()
            if native == "extra hard":
                return "extra"
            if native in _SPLIT:
                return native
            # Heuristic fallback
            score = estimate_difficulty(ex)
            if score == 0:
                return "easy"
            elif score <= 2:
                return "medium"
            elif score <= 4:
                return "hard"
            else:
                return "extra"

        buckets = {t: [] for t in _TIER_ORDER}
        for ex in dev_data:
            buckets[_get_tier(ex)].append(ex)

        # Shuffle within each bucket for randomness
        for tier in _TIER_ORDER:
            random.shuffle(buckets[tier])

        # 2. Compute per-tier quotas (floor + remainder redistribution)
        quotas_f = {t: _SPLIT[t] * n_samples for t in _TIER_ORDER}
        quotas   = {t: int(v) for t, v in quotas_f.items()}
        remainder = n_samples - sum(quotas.values())
        for t in sorted(quotas_f, key=lambda x: quotas_f[x] - quotas[x], reverse=True):
            if remainder <= 0:
                break
            quotas[t] += 1
            remainder -= 1

        # 3. Sample from each bucket 
        sampled = []
        for tier in _TIER_ORDER:
            n = min(quotas[tier], len(buckets[tier]))
            sampled.extend(buckets[tier][:n])

        # Top-up if any tier was underpopulated
        if len(sampled) < n_samples:
            picked_ids = {id(s) for s in sampled}
            pool = [ex for ex in dev_data if id(ex) not in picked_ids]
            random.shuffle(pool)
            sampled.extend(pool[: n_samples - len(sampled)])

        # ── 4. Return sorted easy → extra hard (natural eval progression) ───
        tier_rank = {t: i for i, t in enumerate(_TIER_ORDER)}
        sampled.sort(key=lambda ex: tier_rank[_get_tier(ex)])
        return sampled

    # ── Start fresh evaluation ──
    if run_eval:
        # Clear any existing checkpoint
        clear_checkpoint()
        st.session_state["evaluation_paused"] = False
        st.session_state["evaluation_running"] = True

        # Initialize checkpoint data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        samples = prepare_samples(dev_data, n_samples)

        checkpoint_data = {
            "timestamp": timestamp,
            "n_samples": n_samples,
            "configs_to_run": configs_to_run,
            "samples": samples,  # Store samples to ensure consistency
            "results_store": {},
            "current_config_index": 0,
            "current_example_index": 0,
            "per_config_state": {}  # Store state for each config
        }

        # Initialize LogManager
        st.session_state["log_manager"] = LogManager(timestamp)

        # Save initial checkpoint
        save_checkpoint(checkpoint_data, timestamp)
        st.rerun()

    # ── Resume from checkpoint ──
    if resume_eval:
        st.session_state["evaluation_paused"] = False
        st.session_state["evaluation_running"] = True
        st.rerun()

    # ── Run the evaluation loop (handles both fresh and resumed) ──
    if "evaluation_running" in st.session_state and st.session_state["evaluation_running"]:
        checkpoint_data = load_checkpoint_from_session()

        if checkpoint_data is None:
            st.error("Checkpoint data lost! Please start a new evaluation.")
            st.session_state["evaluation_running"] = False
            st.stop()

        configs_to_run = checkpoint_data["configs_to_run"]
        samples = checkpoint_data["samples"]
        n_samples = checkpoint_data["n_samples"]
        results_store = checkpoint_data.get("results_store", {})
        current_config_index = checkpoint_data.get("current_config_index", 0)

        try:
            # Resume from the current configuration
            for cfg_idx in range(current_config_index, len(configs_to_run)):
                cfg = configs_to_run[cfg_idx]

                st.subheader(f"Running: {cfg}")
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Restore state for this config or initialize new
                if cfg in checkpoint_data.get("per_config_state", {}):
                    state = checkpoint_data["per_config_state"][cfg]
                    correct = state["correct"]
                    total = state["total"]
                    latencies = state["latencies"]
                    corrections = state["corrections"]
                    start_idx = state["next_example_index"]
                    st.info(
                        f"Resuming {cfg} from example {start_idx + 1}/{n_samples}")
                else:
                    correct, total, latencies, corrections = 0, 0, [], []
                    start_idx = 0

                # Run examples for this config
                for i in range(start_idx, len(samples)):
                    sample = samples[i]
                    db_id = sample["db_id"]
                    db_file = os.path.join(
                        spider_path, db_id, f"{db_id}.sqlite")

                    if not os.path.exists(db_file):
                        continue

                    status_text.text(
                        f"[{i+1}/{n_samples}] DB: {db_id} | EX so far: {correct}/{total} ({100*correct/max(total, 1):.1f}%)")
                    progress_bar.progress((i + 1) / n_samples)

                    try:
                        trace = run_pipeline(
                            nl_query=sample["question"],
                            db_path=db_file,
                            api_key=api_key,
                            top_k=top_k,
                            max_retries=max_retries,
                            config=cfg
                        )
                        is_correct = execution_accuracy(
                            trace["final_sql"], sample["query"], db_file)
                        correct += int(is_correct)
                        total += 1
                        latencies.append(trace["elapsed_ms"])
                        corrections.append(trace["correction_attempts"])

                        # Log this trace
                        if "log_manager" in st.session_state:
                            log_entry = create_trace_entry(
                                example_id=i + 1,
                                config=cfg,
                                question=sample["question"],
                                trace=trace,
                                golden_sql=sample["query"],
                                execution_match=is_correct,
                                db_id=db_id
                            )
                            st.session_state["log_manager"].add_trace(
                                cfg, log_entry)

                        time.sleep(0.1)

                    except Exception as e:
                        # Check if it's an API limit error
                        if is_api_limit_error(e):
                            st.error(
                                f"**API Limit Reached!** Error: {str(e)[:200]}")

                            # Save checkpoint with current state
                            checkpoint_data["current_config_index"] = cfg_idx
                            checkpoint_data["per_config_state"][cfg] = {
                                "correct": correct,
                                "total": total,
                                "latencies": latencies,
                                "corrections": corrections,
                                "next_example_index": i  # Resume from this example
                            }
                            checkpoint_data["results_store"] = results_store

                            save_checkpoint(checkpoint_data,
                                            checkpoint_data["timestamp"])
                            st.session_state["evaluation_paused"] = True
                            st.session_state["evaluation_running"] = False

                            st.warning(
                                "Progress saved! Replace your API key in the sidebar and click 'Resume from Checkpoint'.")
                            st.stop()
                        else:
                            # Non-API-limit error: log it but CONTINUE the batch
                            # This prevents a single malformed SQL output from halting
                            # the entire evaluation (e.g., model_c output-leakage crash).
                            total += 1
                            err_snippet = str(e)[:300]
                            st.warning(
                                f"Sample {i+1} ({cfg}) failed - skipping: `{type(e).__name__}: {err_snippet}`")
                            # Still log the failed sample so it appears in errors_only.jsonl
                            if "log_manager" in st.session_state:
                                try:
                                    failed_entry = {
                                        "example_id": i + 1,
                                        "config": cfg,
                                        "db_id": sample.get("db_id", "unknown"),
                                        "question": sample.get("question", ""),
                                        "golden_sql": sample.get("query", ""),
                                        "final_sql": "",
                                        "execution_match": False,
                                        "total_latency_ms": 0,
                                        "correction_attempts": 0,
                                        "phases": [],
                                        "error_category": "pipeline_exception",
                                        "failed_at_phase": f"Exception: {type(e).__name__}: {err_snippet}"
                                    }
                                    st.session_state["log_manager"].add_trace(
                                        cfg, failed_entry)
                                except Exception:
                                    pass  # Logging failure must never abort the loop

                    # Save checkpoint every N examples
                    if (i + 1) % CHECKPOINT_INTERVAL == 0:
                        checkpoint_data["current_config_index"] = cfg_idx
                        checkpoint_data["per_config_state"][cfg] = {
                            "correct": correct,
                            "total": total,
                            "latencies": latencies,
                            "corrections": corrections,
                            "next_example_index": i + 1
                        }
                        checkpoint_data["results_store"] = results_store
                        save_checkpoint(checkpoint_data,
                                        checkpoint_data["timestamp"])

                        # Flush logs to disk
                        if "log_manager" in st.session_state:
                            st.session_state["log_manager"].flush_to_disk(cfg)

                # Config completed
                ex = correct / max(total, 1)
                results_store[cfg] = {
                    "EX (%)": round(ex * 100, 1),
                    "Correct": correct,
                    "Total": total,
                    "Avg Latency (ms)": int(sum(latencies)/max(len(latencies), 1)),
                    "Avg Corrections": round(sum(corrections)/max(len(corrections), 1), 2)
                }
                status_text.success(
                    f"{cfg} - EX: {ex*100:.1f}% ({correct}/{total})")

                # Final flush for this config
                if "log_manager" in st.session_state:
                    st.session_state["log_manager"].flush_to_disk(cfg)

                # Update checkpoint
                checkpoint_data["results_store"] = results_store
                checkpoint_data["current_config_index"] = cfg_idx + 1
                if cfg in checkpoint_data.get("per_config_state", {}):
                    # Clear completed config state
                    del checkpoint_data["per_config_state"][cfg]
                save_checkpoint(checkpoint_data, checkpoint_data["timestamp"])

            # All configs completed
            st.session_state["eval_results"] = results_store
            st.session_state["evaluation_running"] = False

            # Generate final log exports
            if "log_manager" in st.session_state:
                log_mgr = st.session_state["log_manager"]

                # Export summary CSV and Excel
                csv_path = log_mgr.export_summary_csv(results_store)
                excel_path = log_mgr.export_detailed_excel(results_store)

                # Store paths for download in Tab 3
                st.session_state["log_csv_path"] = csv_path
                st.session_state["log_excel_path"] = excel_path
                st.session_state["log_root_dir"] = log_mgr.log_root

            clear_checkpoint()  # Clean up

            st.success("Evaluation complete!")

            # Results table
            st.divider()
            st.subheader("Results Table (reproduces Table I)")
            import pandas as pd
            df = pd.DataFrame(results_store).T.reset_index()
            df.columns = ["Configuration"] + list(df.columns[1:])
            st.dataframe(df, width='stretch')

            # Download options
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("Download Results CSV", df.to_csv(index=False),
                                   "spider_results.csv", "text/csv")

            if "log_csv_path" in st.session_state and os.path.exists(st.session_state["log_csv_path"]):
                with col2:
                    with open(st.session_state["log_csv_path"], 'r') as f:
                        st.download_button("Download Summary CSV", f.read(),
                                           "evaluation_summary.csv", "text/csv")

                if "log_excel_path" in st.session_state and st.session_state["log_excel_path"] and os.path.exists(st.session_state["log_excel_path"]):
                    with col3:
                        with open(st.session_state["log_excel_path"], 'rb') as f:
                            st.download_button("Download Detailed Excel", f.read(),
                                               "evaluation_detailed.xlsx",
                                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Unexpected error during evaluation: {str(e)}")
            st.session_state["evaluation_running"] = False
            st.exception(e)



# TAB 3 — RESULTS DASHBOARD

with tab3:
    st.header("Results Dashboard")

    # Use session state results or show paper numbers as default
    if "eval_results" in st.session_state:
        res = st.session_state["eval_results"]
    else:
        st.info("Run the benchmark evaluation first to view results.")
        res = {}

    # ── Key metrics ──
    cfg_order = ["baseline", "model_a", "model_b", "model_c"]
    labels = {"baseline": "Zero-Shot Baseline", "model_a": "Model A\n(+Linking)",
              "model_b": "Model B\n(+Guard)", "model_c": "Model C\n(Full Agentic)"}

    cols = st.columns(4)
    colors = ["#ff6b6b", "#ffa94d", "#69db7c", "#339af0"]
    for i, cfg in enumerate(cfg_order):
        if cfg in res:
            with cols[i]:
                ex = res[cfg]["EX (%)"]
                st.markdown(f"""
                <div class="metric-card" style="border-color:{colors[i]}">
                    <div style="font-size:11px;color:#666">{labels[cfg]}</div>
                    <div style="font-size:28px;font-weight:700;color:{colors[i]}">{ex}%</div>
                    <div style="font-size:11px;color:#888">Execution Accuracy</div>
                </div>""", unsafe_allow_html=True)

    # ── Bar chart ──
    st.divider()
    try:
        import plotly.graph_objects as go

        cfgs_present = [c for c in cfg_order if c in res]
        ex_vals = [res[c]["EX (%)"] for c in cfgs_present]
        lat_vals = [res[c].get("Avg Latency (ms)", 0) for c in cfgs_present]
        lab = [labels[c].replace("\n", " ") for c in cfgs_present]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=lab, y=ex_vals, name="Execution Accuracy (%)",
            marker_color=colors[:len(cfgs_present)],
            text=[f"{v}%" for v in ex_vals], textposition="outside"
        ))
        fig.update_layout(
            title="Execution Accuracy by Pipeline Configuration",
            yaxis=dict(title="EX (%)", range=[50, 100]),
            xaxis_title="Configuration",
            height=400, showlegend=False,
            plot_bgcolor="white",
            yaxis_gridcolor="#eeeeee"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Latency chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=lab, y=lat_vals, name="Avg Latency (ms)",
            marker_color="#adb5bd",
            text=[f"{v}ms" for v in lat_vals], textposition="outside"
        ))
        fig2.update_layout(
            title="Average Pipeline Latency by Configuration",
            yaxis_title="Latency (ms)", xaxis_title="Configuration",
            height=350, showlegend=False,
            plot_bgcolor="white", yaxis_gridcolor="#eeeeee"
        )
        st.plotly_chart(fig2, use_container_width=True)

    except ImportError:
        st.warning("Install plotly for charts: `pip install plotly`")



    # ── Detailed Trace Analysis Section ──
    if "log_root_dir" in st.session_state and os.path.exists(st.session_state["log_root_dir"]):
        st.divider()
        st.subheader("Detailed Trace Analysis")

        log_root = st.session_state["log_root_dir"]

        # Count errors per config
        error_counts = {}
        error_categories = {}

        for cfg in cfg_order:
            errors_file = os.path.join(log_root, cfg, "errors_only.jsonl")
            if os.path.exists(errors_file):
                count = 0
                categories = {}
                with open(errors_file, 'r') as f:
                    for line in f:
                        count += 1
                        try:
                            trace = json.loads(line)
                            cat = trace.get("error_category", "unknown")
                            categories[cat] = categories.get(cat, 0) + 1
                        except:
                            pass
                error_counts[cfg] = count
                error_categories[cfg] = categories

        # Display error summary
        if error_counts:
            st.markdown("### Error Analysis")

            cols = st.columns(len(error_counts))
            for i, (cfg, count) in enumerate(error_counts.items()):
                with cols[i]:
                    total = res.get(cfg, {}).get("Total", 0)
                    error_rate = (count / total * 100) if total > 0 else 0
                    st.metric(
                        label=cfg,
                        value=f"{count} errors",
                        delta=f"{error_rate:.1f}% error rate"
                    )

            # Error category breakdown
            st.markdown("### Error Categories")
            category_data = []
            for cfg, categories in error_categories.items():
                for cat, cnt in categories.items():
                    category_data.append({
                        "Configuration": cfg,
                        "Error Type": cat.replace("_", " ").title(),
                        "Count": cnt
                    })

            if category_data:
                import pandas as pd
                df_errors = pd.DataFrame(category_data)
                st.dataframe(df_errors, width='stretch')

        # Download options for traces
        st.divider()
        st.markdown("### Download Detailed Traces")

        col1, col2, col3 = st.columns(3)

        with col1:
            if "log_csv_path" in st.session_state and os.path.exists(st.session_state["log_csv_path"]):
                with open(st.session_state["log_csv_path"], 'r') as f:
                    st.download_button(
                        "Summary CSV",
                        f.read(),
                        "evaluation_summary.csv",
                        "text/csv",
                        help="High-level metrics per configuration"
                    )

        with col2:
            if "log_excel_path" in st.session_state and st.session_state["log_excel_path"] and os.path.exists(st.session_state["log_excel_path"]):
                with open(st.session_state["log_excel_path"], 'rb') as f:
                    st.download_button(
                        "Detailed Excel",
                        f.read(),
                        "evaluation_detailed.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Multi-sheet workbook with all traces, errors, and latency breakdown"
                    )

        with col3:
            # Offer JSONL traces as zip
            st.caption("Individual trace files available in:")
            st.code(log_root, language="bash")
        # ── Granular Query Analysis Table ──
        if "log_root_dir" in st.session_state and os.path.exists(st.session_state["log_root_dir"]):
            st.divider()
            st.subheader("Query Analysis")

            log_root = st.session_state["log_root_dir"]

            # Dynamically find available configurations
            available_configs = []
            if os.path.exists(log_root):
                for item in os.listdir(log_root):
                    item_path = os.path.join(log_root, item)
                    if os.path.isdir(item_path):
                        traces_file = os.path.join(item_path, "traces.jsonl")
                        if os.path.exists(traces_file):
                            available_configs.append(item)

            if available_configs:
                # Configuration selector
                selected_config = st.selectbox(
                    "Select Configuration to View",
                    options=available_configs,
                    help="Choose which configuration's detailed traces to display"
                )

                # Load and process traces for selected configuration
                traces_file_path = os.path.join(
                    log_root, selected_config, "traces.jsonl")

                try:
                    query_data = []
                    with open(traces_file_path, 'r') as f:
                        for line_num, line in enumerate(f, start=1):
                            try:
                                trace = json.loads(line.strip())
                                query_data.append({
                                    "Serial Number": trace.get("example_id", line_num),
                                    "Test Question": trace.get("question", "N/A"),
                                    "Generated SQL Query": trace.get("final_sql", "N/A"),
                                    "Golden Query": trace.get("golden_sql", "N/A"),
                                    "Status": "Passed" if trace.get("execution_match", False) else "Failed"
                                })
                            except json.JSONDecodeError:
                                # Skip malformed JSON lines
                                continue

                    if query_data:
                        import pandas as pd
                        df_queries = pd.DataFrame(query_data)

                        # Display the table
                        st.dataframe(
                            df_queries, width='stretch', hide_index=True)

                        # Show summary statistics
                        total = len(df_queries)
                        passed = len(
                            df_queries[df_queries["Status"] == "Passed"])
                        failed = total - passed
                        st.caption(
                            f"Total: {total} queries | Passed: {passed} ({100*passed/total:.1f}%) | Failed: {failed} ({100*failed/total:.1f}%)")
                    else:
                        st.warning(
                            f"No valid trace data found in {selected_config}")

                except FileNotFoundError:
                    st.error(f"Trace file not found: {traces_file_path}")
                except Exception as e:
                    st.error(f"Error loading trace data: {str(e)}")
            else:
                st.info(
                    "No configuration trace data available yet. Run an evaluation first.")

