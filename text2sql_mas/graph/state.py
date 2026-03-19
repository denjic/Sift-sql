"""SQLAgentState TypedDict — the shared state schema for the LangGraph pipeline.

FIX LOG (v2):
- ast_errors changed from List[str] to List[dict] — the AST guard returns
  structured dicts {level, type, detail, suggestion}, not plain strings.
  Having the wrong type here caused silent KeyError crashes in self_corrector
  when it called format_errors_for_correction(ast_errors) on a list of strings.
- logical_blueprint made Optional[str] with default None so baseline and
  model_a configs (which skip Phase 2) do not crash when self_corrector reads
  state["logical_blueprint"].
- pipeline_status given a sensible note — callers should initialise it to
  "running" so any crash mid-pipeline is distinguishable from a clean "failed".
"""

from typing import TypedDict, List, Optional, Dict, Any


class SQLAgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    nl_query:     str           # Original natural language question
    full_schema:  dict          # Full database schema (tables, columns, types, FKs)
    db_path:      str           # SQLite path or PostgreSQL connection string

    # ── Phase 1 output ────────────────────────────────────────────────────────
    pruned_schema:      dict               # Top-k relevant tables and their columns
    schema_embeddings:  List[List[float]]  # Cached MiniLM embeddings for tables

    # ── Phase 2 output ────────────────────────────────────────────────────────
    # Optional — baseline and model_a skip Phase 2; must not crash self_corrector
    logical_blueprint:  Optional[str]      # CoT plan: tables, joins, filters, aggs

    # ── Phase 3 output ────────────────────────────────────────────────────────
    generated_sql:  str          # Raw SQL string from generation agent

    # ── Phase 4 state ─────────────────────────────────────────────────────────
    ast_validation_passed:  bool                # True if AST guard found no violations
    ast_errors:             List[Dict[str, Any]] # Structured error dicts from AST guard
                                                 # {level, type, detail, suggestion}
    correction_attempts:    int                  # Number of self-correction retries
    max_corrections:        int                  # Maximum retries (configurable, default 3)

    # ── Terminal outputs ───────────────────────────────────────────────────────
    final_sql:        Optional[str]  # Final SQL (set by router on validated/exhausted)
    pipeline_status:  str            # "running" → "success" | "correction_loop" | "failed"