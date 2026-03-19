"""Pipeline entry point — run a single NL→SQL query through the multi-agent system."""

import argparse
import sqlite3
import sys
import json

from text2sql_mas.graph.builder import build_pipeline
from text2sql_mas.config import config


# ---------------------------------------------------------------------------
# Schema loader (from IMPLEMENTATION_DETAILS.md §5.2)
# ---------------------------------------------------------------------------

def load_schema_from_sqlite(db_path: str) -> dict:
    """
    Introspects a SQLite database to extract the full schema.
    Parses table names, column names, types, and FK relationships.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    schema = {}
    for tbl in tables:
        cursor.execute(f"PRAGMA table_info('{tbl}')")
        cols = [
            {"name": row[1], "type": row[2], "is_pk": bool(row[5])}
            for row in cursor.fetchall()
        ]
        cursor.execute(f"PRAGMA foreign_key_list('{tbl}')")
        for fk in cursor.fetchall():
            # fk[3] = from_col, fk[2] = ref_table, fk[4] = ref_col
            for col in cols:
                if col["name"] == fk[3]:
                    col["fk_ref"] = f"{fk[2]}.{fk[4]}"
        schema[tbl] = {"columns": cols}

    conn.close()
    return schema


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Text-to-SQL Pipeline"
    )
    parser.add_argument(
        "--query", type=str, required=True,
        help="Natural language question to convert to SQL"
    )
    parser.add_argument(
        "--db", type=str, required=True,
        help="Path to the SQLite database file"
    )
    parser.add_argument(
        "--max-corrections", type=int, default=config.max_correction_retries,
        help="Maximum self-correction retries (default: 3)"
    )
    args = parser.parse_args()

    # 1. Load schema from SQLite
    print(f"[1/5] Loading schema from: {args.db}")
    full_schema = load_schema_from_sqlite(args.db)
    print(f"       Found {len(full_schema)} tables")

    # 2. Build the LangGraph pipeline
    print("[2/5] Building LangGraph pipeline...")
    pipeline = build_pipeline(max_corrections=args.max_corrections)

    # 3. Prepare initial state
    initial_state = {
        "nl_query": args.query,
        "full_schema": full_schema,
        "db_path": args.db,
        "correction_attempts": 0,
        "max_corrections": args.max_corrections,
        "ast_errors": [],
        "pipeline_status": "running",
    }

    # 4. Run the pipeline
    print("[3/5] Running multi-agent pipeline...")
    print(f"       Query: {args.query}")
    final_state = pipeline.invoke(initial_state)

    # 5. Output results
    print("[4/5] Pipeline complete!")
    print(f"       Status: {final_state.get('pipeline_status', 'unknown')}")
    print(f"       Correction attempts: {final_state.get('correction_attempts', 0)}")
    print()
    print("[5/5] Generated SQL:")
    print("=" * 60)
    print(final_state.get("final_sql", final_state.get("generated_sql", "No SQL generated")))
    print("=" * 60)

    return final_state


if __name__ == "__main__":
    main()
