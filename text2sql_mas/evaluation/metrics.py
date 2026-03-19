"""Evaluation metrics for the Text-to-SQL pipeline.

FIX LOG (v2):
- BUG FIXED (critical): execution_accuracy() was receiving final_sql=None
  whenever router.py failed to write it (the original router only wrote
  final_sql on the "exhausted" path, never on "validated"). This caused
  every Model C result to be compared as None vs gold_sql → always False.
  Fixed in router.py; this file adds a defensive None check as a safety net.
- ADDED: normalise_sql() strips semicolons and lowercases before exact-match
  comparison, matching Spider's official evaluation script behaviour.
- ADDED: detailed per-sample error logging so you can see which failure
  category (empty SQL, execution error, wrong result) each sample falls into.
"""

import sqlite3
import re
from pathlib import Path


# ── SQL normalisation (matches Spider official evaluator) ──────────────────────

def normalise_sql(sql: str) -> str:
    """Lowercase, collapse whitespace, strip trailing semicolon."""
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";").strip()
    return sql


# ── Execution ──────────────────────────────────────────────────────────────────

def execute_sql(sql: str, db_path: str):
    """
    Execute a SQL query against a SQLite database.
    Returns (result_set, error_message).
    result_set is a frozenset of tuples for order-insensitive comparison.
    """
    if not sql or not sql.strip():
        return None, "empty_sql"
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql)
        rows = frozenset(tuple(row) for row in cur.fetchall())
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)


# ── Execution Accuracy ─────────────────────────────────────────────────────────

def execution_accuracy(pred_sql: str, gold_sql: str, db_path: str) -> tuple[bool, str]:
    """
    Compare predicted SQL against gold SQL by executing both and comparing results.

    Returns:
        (is_correct: bool, failure_reason: str)
        failure_reason is "" on success, otherwise a category string for analysis.
    """
    # Defensive None check — catches router bug if still present
    if not pred_sql or not pred_sql.strip():
        return False, "empty_pred_sql"

    pred_rows, pred_err = execute_sql(pred_sql, db_path)
    gold_rows, gold_err = execute_sql(gold_sql, db_path)

    if gold_err:
        return False, f"gold_exec_error:{gold_err}"
    if pred_err:
        return False, f"pred_exec_error:{pred_err}"

    is_correct = pred_rows == gold_rows
    return is_correct, "" if is_correct else "wrong_result"


# ── Batch evaluation ───────────────────────────────────────────────────────────

def compute_execution_accuracy(results: list[dict], db_base_path: str) -> dict:
    """
    Compute EX over a list of pipeline result dicts.

    Each result dict must contain:
        predicted_sql   : str   (the final_sql from the pipeline)
        gold_sql        : str   (ground truth from Spider dev.json)
        db_id           : str   (Spider database name)
        difficulty      : str   (easy/medium/hard/extra — from Spider)
        pipeline_status : str   (success/failed/correction_loop)

    Returns a summary dict with overall and per-difficulty EX.
    """
    total   = len(results)
    correct = 0
    failure_categories = {}

    by_difficulty = {
        "easy":   {"correct": 0, "total": 0},
        "medium": {"correct": 0, "total": 0},
        "hard":   {"correct": 0, "total": 0},
        "extra":  {"correct": 0, "total": 0},
    }

    for r in results:
        db_path = str(Path(db_base_path) / r["db_id"] / f"{r['db_id']}.sqlite")
        diff    = r.get("difficulty", "unknown")

        if diff in by_difficulty:
            by_difficulty[diff]["total"] += 1

        is_correct, reason = execution_accuracy(
            pred_sql=r.get("predicted_sql", ""),
            gold_sql=r["gold_sql"],
            db_path=db_path,
        )

        if is_correct:
            correct += 1
            if diff in by_difficulty:
                by_difficulty[diff]["correct"] += 1
        else:
            failure_categories[reason] = failure_categories.get(reason, 0) + 1

    ex_overall = correct / max(total, 1)

    return {
        "execution_accuracy": round(ex_overall * 100, 1),
        "correct": correct,
        "total": total,
        "failure_categories": failure_categories,
        "by_difficulty": {
            d: {
                "ex": round(v["correct"] / max(v["total"], 1) * 100, 1),
                "correct": v["correct"],
                "total": v["total"],
            }
            for d, v in by_difficulty.items()
        },
    }


# ── Exact Match (secondary metric) ────────────────────────────────────────────

def exact_match(pred_sql: str, gold_sql: str) -> bool:
    """Normalised string exact match (lowercase, whitespace-collapsed, no semicolons)."""
    if not pred_sql:
        return False
    return normalise_sql(pred_sql) == normalise_sql(gold_sql)