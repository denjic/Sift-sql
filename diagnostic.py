"""
diagnostic.py — run this from your workspace root to verify every module
is importable and the full pipeline produces a non-None final_sql.

Usage:
    python diagnostic.py

Expected output (all PASS):
    [1/6] schema.serializer        PASS
    [2/6] schema.embedder          PASS
    [3/6] schema.fk_closure        PASS
    [4/6] schema.formatter         PASS
    [5/6] graph.router final_sql   PASS
    [6/6] full pipeline end-to-end PASS
    All checks passed. Re-run the benchmark now.
"""

import sys, os

# ── make sure we can find the package ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONCERT_DB = "./databases/concert_singer/concert_singer.sqlite"

results = []

# ── Check 1: serializer ───────────────────────────────────────────────────────
try:
    from text2sql_mas.schema.serializer import load_schema_from_sqlite, serialize_table
    schema = load_schema_from_sqlite(CONCERT_DB)
    assert len(schema) > 0, "Schema is empty"
    first_tbl = list(schema.keys())[0]
    s = serialize_table(first_tbl, schema[first_tbl]["columns"])
    assert isinstance(s, str) and len(s) > 0
    results.append(("[1/6] schema.serializer       ", True, None))
except Exception as e:
    results.append(("[1/6] schema.serializer       ", False, e))

# ── Check 2: embedder ─────────────────────────────────────────────────────────
try:
    from text2sql_mas.schema.embedder import embed_schema, retrieve_top_k
    embs = embed_schema(schema)
    assert len(embs) == len(schema)
    top = retrieve_top_k("how many singers are there", embs, k=3)
    assert isinstance(top, list) and len(top) == 3
    results.append(("[2/6] schema.embedder         ", True, None))
except Exception as e:
    results.append(("[2/6] schema.embedder         ", False, e))

# ── Check 3: fk_closure ───────────────────────────────────────────────────────
try:
    from text2sql_mas.schema.fk_closure import fk_closure
    closed = fk_closure(["singer"], schema, hops=2)
    assert isinstance(closed, list) and len(closed) >= 1
    results.append(("[3/6] schema.fk_closure       ", True, None))
except Exception as e:
    results.append(("[3/6] schema.fk_closure       ", False, e))

# ── Check 4: formatter ────────────────────────────────────────────────────────
try:
    from text2sql_mas.schema.formatter import format_pruned_schema
    pruned = {k: schema[k] for k in list(schema.keys())[:2]}
    fmt    = format_pruned_schema(list(pruned.keys()), pruned)
    assert "TABLE:" in fmt
    results.append(("[4/6] schema.formatter        ", True, None))
except Exception as e:
    results.append(("[4/6] schema.formatter        ", False, e))

# ── Check 5: router writes final_sql ─────────────────────────────────────────
try:
    from text2sql_mas.graph.router import route_after_guard

    # Simulate a passing state
    state_pass = {
        "ast_validation_passed": True,
        "generated_sql": "SELECT COUNT(*) FROM singer",
        "correction_attempts": 0,
        "max_corrections": 3,
        "final_sql": None,
        "pipeline_status": "running",
    }
    route = route_after_guard(state_pass)
    assert route == "validated", f"Expected 'validated', got '{route}'"
    assert state_pass["final_sql"] is not None, \
        "BUG: router did not write final_sql on validated path"
    assert state_pass["pipeline_status"] == "success"
    results.append(("[5/6] router final_sql check  ", True, None))
except Exception as e:
    results.append(("[5/6] router final_sql check  ", False, e))

# ── Check 6: full pipeline end-to-end ────────────────────────────────────────
try:
    from text2sql_mas.graph.builder import build_pipeline

    pipeline = build_pipeline()
    state = pipeline.invoke({
        "nl_query":            "How many singers are there?",
        "full_schema":         schema,
        "db_path":             CONCERT_DB,
        "correction_attempts": 0,
        "max_corrections":     3,
        "pipeline_status":     "running",
    })

    final_sql = state.get("final_sql")
    assert final_sql is not None,    "final_sql is None — router bug still present"
    assert len(final_sql.strip()) > 0, "final_sql is empty string"
    assert "SELECT" in final_sql.upper(), f"final_sql doesn't look like SQL: {final_sql}"

    results.append(("[6/6] full pipeline e2e       ", True, None))
except Exception as e:
    results.append(("[6/6] full pipeline e2e       ", False, e))

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
all_pass = True
for label, passed, err in results:
    status = "PASS" if passed else "FAIL"
    print(f"  {label}  {status}")
    if not passed:
        print(f"         └─ {err}")
        all_pass = False

print("="*55)
if all_pass:
    print("\n  All checks passed. Re-run the benchmark now.")
    print("  Expected: Model C → 65-72% EX\n")
else:
    print("\n  Fix the FAIL items above before running the benchmark.\n")