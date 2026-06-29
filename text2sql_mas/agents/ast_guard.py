import re
import sqlite3
from difflib import get_close_matches
 
import sqlglot
from sqlglot import exp
 
 
# Core validation
 
def full_ast_validation(
    sql: str,
    pruned_schema: dict
) -> tuple[bool, list[dict]]:
    """
    Performs two-pass AST validation:
    Pass 1: Syntax parsing
    Pass 2: Schema entity membership validation with alias resolution
 
    Returns:
        (is_valid: bool, errors: list of structured error dicts)
    """
    errors = []
 
    # --- Pass 1: Syntax ---
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError as e:
        return False, [{"level": 1, "type": "SYNTAX_ERROR", "detail": str(e), "suggestion": None}]
 
    # Build schema lookup structures
    valid_tables = {t.lower(): data for t, data in pruned_schema.items()}
    valid_columns: dict[tuple, bool] = {}
    for tbl, data in valid_tables.items():
        for col in data['columns']:
            valid_columns[(tbl, col['name'].lower())] = True
            valid_columns[(None, col['name'].lower())] = True  # Unqualified
 
    # --- Build alias map: alias → real table name ---
    # e.g. "singer AS T1" → {"t1": "singer"}
    alias_map: dict[str, str] = {}
    for tbl_node in ast.find_all(exp.Table):
        real_name = tbl_node.name.lower()
        alias = tbl_node.alias.lower() if tbl_node.alias else None
        if alias:
            alias_map[alias] = real_name
 
    # --- Pass 2A: Table entity validation ---
    # Only validate real table names, not aliases
    for tbl_node in ast.find_all(exp.Table):
        tbl_name = tbl_node.name.lower()
        # Skip if this is actually an alias reference
        if tbl_name in alias_map.values():
            pass  # real table name — validate it
        if tbl_name and tbl_name not in valid_tables:
            # Don't flag aliases as missing tables
            if tbl_name in alias_map:
                continue
            suggestions = get_close_matches(tbl_name, valid_tables.keys(), n=1, cutoff=0.5)
            errors.append({
                "level": 2,
                "type": "NONEXISTENT_TABLE",
                "entity": tbl_name,
                "suggestion": suggestions[0] if suggestions else None,
                "detail": f"Table '{tbl_name}' not found in schema."
            })
 
    # --- Pass 2B: Column entity validation with alias resolution ---
    for col_node in ast.find_all(exp.Column):
        col_name = col_node.name.lower()
        tbl_ref  = col_node.table.lower() if col_node.table else None
 
        # Resolve alias to real table name
        resolved_tbl = alias_map.get(tbl_ref, tbl_ref) if tbl_ref else None
 
        # Skip validation for short aliases not in alias_map and not in schema
        # Pattern: single letter (a,b,s) or T+digit (T1,T2,T3)
        if tbl_ref and tbl_ref not in alias_map and tbl_ref not in valid_tables:
            if len(tbl_ref) <= 2 or re.match(r'^t\d+$', tbl_ref):
                continue  # looks like an unresolved alias — skip silently
 
        key_qualified   = (resolved_tbl, col_name) if resolved_tbl else None
        key_unqualified = (None, col_name)
 
        if key_qualified and key_qualified not in valid_columns:
            table_cols = [
                c['name'].lower()
                for c in valid_tables.get(resolved_tbl, {}).get('columns', [])
            ]
            suggestions = get_close_matches(col_name, table_cols, n=1, cutoff=0.5)
            errors.append({
                "level": 2,
                "type": "NONEXISTENT_COLUMN",
                "entity": f"{resolved_tbl}.{col_name}",
                "suggestion": f"{resolved_tbl}.{suggestions[0]}" if suggestions else None,
                "detail": f"Column '{col_name}' not found in table '{resolved_tbl}'."
            })
        elif not tbl_ref and key_unqualified not in valid_columns:
            errors.append({
                "level": 2,
                "type": "NONEXISTENT_COLUMN_UNQUALIFIED",
                "entity": col_name,
                "suggestion": None,
                "detail": f"Unqualified column '{col_name}' not found in any selected table."
            })
 
    return len(errors) == 0, errors
 
 

# Error formatting for the correction agent
 
def format_errors_for_correction(errors: list[dict]) -> str:
    """Formats structured AST errors into a human-readable correction directive."""
    lines = ["The following schema violations were detected by static analysis:\n"]
    for i, err in enumerate(errors, 1):
        line = f"{i}. [{err['type']}] {err['detail']}"
        if err.get('suggestion'):
            line += f"\n   → Suggested correction: use '{err['suggestion']}' instead."
        lines.append(line)
    return "\n".join(lines)
 
 
# Execution Dry-Run (Step B)


def phase4_execute_validate(sql: str, db_path: str) -> tuple[bool, str]:
    """Execute SQL against SQLite db_path to catch runtime errors that AST misses."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.close()
        return True, ""
    except Exception as e:
        return False, str(e)



# LangGraph node

 
def ast_guard_node(state: dict) -> dict:
    """
    LangGraph node: Phase 4A — AST Schema Guard + Execution Dry Run.
 
    Input state keys:  generated_sql, pruned_schema, db_path
    Output state keys: ast_validation_passed, ast_errors, final_sql, pipeline_status
    """
    generated_sql = state["generated_sql"]
    pruned_schema = state["pruned_schema"]
    db_path = state["db_path"]
 
    # Step A: Static AST Check
    is_valid, errors = full_ast_validation(generated_sql, pruned_schema)
    
    # Step B: Execution Dry Run
    if is_valid:
        exec_ok, exec_err = phase4_execute_validate(generated_sql, db_path)
        if not exec_ok:
            is_valid = False
            errors.append({
                "level": 3,
                "type": "EXECUTION_ERROR",
                "detail": exec_err,
                "suggestion": None
            })
 
    result = {
        "ast_validation_passed": is_valid,
        "ast_errors":            errors,
    }
 
    if is_valid:
        result["final_sql"]       = generated_sql
        result["pipeline_status"] = "success"
 
    return result
 