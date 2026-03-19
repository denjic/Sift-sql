"""Phase 4A: AST Schema Guard — pre-execution structural validation.

Parses generated SQL into an AST via sqlglot and verifies every referenced
entity (table, column) exists in the pruned schema before any DB interaction.
"""

from difflib import get_close_matches

import sqlglot
from sqlglot import exp


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def full_ast_validation(
    sql: str,
    pruned_schema: dict
) -> tuple[bool, list[dict]]:
    """
    Performs two-pass AST validation:
    Pass 1: Syntax parsing
    Pass 2: Schema entity membership + FK relationship validation

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
    valid_columns: dict[tuple, bool] = {}  # (table, col) -> True
    for tbl, data in valid_tables.items():
        for col in data['columns']:
            valid_columns[(tbl, col['name'].lower())] = True
            valid_columns[(None, col['name'].lower())] = True  # Unqualified

    fk_relations = set()  # (from_table.from_col, to_table.to_col)
    for tbl, data in valid_tables.items():
        for col in data['columns']:
            if col.get('fk_ref'):
                ref_parts = col['fk_ref'].lower().split('.')
                fk_relations.add((tbl, col['name'].lower(), ref_parts[0], ref_parts[1]))

    # --- Pass 2A: Table entity validation ---
    for tbl_node in ast.find_all(exp.Table):
        tbl_name = tbl_node.name.lower()
        if tbl_name and tbl_name not in valid_tables:
            suggestions = get_close_matches(tbl_name, valid_tables.keys(), n=1, cutoff=0.5)
            errors.append({
                "level": 2,
                "type": "NONEXISTENT_TABLE",
                "entity": tbl_name,
                "suggestion": suggestions[0] if suggestions else None,
                "detail": f"Table '{tbl_name}' not found in schema."
            })

    # --- Pass 2B: Column entity validation ---
    for col_node in ast.find_all(exp.Column):
        col_name = col_node.name.lower()
        tbl_name = col_node.table.lower() if col_node.table else None

        key_qualified = (tbl_name, col_name) if tbl_name else None
        key_unqualified = (None, col_name)

        if key_qualified and key_qualified not in valid_columns:
            table_cols = [c['name'].lower() for c in valid_tables.get(tbl_name, {}).get('columns', [])]
            suggestions = get_close_matches(col_name, table_cols, n=1, cutoff=0.5)
            errors.append({
                "level": 2,
                "type": "NONEXISTENT_COLUMN",
                "entity": f"{tbl_name}.{col_name}",
                "suggestion": f"{tbl_name}.{suggestions[0]}" if suggestions else None,
                "detail": f"Column '{col_name}' not found in table '{tbl_name}'."
            })
        elif not tbl_name and key_unqualified not in valid_columns:
            errors.append({
                "level": 2,
                "type": "NONEXISTENT_COLUMN_UNQUALIFIED",
                "entity": col_name,
                "suggestion": None,
                "detail": f"Unqualified column '{col_name}' not found in any selected table."
            })

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Error formatting for the correction agent
# ---------------------------------------------------------------------------

def format_errors_for_correction(errors: list[dict]) -> str:
    """Formats structured AST errors into a human-readable correction directive."""
    lines = ["The following schema violations were detected by static analysis:\n"]
    for i, err in enumerate(errors, 1):
        line = f"{i}. [{err['type']}] {err['detail']}"
        if err.get('suggestion'):
            line += f"\n   → Suggested correction: use '{err['suggestion']}' instead."
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def ast_guard_node(state: dict) -> dict:
    """
    LangGraph node: Phase 4A — AST Schema Guard.

    Input state keys: generated_sql, pruned_schema
    Output state keys: ast_validation_passed, ast_errors, final_sql, pipeline_status
    """
    generated_sql = state["generated_sql"]
    pruned_schema = state["pruned_schema"]

    is_valid, errors = full_ast_validation(generated_sql, pruned_schema)

    result = {
        "ast_validation_passed": is_valid,
        "ast_errors": errors,
    }

    if is_valid:
        result["final_sql"] = generated_sql
        result["pipeline_status"] = "success"

    return result
