"""Schema serialization and pruned schema formatting for downstream agents."""


def serialize_table(table_name: str, columns: list[dict], include_types: bool = True) -> str:
    """
    Serializes a table schema to a natural-language-friendly string for embedding.

    Example output:
    "orders: order_id (INTEGER, PK), customer_id (INTEGER, FK→customers.customer_id),
     order_date (TEXT), total_amount (REAL), status (TEXT)"
    """
    col_parts = []
    for col in columns:
        part = f"{col['name']} ({col['type']}"
        if col.get('is_pk'):
            part += ", PK"
        if col.get('fk_ref'):
            part += f", FK→{col['fk_ref']}"
        part += ")"
        col_parts.append(part)

    return f"{table_name}: {', '.join(col_parts)}"


def format_pruned_schema(
    selected_tables: list[str],
    full_schema: dict
) -> str:
    """
    Formats the pruned schema subset as a structured DDL-like string.
    Includes column names, types, primary keys, and foreign key references.
    """
    lines = []
    for tbl in selected_tables:
        tbl_data = full_schema[tbl]
        lines.append(f"TABLE: {tbl}")
        for col in tbl_data['columns']:
            col_str = f"  - {col['name']} ({col['type']})"
            if col.get('is_pk'):
                col_str += " [PK]"
            if col.get('fk_ref'):
                col_str += f" [FK → {col['fk_ref']}]"
            lines.append(col_str)
        lines.append("")
    return "\n".join(lines)
