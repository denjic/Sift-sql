"""Formats a pruned schema subset into a structured string for LLM prompts."""


def format_pruned_schema(selected_tables: list, schema: dict) -> str:
    """
    Format a subset of the schema as a structured DDL-like string for
    injection into planning and generation prompts.

    Example output:
        TABLE: singer
          - Singer_ID (INTEGER) [PK]
          - Name (TEXT)
          - Country (TEXT)

        TABLE: singer_in_concert
          - concert_ID (INTEGER) [FK → concert.concert_ID]
          - Singer_ID (INTEGER) [FK → singer.Singer_ID]

    Args:
        selected_tables : ordered list of table names to include
        schema          : full or pruned schema dict

    Returns:
        Multi-line string ready for prompt injection
    """
    lines = []
    for tbl in selected_tables:
        if tbl not in schema:
            continue
        lines.append(f"TABLE: {tbl}")
        for col in schema[tbl]["columns"]:
            s = f"  - {col['name']} ({col['type']})"
            if col.get("is_pk"):
                s += " [PK]"
            if col.get("fk_ref"):
                s += f" [FK → {col['fk_ref']}]"
            lines.append(s)
        lines.append("")          # blank line between tables
    return "\n".join(lines).strip()