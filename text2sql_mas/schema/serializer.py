"""Schema serialization: converts table metadata to strings for embedding."""

import sqlite3


def serialize_table(tbl_name: str, cols: list) -> str:
    """
    Convert a table's metadata into a natural-language-friendly string
    suitable for sentence-transformer embedding.

    Example output:
        "singer: Singer_ID (INTEGER, PK), Name (TEXT), Country (TEXT), Song_Name (TEXT)"
    """
    parts = []
    for c in cols:
        p = f"{c['name']} ({c['type']}"
        if c.get("is_pk"):
            p += ", PK"
        if c.get("fk_ref"):
            p += f", FK→{c['fk_ref']}"
        p += ")"
        parts.append(p)
    return f"{tbl_name}: {', '.join(parts)}"


def load_schema_from_sqlite(db_path: str) -> dict:
    """
    Introspect a SQLite database file and return the full schema dict.

    Returns:
        {
            "table_name": {
                "columns": [
                    {"name": str, "type": str, "is_pk": bool, "fk_ref": str|None},
                    ...
                ]
            },
            ...
        }
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    schema = {}
    for tbl in tables:
        cursor.execute(f"PRAGMA table_info('{tbl}')")
        cols = []
        for row in cursor.fetchall():
            cols.append({
                "name":   row[1],
                "type":   row[2] if row[2] else "TEXT",
                "is_pk":  bool(row[5]),
                "fk_ref": None,
            })

        # Attach foreign key references
        cursor.execute(f"PRAGMA foreign_key_list('{tbl}')")
        for fk in cursor.fetchall():
            from_col  = fk[3]
            ref_table = fk[2]
            ref_col   = fk[4]
            for col in cols:
                if col["name"] == from_col:
                    col["fk_ref"] = f"{ref_table}.{ref_col}"

        schema[tbl] = {"columns": cols}

    conn.close()
    return schema