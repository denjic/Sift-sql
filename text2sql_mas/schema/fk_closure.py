"""Foreign key closure: expands a selected table set to include bridge tables."""


def fk_closure(selected: list, schema: dict, hops: int = 2, max_hops: int = None) -> list:
    if max_hops is not None:
        hops = max_hops
    """
    Expand the selected table list to include all tables reachable within
    `hops` foreign-key steps from any already-selected table.

    This prevents missing bridge tables (e.g. singer_in_concert) that are
    required to join two directly selected tables but aren't semantically
    close enough to the query to be retrieved by cosine similarity alone.

    Args:
        selected : list of table names chosen by cosine retrieval
        schema   : full schema dict {tbl: {"columns": [...]}}
        hops     : maximum FK hops to follow (default 2)

    Returns:
        List of table names (original selected + any bridge tables added)
    """
    closed   = set(t for t in selected if t in schema)
    frontier = set(closed)

    for _ in range(hops):
        new_frontier = set()
        for tbl in frontier:
            for col in schema.get(tbl, {}).get("columns", []):
                if col.get("fk_ref"):
                    ref_table = col["fk_ref"].split(".")[0]
                    if ref_table in schema and ref_table not in closed:
                        closed.add(ref_table)
                        new_frontier.add(ref_table)
        frontier = new_frontier
        if not frontier:
            break

    return list(closed)
# Alias for backwards compatibility with generated agent code
apply_fk_closure = fk_closure    