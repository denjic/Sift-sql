"""Foreign key closure algorithm — BFS over FK edges to ensure join-path completeness."""


def apply_fk_closure(
    selected_tables: list[str],
    full_schema: dict,
    max_hops: int = 2
) -> list[str]:
    """
    Expands the selected table set to include all tables reachable
    within max_hops foreign key steps from any selected table.
    """
    closed = set(selected_tables)
    frontier = set(selected_tables)

    for _ in range(max_hops):
        new_frontier = set()
        for tbl in frontier:
            if tbl not in full_schema:
                continue
            for col in full_schema[tbl]['columns']:
                if col.get('fk_ref'):
                    ref_table = col['fk_ref'].split('.')[0]
                    if ref_table not in closed:
                        closed.add(ref_table)
                        new_frontier.add(ref_table)
        frontier = new_frontier
        if not frontier:
            break

    return list(closed)
