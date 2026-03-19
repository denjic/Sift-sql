"""Phase 1: Semantic Schema Linking Agent.

Retrieves the top-k most semantically relevant tables from the full database
schema using all-MiniLM-L6-v2 cosine similarity, then expands the selection
via foreign key closure to preserve join-path completeness.

FIX LOG (v2):
- retrieve_top_k_tables → retrieve_top_k       (matches embedder.py)
- apply_fk_closure      → fk_closure           (matches fk_closure.py)
- max_hops=             → hops=                (matches fk_closure.py signature)
"""

from text2sql_mas.schema.serializer import load_schema_from_sqlite
from text2sql_mas.schema.embedder import embed_schema, retrieve_top_k
from text2sql_mas.schema.fk_closure import fk_closure
from text2sql_mas.schema.formatter import format_pruned_schema
from text2sql_mas.config import config


def schema_linking_node(state: dict) -> dict:
    """
    LangGraph node: Phase 1 — Semantic Schema Linking Agent.

    Input state keys:  nl_query, full_schema
    Output state keys: pruned_schema, schema_embeddings
    """
    nl_query    = state["nl_query"]
    full_schema = state["full_schema"]

    # Embed all tables in the schema
    schema_embeddings = embed_schema(full_schema)

    # Retrieve top-k tables by cosine similarity
    top_k_tables = retrieve_top_k(
        nl_query=nl_query,
        schema_embeddings=schema_embeddings,
        k=config.schema_top_k,
    )

    # Expand via foreign key closure to include bridge tables
    closed_tables = fk_closure(
        selected=top_k_tables,
        schema=full_schema,
        hops=config.fk_closure_hops,
    )

    # Build pruned schema dict (only selected tables)
    pruned_schema = {
        tbl: full_schema[tbl]
        for tbl in closed_tables
        if tbl in full_schema
    }

    # Format as string for downstream prompt injection
    pruned_schema_str = format_pruned_schema(
        list(pruned_schema.keys()), pruned_schema
    )

    return {
        "pruned_schema":      pruned_schema,
        "schema_embeddings":  schema_embeddings,
        "pruned_schema_str":  pruned_schema_str,
    }