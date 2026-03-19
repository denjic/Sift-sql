"""Phase 1: Semantic Schema Linking Agent node.

Uses MiniLM cosine retrieval to select top-k relevant tables,
applies FK closure, and formats the pruned schema for downstream agents.
"""

from text2sql_mas.schema.embedder import embed_schema, retrieve_top_k_tables
from text2sql_mas.schema.fk_closure import apply_fk_closure
from text2sql_mas.schema.formatter import format_pruned_schema
from text2sql_mas.config import config


def schema_linking_node(state: dict) -> dict:
    """
    LangGraph node: Phase 1 — Semantic Schema Linking.

    Input state keys: nl_query, full_schema
    Output state keys: pruned_schema, schema_embeddings
    """
    full_schema = state["full_schema"]
    nl_query = state["nl_query"]

    # Step 1: Embed all tables (cached per schema)
    schema_embeddings = embed_schema(full_schema)

    # Step 2: Retrieve top-k tables by cosine similarity
    top_k_tables = retrieve_top_k_tables(
        nl_query, schema_embeddings, k=config.schema_top_k
    )

    # Step 3: Apply foreign key closure to ensure join-path completeness
    closed_tables = apply_fk_closure(
        top_k_tables, full_schema, max_hops=config.fk_closure_hops
    )

    # Step 4: Build pruned schema dict (subset of full_schema)
    pruned_schema = {
        tbl: full_schema[tbl]
        for tbl in closed_tables
        if tbl in full_schema
    }

    # Step 5: Format pruned schema as structured string for prompts
    pruned_schema_str = format_pruned_schema(closed_tables, full_schema)

    # Convert embeddings to serializable lists
    embeddings_list = [
        emb.tolist() for emb in schema_embeddings.values()
    ]

    return {
        "pruned_schema": pruned_schema,
        "schema_embeddings": embeddings_list,
    }
