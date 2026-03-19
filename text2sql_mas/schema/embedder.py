"""MiniLM embedding and cosine similarity retrieval for semantic schema linking."""

from sentence_transformers import SentenceTransformer
import numpy as np

from text2sql_mas.schema.formatter import serialize_table

# Model is loaded once at import time and reused
model = SentenceTransformer('all-MiniLM-L6-v2')


def embed_schema(full_schema: dict) -> dict[str, np.ndarray]:
    """
    Returns a mapping from table_name → embedding vector.
    Embeddings are computed once and cached.
    """
    serialized = {
        tbl_name: serialize_table(tbl_name, tbl_data['columns'])
        for tbl_name, tbl_data in full_schema.items()
    }
    embeddings = model.encode(list(serialized.values()), normalize_embeddings=True)
    return {name: emb for name, emb in zip(serialized.keys(), embeddings)}


def retrieve_top_k_tables(
    nl_query: str,
    schema_embeddings: dict[str, np.ndarray],
    k: int = 5
) -> list[str]:
    """
    Returns names of the top-k most semantically relevant tables for the query.
    """
    query_embedding = model.encode([nl_query], normalize_embeddings=True)[0]
    scores = {
        tbl: float(np.dot(query_embedding, emb))
        for tbl, emb in schema_embeddings.items()
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [tbl for tbl, _ in ranked[:k]]
