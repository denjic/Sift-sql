"""Schema embedding and cosine-similarity retrieval using all-MiniLM-L6-v2."""

import numpy as np
from sentence_transformers import SentenceTransformer
from text2sql_mas.schema.serializer import serialize_table

_MODEL = None


def get_model() -> SentenceTransformer:
    """Singleton loader — model is loaded once and reused."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed_schema(full_schema: dict) -> dict:
    """
    Encode every table in full_schema into a 384-dim embedding vector.

    Returns:
        {table_name: np.ndarray of shape (384,)}
    """
    model = get_model()
    names = list(full_schema.keys())
    texts = [serialize_table(t, full_schema[t]["columns"]) for t in names]
    vecs  = model.encode(texts, normalize_embeddings=True)
    return {name: vecs[i] for i, name in enumerate(names)}


def retrieve_top_k(
    nl_query: str,
    schema_embeddings: dict,
    k: int = 5,
) -> list:
    """
    Return the names of the top-k most semantically similar tables
    to the natural language query, ranked by cosine similarity.
    """
    model  = get_model()
    q_vec  = model.encode([nl_query], normalize_embeddings=True)[0]
    scores = {
        tbl: float(np.dot(q_vec, emb))
        for tbl, emb in schema_embeddings.items()
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [tbl for tbl, _ in ranked[:k]]
# Alias for backwards compatibility with generated agent code
retrieve_top_k_tables = retrieve_top_k