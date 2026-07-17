"""Vector Search retriever factory (Task 1.4 support / rag/store.py)."""

from __future__ import annotations

from config import get_settings

# Only request columns that exist on the Delta Sync index.
CITATION_COLUMNS = ["chunk_id", "source", "page", "chunk_to_retrieve"]


def get_vector_store():
    from databricks_langchain import DatabricksVectorSearch

    s = get_settings()
    if not s["vs_endpoint"] or not s["vs_index"]:
        raise OSError(
            "VECTOR_SEARCH_ENDPOINT and VECTOR_SEARCH_INDEX must be set "
            "(local .env or endpoint environment_vars)."
        )
    return DatabricksVectorSearch(
        index_name=s["vs_index"],
        endpoint=s["vs_endpoint"],
        columns=CITATION_COLUMNS,
        # no text_column= — the index's embedding_source_column (chunk_to_embed)
        # is auto-detected; passing text_column explicitly conflicts with it.
    )


def get_retriever(k: int = 4):
    return get_vector_store().as_retriever(search_kwargs={"k": k})