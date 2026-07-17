"""Vector Search retriever factory (Task 1.4 support / rag/store.py)."""

from __future__ import annotations

from config import get_settings

# NOTE: "page" is NOT in CITATION_COLUMNS because the live index
# (built via the notebook) does not have a page column yet.
CITATION_COLUMNS = ["chunk_id", "source", "chunk_to_retrieve"]


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
    )


def get_retriever(k: int = 4):
    return get_vector_store().as_retriever(search_kwargs={"k": k})
