"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

TODO: Implement `get_retriever(k=4)` that returns a LangChain retriever over the
Databricks Vector Search index built by `ingest.py`, using
`DatabricksVectorSearch` from `databricks_langchain`. Read endpoint/index names
from config.get_settings(). This exact retriever is reused by the deployed model.
"""

from __future__ import annotations

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
# Only request columns that exist on the Delta Sync index. `page` is optional —
# if your ingest notebook did not write a page column, asking for it here breaks
# retrieval. format_docs() already falls back to p.? when metadata lacks page.
CITATION_COLUMNS = ["chunk_id", "source"]


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
        text_column=TEXT_COLUMN,
        columns=CITATION_COLUMNS,
    )


def get_retriever(k: int = 4):
    return get_vector_store().as_retriever(search_kwargs={"k": k})
