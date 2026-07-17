"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook (needs Spark + ai_parse_document/ai_prep_search).
Mirror PA2 Part 1:

TODO:
  - `build_chunks_table(spark, volume_path, chunks_table)`: parse the PDF with
    ai_parse_document, chunk with ai_prep_search into a Delta table with columns
    chunk_id, chunk_to_retrieve, chunk_to_embed, source, page. Enable Change Data
    Feed on the table.
  - `create_index()`: create a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index (primary_key='chunk_id',
    embedding_source_column='chunk_to_retrieve',
    embedding_model_endpoint_name=$EMBEDDINGS_ENDPOINT).
"""

from __future__ import annotations

import os


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    # Parse the PDF from a UC volume, then chunk for retrieval.
    parsed_df = spark.sql(
        f"""
        SELECT
          path AS source,
          ai_parse_document(content) AS parsed
        FROM READ_FILES('{volume_path}', format => 'binaryFile')
        """
    )
    parsed_df.createOrReplaceTempView("parsed_docs")

    chunks_df = spark.sql(
        """
        SELECT
          explode(
            ai_prep_search(
              parsed,
              map(
                'chunking_strategy', 'recursive',
                'chunk_size', 512,
                'chunk_overlap', 64
              )
            )
          ) AS chunk
        FROM parsed_docs
        """
    )
    chunks_df.createOrReplaceTempView("raw_chunks")

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {chunks_table} AS
        SELECT
          md5(cast(chunk.chunk_to_retrieve AS STRING) || cast(monotonically_increasing_id() AS STRING))
            AS chunk_id,
          chunk.chunk_to_retrieve AS chunk_to_retrieve,
          chunk.chunk_to_embed AS chunk_to_embed,
          '{volume_path}' AS source,
          coalesce(cast(chunk.metadata.page AS INT), 0) AS page
        FROM raw_chunks
        """
    )

    # Change Data Feed is required for Delta Sync Vector Search indexes.
    spark.sql(
        f"ALTER TABLE {chunks_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )


def create_index() -> None:
    from databricks.vector_search.client import VectorSearchClient

    endpoint_name = os.environ["VECTOR_SEARCH_ENDPOINT"]
    index_name = os.environ["VECTOR_SEARCH_INDEX"]
    source_table = os.environ.get(
        "SOURCE_TABLE",
        os.environ.get("UC_CATALOG", "main")
        + "."
        + os.environ.get("UC_SCHEMA", "default")
        + ".analyst_chunks",
    )
    embeddings = os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en")

    vsc = VectorSearchClient()

    try:
        vsc.get_endpoint(endpoint_name)
    except Exception:
        vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")

    try:
        vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
    except Exception:
        vsc.create_delta_sync_index(
            endpoint_name=endpoint_name,
            index_name=index_name,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_embed",
            embedding_model_endpoint_name=embeddings,
        )
