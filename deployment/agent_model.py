"""MLflow models-from-code definition (Task 2.1)."""

from __future__ import annotations

import os
import sys

# Databricks Model Serving replaces sys.stdout/sys.stderr with a
# StreamToLogger object that lacks .fileno(). Several dependencies in this
# model's import chain (mcp, databricks-mcp, langchain-mcp-adapters) bind
# sys.stderr as a default argument at *import time*, so this must run before
# any other import in this file -- patching later is too late.
def _ensure_real_fileno_streams() -> None:
    def _has_fileno(stream) -> bool:
        try:
            stream.fileno()
            return True
        except Exception:
            return False

    if not _has_fileno(sys.stdout):
        sys.stdout = os.fdopen(1, "w", closefd=False)
    if not _has_fileno(sys.stderr):
        sys.stderr = os.fdopen(2, "w", closefd=False)


_ensure_real_fileno_streams()

import mlflow

_REQUIRED = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_MODEL")
_missing = [name for name in _REQUIRED if not os.environ.get(name)]
if _missing:
    raise OSError(
        f"Missing required environment variable(s): {', '.join(_missing)}. "
        "Set them in your .env (local) or the endpoint secret scope (deployed)."
    )

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm
from rag.store import get_retriever

_mcp_server = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "tools", "mcp_server.py"
)

graph = build_graph(
    llm=get_chat_llm(),
    retriever=get_retriever(),
    tools=load_mcp_tools(),
)

mlflow.models.set_model(graph)
