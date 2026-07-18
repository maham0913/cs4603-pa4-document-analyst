"""MLflow models-from-code definition (Task 2.1).

TODO: Make this file self-contained so MLflow can serialise it:
  - validate DATABRICKS_HOST/TOKEN/MODEL at import time (clear error if missing),
  - rebuild the graph with production clients (LLM, Vector Search retriever,
    MCP tools),
  - end with `mlflow.models.set_model(graph)`.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations

import os

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
