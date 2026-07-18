"""Full Document Analyst graph (Tasks 1.5 + 1.7).

TODO:
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio
    (see langchain-mcp-adapters) and return its tools.
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM
    call exactly one MCP tool, then append the result and increment the index.
  - `build_graph(llm=None, retriever=None, tools=None)`: assemble
    planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.
    Inject dependencies so the graph can be unit-tested offline with fakes.
"""

from __future__ import annotations

import asyncio
import os
import sys

from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


def load_mcp_tools(server_path: str | None = None):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    if server_path is None:
        import tools.mcp_server as _mcp_module
        server_path = _mcp_module.__file__

    mcp_url = os.environ.get("MCP_SERVER_URL")
    token = os.environ.get("DATABRICKS_TOKEN", "")

    if mcp_url:
        client = MultiServerMCPClient(
            {
                "analyst": {
                    "url": f"{mcp_url.rstrip('/')}/mcp",
                    "transport": "streamable_http",
                    "headers": {"Authorization": f"Bearer {token}"},
                }
            }
        )
    else:
        client = MultiServerMCPClient(
            {
                "analyst": {
                    "command": sys.executable,
                    "args": [server_path],
                    "transport": "stdio",
                }
            }
        )

    return asyncio.run(_get_tools_with_real_stdio(client))


async def _get_tools_with_real_stdio(client):
    """Temporarily swap sys.stdout/stderr for real files with .fileno()
    while the stdio MCP subprocess is spawned. Databricks Model Serving
    replaces sys.stdout/stderr with a StreamToLogger object that lacks
    .fileno(), which the stdio subprocess machinery requires.
    """
    import os as _os

    real_stdout, real_stderr = sys.stdout, sys.stderr
    devnull_out = open(_os.devnull, "w")
    devnull_err = open(_os.devnull, "w")
    try:
        if not hasattr(sys.stdout, "fileno") or not _has_working_fileno(sys.stdout):
            sys.stdout = devnull_out
        if not hasattr(sys.stderr, "fileno") or not _has_working_fileno(sys.stderr):
            sys.stderr = devnull_err
        return await client.get_tools()
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr
        devnull_out.close()
        devnull_err.close()


def _has_working_fileno(stream) -> bool:
    try:
        stream.fileno()
        return True
    except Exception:
        return False