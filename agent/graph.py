"""Full Document Analyst graph (Tasks 1.5 + 1.7)."""

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


def _run_async(coro):
    """Run a coroutine safely whether or not an event loop is already
    running (e.g. inside Jupyter/ipykernel)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


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

    # Jupyter/ipykernel already runs an event loop, so asyncio.run() would
    # raise "cannot be called from a running event loop". _run_async handles
    # both the script and notebook cases.
    return _run_async(client.get_tools())


def _call_tool_sync(tool, args):
    """Call a tool with whatever invocation style it supports.

    Real MCP-derived StructuredTools only implement async invocation
    (ainvoke) and raise NotImplementedError on sync .invoke(). Test
    doubles / plain LangChain tools may only implement sync invoke.
    Try async first when available, fall back to sync otherwise.
    """
    if hasattr(tool, "ainvoke"):
        try:
            return _run_async(tool.ainvoke(args))
        except NotImplementedError:
            pass
    return tool.invoke(args)


def make_mcp_node(tools, llm):
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def mcp_tools(state: AnalystState) -> dict:
        step = state["plan"][state["current_step_index"]]
        prior = "\n".join(state.get("step_results") or [])
        ai_msg = llm_with_tools.invoke(
            [
                {"role": "system", "content": MCP_STEP_PROMPT},
                {
                    "role": "user",
                    "content": f"Step: {step}\nPrior results:\n{prior}",
                },
            ]
        )

        result_text = ""
        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                name = tc["name"] if isinstance(tc, dict) else tc.get("name")
                args = tc["args"] if isinstance(tc, dict) else tc.get("args", {})
                tool = tool_map[name]
                out = _call_tool_sync(tool, args)
                result_text += str(out) + "\n"
        else:
            result_text = (ai_msg.content or "").strip() or "No tool was called"

        step_results = list(state.get("step_results") or [])
        step_results.append(result_text.strip())
        return {
            "step_results": step_results,
            "current_step_index": state["current_step_index"] + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    if llm is None:
        from config import get_chat_llm

        llm = get_chat_llm()
    if retriever is None:
        from rag.store import get_retriever

        retriever = get_retriever()
    if tools is None:
        tools = load_mcp_tools()

    builder = StateGraph(AnalystState)
    builder.add_node("planner", make_planner(llm))
    builder.add_node("supervisor", make_supervisor(llm))
    builder.add_node("rag_agent", make_rag_agent(retriever, llm))
    builder.add_node("mcp_tools", make_mcp_node(tools, llm))
    builder.add_node("synthesizer", make_synthesizer(llm))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {RAG: "rag_agent", MCP: "mcp_tools", SYNTH: "synthesizer"},
    )
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("mcp_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    return builder.compile()