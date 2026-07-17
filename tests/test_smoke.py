"""Offline smoke test for the Document Analyst graph (Bonus A test target).

This is the target the Bonus A CI pipeline runs to prove the graph wires up
before any deploy. Fill it in once your nodes are implemented.

TODO (Task 1.7 / Bonus A):
  - Build fake LLM / retriever / tool objects (no Databricks, no network).
  - Call `build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])`.
  - Invoke it on a combined retrieval+calculation query and assert that a plan was
    produced, both specialists ran, and the final answer surfaced on messages[-1].

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


class _FakeMsg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    def invoke(self, messages):
        system = ""
        user = ""
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if role in ("system", "SystemMessage") or (
                isinstance(m, dict) and m.get("role") == "system"
            ):
                system = content or system
            else:
                user = content or user

        system_l = system.lower()
        user_l = user.lower()

        if "json list" in system_l or "atomic steps" in system_l or "planning" in system_l:
            return _FakeMsg(
                '["Find Meridian net revenue for FY2023", '
                '"Calculate 10 percent growth on that revenue"]'
            )
        if "route" in system_l or "rag_agent" in system_l or "mcp_tools" in system_l:
            if any(w in user_l for w in ("calculate", "growth", "percent", "%")):
                return _FakeMsg("mcp_tools")
            return _FakeMsg("rag_agent")
        if "extract" in system_l or "retrieved chunks" in system_l:
            return _FakeMsg(
                "Meridian net revenue FY2023 was 16.91 trillion "
                "[source: annual_report.pdf, p.4]"
            )
        if "combine" in system_l or "final answer" in system_l:
            return _FakeMsg(
                "Net revenue was 16.91 trillion; after 10% growth it is 18.601 trillion."
            )
        return _FakeMsg(user or "ok")

    def bind_tools(self, tools):
        return FakeToolLLM(tools)


class FakeToolLLM:
    def __init__(self, tools):
        self.tools = tools

    def invoke(self, messages):
        name = self.tools[0].name if self.tools else "calculate"
        if name == "growth_rate":
            args = {"start_value": 16.91, "rate": 0.10, "years": 1}
        elif name == "calculate":
            args = {"expression": "16.91 * 1.10"}
        else:
            args = {}
        return _FakeMsg("", tool_calls=[{"name": name, "args": args}])


class FakeRetriever:
    def invoke(self, query):
        from langchain_core.documents import Document

        return [
            Document(
                page_content="Meridian Motor Corporation net revenue FY2023: 16.91 trillion yen.",
                metadata={"source": "annual_report.pdf", "page": 4, "chunk_id": "c1"},
            )
        ]


class FakeTool:
    name = "calculate"
    description = "Evaluate a math expression"

    def invoke(self, args):
        expr = args.get("expression", "") if isinstance(args, dict) else str(args)
        if "16.91" in expr and "1.10" in expr:
            return f"{expr} = 18.601"
        return f"{expr} = 0"


def test_graph_combined_query_offline():
    from agent.graph import build_graph

    graph = build_graph(
        llm=FakeLLM(),
        retriever=FakeRetriever(),
        tools=[FakeTool()],
    )
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What was 2023 revenue, and what is it after 10% growth?",
                }
            ]
        }
    )

    assert result.get("plan"), "expected a plan"
    assert len(result.get("step_results") or []) >= 2, "expected rag + mcp step results"
    assert result["messages"][-1].content, "expected final answer on messages[-1]"
