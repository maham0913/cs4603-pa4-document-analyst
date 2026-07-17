"""Supervisor node + routing edge (Task 1.3).

TODO:
  - `make_supervisor(llm)`: if current_step_index >= len(plan) -> next_agent =
    'synthesizer'; else classify the current step to 'rag_agent' or 'mcp_tools'.
  - `route_from_supervisor(state)`: return state["next_agent"] for the
    conditional edge.
"""

from __future__ import annotations

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        plan = state.get("plan") or []
        idx = state.get("current_step_index", 0)
        if idx >= len(plan):
            return {"next_agent": SYNTH}

        step = plan[idx]
        response = llm.invoke(
            [
                {"role": "system", "content": SUPERVISOR_PROMPT},
                {"role": "user", "content": step},
            ]
        )
        decision = (response.content or "").strip().lower()
        if MCP in decision or "mcp" in decision or "calc" in decision:
            next_agent = MCP
        elif RAG in decision or "rag" in decision:
            next_agent = RAG
        else:
            next_agent = RAG
        return {"next_agent": next_agent}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    return state["next_agent"]
