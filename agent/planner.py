"""Planner node (Task 1.2).

TODO: Implement `make_planner(llm)` returning a node that:
  - reads the user question from state["messages"],
  - asks the LLM (PLANNER_PROMPT) for a JSON list of 2-5 steps,
  - parses it robustly (fallback to a single step on parse failure),
  - returns {"plan": [...], "current_step_index": 0, "step_results": []}.
"""

from __future__ import annotations

from agent.state import AnalystState
from agent.prompts import PLANNER_PROMPT
import json
import re

def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        user_question = state["messages"][-1].content

        response = llm.invoke([
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": user_question},
        ])

        raw = response.content.strip()

        # Models sometimes wrap JSON in markdown fences despite instructions —
        # strip those before parsing.
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())

        try:
            plan = json.loads(raw)
            if not isinstance(plan, list) or not all(isinstance(s, str) for s in plan):
                raise ValueError("Parsed JSON is not a list of strings")
        except (json.JSONDecodeError, ValueError):
            # Fallback: treat the whole question as a single atomic step.
            plan = [user_question]

        return {"plan": plan, "current_step_index": 0, "step_results": []}

    return planner
