"""Synthesizer node (Task 1.6).

TODO: Implement `make_synthesizer(llm)` returning a node that combines
step_results into one cited answer and writes it to BOTH `final_answer` AND
the `messages` channel as an AIMessage (required for the OpenAI-compatible
serving contract — see spec Task 1.6).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        step_results = state.get("step_results") or []
        lines = [f"Step {i + 1}: {r}" for i, r in enumerate(step_results)]
        question = ""
        if state.get("messages"):
            question = state["messages"][0].content
        response = llm.invoke(
            [
                {"role": "system", "content": SYNTHESIZER_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nResults:\n" + "\n".join(lines),
                },
            ]
        )
        answer = (response.content or "").strip()
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    return synthesizer
