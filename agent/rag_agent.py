"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

TODO: Implement `make_rag_agent(retriever, llm)` returning a node that:
  - retrieves top-k chunks for the current step,
  - formats them with [source: file, p.N] citations,
  - extracts a single cited fact via the LLM (or 'not found in documents'),
  - appends the fact to step_results and increments current_step_index.
Reuse `rag/store.py::get_retriever()` so local and deployed retrieval match.
"""

from __future__ import annotations

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState


def format_docs(docs) -> str:
    parts = []
    for doc in docs:
        meta = getattr(doc, "metadata", None) or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        parts.append(f"{doc.page_content}\n[source: {source}, p.{page}]")
    return "\n\n".join(parts)


def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        step = state["plan"][state["current_step_index"]]
        docs = retriever.invoke(step)
        if not docs:
            fact = "not found in documents"
        else:
            context = format_docs(docs)
            response = llm.invoke(
                [
                    {"role": "system", "content": RAG_EXTRACT_PROMPT},
                    {
                        "role": "user",
                        "content": f"Step: {step}\n\nRetrieved chunks:\n{context}",
                    },
                ]
            )
            fact = (response.content or "").strip() or "not found in documents"

        step_results = list(state.get("step_results") or [])
        step_results.append(fact)
        return {
            "step_results": step_results,
            "current_step_index": state["current_step_index"] + 1,
        }

    return rag_agent
