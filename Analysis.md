# CS4603 PA4 — Document Analyst

> This `README.md` is a **graded deliverable**:
>
> - Document how to set up, run, and deploy your Document Analyst so a TA can reproduce your results.
> - **Answer every ANALYSIS QUESTION** from the assignment in the sections below.
> - Code that runs but is not explained will not receive full marks.
> - Replace every `TODO` before submitting.
> - Keep it self-contained: a reader should be able to follow this file top-to-bottom —
>   setup → ingest → run → deploy → results — without opening the assignment PDF.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your values
```

## Running locally

1. **Ingest the corpus** (run once, from a Databricks notebook with Spark):
   ```python
   from rag.ingest import build_chunks_table, create_index
   build_chunks_table(
       spark,
       volume_path="/Volumes/main/default/pa4/annual_report.pdf",
       chunks_table="main.default.<your-name>_analyst_chunks",
   )
   create_index()  # creates VECTOR_SEARCH_ENDPOINT + VECTOR_SEARCH_INDEX
   ```
   Wait until the Vector Search index is `READY`.

2. **Build and run the graph** in `pa4.ipynb`:
   ```python
   from agent.graph import build_graph
   graph = build_graph()
   result = graph.invoke({"messages": [{"role": "user",
             "content": "What was the net revenue in 2023?"}]})
   print(result["messages"][-1].content)
   ```

3. **Test queries** (retrieval-only, computation-only, combined):
   | Query | Expected shape of answer |
   |-------|--------------------------|
   | "What was the net income in 2023?" | Fact + citation from the report |
   | "What is 15% of 2.4 billion?" | Numeric result via MCP tools |
   | "What was 2023 revenue, and its value after 10% growth?" | Retrieval + growth calculation |

## Deployment

```bash
# One-time secrets (never put tokens in the endpoint config as plaintext)
databricks secrets create-scope cs4603-deploy
databricks secrets put-secret cs4603-deploy DATABRICKS_TOKEN --string-value "dapi..."
databricks secrets put-secret cs4603-deploy DATABRICKS_HOST  --string-value "https://<workspace>.databricks.com"
databricks secrets put-secret cs4603-deploy DATABRICKS_MODEL --string-value "databricks-meta-llama-3-3-70b-instruct"

# Log + register + create/update serving endpoint
uv run python deployment/deploy.py
```

- Model definition: `deployment/agent_model.py` (models-from-code + `mlflow.models.set_model`)
- Endpoint name: value of `SERVING_ENDPOINT_NAME` in `.env`
- URL: `$DATABRICKS_HOST/serving-endpoints/<SERVING_ENDPOINT_NAME>/invocations`
- Alternative (Bonus B): `uv run python deployment/deploy_agents.py`

## Design decisions

- **Planner → supervisor loop → specialists → synthesizer**: planning first makes steps auditable; the supervisor routes per step so mixed retrieval+math queries work.
- **RAG and MCP as separate nodes**: retrieval prompts stay clean; math stays deterministic on the MCP server.
- **Databricks Vector Search only**: same retriever locally and in the serving container (no local DB).
- **Secrets via `{{secrets/cs4603-deploy/...}}`**: serving container has no `.env`.
- **`code_paths` ships `agent/`, `rag/`, `tools/`, `config.py`**: avoids the #1 deploy failure (`ModuleNotFoundError: agent`).
- **Synthesizer writes `AIMessage` to `messages`**: required for the OpenAI-compatible serving contract.

---

## Analysis Questions

### Task 1.2 — Planner
1. Dependent steps are handled implicitly via `step_results`: each completed step appends its result, and later nodes (especially MCP) see prior results when building tool args. There is no explicit dependency graph — the planner must order retrieval before calculation. If ordering is wrong, later steps fail silently or compute with missing numbers.
2. For typical 2–3 step CAGR-style queries, replanning after every step mostly adds latency/cost without benefit. It helps when a later step depends on *which* fact was retrieved (e.g. “find the best-margin segment, then project its growth”) because the fixed plan cannot name the segment until step 1 finishes.

### Task 1.3 — Supervisor
1. Misroute failure mode: a calc step sent to RAG returns “not found”; a lookup sent to MCP may call tools with invented numbers. Detect by validating step results (empty/not-found after RAG, or tool errors after MCP) and recover by re-classifying once or falling back to the other specialist before advancing `current_step_index`.
2. A single ReAct agent with all tools is simpler and fine for short tool chains. The supervisor pattern is worth it when you want auditable plans, specialist prompts, independent tuning of retrieval vs tools, and clearer failure isolation on multi-step analytical queries.

### Task 1.4 — RAG Agent
1. Retrieving for a decomposed step usually improves precision (narrower query → better chunks) versus the full user question, which mixes retrieval and math intent and can dilute similarity. The tradeoff is that a poorly phrased step can retrieve worse than a well-scoped original question.
2. Rewrite vague steps before retrieval: expand with entity/year from the original query, prior `step_results`, and report vocabulary (e.g. “net revenue FY2023 Meridian”) via a small query-rewrite LLM call or template.

### Task 2.1 — Model Definition
1. `models-from-code` packages what the serving container can import and run. External laptop-only state (local Postgres, open files, env-only paths) is invisible inside the container, so imports fail or retrieval dies at inference. Everything the model needs must be shipped (`code_paths` / `pip_requirements`) or reached as a managed cloud service.
2. External Vector Search: fresher corpus (re-sync index without rebuilding the model), smaller cold-start artifact, but adds network latency and a new failure mode (index down / auth). Baking the corpus into the artifact: offline inference and no VS dependency, but large images, stale data until re-deploy, and slower cold starts.

### Task 2.3 — Serving Endpoint
1. The serving principal authenticates *to serve traffic*; your graph still calls Databricks APIs (LLM endpoint, Vector Search) as a *client*. Those outbound calls need `DATABRICKS_TOKEN`/`HOST` (and related env) inside the container — serving auth does not automatically forward as your app credentials.
2. Databricks updates served entities with a rolling transition: new version is brought up, traffic shifts when healthy, old version drains. In-flight requests on the old replica typically finish there; new requests go to the new version once it is ready (brief mixed-version window possible during the swap).

### Task 3.2 — Client
1. Exponential backoff reduces thundering-herd load when the endpoint is rate-limiting (429) or scaling from zero (503). Fixed short retries keep hammering an overloaded/cold endpoint; backoff gives scale-up time and spaces retries.
2. Very high `max_retries` with many concurrent clients multiplies load (retry storms), inflates tail latency, and can exhaust client thread/connection pools while users wait.
3. Prefer `ask_streaming()` when the UI should show partial text early (chat typing indicator). Prefer `ask()` for batch jobs, tests, or when you only need the final string. Note: LangChain models-from-code may still return a single chunk — streaming then falls back to one full answer.

### Bonus A / B / C (if attempted)
**Bonus A:** Deploy only on `main` so feature branches cannot overwrite production with unfinished work; merge is the “ready” signal. Add an eval gate (held-out queries + faithfulness/accuracy vs current production metrics) between test and deploy so worse models never roll out.

**Bonus B:** `agents.deploy()` trades fine-grained control (manual secret wiring, exact endpoint config) for speed: one call provisions endpoint + Review App with automatic auth. Use Review App ratings in an MLflow feedback loop: sample low-rated traces, add them to an eval set, fix prompts/routing, re-deploy, compare metrics.

**Bonus C:** Separating MCP gains independent scale/deploy/observe of tools and a smaller model artifact; new failure modes are network, auth, latency, and tool-service downtime. Secure with Databricks App auth / IP allowlists / private connectivity so only the serving endpoint’s identity can call tools. Bundle tools in-container when the tool surface is tiny and tightly coupled; use a remote MCP service when tools are shared across agents or need independent release cadence.
