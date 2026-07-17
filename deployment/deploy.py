"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

TODO:
  - `log_and_register()`: set registry uri to 'databricks-uc', log the model via
    `mlflow.langchain.log_model(lc_model="deployment/agent_model.py", name=...,
    code_paths=[...], pip_requirements=[...], input_example={...})`, then
    `mlflow.register_model(...)` into $UC_CATALOG.$UC_SCHEMA.<model>.
  - `create_or_update_endpoint(uc_name, version)`: create/update a Model Serving
    endpoint with `WorkspaceClient().serving_endpoints`, workload_size='Small',
    scale_to_zero_enabled=True, and environment_vars supplied as secret refs
    ({{secrets/cs4603-deploy/...}}). Wait for READY and print the URL.
"""

from __future__ import annotations

import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from dotenv import load_dotenv

load_dotenv()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log_and_register():
    catalog = os.environ.get("UC_CATALOG", "main")
    schema = os.environ.get("UC_SCHEMA", "default")
    model_name = os.environ.get("MODEL_NAME", "document_analyst")
    uc_name = f"{catalog}.{schema}.{model_name}"

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "/Shared/pa4-document-analyst"))

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model=os.path.join(os.path.dirname(__file__), "agent_model.py"),
            name="agent",
            code_paths=[
                os.path.join(root, "agent"),
                os.path.join(root, "rag"),
                os.path.join(root, "tools"),
                os.path.join(root, "config.py"),
            ],
            pip_requirements=[
                "mlflow",
                "langgraph",
                "langchain-openai",
                "databricks-langchain",
                "databricks-vectorsearch",
                "langchain-mcp-adapters",
                "mcp",
                "openai",
            ],
            input_example={"messages": [{"role": "user", "content": "What was the revenue?"}]},
        )
        registered = mlflow.register_model(model_info.model_uri, uc_name)

    print(f"Registered model version: {registered.version}")
    return uc_name, str(registered.version)


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    endpoint_name = os.environ.get("SERVING_ENDPOINT_NAME", "document-analyst")

    environment_vars = {
        "DATABRICKS_HOST": "{{secrets/cs4603-deploy/DATABRICKS_HOST}}",
        "DATABRICKS_TOKEN": "{{secrets/cs4603-deploy/DATABRICKS_TOKEN}}",
        "DATABRICKS_MODEL": "{{secrets/cs4603-deploy/DATABRICKS_MODEL}}",
        "VECTOR_SEARCH_ENDPOINT": os.environ.get("VECTOR_SEARCH_ENDPOINT", ""),
        "VECTOR_SEARCH_INDEX": os.environ.get("VECTOR_SEARCH_INDEX", ""),
        "EMBEDDINGS_ENDPOINT": os.environ.get(
            "EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"
        ),
        "MCP_SERVER_URL": os.environ.get("MCP_SERVER_URL", ""),
    }

    w = WorkspaceClient()
    config = EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=uc_name,
                entity_version=version,
                workload_size="Small",
                scale_to_zero_enabled=True,
                environment_vars=environment_vars,
            )
        ]
    )

    existing = {e.name for e in w.serving_endpoints.list()}
    if endpoint_name in existing:
        w.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=config.served_entities,
        )
    else:
        w.serving_endpoints.create_and_wait(name=endpoint_name, config=config)

    host = os.environ["DATABRICKS_HOST"].rstrip("/")
    url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    print(f"Endpoint URL: {url}")
    print(f"Deployed model version: {version}")
    print(f"Endpoint status: READY (name={endpoint_name})")
    return url


if __name__ == "__main__":
    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)
