"""Bonus B — deploy via the databricks-agents SDK (deployment/deploy_agents.py).

TODO: Log + register the model (reuse the pattern from deploy.py), then call
`databricks.agents.deploy(model_name=..., model_version=...)` to provision the
serving endpoint AND the Review App in one call. Print the endpoint + review URL.
"""

from __future__ import annotations

from deployment.deploy import log_and_register


def main() -> None:
    from databricks import agents

    uc_name, version = log_and_register()
    deployment = agents.deploy(
        model_name=uc_name,
        model_version=version,
        scale_to_zero=True,
    )
    print(f"Endpoint name: {deployment.endpoint_name}")
    print(f"Review app URL: {deployment.review_app_url}")


if __name__ == "__main__":
    main()
