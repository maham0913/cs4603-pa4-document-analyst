"""Python client SDK for the deployed Document Analyst (Part 3).

TODO: Implement `DocumentAnalystClient` and `AnalystClientError` per Task 3.1:
  - __init__(endpoint_name, host=None, token=None, timeout=120.0, max_retries=3):
    read DATABRICKS_HOST/DATABRICKS_TOKEN from env when not provided.
  - ask(question) -> str
  - ask_streaming(question) -> Iterator[str]   (yield chunks as they arrive)
  - health_check() -> bool                      (True only when endpoint READY)
  - exponential backoff on 429/503, TimeoutError with elapsed time, and wrap HTTP
    errors in AnalystClientError(status_code, message, request_id).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import httpx


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.host = (host or os.environ["DATABRICKS_HOST"]).rstrip("/")
        self.token = token or os.environ["DATABRICKS_TOKEN"]
        self.timeout = timeout
        self.max_retries = max_retries
        self._url = f"{self.host}/serving-endpoints/{endpoint_name}/invocations"
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def ask(self, question: str) -> str:
        payload = {"messages": [{"role": "user", "content": question}]}
        started = time.time()

        for attempt in range(self.max_retries + 1):
            elapsed = time.time() - started
            remaining = self.timeout - elapsed
            if remaining <= 0:
                raise TimeoutError(f"Request timed out after {elapsed:.2f}s")

            try:
                with httpx.Client(timeout=remaining) as client:
                    response = client.post(
                        self._url, headers=self._headers, json=payload
                    )
            except httpx.TimeoutException as exc:
                elapsed = time.time() - started
                raise TimeoutError(f"Request timed out after {elapsed:.2f}s") from exc

            if response.status_code in (429, 503) and attempt < self.max_retries:
                time.sleep(2**attempt)
                continue

            if response.status_code >= 400:
                raise AnalystClientError(
                    response.text,
                    status_code=response.status_code,
                    request_id=response.headers.get("x-request-id"),
                )

            data = response.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            if "messages" in data:
                return data["messages"][-1]["content"]
            return json.dumps(data)

        raise AnalystClientError("Exhausted retries")

    def ask_streaming(self, question: str) -> Iterator[str]:
        payload = {
            "messages": [{"role": "user", "content": question}],
            "stream": True,
        }
        started = time.time()

        for attempt in range(self.max_retries + 1):
            elapsed = time.time() - started
            remaining = self.timeout - elapsed
            if remaining <= 0:
                raise TimeoutError(f"Request timed out after {elapsed:.2f}s")

            try:
                with httpx.Client(timeout=remaining) as client:
                    with client.stream(
                        "POST", self._url, headers=self._headers, json=payload
                    ) as response:
                        if (
                            response.status_code in (429, 503)
                            and attempt < self.max_retries
                        ):
                            response.read()
                            time.sleep(2**attempt)
                            continue

                        if response.status_code >= 400:
                            response.read()
                            raise AnalystClientError(
                                response.text,
                                status_code=response.status_code,
                                request_id=response.headers.get("x-request-id"),
                            )

                        yielded = False
                        for line in response.iter_lines():
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            text = delta.get("content") or (
                                choices[0].get("message") or {}
                            ).get("content")
                            if text:
                                yielded = True
                                yield text

                        if not yielded:
                            yield self.ask(question)
                        return
            except httpx.TimeoutException as exc:
                elapsed = time.time() - started
                raise TimeoutError(f"Request timed out after {elapsed:.2f}s") from exc

        raise AnalystClientError("Exhausted retries")

    def health_check(self) -> bool:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient(host=self.host, token=self.token)
        ep = w.serving_endpoints.get(self.endpoint_name)
        ready = getattr(getattr(ep, "state", None), "ready", None)
        return ready is not None and "READY" in str(ready).upper()
