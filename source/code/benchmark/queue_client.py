"""Stdlib-only HTTP client for the agents-metrics central queue API.

Workers use this to claim benchmark cells, push run records and score rows,
and report preflight capabilities to the central VPS. The binding contract is
archive/quality_reports_2026-08-10/plans/2026-07-04_central-queue-and-db.md.

This module must import without numpy/pandas (it runs under the repo uv venv),
so it uses only the standard library.

Configuration:
- ``AGENTS_METRICS_API``        base URL (default https://agents-metrics.falstuff.com)
- ``AGENTS_METRICS_WORKER_KEY`` bearer key; when unset, the key is read from
  ``~/.config/agents-metrics-worker.env`` (a ``KEY=...`` line, mode 600).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://agents-metrics.falstuff.com"
WORKER_ENV_FILE = Path.home() / ".config" / "agents-metrics-worker.env"
ENV_KEY_NAMES = ("AGENTS_METRICS_WORKER_KEY", "KEY")
REQUEST_TIMEOUT_SECONDS = 60
RECORDS_CHUNK_SIZE = 100
SCORES_CHUNK_SIZE = 500
MAX_RETRIES = 3  # retries after the first attempt; idempotent calls only
RETRY_STATUS_CODES = frozenset({502, 503})
# Client-side backoff policy between retries (a design choice of this client,
# not a measured server pacing constant).
RETRY_BACKOFF_SECONDS = (2.0, 5.0, 10.0)


class QueueError(RuntimeError):
    """HTTP or transport failure talking to the queue API."""

    def __init__(
        self,
        message: str,
        status: int | None = None,
        body: str | None = None,
    ) -> None:
        snippet = (body or "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        full = f"{message} [HTTP {status}]" if status is not None else message
        if snippet:
            full = f"{full}: {snippet}"
        super().__init__(full)
        self.status = status
        self.body = snippet


def read_worker_env_key(env_file: Path = WORKER_ENV_FILE) -> str | None:
    """Parse the bearer key from a ``KEY=...`` style env file."""
    try:
        text = env_file.read_text()
    except OSError:
        return None
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip().strip('"').strip("'")
    for name in ENV_KEY_NAMES:
        if values.get(name):
            return values[name]
    return None


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Split ``items`` into consecutive chunks of at most ``size``."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def _sum_key(responses: list[dict[str, Any]], key: str) -> int | None:
    """Best-effort sum of an integer field across per-chunk responses."""
    values = [r.get(key) for r in responses if isinstance(r.get(key), int)]
    return sum(values) if values else None


class QueueClient:
    """Minimal bearer-auth JSON client for the central queue API."""

    def __init__(
        self,
        base_url: str | None = None,
        worker_key: str | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        env_file: Path = WORKER_ENV_FILE,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("AGENTS_METRICS_API") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.worker_key = (
            worker_key
            or os.environ.get("AGENTS_METRICS_WORKER_KEY")
            or read_worker_env_key(env_file)
        )
        self.timeout = timeout

    # -- transport ---------------------------------------------------------

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        retry: bool = False,
    ) -> dict[str, Any]:
        """One API call. ``retry=True`` (idempotent calls only) retries
        transient failures (HTTP 502/503, timeouts, connection errors) up to
        MAX_RETRIES times with backoff."""
        url = f"{self.base_url}{path}"
        data: bytes | None = None
        headers = {"Accept": "application/json"}
        if self.worker_key:
            headers["Authorization"] = f"Bearer {self.worker_key}"
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        max_attempts = 1 + MAX_RETRIES if retry else 1
        for attempt in range(max_attempts):
            req = Request(url, data=data, headers=headers, method=method)
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8", "replace")
                return json.loads(body) if body.strip() else {}
            except HTTPError as err:
                err_body = ""
                try:
                    err_body = err.read().decode("utf-8", "replace")
                except (OSError, ValueError):
                    pass
                if err.code in RETRY_STATUS_CODES and attempt < max_attempts - 1:
                    self._sleep(attempt)
                    continue
                raise QueueError(
                    f"{method} {path} failed", status=err.code, body=err_body
                ) from err
            except json.JSONDecodeError as err:
                raise QueueError(f"{method} {path} returned non-JSON body") from err
            except OSError as err:  # URLError, socket timeouts, conn resets
                if attempt < max_attempts - 1:
                    self._sleep(attempt)
                    continue
                raise QueueError(f"{method} {path} failed: {err}") from err
        raise QueueError(f"{method} {path} failed")  # pragma: no cover

    def _sleep(self, attempt: int) -> None:
        delay = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
        time.sleep(delay)

    # -- endpoints ---------------------------------------------------------

    def healthz(self) -> dict[str, Any]:
        return self.request("GET", "/healthz", retry=True)

    def preflight(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Report worker capabilities; idempotent upsert, safe to retry."""
        return self.request("POST", "/v1/preflight", payload, retry=True)

    def claim(
        self, max_cells: int = 1, usage: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Lease up to ``max_cells`` queue slots. NOT idempotent: never retried."""
        payload: dict[str, Any] = {"max_cells": max_cells}
        if usage is not None:
            payload["usage"] = usage
        return self.request("POST", "/v1/claim", payload)

    def heartbeat(self, queue_id: int) -> dict[str, Any]:
        return self.request("POST", "/v1/heartbeat", {"queue_id": queue_id})

    def release(self, queue_id: int, reason: str) -> dict[str, Any]:
        return self.request(
            "POST", "/v1/release", {"queue_id": queue_id, "reason": reason}
        )

    def push_records(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Push run-record items (each {record, script_name?, script?,
        result_json?}), auto-chunked to <= RECORDS_CHUNK_SIZE per request.
        Server dedupes by record uuid, so retries are safe."""
        responses = [
            self.request("POST", "/v1/records", {"items": chunk}, retry=True)
            for chunk in chunked(list(items), RECORDS_CHUNK_SIZE)
        ]
        return {
            "items": len(items),
            "requests": len(responses),
            "inserted": _sum_key(responses, "inserted"),
            "exists": _sum_key(responses, "exists"),
            "responses": responses,
        }

    def push_scores(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Push score rows, auto-chunked to <= SCORES_CHUNK_SIZE per request.
        Keyed by run_uuid server-side, so retries are safe."""
        responses = [
            self.request("POST", "/v1/scores", {"rows": chunk}, retry=True)
            for chunk in chunked(list(rows), SCORES_CHUNK_SIZE)
        ]
        return {
            "rows": len(rows),
            "requests": len(responses),
            "inserted": _sum_key(responses, "inserted"),
            "exists": _sum_key(responses, "exists"),
            "responses": responses,
        }

    def status(self) -> dict[str, Any]:
        return self.request("GET", "/v1/status", retry=True)
