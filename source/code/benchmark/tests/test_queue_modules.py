"""Tests for the worker-side queue modules (no network, no subprocesses)."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import preflight  # noqa: E402
import push_records  # noqa: E402
import push_scores  # noqa: E402
import queue_client  # noqa: E402


# ---------------------------------------------------------------------------
# queue_client
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = 200

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def http_error(code: int, body: bytes = b"boom") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.test/x", code, "err", {}, io.BytesIO(body)
    )


def make_client() -> queue_client.QueueClient:
    return queue_client.QueueClient(
        base_url="https://example.test", worker_key="sekrit"
    )


def test_push_records_chunks_auth_and_payload(monkeypatch):
    calls: list[tuple[str, str | None, dict[str, Any], float | None]] = []

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        calls.append((req.full_url, req.get_header("Authorization"), payload, timeout))
        return FakeResponse({"inserted": len(payload["items"]), "exists": 0})

    monkeypatch.setattr(queue_client, "urlopen", fake_urlopen)
    client = make_client()
    items = [{"record": {"uuid": f"u{i}"}} for i in range(250)]
    out = client.push_records(items)

    assert [len(call[2]["items"]) for call in calls] == [100, 100, 50]
    assert all(call[0] == "https://example.test/v1/records" for call in calls)
    assert all(call[1] == "Bearer sekrit" for call in calls)
    assert all(call[3] == queue_client.REQUEST_TIMEOUT_SECONDS for call in calls)
    assert out["items"] == 250
    assert out["requests"] == 3
    assert out["inserted"] == 250
    assert out["exists"] == 0


def test_push_scores_chunks_to_500(monkeypatch):
    sizes: list[int] = []

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        sizes.append(len(payload["rows"]))
        return FakeResponse({"inserted": len(payload["rows"])})

    monkeypatch.setattr(queue_client, "urlopen", fake_urlopen)
    rows = [{"run_uuid": f"u{i}"} for i in range(501)]
    out = make_client().push_scores(rows)
    assert sizes == [500, 1]
    assert out["rows"] == 501
    assert out["inserted"] == 501


def test_idempotent_get_retries_on_503(monkeypatch):
    attempts: list[str] = []
    sleeps: list[float] = []

    def fake_urlopen(req, timeout=None):
        attempts.append(req.full_url)
        if len(attempts) < 3:
            raise http_error(503, b"maintenance")
        return FakeResponse({"ok": True})

    monkeypatch.setattr(queue_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(queue_client.time, "sleep", lambda s: sleeps.append(s))
    assert make_client().healthz() == {"ok": True}
    assert len(attempts) == 3
    assert len(sleeps) == 2


def test_client_error_is_not_retried_and_raises(monkeypatch):
    attempts: list[int] = []

    def fake_urlopen(req, timeout=None):
        attempts.append(1)
        raise http_error(400, b"bad payload")

    monkeypatch.setattr(queue_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(queue_client.time, "sleep", lambda s: None)
    with pytest.raises(queue_client.QueueError) as excinfo:
        make_client().status()
    assert len(attempts) == 1
    assert excinfo.value.status == 400
    assert "bad payload" in str(excinfo.value)


def test_claim_is_never_retried(monkeypatch):
    attempts: list[int] = []

    def fake_urlopen(req, timeout=None):
        attempts.append(1)
        raise http_error(503)

    monkeypatch.setattr(queue_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(queue_client.time, "sleep", lambda s: None)
    with pytest.raises(queue_client.QueueError):
        make_client().claim(max_cells=2)
    assert len(attempts) == 1


def test_worker_key_falls_back_to_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTS_METRICS_WORKER_KEY", raising=False)
    monkeypatch.delenv("AGENTS_METRICS_API", raising=False)
    env_file = tmp_path / "worker.env"
    env_file.write_text("# worker credentials\nKEY=abc123\n")
    client = queue_client.QueueClient(env_file=env_file)
    assert client.worker_key == "abc123"
    assert client.base_url == queue_client.DEFAULT_BASE_URL


# ---------------------------------------------------------------------------
# push_records
# ---------------------------------------------------------------------------


class FakeQueue:
    def __init__(self, fail: bool = False) -> None:
        self.chunks: list[list[dict[str, Any]]] = []
        self.fail = fail

    def push_records(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if self.fail:
            raise queue_client.QueueError("boom", status=503)
        self.chunks.append(list(items))
        return {"inserted": len(items), "exists": 0}


def write_run(
    runs_dir: Path,
    name: str,
    record: dict[str, Any],
    files: dict[str, str] | None = None,
) -> Path:
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "run_record.json").write_text(json.dumps(record))
    for fname, content in (files or {}).items():
        (run_dir / fname).write_text(content)
    return run_dir


def make_fake_runs(tmp_path: Path) -> tuple[Path, Path]:
    runs = tmp_path / "runs"
    write_run(runs, "cell_a", {"uuid": "aaa", "platform": "python"}, {"run.py": "a\n"})
    write_run(
        runs,
        "cell_b",
        {"uuid": "bbb", "platform": "python"},
        {"run.py": "print(2)\n", "result.json": json.dumps({"est": 1.5})},
    )
    write_run(runs, "cell_c", {"platform": "r"}, {"run.R": "cat(1)\n"})
    state_file = runs / ".pushed_uuids.json"
    state_file.write_text(json.dumps(["aaa"]))
    return runs, state_file


def test_sync_records_state_file_behavior(tmp_path):
    runs, state_file = make_fake_runs(tmp_path)
    fake = FakeQueue()
    summary = push_records.sync_records(runs, state_file, client=fake)
    assert summary == {
        "scanned": 3,
        "pushed_new": 1,
        "already_present": 1,
        "skipped_no_uuid": 1,
        "errors": 0,
    }
    assert len(fake.chunks) == 1
    assert len(fake.chunks[0]) == 1
    item = fake.chunks[0][0]
    assert item["record"]["uuid"] == "bbb"
    assert item["script_name"] == "run.py"
    assert item["script"] == "print(2)\n"
    assert item["result_json"] == {"est": 1.5}
    assert push_records.load_state(state_file) == {"aaa", "bbb"}


def test_sync_records_all_flag_repushes_everything(tmp_path):
    runs, state_file = make_fake_runs(tmp_path)
    fake = FakeQueue()
    summary = push_records.sync_records(runs, state_file, push_all=True, client=fake)
    assert summary["pushed_new"] == 2
    assert summary["already_present"] == 0
    assert summary["skipped_no_uuid"] == 1
    pushed_uuids = {item["record"]["uuid"] for chunk in fake.chunks for item in chunk}
    assert pushed_uuids == {"aaa", "bbb"}


def test_sync_records_dry_run_touches_nothing(tmp_path):
    runs, state_file = make_fake_runs(tmp_path)
    summary = push_records.sync_records(runs, state_file, dry_run=True, client=None)
    assert summary["would_push"] == 1
    assert summary["pushed_new"] == 0
    assert push_records.load_state(state_file) == {"aaa"}


def test_sync_records_push_failure_counts_errors(tmp_path):
    runs, state_file = make_fake_runs(tmp_path)
    summary = push_records.sync_records(runs, state_file, client=FakeQueue(fail=True))
    assert summary["errors"] == 1
    assert summary["pushed_new"] == 0
    assert push_records.load_state(state_file) == {"aaa"}


def test_find_script_prefers_last_self_repair_attempt(tmp_path):
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    (run_dir / "attempt1_run.R").write_text("first")
    (run_dir / "attempt2_run.R").write_text("second")
    found = push_records.find_script(run_dir, "r")
    assert found is not None
    assert found.name == "attempt2_run.R"
    item = push_records.build_item(run_dir, {"uuid": "x", "platform": "r"})
    assert item["script_name"] == "attempt2_run.R"
    assert item["script"] == "second"


def test_build_item_skips_invalid_result_json(tmp_path):
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    (run_dir / "run.do").write_text("display 1")
    (run_dir / "result.json").write_text("{not json")
    item = push_records.build_item(run_dir, {"uuid": "x", "platform": "stata"})
    assert item["script_name"] == "run.do"
    assert "result_json" not in item


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

EXPECTED_R_PACKAGES = [
    "jsonlite",
    "sandwich",
    "lmtest",
    "car",
    "AER",
    "ivreg",
    "lfe",
    "fixest",
    "plm",
    "multiwayvcov",
    "vars",
    "forecast",
]


def test_parse_r_packages_matches_installer_script():
    assert preflight.parse_r_packages() == EXPECTED_R_PACKAGES


def test_canonical_r_packages_appends_survival():
    pkgs = preflight.canonical_r_packages()
    assert pkgs == EXPECTED_R_PACKAGES + ["survival"]
    assert len(pkgs) == 13
    assert len(set(pkgs)) == 13


# ---------------------------------------------------------------------------
# push_scores
# ---------------------------------------------------------------------------


def test_build_score_rows_maps_run_dirs_to_uuids(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "cell_x", {"uuid": "ux"})
    write_run(runs, "cell_y", {"uuid": "uy"})
    write_run(runs, "cell_no_uuid", {"cell_id": "cell_no_uuid"})
    report = {
        "scoring_version": "strict_posthoc_2026-05-20",
        "scores": [
            {
                "run_dir": "/other/host/runs/cell_x",
                "task_success": True,
                "quarantined": False,
                "prompt_wave": "pinned-2026-07",
            },
            {
                "run_dir": "/other/host/runs/cell_y",
                "task_success": False,
                "quarantined": True,
            },
            {
                "run_dir": "/other/host/runs/cell_missing",
                "task_success": True,
                "quarantined": False,
            },
            {
                "run_dir": "/other/host/runs/cell_no_uuid",
                "task_success": True,
                "quarantined": False,
            },
        ],
    }
    rows, counts = push_scores.build_score_rows(report, runs)
    assert counts == {
        "rows_in_report": 4,
        "mapped": 2,
        "skipped_no_record": 1,
        "skipped_no_uuid": 1,
    }
    first, second = rows
    assert first["run_uuid"] == "ux"
    assert first["scoring_version"] == "strict_posthoc_2026-05-20"
    assert first["task_success"] is True
    assert first["quarantined"] is False
    assert first["prompt_wave"] == "pinned-2026-07"
    assert first["score"] is report["scores"][0]
    assert second["run_uuid"] == "uy"
    assert second["task_success"] is False
    assert second["quarantined"] is True
    assert second["prompt_wave"] == "legacy"
