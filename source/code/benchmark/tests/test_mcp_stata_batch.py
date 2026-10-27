from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from mcp_stata_batch import server


def test_initialize_and_tools_list_round_trip() -> None:
    requests = "\n".join(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {},
                }
            ),
        )
    )
    output = io.StringIO()

    server.serve(io.StringIO(requests), output)

    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[0]["result"]["protocolVersion"] == "2025-06-18"
    assert responses[0]["result"]["serverInfo"]["name"] == "mcp-stata"
    tool_names = {tool["name"] for tool in responses[1]["result"]["tools"]}
    assert tool_names == {
        "stata_run",
        "stata_run_file",
        "stata_doctor",
        "stata_versions",
        "stata_set_default_version",
    }


def test_run_stata_code_uses_batch_mode_and_reads_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "stata-mp"
    binary.write_text("", encoding="utf-8")
    binary.chmod(0o700)

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
        start_new_session: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert command[:4] == [str(binary.resolve()), "-q", "-b", "do"]
        assert cwd == tmp_path.resolve()
        assert capture_output is True
        assert text is True
        assert timeout == 12
        assert check is False
        assert start_new_session is True
        Path(command[-1]).with_suffix(".log").write_text(
            ". display 2 + 2\n4\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    result = server.run_stata_code(
        "display 2 + 2",
        cwd=str(tmp_path),
        timeout_seconds=12,
        stata_path=str(binary),
    )

    assert result["ok"] is True
    assert result["stata_rc"] is None
    assert result["timed_out"] is False
    assert result["truncated"] is False
    assert "display 2 + 2" in result["log"]
    assert "\n4\n" in result["log"]


def test_run_stata_code_cleans_up_cwd_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real Stata batch mode writes ``<stem>.log`` into cwd; it must not linger."""
    binary = tmp_path / "stata-mp"
    binary.write_text("", encoding="utf-8")
    binary.chmod(0o700)
    workdir = tmp_path / "work"
    workdir.mkdir()

    def fake_run(command: list[str], *, cwd: Path, **_: object):
        # Mirror real Stata: the batch log lands in cwd, named after the do-file.
        stem = Path(command[-1]).stem
        (Path(cwd) / f"{stem}.log").write_text(". display 1\n1\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    result = server.run_stata_code(
        "display 1", cwd=str(workdir), stata_path=str(binary)
    )

    assert result["ok"] is True
    assert "\n1\n" in result["log"]  # log was still read before cleanup
    assert list(workdir.glob("*.log")) == []  # nothing left behind in cwd


def test_run_stata_code_detects_stata_return_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "stata-mp"
    binary.write_text("", encoding="utf-8")
    binary.chmod(0o700)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).with_suffix(".log").write_text(
            ". invalid_command\ncommand invalid_command is unrecognized\nr(199);\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    result = server.run_stata_code(
        "invalid_command",
        cwd=str(tmp_path),
        stata_path=str(binary),
    )

    assert result["ok"] is False
    assert result["stata_rc"] == 199
    assert result["error"] == "Stata returned r(199)"


def test_run_file_uses_parent_as_default_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    do_file = tmp_path / "example.do"
    do_file.write_text("display 42\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_stata_code(code: str, **kwargs: object) -> dict[str, object]:
        captured["code"] = code
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(server, "run_stata_code", fake_run_stata_code)

    result = server.run_stata_file(str(do_file))

    assert result == {"ok": True}
    assert captured["code"] == "display 42\n"
    assert captured["cwd"] == str(tmp_path.resolve())


def test_tool_errors_are_returned_as_mcp_tool_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_: str | None = None) -> dict[str, object]:
        raise FileNotFoundError("Stata missing")

    monkeypatch.setattr(server, "stata_doctor", fail)
    response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "stata_doctor", "arguments": {}},
        }
    )

    assert response is not None
    result = response["result"]
    assert result["isError"] is True
    assert result["structuredContent"]["error_type"] == "FileNotFoundError"
    assert "Stata missing" in result["content"][0]["text"]


def test_license_serial_is_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "stata-mp"
    binary.write_text("", encoding="utf-8")
    binary.chmod(0o700)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).with_suffix(".log").write_text(
            "Single-user license:\n       Serial number:  12345678\n. display 1\n1\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    result = server.run_stata_code(
        "display 1",
        cwd=str(tmp_path),
        stata_path=str(binary),
    )

    assert "12345678" not in result["log"]
    assert "Serial number:  [redacted]" in result["log"]


def test_default_profile_persists_and_normalizes_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_file = tmp_path / "default-profile"
    monkeypatch.setenv("MCP_STATA_STATE_FILE", str(state_file))

    assert server.get_default_profile() == "auto"
    result = server.set_default_profile("statanow")

    assert result["default_profile"] == "19"
    assert state_file.read_text(encoding="utf-8") == "19\n"
    assert server.get_default_profile() == "19"
    assert server.normalize_profile("stata-14") == "14"


def test_auto_falls_back_only_after_stata19_license_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stata19 = tmp_path / "stata19"
    stata14 = tmp_path / "stata14"
    calls: list[str] = []

    monkeypatch.setattr(
        server,
        "resolve_stata_candidates",
        lambda profile, explicit_path: (
            "auto",
            [("19", stata19), ("14", stata14)],
        ),
    )

    def fake_execute(
        code: str,
        *,
        binary: Path,
        requested_profile: str,
        selected_profile: str,
        cwd: str | None,
        timeout_seconds: int | float | None,
    ) -> dict[str, object]:
        calls.append(selected_profile)
        if selected_profile == "19":
            return {
                "ok": False,
                "error": "Your license has expired",
                "license_error": True,
                "selected_profile": "19",
            }
        return {
            "ok": True,
            "error": None,
            "license_error": False,
            "selected_profile": "14",
        }

    monkeypatch.setattr(server, "_execute_stata_code", fake_execute)

    result = server.run_stata_code("display 1", version="auto")

    assert calls == ["19", "14"]
    assert result["ok"] is True
    assert result["selected_profile"] == "14"
    assert result["fallback_from"] == "19"
    assert result["fallback_reason"] == "Your license has expired"


def test_auto_does_not_fallback_after_ordinary_stata_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stata19 = tmp_path / "stata19"
    stata14 = tmp_path / "stata14"
    calls: list[str] = []

    monkeypatch.setattr(
        server,
        "resolve_stata_candidates",
        lambda profile, explicit_path: (
            "auto",
            [("19", stata19), ("14", stata14)],
        ),
    )

    def fake_execute(
        code: str,
        *,
        binary: Path,
        requested_profile: str,
        selected_profile: str,
        cwd: str | None,
        timeout_seconds: int | float | None,
    ) -> dict[str, object]:
        calls.append(selected_profile)
        return {
            "ok": False,
            "error": "Stata returned r(199)",
            "license_error": False,
            "selected_profile": selected_profile,
        }

    monkeypatch.setattr(server, "_execute_stata_code", fake_execute)

    result = server.run_stata_code("invalid_command", version="auto")

    assert calls == ["19"]
    assert result["ok"] is False
    assert result["selected_profile"] == "19"
    assert result["fallback_from"] is None


def test_explicit_profile_resolves_only_requested_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stata14 = tmp_path / "stata-mp"
    stata19 = tmp_path / "stata-se"
    for binary in (stata14, stata19):
        binary.write_text("", encoding="utf-8")
        binary.chmod(0o700)
    monkeypatch.setattr(
        server,
        "STATA_PROFILES",
        {"14": stata14, "19": stata19},
    )

    requested, candidates = server.resolve_stata_candidates("19")

    assert requested == "19"
    assert candidates == [("19", stata19.resolve())]
