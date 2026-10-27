"""Generated Stata logs must not keep the license serial number or licensee."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "code" / "benchmark"
SITE_DIR = PROJECT_ROOT / "code" / "falstuff_site"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(SITE_DIR))

import factorial_analysis as fa  # noqa: E402
import production_paper_assets as assets  # noqa: E402
import production_statistics as ps  # noqa: E402
import stata_log_privacy  # noqa: E402

SERIAL_RE = re.compile(r"Serial number:\s*[0-9]{4,}")

STATA14_HEADER = (
    "  ___  ____  ____  ____  ____ (R)\n"
    "___/   /   /___/   /   /___/   14.0   Copyright 1985-2015 StataCorp LP\n"
    "\n"
    "Single-user 8-core Stata perpetual license:\n"
    "       Serial number:  99990001\n"
    "         Licensed to:  EXAMPLE OWNER\n"
    "                       \n"
    "\n"
    "Notes:\n"
    "      1.  Stata is running in batch mode.\n"
)
STATA19_HEADER = (
    "Stata license: Unlimited-user network, expiring 1 Jan 2099\n"
    "Serial number: 999900000001\n"
    "  Licensed to: EXAMPLE OWNER\n"
    "               EXAMPLE ORGANIZATION\n"
    "\n"
    "Notes:\n"
    "      1. Stata is running in batch mode.\n"
)
PRIVATE = ("99990001", "999900000001", "EXAMPLE OWNER", "EXAMPLE ORGANIZATION")


@pytest.mark.parametrize("header", [STATA14_HEADER, STATA19_HEADER])
def test_strip_license_details_removes_identifiers_and_keeps_output(header: str) -> None:
    text = header + ". display 1\n1\nPKGEST H1_r 0.5 0.1\n"

    cleaned = stata_log_privacy.strip_license_details(text)

    for private in PRIVATE:
        assert private not in cleaned
    assert "Serial number" not in cleaned
    assert "Licensed to" not in cleaned
    assert "Stata is running in batch mode." in cleaned
    assert cleaned.endswith(". display 1\n1\nPKGEST H1_r 0.5 0.1\n")


def test_strip_license_details_leaves_other_text_unchanged() -> None:
    text = "  Serial correlation test\n. display 1\n1\n"

    assert stata_log_privacy.strip_license_details(text) == text


def _fake_stata_log(ws: dict, contrasts: dict) -> str:
    lines = [STATA19_HEADER]
    for name in ws:
        lines.append(f"PKGEST {name} {contrasts[name]['estimate']:.15g} {contrasts[name]['se_cr1']:.15g}")
        lines.append(f"PKGBOOT {name} {200 / 9999:.15g} 9999 9999")
    lines += [f"PKGBOOT {name} {200 / 9999:.15g} 9999 9999" for name in ("H1_joint", "H4_joint")]
    return "\n".join(lines) + "\n"


def test_bootstrap_saves_inference_log_without_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = ps.weights()
    contrasts = {name: {"estimate": 0.01 * (i + 1), "se_cr1": 0.05} for i, name in enumerate(ws)}
    frame = pd.DataFrame({"success": [1, 0], "task": ["a", "b"], "cell": [1, 2], "armtask": ["x_a", "x_b"]})

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        (Path(kwargs["cwd"]) / "inference.log").write_text(_fake_stata_log(ws, contrasts))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ps.subprocess, "run", fake_run)

    result = assets.bootstrap_with_clean_log(ps, frame, ws, contrasts, tmp_path, False)

    log = (tmp_path / "inference.log").read_text()
    assert SERIAL_RE.search(log) is None
    for private in PRIVATE:
        assert private not in log
    assert "PKGEST H1_r" in log
    assert result["H1_r"]["estimate_stata"] == pytest.approx(contrasts["H1_r"]["estimate"])


def test_factorial_wcb_log_is_written_without_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = {"H1_r": np.eye(18)[1]}
    frame = pd.DataFrame({"success": [1, 0], "task": ["a", "b"], "cell": [1, 2]})

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        (Path(kwargs["cwd"]) / "factorial_wcb.log").write_text(
            STATA14_HEADER
            + "LINCOM_EST H1_r = .25\n"
            + f"BOOTTEST_P H1_r = {200 / 9999:.15g}\n"
            + f"BOOTTEST_P J = {200 / 9999:.15g}\n"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(fa.subprocess, "run", fake_run)

    parsed = fa.run_stata_wcb(frame, weights, tmp_path, {"J": ("H1_r",)})

    log = (tmp_path / "factorial_wcb.log").read_text()
    assert SERIAL_RE.search(log) is None
    assert "EXAMPLE OWNER" not in log
    assert parsed["H1_r"]["estimate_stata"] == pytest.approx(0.25)


def test_tracked_inference_logs_have_no_license_serial() -> None:
    logs = sorted((PROJECT_ROOT / "data/analysis/main/data/inference").glob("*/inference.log"))

    assert logs
    for path in logs:
        text = path.read_text(errors="replace")
        assert SERIAL_RE.search(text) is None, path
        assert "Licensed to:" not in text, path


def test_main_trace_result_and_repair_request_drop_license(tmp_path: Path) -> None:
    import build_run_traces as traces

    run = tmp_path / "run"
    run.mkdir()
    (run / "prompt_attempt1.txt").write_text("Initial\n")
    (run / "prompt_attempt2.txt").write_text("The script failed. Log:\n" + STATA19_HEADER + ". regress y x\n")
    # Later repair requests quote the execution record as JSON, so the log
    # lines arrive with escaped line breaks.
    (run / "prompt_attempt3.txt").write_text(
        "Execution record:\n" + json.dumps({"stata_log_tail": STATA14_HEADER + STATA19_HEADER + ". list\n"})
    )
    (run / "run_record.json").write_text(
        json.dumps({"calls": [{"text": "first"}, {"text": "second"}, {"text": "last\n"}], "executions": [{}, {}, {}]})
    )
    output = run / "final.log"
    output.write_text(STATA19_HEADER + ". display 1\n1\n")
    row = {
        "record_path": str(run / "run_record.json"), "output_path": str(output),
        "platform": "stata", "interface": "self_repair", "prompt": "zero_shot",
        "position_id": "example", "primary": "True",
    }

    combo = traces.build_main_combination(row, 1)

    payload = json.dumps(combo["flow"])
    for private in PRIVATE:
        assert private not in payload
    assert ". display 1\\n1\\n" in payload
    assert ". regress y x" in payload
    assert "Stata is running in batch mode." in payload
    assert ". list" in payload
    assert combo["flow"]["final"]["text"] == "last\n"
