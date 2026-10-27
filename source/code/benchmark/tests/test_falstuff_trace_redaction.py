"""Regression tests for public run-trace redaction."""

from __future__ import annotations

import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "falstuff_site"))

import build_run_traces  # noqa: E402


PUBLIC_TRACE_FORBIDDEN = (
    "/private/",
    "/var/folders/",
    "/tmp/",
    "Serial number:",
    "Licensed to:",
    "Stata license:",
    "4905 Lakeway Drive",
    "College Station, Texas",
    "@stata.com",
    "Standard Edition",
)


def test_redact_replaces_private_temp_paths() -> None:
    source = (
        'File "/private/tmp/session-503/-Users-account-owner-LOCAL/'
        'scratchpad/testvenv/lib/python3.14/site-packages/pandas/io.py", line 42\n'
        "ValueError: bad input"
    )

    cleaned = build_run_traces.redact(source)

    assert "/private/" not in cleaned
    assert "account-owner" not in cleaned
    assert "<temp>" in cleaned
    assert "ValueError: bad input" in cleaned


def test_redact_replaces_account_owner_in_file_listing() -> None:
    source = "-rw-r--r--@ 1 account-owner  staff  479 Jun 14 23:17 result.json"

    cleaned = build_run_traces.redact(source)

    assert "account-owner" not in cleaned
    assert "<user>" in cleaned
    assert "result.json" in cleaned


def test_redact_removes_serial_and_licensee_lines() -> None:
    source = (
        "Model output before license details\n"
        "Serial number: SERIAL-REDACT-ME\n"
        "  Licensed to: ACCOUNT OWNER\n"
        "               TEST ORGANIZATION\n"
        "Coefficient = 0.492664\n"
    )

    cleaned = build_run_traces.redact(source)

    assert "SERIAL-REDACT-ME" not in cleaned
    assert "ACCOUNT OWNER" not in cleaned
    assert "TEST ORGANIZATION" not in cleaned
    assert "Model output before license details" in cleaned
    assert "Coefficient = 0.492664" in cleaned


def test_redact_removes_full_stata_banner_and_keeps_analysis() -> None:
    source = """[Stata log]
  ___  ____  ____  ____  ____ (R)
 /__    /   ____/   /   ____/      StataNow 19.5
___/   /   /___/   /   /___/       SE-Standard Edition

 Statistics and Data Science       Copyright 1985-2025 StataCorp LLC
                                   StataCorp
                                   4905 Lakeway Drive
                                   College Station, Texas 77845 USA
                                   800-782-8272        https://www.stata.com
                                   979-696-4600        service@stata.com

Stata license: Unlimited-user network, expiring 14 Jul 2027
Serial number: SERIAL-REDACT-ME
  Licensed to: ACCOUNT OWNER
               TEST ORGANIZATION

Notes:
      1. Stata is running in batch mode.
      2. Unicode is supported; see help unicode_advice.
      3. Maximum number of variables is set to 5,000 but can be increased;
          see help set_maxvar.

. do run.do

. regress y x1, vce(robust)

------------------------------------------------------------------------------
           y | Coefficient  std. err.      t    P>|t|
-------------+----------------------------------------------------------------
          x1 |    .492664     .022807   21.60   0.000
------------------------------------------------------------------------------
"""

    cleaned = build_run_traces.redact(source)

    for private_text in (
        "StataNow 19.5",
        "Standard Edition",
        "StataCorp",
        "4905 Lakeway Drive",
        "Stata license",
        "SERIAL-REDACT-ME",
        "ACCOUNT OWNER",
        "TEST ORGANIZATION",
        "Notes:",
    ):
        assert private_text not in cleaned
    assert "[Stata log]" in cleaned
    assert ". do run.do" in cleaned
    assert ". regress y x1, vce(robust)" in cleaned
    assert "x1 |    .492664" in cleaned


def test_redact_removes_truncated_stata_banner_fragments() -> None:
    fragments = (
        """[Stata log]
icense: Unlimited-user network, expiring 14 Jul 2026
               TEST ORGANIZATION

Notes:
      1. Stata is running in batch mode.
      2. Unicode is supported; see help unicode_advice.

. do run.do
. display "analysis retained"
analysis retained
""",
        """... output truncated ...
               TEST ORGANIZATION

Notes:
      1. Stata is running in batch mode.
      2. Unicode is supported; see help unicode_advice.

. do run.do
. display "analysis retained"
analysis retained
""",
        """... output truncated ...
   /___/       SE-Standard Edition

. do run.do
. display "analysis retained"
analysis retained
""",
        """... output truncated ...
Corp
                                   4905 Lakeway Drive
                                   College Station, Texas 77845 USA

Stata license: Unlimited-user network, expiring 14 Jul 2026
               TEST ORGANIZATION

Notes:
      1. Stata is running in batch mode.

. do run.do
. display "analysis retained"
analysis retained
""",
        """ht 1985-2025 StataCorp LLC
                                   StataCorp
                                   4905 Lakeway Drive
                                   College Station, Texas 77845 USA

Notes:
      1. Stata is running in batch mode.

. do run.do
. display "analysis retained"
analysis retained
""",
        """s 77845 USA
                                   800-782-8272        https://www.stata.com
                                   979-696-4600        service@stata.com

Notes:
      1. Stata is running in batch mode.

. do run.do
. display "analysis retained"
analysis retained
""",
    )

    for source in fragments:
        cleaned = build_run_traces.redact(source)

        assert "TEST ORGANIZATION" not in cleaned
        assert "Standard Edition" not in cleaned
        assert "Unlimited-user network" not in cleaned
        assert "4905 Lakeway Drive" not in cleaned
        assert ". do run.do" in cleaned
        assert "analysis retained" in cleaned


def test_redact_keeps_banner_words_in_analytic_stata_output() -> None:
    source = (
        '. display "StataCorp"\n'
        "StataCorp\n"
        '. display "StataNow 19.5"\n'
        "StataNow 19.5\n"
        ". regress y x1\n"
        "x1 | .492664\n"
    )

    assert build_run_traces.redact(source) == source


def test_run_report_redacts_banner_before_json_encoding(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    block = {
        "ok": False,
        "stata_log_tail": """[Stata log]
___  ____  ____  ____  ____
___/   /   /___/   /   /___/       SE-Standard Edition

                                   StataCorp
                                   4905 Lakeway Drive
                                   College Station, Texas 77845 USA

Stata license: Unlimited-user network
Serial number: SERIAL-REDACT-ME
  Licensed to: ACCOUNT OWNER
               TEST ORGANIZATION

Notes:
      1. Stata is running in batch mode.

. do run.do
. display "analysis retained"
analysis retained
""",
    }

    cleaned = build_run_traces.run_report_text(run_dir, block)

    for private_text in (
        "Standard Edition",
        "StataCorp",
        "4905 Lakeway Drive",
        "Stata license",
        "SERIAL-REDACT-ME",
        "ACCOUNT OWNER",
        "TEST ORGANIZATION",
    ):
        assert private_text not in cleaned
    assert "analysis retained" in cleaned


def test_generated_run_traces_contain_no_private_markers() -> None:
    trace_paths = sorted(
        path
        for model_id in build_run_traces.MODEL_CONFIG
        for path in (build_run_traces.TRACE_DIR / model_id).glob("*.js")
    )

    assert len(trace_paths) == 40 * len(build_run_traces.MODEL_CONFIG)
    for path in trace_paths:
        import json
        # The exact saved prompt and answer may name the configured interpreter.
        # Redaction applies to operational evidence, not these source texts.
        text = path.read_text(encoding="utf-8")
        payload_object = json.loads(text.split("\n", 2)[2].split(" = ", 1)[1].rstrip(";\n"))
        operational = []
        for combo in payload_object["combos"].values():
            flow = combo["flow"]
            operational.append(flow.get("executionText", ""))
            operational.extend(event["text"] for event in flow["process"]["events"] if event["kind"] in {"log", "tool"})
        payload = json.dumps(operational)
        has_home_path = re.search(r"/Users/[^/\\s]+", payload) is not None
        assert not has_home_path, path
        for marker in PUBLIC_TRACE_FORBIDDEN:
            contains_marker = marker in payload
            assert not contains_marker, (path, marker)


def test_generated_run_traces_contain_no_license_identifiers() -> None:
    """Results and repair requests also quote Stata logs. Check every field."""
    trace_paths = sorted(
        path
        for model_id in build_run_traces.MODEL_CONFIG
        for path in (build_run_traces.TRACE_DIR / model_id).glob("*.js")
    )

    assert len(trace_paths) == 40 * len(build_run_traces.MODEL_CONFIG)
    for path in trace_paths:
        text = path.read_text(encoding="utf-8")
        assert re.search(r"Serial number:\s*[0-9]{4,}", text) is None, path
        assert "Licensed to:" not in text, path
