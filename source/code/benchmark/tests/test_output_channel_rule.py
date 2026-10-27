"""Tests for the output-channel rule in the canonical outcome (code/paper/dataset.py).

The rule is in decision record 2026-09-24_output-channel-rule.md. A Stata run
also passes when its final execution wrote correct ANSWER lines to another file
or to stdout, provided that no two sources give conflicting values for a term.
The "log" rule reproduces the earlier outcome. All fixtures are synthetic.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("paper_dataset_channel", ROOT / "code/paper/dataset.py")
dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dataset)

CATALOG = ROOT / "data/production/main/catalog.json"
if not CATALOG.is_file():
    pytest.skip("task catalog not on disk", allow_module_level=True)
SPEC = json.loads(CATALOG.read_text())["broad_oaxaca"]
REFERENCE = {row["term"]: row["estimate"] for row in SPEC["reference"]["estimates"]}
assert set(REFERENCE) == {"gap", "explained", "unexplained", "explained_x"}
GOOD = "".join(f"ANSWER: {term} estimate={value:.6f} se=. n_obs=360\n" for term, value in REFERENCE.items())


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    return dataset.analysis.runtime.load_runtime(tmp_path_factory.mktemp("scoring")).c


def make_run(tmp_path, log_text="no answer lines\n", files=None, stdout="", old_files=None, window=True):
    """A saved Stata attempt folder with a final execution window at 'now'."""
    attempt = tmp_path / "attempt_0001"
    workspace = attempt / "final_workspace"
    workspace.mkdir(parents=True)
    log = attempt / "printed_output.txt"
    log.write_text(log_text)
    now = time.time()
    written = [log]
    for name, text in (files or {}).items():
        (workspace / name).write_text(text)
        written.append(workspace / name)
    for name, text in (old_files or {}).items():
        path = workspace / name
        path.write_text(text)
        os.utime(path, (now - 600, now - 600))
    if window:
        (attempt / "execution_attempt1.stdout").write_text(stdout)
        (attempt / "execution_attempt1.json").write_text("{}")
        written += [attempt / "execution_attempt1.stdout", attempt / "execution_attempt1.json"]
    for path in written:
        os.utime(path, (now, now))
    record = dict(executions=[dict(wall_seconds=1.0, stdout="")])
    record_path = attempt / "run_record.json"
    record_path.write_text(json.dumps(record))
    return log, record_path, record


def score(runtime, log, record_path, record, rule, platform="stata"):
    return dataset.score_output(log, record_path, record, platform, SPEC, runtime, output_rule=rule)


def test_answer_lines_in_another_final_file_pass_under_channel_rule(runtime, tmp_path):
    log, record_path, record = make_run(tmp_path, files={"output.txt": GOOD})
    assert not score(runtime, log, record_path, record, "log")["passed"]
    result = score(runtime, log, record_path, record, "channel")
    assert result["passed"]
    assert result["channel_recovered"] == ["explained", "explained_x", "gap", "unexplained"]
    assert result["channel_sources"] == ["final_workspace/output.txt"]
    assert result["delivered"]


def test_answer_lines_on_stdout_pass_under_channel_rule(runtime, tmp_path):
    log, record_path, record = make_run(tmp_path, stdout=GOOD)
    assert score(runtime, log, record_path, record, "channel")["passed"]
    assert not score(runtime, log, record_path, record, "log")["passed"]


def test_conflicting_sources_fail_the_run(runtime, tmp_path):
    wrong = GOOD.replace("0.572298", "0.900000")
    log, record_path, record = make_run(tmp_path, log_text=GOOD, files={"results.txt": wrong})
    assert score(runtime, log, record_path, record, "log")["passed"]
    result = score(runtime, log, record_path, record, "channel")
    assert not result["passed"]
    assert result["channel_conflicts"] == ["explained"]


def test_files_outside_the_final_execution_window_are_ignored(runtime, tmp_path):
    log, record_path, record = make_run(tmp_path, old_files={"output.txt": GOOD})
    result = score(runtime, log, record_path, record, "channel")
    assert not result["passed"]
    assert result["channel_sources"] == []


def test_workspace_files_need_a_recorded_final_execution_window(runtime, tmp_path):
    log, record_path, record = make_run(tmp_path, files={"output.txt": GOOD}, window=False)
    assert not score(runtime, log, record_path, record, "channel")["passed"]


def test_same_value_at_lower_precision_is_not_a_conflict(runtime, tmp_path):
    coarse = GOOD.replace("1.435517", "1.4355").replace("0.572298", "0.5723")
    log, record_path, record = make_run(tmp_path, log_text=GOOD, files={"output.txt": coarse})
    result = score(runtime, log, record_path, record, "channel")
    assert result["passed"]
    assert result["channel_conflicts"] == []


def test_without_extra_sources_the_channel_rule_equals_the_log_rule(runtime, tmp_path):
    log, record_path, record = make_run(tmp_path, log_text=GOOD.replace("1.435517", "2.0"))
    log_rule = score(runtime, log, record_path, record, "log")
    channel = score(runtime, log, record_path, record, "channel")
    assert log_rule["passed"] is channel["passed"] is False
    assert log_rule["values"] == channel["values"]
    assert log_rule["notes"] == channel["notes"]


def test_rule_applies_only_to_stata(runtime, tmp_path):
    log, record_path, record = make_run(tmp_path, files={"output.txt": GOOD})
    result = dataset.score_output(record_path.parent / "result.json", record_path, record, "python", SPEC, runtime,
                                  output_rule="channel")
    assert not result["passed"]
    assert result["channel_sources"] == []


def test_unknown_rule_is_refused(runtime, tmp_path):
    log, record_path, record = make_run(tmp_path)
    with pytest.raises(ValueError):
        score(runtime, log, record_path, record, "other")


def test_default_rule_adds_se_scoring_to_the_channel_rule():
    # Decision record 2026-09-25_se-scored-primary-outcome.md: the canonical
    # outcome is the channel rule plus the SE check. The channel and log rules
    # stay available for comparison.
    assert dataset.OUTPUT_RULE == "channel_se"
    assert set(dataset.OUTPUT_RULES) == {"channel", "log", "channel_se"}


def test_default_rule_keeps_the_channel_sources(runtime, tmp_path):
    log, record_path, record = make_run(tmp_path, files={"output.txt": GOOD})
    result = dataset.score_output(log, record_path, record, "stata", SPEC, runtime)
    assert result["output_rule"] == "channel_se"
    assert result["passed"]
    assert result["channel_sources"] == ["final_workspace/output.txt"]
    assert result["se_checked"] == []
