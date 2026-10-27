"""Tests for the "channel_se" rule in code/paper/dataset.py.

The rule is the output-channel outcome plus a standard-error check in the nine
tasks whose named decision is the standard error. The SE conventions and
tolerances come from variant V3 of code/benchmark/rescore_v2.py
(data/analysis/rescore_v2/scoring_spec.json). Since decision record
2026-09-25_se-scored-primary-outcome.md, "channel_se" is the default and
canonical rule. "channel" and "log" stay available. All fixtures are synthetic.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("paper_dataset_se", ROOT / "code/paper/dataset.py")
dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dataset)

CATALOG = ROOT / "data/production/main/catalog.json"
SCORING_SPEC = ROOT / "data/analysis/rescore_v2/scoring_spec.json"
if not CATALOG.is_file() or not SCORING_SPEC.is_file():
    pytest.skip("task catalog or scoring spec not on disk", allow_module_level=True)
CAT = json.loads(CATALOG.read_text())
TWC = CAT["two_way_cluster"]
OAX = CAT["broad_oaxaca"]


def answer_lines(spec, se=None, override=None):
    """ANSWER lines at the reference estimates, with the reference SE unless given."""
    lines = []
    for row in spec["reference"]["estimates"]:
        term = row["term"]
        value = (override or {}).get(term, row["estimate"])
        s = (se or {}).get(term, row.get("se"))
        s_text = "." if s is None else f"{s:.8f}"
        lines.append(f"ANSWER: {term} estimate={value:.8f} se={s_text} n_obs=100\n")
    return "".join(lines)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    return dataset.analysis.runtime.load_runtime(tmp_path_factory.mktemp("scoring")).c


def make_run(tmp_path, log_text, files=None, stdout=""):
    """A saved Stata attempt folder with a final execution window at 'now'."""
    attempt = tmp_path / "attempt_0001"
    workspace = attempt / "final_workspace"
    workspace.mkdir(parents=True)
    log = attempt / "printed_output.txt"
    log.write_text(log_text)
    written = [log]
    for name, text in (files or {}).items():
        (workspace / name).write_text(text)
        written.append(workspace / name)
    (attempt / "execution_attempt1.stdout").write_text(stdout)
    (attempt / "execution_attempt1.json").write_text("{}")
    written += [attempt / "execution_attempt1.stdout", attempt / "execution_attempt1.json"]
    now = time.time()
    for path in written:
        os.utime(path, (now, now))
    record = dict(executions=[dict(wall_seconds=1.0, stdout="")])
    record_path = attempt / "run_record.json"
    record_path.write_text(json.dumps(record))
    return log, record_path, record


def score(runtime, output, record_path, record, spec, rule, platform="stata"):
    return dataset.score_output(output, record_path, record, platform, spec, runtime, output_rule=rule)


def test_se_rule_is_the_default_and_other_rules_remain():
    assert dataset.OUTPUT_RULE == "channel_se"
    assert set(dataset.OUTPUT_RULES) == {"channel", "log", "channel_se"}


def test_default_scoring_checks_the_se(runtime, tmp_path):
    log, rp, rec = make_run(tmp_path, answer_lines(TWC, se={"x": 0.0100}))
    result = dataset.score_output(log, rp, rec, "stata", TWC, runtime)
    assert result["output_rule"] == "channel_se"
    assert not result["passed"]
    assert result["se_failed"] == ["x"]
    assert score(runtime, log, rp, rec, TWC, "channel")["passed"]


def test_default_build_argument_is_the_se_rule():
    import inspect
    assert inspect.signature(dataset.build).parameters["output_rule"].default == "channel_se"
    assert inspect.signature(dataset.score_output).parameters["output_rule"].default == "channel_se"


def load_rebuild():
    sys.path.insert(0, str(ROOT / "code/paper"))
    spec = importlib.util.spec_from_file_location("paper_rebuild_se", ROOT / "code/paper/rebuild.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rebuild_writes_the_canonical_folder_only_with_the_se_rule(tmp_path, capsys):
    rebuild = load_rebuild()
    args = rebuild.parse_args([])
    assert args.output_rule == "channel_se"
    assert args.out == rebuild.dataset.OUT.resolve()
    for rule in ("channel", "log"):
        with pytest.raises(SystemExit) as stop:
            rebuild.parse_args(["--output-rule", rule])
        assert stop.value.code == 2
        assert "separate --out folder" in capsys.readouterr().err
        args = rebuild.parse_args(["--output-rule", rule, "--out", str(tmp_path)])
        assert args.output_rule == rule and args.out == tmp_path.resolve()


def test_reference_se_passes(runtime, tmp_path):
    log, rp, rec = make_run(tmp_path, answer_lines(TWC))
    result = score(runtime, log, rp, rec, TWC, "channel_se")
    assert result["passed"]
    assert result["se_checked"] == ["const", "x"]
    assert result["se_failed"] == []


def test_wrong_se_fails_only_under_channel_se(runtime, tmp_path):
    # One-way clustering on firm gives a smaller SE. The estimate is unchanged.
    log, rp, rec = make_run(tmp_path, answer_lines(TWC, se={"x": 0.0100}))
    assert score(runtime, log, rp, rec, TWC, "channel")["passed"]
    result = score(runtime, log, rp, rec, TWC, "channel_se")
    assert not result["passed"]
    assert result["se_failed"] == ["x"]
    assert any("x: standard error" in note for note in result["notes"])


def test_missing_se_fails(runtime, tmp_path):
    log, rp, rec = make_run(tmp_path, answer_lines(TWC, se={"const": None}))
    result = score(runtime, log, rp, rec, TWC, "channel_se")
    assert not result["passed"]
    assert result["se_failed"] == ["const"]


def test_accepted_alternative_convention_passes(runtime, tmp_path):
    # CR0 components are an accepted convention in the V3 specification.
    log, rp, rec = make_run(tmp_path, answer_lines(TWC, se={"const": 0.14631300292389593, "x": 0.01564593827933807}))
    assert score(runtime, log, rp, rec, TWC, "channel_se")["passed"]


def test_se_tolerance_is_the_v3_tolerance(runtime, tmp_path):
    ref = TWC["reference"]["estimates"][1]["se"]
    inside, outside = ref * 1.004, ref * 1.02
    log, rp, rec = make_run(tmp_path, answer_lines(TWC, se={"x": inside}))
    assert score(runtime, log, rp, rec, TWC, "channel_se")["passed"]
    other = tmp_path / "second"
    other.mkdir()
    log, rp, rec = make_run(other, answer_lines(TWC, se={"x": outside}))
    assert not score(runtime, log, rp, rec, TWC, "channel_se")["passed"]


def test_estimate_failure_still_fails(runtime, tmp_path):
    log, rp, rec = make_run(tmp_path, answer_lines(TWC, override={"x": 0.9}))
    assert not score(runtime, log, rp, rec, TWC, "channel_se")["passed"]


def test_task_without_se_decision_equals_channel_rule(runtime, tmp_path):
    lines = answer_lines(OAX).replace("se=", "se=9")
    log, rp, rec = make_run(tmp_path, lines)
    channel = score(runtime, log, rp, rec, OAX, "channel")
    se_rule = score(runtime, log, rp, rec, OAX, "channel_se")
    assert channel["passed"] is se_rule["passed"] is True
    assert se_rule["se_checked"] == []


def test_se_on_final_stdout_counts_for_stata(runtime, tmp_path):
    no_se = answer_lines(TWC, se={"const": None, "x": None})
    log, rp, rec = make_run(tmp_path, no_se, stdout=answer_lines(TWC))
    result = score(runtime, log, rp, rec, TWC, "channel_se")
    assert result["passed"]


def test_conflicting_se_across_channels_fails(runtime, tmp_path):
    log, rp, rec = make_run(tmp_path, answer_lines(TWC), files={"out.txt": answer_lines(TWC, se={"x": 0.0100})})
    assert score(runtime, log, rp, rec, TWC, "channel")["passed"]
    result = score(runtime, log, rp, rec, TWC, "channel_se")
    assert not result["passed"]
    assert result["se_conflicts"] == ["x"]


def test_json_result_se_is_scored(runtime, tmp_path):
    attempt = tmp_path / "attempt_0001"
    attempt.mkdir()
    rows = [dict(term=r["term"], estimate=r["estimate"], se=r["se"], n_obs=100) for r in TWC["reference"]["estimates"]]
    good = attempt / "result.json"
    good.write_text(json.dumps(dict(estimates=rows)))
    record = dict(executions=[])
    rp = attempt / "run_record.json"
    rp.write_text(json.dumps(record))
    assert score(runtime, good, rp, record, TWC, "channel_se", platform="python")["passed"]
    rows[1]["se"] = 0.0100
    good.write_text(json.dumps(dict(estimates=rows)))
    assert score(runtime, good, rp, record, TWC, "channel", platform="python")["passed"]
    assert not score(runtime, good, rp, record, TWC, "channel_se", platform="python")["passed"]


def test_rebuild_accepts_the_rule():
    text = (ROOT / "code/paper/rebuild.py").read_text()
    assert "choices=dataset.OUTPUT_RULES" in text


def test_rebuild_rejects_a_float_taller_than_the_page():
    """A float taller than the text block prints over the page number (Table D1, 2026-09-30)."""
    rebuild = load_rebuild()
    log = "LaTeX Warning: Float too large for page by 52.63342pt on input line 71.\nOverfull \\hbox (1pt too wide)"
    assert len(rebuild.log_problems(log)) == 2
    assert rebuild.log_problems("LaTeX Warning: Label(s) may have changed.") == []
