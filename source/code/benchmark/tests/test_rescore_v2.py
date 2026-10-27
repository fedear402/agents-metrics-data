"""Tests for the offline version 2 rescore (code/benchmark/rescore_v2.py).

The rule tests use synthetic values. The V0 test reads a fixed sample of the
selected runs and is skipped when the saved outputs are not on disk.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rescore_v2 as rs  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "data/analysis/main/data/catalog.json"


def spec_for(**task):
    return {"rel_tol": 1e-6, **task}


def catalog_spec(terms):
    return dict(estimands=list(terms), reference=dict(estimates=[dict(term=t, estimate=v) for t, v in terms.items()]))


# --------------------------------------------------------------------------- precision rule

def test_decimals_and_allowance():
    assert rs.decimals("0.1928") == 4
    assert rs.decimals("7.743e-06") == 9
    assert rs.decimals("12") == 0
    assert rs.allowance("0.1928") == pytest.approx(5e-5)
    assert rs.allowance("0.028549") == pytest.approx(5e-7)
    # Tokens with fewer than 4 decimal places get no allowance.
    assert rs.allowance("0") == 0.0
    assert rs.allowance("0.06") == 0.0
    assert rs.allowance(None) == 0.0


def test_precision_rule_accepts_four_decimal_rounding_only():
    cat = catalog_spec({"x": 0.192838})
    task = spec_for()
    rounded = {"x": rs.Value("0.1928", "0.1928")}
    assert rs.score_run("t", rounded, cat, task, precision=True).passed
    assert not rs.score_run("t", rounded, cat, task, precision=False).passed
    # A value that is wrong by more than the rounding bound still fails.
    wrong = {"x": rs.Value("0.1930", "0.1930")}
    assert not rs.score_run("t", wrong, cat, task, precision=True).passed
    # A zero printed as "0" is not rounding and fails.
    zero = {"x": rs.Value("0", "0")}
    assert not rs.score_run("t", zero, catalog_spec({"x": 0.11}), task, precision=True).passed


def test_precision_rule_marks_allowance_use():
    cat = catalog_spec({"x": 0.192838})
    score = rs.score_run("t", {"x": rs.Value("0.1928", "0.1928")}, cat, spec_for(), precision=True)
    assert score.checks[0].used_allowance


def test_json_tokens_keep_raw_text():
    raw = '{"estimates": [{"term": "x", "estimate": 0.1928, "se": 1.50e-2}]}'
    rows, rule = rs.json_rows(raw)
    assert rule == "none needed"
    assert rows["x"].est_token == "0.1928"
    assert rows["x"].se_token == "1.50e-2"
    assert rows["x"].number == pytest.approx(0.1928)


# --------------------------------------------------------------------------- exact integers and conventions

def test_exact_integer_terms():
    cat = catalog_spec({"F": 9.7385, "df_den": 196.0})
    task = spec_for(rel_tol=1e-4, terms={"df_den": {"exact_integer": True}})
    base = {"F": rs.Value(9.7385, "9.7385")}
    for got, ok in ((196, True), ("196.0", True), (195, False), (196.4, False), (True, False)):
        values = dict(base, df_den=rs.Value(got, str(got)))
        assert rs.score_run("anova", values, cat, task).passed is ok, got


def test_exact_integer_accepts_listed_alternative():
    cat = catalog_spec({"df_den": 193.0})
    task = spec_for(terms={"df_den": {"exact_integer": True, "accepted_values": [193.0, 386.0]}})
    assert rs.score_run("g", {"df_den": rs.Value(386, "386")}, cat, task).passed
    assert not rs.score_run("g", {"df_den": rs.Value(194, "194")}, cat, task).passed


def test_signed_compare_and_alternative_convention():
    cat = catalog_spec({"t": 2.985073})
    assert not rs.score_run("ttest_welch", {"t": rs.Value(-2.985073, "-2.985073")}, cat, spec_for(rel_tol=1e-4)).passed
    cat = catalog_spec({"irf": 0.0817413})
    task = spec_for(rel_tol=1e-3, terms={"irf": {"accepted_values": [0.0817413, 0.0807026]}})
    check = rs.score_run("var_irf", {"irf": rs.Value(0.0807026, "0.0807026")}, cat, task).checks[0]
    assert check.passed and check.matched_alternative


def test_se_scoring_only_in_v3_style_call():
    cat = catalog_spec({"x": 0.5})
    task = spec_for(se_scored=True, se_terms={"x": {"rel_tol": 1e-3, "accepted_values": [0.1]}})
    values = {"x": rs.Value(0.5, "0.5", 0.2, "0.2")}
    assert rs.score_run("t", values, cat, task).passed
    assert not rs.score_run("t", values, cat, task, score_se=True).passed
    values["x"].se, values["x"].se_token = 0.10005, "0.10005"
    assert rs.score_run("t", values, cat, task, score_se=True).passed


# --------------------------------------------------------------------------- channel rule

def test_channel_rule_recovers_missing_terms():
    primary = {}
    sources = [("final_workspace/output.txt", {"a": [rs.Value(1.25, "1.25")], "b": [rs.Value(2.0, "2.0")]})]
    merged, conflicts, recovered = rs.merge_channels(primary, sources, ["a", "b"])
    assert conflicts == []
    assert sorted(recovered) == ["a", "b"]
    assert merged["a"].source == "final_workspace/output.txt"


def test_channel_rule_conflict_fails():
    primary = {"a": rs.Value(1.25, "1.25")}
    sources = [("final_workspace/run2.log", {"a": [rs.Value(1.40, "1.40")]})]
    merged, conflicts, _ = rs.merge_channels(primary, sources, ["a"])
    assert conflicts == ["a"]
    assert "a" not in merged


def test_channel_rule_same_value_at_lower_precision_is_not_a_conflict():
    primary = {"a": rs.Value(0.192838, "0.192838")}
    sources = [("stdout", {"a": [rs.Value(0.1928, "0.1928")]})]
    merged, conflicts, _ = rs.merge_channels(primary, sources, ["a"])
    assert conflicts == []
    assert merged["a"].est_token == "0.192838"


def test_channel_rule_uses_only_final_execution_files(tmp_path):
    attempt = tmp_path / "attempt_0001"
    workspace = attempt / "final_workspace"
    workspace.mkdir(parents=True)
    old = workspace / "run2.log"
    old.write_text("ANSWER: a estimate=9.99 se=. n_obs=10\n")
    new = workspace / "output.txt"
    new.write_text("ANSWER: a estimate=1.25 se=. n_obs=10\n")
    log = workspace / "printed_output.txt"
    log.write_text("no answer lines\n")
    (attempt / "execution_attempt1.stdout").write_text("")
    (attempt / "execution_attempt1.json").write_text("{}")
    record = attempt / "run_record.json"
    record.write_text(json.dumps({"executions": [{"wall_seconds": 1.0, "stdout": ""}]}))
    now = time.time()
    os.utime(old, (now - 600, now - 600))
    for path in (new, log, attempt / "execution_attempt1.stdout", attempt / "execution_attempt1.json"):
        os.utime(path, (now, now))
    sources = rs.extra_sources(record, log, "stata", ["a"])
    assert [label for label, _ in sources] == ["final_workspace/output.txt"]


# --------------------------------------------------------------------------- V0 reproduction

@pytest.mark.skipif(not CATALOG.is_file(), reason="selected sample not on disk")
def test_v0_with_the_se_check_reproduces_stored_primary_for_fixed_sample():
    # The stored primary is the "channel_se" rule (decision record
    # 2026-09-25_se-scored-primary-outcome.md): the estimate check of V0 plus
    # the V3 standard-error check in the nine SE-decision tasks.
    import se_scoring
    runs = pd.read_csv(ROOT / "data/analysis/main/data/runs.csv")
    assert set(runs.output_rule) == {"channel_se"}
    catalog = json.loads(CATALOG.read_text())
    # Fixed sample: per task, the first two successes and the first two failures by position_id.
    sample = (runs.sort_values("position_id").groupby(["task", "primary"], group_keys=False).head(2))
    if not all(Path(p).is_file() for p in sample.record_path):
        pytest.skip("saved run records not on disk")
    score = rs.v0_scorer()
    se_only_failures = 0
    for row in sample.itertuples():
        passed, _, _, _ = score(Path(row.output_path), row.platform, catalog[row.task])
        record_path = Path(row.record_path)
        se = se_scoring.check(Path(row.output_path), record_path, json.loads(record_path.read_text()),
                              row.platform, catalog[row.task])
        assert (passed and not se["failed"]) == bool(row.primary), row.position_id
        se_only_failures += passed and bool(se["failed"])
    assert len(sample) >= 120
    # The sample exercises the SE check: some runs pass every estimate and fail on the SE.
    assert se_only_failures >= 1
