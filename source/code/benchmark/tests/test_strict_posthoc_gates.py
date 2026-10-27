"""Regression tests for the 2026-06-26 scorer fixes.

Covers:
- the att_gt method gate distinguishing a call from an assignment;
- compare_numeric skipping n_obs for staggered_did (ill-defined estimation sample);
- compare_numeric skipping se for arima_forecast (forecast SE not required).

Runnable with `pytest` or directly: `python3 test_strict_posthoc.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import strict_posthoc as sp

HERE = Path(__file__).resolve()
STAGGERED_EVIDENCE_BASE = "df.groupby(['cohort']).mean()  # never-treated control\n"


def test_att_gt_assignment_is_not_rejected():
    # A manual ATT(g,t) that binds a local named att_gt must pass the method gate.
    evidence = STAGGERED_EVIDENCE_BASE + "att_gt = delta_treated - delta_control\n"
    ok, notes = sp.score_method("staggered_did", evidence, HERE)
    assert ok, f"assignment att_gt wrongly rejected: {notes}"


def test_att_gt_call_is_rejected():
    # An actual R did::att_gt(...) call must still be rejected.
    evidence = STAGGERED_EVIDENCE_BASE + "res = att_gt(yname='y', tname='t', idname='id')\n"
    ok, notes = sp.score_method("staggered_did", evidence, HERE)
    assert not ok, "att_gt(...) call should be rejected"
    assert any("att_gt" in n for n in notes)


def test_feols_call_rejected_but_variable_ok():
    assert not sp.score_method("staggered_did", STAGGERED_EVIDENCE_BASE + "feols(y ~ x)", HERE)[0]
    assert sp.score_method("staggered_did", STAGGERED_EVIDENCE_BASE + "feols_result = 1", HERE)[0]


def _task(task_id, estimands, reference_key=None):
    return SimpleNamespace(task_id=task_id, estimands=estimands, reference_key=reference_key or task_id)


def test_compare_numeric_skips_nobs_for_staggered_did():
    refs = {"staggered_did": {"estimates": [{"term": "att", "estimate": 1.5886894029357215, "se": None, "n_obs": 2000}]}}
    est = {"att": {"term": "att", "estimate": 1.588689402935722, "se": None, "n_obs": 616}}  # right value, "wrong" n_obs
    ok, _, notes = sp.compare_numeric(est, _task("staggered_did", ["att"]), refs)
    assert ok, f"staggered_did should ignore n_obs: {notes}"


def test_compare_numeric_enforces_nobs_for_other_tasks():
    refs = {"gen": {"estimates": [{"term": "att", "estimate": 1.5886894029357215, "se": None, "n_obs": 2000}]}}
    est = {"att": {"term": "att", "estimate": 1.588689402935722, "se": None, "n_obs": 616}}
    ok, _, _ = sp.compare_numeric(est, _task("gen", ["att"]), refs)
    assert not ok, "non-skipped tasks must still enforce n_obs"


def test_compare_numeric_skips_se_for_arima():
    refs = {"arima_forecast": {"estimates": [{"term": "forecast", "estimate": 42.695, "se": 0.95, "n_obs": 249}]}}
    est = {"forecast": {"term": "forecast", "estimate": 42.695, "se": None, "n_obs": 249}}  # right value, no SE
    ok, _, notes = sp.compare_numeric(est, _task("arima_forecast", ["forecast"]), refs)
    assert ok, f"arima_forecast should not require se: {notes}"


def test_compare_numeric_enforces_se_for_other_tasks():
    refs = {"gen2": {"estimates": [{"term": "x", "estimate": 0.5, "se": 0.02, "n_obs": 100}]}}
    est = {"x": {"term": "x", "estimate": 0.5, "se": None, "n_obs": 100}}
    ok, _, _ = sp.compare_numeric(est, _task("gen2", ["x"]), refs)
    assert not ok, "non-skipped tasks must still require se when the reference has one"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
