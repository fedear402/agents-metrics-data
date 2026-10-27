"""Planted-fixture suite against the real post-hoc scoring entry point.

Each test builds a synthetic run directory in ``tmp_path`` and scores it with
``strict_posthoc.score_one_run``, the same function the deployed report uses.
The fixtures are the runnable regression suite for the 2026-07-30 scorer fixes
(review finding F4):

- integrity: a FULL-precision hard-coded copy of the reference must fail, not
  only the rounded copy the stored truncated literals catch exactly;
- method: the mandatory task-id string (``ols_robust`` contains ``robust``)
  must not satisfy method-evidence groups vacuously;
- method: genuine ``vce(robust)`` evidence must survive the task-id masking;
- a documented known limit: a wrong-estimator run whose numbers land inside
  tolerance and whose evidence comes from comments still passes today.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import strict_posthoc  # noqa: E402
from tasks import TASKS  # noqa: E402

TASKS_BY_ID = {task.task_id: task for task in TASKS}
REFERENCES = strict_posthoc.load_references()


def ref_rows(task_id: str) -> list[dict[str, Any]]:
    task = TASKS_BY_ID[task_id]
    return json.loads(json.dumps(REFERENCES[task.reference_key]["estimates"]))


def make_record(task_id: str, platform: str) -> dict[str, Any]:
    return {
        "cell_id": f"{task_id}__{platform}__zero_shot__no_feedback__rep00",
        "task_id": task_id,
        "platform": platform,
        "exec_mode": "no_feedback",
        "prompt_mode": "zero_shot",
        "rep": 0,
        "prompt_sha256": "ab" * 32,
        "claude": {"ok": True, "total_cost_usd": 0.1, "usage": {"input_tokens": 1}},
        "execution_success": True,
    }


def make_result(
    task_id: str,
    platform: str,
    estimates: list[dict[str, Any]],
    method_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "platform": platform,
        "estimates": estimates,
        "method_metadata": method_metadata,
        "diagnostics": {},
    }


def score_planted_run(
    tmp_path: Path,
    task_id: str,
    platform: str,
    script_name: str,
    script_text: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build a synthetic run dir and score it with the real entry point."""
    task = TASKS_BY_ID[task_id]
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "runs" / "cell"
    data_dir.mkdir(exist_ok=True)
    run_dir.mkdir(parents=True)
    (data_dir / task.data_file).write_text("x,y\n1,2\n")
    (run_dir / "input.csv").write_text("x,y\n1,2\n")
    (run_dir / script_name).write_text(script_text)
    (run_dir / "result.json").write_text(json.dumps(result))
    return strict_posthoc.score_one_run(
        run_dir,
        make_record(task_id, platform),
        TASKS_BY_ID,
        REFERENCES,
        data_dir,
    )


LEGIT_ROBUST_SCRIPT = """\
import json

import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("input.csv")
X = sm.add_constant(df[["x1", "x2"]])
fit = sm.OLS(df["y_lin"], X).fit(cov_type="HC1")
rows = [
    {
        "term": term,
        "estimate": float(fit.params[term]),
        "se": float(fit.bse[term]),
        "n_obs": int(fit.nobs),
    }
    for term in ["const", "x1", "x2"]
]
out = {
    "task_id": "ols_robust",
    "platform": "python",
    "estimates": rows,
    "method_metadata": {"estimator": "OLS", "vcov": "HC1"},
    "diagnostics": {},
}
with open("result.json", "w") as fh:
    json.dump(out, fh)
"""


def test_fixture_a_legitimate_script_passes_all_gates(tmp_path: Path) -> None:
    score = score_planted_run(
        tmp_path,
        "ols_robust",
        "python",
        "run.py",
        LEGIT_ROBUST_SCRIPT,
        make_result(
            "ols_robust",
            "python",
            ref_rows("ols_robust"),
            {"estimator": "OLS", "vcov": "HC1"},
        ),
    )

    for gate in ("execution", "artifact", "numeric", "method", "integrity"):
        assert score["gates"][gate]["pass"] is True, (gate, score["gates"][gate])
    for criterion, payload in score["criteria"].items():
        assert payload["pass"] is True, (criterion, payload)
    assert score["task_success"] is True
    assert score["strict_success"] is True


def test_fixture_b_full_precision_hardcode_fails_integrity(tmp_path: Path) -> None:
    # Regression for fix 1 (review F4): the stored literals are 6-12
    # significant-digit forms, so a script hard-coding the reference at FULL
    # precision contains none of them verbatim. The prefix rule must catch it.
    task = TASKS_BY_ID["ols_robust"]
    rows = ref_rows("ols_robust")
    literals = strict_posthoc.reference_number_literals(REFERENCES[task.reference_key])
    row_lines = []
    for row in rows:
        estimate = repr(float(row["estimate"]))
        se = repr(float(row["se"]))
        # The exact-match channel cannot fire: no stored literal equals the
        # full-precision string, so a failure proves the prefix rule works.
        assert estimate not in literals
        assert se not in literals
        row_lines.append(
            f'    {{"term": "{row["term"]}", "estimate": {estimate}, '
            f'"se": {se}, "n_obs": 2000}},'
        )
    script_text = (
        "import json\n\n"
        "# estimator: OLS, vcov: HC1\n"
        "rows = [\n" + "\n".join(row_lines) + "\n]\n"
        'out = {"task_id": "ols_robust", "platform": "python",\n'
        '       "estimates": rows,\n'
        '       "method_metadata": {"estimator": "OLS", "vcov": "HC1"},\n'
        '       "diagnostics": {}}\n'
        'with open("result.json", "w") as fh:\n'
        "    json.dump(out, fh)\n"
    )

    score = score_planted_run(
        tmp_path,
        "ols_robust",
        "python",
        "run.py",
        script_text,
        make_result(
            "ols_robust",
            "python",
            rows,
            {"estimator": "OLS", "vcov": "HC1"},
        ),
    )

    integrity = score["gates"]["integrity"]
    assert integrity["pass"] is False
    assert any("reference numeric literal" in note for note in integrity["notes"])
    assert score["criteria"]["submission_validity"]["pass"] is False
    assert score["task_success"] is False


def test_fixture_c_truncated_hardcode_still_fails_integrity(tmp_path: Path) -> None:
    # Existing behavior preserved: a rounded 10-significant-digit copy of the
    # x1 reference estimate matches a stored literal exactly.
    script_text = (
        "import json\n\n"
        "# estimator: OLS, vcov: HC1\n"
        "beta_x1 = 0.4926639841\n"
        'out = {"task_id": "ols_robust", "platform": "python"}\n'
        'with open("result.json", "w") as fh:\n'
        "    json.dump(out, fh)\n"
    )

    score = score_planted_run(
        tmp_path,
        "ols_robust",
        "python",
        "run.py",
        script_text,
        make_result(
            "ols_robust",
            "python",
            ref_rows("ols_robust"),
            {"estimator": "OLS", "vcov": "HC1"},
        ),
    )

    integrity = score["gates"]["integrity"]
    assert integrity["pass"] is False
    assert any("0.4926639841" in note for note in integrity["notes"])


CLASSICAL_SE_SCRIPT = """\
import json

import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("input.csv")
X = sm.add_constant(df[["x1", "x2"]])
fit = sm.OLS(df["y_lin"], X).fit()
rows = [
    {
        "term": term,
        "estimate": float(fit.params[term]),
        "se": float(fit.bse[term]),
        "n_obs": int(fit.nobs),
    }
    for term in ["const", "x1", "x2"]
]
out = {
    "task_id": "ols_robust",
    "platform": "python",
    "estimates": rows,
    "method_metadata": {"estimator": "OLS", "vcov": "classical"},
    "diagnostics": {},
}
with open("result.json", "w") as fh:
    json.dump(out, fh)
"""


def test_fixture_d_classical_se_script_fails_method_gate(tmp_path: Path) -> None:
    # Regression for fix 2 (review F4): an honestly labeled classical-SE script
    # on the robust-inference task, whose only "robust" text is the mandatory
    # task id, must fail the robust-evidence group.
    lowered = CLASSICAL_SE_SCRIPT.lower()
    assert lowered.count("robust") == lowered.count("ols_robust")

    score = score_planted_run(
        tmp_path,
        "ols_robust",
        "python",
        "run.py",
        CLASSICAL_SE_SCRIPT,
        make_result(
            "ols_robust",
            "python",
            ref_rows("ols_robust"),
            {"estimator": "OLS", "vcov": "classical"},
        ),
    )

    method = score["gates"]["method"]
    assert method["pass"] is False
    assert any("missing method evidence group 2" in note for note in method["notes"])
    assert score["criteria"]["specification_correctness"]["pass"] is False
    assert score["task_success"] is False


GENUINE_ROBUST_STATA_SCRIPT = """\
import delimited "input.csv", clear
regress y_lin x1 x2, vce(robust)
local n_obs = e(N)
file open jf using "result.json", write replace
file write jf `"{"task_id": "ols_robust", "platform": "stata","' _n
file write jf `" "estimates": [],"' _n
file write jf `" "method_metadata": {"estimator": "OLS", "vcov": "robust"},"' _n
file write jf `" "diagnostics": {}}"' _n
file close jf
"""


def test_fixture_e_genuine_robust_evidence_survives_masking(tmp_path: Path) -> None:
    # No false positive from the task-id masking: vce(robust) is real code
    # evidence and must keep satisfying the robust-evidence group even though
    # the script also declares the task id.
    score = score_planted_run(
        tmp_path,
        "ols_robust",
        "stata",
        "run.do",
        GENUINE_ROBUST_STATA_SCRIPT,
        make_result(
            "ols_robust",
            "stata",
            ref_rows("ols_robust"),
            {"estimator": "OLS", "vcov": "robust"},
        ),
    )

    assert score["gates"]["method"]["pass"] is True, score["gates"]["method"]
    assert score["criteria"]["specification_correctness"]["pass"] is True
    assert score["task_success"] is True


UNCLUSTERED_TWOWAY_FE_SCRIPT = """\
import json

import pandas as pd
from linearmodels import PanelOLS

df = pd.read_csv("input.csv").set_index(["firm", "year"])
m = PanelOLS(df["y_twocluster"], df[["x"]], entity_effects=True, time_effects=True)
res = m.fit()  # NOTE: we do NOT cluster the standard errors by firm or year
out = {
    "task_id": "twoway_fe",
    "platform": "python",
    "estimates": [
        {
            "term": "x",
            "estimate": float(res.params["x"]),
            "se": float(res.std_errors["x"]),
            "n_obs": int(res.nobs),
        }
    ],
    "method_metadata": {"estimator": "PanelOLS", "vcov": "unadjusted"},
    "diagnostics": {},
}
with open("result.json", "w") as fh:
    json.dump(out, fh)
"""


def test_fixture_f_wrong_estimator_within_tolerance_passes_today(
    tmp_path: Path,
) -> None:
    # KNOWN LIMIT, documented on purpose: this fixture is an UNCLUSTERED
    # two-way FE run (wrong variance estimator for the task) and the scorer
    # passes it today. Two mechanisms, both measured in review finding F4:
    # 1. the unclustered-vs-clustered SE gap on this near-homoskedastic draw
    #    (about 1.9e-3) sits inside the SE_ABS_TOL = 5e-3 absolute floor, so
    #    the numeric gate cannot distinguish the estimators; and
    # 2. require-evidence groups scan raw text including comments, so the
    #    comment "we do NOT cluster ..." satisfies the cluster-evidence group
    #    (only reject patterns are comment-stripped).
    # The asserts below pin what the scorer actually does; they do not claim
    # the scorer catches this class of submission.
    rows = ref_rows("twoway_fe")
    for row in rows:
        if row.get("se") is not None:
            row["se"] = float(row["se"]) + 1.9e-3  # measured unclustered gap

    score = score_planted_run(
        tmp_path,
        "twoway_fe",
        "python",
        "run.py",
        UNCLUSTERED_TWOWAY_FE_SCRIPT,
        make_result(
            "twoway_fe",
            "python",
            rows,
            {"estimator": "PanelOLS", "vcov": "unadjusted"},
        ),
    )

    assert score["gates"]["method"]["pass"] is True, score["gates"]["method"]
    assert score["gates"]["numeric"]["pass"] is True, score["gates"]["numeric"]
    assert score["task_success"] is True


def test_scoring_version_stamp_is_bumped() -> None:
    assert strict_posthoc.SCORING_VERSION == "strict_posthoc_2026-08-03"
    report = strict_posthoc.build_report([], references=REFERENCES)
    assert report["scoring_version"] == "strict_posthoc_2026-08-03"
