"""Pins for the scorer tolerances, the numeric comparison boundaries, the
Claude and Codex command lines, and the strict_posthoc CLI defaults.

Every test records what the code does today. A later change to a pinned value
must edit this file on purpose. Source: the 2026-09-01 code re-audit
(``quality_reports/audits/2026-09-01_full_reaudit/CODE.md``), findings D09-02,
D09-03, D09-08, D01-2, D03-1, D06-1, D06-2, and next actions 11 and 13. Where a
behavior change is pending, the comment names the OPEN_ITEMS item.

No test executes a model. A fake replaces every subprocess call.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import cells  # noqa: E402
import codex_backend  # noqa: E402
import harness  # noqa: E402
import strict_posthoc as sp  # noqa: E402
import worker_loop  # noqa: E402
from tasks import ALL_TASKS, EXPLORATORY_TASKS, TASKS  # noqa: E402

PROJECT_ROOT = BENCHMARK_ROOT.parent.parent
TASK_IDS = {task.task_id for task in TASKS}
ALL_TASK_IDS = {task.task_id for task in ALL_TASKS}
# The model id the paper names. cells.run_haiku_cell and worker_loop rebind
# harness.MODEL to this value before any corpus call.
PINNED_MODEL = "claude-sonnet-4-6"
# Captured at collection time, before any test runs. cells.run_haiku_cell
# rebinds harness.MODEL in place and test_cells.py leaves it rebound, so the
# runtime value is order dependent after that file runs.
HARNESS_DEFAULT_MODEL = harness.MODEL


def task_by_id(task_id: str):
    return next(task for task in TASKS if task.task_id == task_id)


# --------------------------------------------------------------------------- #
# 1. Tolerance constants and task sets (D09-02, D01-2, D03-1)
# --------------------------------------------------------------------------- #


def test_tolerance_constants_equal_the_appendix_b_values() -> None:
    # Appendix B, Numerical tolerances: a coefficient matches within 1 percent
    # relative or 0.005 absolute, a standard error within 5 percent relative or
    # 0.005 absolute. D09-02 found that no test pins the four constants.
    assert sp.COEF_REL_TOL == 1e-2
    assert sp.COEF_ABS_TOL == 5e-3
    assert sp.SE_REL_TOL == 5e-2
    assert sp.SE_ABS_TOL == 5e-3


def test_no_other_tolerance_constant_exists() -> None:
    names = sorted(name for name in vars(sp) if name.endswith("_TOL"))
    assert names == ["COEF_ABS_TOL", "COEF_REL_TOL", "SE_ABS_TOL", "SE_REL_TOL"]


def test_live_harness_scorer_mirrors_the_posthoc_tolerances() -> None:
    assert harness.COEF_REL_TOL == sp.COEF_REL_TOL
    assert harness.COEF_ABS_TOL == sp.COEF_ABS_TOL
    assert harness.SE_REL_TOL == sp.SE_REL_TOL
    assert harness.SE_ABS_TOL == sp.SE_ABS_TOL


def test_skip_nobs_tasks_membership() -> None:
    assert sp.SKIP_NOBS_TASKS == {
        "staggered_did",
        "var_irf",
        "arima_forecast",
        "cox_hazard",
        "dynpanel_ab",
        "synth_weights_fixedv",
    }
    # synth_weights_fixedv lives in EXPLORATORY_TASKS, outside the main list.
    assert sp.SKIP_NOBS_TASKS <= ALL_TASK_IDS
    assert sp.SKIP_NOBS_TASKS - TASK_IDS == {"synth_weights_fixedv"}
    assert "synth_weights_fixedv" in {task.task_id for task in EXPLORATORY_TASKS}


def test_skip_nobs_tasks_inside_the_analysis_sample_match_appendix_a() -> None:
    import analysis_dataset as ad

    # Appendix A names five tasks whose sample size the scorer does not compare.
    # The sixth member, synth_weights_fixedv, is outside the 21 analyzed tasks.
    assert len(ad.ANALYSIS_TASK_IDS) == 21
    assert sp.SKIP_NOBS_TASKS & set(ad.ANALYSIS_TASK_IDS) == {
        "staggered_did",
        "var_irf",
        "arima_forecast",
        "cox_hazard",
        "dynpanel_ab",
    }
    assert "synth_weights_fixedv" not in ad.ANALYSIS_TASK_IDS


def test_skip_se_abs_compare_and_quarantine_sets() -> None:
    assert sp.SKIP_SE_TASKS == {"arima_forecast"}
    assert sp.ABS_COMPARE_ESTIMANDS == {("ttest_welch", "t")}
    assert sp.QUARANTINED_TASKS == {"wcb_few"}
    assert sp.SKIP_ESTIMANDS == set()
    assert sp.LEGACY_SKIP_ESTIMANDS == {("wcb_few", "p_wcb")}
    assert sp.EXACT_INTEGER_ESTIMANDS == {
        ("cooks_distance", f"rank{rank}_obs_id") for rank in range(1, 6)
    }
    assert sp.MULTIPLE_CONVENTION_ESTIMANDS == {("var_granger", "df_den"): (2.0,)}


def test_every_task_term_pair_in_the_sets_names_a_real_estimand() -> None:
    pairs = (
        set(sp.ABS_COMPARE_ESTIMANDS)
        | set(sp.LEGACY_SKIP_ESTIMANDS)
        | set(sp.EXACT_INTEGER_ESTIMANDS)
        | set(sp.MULTIPLE_CONVENTION_ESTIMANDS)
    )
    for task_id, term in pairs:
        assert term in task_by_id(task_id).estimands, (task_id, term)
    for task_id in sp.SKIP_SE_TASKS | sp.QUARANTINED_TASKS:
        assert task_id in TASK_IDS


# --------------------------------------------------------------------------- #
# 2. compare_numeric boundaries (D09-02, D01-2, D03-1)
# --------------------------------------------------------------------------- #


def _task(task_id: str, *terms: str) -> SimpleNamespace:
    return SimpleNamespace(task_id=task_id, estimands=list(terms), reference_key=task_id)


def _row(term: str, estimate, se=None, n_obs=None) -> dict:
    return {"term": term, "estimate": estimate, "se": se, "n_obs": n_obs}


def _compare(
    task_id: str,
    term: str,
    ref_estimate,
    got_estimate,
    ref_se=None,
    got_se=None,
    ref_n=None,
    got_n=None,
):
    """Score one synthetic term and return (passed, per_term_row, notes)."""

    references = {task_id: {"estimates": [_row(term, ref_estimate, ref_se, ref_n)]}}
    estimates = {term: _row(term, got_estimate, got_se, got_n)}
    passed, per_term, notes = sp.compare_numeric(estimates, _task(task_id, term), references)
    return passed, per_term.get(term), notes


# Reference 100.0 gives a coefficient tolerance of exactly 1.0, so the
# boundary arithmetic is exact in binary floating point.
@pytest.mark.parametrize(
    "got, expected",
    [
        (101.0, True),  # exactly at the tolerance, the operator is <=
        (99.0, True),  # exactly at the tolerance on the low side
        (100.9, True),  # 0.9 percent inside
        (99.1, True),
        (101.1, False),  # 1.1 percent outside
        (98.9, False),
        (100.0, True),
    ],
)
def test_coefficient_relative_tolerance_boundary(got: float, expected: bool) -> None:
    passed, row, notes = _compare("ols_robust", "x1", 100.0, got)
    assert row["coef_tol"] == 1.0
    assert row["coef_pass"] is expected
    assert passed is expected
    assert ("x1: estimate outside tolerance" in notes) is (not expected)


# Below a reference magnitude of 0.5 the absolute floor of 0.005 binds, so a
# small coefficient passes with a relative error far above 1 percent. D03-1
# and next action 16 in CODE.md describe the terms where the floor binds.
@pytest.mark.parametrize(
    "ref, got, expected",
    [
        (0.1, 0.104, True),  # 4 percent relative error passes under the floor
        (0.1, -0.1 + 0.204, True),
        (0.1, 0.106, False),
        (0.0, 0.004, True),  # a zero reference passes any value inside 0.005
        (0.0, -0.004, True),
        (0.0, 0.006, False),
        (0.4, 0.4049, True),
        (0.4, 0.4051, False),
    ],
)
def test_coefficient_absolute_floor_regime(ref: float, got: float, expected: bool) -> None:
    passed, row, _ = _compare("ols_robust", "x1", ref, got)
    assert row["coef_tol"] == 0.005
    assert row["coef_pass"] is expected
    assert passed is expected


def test_coefficient_tolerance_switches_from_floor_to_relative_at_half() -> None:
    # max(0.01 * |ref|, 0.005) equals 0.005 up to |ref| = 0.5 and grows after.
    _, row_small, _ = _compare("ols_robust", "x1", 0.5, 0.5)
    _, row_large, _ = _compare("ols_robust", "x1", 0.6, 0.6)
    _, row_neg, _ = _compare("ols_robust", "x1", -50.0, -50.0)
    assert row_small["coef_tol"] == 0.005
    assert row_large["coef_tol"] == pytest.approx(0.006)
    assert row_neg["coef_tol"] == 0.5


# Reference standard error 10.0 gives a tolerance of exactly 0.5.
@pytest.mark.parametrize(
    "got_se, expected",
    [
        (10.5, True),  # exactly at the tolerance
        (9.5, True),
        (10.49, True),  # 4.9 percent inside
        (10.51, False),  # 5.1 percent outside
        (9.49, False),
    ],
)
def test_standard_error_relative_tolerance_boundary(got_se: float, expected: bool) -> None:
    passed, row, notes = _compare("ols_robust", "x1", 1.0, 1.0, ref_se=10.0, got_se=got_se)
    assert row["coef_pass"] is True
    assert row["se_tol"] == 0.5
    assert row["se_pass"] is expected
    assert passed is expected
    assert ("x1: standard error outside tolerance or missing" in notes) is (not expected)


@pytest.mark.parametrize(
    "ref_se, got_se, expected",
    [
        (0.05, 0.054, True),  # 8 percent relative error passes under the floor
        (0.05, 0.056, False),
        (0.0, 0.004, True),
        (0.0, 0.006, False),
    ],
)
def test_standard_error_absolute_floor_regime(ref_se, got_se, expected: bool) -> None:
    passed, row, _ = _compare("ols_robust", "x1", 1.0, 1.0, ref_se=ref_se, got_se=got_se)
    assert row["se_tol"] == 0.005
    assert row["se_pass"] is expected
    assert passed is expected


def test_standard_error_is_not_scored_when_the_reference_has_none() -> None:
    for got_se in (None, 3.0, math.nan):
        passed, row, notes = _compare("ols_robust", "x1", 1.0, 1.0, ref_se=None, got_se=got_se)
        assert passed is True
        assert row["se_pass"] is None
        assert row["se_tol"] is None
        assert row["se_err"] is None
        assert notes == []


@pytest.mark.parametrize("got_se", [None, math.nan, math.inf, "n/a"])
def test_missing_or_non_finite_standard_error_fails_when_the_reference_has_one(got_se) -> None:
    passed, row, notes = _compare("ols_robust", "x1", 1.0, 1.0, ref_se=0.1, got_se=got_se)
    assert passed is False
    assert row["se_pass"] is False
    assert row["got_se"] is None
    assert "x1: missing finite standard error" in notes
    assert "x1: standard error outside tolerance or missing" in notes


def test_skip_se_task_ignores_the_reference_standard_error() -> None:
    passed, row, notes = _compare(
        "arima_forecast", "forecast", 42.0, 42.0, ref_se=0.9, got_se=None
    )
    assert passed is True
    assert row["se_pass"] is None
    assert notes == []


# The exact sample-size rule. D01-2 records that the paper does not state it
# and that 12 sample runs fail on this rule alone. The pin documents the rule
# as implemented: int(round(got)) must equal int(round(ref)).
@pytest.mark.parametrize(
    "got_n, expected",
    [
        (2000, True),
        (2000.0, True),
        (2000.4, True),  # rounds to the reference
        (1999.6, True),
        (1999, False),  # one observation off fails the run
        (2001, False),
        (1999.4, False),
        (None, False),
        (math.nan, False),
    ],
)
def test_n_obs_exact_match_rule(got_n, expected: bool) -> None:
    passed, row, notes = _compare(
        "ols_robust", "x1", 1.0, 1.0, ref_se=0.1, got_se=0.1, ref_n=2000, got_n=got_n
    )
    assert row["coef_pass"] is True
    assert row["se_pass"] is True
    assert row["n_obs_pass"] is expected
    assert passed is expected
    assert ("x1: n_obs mismatch" in notes) is (not expected)
    assert ("x1: sample size outside exact match" in notes) is (not expected)


def test_n_obs_is_not_checked_when_the_reference_has_none() -> None:
    passed, row, notes = _compare("ols_robust", "x1", 1.0, 1.0, ref_n=None, got_n=17)
    assert passed is True
    assert row["n_obs_pass"] is True
    assert row["ref_n_obs"] is None
    assert notes == []


def test_skip_nobs_task_accepts_any_sample_size() -> None:
    # cox_hazard: 2,000 subjects against 1,470 events, both conventions pass.
    passed, row, notes = _compare("cox_hazard", "hr_x", 1.5, 1.5, ref_n=2000, got_n=1470)
    assert passed is True
    assert row["n_obs_pass"] is True
    assert notes == []


def test_sign_flip_passes_only_for_the_magnitude_compared_term() -> None:
    # ttest_welch t is compared on magnitude because Stata orders the groups
    # the other way. The same flip on any other term fails.
    passed_t, row_t, _ = _compare("ttest_welch", "t", 2.5, -2.5)
    assert passed_t is True
    assert row_t["coef_pass"] is True
    assert row_t["coef_err"] == 0.0
    assert row_t["got_estimate"] == -2.5

    passed_x, row_x, notes_x = _compare("ols_robust", "x1", 2.5, -2.5)
    assert passed_x is False
    assert row_x["coef_pass"] is False
    assert row_x["coef_err"] == 5.0
    assert "x1: estimate outside tolerance" in notes_x


@pytest.mark.parametrize("got", [None, math.nan, math.inf, -math.inf, "abc", ""])
def test_non_finite_or_non_numeric_estimate_fails_with_a_short_row(got) -> None:
    passed, row, notes = _compare("ols_robust", "x1", 1.0, got)
    assert passed is False
    assert row == {"estimate_pass": False}
    assert notes == ["x1: non-finite estimate"]


def test_numeric_string_estimate_is_accepted() -> None:
    # number_or_none coerces through float(), so a numeric string scores.
    passed, row, notes = _compare("ols_robust", "x1", 1.0, "1.005", ref_n=10, got_n="10")
    assert passed is True
    assert row["got_estimate"] == 1.005
    assert row["got_n_obs"] == 10.0
    assert notes == []


def test_non_finite_reference_estimate_fails_the_term() -> None:
    passed, row, notes = _compare("ols_robust", "x1", math.nan, 1.0)
    assert passed is False
    assert row == {"estimate_pass": False}
    assert notes == ["x1: non-finite estimate"]


def test_missing_result_term_fails_without_a_per_term_row() -> None:
    references = {"ols_robust": {"estimates": [_row("x1", 1.0), _row("x2", 2.0)]}}
    estimates = {"x1": _row("x1", 1.0)}
    passed, per_term, notes = sp.compare_numeric(
        estimates, _task("ols_robust", "x1", "x2"), references
    )
    assert passed is False
    assert set(per_term) == {"x1"}
    assert per_term["x1"]["coef_pass"] is True
    assert notes == ["x2: missing result or reference row"]


def test_missing_reference_row_fails_the_term() -> None:
    references = {"ols_robust": {"estimates": [_row("x1", 1.0)]}}
    estimates = {"x1": _row("x1", 1.0), "x2": _row("x2", 2.0)}
    passed, per_term, notes = sp.compare_numeric(
        estimates, _task("ols_robust", "x1", "x2"), references
    )
    assert passed is False
    assert set(per_term) == {"x1"}
    assert notes == ["x2: missing result or reference row"]


def test_missing_reference_key_fails_before_any_term() -> None:
    passed, per_term, notes = sp.compare_numeric(
        {"x1": _row("x1", 1.0)}, _task("ols_robust", "x1"), {}
    )
    assert passed is False
    assert per_term == {}
    assert notes == ["reference key missing: ols_robust"]


def test_one_failing_term_fails_the_run_and_keeps_every_row() -> None:
    references = {"ols_robust": {"estimates": [_row("x1", 1.0), _row("x2", 2.0)]}}
    estimates = {"x1": _row("x1", 1.0), "x2": _row("x2", 2.5)}
    passed, per_term, notes = sp.compare_numeric(
        estimates, _task("ols_robust", "x1", "x2"), references
    )
    assert passed is False
    assert per_term["x1"]["coef_pass"] is True
    assert per_term["x2"]["coef_pass"] is False
    assert notes == ["x2: estimate outside tolerance"]


@pytest.mark.parametrize(
    "got, expected",
    [
        (17, True),
        (17.0, True),
        (18, False),  # one row off fails even inside the 1 percent tolerance
        (17.4, False),  # a non-integer never matches
        (16, False),
    ],
)
def test_exact_integer_estimands_use_a_zero_tolerance(got, expected: bool) -> None:
    passed, row, _ = _compare("cooks_distance", "rank1_obs_id", 17, got)
    assert row["coef_tol"] == 0.0
    assert row["coef_pass"] is expected
    assert passed is expected


@pytest.mark.parametrize(
    "got, expected",
    [
        (100.0, True),  # the reference convention
        (200.0, True),  # the documented 2x convention
        (202.0, True),  # inside 1 percent of the rescaled reference
        (203.0, False),
        (50.0, False),  # the retired 0.5x allowance no longer passes
        (150.0, False),
    ],
)
def test_var_granger_df_den_accepts_only_the_documented_multiplier(got, expected) -> None:
    passed, row, _ = _compare("var_granger", "df_den", 100.0, got)
    assert row["coef_pass"] is expected
    assert passed is expected
    if got == 200.0:
        assert row["coef_err"] == 0.0


# --------------------------------------------------------------------------- #
# 3. Claude argv snapshot (D09-03)
# --------------------------------------------------------------------------- #

# One "result" event serves both the json and the stream-json readers.
CLAUDE_STDOUT = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "result": "```python\nprint(1)\n```",
        "usage": {},
        "num_turns": 1,
    }
)

CHATBOT_ARGV = [
    "claude",
    "-p",
    "--model",
    PINNED_MODEL,
    "--output-format",
    "json",
    "--max-turns",
    "1",
    "--no-session-persistence",
    "--permission-mode",
    "default",
    "--disallowedTools",
    "Bash,Write,Edit,Read",
]

AGENT_ARGV = [
    "claude",
    "-p",
    "--model",
    PINNED_MODEL,
    "--output-format",
    "stream-json",
    "--max-turns",
    "12",
    "--no-session-persistence",
    "--permission-mode",
    "default",
    "--verbose",
    "--allowedTools",
    "Bash",
]


class _FakeCompleted:
    def __init__(self, stdout: str = "") -> None:
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


def _install_fake_run(monkeypatch, stdout: str) -> list[dict]:
    """Replace subprocess.run everywhere and record every call."""

    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        calls.append(
            {
                "cmd": list(cmd),
                "timeout": kwargs.get("timeout"),
                "cwd": kwargs.get("cwd"),
                "input": kwargs.get("input"),
                "stdin": kwargs.get("stdin"),
            }
        )
        return _FakeCompleted(stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _pin_claude_environment(monkeypatch) -> None:
    monkeypatch.setattr(harness, "claude_bin", lambda: "claude")
    monkeypatch.setattr(harness, "CLAUDE_EXTRA_CLI_ARGS", [])
    monkeypatch.setattr(harness, "TOOL_LOOP_TURN_CAP_ENABLED", True)
    monkeypatch.setattr(harness, "MODEL_CALL_TIMEOUT_OVERRIDE", None)


def _claude_calls(calls: list[dict]) -> list[dict]:
    return [call for call in calls if call["cmd"][:2] == ["claude", "-p"]]


def test_harness_timeout_turn_and_tool_constants() -> None:
    assert harness.TIMEOUT_NO_FEEDBACK == 180
    assert harness.TIMEOUT_TOOL == 600
    assert harness.TIMEOUT_AGENTIC == 1800
    assert harness.MAX_TURNS_TOOL == 12
    assert harness.MAX_TURNS_AGENTIC == 60
    assert harness.MAX_EXECUTIONS_REPAIR == 4  # one first attempt and three revisions
    assert harness.TOOL_LOOP_TURN_CAP_ENABLED is True
    assert harness.MODEL_CALL_TIMEOUT_OVERRIDE is None
    assert harness.model_call_timeout() == 180
    assert harness.PERMISSION_MODE == "default"
    assert harness.AGENTIC_PERMISSION_MODE == "default"
    assert harness.TOOL_LOOP_ALLOWED_TOOLS == ["Bash"]
    assert harness.NO_FEEDBACK_DISALLOWED_TOOLS == ["Bash", "Write", "Edit", "Read"]
    assert harness.AGENTIC_ALLOWED_TOOLS == "Read Write Edit Bash"
    assert harness.CLAUDE_EXTRA_CLI_ARGS == []
    assert harness.DEFAULT_EXEC_MODES == ["no_feedback", "self_repair", "tool_loop"]


def test_harness_default_model_is_the_bare_alias_the_worker_forbids() -> None:
    # D09-03 and the CODE.md Coverage review bullet on harness.py:50. The
    # module default is the floating alias "sonnet", which resolved to
    # claude-sonnet-5 on 2026-07-04. worker_loop refuses the alias and rebinds
    # harness.MODEL to the pin. cells.run_haiku_cell rebinds it per call. A
    # direct harness.run_cell call without a rebind still sends the alias.
    assert HARNESS_DEFAULT_MODEL == "sonnet"
    assert re.search(r'^MODEL = "sonnet"$', inspect.getsource(harness), re.MULTILINE)
    assert worker_loop.PINNED_MODEL == PINNED_MODEL
    assert worker_loop.FORBIDDEN_MODEL_ALIASES == {"sonnet", "opus", "haiku", "default"}
    assert HARNESS_DEFAULT_MODEL in worker_loop.FORBIDDEN_MODEL_ALIASES


def test_run_claude_chatbot_argv_snapshot(monkeypatch, tmp_path: Path) -> None:
    _pin_claude_environment(monkeypatch)
    monkeypatch.setattr(harness, "MODEL", PINNED_MODEL)
    calls = _install_fake_run(monkeypatch, CLAUDE_STDOUT)

    result = harness.run_claude(
        "PROMPT",
        cwd=tmp_path,
        allow_tools=False,
        max_turns=1,
        timeout=harness.model_call_timeout(),
    )

    assert calls[0]["cmd"] == CHATBOT_ARGV
    assert calls[0]["timeout"] == 180
    assert calls[0]["cwd"] == str(tmp_path)
    assert calls[0]["input"] == "PROMPT"
    assert result["ok"] is True
    assert result["command"] == CHATBOT_ARGV
    assert result["permission_mode"] == "default"
    assert result["allowed_tools"] == []
    assert result["disallowed_tools"] == ["Bash", "Write", "Edit", "Read"]


def test_run_claude_constrained_agent_argv_snapshot(monkeypatch, tmp_path: Path) -> None:
    _pin_claude_environment(monkeypatch)
    monkeypatch.setattr(harness, "MODEL", PINNED_MODEL)
    calls = _install_fake_run(monkeypatch, CLAUDE_STDOUT)

    result = harness.run_claude(
        "PROMPT",
        cwd=tmp_path,
        allow_tools=True,
        max_turns=harness.MAX_TURNS_TOOL,
        timeout=harness.TIMEOUT_TOOL,
        stream=True,
    )

    assert calls[0]["cmd"] == AGENT_ARGV
    assert calls[0]["timeout"] == 600
    assert result["ok"] is True
    assert result["output_format"] == "stream-json"
    assert result["allowed_tools"] == ["Bash"]
    assert result["disallowed_tools"] == []


def test_run_claude_without_a_rebind_sends_the_default_alias(monkeypatch, tmp_path) -> None:
    # The alias pin above, seen from the argv side.
    _pin_claude_environment(monkeypatch)
    monkeypatch.setattr(harness, "MODEL", HARNESS_DEFAULT_MODEL)
    calls = _install_fake_run(monkeypatch, CLAUDE_STDOUT)
    harness.run_claude("PROMPT", cwd=tmp_path, allow_tools=False, max_turns=1, timeout=5)
    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--model") + 1] == "sonnet"


def test_run_claude_omits_max_turns_when_none(monkeypatch, tmp_path: Path) -> None:
    # The limit re-run wave lifts the twelve-turn cap this way. Every corpus
    # call passes an integer.
    _pin_claude_environment(monkeypatch)
    monkeypatch.setattr(harness, "MODEL", PINNED_MODEL)
    calls = _install_fake_run(monkeypatch, CLAUDE_STDOUT)
    harness.run_claude(
        "PROMPT", cwd=tmp_path, allow_tools=True, max_turns=None, timeout=600, stream=True
    )
    expected = [item for item in AGENT_ARGV if item not in ("--max-turns", "12")]
    assert calls[0]["cmd"] == expected


def test_production_chatbot_path_sends_the_pin_and_both_timeouts(monkeypatch, tmp_path) -> None:
    # cells.run_haiku_cell is the production entry for the Sonnet arms. It
    # rebinds harness.MODEL and calls harness.run_cell. The fake returns a
    # script, so the second call is the script execution.
    task = task_by_id("ols_robust")
    _pin_claude_environment(monkeypatch)
    monkeypatch.setattr(harness, "MODEL", harness.MODEL)  # restore the default afterwards
    monkeypatch.setattr(cells.pilot_scoring, "rescore_record", lambda *a, **k: None)
    calls = _install_fake_run(monkeypatch, CLAUDE_STDOUT)

    record = cells.run_haiku_cell(
        task,
        "python",
        "zero_shot",
        "no_feedback",
        rep=0,
        overwrite=True,
        model=PINNED_MODEL,
        runs_dir=tmp_path,
    )

    cell_dir = tmp_path / "ols_robust__python__zero_shot__no_feedback__rep00"
    model_calls = _claude_calls(calls)
    assert len(model_calls) == 1
    assert model_calls[0]["cmd"] == CHATBOT_ARGV
    assert model_calls[0]["timeout"] == 180
    assert model_calls[0]["cwd"] == str(cell_dir)
    script_call = calls[calls.index(model_calls[0]) + 1]
    assert script_call["cmd"] == [sys.executable, str(cell_dir / "run.py")]
    assert script_call["timeout"] == 180
    assert record["model"] == PINNED_MODEL
    assert record["claude_timeout_seconds"] == 180
    assert record["claude_extra_cli_args"] == []
    assert record["arm"] == "sonnet_pilot"
    assert (cell_dir / "run_record.json").exists()


def test_production_self_repair_path_makes_one_call_and_three_revisions(monkeypatch, tmp_path):
    # Appendix B: the self-repair condition ends on a clean execution with the
    # result file, or after three revisions. The fake never writes result.json,
    # so every revision runs. Every model call is a one-turn chatbot call.
    task = task_by_id("ols_robust")
    _pin_claude_environment(monkeypatch)
    monkeypatch.setattr(harness, "MODEL", PINNED_MODEL)
    calls = _install_fake_run(monkeypatch, CLAUDE_STDOUT)

    harness.run_cell(task, "python", "zero_shot", "self_repair", rep=0, runs_dir=tmp_path)

    model_calls = _claude_calls(calls)
    assert len(model_calls) == harness.MAX_EXECUTIONS_REPAIR == 4
    assert all(call["cmd"] == CHATBOT_ARGV for call in model_calls)
    assert all(call["timeout"] == 180 for call in model_calls)
    script_calls = [call for call in calls if call["cmd"][0] == sys.executable]
    assert len(script_calls) == 4
    assert all(call["timeout"] == 180 for call in script_calls)


def test_production_constrained_agent_path_sends_the_agent_argv(monkeypatch, tmp_path) -> None:
    task = task_by_id("ols_robust")
    _pin_claude_environment(monkeypatch)
    monkeypatch.setattr(harness, "MODEL", PINNED_MODEL)
    calls = _install_fake_run(monkeypatch, CLAUDE_STDOUT)

    record = harness.run_cell(task, "python", "zero_shot", "tool_loop", rep=0, runs_dir=tmp_path)

    cell_dir = tmp_path / "ols_robust__python__zero_shot__tool_loop__rep00"
    model_calls = _claude_calls(calls)
    assert len(model_calls) == 1
    assert model_calls[0]["cmd"] == AGENT_ARGV
    assert model_calls[0]["timeout"] == 600
    assert model_calls[0]["cwd"] == str(cell_dir)
    assert record["model"] == PINNED_MODEL
    # The harness never executes a script itself in this condition.
    assert all(call["cmd"][0] != sys.executable for call in calls)
    assert (cell_dir / "agent_stream.jsonl").exists()


def test_agentic_claude_cmd_snapshot(monkeypatch, tmp_path: Path) -> None:
    # The full-agent condition is outside the paper. The pin keeps its command
    # visible so a change is deliberate.
    monkeypatch.setattr(harness, "claude_bin", lambda: "claude")
    assert harness.agentic_claude_cmd(tmp_path, model=PINNED_MODEL) == [
        "claude",
        "-p",
        "--model",
        PINNED_MODEL,
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--max-turns",
        "60",
        "--no-session-persistence",
        "--permission-mode",
        "default",
        "--allowedTools",
        "Read Write Edit Bash",
        "--add-dir",
        str(tmp_path),
    ]


def test_execute_script_argv_by_platform(monkeypatch, tmp_path: Path) -> None:
    calls = _install_fake_run(monkeypatch, "")
    script = tmp_path / "run.py"
    script.write_text("print(1)\n")

    harness.execute_script(script, "python", tmp_path, timeout=180)
    harness.execute_script(tmp_path / "run.R", "r", tmp_path, timeout=180)

    assert calls[0]["cmd"] == [sys.executable, str(script)]
    assert calls[1]["cmd"] == ["Rscript", "--vanilla", str(tmp_path / "run.R")]
    assert all(call["timeout"] == 180 and call["cwd"] == str(tmp_path) for call in calls)
    assert harness.execute_script(script, "julia", tmp_path, timeout=1) == {
        "ok": False,
        "error": "unknown platform julia",
    }


@pytest.mark.skipif(not harness.STATA_BIN.exists(), reason="no Stata binary on this host")
def test_execute_script_stata_argv(monkeypatch, tmp_path: Path) -> None:
    calls = _install_fake_run(monkeypatch, "")
    harness.execute_script(tmp_path / "run.do", "stata", tmp_path, timeout=180)
    batch_flag = "/e" if harness.os.name == "nt" else "-b"
    assert calls[0]["cmd"] == [str(harness.STATA_BIN), batch_flag, "do", str(tmp_path / "run.do")]


# --------------------------------------------------------------------------- #
# 4. Codex argv snapshot (D06-1, D06-2)
# --------------------------------------------------------------------------- #


def _install_codex_fake(monkeypatch) -> list[dict]:
    calls = _install_fake_run(monkeypatch, "")
    monkeypatch.setattr(codex_backend, "CODEX_BIN", "codex")
    monkeypatch.setattr(codex_backend, "codex_version", lambda: "codex-cli test")
    return calls


def _codex_argv(sandbox: str, cwd: Path, last_message_path: Path) -> list[str]:
    return [
        "codex",
        "exec",
        "--json",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--skip-git-repo-check",
        "-m",
        "gpt-5.4",
        "-c",
        "model_reasoning_effort=medium",
        "--sandbox",
        sandbox,
        "-C",
        str(cwd),
        "-o",
        str(last_message_path),
        "PROMPT",
    ]


def test_codex_defaults_and_timeout_constants() -> None:
    assert cells.DEFAULT_CODEX_MODEL == "gpt-5.4"
    assert cells.DEFAULT_CODEX_REASONING == "medium"
    # D06-1: Appendix B line 39 states that Codex runs under the same
    # 600-second limit as Sonnet. The code caps a chatbot or self-repair call
    # at 900 seconds and an agent session at 1,500 seconds, and every stored
    # Codex record carries these values. OPEN_ITEMS Fix 3 rewrites the
    # sentence. The pin records the code values.
    assert cells.DEFAULT_CODEX_TIMEOUT == 900
    assert cells.DEFAULT_CODEX_AGENT_TIMEOUT == 1500
    params = inspect.signature(cells.run_codex_cell).parameters
    assert params["codex_timeout"].default == 900
    assert params["codex_agent_timeout"].default == 1500
    assert codex_backend.BASE_FLAGS == [
        "exec",
        "--json",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--skip-git-repo-check",
    ]


def test_codex_sandbox_by_condition() -> None:
    assert codex_backend.sandbox_for("codex_generate") == "read-only"
    assert codex_backend.sandbox_for("codex_repair") == "read-only"
    assert codex_backend.sandbox_for("codex_agent") == "workspace-write"
    with pytest.raises(ValueError):
        codex_backend.sandbox_for("codex_chat")


@pytest.mark.parametrize(
    "condition, sandbox, timeout",
    [
        ("codex_generate", "read-only", 900),
        ("codex_repair", "read-only", 900),
        ("codex_agent", "workspace-write", 1500),
    ],
)
def test_run_codex_argv_snapshot(monkeypatch, tmp_path, condition, sandbox, timeout) -> None:
    # D06-2: the argv carries no flag that restricts Codex's own tools. Under
    # workspace-write the native file-change and web-search tools stay
    # available, and 156 analyzed agent runs used the file-change tool.
    # OPEN_ITEMS Fix 3 adds the disclosure. The pin records the argv as sent.
    calls = _install_codex_fake(monkeypatch)
    events_path = tmp_path / "codex_events.jsonl"
    last_path = tmp_path / "codex_last.txt"

    block = codex_backend.run_codex(
        "PROMPT",
        tmp_path,
        condition,
        cells.DEFAULT_CODEX_MODEL,
        cells.DEFAULT_CODEX_REASONING,
        timeout,
        events_path,
        last_path,
    )

    expected = _codex_argv(sandbox, tmp_path, last_path)
    assert calls[0]["cmd"] == expected
    assert calls[0]["timeout"] == timeout
    assert calls[0]["cwd"] == str(tmp_path)
    assert calls[0]["stdin"] is subprocess.DEVNULL
    assert block["argv"] == expected[:-1] + ["<PROMPT>"]
    assert block["sandbox"] == sandbox
    assert block["timeout_seconds"] == timeout
    assert block["model"] == "gpt-5.4"
    assert block["reasoning_effort"] == "medium"
    assert block["condition"] == condition
    assert block["cost_usd"] is None
    assert events_path.exists()


@pytest.mark.parametrize(
    "condition, timeout_key, expected",
    [
        ("codex_generate", "codex_timeout_seconds", 900),
        ("codex_repair", "codex_timeout_seconds", 900),
        ("codex_agent", "codex_agent_timeout_seconds", 1500),
    ],
)
def test_run_codex_cell_passes_the_default_timeouts(
    monkeypatch, tmp_path: Path, condition: str, timeout_key: str, expected: int
) -> None:
    task = task_by_id("ols_robust")
    seen: dict = {}

    def fake_run_codex(prompt, cwd, cond, model, reasoning, timeout, events, last):
        seen.update(condition=cond, model=model, reasoning=reasoning, timeout=timeout)
        return {
            "backend": "codex",
            "ok": True,
            "text": "no script in this reply",
            "provider_limit": False,
            "error_message": "",
            "num_tool_calls": 0,
            "used_tools": False,
        }

    monkeypatch.setattr(cells.codex_backend, "run_codex", fake_run_codex)
    monkeypatch.setattr(cells, "CODEX_TMP_ROOT", tmp_path / "codex_tmp")
    monkeypatch.setattr(
        cells.pilot_scoring,
        "canonical_score",
        lambda *a, **k: {"numeric_success": False, "notes": []},
    )

    record = cells.run_codex_cell(
        task,
        "python",
        "zero_shot",
        condition,
        0,
        cells.DEFAULT_CODEX_MODEL,
        cells.DEFAULT_CODEX_REASONING,
        overwrite=True,
        runs_dir=tmp_path / "arm",
    )

    assert seen == {
        "condition": condition,
        "model": "gpt-5.4",
        "reasoning": "medium",
        "timeout": expected,
    }
    assert record[timeout_key] == expected
    assert record["model"] == "gpt-5.4"
    assert record["reasoning_effort"] == "medium"
    assert record["arm"] == "codex_pilot"
    assert record["backend"] == "codex"


def test_codex_generator_note_is_appended_to_chatbot_and_repair_prompts(monkeypatch, tmp_path):
    # The read-only sandbox still lets Codex act as an agent, so the chatbot
    # and self-repair prompts carry an appended note. The agent prompt does not.
    task = task_by_id("ols_robust")
    prompts: dict[str, str] = {}

    def fake_run_codex(prompt, cwd, cond, *rest):
        prompts[cond] = prompt
        return {
            "backend": "codex",
            "ok": True,
            "text": "no script in this reply",
            "provider_limit": False,
            "error_message": "",
            "num_tool_calls": 0,
            "used_tools": False,
        }

    monkeypatch.setattr(cells.codex_backend, "run_codex", fake_run_codex)
    monkeypatch.setattr(cells, "CODEX_TMP_ROOT", tmp_path / "codex_tmp")
    monkeypatch.setattr(
        cells.pilot_scoring,
        "canonical_score",
        lambda *a, **k: {"numeric_success": False, "notes": []},
    )
    for condition in ("codex_generate", "codex_repair", "codex_agent"):
        cells.run_codex_cell(
            task, "python", "zero_shot", condition, 0, "gpt-5.4", "medium",
            overwrite=True, runs_dir=tmp_path / "arm",
        )

    note = cells.CODEX_GENERATOR_NOTE
    assert note
    assert prompts["codex_generate"].endswith(note)
    assert prompts["codex_repair"].endswith(note)
    assert prompts["codex_generate"] == harness.no_feedback_prompt(task, "python", "zero_shot") + note
    assert prompts["codex_agent"] == harness.tool_loop_prompt(task, "python", "zero_shot")
    assert not prompts["codex_agent"].endswith(note)


# --------------------------------------------------------------------------- #
# 5. strict_posthoc CLI defaults and scoring version (D09-08, OPEN_ITEMS Fix 1)
# --------------------------------------------------------------------------- #


def test_scoring_version_and_run_score_name() -> None:
    assert sp.SCORING_VERSION == "strict_posthoc_2026-08-03"
    assert sp.RUN_SCORE_NAME == "score.json"


def test_strict_posthoc_cli_defaults_target_the_stored_sonnet_zero_shot_arm(monkeypatch) -> None:
    # D09-08 and OPEN_ITEMS Fix 1. A bare invocation rewrites four derived
    # summaries under the stored arm's _working directory. Fix 1 asks for a
    # guard. This pin records today's defaults so the guard is a deliberate
    # change to this test.
    monkeypatch.setattr(sys, "argv", ["strict_posthoc.py"])
    args = sp.parse_args()
    assert sp.PROJECT_ROOT == PROJECT_ROOT
    assert args.runs_dir == PROJECT_ROOT / "data" / "runs" / "sonnet_zero_shot"
    assert args.out_dir == PROJECT_ROOT / "data" / "runs" / "sonnet_zero_shot" / "_working"
    assert args.data_dir == PROJECT_ROOT / "data" / "tasks" / "inputs"
    assert args.reference_dir == PROJECT_ROOT / "data" / "tasks" / "references"
    assert args.write_run_scores is False


def test_strict_posthoc_out_dir_does_not_follow_runs_dir(monkeypatch, tmp_path: Path) -> None:
    # A call that names --runs-dir and omits --out-dir still writes under the
    # stored Sonnet zero-shot arm. OPEN_ITEMS Fix 1.
    monkeypatch.setattr(sys, "argv", ["strict_posthoc.py", "--runs-dir", str(tmp_path)])
    args = sp.parse_args()
    assert args.runs_dir == tmp_path
    assert args.out_dir == sp.DEFAULT_OUT_DIR
    assert sp.DEFAULT_OUT_DIR == sp.DEFAULT_RUNS_DIR / "_working"
    assert sp.DEFAULT_RUNS_DIR == PROJECT_ROOT / "data" / "runs" / "sonnet_zero_shot"


def _write_minimal_record(runs_dir: Path) -> Path:
    cell = runs_dir / "ols_robust__python__zero_shot__no_feedback__rep00"
    cell.mkdir(parents=True)
    (cell / "run_record.json").write_text(
        json.dumps(
            {
                "cell_id": cell.name,
                "task_id": "ols_robust",
                "platform": "python",
                "prompt_mode": "zero_shot",
                "exec_mode": "no_feedback",
                "rep": 0,
                "execution_success": True,
                "scoring": {"numeric_success": True},
                "claude": {"ok": True, "total_cost_usd": 0.01, "usage": {}},
            }
        )
    )
    return cell


def test_strict_posthoc_main_with_explicit_dirs_writes_only_the_four_reports(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    runs_dir = tmp_path / "arm"
    out_dir = runs_dir / "_working"
    cell = _write_minimal_record(runs_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["strict_posthoc.py", "--runs-dir", str(runs_dir), "--out-dir", str(out_dir)],
    )

    sp.main()

    assert sorted(path.name for path in out_dir.iterdir()) == [
        "strict_posthoc_report.json",
        "strict_posthoc_report.md",
        "task_success_report.json",
        "task_success_report.md",
    ]
    # Without --write-run-scores no score.json enters the run directory.
    assert sorted(path.name for path in cell.iterdir()) == ["run_record.json"]
    assert {path.name for path in runs_dir.iterdir()} == {cell.name, "_working"}
    report = json.loads((out_dir / "task_success_report.json").read_text())
    assert report["scoring_version"] == sp.SCORING_VERSION
    assert report["n_saved_records"] == 1
    assert capsys.readouterr().out.strip() == str(out_dir / "task_success_report.md")


def test_strict_posthoc_write_run_scores_adds_only_score_json(monkeypatch, tmp_path) -> None:
    runs_dir = tmp_path / "arm"
    out_dir = runs_dir / "_working"
    cell = _write_minimal_record(runs_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "strict_posthoc.py",
            "--runs-dir",
            str(runs_dir),
            "--out-dir",
            str(out_dir),
            "--write-run-scores",
        ],
    )

    sp.main()

    assert sorted(path.name for path in cell.iterdir()) == ["run_record.json", "score.json"]
    score = json.loads((cell / "score.json").read_text())
    assert score["task_id"] == "ols_robust"
    assert score["cell_id"] == cell.name
