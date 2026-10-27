"""Offline tests for extended_analyses (synthetic rows only, no data/ reads).

Fixture note: make_row is defined locally so this file runs even while the
shared tests/analysis_fixtures.py helper is still landing; the row shape
mirrors the RunRow contract in the implementation spec.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import extended_analyses as ea  # noqa: E402

GATE_NAMES = ("execution", "artifact", "numeric", "method", "integrity")


def make_row(**overrides) -> dict:
    """Valid default RunRow-shaped dict, overridable per test."""
    interface = overrides.pop("interface", "no_feedback")
    gates = {gate: True for gate in GATE_NAMES}
    gates.update(overrides.pop("gates", {}))
    row = {
        "model": "sonnet",
        "model_id": "claude-sonnet-4-6",
        "prompt": "zero_shot",
        "interface": interface,
        "interface_label": ea.INTERFACE_LABELS[interface],
        "platform": "python",
        "task": "ols_robust",
        "rep": 0,
        "task_success": all(gates.values()),
        "gates": gates,
        "criteria": {
            "specification_correctness": gates["method"],
            "executability": gates["execution"],
            "output_correctness": gates["artifact"] and gates["numeric"],
            "submission_validity": gates["integrity"],
        },
        "per_term": {},
        "cost_usd": 0.01,
        "tokens": {key: 0 for key in ea.TOKEN_KEYS},
        "wall_seconds": 10.0,
        "turns": 3,
        "provenance": {
            "arm_id": "sonnet_zero_shot",
            "run_dir": "synthetic",
            "source_kind": "test",
            "prompt_sha256": None,
            "scoring_source": "report",
            "usable_reason": "ok",
        },
    }
    row.update(overrides)
    return row


def make_cell(n: int, c: int, **common) -> list[dict]:
    """One design cell with n reps, the first c of which succeed."""
    return [
        make_row(
            rep=rep,
            gates={"numeric": rep < c},
            task_success=rep < c,
            **common,
        )
        for rep in range(n)
    ]


def term_entry(
    got: float,
    ref: float = 1.0,
    coef_tol: float = 0.01,
    *,
    coef_err: float | None = None,
    coef_pass: bool | None = None,
    se_pass: bool | None = None,
    se_err: float | None = None,
    se_tol: float | None = None,
    n_obs_pass: bool = True,
) -> dict:
    if coef_err is None:
        coef_err = abs(got - ref)
    if coef_pass is None:
        coef_pass = coef_err <= coef_tol
    return {
        "ref_estimate": ref,
        "got_estimate": got,
        "coef_err": coef_err,
        "coef_tol": coef_tol,
        "coef_pass": coef_pass,
        "ref_se": None,
        "got_se": None,
        "se_err": se_err,
        "se_tol": se_tol,
        "se_pass": se_pass,
        "ref_n_obs": None,
        "got_n_obs": None,
        "n_obs_pass": n_obs_pass,
    }


def chatbot_block(out: dict, section: str = "models") -> dict:
    return out[section]["sonnet"]["zero_shot"]["chatbot"]


# ---------------------------------------------------------------------------
# 1. pass_at_k
# ---------------------------------------------------------------------------


def test_pass_at_k_closed_form_extremes() -> None:
    out = ea.pass_at_k(make_cell(5, 0), k_values=(1, 2, 3, 5))
    by_itf = out["models"]["sonnet"]["zero_shot"]["by_interface"]["chatbot"]
    for k in ("1", "2", "3", "5"):
        assert by_itf[k]["pass_at_k_pct"] == 0.0

    out = ea.pass_at_k(make_cell(5, 5), k_values=(1, 2, 3, 5))
    by_itf = out["models"]["sonnet"]["zero_shot"]["by_interface"]["chatbot"]
    for k in ("1", "2", "3", "5"):
        assert by_itf[k]["pass_at_k_pct"] == 100.0


def test_pass_at_k_single_success_cell() -> None:
    out = ea.pass_at_k(make_cell(5, 1), k_values=(1, 5))
    by_itf = out["models"]["sonnet"]["zero_shot"]["by_interface"]["chatbot"]
    assert by_itf["1"]["pass_at_k_pct"] == 20.0
    assert by_itf["5"]["pass_at_k_pct"] == 100.0


def test_pass_at_k_skips_short_cells_and_counts_them() -> None:
    rows = make_cell(5, 3, task="t_a") + make_cell(4, 2, task="t_b")
    out = ea.pass_at_k(rows, k_values=(1, 5))
    by_itf = out["models"]["sonnet"]["zero_shot"]["by_interface"]["chatbot"]
    assert by_itf["1"] == {"pass_at_k_pct": 55.0, "cells_used": 2, "cells_skipped": 0}
    assert by_itf["5"]["cells_used"] == 1
    assert by_itf["5"]["cells_skipped"] == 1
    # (n=5, c=3): n - c = 2 < 5 so pass@5 is 1.0 by definition.
    assert by_itf["5"]["pass_at_k_pct"] == 100.0


def test_pass_at_k_unbiasedness_sanity() -> None:
    out = ea.pass_at_k(make_cell(10, 3), k_values=(3,))
    by_itf = out["models"]["sonnet"]["zero_shot"]["by_interface"]["chatbot"]
    expected = 1.0 - math.comb(7, 3) / math.comb(10, 3)
    assert by_itf["3"]["pass_at_k_pct"] == round(100.0 * expected, 2)


def test_pass_at_k_overall_pools_cells_across_interfaces() -> None:
    rows = make_cell(5, 5, task="t_a") + make_cell(
        5, 0, task="t_a", interface="tool_loop"
    )
    out = ea.pass_at_k(rows, k_values=(1,))
    overall = out["models"]["sonnet"]["zero_shot"]["overall"]["1"]
    assert overall["cells_used"] == 2
    assert overall["pass_at_k_pct"] == 50.0


# ---------------------------------------------------------------------------
# 2. majority_vote
# ---------------------------------------------------------------------------


def vote_cell(per_terms: list[dict], successes: list[bool]) -> list[dict]:
    return [
        make_row(
            rep=rep,
            per_term=per_terms[rep],
            gates={"numeric": successes[rep]},
            task_success=successes[rep],
        )
        for rep in range(len(per_terms))
    ]


def test_majority_vote_correct_majority_beats_garbage() -> None:
    per_terms = [{"beta": term_entry(1.0)}] * 3 + [
        {"beta": term_entry(50.0)},
        {"beta": term_entry(80.0)},
    ]
    rows = vote_cell(per_terms, [True, True, True, False, False])
    block = chatbot_block(ea.majority_vote(rows))
    assert block["vote_success_pct"] == 100.0
    assert block["pass_at_1_pct"] == 60.0
    assert block["gain_pp"] == 40.0


def test_majority_vote_wrong_majority_beats_correct_minority() -> None:
    per_terms = [{"beta": term_entry(5.0)}] * 3 + [{"beta": term_entry(1.0)}] * 2
    rows = vote_cell(per_terms, [False, False, False, True, True])
    block = chatbot_block(ea.majority_vote(rows))
    assert block["vote_success_pct"] == 0.0
    assert block["pass_at_1_pct"] == 40.0  # majority can be worse than pass@1


def test_majority_vote_tie_breaks_to_smallest_median() -> None:
    per_terms = [
        {"beta": term_entry(1.0)},
        {"beta": term_entry(1.0)},
        {"beta": term_entry(5.0)},
        {"beta": term_entry(5.0)},
        {"beta": {"estimate_pass": False}},  # no-answer ballot for the term
    ]
    rows = vote_cell(per_terms, [True, True, False, False, False])
    block = chatbot_block(ea.majority_vote(rows))
    # Two clusters of size 2 tie; the smaller median (1.0) wins and matches ref.
    assert block["vote_success_pct"] == 100.0


def test_majority_vote_no_answer_majority_fails() -> None:
    per_terms = [{"beta": {"estimate_pass": False}}] * 3 + [
        {"beta": term_entry(1.0)}
    ] * 2
    rows = vote_cell(per_terms, [False, False, False, True, True])
    block = chatbot_block(ea.majority_vote(rows))
    # Ballots are non-empty (threshold met) but no-answer outnumbers the modal
    # cluster, so the term fails.
    assert block["vote_success_pct"] == 0.0


def test_majority_vote_empty_ballot_majority_fails() -> None:
    per_terms = [{}] * 3 + [{"beta": term_entry(1.0)}] * 2
    rows = vote_cell(per_terms, [False, False, False, True, True])
    block = chatbot_block(ea.majority_vote(rows))
    assert block["vote_success_pct"] == 0.0


def test_majority_vote_exact_integer_estimand_votes_on_equality() -> None:
    per_terms = [{"df": term_entry(4.0, ref=4.0, coef_tol=0.0)}] * 3 + [
        {"df": term_entry(5.0, ref=4.0, coef_tol=0.0)}
    ] * 2
    rows = vote_cell(per_terms, [True, True, True, False, False])
    assert chatbot_block(ea.majority_vote(rows))["vote_success_pct"] == 100.0

    per_terms = [{"df": term_entry(5.0, ref=4.0, coef_tol=0.0)}] * 3 + [
        {"df": term_entry(4.0, ref=4.0, coef_tol=0.0)}
    ] * 2
    rows = vote_cell(per_terms, [False, False, False, True, True])
    assert chatbot_block(ea.majority_vote(rows))["vote_success_pct"] == 0.0


def test_majority_vote_short_cell_flagged() -> None:
    per_terms = [{"beta": term_entry(1.0)}] * 3
    rows = vote_cell(per_terms, [True, True, True])
    block = chatbot_block(ea.majority_vote(rows, k=5))
    assert block["n_cells"] == 1  # the vote_success_pct denominator
    assert block["cells_short"] == 1
    assert block["cells_full_k"] == 0
    assert block["vote_success_pct"] == 100.0


def test_convention_mirrors_match_strict_posthoc() -> None:
    import strict_posthoc as sp

    assert ea.ABS_COMPARE_ESTIMANDS == frozenset(sp.ABS_COMPARE_ESTIMANDS)
    assert ea.MULTIPLE_CONVENTION_ESTIMANDS == dict(sp.MULTIPLE_CONVENTION_ESTIMANDS)


def convention_cell(task: str, term: str, per_terms: list[dict]) -> list[dict]:
    return [
        make_row(rep=rep, task=task, per_term=per_terms[rep])
        for rep in range(len(per_terms))
    ]


def test_majority_vote_abs_compare_sign_convention_passes() -> None:
    # Modal cluster at the accepted opposite sign (Stata t convention) passes.
    per_terms = [{"t": term_entry(-2.985, ref=2.985, coef_tol=0.02985)}] * 3 + [
        {"t": term_entry(2.985, ref=2.985, coef_tol=0.02985)}
    ] * 2
    rows = convention_cell("ttest_welch", "t", per_terms)
    assert chatbot_block(ea.majority_vote(rows))["vote_success_pct"] == 100.0


def test_majority_vote_abs_compare_sign_tie_merges_clusters() -> None:
    # A 2-2 sign split is one answer under magnitude comparison, so the merged
    # cluster of four beats the garbage ballot.
    per_terms = (
        [{"t": term_entry(-2.985, ref=2.985, coef_tol=0.02985)}] * 2
        + [{"t": term_entry(2.985, ref=2.985, coef_tol=0.02985)}] * 2
        + [{"t": term_entry(50.0, ref=2.985, coef_tol=0.02985)}]
    )
    rows = convention_cell("ttest_welch", "t", per_terms)
    assert chatbot_block(ea.majority_vote(rows))["vote_success_pct"] == 100.0


def test_majority_vote_accepts_declared_convention_multiplier() -> None:
    # Modal cluster at 2x ref (vars::causality df_den convention) passes.
    per_terms = [{"df_den": term_entry(976.0, ref=488.0, coef_tol=4.88)}] * 3 + [
        {"df_den": term_entry(488.0, ref=488.0, coef_tol=4.88)}
    ] * 2
    rows = convention_cell("var_granger", "df_den", per_terms)
    assert chatbot_block(ea.majority_vote(rows))["vote_success_pct"] == 100.0


def test_majority_vote_convention_rules_are_task_scoped() -> None:
    # The same ballots under a task with no declared convention still fail.
    per_terms = [{"df_den": term_entry(976.0, ref=488.0, coef_tol=4.88)}] * 3 + [
        {"df_den": term_entry(488.0, ref=488.0, coef_tol=4.88)}
    ] * 2
    rows = convention_cell("ols_robust", "df_den", per_terms)
    assert chatbot_block(ea.majority_vote(rows))["vote_success_pct"] == 0.0

    per_terms = [{"t": term_entry(-2.985, ref=2.985, coef_tol=0.02985)}] * 3 + [
        {"t": term_entry(2.985, ref=2.985, coef_tol=0.02985)}
    ] * 2
    rows = convention_cell("ols_robust", "t", per_terms)
    assert chatbot_block(ea.majority_vote(rows))["vote_success_pct"] == 0.0


# ---------------------------------------------------------------------------
# 3. silent_failure
# ---------------------------------------------------------------------------


def test_silent_failure_exact_share_and_empty_denominator() -> None:
    rows = [
        make_row(rep=i, gates={"numeric": i != 0}, task_success=i != 0)
        for i in range(4)
    ]
    rows.append(
        make_row(
            interface="self_repair",
            gates={"execution": False, "artifact": False, "numeric": False},
            task_success=False,
        )
    )
    out = ea.silent_failure(rows)
    strata = out["models"]["sonnet"]["zero_shot"]
    assert strata["chatbot"] == {
        "n_clean": 4,
        "silent_failures": 1,
        "silent_fail_pct": 25.0,
    }
    assert strata["one-repair"]["n_clean"] == 0
    assert strata["one-repair"]["silent_fail_pct"] is None


# ---------------------------------------------------------------------------
# 4. criterion_cascade and interface_gain_decomposition
# ---------------------------------------------------------------------------


def cascade_rows() -> list[dict]:
    specs = [
        {"execution": False, "method": False},  # must land in execution only
        {"execution": False},
        {"execution": False},
        {"artifact": False},
        {"numeric": False},
        {"method": False},
        {},
        {},
        {},
        {},
    ]
    return [
        make_row(rep=0, task=f"t{i:02d}", gates=dict(gates))
        for i, gates in enumerate(specs)
    ]


def test_criterion_cascade_is_a_partition() -> None:
    out = ea.criterion_cascade(cascade_rows())
    block = chatbot_block(out)
    assert block["counts"] == {
        "execution": 3,
        "artifact": 1,
        "numeric": 1,
        "method": 1,
        "integrity": 0,
        "success": 4,
    }
    assert sum(block["counts"].values()) == block["n"]
    assert sum(block["shares_pct"].values()) == 100.0


def test_interface_gain_decomposition_sums_exactly_to_gap() -> None:
    agent_specs = [{"execution": False}, {"numeric": False}] + [{}] * 8
    rows = cascade_rows() + [
        make_row(rep=0, task=f"t{i:02d}", interface="tool_loop", gates=dict(gates))
        for i, gates in enumerate(agent_specs)
    ]
    out = ea.interface_gain_decomposition(rows, "no_feedback", "tool_loop")
    block = out["models"]["sonnet"]["zero_shot"]
    assert block["success_from_pct"] == 40.0
    assert block["success_to_pct"] == 80.0
    assert block["total_gain_pp"] == 40.0
    contribs = block["stage_contributions_pp"]
    assert contribs == {
        "execution": 20.0,
        "artifact": 10.0,
        "numeric": 0.0,
        "method": 10.0,
        "integrity": 0.0,
    }
    assert sum(contribs.values()) == block["total_gain_pp"]

    # Display labels are accepted as interface arguments.
    relabeled = ea.interface_gain_decomposition(rows, "chatbot", "constrained agent")
    assert relabeled["models"]["sonnet"]["zero_shot"] == block


def test_interface_gain_decomposition_insufficient_data() -> None:
    out = ea.interface_gain_decomposition(cascade_rows(), "no_feedback", "tool_loop")
    block = out["models"]["sonnet"]["zero_shot"]
    assert block["status"] == "insufficient data"
    assert block["n_to"] == 0


# ---------------------------------------------------------------------------
# 5. tolerance_sensitivity
# ---------------------------------------------------------------------------


def numeric_row(per_term: dict, numeric_pass: bool, **kw) -> dict:
    gates = {"numeric": numeric_pass}
    gates.update(kw.pop("gates", {}))
    kw.setdefault("task_success", numeric_pass)
    return make_row(per_term=per_term, gates=gates, **kw)


def scale_pcts(out: dict) -> dict:
    return out["models"]["sonnet"]["zero_shot"]["overall"]["task_success_pct_by_scale"]


def test_tolerance_sensitivity_reproduces_recorded_at_one_and_flips_at_two() -> None:
    rows = [
        numeric_row({"beta": term_entry(1.0)}, True, rep=0),
        numeric_row({"beta": term_entry(1.015, coef_tol=0.01)}, False, rep=1),
    ]
    out = ea.tolerance_sensitivity(rows)
    pcts = scale_pcts(out)
    assert pcts["1.0"] == 50.0  # recorded verdicts reproduced, no raise
    assert pcts["0.5"] == 50.0
    assert pcts["2.0"] == 100.0
    assert pcts["4.0"] == 100.0


def test_tolerance_sensitivity_structural_failures_never_flip() -> None:
    rows = [
        # (c) execution-failed row with passing stored terms.
        numeric_row(
            {"beta": term_entry(1.0)},
            True,
            gates={"execution": False, "numeric": True},
            task_success=False,
            rep=0,
        ),
        # (d) empty per_term: structural numeric failure.
        numeric_row({}, False, rep=1, task="t_b"),
        # (e) missing finite se: se_pass False with no stored se_err.
        numeric_row({"beta": term_entry(1.0, se_pass=False)}, False, rep=2, task="t_c"),
        # (f) n_obs failure: exact-match rule, never scaled.
        numeric_row(
            {"beta": term_entry(1.0, n_obs_pass=False)}, False, rep=3, task="t_d"
        ),
        # missing-term keep-fail: recorded failure, stored terms all pass.
        numeric_row({"beta": term_entry(1.0)}, False, rep=4, task="t_e"),
    ]
    out = ea.tolerance_sensitivity(rows)
    pcts = scale_pcts(out)
    for scale in ("0.25", "0.5", "1.0", "2.0", "4.0"):
        assert pcts[scale] == 0.0
    overall = out["models"]["sonnet"]["zero_shot"]["overall"]
    # Gate, empty-per_term, and missing-term failures count as structural; the
    # se and n_obs rows fail at every scale through the recompute path.
    assert overall["structurally_failed"] == 3


def test_tolerance_sensitivity_inconsistent_recorded_success_raises() -> None:
    row = numeric_row({"beta": term_entry(5.0, coef_tol=0.01)}, True)
    with pytest.raises(ValueError, match="scale 1.0"):
        ea.tolerance_sensitivity([row])


def test_tolerance_sensitivity_targets_strict_verdict_on_annotated_rows() -> None:
    # A loader-annotated row can pass the unified primary outcome while the
    # strict numeric gate fails; the recompute must target the strict verdict
    # and must not raise.
    row = numeric_row(
        {"beta": term_entry(1.015, coef_tol=0.01)},
        False,
        task_success=True,
        task_success_strict=False,
    )
    pcts = scale_pcts(ea.tolerance_sensitivity([row]))
    assert pcts["1.0"] == 0.0
    assert pcts["2.0"] == 100.0


def test_tolerance_sensitivity_convention_rescue_uses_rescue_tolerance() -> None:
    # strict_posthoc stores the post-rescue coef_err (vs 2 * ref = 976) but
    # keeps coef_tol anchored to the unscaled ref: coef_err 6.0 > coef_tol
    # 4.88 while the recorded verdict is a pass via the rescue tolerance
    # (max(0.01 * 976, 0.005) = 9.76). The recompute must not raise and must
    # scale from the rescue tolerance.
    entry = term_entry(970.0, ref=488.0, coef_tol=4.88, coef_err=6.0, coef_pass=True)
    row = numeric_row({"df_den": entry}, True, task="var_granger")
    pcts = scale_pcts(ea.tolerance_sensitivity([row]))
    assert pcts["1.0"] == 100.0
    assert pcts["0.5"] == 0.0  # 6.0 > 0.5 * 9.76
    assert pcts["2.0"] == 100.0


def test_tolerance_sensitivity_rescue_shape_without_convention_still_raises() -> None:
    # A recorded pass with coef_err above coef_tol on a task with no declared
    # convention remains irreproducible and must keep tripping the guard.
    entry = term_entry(970.0, ref=488.0, coef_tol=4.88, coef_err=6.0, coef_pass=True)
    row = numeric_row({"df_den": entry}, True)
    with pytest.raises(ValueError, match="scale 1.0"):
        ea.tolerance_sensitivity([row])


# ---------------------------------------------------------------------------
# 6. per_cell_dispersion
# ---------------------------------------------------------------------------


def test_per_cell_dispersion_classification_and_short_exclusion() -> None:
    rows = (
        make_cell(5, 5, task="t_a")
        + make_cell(5, 0, task="t_b")
        + make_cell(5, 3, task="t_c")
        + make_cell(1, 1, task="t_d")  # below min_reps, excluded and counted
    )
    block = chatbot_block(ea.per_cell_dispersion(rows))
    assert block["n_cells"] == 3
    assert block["cells_excluded_short"] == 1
    assert block["always_pass_pct"] == 33.33
    assert block["always_fail_pct"] == 33.33
    assert block["mixed_pct"] == 33.33


def test_per_cell_dispersion_icc_none_for_single_cell() -> None:
    block = chatbot_block(ea.per_cell_dispersion(make_cell(4, 2)))
    assert block["n_cells"] == 1
    assert block["icc_anova"] is None


def test_per_cell_dispersion_icc_matches_hand_computation() -> None:
    # Cells [1,1,0] and [0,0,0]: N=6, I=2, k0=3, MSB=2/3, MSW=1/6, ICC=0.5.
    rows = make_cell(3, 2, task="t_a") + make_cell(3, 0, task="t_b")
    block = chatbot_block(ea.per_cell_dispersion(rows))
    assert block["icc_anova"] == 0.5


# ---------------------------------------------------------------------------
# 7. turn_distributions
# ---------------------------------------------------------------------------


def test_turn_distributions_percentiles_and_codex_skip() -> None:
    rows = [make_row(rep=0, task=f"t{i:02d}", turns=i + 1) for i in range(10)]
    rows.append(
        make_row(
            model="codex",
            model_id="gpt-5.4",
            turns=None,
            cost_usd=None,
            task="t_x",
        )
    )
    out = ea.turn_distributions(rows)
    block = chatbot_block(out)
    assert block["n_with_turns"] == 10
    assert block["p90"] == 9  # nearest-rank on 1..10
    assert block["median"] == 5.5
    assert block["max"] == 10
    assert block["cap"] is None
    assert out["skipped_arms"] == [
        {
            "model": "codex",
            "prompt": "zero_shot",
            "reason": "no turn telemetry in codex CLI records",
        }
    ]
    assert "codex" not in out["models"]


def test_turn_distributions_sonnet_tool_loop_cap_share() -> None:
    turns = [12, 12, 6, 8]
    rows = [
        make_row(rep=0, task=f"t{i}", interface="tool_loop", turns=t)
        for i, t in enumerate(turns)
    ]
    block = ea.turn_distributions(rows)["models"]["sonnet"]["zero_shot"][
        "constrained agent"
    ]
    assert block["cap"] == 12
    assert block["share_at_cap_pct"] == 50.0


# ---------------------------------------------------------------------------
# 8. cost_aggregation
# ---------------------------------------------------------------------------


def test_cost_aggregation_native_excludes_unpriced_rows() -> None:
    rows = [
        make_row(rep=0, cost_usd=0.01),
        make_row(rep=1, cost_usd=0.03),
        make_row(rep=2, cost_usd=None),
    ]
    native = chatbot_block(ea.cost_aggregation(rows))["native"]
    assert native["n_priced"] == 2
    assert native["total_cost_usd"] == 0.04
    assert native["mean_cost_usd"] == 0.02
    assert native["median_cost_usd"] == 0.02
    assert native["cost_per_success_usd"] == 0.02


def codex_row() -> dict:
    tokens = {key: 0 for key in ea.TOKEN_KEYS}
    tokens.update(
        {
            "input_tokens": 50_000,  # includes the 40_000 cached (OpenAI shape)
            "cached_input_tokens": 40_000,
            "output_tokens": 2_000,
        }
    )
    return make_row(model="codex", model_id="gpt-5.4", cost_usd=None, tokens=tokens)


def test_cost_aggregation_codex_imputation_to_the_cent() -> None:
    out = ea.cost_aggregation([codex_row()])
    block = out["models"]["codex"]["zero_shot"]["chatbot"]
    # (10_000 * 2.50 + 40_000 * 0.25 + 2_000 * 15.00) / 1e6 = 0.065
    assert block["imputed"]["total_cost_usd"] == 0.065
    assert block["imputed"]["available"] is True
    assert block["imputed"]["basis"] == ea.GPT54_IMPUTATION_BASIS
    # Never merged into native aggregates.
    assert block["native"]["n_priced"] == 0
    assert block["native"]["total_cost_usd"] == 0.0
    assert block["native"]["mean_cost_usd"] is None


def test_cost_aggregation_uses_plain_model_name_in_price_source() -> None:
    assert ea.GPT54_PRICING_SOURCE["model_row"] == "gpt-5.4"


def test_cost_aggregation_pricing_none_marks_unavailable() -> None:
    out = ea.cost_aggregation([codex_row()], pricing=None)
    block = out["models"]["codex"]["zero_shot"]["chatbot"]
    assert block["imputed"] == {
        "available": False,
        "reason": "gpt-5.4 API pricing not verified",
    }


def test_cost_aggregation_non_codex_strata_have_no_imputed_block() -> None:
    out = ea.cost_aggregation([make_row()])
    assert "imputed" not in chatbot_block(out)


def test_cost_aggregation_token_totals_and_means() -> None:
    tokens_a = {key: 0 for key in ea.TOKEN_KEYS}
    tokens_a.update({"input_tokens": 100, "output_tokens": 10})
    tokens_b = {key: 0 for key in ea.TOKEN_KEYS}
    tokens_b.update({"input_tokens": 300, "output_tokens": 30})
    rows = [
        make_row(rep=0, tokens=tokens_a),
        make_row(rep=1, tokens=tokens_b),
    ]
    block = chatbot_block(ea.cost_aggregation(rows))["tokens"]
    assert block["totals"]["input_tokens"] == 400
    assert block["mean_per_run"]["input_tokens"] == 200.0
    assert block["totals"]["reasoning_output_tokens"] == 0
