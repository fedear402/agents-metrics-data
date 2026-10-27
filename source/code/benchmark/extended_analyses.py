"""Extended analyses for the six-arm twelve-rep benchmark sample.

Every public function is a pure function over a sequence of ``RunRow``
mappings, the normalized record shape produced by
``analysis_dataset.load_runs``. This module treats rows structurally as
plain mappings so it stays import-free, offline, and stdlib-only
(``math``, ``statistics``, ``fractions``). All outputs are JSON-safe
dicts with rates as percentages rounded to 2 decimals.

Conventions shared by the outputs:

- Result dicts nest as ``{"models": {model: {prompt: ...}}}`` with the
  interface display labels (chatbot, one-repair, constrained agent) as
  stratum keys, in the canonical interface order.
- ``majority_vote`` is a POINT-ESTIMATE plurality vote. It covers point
  estimates only; the se, n_obs, and method gates are outside the vote.
  It applies the same declared numeric-gate conventions as
  ``strict_posthoc``: magnitude-only comparison for
  ``ABS_COMPARE_ESTIMANDS`` terms and the accepted reference multipliers
  for ``MULTIPLE_CONVENTION_ESTIMANDS`` terms. The JSON labels say so to
  prevent misquotation as a five-gate ensemble result.
- ``tolerance_sensitivity`` rescales stored per-term errors and targets the
  STRICT five-gate outcome (``task_success_strict`` on loader-annotated
  rows), because the stored errors describe the strict numeric gate.
  Structural failures (execution, artifact, method, integrity, empty
  ``per_term``, missing result rows) never flip.
  ``MULTIPLE_CONVENTION_ESTIMANDS`` rescues are already folded into the
  stored ``coef_err``, so scaling operates on the post-rescue error against
  the rescue tolerance the term actually passed with.
- ``pass_at_k``, ``majority_vote``, ``silent_failure``,
  ``per_cell_dispersion``, and the cost blocks read ``task_success``, so
  they inherit the row's primary outcome (unified since 2026-08-12).
  ``criterion_cascade`` and ``interface_gain_decomposition`` are gate
  anatomy over the strict outcome.
- The GPT-5.4 cost block is an imputed counterfactual at OpenAI API
  Standard-tier list prices. The codex runs were collected on a
  ChatGPT-subscription Codex CLI and were not API-billed. Imputed and
  native costs are never summed together.
- ``icc_anova`` is a one-way ANOVA ICC(1) on 0/1 outcomes with the
  unbalanced-design ``k0`` correction; it is a dispersion diagnostic,
  and can be slightly negative.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

# Structural alias for analysis_dataset.RunRow (kept import-free on purpose).
RunRow = Mapping[str, Any]
CellKey = tuple[str, str, str, str, str]  # model, prompt, interface, platform, task

MODELS = ("sonnet", "codex", "deepseek")
PROMPTS = ("zero_shot", "few_shot")
INTERFACES = ("no_feedback", "self_repair", "tool_loop")
INTERFACE_LABELS = {
    "no_feedback": "chatbot",
    "self_repair": "one-repair",
    "tool_loop": "constrained agent",
}
_LABEL_TO_INTERFACE = {label: itf for itf, label in INTERFACE_LABELS.items()}

GATE_ORDER = ("execution", "artifact", "numeric", "method", "integrity")
CASCADE_STAGES = GATE_ORDER + ("success",)
TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)

# strict_posthoc coefficient tolerance constants (majority-vote fallback and
# convention re-checks).
COEF_REL_TOL = 1e-2
COEF_ABS_TOL = 5e-3

# Declared numeric-gate conventions, mirrored from strict_posthoc so this
# module stays import-free; tests/test_extended_analyses.py asserts the
# mirror matches strict_posthoc exactly.
# ABS_COMPARE_ESTIMANDS: (task, term) pairs compared on magnitude only
# (benign sign convention, e.g. Stata's mean(g0)-mean(g1) ttest_welch t).
# MULTIPLE_CONVENTION_ESTIMANDS: (task, term) -> accepted reference
# multipliers (var_granger df_den: system Wald = 2x aligned-OLS df_den).
ABS_COMPARE_ESTIMANDS: frozenset[tuple[str, str]] = frozenset({("ttest_welch", "t")})
MULTIPLE_CONVENTION_ESTIMANDS: dict[tuple[str, str], tuple[float, ...]] = {
    ("var_granger", "df_den"): (2.0,)
}

SONNET_TOOL_LOOP_TURN_CAP = 12

# GPT-5.4 API list prices, verified 2026-07-30 (see GPT54_PRICING_SOURCE).
# If a future re-verification fails, set this dict to None; cost_aggregation
# then reports the codex imputed block as unavailable.
GPT54_PRICING_USD_PER_MTOK: dict[str, float] | None = {
    "input": 2.50,
    "cached_input": 0.25,
    "output": 15.00,
}
GPT54_PRICING_SOURCE: dict[str, str] = {
    "model_row": "gpt-5.4",
    "tier": "Standard",
    "url": "https://developers.openai.com/api/docs/pricing",
    "accessed": "2026-07-30",
    "note": (
        "Imputed counterfactual API cost; the codex arms ran on a "
        "ChatGPT-subscription Codex CLI with cost_usd unrecorded (null). "
        "reasoning_output_tokens and cached_input_tokens are subsets of "
        "output_tokens / input_tokens (verified over 3,537 usage blocks)."
    ),
}
GPT54_IMPUTATION_BASIS = (
    "OpenAI API Standard tier list prices; runs were collected on a "
    "ChatGPT-subscription Codex CLI and were not API-billed"
)


def pct(successes: float, n: float) -> float:
    return round(100.0 * successes / n, 2) if n else 0.0


def _seq_rank(seq: Sequence[str], value: str) -> tuple[int, str]:
    return (seq.index(value) if value in seq else len(seq), value)


def _cell_order(key: CellKey) -> tuple:
    model, prompt, interface, platform, task = key
    return (
        _seq_rank(MODELS, model),
        _seq_rank(PROMPTS, prompt),
        _seq_rank(INTERFACES, interface),
        platform,
        task,
    )


def _label(interface: str) -> str:
    return INTERFACE_LABELS.get(interface, interface)


def _canon_interface(value: str) -> str:
    """Accept a canonical interface name or its display label."""
    if value in INTERFACES:
        return value
    if value in _LABEL_TO_INTERFACE:
        return _LABEL_TO_INTERFACE[value]
    raise ValueError(f"unknown interface: {value!r}")


def group_cells(rows: Sequence[RunRow]) -> dict[CellKey, list[RunRow]]:
    """Group rows by design cell; rows within a cell are sorted by rep."""
    cells: dict[CellKey, list[RunRow]] = {}
    for row in rows:
        key = (
            row["model"],
            row["prompt"],
            row["interface"],
            row["platform"],
            row["task"],
        )
        cells.setdefault(key, []).append(row)
    return {
        key: sorted(cells[key], key=lambda r: r["rep"])
        for key in sorted(cells, key=_cell_order)
    }


def _strata(rows: Sequence[RunRow]) -> dict[tuple[str, str, str], list[RunRow]]:
    """Rows grouped by (model, prompt, interface) in canonical order."""
    groups: dict[tuple[str, str, str], list[RunRow]] = {}
    for row in rows:
        key = (row["model"], row["prompt"], row["interface"])
        groups.setdefault(key, []).append(row)
    ordered = sorted(
        groups,
        key=lambda k: (
            _seq_rank(MODELS, k[0]),
            _seq_rank(PROMPTS, k[1]),
            _seq_rank(INTERFACES, k[2]),
        ),
    )
    return {key: groups[key] for key in ordered}


def _nest(entries: Mapping[tuple[str, str, str], Any]) -> dict[str, Any]:
    """{(model, prompt, interface): block} -> {model: {prompt: {label: block}}}."""
    out: dict[str, Any] = {}
    for (model, prompt, interface), block in entries.items():
        out.setdefault(model, {}).setdefault(prompt, {})[_label(interface)] = block
    return out


# ---------------------------------------------------------------------------
# pass@k
# ---------------------------------------------------------------------------


def _pass_at_k_cell(n: int, c: int, k: int) -> float:
    """HumanEval unbiased estimator for one cell (requires n >= k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _pass_at_k_summary(
    counts: Sequence[tuple[int, int]], k_values: Sequence[int]
) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for k in k_values:
        eligible = [(n, c) for n, c in counts if n >= k]
        value = (
            round(
                100.0 * statistics.fmean(_pass_at_k_cell(n, c, k) for n, c in eligible),
                2,
            )
            if eligible
            else None
        )
        block[str(k)] = {
            "pass_at_k_pct": value,
            "cells_used": len(eligible),
            "cells_skipped": len(counts) - len(eligible),
        }
    return block


def pass_at_k(
    rows: Sequence[RunRow], k_values: Sequence[int] = (1, 2, 3, 5)
) -> dict[str, Any]:
    """Unweighted mean pass@k over design cells, per model x prompt x interface.

    Callers pass ALL-reps rows (decision record: pass@k uses all available
    repetitions). Cells with fewer than k reps are skipped for that k and
    counted in ``cells_skipped``.
    """
    ks = [int(k) for k in k_values]
    per_stratum: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for (model, prompt, interface, _p, _t), cell_rows in group_cells(rows).items():
        n = len(cell_rows)
        c = sum(bool(r["task_success"]) for r in cell_rows)
        per_stratum.setdefault((model, prompt, interface), []).append((n, c))

    arm_counts: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for (model, prompt, _interface), counts in per_stratum.items():
        arm_counts.setdefault((model, prompt), []).extend(counts)

    models: dict[str, Any] = {}
    for (model, prompt), counts in arm_counts.items():
        models.setdefault(model, {})[prompt] = {
            "overall": _pass_at_k_summary(counts, ks),
            "by_interface": {},
        }
    for (model, prompt, interface), counts in per_stratum.items():
        models[model][prompt]["by_interface"][_label(interface)] = _pass_at_k_summary(
            counts, ks
        )
    return {
        "estimator": (
            "HumanEval unbiased pass@k = 1 - C(n-c, k)/C(n, k) per cell; "
            "unweighted mean over cells with n >= k"
        ),
        "reps_rule": "all available reps",
        "k_values": ks,
        "models": models,
    }


# ---------------------------------------------------------------------------
# majority vote (point estimates only)
# ---------------------------------------------------------------------------


def _cluster(sorted_values: Sequence[float], tol: float) -> list[list[float]]:
    """Single-link agglomeration: consecutive gaps <= tol join a cluster."""
    clusters: list[list[float]] = []
    for value in sorted_values:
        if clusters and value - clusters[-1][-1] <= tol:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _term_vote(
    entries: Sequence[Mapping[str, Any]], n_ballots: int, task: str, term: str
) -> bool:
    """Vote one estimand term. Missing or non-finite values are no-answer ballots.

    Applies the declared numeric-gate conventions: ``ABS_COMPARE_ESTIMANDS``
    terms cluster and compare on magnitude (a sign flip is the same answer),
    and a modal cluster matching an accepted ``MULTIPLE_CONVENTION_ESTIMANDS``
    reference multiple passes under the rescue tolerance.
    """
    abs_compare = (task, term) in ABS_COMPARE_ESTIMANDS
    values: list[float] = []
    tols: list[float] = []
    ref: float | None = None
    for entry in entries:
        got = entry.get("got_estimate")
        if _finite(got):
            value = float(got)
            values.append(abs(value) if abs_compare else value)
        if ref is None and _finite(entry.get("ref_estimate")):
            ref = float(entry["ref_estimate"])
        if _finite(entry.get("coef_tol")):
            tols.append(float(entry["coef_tol"]))
    no_answer = n_ballots - len(values)
    if not values or ref is None:
        return False
    if abs_compare:
        ref = abs(ref)
    # Exact-integer estimands store coef_tol == 0.0: clustering and the final
    # comparison then both operate on exact equality.
    tol = max(tols) if tols else max(COEF_REL_TOL * abs(ref), COEF_ABS_TOL)
    clusters = _cluster(sorted(values), tol)
    modal = min(clusters, key=lambda c: (-len(c), statistics.median(c)))
    if no_answer > len(modal):
        return False
    median = statistics.median(modal)
    if abs(median - ref) <= tol:
        return True
    for factor in MULTIPLE_CONVENTION_ESTIMANDS.get((task, term), ()):
        scaled = factor * ref
        if abs(median - scaled) <= max(COEF_REL_TOL * abs(scaled), COEF_ABS_TOL):
            return True
    return False


def _cell_vote(ballots: Sequence[RunRow], k: int, task: str) -> bool:
    """Vote one cell over its first-k ballots."""
    n_ballots = len(ballots)
    answered = sum(1 for r in ballots if r["per_term"])
    if answered < math.ceil(min(k, n_ballots) / 2):
        return False
    term_entries: dict[str, list[Mapping[str, Any]]] = {}
    for row in ballots:
        for term, entry in (row["per_term"] or {}).items():
            if not isinstance(entry, Mapping) or entry.get("skipped"):
                continue
            term_entries.setdefault(term, []).append(entry)
    return all(
        _term_vote(term_entries[term], n_ballots, task, term)
        for term in sorted(term_entries)
    )


def majority_vote(rows: Sequence[RunRow], k: int = 5) -> dict[str, Any]:
    """Point-estimate plurality vote per cell on the first k reps by rep index.

    The vote covers point estimates only (no se, n_obs, or method voting) and
    applies the declared numeric-gate conventions (magnitude-only terms,
    accepted reference multipliers), so it approximates ensemble numeric
    agreement from above; it is labeled as a point-estimate vote everywhere to
    prevent misquotation as a five-gate ensemble result. ``vote_success_pct``
    is over all ``n_cells`` cells with at least one rep; ``cells_full_k``
    counts the cells that supplied all k ballots.
    """
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, cell_rows in group_cells(rows).items():
        stratum = key[:3]
        ballots = cell_rows[:k]
        bucket = buckets.setdefault(stratum, {"votes": [], "rates": [], "short": 0})
        if len(cell_rows) < k:
            bucket["short"] += 1
        bucket["votes"].append(_cell_vote(ballots, k, key[4]))
        bucket["rates"].append(
            statistics.fmean(bool(r["task_success"]) for r in ballots)
        )

    entries: dict[tuple[str, str, str], Any] = {}
    for stratum, bucket in buckets.items():
        n_cells = len(bucket["votes"])
        vote_pct = pct(sum(bucket["votes"]), n_cells)
        pass1 = (
            round(100.0 * statistics.fmean(bucket["rates"]), 2)
            if bucket["rates"]
            else None
        )
        entries[stratum] = {
            "n_cells": n_cells,
            "cells_full_k": n_cells - bucket["short"],
            "cells_short": bucket["short"],
            "vote_success_pct": vote_pct,
            "pass_at_1_pct": pass1,
            "gain_pp": round(vote_pct - pass1, 2) if pass1 is not None else None,
        }
    return {
        "definition": (
            "point-estimate plurality vote over the first k reps per cell, "
            "tolerance-clustered per estimand term under the declared "
            "sign/convention rules; se, n_obs, and method gates are outside "
            "the vote"
        ),
        "k": k,
        "short_cell_rule": (
            "cells with fewer than k reps vote over the reps that exist "
            "(counted in cells_short) with majority threshold ceil(n/2); "
            "vote_success_pct is over all n_cells voting cells"
        ),
        "models": _nest(entries),
    }


# ---------------------------------------------------------------------------
# silent failure
# ---------------------------------------------------------------------------


def silent_failure(rows: Sequence[RunRow]) -> dict[str, Any]:
    """P(strict failure | execution passes AND artifact passes), per stratum.

    Part of the failure anatomy, so it decomposes the STRICT five-gate
    verdict (``task_success_strict`` on loader-annotated rows). Only that
    outcome satisfies the anatomy identity: every strict failure is either
    loud (execution or artifact) or silent (clean but failing a later gate).
    """
    entries: dict[tuple[str, str, str], Any] = {}
    for key, srows in _strata(rows).items():
        clean = [r for r in srows if r["gates"]["execution"] and r["gates"]["artifact"]]
        silent = sum(
            1
            for r in clean
            if not bool(r.get("task_success_strict", r["task_success"]))
        )
        entries[key] = {
            "n_clean": len(clean),
            "silent_failures": silent,
            "silent_fail_pct": pct(silent, len(clean)) if clean else None,
        }
    return {
        "definition": (
            "share of runs that fail the strict five-gate verdict "
            "(task_success_strict) among runs that pass both the execution "
            "and artifact gates"
        ),
        "models": _nest(entries),
    }


# ---------------------------------------------------------------------------
# criterion cascade and interface gain decomposition
# ---------------------------------------------------------------------------


def _first_failure(row: RunRow) -> str:
    for gate in GATE_ORDER:
        if not row["gates"].get(gate):
            return gate
    return "success"


def criterion_cascade(rows: Sequence[RunRow]) -> dict[str, Any]:
    """First-failure partition in fixed gate order; shares sum to 100."""
    entries: dict[tuple[str, str, str], Any] = {}
    for key, srows in _strata(rows).items():
        counts = {stage: 0 for stage in CASCADE_STAGES}
        for row in srows:
            counts[_first_failure(row)] += 1
        n = len(srows)
        entries[key] = {
            "n": n,
            "counts": counts,
            "shares_pct": {stage: pct(counts[stage], n) for stage in CASCADE_STAGES},
        }
    return {
        "stage_order": list(CASCADE_STAGES),
        "definition": (
            "each run lands in exactly one bucket: its first failing gate in "
            "the order execution, artifact, numeric, method, integrity, or "
            "success when all five pass (the strict five-gate outcome)"
        ),
        "models": _nest(entries),
    }


def _exact_shares(srows: Sequence[RunRow]) -> dict[str, Fraction]:
    counts = {stage: 0 for stage in CASCADE_STAGES}
    for row in srows:
        counts[_first_failure(row)] += 1
    n = len(srows)
    return {stage: 100 * Fraction(counts[stage], n) for stage in CASCADE_STAGES}


def interface_gain_decomposition(
    rows: Sequence[RunRow], from_interface: str, to_interface: str
) -> dict[str, Any]:
    """Decompose the success gap between two interfaces into stage shares.

    Within each model x prompt, ``success(to) - success(from)`` equals the sum
    over failure stages of ``share_from - share_to``. Contributions are exact
    by construction (Fraction arithmetic); an internal invariant re-checks the
    identity on every call.
    """
    frm = _canon_interface(from_interface)
    to = _canon_interface(to_interface)
    arms: dict[tuple[str, str], dict[str, list[RunRow]]] = {}
    for row in rows:
        if row["interface"] not in (frm, to):
            continue
        arm = arms.setdefault((row["model"], row["prompt"]), {frm: [], to: []})
        arm[row["interface"]].append(row)

    models: dict[str, Any] = {}
    ordered = sorted(
        arms, key=lambda k: (_seq_rank(MODELS, k[0]), _seq_rank(PROMPTS, k[1]))
    )
    for model, prompt in ordered:
        rows_from = arms[(model, prompt)][frm]
        rows_to = arms[(model, prompt)][to]
        if not rows_from or not rows_to:
            block: dict[str, Any] = {
                "status": "insufficient data",
                "n_from": len(rows_from),
                "n_to": len(rows_to),
            }
        else:
            sh_from = _exact_shares(rows_from)
            sh_to = _exact_shares(rows_to)
            contribs = {g: sh_from[g] - sh_to[g] for g in GATE_ORDER}
            gap = sh_to["success"] - sh_from["success"]
            if sum(contribs.values()) != gap:
                raise AssertionError(
                    "stage contributions do not sum to the success gap"
                )
            block = {
                "n_from": len(rows_from),
                "n_to": len(rows_to),
                "success_from_pct": round(float(sh_from["success"]), 2),
                "success_to_pct": round(float(sh_to["success"]), 2),
                "total_gain_pp": round(float(gap), 2),
                "stage_contributions_pp": {
                    g: round(float(contribs[g]), 2) for g in GATE_ORDER
                },
            }
        models.setdefault(model, {})[prompt] = block
    return {
        "from_interface": frm,
        "to_interface": to,
        "from_label": _label(frm),
        "to_label": _label(to),
        "stage_order": list(GATE_ORDER),
        "definition": (
            "strict five-gate success(to) - success(from) decomposed into "
            "first-failure share differences (share_from - share_to) per "
            "gate; the stage contributions sum exactly to the total gain"
        ),
        "models": models,
    }


# ---------------------------------------------------------------------------
# tolerance sensitivity
# ---------------------------------------------------------------------------


def _effective_coef_tol(task: str, term: str, entry: Mapping[str, Any]) -> float:
    """Stored coef_tol, or the rescue tolerance for a convention-rescued term.

    ``strict_posthoc.compare_numeric`` stores the POST-rescue ``coef_err``
    (measured against the scaled reference) while ``coef_tol`` stays anchored
    to the unscaled reference, so a rescued term (recorded ``coef_pass`` with
    ``coef_err`` above ``coef_tol``) must be re-checked against the rescue
    tolerance it actually passed with.
    """
    coef_tol = float(entry["coef_tol"])
    if not entry.get("coef_pass") or entry["coef_err"] <= coef_tol:
        return coef_tol
    ref = entry.get("ref_estimate")
    rescue_tols = [
        max(COEF_REL_TOL * abs(factor * float(ref)), COEF_ABS_TOL)
        for factor in MULTIPLE_CONVENTION_ESTIMANDS.get((task, term), ())
        if _finite(ref)
    ]
    return max(rescue_tols, default=coef_tol)


def _recompute_numeric(task: str, per_term: Mapping[str, Any], scale: float) -> bool:
    """Recompute the numeric gate from stored per-term errors at one scale."""
    for term, entry in per_term.items():
        if not isinstance(entry, Mapping):
            return False
        if entry.get("skipped"):
            continue
        if "coef_err" not in entry or "coef_tol" not in entry:
            # Degenerate row ({"estimate_pass": False}): non-finite estimate,
            # unaffected by tolerance scaling.
            if not entry.get("estimate_pass", False):
                return False
            continue
        if entry["coef_err"] > scale * _effective_coef_tol(task, term, entry):
            return False
        se_pass = entry.get("se_pass")
        if se_pass is not None:
            se_err = entry.get("se_err")
            se_tol = entry.get("se_tol")
            if se_err is not None and se_tol is not None:
                if se_err > scale * se_tol:
                    return False
            elif not se_pass:
                # Missing finite se: no scale can fix it.
                return False
        if not entry.get("n_obs_pass", True):
            # n_obs is an exact-match rule, never scaled.
            return False
    return True


def _row_scale_outcomes(
    row: RunRow, scales: Sequence[float]
) -> tuple[dict[float, bool], bool, bool]:
    """Return (outcome per scale, structurally_failed, consistent_at_1).

    Operates on the strict five-gate verdict (``task_success_strict`` when the
    loader has annotated the row; the raw ``task_success`` otherwise), because
    the stored per-term errors describe the strict numeric gate.
    """
    gates = row["gates"]
    recorded = bool(row.get("task_success_strict", row["task_success"]))
    fail_all = {s: False for s in scales}
    structural_gates = ("execution", "artifact", "method", "integrity")
    if not all(bool(gates.get(g)) for g in structural_gates):
        return fail_all, True, not recorded
    per_term = row.get("per_term") or {}
    recorded_numeric = bool(gates.get("numeric"))
    if not per_term:
        # Structural numeric verdict (no parseable estimates or missing
        # reference): the recorded gate holds at every scale.
        outcomes = {s: recorded_numeric for s in scales}
        return outcomes, not recorded_numeric, recorded == recorded_numeric
    task = str(row["task"])
    base = _recompute_numeric(task, per_term, 1.0)
    if base != recorded:
        if recorded:
            return fail_all, False, False
        # Recorded failure with passing stored terms: a failed term is missing
        # from per_term (missing result row), so no scale can flip the verdict.
        return fail_all, True, True
    return {s: _recompute_numeric(task, per_term, s) for s in scales}, False, True


def _scale_block(
    items: Sequence[tuple[RunRow, dict[float, bool], bool]],
    scales: Sequence[float],
) -> dict[str, Any]:
    n = len(items)
    return {
        "n": n,
        "structurally_failed": sum(1 for _r, _o, s in items if s),
        "task_success_pct_by_scale": {
            str(s): pct(sum(1 for _r, o, _s in items if o[s]), n) for s in scales
        },
    }


def tolerance_sensitivity(
    rows: Sequence[RunRow],
    scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> dict[str, Any]:
    """Strict five-gate task success under rescaled numeric tolerances.

    A strict-outcome diagnostic: the stored per-term errors describe the
    strict numeric gate, so the recomputation targets ``task_success_strict``
    on loader-annotated rows. Raises ``ValueError`` when the recomputation at
    scale 1.0 disagrees with a recorded success (the recorded verdict must be
    reproducible from the stored per-term errors). Recorded failures whose
    stored terms all pass are missing-term structural failures and stay
    failed at every scale.
    """
    scale_list = [float(s) for s in scales]
    evaluated: list[tuple[RunRow, dict[float, bool], bool]] = []
    mismatched: list[str] = []
    for row in rows:
        outcomes, structural, consistent = _row_scale_outcomes(row, scale_list)
        if not consistent:
            mismatched.append(str(row.get("provenance", {}).get("run_dir")))
            continue
        evaluated.append((row, outcomes, structural))
    if mismatched:
        shown = ", ".join(mismatched[:20])
        raise ValueError(
            "tolerance_sensitivity: recomputed task_success at scale 1.0 "
            f"diverges from the recorded verdict for {len(mismatched)} "
            f"row(s): {shown}"
        )

    by_arm: dict[tuple[str, str], list[tuple[RunRow, dict[float, bool], bool]]] = {}
    for item in evaluated:
        by_arm.setdefault((item[0]["model"], item[0]["prompt"]), []).append(item)
    models: dict[str, Any] = {}
    ordered = sorted(
        by_arm, key=lambda k: (_seq_rank(MODELS, k[0]), _seq_rank(PROMPTS, k[1]))
    )
    for model, prompt in ordered:
        items = by_arm[(model, prompt)]
        by_interface: dict[str, Any] = {}
        present = {item[0]["interface"] for item in items}
        for interface in sorted(present, key=lambda i: _seq_rank(INTERFACES, i)):
            sub = [item for item in items if item[0]["interface"] == interface]
            by_interface[_label(interface)] = _scale_block(sub, scale_list)
        models.setdefault(model, {})[prompt] = {
            "overall": _scale_block(items, scale_list),
            "by_interface": by_interface,
        }
    return {
        "scales": scale_list,
        "note": (
            "strict five-gate outcome (task_success_strict) with the numeric "
            "gate recomputed from stored per-term errors at each tolerance "
            "scale; execution, artifact, method, and integrity failures never "
            "flip; multiple-convention rescues are folded into the stored "
            "coef_err and re-checked against the rescue tolerance they "
            "passed with"
        ),
        "models": models,
    }


# ---------------------------------------------------------------------------
# per-cell dispersion
# ---------------------------------------------------------------------------


def _icc_anova(cells: Sequence[Sequence[int]]) -> float | None:
    """One-way ANOVA ICC(1) with the unbalanced-design k0 correction."""
    counts = [len(cell) for cell in cells]
    total_n = sum(counts)
    n_cells = len(cells)
    if n_cells < 2 or total_n <= n_cells:
        return None
    grand = sum(sum(cell) for cell in cells) / total_n
    means = [sum(cell) / len(cell) for cell in cells]
    k0 = (total_n - sum(n * n for n in counts) / total_n) / (n_cells - 1)
    msb = sum(n * (m - grand) ** 2 for n, m in zip(counts, means)) / (n_cells - 1)
    msw = sum(sum((y - m) ** 2 for y in cell) for cell, m in zip(cells, means)) / (
        total_n - n_cells
    )
    denom = msb + (k0 - 1.0) * msw
    if denom == 0:
        return None
    return round((msb - msw) / denom, 4)


def per_cell_dispersion(rows: Sequence[RunRow], min_reps: int = 2) -> dict[str, Any]:
    """Always-pass / always-fail / mixed cell shares plus icc_anova per stratum.

    All-reps rows are the intended input (decision record).
    """
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, cell_rows in group_cells(rows).items():
        bucket = buckets.setdefault(key[:3], {"cells": [], "excluded": 0})
        if len(cell_rows) < min_reps:
            bucket["excluded"] += 1
            continue
        bucket["cells"].append([int(bool(r["task_success"])) for r in cell_rows])

    entries: dict[tuple[str, str, str], Any] = {}
    for stratum, bucket in buckets.items():
        cells = bucket["cells"]
        n_cells = len(cells)
        always_pass = sum(1 for cell in cells if all(cell))
        always_fail = sum(1 for cell in cells if not any(cell))
        entries[stratum] = {
            "n_cells": n_cells,
            "cells_excluded_short": bucket["excluded"],
            "always_pass_pct": pct(always_pass, n_cells) if n_cells else None,
            "always_fail_pct": pct(always_fail, n_cells) if n_cells else None,
            "mixed_pct": (
                pct(n_cells - always_pass - always_fail, n_cells) if n_cells else None
            ),
            "icc_anova": _icc_anova(cells),
        }
    return {
        "reps_rule": "all available reps",
        "min_reps": min_reps,
        "definition": (
            "share of design cells whose reps all pass, all fail, or mix, "
            "over cells with at least min_reps reps; icc_anova is a one-way "
            "ANOVA ICC(1) approximation on 0/1 outcomes (diagnostic only)"
        ),
        "models": _nest(entries),
    }


# ---------------------------------------------------------------------------
# turn distributions
# ---------------------------------------------------------------------------


def _nearest_rank(sorted_values: Sequence[int], percentile: float) -> int:
    rank = max(math.ceil(percentile / 100.0 * len(sorted_values)), 1)
    return sorted_values[rank - 1]


def turn_distributions(rows: Sequence[RunRow]) -> dict[str, Any]:
    """Turn-count distributions for rows that carry turn telemetry."""
    entries: dict[tuple[str, str, str], Any] = {}
    arms_seen: dict[tuple[str, str], bool] = {}
    for key, srows in _strata(rows).items():
        arm = key[:2]
        arms_seen.setdefault(arm, False)
        turns = sorted(int(r["turns"]) for r in srows if r["turns"] is not None)
        if not turns:
            continue
        arms_seen[arm] = True
        model, _prompt, interface = key
        cap = (
            SONNET_TOOL_LOOP_TURN_CAP
            if model == "sonnet" and interface == "tool_loop"
            else None
        )
        entries[key] = {
            "n_with_turns": len(turns),
            "mean": round(statistics.fmean(turns), 2),
            "median": statistics.median(turns),
            "p90": _nearest_rank(turns, 90),
            "max": turns[-1],
            "cap": cap,
            "share_at_cap_pct": (
                pct(sum(1 for t in turns if t >= cap), len(turns))
                if cap is not None
                else None
            ),
        }
    skipped_arms = [
        {
            "model": model,
            "prompt": prompt,
            "reason": (
                "no turn telemetry in codex CLI records"
                if model == "codex"
                else "no rows with turn telemetry"
            ),
        }
        for (model, prompt), has_turns in arms_seen.items()
        if not has_turns
    ]
    return {
        "p90_rule": "nearest-rank percentile",
        "models": _nest(entries),
        "skipped_arms": skipped_arms,
    }


# ---------------------------------------------------------------------------
# cost aggregation
# ---------------------------------------------------------------------------


def _impute_codex_cost(
    tokens: Mapping[str, Any], pricing: Mapping[str, float]
) -> float:
    """Imputed API cost of one codex run at the given USD-per-MTok prices.

    OpenAI convention: input_tokens includes cached_input_tokens and
    output_tokens includes reasoning_output_tokens (subset relations verified
    over all 3,537 codex usage blocks on 2026-07-30).
    """
    input_tokens = int(tokens.get("input_tokens", 0))
    cached = int(tokens.get("cached_input_tokens", 0))
    output = int(tokens.get("output_tokens", 0))
    return (
        (input_tokens - cached) * pricing["input"]
        + cached * pricing["cached_input"]
        + output * pricing["output"]
    ) / 1e6


def _cost_stats(costs: Sequence[float], successes: int) -> dict[str, Any]:
    return {
        "total_cost_usd": round(sum(costs), 4),
        "mean_cost_usd": round(statistics.fmean(costs), 6) if costs else None,
        "median_cost_usd": round(statistics.median(costs), 6) if costs else None,
        "cost_per_success_usd": (
            round(sum(costs) / successes, 6) if successes else None
        ),
    }


def cost_aggregation(
    rows: Sequence[RunRow],
    pricing: Mapping[str, float] | None = GPT54_PRICING_USD_PER_MTOK,
) -> dict[str, Any]:
    """Native cost, token totals, and the labeled codex imputation per stratum.

    Imputed and native costs are never summed together: the codex imputed
    block is separate and carries its counterfactual basis label.
    """
    entries: dict[tuple[str, str, str], Any] = {}
    for key, srows in _strata(rows).items():
        model = key[0]
        priced = [r for r in srows if r["cost_usd"] is not None]
        costs = [float(r["cost_usd"]) for r in priced]
        successes = sum(bool(r["task_success"]) for r in priced)
        totals = {
            token_key: sum(int(r["tokens"].get(token_key, 0)) for r in srows)
            for token_key in TOKEN_KEYS
        }
        block: dict[str, Any] = {
            "native": {
                "n_priced": len(priced),
                "successes_priced": successes,
                **_cost_stats(costs, successes),
            },
            "tokens": {
                "n": len(srows),
                "totals": totals,
                "mean_per_run": {
                    token_key: round(totals[token_key] / len(srows), 2)
                    for token_key in TOKEN_KEYS
                },
            },
        }
        if model == "codex":
            if pricing is None:
                block["imputed"] = {
                    "available": False,
                    "reason": "gpt-5.4 API pricing not verified",
                }
            else:
                imputed = [_impute_codex_cost(r["tokens"], pricing) for r in srows]
                imputed_successes = sum(bool(r["task_success"]) for r in srows)
                block["imputed"] = {
                    "available": True,
                    "basis": GPT54_IMPUTATION_BASIS,
                    "pricing_usd_per_mtok": dict(pricing),
                    "n": len(srows),
                    **_cost_stats(imputed, imputed_successes),
                }
        entries[key] = block
    return {
        "native_field": "cost_usd as billed by the provider (None for codex)",
        "imputed_rule": (
            "codex only; imputed and native costs are never summed together"
        ),
        "pricing_source": dict(GPT54_PRICING_SOURCE),
        "models": _nest(entries),
    }
