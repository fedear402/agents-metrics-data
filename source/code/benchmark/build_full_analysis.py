"""Build the six-arm twelve-rep analysis file for the manuscript.

The analysis sample is the first twelve eligible repetitions in every design
cell of the six model x prompt arms (decision record
2026-08-18_twelve-repetition-sample). This script loads one all-reps snapshot
through ``analysis_dataset.load_runs``, applies the shared balanced selector
in memory, and writes ``full_analysis.json``, the single data source for the
six-arm exhibits and the public site, plus two pinned sample manifests:
``sample_manifest.json`` for the 13,608-run corpus and
``paper_sample_manifest.json`` for the 9,072 rows of the two-model paper
scope. A single snapshot load guarantees that the twelve-rep sample and the
all-reps sensitivity block describe the same state of the live collection
directories.

The ``paper_scope`` block carries the headline rates of the paper's main
specification: the Claude Sonnet 4.6 and GPT-5.4 through Codex arms, pooled
and per arm (decision record 2026-08-21_two-model-main-specification). The
rest of the file stays a three-arm corpus snapshot.

``task_success`` on every loaded row is the final-artifact primary outcome
(decision record 2026-08-13_final-artifact-outcome): the repaired
``result.json`` matches the reference under the canonical tolerances and the
integrity gate passes. The scorer judges the final saved file; model-side
timeouts stay failures. The loader computes it centrally, so every block here
inherits it. The ``strict_vs_unified`` block compares the primary outcome
against the strict five-gate baseline and reports flips, repair-rule counts,
and the with-method sensitivity variant.

The ``extended.runtime`` block aggregates the harmonized
``provider_wall_seconds`` measure over the twelve-rep sample, per model arm,
agency condition, and software, and carries its own measure definition.

Deterministic and offline: same input data, byte-identical output. It performs
no model calls, no network calls, and never writes inside a collection
directory. Run from the repo root:

    uv run python code/benchmark/build_full_analysis.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analysis_dataset as ad  # noqa: E402
import estimation_outcome as eo  # noqa: E402
import extended_analyses as ea  # noqa: E402
import strict_posthoc as sp  # noqa: E402

DEFAULT_OUT_SUBDIR = Path("data") / "analysis"
OUT_NAME = "full_analysis.json"
SAMPLE_MANIFEST_NAME = "sample_manifest.json"
PAPER_SAMPLE_MANIFEST_NAME = "paper_sample_manifest.json"
DECISION_RECORD = (
    "quality_reports/decision_records/2026-08-18_twelve-repetition-sample.md"
)
PAPER_SCOPE_DECISION_RECORD = (
    "quality_reports/decision_records/2026-08-21_two-model-main-specification.md"
)
PAPER_SCOPE_NOTE = (
    "The paper's main specification covers Claude Sonnet 4.6 and GPT-5.4. "
    "Pooled numbers in the paper pool those two models. This file stays a "
    "corpus snapshot over all three models because the public site and the "
    "diagnostics read it. The paper scope is the block below and the pinned "
    "paper_sample_manifest.json."
)
PRIMARY_OUTCOME_DECISION_RECORD = (
    "quality_reports/decision_records/2026-08-13_final-artifact-outcome.md"
)
SAMPLE_RULE = (
    "first twelve eligible repetitions by repetition index per "
    "arm/task/platform/interface cell; "
    "intention-to-treat: attempted slots without a scored script within the "
    "operating cap enter as failures (provenance.itt_failure); provider-side "
    "interruptions stay excluded; the legacy corpus is primary and "
    "pinned-2026-07 fills only repetition keys with no legacy record"
)
TASK_SUCCESS_RULE = (
    "primary (final artifact): the run's result.json, after model-blind "
    "digit-preserving repair, matches the reference under the canonical "
    "per-task tolerances and the canonical integrity gate passes; execution, "
    "artifact-shape, and method-evidence gates are not part of the primary "
    "outcome; the scorer judges the final saved file and model-side timeouts "
    "stay failures. The strict five-gate verdict (execution, artifact, "
    "numeric, method, integrity; all five gates recomputed for pilot arms) is "
    "preserved per row as task_success_strict for diagnostics"
)
CRITERIA = (
    "specification_correctness",
    "executability",
    "output_correctness",
    "submission_validity",
)
NOTES = (
    "Retired few-shot demonstrations inside data/runs/codex_zero_shot and "
    "data/runs/deepseek_zero_shot are excluded by prompt_mode within those "
    "directories, never by prompt hash.",
    "data/runs/_excluded_deepseek_fewshot_strays is excluded by "
    "path; its prompt hashes match the respecified set, so a hash allowlist "
    "would wrongly admit it.",
    "All model arms report API-equivalent costs from recorded token use. Codex "
    "uses OpenAI API Standard base rates because its ChatGPT-subscription CLI "
    "records no billed dollar amount.",
    "Sonnet few-shot and pilot inline scoring blocks are never consumed for "
    "analysis; all five gates are recomputed canonically.",
    "Live arms move between builds; this file is a labeled snapshot whose "
    "per-arm max_created_at values are data derived.",
    "Manifest coverage keys (usable_runs, cells_*) describe the twelve-rep "
    "sample; the embedded attrition counters describe the all-reps snapshot "
    "load (attrition_basis). Model-side timeout failures enter the twelve-rep "
    "sample, so usable_runs = attrition.usable + itt_timeout_failure - "
    "usable_over_rep_cap per arm. The compatibility field "
    "usable_over_rep_cap counts eligible rows outside the balanced selection, "
    "including later repetitions and duplicate repetition records.",
    "The primary sample is intention-to-treat: model-side timeouts enter as "
    "failure rows on every criterion. responded_only_sensitivity recomputes "
    "the headline after dropping those rows (the pre-fix definition).",
    "Self-repair rows keep their whole-run outcome, cost, tokens, and wall "
    "time; each harness bounds revisions its own way, and "
    "extended.revision_disclosure reports the measured revision use per "
    "model arm.",
    "The strict_vs_unified block compares the primary unified outcome "
    "against the strict five-gate baseline on the same rows; the "
    "tolerance_sensitivity, criterion_cascade, and "
    "interface_gain_decomposition diagnostics operate on the strict "
    "gate anatomy.",
    "This file is a corpus snapshot over all three model arms. The paper's "
    "main specification covers the two arms named in paper_scope; the "
    "estimation and numbers chain reads that scope and pins it with "
    "data/analysis/paper_sample_manifest.json.",
)


def pct(successes: int, n: int) -> float:
    return round(100.0 * successes / n, 2) if n else 0.0


def success_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(r["task_success"]) for r in rows)
    return {
        "n": len(rows),
        "successes": successes,
        "success_pct": pct(successes, len(rows)),
    }


def _rows_for(
    rows: Sequence[Mapping[str, Any]], model: str, prompt: str
) -> list[Mapping[str, Any]]:
    return [r for r in rows if r["model"] == model and r["prompt"] == prompt]


def _arm_absent(reports: Mapping[str, Any], arm_id: str) -> bool:
    return not reports[arm_id].present


def _stats_for(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "overall": success_block(rows),
        "by_interface": {
            ad.INTERFACE_LABELS[i]: success_block(
                [r for r in rows if r["interface"] == i]
            )
            for i in ad.INTERFACES
        },
        "by_platform": {
            p: success_block([r for r in rows if r["platform"] == p])
            for p in ad.PLATFORMS
        },
    }


def _headline_section(
    rows: Sequence[Mapping[str, Any]], reports: Mapping[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm_id, spec in ad.ARMS.items():
        if arm_id not in reports:
            continue
        entry = out.setdefault(spec.model, {})
        if _arm_absent(reports, arm_id):
            entry[spec.prompt_mode] = {"status": "absent"}
            continue
        entry[spec.prompt_mode] = _stats_for(
            _rows_for(rows, spec.model, spec.prompt_mode)
        )
    return out


def _paper_scope_section(
    five_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Headline task success on the two-model paper scope.

    ``pooled`` pools the two paper arms, which is what "pooled" means in the
    manuscript. The per-arm blocks repeat the same cuts so a reader never has
    to add the three-arm blocks back together.
    """
    scope_rows = ad.paper_rows(five_rows)
    return {
        "decision_record": PAPER_SCOPE_DECISION_RECORD,
        "note": PAPER_SCOPE_NOTE,
        "models": list(ad.PAPER_MODELS),
        "arm_ids": list(ad.PAPER_ARM_IDS),
        "sample_version": ad.PAPER_SAMPLE_VERSION,
        "n": len(scope_rows),
        "pooled": _stats_for(scope_rows),
        "pooled_by_prompt": {
            prompt: _stats_for([r for r in scope_rows if r["prompt"] == prompt])
            for prompt in ad.PROMPTS
        },
        "by_model": {
            model: {
                "overall": _stats_for([r for r in scope_rows if r["model"] == model]),
                "by_prompt": {
                    prompt: _stats_for(
                        [
                            r
                            for r in scope_rows
                            if r["model"] == model and r["prompt"] == prompt
                        ]
                    )
                    for prompt in ad.PROMPTS
                },
            }
            for model in ad.PAPER_MODELS
        },
    }


def _nine_cell_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for platform in ad.PLATFORMS:
        for interface in ad.INTERFACES:
            cell = [
                r
                for r in rows
                if r["platform"] == platform and r["interface"] == interface
            ]
            costs = [r["cost_usd"] for r in cell if r["cost_usd"] is not None]
            imputed_costs = (
                [
                    ea._impute_codex_cost(r["tokens"], ea.GPT54_PRICING_USD_PER_MTOK)
                    for r in cell
                ]
                if cell
                and cell[0]["model"] == "codex"
                and ea.GPT54_PRICING_USD_PER_MTOK is not None
                else []
            )
            out.append(
                {
                    "platform": platform,
                    "interface": interface,
                    "interface_label": ad.INTERFACE_LABELS[interface],
                    **success_block(cell),
                    "n_priced": len(costs),
                    "median_cost_usd": (
                        round(statistics.median(costs), 6) if costs else None
                    ),
                    "mean_cost_usd": (
                        round(statistics.fmean(costs), 6) if costs else None
                    ),
                    "median_imputed_cost_usd": (
                        round(statistics.median(imputed_costs), 6)
                        if imputed_costs
                        else None
                    ),
                    "mean_imputed_cost_usd": (
                        round(statistics.fmean(imputed_costs), 6)
                        if imputed_costs
                        else None
                    ),
                }
            )
    return out


def _nine_cell_section(
    rows: Sequence[Mapping[str, Any]], reports: Mapping[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm_id, spec in ad.ARMS.items():
        if arm_id not in reports:
            continue
        entry = out.setdefault(spec.model, {})
        if _arm_absent(reports, arm_id):
            entry[spec.prompt_mode] = {"status": "absent"}
            continue
        entry[spec.prompt_mode] = _nine_cell_rows(
            _rows_for(rows, spec.model, spec.prompt_mode)
        )
    return out


def _criteria_by_interface(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for interface in ad.INTERFACES:
        subset = [r for r in rows if r["interface"] == interface]
        entry: dict[str, Any] = {"n": len(subset)}
        for criterion in CRITERIA:
            passes = sum(bool(r["criteria"][criterion]) for r in subset)
            entry[f"{criterion}_pct"] = pct(passes, len(subset))
        entry["task_success_pct"] = success_block(subset)["success_pct"]
        strict = sum(
            bool(r.get("task_success_strict", r["task_success"])) for r in subset
        )
        entry["task_success_strict_pct"] = pct(strict, len(subset))
        out[ad.INTERFACE_LABELS[interface]] = entry
    return out


def _criteria_section(
    rows: Sequence[Mapping[str, Any]], reports: Mapping[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm_id, spec in ad.ARMS.items():
        if arm_id not in reports:
            continue
        entry = out.setdefault(spec.model, {})
        if _arm_absent(reports, arm_id):
            entry[spec.prompt_mode] = {"status": "absent"}
            continue
        entry[spec.prompt_mode] = _criteria_by_interface(
            _rows_for(rows, spec.model, spec.prompt_mode)
        )
    return out


def _cell_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (row["task"], row["platform"], row["interface"])


def _interface_pp(
    few_rows: Sequence[Mapping[str, Any]], zero_rows: Sequence[Mapping[str, Any]]
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for interface in ad.INTERFACES:
        few = [r for r in few_rows if r["interface"] == interface]
        zero = [r for r in zero_rows if r["interface"] == interface]
        if not few or not zero:
            out[ad.INTERFACE_LABELS[interface]] = None
            continue
        out[ad.INTERFACE_LABELS[interface]] = round(
            success_block(few)["success_pct"] - success_block(zero)["success_pct"], 2
        )
    return out


def _contrast_block(
    few_rows: Sequence[Mapping[str, Any]], zero_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    few_pct = success_block(few_rows)["success_pct"]
    zero_pct = success_block(zero_rows)["success_pct"]
    return {
        "zero_shot_success_pct": zero_pct,
        "few_shot_success_pct": few_pct,
        "overall_pp": round(few_pct - zero_pct, 2),
        "by_interface_pp": _interface_pp(few_rows, zero_rows),
    }


def _manifest_status(manifest: Mapping[str, Any], arm_id: str) -> str:
    return str(manifest.get("arms", {}).get(arm_id, {}).get("status", "unknown"))


def _prompt_contrast_for_model(
    model: str,
    five_rows: Sequence[Mapping[str, Any]],
    reports: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    statuses: dict[str, str] = {}
    rows_by_prompt: dict[str, list[Mapping[str, Any]]] = {}
    for prompt in ad.PROMPTS:
        arm_id = f"{model}_{prompt}"
        rows_by_prompt[prompt] = _rows_for(five_rows, model, prompt)
        if arm_id not in reports:
            statuses[prompt] = "not_loaded"
        elif _arm_absent(reports, arm_id) or not rows_by_prompt[prompt]:
            statuses[prompt] = "absent"
        else:
            statuses[prompt] = _manifest_status(manifest, arm_id)
    if statuses["few_shot"] in ("absent", "not_loaded"):
        return {"status": f"few_shot_{statuses['few_shot']}", "zero_shot_only": True}
    if statuses["zero_shot"] in ("absent", "not_loaded"):
        return {"status": f"zero_shot_{statuses['zero_shot']}", "few_shot_only": True}
    zero_rows = rows_by_prompt["zero_shot"]
    few_rows = rows_by_prompt["few_shot"]
    shared = set(map(_cell_key, zero_rows)) & set(map(_cell_key, few_rows))
    z_shared = [r for r in zero_rows if _cell_key(r) in shared]
    f_shared = [r for r in few_rows if _cell_key(r) in shared]
    expected_tp = len(ad.ANALYSIS_TASK_IDS) * len(ad.PLATFORMS)
    parts = []
    for prompt in ad.PROMPTS:
        if statuses[prompt] == "complete":
            continue
        tp = len({(r["task"], r["platform"]) for r in rows_by_prompt[prompt]})
        parts.append(
            f"{prompt}_{statuses[prompt]} ({tp} of {expected_tp} task x platform cells)"
        )
    return {
        "unrestricted": _contrast_block(few_rows, zero_rows),
        "shared_cells_only": {
            "n_shared_cells": len(shared),
            **_contrast_block(f_shared, z_shared),
        },
        "status": "ok" if not parts else "; ".join(parts),
    }


def _prompt_contrasts_section(
    five_rows: Sequence[Mapping[str, Any]],
    reports: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    models = []
    for arm_id, spec in ad.ARMS.items():
        if arm_id in reports and spec.model not in models:
            models.append(spec.model)
    return {
        model: _prompt_contrast_for_model(model, five_rows, reports, manifest)
        for model in models
    }


RUNTIME_NOTE = (
    "provider_wall_seconds is the harmonized runtime measure: the sum of "
    "wall_seconds over the arm's provider blocks (claude for Sonnet, codex "
    "for Codex, openrouter for DeepSeek), so it covers model-call time only "
    "and is comparable across arms. Self-repair runs count every attempt, "
    "matching the whole-run cost and token accounting. "
    "wall_seconds keeps its per-arm historical definition and is not "
    "comparable across arms: Sonnet rows report the same provider-call sum, "
    "while Codex and DeepSeek rows prefer the record's total_wall_seconds, "
    "which also covers local script execution and harness overhead. Read "
    "median_wall_seconds within a model arm, never across model arms. "
    "Runtime reflects operating conditions during collection (retries, "
    "provider queueing, local machine load) and is not a model speed "
    "benchmark."
)


def _runtime_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    provider = [
        float(r["provider_wall_seconds"])
        for r in rows
        if r.get("provider_wall_seconds") is not None
    ]
    total = [
        float(r["wall_seconds"]) for r in rows if r.get("wall_seconds") is not None
    ]
    return {
        "n": len(rows),
        "n_with_provider_wall": len(provider),
        "median_provider_wall_seconds": (
            round(statistics.median(provider), 1) if provider else None
        ),
        "mean_provider_wall_seconds": (
            round(statistics.fmean(provider), 1) if provider else None
        ),
        "total_provider_wall_hours": round(sum(provider) / 3600.0, 2),
        "n_with_wall": len(total),
        "median_wall_seconds": round(statistics.median(total), 1) if total else None,
    }


def _runtime_section(five_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Runtime aggregation on the twelve-rep sample, per model arm and stratum.

    Every entry reports the harmonized ``provider_wall_seconds`` median and
    mean plus the historical ``wall_seconds`` median. The pooled block covers
    the whole analyzed sample.
    """
    models = [m for m in ad.MODELS if any(r["model"] == m for r in five_rows)]
    by_model: dict[str, Any] = {}
    for model in models:
        model_rows = [r for r in five_rows if r["model"] == model]
        by_model[model] = {
            "overall": _runtime_stats(model_rows),
            "by_interface": {
                ad.INTERFACE_LABELS[interface]: _runtime_stats(
                    [r for r in model_rows if r["interface"] == interface]
                )
                for interface in ad.INTERFACES
            },
            "by_platform": {
                platform: _runtime_stats(
                    [r for r in model_rows if r["platform"] == platform]
                )
                for platform in ad.PLATFORMS
            },
            "by_prompt": {
                prompt: _runtime_stats(
                    [r for r in model_rows if r["prompt"] == prompt]
                )
                for prompt in ad.PROMPTS
            },
        }
    pooled = _runtime_stats(five_rows)
    return {
        "note": RUNTIME_NOTE,
        "measure": "provider_wall_seconds",
        "sample": "twelve-rep analysis sample",
        "pooled": pooled,
        "pooled_by_interface": {
            ad.INTERFACE_LABELS[interface]: _runtime_stats(
                [r for r in five_rows if r["interface"] == interface]
            )
            for interface in ad.INTERFACES
        },
        "pooled_by_platform": {
            platform: _runtime_stats(
                [r for r in five_rows if r["platform"] == platform]
            )
            for platform in ad.PLATFORMS
        },
        "models": by_model,
        "total_provider_wall_hours": pooled["total_provider_wall_hours"],
        "total_provider_wall_hours_by_model": {
            model: by_model[model]["overall"]["total_provider_wall_hours"]
            for model in models
        },
    }


REVISION_DISCLOSURE_NOTE = (
    "Self-repair revision use per model arm on the analysis sample. "
    "revisions_used = provider attempts minus one, floor zero; rows with an "
    "unknown attempt count are excluded from the share and reported under "
    "n_unknown. share_gt1_pct is the share of self-repair rows with known "
    "attempt counts that used more than one revision; max_revisions is the "
    "largest revision count observed in the arm."
)


def _revision_disclosure(
    sample_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measured self-repair revision use, per model arm and prompt condition."""

    def block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        repair = [r for r in rows if r["interface"] == "self_repair"]
        known = [
            int(r["revisions_used"])
            for r in repair
            if r.get("revisions_used") is not None
        ]
        gt1 = sum(1 for value in known if value > 1)
        return {
            "n_self_repair": len(repair),
            "n_known": len(known),
            "n_unknown": len(repair) - len(known),
            "n_gt1_revision": gt1,
            "share_gt1_pct": pct(gt1, len(known)),
            "max_revisions": max(known) if known else None,
        }

    out: dict[str, Any] = {"note": REVISION_DISCLOSURE_NOTE, "models": {}}
    for model in ad.MODELS:
        model_rows = [r for r in sample_rows if r["model"] == model]
        if not model_rows:
            continue
        out["models"][model] = {
            "overall": block(model_rows),
            "by_prompt": {
                prompt: block([r for r in model_rows if r["prompt"] == prompt])
                for prompt in ad.PROMPTS
            },
        }
    return out


def _extended_section(
    five_rows: Sequence[Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
    pricing: Mapping[str, float] | None,
) -> dict[str, Any]:
    return {
        "pass_at_k": ea.pass_at_k(all_rows),
        "majority_vote": ea.majority_vote(five_rows),
        "silent_failure": ea.silent_failure(five_rows),
        "criterion_cascade": ea.criterion_cascade(five_rows),
        "interface_gain_decomposition": {
            "chatbot_to_agent": ea.interface_gain_decomposition(
                five_rows, "no_feedback", "tool_loop"
            ),
            "chatbot_to_one_repair": ea.interface_gain_decomposition(
                five_rows, "no_feedback", "self_repair"
            ),
        },
        "tolerance_sensitivity": ea.tolerance_sensitivity(five_rows),
        "per_cell_dispersion": ea.per_cell_dispersion(all_rows),
        "turn_distributions": ea.turn_distributions(five_rows),
        "cost": ea.cost_aggregation(five_rows, pricing=pricing),
        "runtime": _runtime_section(five_rows),
        "revision_disclosure": _revision_disclosure(five_rows),
    }


def annotate_attrition_basis(
    manifest: Mapping[str, Any],
    rows_all_reps: Sequence[Mapping[str, Any]],
    max_reps: int,
) -> None:
    """Label the manifest's attrition counters with their all-reps basis.

    The coverage keys (``usable_runs``, ``cells_*``) describe the twelve-rep
    sample, while the embedded attrition counters come from the single
    all-reps snapshot load. Model-side timeout failures are materialized in
    the twelve-rep sample. ``usable_over_rep_cap`` is retained for output-schema
    compatibility and counts eligible rows outside the balanced selection. It
    reconciles the two:
    ``usable_runs == attrition.usable + itt_timeout_failure -
    usable_over_rep_cap`` per arm.
    """
    selected = ad.select_first_eligible_rows(rows_all_reps, max_reps)
    all_by_arm = _n_by_arm(rows_all_reps, manifest.get("arms", {}))
    selected_by_arm = _n_by_arm(selected, manifest.get("arms", {}))
    for arm_id, arm in manifest.get("arms", {}).items():
        arm["attrition_basis"] = "all-reps snapshot load"
        arm["usable_over_rep_cap"] = all_by_arm.get(arm_id, 0) - selected_by_arm.get(
            arm_id, 0
        )


def _n_by_arm(
    rows: Sequence[Mapping[str, Any]], reports: Mapping[str, Any]
) -> dict[str, int]:
    return {
        arm_id: sum(1 for r in rows if r["provenance"]["arm_id"] == arm_id)
        for arm_id in ad.ARMS
        if arm_id in reports
    }


def build_payload(
    rows_all_reps: Sequence[Mapping[str, Any]],
    reports: Mapping[str, Any],
    manifest: Mapping[str, Any],
    max_reps: int,
    pricing: Mapping[str, float] | None,
) -> dict[str, Any]:
    """Assemble the full_analysis.json payload. Deterministic.

    ``rows_all_reps`` is the single all-reps snapshot. The shared balanced
    selector derives the twelve-rep sample in memory, so both samples always
    describe the same snapshot of the live collection directories.

    Every block is a pure function of the loaded rows: the unified primary
    outcome and its repair detail are annotated on each row by the loader, so
    ``strict_vs_unified`` formats stored fields without re-reading any run
    directory.
    """

    five_rows = ad.select_first_eligible_rows(rows_all_reps, max_reps)
    return {
        "generated_by": "code/benchmark/build_full_analysis.py",
        "decision_record": DECISION_RECORD,
        "primary_outcome_decision_record": PRIMARY_OUTCOME_DECISION_RECORD,
        "scoring_version": sp.SCORING_VERSION,
        "primary_outcome_version": ad.PRIMARY_OUTCOME_VERSION,
        "sample_rule": SAMPLE_RULE,
        "task_success_rule": TASK_SUCCESS_RULE,
        "gpt54_pricing": {
            "prices_usd_per_mtok": dict(pricing) if pricing is not None else None,
            "source": dict(ea.GPT54_PRICING_SOURCE),
        },
        "manifest": manifest,
        "paper_scope": _paper_scope_section(five_rows),
        "headline": _headline_section(five_rows, reports),
        "nine_cell": _nine_cell_section(five_rows, reports),
        "criteria": _criteria_section(five_rows, reports),
        "prompt_contrasts": _prompt_contrasts_section(five_rows, reports, manifest),
        "extended": _extended_section(five_rows, rows_all_reps, pricing),
        "all_reps_sensitivity": {
            "note": "headline success recomputed on all usable reps (unbalanced)",
            "headline": _headline_section(rows_all_reps, reports),
            "n_by_arm": _n_by_arm(rows_all_reps, reports),
        },
        "responded_only_sensitivity": {
            "note": (
                "headline success recomputed after dropping intention-to-treat "
                "failure rows (the pre-fix responded-only sample definition)"
            ),
            "headline": _headline_section(
                [r for r in five_rows if not r["provenance"].get("itt_failure")],
                reports,
            ),
            "n_itt_rows_dropped": sum(
                1 for r in five_rows if r["provenance"].get("itt_failure")
            ),
        },
        "strict_vs_unified": eo.summarize(eo.score_rows(five_rows)),
        "notes": list(NOTES),
    }


def _print_summary(
    payload: Mapping[str, Any],
    five_rows: Sequence[Mapping[str, Any]],
    max_reps: int,
    out_path: Path,
) -> None:
    manifest = payload["manifest"]
    expected = ad.EXPECTED_CELLS_PER_ARM * max_reps
    for arm_id, arm in manifest.get("arms", {}).items():
        if not arm.get("present", False):
            print(f"{arm_id}: absent")
            continue
        arm_rows = [r for r in five_rows if r["provenance"]["arm_id"] == arm_id]
        by_iface = []
        for interface in ad.INTERFACES:
            subset = [r for r in arm_rows if r["interface"] == interface]
            successes = sum(bool(r["task_success"]) for r in subset)
            label = ad.INTERFACE_LABELS[interface]
            by_iface.append(f"{label} {successes}/{len(subset)}")
        print(
            f"{arm_id}: {arm.get('status', 'unknown')} "
            f"usable {arm.get('usable_runs', len(arm_rows))}/{expected} "
            f"empty_cells {len(arm.get('empty_cells', []))} "
            f"success_by_interface [{', '.join(by_iface)}]"
        )
    prices = payload["gpt54_pricing"]["prices_usd_per_mtok"]
    if prices is None:
        print("gpt-5.4 pricing: unavailable (imputed codex cost not reported)")
    else:
        source = payload["gpt54_pricing"]["source"]
        print(
            f"gpt-5.4 pricing: input {prices['input']}/cached "
            f"{prices['cached_input']}/output {prices['output']} USD per MTok "
            f"({source.get('tier', '?')} tier, accessed "
            f"{source.get('accessed', '?')})"
        )
    print(f"wrote {out_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the six-arm twelve-rep full_analysis.json snapshot."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            f"output directory; default <project-root>/{DEFAULT_OUT_SUBDIR.as_posix()}"
        ),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list(ad.ARMS),
        default=None,
        help="arm ids to load (default: all six)",
    )
    parser.add_argument(
        "--max-reps",
        type=int,
        default=12,
        help="eligible repetitions retained per design cell (default 12)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="repo root override for tests (default: the live repo root)",
    )
    parser.add_argument(
        "--corpus-source",
        choices=["records", "report"],
        default=None,
        help=(
            "storage path for the Sonnet zero-shot arm: per-run score records "
            "(default) or the derived aggregate report (audit path)"
        ),
    )
    args = parser.parse_args(argv)

    project_root = (args.project_root or ad.PROJECT_ROOT).resolve()
    out_dir = (
        args.out_dir if args.out_dir is not None else project_root / DEFAULT_OUT_SUBDIR
    )

    # Single snapshot load; the twelve-rep sample is filtered in memory so live
    # arm growth between two loads can never split the two samples.
    result = ad.load_runs(
        arm_ids=args.arms,
        max_reps=None,
        project_root=project_root,
        corpus_source=args.corpus_source,
    )
    five_rows = ad.select_first_eligible_rows(result.rows, args.max_reps)
    manifest = ad.completeness_manifest(five_rows, result.reports, args.max_reps)
    annotate_attrition_basis(manifest, result.rows, args.max_reps)
    payload = build_payload(
        result.rows,
        result.reports,
        manifest,
        args.max_reps,
        pricing=ea.GPT54_PRICING_USD_PER_MTOK,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_NAME
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    if args.arms is None:
        # The pinned sample manifest covers all six arms; a partial build
        # must never overwrite it.
        sample_manifest = ad.build_sample_manifest(
            five_rows, args.max_reps, project_root=project_root
        )
        sample_manifest_path = out_dir / SAMPLE_MANIFEST_NAME
        sample_manifest_path.write_text(json.dumps(sample_manifest, indent=2) + "\n")
        print(
            f"wrote {sample_manifest_path} "
            f"({sample_manifest['n_rows']} pinned runs, "
            f"sample {sample_manifest['sample_version']})"
        )
        # The paper pin sits next to the corpus pin. The estimation and
        # numbers chain verifies against this file on every build.
        paper_manifest = ad.build_paper_sample_manifest(
            five_rows, args.max_reps, project_root=project_root
        )
        if (
            args.max_reps == ad.DEFAULT_MAX_REPS
            and paper_manifest["n_rows"] != ad.EXPECTED_PAPER_RUNS_12REP
        ):
            raise SystemExit(
                "paper sample size "
                f"{paper_manifest['n_rows']} != "
                f"{ad.EXPECTED_PAPER_RUNS_12REP}; refusing to pin"
            )
        paper_manifest_path = out_dir / PAPER_SAMPLE_MANIFEST_NAME
        paper_manifest_path.write_text(json.dumps(paper_manifest, indent=2) + "\n")
        print(
            f"wrote {paper_manifest_path} "
            f"({paper_manifest['n_rows']} pinned runs, "
            f"sample {paper_manifest['sample_version']})"
        )
    _print_summary(payload, five_rows, args.max_reps, out_path)


if __name__ == "__main__":
    main()
