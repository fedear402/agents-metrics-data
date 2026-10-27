"""Build ``manuscript_numbers.json`` for the manuscript.

Scope: the paper's main specification covers two model arms, Claude Sonnet 4.6
and GPT-5.4 through Codex (decision record
2026-08-21_two-model-main-specification). Everything this file writes stays
inside that scope, and "pooled" means the two arms pooled. The run corpus and
``full_analysis.json`` keep all three arms; this builder selects the paper
arms out of them.

Assembles every number the manuscript writers quote into one file: factorial
contrasts per arm and pooled, twelve-rep headline rates per arm and pooled,
Sonnet per-task task-success rates, difficulty thirds ranked by chatbot
zero-shot success, the three failure-archetype task rates, cost aggregates
with two-decimal display strings, prompt contrasts per arm, silent-failure
rates, the pass@k tables, tolerance sensitivity, the constrained-agent
turn-distribution summary, the runtime medians, and the self-repair revision
disclosure.

Two pass@k blocks are written. ``pass_at_k_sonnet`` is the older all-reps
Sonnet block copied from the snapshot. ``pass_at_k_paper`` is the paper block
that ``code/benchmark/pass_at_k.py`` computes on the twelve-repetition paper
sample, by model and agency condition with the prompt conditions pooled, for
k from 1 to 12.

Sources, in order of authority:

* ``full_analysis.json``: the corpus twelve-rep snapshot (prompt contrasts,
  silent failure, pass@k, tolerance sensitivity, per-interface cost blocks),
  read for the paper arms only;
* ``factorial_fiverep.json``: the Sonnet 2x3x3 inference run;
* ``factorial_by_model.json``: the paper inference run, read for the per-arm
  and pooled contrasts, the joint tests, the software-gap diagnostics, and the
  cross-arm differences (section ``factorial_extensions``);
* ``analysis_dataset.load_paper_runs``: the same first-twelve-eligible paper
  sample re-loaded for the row-level cuts the snapshot does not carry
  (per-task rates, difficulty thirds, per-platform cost, turn shares). The
  load verifies the 9,072 rows against ``paper_sample_manifest.json``.

The script asserts that the re-loaded sample reproduces the snapshot headline
counts and per-interface cost statistics exactly, so all three sources
describe the same sample. Cost provenance labels are attached where costs are
reported: Sonnet dollars are client-side estimates and GPT-5.4 dollars are an
imputed API counterfactual. Imputed and native costs are never merged.

Deterministic and offline. Run from the repo root:

    uv run python code/benchmark/build_manuscript_numbers.py
"""

from __future__ import annotations

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
import pass_at_k as pak  # noqa: E402

OUT_DIR = ad.PROJECT_ROOT / "data" / "analysis"
SNAPSHOT_PATH = OUT_DIR / "full_analysis.json"
FACTORIAL_PATH = OUT_DIR / "factorial_fiverep.json"
FACTORIAL_BY_MODEL_PATH = OUT_DIR / "factorial_by_model.json"
OUT_PATH = OUT_DIR / "manuscript_numbers.json"

#: The paper scope: two model arms, four arm ids.
MODELS = ad.PAPER_MODELS
ARM_IDS = ad.PAPER_ARM_IDS
MAX_REPS = 12
INTERFACE_LABELS = ad.INTERFACE_LABELS
LABELS = tuple(INTERFACE_LABELS[i] for i in ad.INTERFACES)
PROMPTS = ad.PROMPTS
PLATFORMS = ad.PLATFORMS
TURN_CAP = 12
PAPER_SCOPE_DECISION_RECORD = (
    "quality_reports/decision_records/2026-08-21_two-model-main-specification.md"
)
AGENT_TURN_CAVEAT = (
    "the constrained-agent condition runs under the same 600-second session "
    "wall for both arms, and no retained Sonnet run is bound by a model-turn "
    "cap; the runs the cap once stopped were recollected without it (decision "
    "record 2026-08-24_sonnet-rerun-replacements)"
)
ARCHETYPES = {
    "dynpanel_ab": "platform",
    "staggered_did": "interface_label",
    "anova_oneway": "platform",
}

SONNET_COST_LABEL = (
    "client-side estimate reported by the Claude CLI; not a provider invoice"
)

RunRow = Mapping[str, Any]


def pct(successes: int, n: int) -> float:
    return round(100.0 * successes / n, 2) if n else 0.0


def fmt2(value: float | None) -> str | None:
    return None if value is None else f"{value:.2f}"


def success_block(rows: Sequence[RunRow]) -> dict[str, Any]:
    successes = sum(bool(r["task_success"]) for r in rows)
    return {
        "n": len(rows),
        "successes": successes,
        "success_pct": pct(successes, len(rows)),
    }


def subset(rows: Sequence[RunRow], **field_values: str) -> list[RunRow]:
    return [
        r
        for r in rows
        if all(r[field] == value for field, value in field_values.items())
    ]


def stats_block(rows: Sequence[RunRow]) -> dict[str, Any]:
    return {
        "overall": success_block(rows),
        "by_interface": {
            label: success_block(subset(rows, interface_label=label))
            for label in LABELS
        },
        "by_platform": {
            platform: success_block(subset(rows, platform=platform))
            for platform in PLATFORMS
        },
    }


def headline_section(rows: Sequence[RunRow]) -> dict[str, Any]:
    return {
        "zero_shot": stats_block(subset(rows, prompt="zero_shot")),
        "few_shot": stats_block(subset(rows, prompt="few_shot")),
        "pooled_prompts": stats_block(list(rows)),
    }


def per_task_section(rows: Sequence[RunRow]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in ad.ANALYSIS_TASK_IDS:
        task_rows = subset(rows, task=task)
        out[task] = {
            "pooled": success_block(task_rows),
            "by_prompt": {
                prompt: success_block(subset(task_rows, prompt=prompt))
                for prompt in PROMPTS
            },
            "by_interface": {
                label: success_block(subset(task_rows, interface_label=label))
                for label in LABELS
            },
        }
    return out


def per_task_model_platform_section(rows: Sequence[RunRow]) -> dict[str, Any]:
    """Task success by model, prompt, platform, and interface.

    Empty cells remain explicit through a zero-record success block. This lets
    displays distinguish unavailable pilot coverage from a zero success rate.
    """
    return {
        model: {
            prompt: {
                task: {
                    platform: {
                        label: success_block(
                            subset(
                                rows,
                                model=model,
                                prompt=prompt,
                                task=task,
                                platform=platform,
                                interface_label=label,
                            )
                        )
                        for label in LABELS
                    }
                    for platform in PLATFORMS
                }
                for task in ad.ANALYSIS_TASK_IDS
            }
            for prompt in PROMPTS
        }
        for model in MODELS
    }


def _task_rate(rows: Sequence[RunRow]) -> float:
    return success_block(rows)["success_pct"]


def difficulty_thirds(rows: Sequence[RunRow]) -> dict[str, Any]:
    """Tasks ranked by chatbot zero-shot success, split 7/7/7, agent gains.

    Ranking pools the three platforms. ``mean_task_gain_pp`` is the unweighted
    mean over tasks of the per-task constrained-agent minus chatbot gap (the
    manuscript figure convention); ``pooled_gain_pp`` pools runs within the
    third instead.
    """
    ranked = sorted(
        ad.ANALYSIS_TASK_IDS,
        key=lambda task: (
            -_task_rate(
                subset(rows, task=task, prompt="zero_shot", interface_label="chatbot")
            ),
            task,
        ),
    )
    thirds = {"easiest": ranked[:7], "middle": ranked[7:14], "hardest": ranked[14:]}
    out: dict[str, Any] = {
        "ranking": (
            "tasks ranked by Sonnet chatbot zero-shot task success, pooled over "
            "platforms; ties broken by task id; thirds of 7"
        )
    }
    for name, tasks in thirds.items():
        entry: dict[str, Any] = {"tasks": tasks}
        for prompt in PROMPTS:
            chatbot = [
                subset(rows, task=task, prompt=prompt, interface_label="chatbot")
                for task in tasks
            ]
            agent = [
                subset(
                    rows, task=task, prompt=prompt, interface_label="constrained agent"
                )
                for task in tasks
            ]
            gains = [
                _task_rate(a) - _task_rate(c)
                for a, c in zip(agent, chatbot, strict=True)
            ]
            pooled_chatbot = [r for rs in chatbot for r in rs]
            pooled_agent = [r for rs in agent for r in rs]
            entry[prompt] = {
                "chatbot": success_block(pooled_chatbot),
                "constrained_agent": success_block(pooled_agent),
                "mean_task_gain_pp": round(statistics.fmean(gains), 2),
                "pooled_gain_pp": round(
                    _task_rate(pooled_agent) - _task_rate(pooled_chatbot), 2
                ),
            }
        out[name] = entry
    return out


def archetype_section(rows: Sequence[RunRow]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task, field in ARCHETYPES.items():
        task_rows = subset(rows, task=task)
        levels = PLATFORMS if field == "platform" else LABELS
        axis = "by_platform" if field == "platform" else "by_interface"
        out[task] = {
            axis: {
                level: {
                    "pooled": success_block(subset(task_rows, **{field: level})),
                    **{
                        prompt: success_block(
                            subset(task_rows, prompt=prompt, **{field: level})
                        )
                        for prompt in PROMPTS
                    },
                }
                for level in levels
            }
        }
    return out


def _cost_stats(rows: Sequence[RunRow]) -> dict[str, Any]:
    priced = [r for r in rows if r["cost_usd"] is not None]
    costs = [float(r["cost_usd"]) for r in priced]
    successes = sum(bool(r["task_success"]) for r in priced)
    mean = round(statistics.fmean(costs), 6) if costs else None
    median = round(statistics.median(costs), 6) if costs else None
    per_success = round(sum(costs) / successes, 6) if successes else None
    return {
        "n_priced": len(priced),
        "successes_priced": successes,
        "mean_cost_usd": mean,
        "median_cost_usd": median,
        "cost_per_success_usd": per_success,
        "mean_cost_usd_2dp": fmt2(mean),
        "median_cost_usd_2dp": fmt2(median),
        "cost_per_success_usd_2dp": fmt2(per_success),
    }


def _conversion_price(rows: Sequence[RunRow]) -> dict[str, Any]:
    chatbot = _cost_stats(subset(rows, interface_label="chatbot"))
    agent = _cost_stats(subset(rows, interface_label="constrained agent"))
    chatbot_rate = success_block(subset(rows, interface_label="chatbot"))["success_pct"]
    agent_rate = success_block(subset(rows, interface_label="constrained agent"))[
        "success_pct"
    ]
    delta_cost = agent["mean_cost_usd"] - chatbot["mean_cost_usd"]
    delta_pp = round(agent_rate - chatbot_rate, 2)
    price = round(delta_cost / (delta_pp / 100.0), 6)
    return {
        "definition": (
            "mean per-run cost difference divided by the task-success rate "
            "difference, chatbot to constrained agent"
        ),
        "delta_success_pp": delta_pp,
        "delta_mean_cost_usd": round(delta_cost, 6),
        "usd_per_additional_success": price,
        "usd_per_additional_success_2dp": fmt2(price),
    }


def _cents(usd: float | None) -> float | None:
    return None if usd is None else round(usd * 100.0, 2)


def _few_shot_cost_increment(rows: Sequence[RunRow]) -> dict[str, Any]:
    """Few-shot minus zero-shot cost per run, overall and by agency condition.

    Reported in dollars and in cents, because the increment is a fraction of a
    cent to a few cents per run. Mean and median are both carried; the mean is
    the one that adds up over a run budget.
    """
    zero = subset(rows, prompt="zero_shot")
    few = subset(rows, prompt="few_shot")

    def delta(zero_rows: Sequence[RunRow], few_rows: Sequence[RunRow]) -> dict[str, Any]:
        z = _cost_stats(zero_rows)
        f = _cost_stats(few_rows)
        mean_delta = (
            None
            if z["mean_cost_usd"] is None or f["mean_cost_usd"] is None
            else round(f["mean_cost_usd"] - z["mean_cost_usd"], 6)
        )
        median_delta = (
            None
            if z["median_cost_usd"] is None or f["median_cost_usd"] is None
            else round(f["median_cost_usd"] - z["median_cost_usd"], 6)
        )
        return {
            "zero_shot_mean_cost_usd": z["mean_cost_usd"],
            "few_shot_mean_cost_usd": f["mean_cost_usd"],
            "mean_increment_usd": mean_delta,
            "mean_increment_cents": _cents(mean_delta),
            "zero_shot_median_cost_usd": z["median_cost_usd"],
            "few_shot_median_cost_usd": f["median_cost_usd"],
            "median_increment_usd": median_delta,
            "median_increment_cents": _cents(median_delta),
        }

    return {
        "definition": (
            "few-shot minus zero-shot cost per run on the same design cells; "
            "cents are the dollar value times 100"
        ),
        "overall": delta(zero, few),
        "by_interface": {
            label: delta(
                subset(zero, interface_label=label), subset(few, interface_label=label)
            )
            for label in LABELS
        },
    }


def sonnet_cost_section(rows: Sequence[RunRow]) -> dict[str, Any]:
    out: dict[str, Any] = {"label": SONNET_COST_LABEL}
    for prompt in (*PROMPTS, "pooled_prompts"):
        prows = (
            list(rows) if prompt == "pooled_prompts" else subset(rows, prompt=prompt)
        )
        out[prompt] = {
            "by_interface": {
                label: _cost_stats(subset(prows, interface_label=label))
                for label in LABELS
            },
            "by_platform": {
                platform: _cost_stats(subset(prows, platform=platform))
                for platform in PLATFORMS
            },
            "conversion_price_chatbot_to_constrained_agent": _conversion_price(prows),
        }
    out["few_shot_increment"] = _few_shot_cost_increment(rows)
    return out


def _snap_cost(snap: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    return snap["extended"]["cost"]["models"][model]


def gpt54_cost_section(snap: Mapping[str, Any]) -> dict[str, Any]:
    blocks = _snap_cost(snap, "codex")
    first = blocks["zero_shot"]["chatbot"]["imputed"]
    out: dict[str, Any] = {
        "label": first["basis"],
        "pricing_usd_per_mtok": first["pricing_usd_per_mtok"],
        "pricing_source": snap["gpt54_pricing"]["source"],
        "never_merge_with_native": True,
    }
    for prompt in PROMPTS:
        out[prompt] = {
            "by_interface": {
                label: {
                    "imputed_cost_per_success_usd": blocks[prompt][label]["imputed"][
                        "cost_per_success_usd"
                    ],
                    "imputed_cost_per_success_usd_2dp": fmt2(
                        blocks[prompt][label]["imputed"]["cost_per_success_usd"]
                    ),
                    "imputed_mean_cost_usd": blocks[prompt][label]["imputed"][
                        "mean_cost_usd"
                    ],
                    "imputed_median_cost_usd": blocks[prompt][label]["imputed"][
                        "median_cost_usd"
                    ],
                }
                for label in LABELS
            }
        }
    pooled: dict[str, Any] = {"by_interface": {}}
    for label in LABELS:
        means = [blocks[p][label]["imputed"]["mean_cost_usd"] for p in PROMPTS]
        total_cost = sum(
            blocks[p][label]["imputed"]["total_cost_usd"] for p in PROMPTS
        )
        # The imputed block prices every Codex run, so the successes in the
        # headline block are exactly the priced successes.
        successes = sum(
            int(snap["headline"]["codex"][p]["by_interface"][label]["successes"])
            for p in PROMPTS
        )
        per_success = round(total_cost / successes, 6) if successes else None
        pooled["by_interface"][label] = {
            "imputed_mean_cost_usd": round(statistics.fmean(means), 6),
            "imputed_cost_per_success_usd": per_success,
            "imputed_cost_per_success_usd_2dp": fmt2(per_success),
        }
    increments = {
        label: {
            "mean_increment_usd": round(
                blocks["few_shot"][label]["imputed"]["mean_cost_usd"]
                - blocks["zero_shot"][label]["imputed"]["mean_cost_usd"],
                6,
            ),
            "mean_increment_cents": _cents(
                round(
                    blocks["few_shot"][label]["imputed"]["mean_cost_usd"]
                    - blocks["zero_shot"][label]["imputed"]["mean_cost_usd"],
                    6,
                )
            ),
        }
        for label in LABELS
    }
    overall_delta = round(
        statistics.fmean(
            [
                blocks["few_shot"][label]["imputed"]["mean_cost_usd"]
                for label in LABELS
            ]
        )
        - statistics.fmean(
            [
                blocks["zero_shot"][label]["imputed"]["mean_cost_usd"]
                for label in LABELS
            ]
        ),
        6,
    )
    out["pooled_prompts"] = pooled
    out["few_shot_increment"] = {
        "definition": (
            "few-shot minus zero-shot imputed cost per run, by agency "
            "condition; cents are the dollar value times 100. The overall "
            "entry gives each agency condition equal weight, matching the "
            "balanced design"
        ),
        "overall": {
            "mean_increment_usd": overall_delta,
            "mean_increment_cents": _cents(overall_delta),
        },
        "by_interface": increments,
    }
    return out


def pass_at_k_section(
    snap: Mapping[str, Any], cost: Mapping[str, Any]
) -> dict[str, Any]:
    passk = snap["extended"]["pass_at_k"]
    out: dict[str, Any] = {
        "estimator": passk["estimator"],
        "reps_rule": passk["reps_rule"],
        "cost_basis": "Sonnet five-rep mean cost per run; expected cost at k = k x mean",
    }
    for prompt in PROMPTS:
        iface = passk["models"]["sonnet"][prompt]["by_interface"]
        chatbot_mean = cost[prompt]["by_interface"]["chatbot"]["mean_cost_usd"]
        agent_mean = cost[prompt]["by_interface"]["constrained agent"]["mean_cost_usd"]
        out[prompt] = {
            "chatbot": {
                **{
                    f"pass_at_{k}_pct": iface["chatbot"][str(k)]["pass_at_k_pct"]
                    for k in (1, 2, 3)
                },
                "mean_cost_usd": chatbot_mean,
                **{
                    f"expected_cost_at_{k}_usd_2dp": fmt2(k * chatbot_mean)
                    for k in (1, 2, 3)
                },
            },
            "constrained_agent": {
                "pass_at_1_pct": iface["constrained agent"]["1"]["pass_at_k_pct"],
                "mean_cost_usd": agent_mean,
                "expected_cost_at_1_usd_2dp": fmt2(agent_mean),
            },
        }
    return out


def tolerance_section(snap: Mapping[str, Any]) -> dict[str, Any]:
    tol = snap["extended"]["tolerance_sensitivity"]
    out: dict[str, Any] = {"note": tol["note"], "scales": [0.5, 1.0, 2.0]}
    for prompt in PROMPTS:
        overall = tol["models"]["sonnet"][prompt]["overall"]
        out[prompt] = {
            "n": overall["n"],
            "task_success_pct_by_scale": {
                scale: overall["task_success_pct_by_scale"][scale]
                for scale in ("0.5", "1.0", "2.0")
            },
        }
    return out


def turns_section(rows: Sequence[RunRow], snap: Mapping[str, Any]) -> dict[str, Any]:
    dist = snap["extended"]["turn_distributions"]["models"]["sonnet"]
    out: dict[str, Any] = {
        "interface": "constrained agent",
        "cap": TURN_CAP,
        "definition": (
            "shares over constrained-agent runs with turn telemetry; "
            "'by 3 turns' means total turns <= 3"
        ),
    }
    for prompt in PROMPTS:
        turns = [
            int(r["turns"])
            for r in subset(rows, prompt=prompt, interface_label="constrained agent")
            if r["turns"] is not None
        ]
        at_cap = sum(1 for t in turns if t >= TURN_CAP)
        snap_entry = dist[prompt]["constrained agent"]
        entry = {
            "n_with_turns": len(turns),
            "share_finishing_by_3_turns_pct": pct(
                sum(1 for t in turns if t <= 3), len(turns)
            ),
            "share_at_cap_pct": pct(at_cap, len(turns)),
            "mean": snap_entry["mean"],
            "median": snap_entry["median"],
            "p90": snap_entry["p90"],
        }
        if entry["n_with_turns"] != snap_entry["n_with_turns"] or (
            entry["share_at_cap_pct"] != snap_entry["share_at_cap_pct"]
        ):
            raise AssertionError(
                f"turn summary disagrees with snapshot for {prompt}: "
                f"{entry} vs {snap_entry}"
            )
        out[prompt] = entry
    return out


def _runtime_medians(rows: Sequence[RunRow]) -> float | None:
    values = [
        float(r["provider_wall_seconds"])
        for r in rows
        if r.get("provider_wall_seconds") is not None
    ]
    return round(statistics.median(values), 1) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return round(numerator / denominator, 2)


def _runtime_stats(rows: Sequence[RunRow]) -> dict[str, Any]:
    """Provider-call runtime summary for a set of rows."""
    values = [
        float(r["provider_wall_seconds"])
        for r in rows
        if r.get("provider_wall_seconds") is not None
    ]
    return {
        "n": len(rows),
        "n_with_provider_wall": len(values),
        "median_provider_wall_seconds": (
            round(statistics.median(values), 1) if values else None
        ),
        "mean_provider_wall_seconds": (
            round(statistics.fmean(values), 1) if values else None
        ),
        "total_provider_wall_hours": round(sum(values) / 3600.0, 2),
    }


def runtime_section(
    rows: Sequence[RunRow], snap: Mapping[str, Any]
) -> dict[str, Any]:
    """The handful of runtime numbers the manuscript quotes.

    Per-arm blocks read the snapshot's ``extended.runtime`` models, so the
    prose and the exhibits share one computation. The pooled block is
    recomputed over the two paper arms, because the snapshot's own pooled
    block covers the three-arm corpus. ``provider_wall_seconds`` is the
    cross-arm measure; ``median_wall_seconds`` is reported only inside a model
    arm, where its per-arm definition is constant.
    """
    rt = snap["extended"]["runtime"]
    models = [model for model in MODELS if model in rt["models"]]
    pooled = _runtime_stats(rows)
    out: dict[str, Any] = {
        "note": rt["note"],
        "measure": rt["measure"],
        "sample": rt["sample"],
        "scope": (
            "pooled over the two paper arms; the corpus snapshot's own pooled "
            "runtime block covers all three arms and is not used here"
        ),
        "n": pooled["n"],
        "pooled": {
            "median_provider_wall_seconds": pooled["median_provider_wall_seconds"],
            "mean_provider_wall_seconds": pooled["mean_provider_wall_seconds"],
            "by_interface_median_provider_wall_seconds": {
                label: _runtime_stats(subset(rows, interface_label=label))[
                    "median_provider_wall_seconds"
                ]
                for label in LABELS
            },
            "by_software_median_provider_wall_seconds": {
                platform: _runtime_stats(subset(rows, platform=platform))[
                    "median_provider_wall_seconds"
                ]
                for platform in PLATFORMS
            },
        },
        "total_provider_hours": pooled["total_provider_wall_hours"],
        "total_provider_hours_by_model": {
            model: rt["total_provider_wall_hours_by_model"][model]
            for model in models
        },
    }
    by_model: dict[str, Any] = {}
    for model in models:
        block = rt["models"][model]
        interfaces = {
            label: {
                "median_provider_wall_seconds": block["by_interface"][label][
                    "median_provider_wall_seconds"
                ],
                "mean_provider_wall_seconds": block["by_interface"][label][
                    "mean_provider_wall_seconds"
                ],
                "median_wall_seconds": block["by_interface"][label][
                    "median_wall_seconds"
                ],
            }
            for label in LABELS
        }
        chatbot = interfaces["chatbot"]["median_provider_wall_seconds"]
        agent = interfaces["constrained agent"]["median_provider_wall_seconds"]
        repair = interfaces["one-repair"]["median_provider_wall_seconds"]
        by_model[model] = {
            "median_provider_wall_seconds": block["overall"][
                "median_provider_wall_seconds"
            ],
            "mean_provider_wall_seconds": block["overall"][
                "mean_provider_wall_seconds"
            ],
            "by_interface": interfaces,
            "by_software_median_provider_wall_seconds": {
                platform: block["by_platform"][platform][
                    "median_provider_wall_seconds"
                ]
                for platform in PLATFORMS
            },
            "constrained_agent_to_chatbot_ratio": _ratio(agent, chatbot),
            "constrained_agent_to_chatbot_ratio_2dp": fmt2(_ratio(agent, chatbot)),
            "one_repair_to_chatbot_ratio": _ratio(repair, chatbot),
            "total_provider_hours": block["overall"]["total_provider_wall_hours"],
        }
    out["by_model"] = by_model
    return out


def check_runtime_against_snapshot(
    rows: Sequence[RunRow], snap: Mapping[str, Any]
) -> None:
    """The re-loaded sample must reproduce the snapshot runtime medians.

    The check runs per paper arm and agency condition. The snapshot's pooled
    runtime block covers the three-arm corpus, so it is not comparable to the
    two-arm sample and is not checked here.
    """
    rt = snap["extended"]["runtime"]
    for model in MODELS:
        block = rt["models"][model]
        for label in LABELS:
            mine = _runtime_medians(
                subset(rows, model=model, interface_label=label)
            )
            theirs = block["by_interface"][label]["median_provider_wall_seconds"]
            if mine != theirs:
                raise AssertionError(
                    f"runtime median {model}/{label} disagrees with snapshot: "
                    f"{mine} vs {theirs}"
                )


def check_headline_against_snapshot(
    rows: Sequence[RunRow], snap: Mapping[str, Any]
) -> None:
    """Every paper arm must reproduce the snapshot headline counts."""
    for model in MODELS:
        model_rows = subset(rows, model=model)
        for prompt in PROMPTS:
            computed = stats_block(subset(model_rows, prompt=prompt))
            snapshot = snap["headline"][model][prompt]
            if computed["overall"] != snapshot["overall"]:
                raise AssertionError(
                    f"{model} {prompt} overall disagrees with snapshot: "
                    f"{computed['overall']} vs {snapshot['overall']}"
                )
            for label in LABELS:
                if computed["by_interface"][label] != snapshot["by_interface"][label]:
                    raise AssertionError(
                        f"{model} {prompt} by_interface[{label}] disagrees with "
                        "snapshot"
                    )
            for platform in PLATFORMS:
                if (
                    computed["by_platform"][platform]
                    != snapshot["by_platform"][platform]
                ):
                    raise AssertionError(
                        f"{model} {prompt} by_platform[{platform}] disagrees with "
                        "snapshot"
                    )
    paper = snap["paper_scope"]
    pooled = stats_block(list(rows))["overall"]
    if pooled != paper["pooled"]["overall"]:
        raise AssertionError(
            "two-model pooled headline disagrees with the snapshot paper "
            f"scope: {pooled} vs {paper['pooled']['overall']}"
        )


def check_cost_against_snapshot(
    cost: Mapping[str, Any], snap: Mapping[str, Any]
) -> None:
    blocks = _snap_cost(snap, "sonnet")
    for prompt in PROMPTS:
        for label in LABELS:
            mine = cost[prompt]["by_interface"][label]
            snap_native = blocks[prompt][label]["native"]
            for key in ("mean_cost_usd", "median_cost_usd", "cost_per_success_usd"):
                if mine[key] != snap_native[key]:
                    raise AssertionError(
                        f"sonnet cost {prompt}/{label}/{key} disagrees with "
                        f"snapshot: {mine[key]} vs {snap_native[key]}"
                    )


def check_factorial_counts(rows: Sequence[RunRow], fact: Mapping[str, Any]) -> None:
    n_zero = len(subset(rows, prompt="zero_shot"))
    n_few = len(subset(rows, prompt="few_shot"))
    if (fact["n_zero_shot"], fact["n_few_shot"]) != (n_zero, n_few):
        raise AssertionError(
            "factorial_fiverep.json counts disagree with the re-loaded sample: "
            f"({fact['n_zero_shot']}, {fact['n_few_shot']}) vs ({n_zero}, {n_few})"
        )


#: Keys of ``factorial_by_model.json`` the manuscript reads: the two paper
#: arms and the two-model pooled fit.
FACTORIAL_KEYS = (*MODELS, "pooled")

#: The Stata minus Python software gap traced across agency conditions at the
#: zero-shot baseline. The chatbot cell is the H1b contrast itself.
STATA_GAP_BY_AGENCY = {
    "chatbot": "H1_stata",
    "one-repair": "GAP_stata_repair_zs",
    "constrained agent": "GAP_stata_agent_zs",
}


#: The paper reports one prompt-by-agency interaction, few-shot by
#: constrained agent, named H4 (decision record
#: 2026-08-24_h4-single-interaction). The few-shot by self-repair interaction
#: and the joint test of the two stay here as diagnostics and reach no
#: reported surface.
REPORTED_INTERACTION = {
    "decision_record": (
        "quality_reports/decision_records/2026-08-24_h4-single-interaction.md"
    ),
    "reported_name": "H4",
    "reported_contrast": "H4_tool_loop",
    "reported_test": "the wild-bootstrap p of H4_tool_loop",
    "diagnostics_only": ["H4_self_repair", "h4_joint_p_wcb"],
}


def factorial_extensions_section(fbm: Mapping[str, Any]) -> dict[str, Any]:
    """Joint H1 and H4 tests, software-gap diagnostics, and cross-arm
    differences, copied from ``factorial_by_model.json``.

    Every block is keyed by the two paper arms plus ``pooled``. The joint H4
    test and the self-repair interaction are diagnostics; ``reported_interaction``
    records which contrast the paper reports.
    """
    return {
        "source": str(FACTORIAL_BY_MODEL_PATH.relative_to(ad.PROJECT_ROOT)),
        "keys": list(FACTORIAL_KEYS),
        "reported_interaction": dict(REPORTED_INTERACTION),
        "h1_joint_p_wcb": {k: fbm[k]["h1_joint_p_wcb"] for k in FACTORIAL_KEYS},
        "h4_joint_p_wcb": {k: fbm[k]["h4_joint_p_wcb"] for k in FACTORIAL_KEYS},
        "stata_gap_zero_shot_by_agency": {
            k: {
                label: fbm[k]["contrasts"][name]
                for label, name in STATA_GAP_BY_AGENCY.items()
            }
            for k in FACTORIAL_KEYS
        },
        "stata_gap_agent_zero_shot": {
            k: fbm[k]["contrasts"]["GAP_stata_agent_zs"] for k in FACTORIAL_KEYS
        },
        "stata_gap_design_averaged": {
            k: fbm[k]["contrasts"]["GAP_stata_avg"] for k in FACTORIAL_KEYS
        },
        "cross_model": fbm["cross_model"],
        "agent_turn_caveat": AGENT_TURN_CAVEAT,
    }


def strict_vs_unified_section(snap: Mapping[str, Any]) -> dict[str, Any]:
    """Primary outcome against the five-gate verdict, per paper arm and pooled.

    The snapshot's own overall block covers the three-arm corpus, so the
    pooled row is summed over the paper arms here.
    """
    svu = snap["strict_vs_unified"]
    by_model = {model: svu["by_model"][model]["overall"] for model in MODELS}
    keys = (
        "n",
        "task_success_n",
        "task_success_strict_n",
        "task_success_with_method_n",
    )
    totals = {key: sum(int(block[key]) for block in by_model.values()) for key in keys}
    pooled = dict(totals)
    pooled["task_success_pct"] = pct(totals["task_success_n"], totals["n"])
    pooled["task_success_strict_pct"] = pct(
        totals["task_success_strict_n"], totals["n"]
    )
    pooled["task_success_with_method_pct"] = pct(
        totals["task_success_with_method_n"], totals["n"]
    )
    pooled["delta_pp"] = round(
        pooled["task_success_pct"] - pooled["task_success_strict_pct"], 2
    )
    pooled["delta_with_method_pp"] = round(
        pooled["task_success_pct"] - pooled["task_success_with_method_pct"], 2
    )
    return {
        "note": (
            "flip counts between the primary final-artifact outcome and the "
            "five-gate strict verdict on the same rows; pooled covers the two "
            "paper arms"
        ),
        "by_model": by_model,
        "pooled": pooled,
    }


def build_payload(
    rows: Sequence[RunRow],
    snap: Mapping[str, Any],
    fact: Mapping[str, Any],
    fbm: Mapping[str, Any],
) -> dict[str, Any]:
    sonnet_rows = subset(rows, model="sonnet")
    cost = sonnet_cost_section(sonnet_rows)
    factorial_keys = (
        "n_analysis",
        "n_zero_shot",
        "n_few_shot",
        "n_tasks",
        "model",
        "wcb",
        "contrast_convention",
        "contrasts",
        "h1_joint_p_wcb",
        "h4_joint_p_wcb",
        "saturation_check",
    )
    return {
        "generated_by": "code/benchmark/build_manuscript_numbers.py",
        "inputs": {
            "snapshot": str(SNAPSHOT_PATH.relative_to(ad.PROJECT_ROOT)),
            "factorial": str(FACTORIAL_PATH.relative_to(ad.PROJECT_ROOT)),
            "factorial_by_model": str(
                FACTORIAL_BY_MODEL_PATH.relative_to(ad.PROJECT_ROOT)
            ),
            "loader": (
                "analysis_dataset.load_paper_runs(max_reps=12) over "
                f"{list(ARM_IDS)}"
            ),
        },
        "paper_scope": {
            "decision_record": PAPER_SCOPE_DECISION_RECORD,
            "models": list(MODELS),
            "arm_ids": list(ARM_IDS),
            "sample_version": ad.PAPER_SAMPLE_VERSION,
            "n": len(rows),
            "pooled_definition": (
                "pooled means the two paper arms pooled, in the headline rates "
                "and in the estimated contrasts alike"
            ),
            "caveat": AGENT_TURN_CAVEAT,
        },
        "sample_rule": snap["sample_rule"],
        "task_success_rule": snap["task_success_rule"],
        "scoring_version": snap["scoring_version"],
        "primary_outcome_version": snap["primary_outcome_version"],
        "headline_sonnet": headline_section(sonnet_rows),
        "headline_by_model": {
            model: headline_section(subset(rows, model=model)) for model in MODELS
        },
        "headline_pooled": headline_section(list(rows)),
        "factorial": {key: fact[key] for key in factorial_keys},
        "factorial_by_key": {
            key: {name: fbm[key][name] for name in factorial_keys}
            for key in FACTORIAL_KEYS
        },
        "factorial_extensions": factorial_extensions_section(fbm),
        "per_task_sonnet": per_task_section(sonnet_rows),
        "per_task_model_platform": per_task_model_platform_section(rows),
        "difficulty_thirds_sonnet": difficulty_thirds(sonnet_rows),
        "failure_archetypes_sonnet": archetype_section(sonnet_rows),
        "cost": {
            "sonnet_native": cost,
            "gpt54_imputed": gpt54_cost_section(snap),
        },
        "prompt_contrasts": {
            model: snap["prompt_contrasts"][model] for model in MODELS
        },
        "silent_failure": {
            "definition": snap["extended"]["silent_failure"]["definition"],
            "models": {
                model: snap["extended"]["silent_failure"]["models"][model]
                for model in MODELS
            },
        },
        "strict_vs_unified": strict_vs_unified_section(snap),
        "pass_at_k_sonnet": pass_at_k_section(snap, cost),
        "pass_at_k_paper": pak.paper_pass_at_k(rows),
        "tolerance_sensitivity_sonnet": tolerance_section(snap),
        "turns_sonnet_constrained_agent": turns_section(sonnet_rows, snap),
        "runtime": runtime_section(rows, snap),
        "revision_disclosure": {
            "note": snap["extended"]["revision_disclosure"]["note"],
            "models": {
                model: snap["extended"]["revision_disclosure"]["models"][model]
                for model in MODELS
            },
        },
    }


def main() -> None:
    snap = json.loads(SNAPSHOT_PATH.read_text())
    fact = json.loads(FACTORIAL_PATH.read_text())
    fbm = json.loads(FACTORIAL_BY_MODEL_PATH.read_text())
    rows = ad.load_paper_runs(max_reps=MAX_REPS).rows
    if len(rows) != ad.EXPECTED_PAPER_RUNS_12REP:
        raise AssertionError(
            f"paper sample is {len(rows)} rows, expected "
            f"{ad.EXPECTED_PAPER_RUNS_12REP}"
        )
    sonnet_rows = subset(rows, model="sonnet")
    check_headline_against_snapshot(rows, snap)
    check_factorial_counts(sonnet_rows, fact)
    check_runtime_against_snapshot(rows, snap)
    payload = build_payload(rows, snap, fact, fbm)
    check_cost_against_snapshot(payload["cost"]["sonnet_native"], snap)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    for model in MODELS:
        block = payload["headline_by_model"][model]["pooled_prompts"]["overall"]
        print(f"{model} twelve-rep: {block['success_pct']}% (n {block['n']})")
    pooled = payload["headline_pooled"]["pooled_prompts"]["overall"]
    print(f"two-model pooled: {pooled['success_pct']}% (n {pooled['n']})")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
