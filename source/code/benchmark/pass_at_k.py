"""pass@k on the twelve-repetition paper sample.

The benchmark scores one run at a time. pass@k asks a different question:
if a user drew k independent runs of the same task under the same
conditions, how often would at least one of them be correct?

Estimator
---------
Each design cell holds ``n`` runs of which ``c`` succeed. The standard
unbiased estimator of pass@k for that cell is

    pass@k = 1 - C(n - c, k) / C(n, k),

which is the probability that a draw of k runs without replacement from the
cell contains at least one success. The reported value is the unweighted
mean of the cell values, so every cell counts the same.

Cells and averages
------------------
A design cell is one model, one prompt condition, one agency condition, one
programming language, and one task. The paper sample holds twelve runs in
every cell, so k runs from 1 to 12 and no cell is ever skipped.

Two averages are reported, both with the prompt conditions pooled, which
means the average runs over the cells of both prompt conditions:

* ``by_agency``: the mean over the 126 cells of one model and one agency
  condition (21 tasks x 3 languages x 2 prompt conditions);
* ``overall``: the mean over the 378 cells of one model.

Scope: the two models of the paper's main specification, Claude Sonnet 4.6
and GPT-5.4 through Codex (decision record
2026-08-21_two-model-main-specification). The load verifies the pinned
paper sample manifest, so the values can never come from an unpinned set of
runs.

Outputs. ``build_manuscript_numbers.py`` embeds the payload in
``manuscript_numbers.json`` under ``pass_at_k_paper``. Running this module
directly also writes the standalone ``data/analysis/pass_at_k.json``:

    uv run python code/benchmark/pass_at_k.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analysis_dataset as ad  # noqa: E402

#: Repetitions per design cell in the paper sample.
MAX_REPS = 12
#: Every k the curve covers.
K_VALUES: tuple[int, ...] = tuple(range(1, MAX_REPS + 1))
#: Models of the paper's main specification, in display order.
MODELS: tuple[str, ...] = tuple(ad.PAPER_MODELS)
#: Agency conditions in the design order, by their display labels.
AGENCY_LABELS: tuple[str, ...] = tuple(
    ad.INTERFACE_LABELS[interface] for interface in ad.INTERFACES
)
#: Cells behind one model and one agency condition, prompts pooled.
CELLS_PER_AGENCY = 21 * 3 * 2
#: Cells behind one model.
CELLS_PER_MODEL = CELLS_PER_AGENCY * 3

ESTIMATOR = (
    "unbiased pass@k = 1 - C(n-c, k)/C(n, k) per design cell, then the "
    "unweighted mean over cells"
)
SAMPLE_RULE = (
    "twelve-repetition paper sample; a design cell is one model, prompt "
    "condition, agency condition, programming language, and task; the "
    "prompt conditions are pooled by averaging over the cells of both"
)

OUT_PATH = ad.PROJECT_ROOT / "data" / "analysis" / "pass_at_k.json"

RunRow = Mapping[str, Any]


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k for one cell with ``n`` runs of which ``c`` succeed.

    Parameters
    ----------
    n : int
        Runs in the cell. Must satisfy ``n >= k >= 1``.
    c : int
        Successful runs in the cell, ``0 <= c <= n``.
    k : int
        Draw size.

    Returns
    -------
    float
        The probability that a draw of ``k`` runs without replacement holds
        at least one success.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if n < k:
        raise ValueError(f"cell of {n} runs cannot support k = {k}")
    if not 0 <= c <= n:
        raise ValueError(f"successes {c} outside 0..{n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def cell_counts(rows: Sequence[RunRow]) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Run and success counts per cell, keyed by model and agency label.

    The prompt condition, the language, and the task select the
    cell but do not appear in the key, so the returned lists already pool
    the prompt conditions.
    """
    cells: dict[tuple[str, str, str, str, str], list[int]] = {}
    for row in rows:
        key = (
            str(row["model"]),
            str(row["interface_label"]),
            str(row["prompt"]),
            str(row["platform"]),
            str(row["task"]),
        )
        counts = cells.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += int(bool(row["task_success"]))
    out: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for (model, agency, _prompt, _platform, _task), (n, c) in cells.items():
        out.setdefault((model, agency), []).append((n, c))
    return out


def curve(counts: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """The pass@k curve over one set of cells, in percent."""
    if not counts:
        raise ValueError("no cells")
    short = [n for n, _c in counts if n != MAX_REPS]
    if short:
        raise ValueError(f"cells outside the twelve-repetition sample: {short}")
    return {
        "n_cells": len(counts),
        "n_runs_per_cell": MAX_REPS,
        "pass_at_k_pct": {
            str(k): round(
                100.0
                * statistics.fmean(unbiased_pass_at_k(n, c, k) for n, c in counts),
                4,
            )
            for k in K_VALUES
        },
    }


def paper_pass_at_k(rows: Sequence[RunRow]) -> dict[str, Any]:
    """The full pass@k payload for the paper sample."""
    counts = cell_counts(rows)
    models: dict[str, Any] = {}
    for model in MODELS:
        by_agency: dict[str, Any] = {}
        pooled: list[tuple[int, int]] = []
        for agency in AGENCY_LABELS:
            cells = counts.get((model, agency))
            if cells is None:
                raise ValueError(f"no cells for {model} / {agency}")
            if len(cells) != CELLS_PER_AGENCY:
                raise ValueError(
                    f"{model} / {agency}: {len(cells)} cells, expected "
                    f"{CELLS_PER_AGENCY}"
                )
            by_agency[agency] = curve(cells)
            pooled.extend(cells)
        if len(pooled) != CELLS_PER_MODEL:
            raise ValueError(
                f"{model}: {len(pooled)} cells, expected {CELLS_PER_MODEL}"
            )
        models[model] = {"overall": curve(pooled), "by_agency": by_agency}
    return {
        "generated_by": "code/benchmark/pass_at_k.py",
        "estimator": ESTIMATOR,
        "sample": SAMPLE_RULE,
        "sample_version": ad.PAPER_SAMPLE_VERSION,
        "models": list(MODELS),
        "agency_conditions": list(AGENCY_LABELS),
        "k_values": list(K_VALUES),
        "by_model": models,
    }


def main() -> None:
    result = ad.load_paper_runs(max_reps=MAX_REPS, include_turns=False)
    rows = result.rows
    payload = paper_pass_at_k(rows)
    payload["n_runs"] = len(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    for model in MODELS:
        block = payload["by_model"][model]["overall"]["pass_at_k_pct"]
        print(f"  {model:8s} pass@1 {block['1']:6.2f}  pass@12 {block['12']:6.2f}")


if __name__ == "__main__":
    main()
