"""Manual equal-weight ATT(g,t) estimator for T17 (staggered DiD) — Python reference.

Computes ATT(g,t) for each treated cohort g in {4, 6, 8} and each post-treatment
year t with t >= g, using never-treated firms (cohort == -1) as controls and a
long-difference design with pre-treatment year (g - 1) as the base period:

    ATT(g, t) = (E[y_did | cohort=g, year=t]    - E[y_did | cohort=g, year=g-1])
              - (E[y_did | cohort=-1, year=t]   - E[y_did | cohort=-1, year=g-1])

The reference estimand is the uniform average of ATT(g, t) over the supported
(g, t) cells:

    ATT_uniform = (1 / |S|) * sum_{(g, t) in S} ATT(g, t)

with S = { (g, t) : g in {4, 6, 8}, g <= t <= 9 }, |S| = 6 + 4 + 2 = 12.

DGP truth (from generate_data_v3.py):
    cohort 4 -> tau = 1.0, cohort 6 -> tau = 2.0, cohort 8 -> tau = 3.0
    ATT_uniform = (1*6 + 2*4 + 3*2) / 12 = 20/12 = 1.6667

This script reads pilot_v3.csv and prints the sample estimate to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "pilot_v3.csv"

COHORTS = (4, 6, 8)
LAST_YEAR = 9
CONTROL_COHORT = -1


def att_gt(df: pd.DataFrame, g: int, t: int) -> float:
    """Long-difference ATT(g, t) using never-treated as the control."""
    base = g - 1
    treated_t = df.loc[(df["cohort"] == g) & (df["year"] == t), "y_did"].mean()
    treated_base = df.loc[(df["cohort"] == g) & (df["year"] == base), "y_did"].mean()
    control_t = df.loc[(df["cohort"] == CONTROL_COHORT) & (df["year"] == t), "y_did"].mean()
    control_base = df.loc[(df["cohort"] == CONTROL_COHORT) & (df["year"] == base), "y_did"].mean()
    return float((treated_t - treated_base) - (control_t - control_base))


def att_uniform(df: pd.DataFrame) -> dict:
    """Equal-weight ATT_uniform across all supported (g, t) cells."""
    cells: list[dict] = []
    for g in COHORTS:
        for t in range(g, LAST_YEAR + 1):
            cells.append({"g": g, "t": t, "att_gt": att_gt(df, g, t)})
    att = float(np.mean([c["att_gt"] for c in cells]))
    return {
        "estimand": "ATT_uniform = (1/|S|) * sum_{(g,t) in S} ATT(g,t)",
        "n_cells": len(cells),
        "cells": cells,
        "att_uniform": att,
        "population_truth": 20.0 / 12.0,
    }


def main() -> None:
    df = pd.read_csv(DATA)
    result = att_uniform(df)
    print(json.dumps({"platform": "python", **result}, indent=2))


if __name__ == "__main__":
    main()
