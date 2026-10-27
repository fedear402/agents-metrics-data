"""Pilot V data: the remaining nine task DGPs that were not yet calibrated.

Tasks covered (with the dataset they write):
  T05 VIF                            pilot_v5_collinear.csv
  T06 Cook's distance                pilot_v5_collinear.csv (same data as T05)
  T07 Two-sample t-test (Welch)      pilot_v5_ttest.csv
  T08 One-way ANOVA + Tukey HSD       pilot_v5_anova.csv
  T09 Chi-square of independence      pilot_v5_chisq.csv
  T16 Arellano-Bond dynamic panel     pilot_v5_dynpanel.csv
  T18 Sharp regression discontinuity  pilot_v5_rdd.csv
  T19 Local-projection IRF            pilot_v5_lp.csv
  T20 Wild cluster bootstrap (few)    pilot_v5_wcb.csv

T14 (two-way fixed-effects estimation) reuses pilot_v3.csv directly (the
firm-year panel with y_twocluster).

Each DGP is parameter-fixed by a single SEED; the reference values below are
the sample manual computations used by the cross-platform verification.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260517
HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
REF_DIR = HERE / "reference"

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# T05 / T06: collinear linear-regression dataset for VIF and Cook's distance
# ---------------------------------------------------------------------------
def make_collinear() -> pd.DataFrame:
    n = 500
    x1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    x2 = 0.7 * x1 + 0.5 * z2
    x3 = rng.normal(0, 1, n)
    x4 = rng.normal(0, 1, n)
    z5 = rng.normal(0, 1, n)
    x5 = 0.4 * x1 + 0.4 * x2 + 0.5 * z5
    eps = rng.normal(0, 1, n)
    y = 1.0 + x1 + x2 + 0.5 * x3 + x4 + 0.5 * x5 + eps
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})


# ---------------------------------------------------------------------------
# T07: two-sample t-test, unequal variance (Welch)
# ---------------------------------------------------------------------------
def make_ttest() -> pd.DataFrame:
    na, nb = 100, 100
    a = rng.normal(0.0, 1.0, na)
    b = rng.normal(0.5, 1.5, nb)
    group = np.r_[np.zeros(na, dtype=int), np.ones(nb, dtype=int)]
    y = np.r_[a, b]
    return pd.DataFrame({"group": group, "y": y})


# ---------------------------------------------------------------------------
# T08: one-way ANOVA with four groups
# ---------------------------------------------------------------------------
def make_anova() -> pd.DataFrame:
    means = [0.0, 0.3, 0.7, 1.0]
    per = 50
    rows = []
    for g, m in enumerate(means):
        rows.append(pd.DataFrame({"group": g, "y": rng.normal(m, 1.0, per)}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# T09: 2x3 contingency table for chi-square test of independence
# ---------------------------------------------------------------------------
def make_chisq() -> pd.DataFrame:
    # Two-variable categorical sample with small but real association.
    n = 600
    row = rng.choice([0, 1], size=n, p=[0.5, 0.5])
    # Column probs depend on row.
    p_col_given_row = {0: [0.5, 0.3, 0.2], 1: [0.3, 0.4, 0.3]}
    col = np.array([rng.choice([0, 1, 2], p=p_col_given_row[r]) for r in row])
    return pd.DataFrame({"row": row, "col": col})


# ---------------------------------------------------------------------------
# T16: dynamic panel y_{i,t} = rho y_{i,t-1} + beta x_{i,t} + alpha_i + eps
# ---------------------------------------------------------------------------
def make_dynpanel() -> pd.DataFrame:
    n_firm, n_t = 100, 10
    rho, beta = 0.5, 0.5
    alpha = rng.normal(0, 0.5, n_firm)
    x = rng.normal(0, 1, (n_firm, n_t))
    y = np.zeros((n_firm, n_t))
    y[:, 0] = alpha + rng.normal(0, 1, n_firm)
    for t in range(1, n_t):
        y[:, t] = rho * y[:, t - 1] + beta * x[:, t] + alpha + rng.normal(0, 1, n_firm)
    rows = []
    for i in range(n_firm):
        for t in range(n_t):
            rows.append({"firm": i, "year": t, "y": y[i, t], "x": x[i, t]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T18: sharp regression discontinuity at z = 0
# ---------------------------------------------------------------------------
def make_rdd() -> pd.DataFrame:
    n = 1000
    z = rng.uniform(-1.0, 1.0, n)
    d = (z >= 0).astype(int)
    eps = rng.normal(0.0, 0.5, n)
    # True LATE at the cutoff is 1.0; outcome has a linear running-variable trend.
    y = 0.5 + 1.0 * d + 0.3 * z + eps
    return pd.DataFrame({"z": z, "d": d, "y": y})


# ---------------------------------------------------------------------------
# T19: AR(1) time series with shock for local-projection IRF
# ---------------------------------------------------------------------------
def make_lp() -> pd.DataFrame:
    T = 250
    rho, theta = 0.5, 0.5
    shock = rng.normal(0.0, 1.0, T)
    eps = rng.normal(0.0, 1.0, T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = rho * y[t - 1] + theta * shock[t] + eps[t]
    return pd.DataFrame({"t": np.arange(1, T + 1, dtype=int), "y": y, "shock": shock})


# ---------------------------------------------------------------------------
# T20: linear regression with five clusters (wild cluster bootstrap)
# ---------------------------------------------------------------------------
def make_wcb() -> pd.DataFrame:
    n_clusters = 5
    per = 80
    n = n_clusters * per
    cluster = np.repeat(np.arange(n_clusters), per)
    x = rng.normal(0, 1, n)
    cluster_re = rng.normal(0, 0.7, n_clusters)[cluster]
    eps = rng.normal(0, 1, n)
    # True coefficient is 0.4.
    y = 0.4 * x + cluster_re + eps
    return pd.DataFrame({"cluster": cluster, "x": x, "y": y})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "pilot_v5_collinear.csv": make_collinear(),
        "pilot_v5_ttest.csv":     make_ttest(),
        "pilot_v5_anova.csv":     make_anova(),
        "pilot_v5_chisq.csv":     make_chisq(),
        "pilot_v5_dynpanel.csv":  make_dynpanel(),
        "pilot_v5_rdd.csv":       make_rdd(),
        "pilot_v5_lp.csv":        make_lp(),
        "pilot_v5_wcb.csv":       make_wcb(),
    }
    for fname, df in datasets.items():
        path = DATA_DIR / fname
        df.to_csv(path, index=False)
        print(f"wrote {path}  ({len(df)} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
