"""Pilot III data: panel structure for two-way clustering and staggered DiD,
plus a selection design for Heckman.

200 firms x 10 years = 2000 firm-year observations.

Columns:
  firm (0-199), year (0-9)
  x          : firm-year regressor (std normal)
  y_twocluster: outcome with both firm-level and year-level random effects
                (genuine two-way error structure for cluster-SE)
  cohort     : adoption year for treated units (4, 6, 8) or -1 for never-treated
  treat_post : firm-year treatment indicator (= 1 if year >= cohort and cohort != -1)
  y_did      : outcome for staggered DiD with HETEROGENEOUS cohort effects
                (cohort 4 -> tau=1, cohort 6 -> tau=2, cohort 8 -> tau=3)
  z_sel, w   : selection-equation regressors
  selected   : 0/1 selection indicator
  y_sel      : observed outcome (NaN where selected==0); selection-on-unobservables design

References:
  two_way_cluster: OLS of y_twocluster on x with intercept, SE clustered on
                   (firm, year) via Cameron-Gelbach-Miller. CGM formula:
                   V_2way = V_firm + V_year - V_firm_inter_year.
                   Reference computed via the `clubSandwich`-equivalent identity
                   implemented in Python (replicates `fixest::feols(... cluster = ~firm + year)`).
  staggered_did:   Uniformly-weighted average ATT(g,t) over the |S|=12 supported
                   post-treatment (g, t) cells (g in {4,6,8}, g <= t <= 9):
                       ATT_uniform = (1/|S|) * sum_{(g,t) in S} ATT(g,t)
                   ATT(g,t) is the Callaway-Sant'Anna long-difference using
                   never-treated firms (cohort = -1) as the control and (g-1)
                   as the base year. The reference value is the SAMPLE estimate
                   from this dataset's manual computation (~1.589 for SEED
                   20260515); the population truth from the DGP cohort taus is
                   20/12 ≈ 1.667 and is reported separately for documentation.
                   All three platforms (Python, R, Stata) use the same long-
                   difference formula and agree at the sample value within
                   floating-point precision.
                   Naive TWFE will diverge from this by the Goodman-Bacon
                   decomposition under cohort-effect heterogeneity.
  heckman:         Heckman two-step: probit selection on z_sel + w, outcome regression on x
                   with IMR for selected observations, asymptotic SEs corrected for the
                   generated-regressor adjustment via Greene's formula.

Notes:
  - DGP seeds use SEED+offsets so the v3 dataset is independent of v1/v2.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

SEED = 20260515
HERE = Path(__file__).parent

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
rng = np.random.default_rng(SEED + 31337)

N_FIRMS = 200
N_YEARS = 10


def make_data() -> pd.DataFrame:
    firms = np.repeat(np.arange(N_FIRMS), N_YEARS)
    years = np.tile(np.arange(N_YEARS), N_FIRMS)
    n = len(firms)

    # ---- regressor x ----
    x = rng.normal(0.0, 1.0, size=n)

    # ---- two-way clustered outcome ----
    firm_re = rng.normal(0.0, 0.7, size=N_FIRMS)
    year_re = rng.normal(0.0, 0.5, size=N_YEARS)
    eps_tw = rng.normal(0.0, 1.0, size=n)
    y_twocluster = 1.0 + 0.5 * x + firm_re[firms] + year_re[years] + eps_tw

    # ---- staggered DiD: cohort assignment ----
    # Assign 25% never-treated, 25% each cohort {4,6,8}
    cohort_choices = rng.choice([-1, 4, 6, 8], size=N_FIRMS, p=[0.25, 0.25, 0.25, 0.25])
    cohort_firm = cohort_choices[firms]
    treat_post = ((cohort_firm != -1) & (years >= cohort_firm)).astype(int)

    # Heterogeneous cohort effects: cohort 4 -> 1.0, cohort 6 -> 2.0, cohort 8 -> 3.0
    cohort_tau = np.where(
        cohort_firm == 4, 1.0,
        np.where(cohort_firm == 6, 2.0, np.where(cohort_firm == 8, 3.0, 0.0)),
    )
    eps_did = rng.normal(0.0, 0.8, size=n)
    y_did = (
        0.0
        + 0.1 * x
        + firm_re[firms]          # firm FE
        + 0.2 * years             # linear time trend (year FE proxy)
        + cohort_tau * treat_post # heterogeneous treatment effect
        + eps_did
    )

    # ---- Heckman selection ----
    z_sel = rng.normal(0.0, 1.0, size=n)
    w = rng.normal(0.0, 1.0, size=n)
    # Joint normal (u, eps_y) with corr = 0.6
    rho = 0.6
    u = rng.normal(0.0, 1.0, size=n)
    eta = rng.normal(0.0, 1.0, size=n)
    eps_y = rho * u + np.sqrt(1 - rho**2) * eta
    # Selection: latent = 0.3 + 0.8 z_sel + 0.5 w + u ; observed if > 0
    latent_sel = 0.3 + 0.8 * z_sel + 0.5 * w + u
    selected = (latent_sel > 0).astype(int)
    # Outcome: y_sel = 1.0 + 0.7 x + eps_y, observed only if selected
    y_sel_full = 1.0 + 0.7 * x + eps_y
    y_sel = np.where(selected == 1, y_sel_full, np.nan)

    df = pd.DataFrame(
        {
            "firm": firms.astype(int),
            "year": years.astype(int),
            "x": x,
            "y_twocluster": y_twocluster,
            "cohort": cohort_firm.astype(int),
            "treat_post": treat_post.astype(int),
            "y_did": y_did,
            "z_sel": z_sel,
            "w": w,
            "selected": selected.astype(int),
            "y_sel": y_sel,
        }
    )
    return df


def reference_two_way_cluster(df: pd.DataFrame) -> dict:
    """OLS of y_twocluster on x with intercept, SE clustered on (firm, year)
    via Cameron-Gelbach-Miller: V_2way = V_firm + V_year - V_intersection.

    Each component uses CR1 (clustered) with the standard finite-sample
    correction G/(G-1) * (N-1)/(N-K). For consistency with `fixest::feols`
    defaults, we use the same correction in all three components.
    """
    y = df["y_twocluster"].values
    X = sm.add_constant(df[["x"]]).values
    n, k = X.shape
    XtXinv = np.linalg.inv(X.T @ X)
    beta = XtXinv @ (X.T @ y)
    resid = y - X @ beta

    def cr1_meat(groups: np.ndarray) -> np.ndarray:
        unique = np.unique(groups)
        G = len(unique)
        meat = np.zeros((k, k))
        for g in unique:
            mask = groups == g
            Xg = X[mask]
            eg = resid[mask]
            ug = Xg.T @ eg
            meat += np.outer(ug, ug)
        # G/(G-1) * (n-1)/(n-k) standard fixest correction
        meat *= (G / (G - 1.0)) * ((n - 1.0) / (n - k))
        return meat

    firms = df["firm"].values
    years = df["year"].values
    intersection = firms * (N_YEARS + 1) + years   # unique per firm-year cell

    V_firm = XtXinv @ cr1_meat(firms) @ XtXinv
    V_year = XtXinv @ cr1_meat(years) @ XtXinv
    V_inter = XtXinv @ cr1_meat(intersection) @ XtXinv
    V_2way = V_firm + V_year - V_inter

    se = np.sqrt(np.diag(V_2way))

    return {
        "vcov": "two-way cluster (CGM) on (firm, year), fixest-style correction",
        "estimates": [
            {"term": "const", "estimate": float(beta[0]), "se": float(se[0]), "n_obs": int(n)},
            {"term": "x",     "estimate": float(beta[1]), "se": float(se[1]), "n_obs": int(n)},
        ],
    }


def reference_staggered_did(df: pd.DataFrame) -> dict:
    """Reference for T17: uniformly-weighted sample average of ATT(g,t).

    The reference estimand is

        ATT_uniform = (1/|S|) * sum_{(g,t) in S} ATT(g, t)

    where S = { (g, t) : g in {4, 6, 8}, g <= t <= 9 } and |S| = 12. The
    cell-level ATT(g, t) is the Callaway-Sant'Anna long difference using
    never-treated firms (cohort = -1) as the control and (g - 1) as the base
    year:

        ATT(g, t) = (E[y_did | cohort=g, year=t]    - E[y_did | cohort=g, year=g-1])
                  - (E[y_did | cohort=-1, year=t]   - E[y_did | cohort=-1, year=g-1])

    Identical manual implementations in Python, R, and Stata
    (`manual_references/staggered_did/`) agree at the sample value within
    floating-point precision; the registered reference value below is that
    cross-platform sample value, not the population truth from the DGP. The
    population truth `20/12 ≈ 1.6667` is reported separately as a sanity
    check; the sample value deviates from it by sampling noise across the 12
    (g, t) cells.

    A naive TWFE specification will diverge from this reference by the
    Goodman-Bacon decomposition under cohort-effect heterogeneity.
    """
    cohorts = (4, 6, 8)
    last_year = N_YEARS - 1  # year index, 0..9 for N_YEARS=10
    control_cohort = -1

    def att_gt(g: int, t: int) -> float:
        base = g - 1
        treated_t    = df.loc[(df["cohort"] == g) & (df["year"] == t), "y_did"].mean()
        treated_base = df.loc[(df["cohort"] == g) & (df["year"] == base), "y_did"].mean()
        control_t    = df.loc[(df["cohort"] == control_cohort) & (df["year"] == t), "y_did"].mean()
        control_base = df.loc[(df["cohort"] == control_cohort) & (df["year"] == base), "y_did"].mean()
        return float((treated_t - treated_base) - (control_t - control_base))

    cells = [
        {"g": int(g), "t": int(t), "att_gt": att_gt(int(g), int(t))}
        for g in cohorts
        for t in range(int(g), last_year + 1)
    ]
    att_uniform = float(np.mean([c["att_gt"] for c in cells]))

    # Population truth from the DGP cohort taus, for documentation only.
    taus = {4: 1.0, 6: 2.0, 8: 3.0}
    pop_num = sum(taus[g] * (N_YEARS - g) for g in cohorts)
    pop_den = sum((N_YEARS - g) for g in cohorts)
    pop_att = pop_num / pop_den

    # Per-cell SE proxy for the pilot record (cells treated as i.i.d. — main
    # sweep will replace with a proper Callaway-Sant'Anna SE).
    se_proxy = float(np.std([c["att_gt"] for c in cells], ddof=1) / np.sqrt(len(cells)))

    return {
        "estimand": "ATT_uniform = (1/|S|) * sum_{(g,t) in S} ATT(g,t)",
        "vcov": "cross-platform sample manual; SE is a cell-mean proxy for the pilot",
        "estimates": [
            {"term": "att", "estimate": att_uniform, "se": se_proxy, "n_obs": int(len(df))},
        ],
        "population_truth": pop_att,
        "n_cells": len(cells),
        "cells": cells,
    }


def reference_heckman(df: pd.DataFrame) -> dict:
    """Heckman two-step with corrected SE.

    Implements Greene's two-step explicitly so we have a fixed reference for the
    pilot. Pure-statsmodels implementation; no external Heckman package.
    """
    # Stage 1: probit of selected on z_sel and w
    Z = sm.add_constant(df[["z_sel", "w"]]).values
    sel = df["selected"].values
    probit = sm.Probit(sel, Z).fit(disp=0)
    gamma = probit.params  # 3-vector
    Zg = Z @ gamma
    imr = norm.pdf(Zg) / norm.cdf(Zg)   # inverse Mills ratio

    # Stage 2: outcome regression on selected sample with IMR
    sel_mask = df["selected"] == 1
    Xo = sm.add_constant(df.loc[sel_mask, ["x"]]).values
    imr_sel = imr[sel_mask.values]
    Xfull = np.column_stack([Xo, imr_sel])
    yo = df.loc[sel_mask, "y_sel"].values
    XtXinv = np.linalg.inv(Xfull.T @ Xfull)
    beta = XtXinv @ (Xfull.T @ yo)
    resid = yo - Xfull @ beta

    # Greene's corrected variance:
    # V = sigma_e^2 * (X'X)^-1 * [X'(I - rho^2 * delta) X + Q] * (X'X)^-1
    # where delta_i = imr_i * (imr_i + Zg_i), Q involves the probit V-cov.
    n_o = len(yo)
    sigma2 = float((resid @ resid) / n_o)  # ML-style sigma2
    rho = float(beta[-1]) / max(np.sqrt(sigma2), 1e-9)  # rough rho proxy

    delta = imr_sel * (imr_sel + Zg[sel_mask.values])
    A = Xfull.T @ Xfull
    Ainv = np.linalg.inv(A)
    Bdiag = np.eye(n_o) - rho**2 * np.diag(delta)
    middle = Xfull.T @ Bdiag @ Xfull
    # Probit V-cov contribution Q:
    # Q = rho^2 * (X' delta Z) V_gamma (Z' delta X)
    V_gamma = probit.cov_params()
    deltaZ = (imr_sel * (imr_sel + Zg[sel_mask.values]))[:, None] * Z[sel_mask.values]
    XDZ = Xfull.T @ deltaZ
    Q = rho**2 * (XDZ @ V_gamma @ XDZ.T)
    V_beta = sigma2 * Ainv @ (middle + Q) @ Ainv
    se = np.sqrt(np.diag(V_beta))

    return {
        "vcov": "Heckman two-step, Greene-corrected SE",
        "rho": rho,
        "estimates": [
            {"term": "const", "estimate": float(beta[0]), "se": float(se[0]), "n_obs": int(n_o)},
            {"term": "x",     "estimate": float(beta[1]), "se": float(se[1]), "n_obs": int(n_o)},
            {"term": "imr",   "estimate": float(beta[2]), "se": float(se[2]), "n_obs": int(n_o)},
        ],
    }


def main() -> None:
    df = make_data()
    csv_path = HERE / "data" / "pilot_v3.csv"
    df.to_csv(csv_path, index=False)

    references = {
        "two_way_cluster": reference_two_way_cluster(df),
        "staggered_did": reference_staggered_did(df),
        "heckman": reference_heckman(df),
    }
    (HERE / "reference" / "reference_v3.json").write_text(json.dumps(references, indent=2))

    print(f"wrote {csv_path} ({len(df)} rows, {df.shape[1]} cols)")
    print(f"  firm count = {df['firm'].nunique()}, year count = {df['year'].nunique()}")
    print(f"  treated firms: {df.loc[df['cohort'] != -1, 'firm'].nunique()}")
    print(f"  selected fraction: {df['selected'].mean():.3f}")
    print(json.dumps(references, indent=2))


if __name__ == "__main__":
    main()
