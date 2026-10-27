"""Extend the pilot dataset with structure needed for harder tasks.

Adds to pilot.csv:
  - z          : instrument for x1 (correlated with x1, uncorrelated with eps_iv)
  - x1_iv      : endogenous regressor (correlation between x1_iv and eps_iv)
  - y_iv       : outcome with endogenous x1_iv
  - cluster    : cluster id (50 clusters of size 40)
  - y_panel    : outcome with cluster-level intraclass correlation
  - exposure   : positive exposure variable
  - y_count    : Poisson count outcome with offset = log(exposure)

Writes:
  data/pilot_v2.csv
  reference/reference_v2.json
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

SEED = 20260515
N = 2000
HERE = Path(__file__).parent

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
rng = np.random.default_rng(SEED)


def make_data(n: int) -> pd.DataFrame:
    # Re-create the v1 dataset (same seed sequence on the same draws)
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)
    eps = rng.normal(0.0, 1.0, size=n)
    y_lin = 1.0 + 0.5 * x1 - 0.3 * x2 + eps

    from scipy.stats import norm

    logit_p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x1 - 0.4 * x2)))
    y_bin_logit = (rng.uniform(size=n) < logit_p).astype(int)

    probit_p = norm.cdf(0.1 + 0.6 * x1 - 0.3 * x2)
    y_bin_probit = (rng.uniform(size=n) < probit_p).astype(int)

    # ---- New: IV system ----
    z = rng.normal(0.0, 1.0, size=n)              # instrument
    v = rng.normal(0.0, 1.0, size=n)              # first-stage shock
    eps_iv = rng.normal(0.0, 1.0, size=n)
    # First stage: x1_iv = 0.6*z + 0.4*v ; v correlated with eps_iv (endogeneity)
    x1_iv = 0.6 * z + 0.4 * v
    eps_iv_corr = eps_iv + 0.5 * v                # corr with x1_iv via v
    y_iv = 1.0 + 0.8 * x1_iv + eps_iv_corr        # true beta = 0.8

    # ---- New: clustered panel ----
    n_clusters = 50
    cluster = rng.integers(0, n_clusters, size=n)
    cluster_re = rng.normal(0.0, 0.8, size=n_clusters)[cluster]   # random effect
    eps_panel = rng.normal(0.0, 1.0, size=n)
    y_panel = 0.5 + 0.7 * x1 - 0.2 * x2 + cluster_re + eps_panel  # true beta_x1 = 0.7

    # ---- New: Poisson with offset ----
    exposure = rng.uniform(0.5, 3.0, size=n)
    log_mu = 0.3 + 0.4 * x1 - 0.2 * x2 + np.log(exposure)         # true beta_x1 = 0.4
    y_count = rng.poisson(np.exp(log_mu), size=n)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1, dtype=int),
            "x1": x1,
            "x2": x2,
            "y_lin": y_lin,
            "y_bin_logit": y_bin_logit,
            "y_bin_probit": y_bin_probit,
            "z": z,
            "x1_iv": x1_iv,
            "y_iv": y_iv,
            "cluster": cluster.astype(int),
            "y_panel": y_panel,
            "exposure": exposure,
            "y_count": y_count.astype(int),
        }
    )
    return df


def fit_reference(df: pd.DataFrame) -> dict:
    out: dict = {}

    # IV / 2SLS with HC1 robust SE on y_iv ~ x1_iv, instrument z
    from linearmodels.iv import IV2SLS

    iv = IV2SLS(
        dependent=df["y_iv"],
        exog=sm.add_constant(pd.DataFrame(index=df.index)),  # only constant in exog
        endog=df[["x1_iv"]],
        instruments=df[["z"]],
    ).fit(cov_type="robust", debiased=True)
    out["iv2sls_robust"] = {
        "vcov": "robust (HC1-like)",
        "estimates": [
            {
                "term": "const",
                "estimate": float(iv.params["const"]),
                "se": float(iv.std_errors["const"]),
                "n_obs": int(iv.nobs),
            },
            {
                "term": "x1_iv",
                "estimate": float(iv.params["x1_iv"]),
                "se": float(iv.std_errors["x1_iv"]),
                "n_obs": int(iv.nobs),
            },
        ],
    }

    # Cluster-robust OLS of y_panel on x1, x2 with intercept; cluster on cluster id
    X = sm.add_constant(df[["x1", "x2"]])
    cluster_ols = sm.OLS(df["y_panel"], X).fit(
        cov_type="cluster", cov_kwds={"groups": df["cluster"].values}
    )
    out["cluster_ols"] = {
        "vcov": "cluster (CR1)",
        "estimates": [
            {
                "term": "const",
                "estimate": float(cluster_ols.params["const"]),
                "se": float(cluster_ols.bse["const"]),
                "n_obs": int(cluster_ols.nobs),
            },
            {
                "term": "x1",
                "estimate": float(cluster_ols.params["x1"]),
                "se": float(cluster_ols.bse["x1"]),
                "n_obs": int(cluster_ols.nobs),
            },
            {
                "term": "x2",
                "estimate": float(cluster_ols.params["x2"]),
                "se": float(cluster_ols.bse["x2"]),
                "n_obs": int(cluster_ols.nobs),
            },
        ],
    }

    # Probit Average Marginal Effects (AME)
    probit = sm.Probit(df["y_bin_probit"], X).fit(disp=0)
    margeff = probit.get_margeff(at="overall", method="dydx")
    # margeff has tables — pull means and SEs for x1 and x2
    summ = margeff.summary_frame()
    out["probit_ame"] = {
        "vcov": "delta-method on probit AME",
        "estimates": [
            {
                "term": "x1",
                "estimate": float(summ.loc["x1", "dy/dx"]),
                "se": float(summ.loc["x1", "Std. Err."]),
                "n_obs": int(probit.nobs),
            },
            {
                "term": "x2",
                "estimate": float(summ.loc["x2", "dy/dx"]),
                "se": float(summ.loc["x2", "Std. Err."]),
                "n_obs": int(probit.nobs),
            },
        ],
    }

    # Poisson with offset = log(exposure)
    offset = np.log(df["exposure"].values)
    pois = sm.GLM(df["y_count"], X, family=sm.families.Poisson(), offset=offset).fit()
    out["poisson_offset"] = {
        "vcov": "asymptotic",
        "estimates": [
            {
                "term": "const",
                "estimate": float(pois.params["const"]),
                "se": float(pois.bse["const"]),
                "n_obs": int(pois.nobs),
            },
            {
                "term": "x1",
                "estimate": float(pois.params["x1"]),
                "se": float(pois.bse["x1"]),
                "n_obs": int(pois.nobs),
            },
            {
                "term": "x2",
                "estimate": float(pois.params["x2"]),
                "se": float(pois.bse["x2"]),
                "n_obs": int(pois.nobs),
            },
        ],
    }

    return out


def main() -> None:
    df = make_data(N)
    csv_path = HERE / "data" / "pilot_v2.csv"
    df.to_csv(csv_path, index=False)

    reference = fit_reference(df)
    (HERE / "reference" / "reference_v2.json").write_text(json.dumps(reference, indent=2))

    print(f"wrote {csv_path} ({len(df)} rows, {df.shape[1]} cols)")
    print(json.dumps(reference, indent=2))


if __name__ == "__main__":
    main()
