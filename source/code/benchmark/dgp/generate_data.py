"""Generate the pilot dataset and reference (reference) estimates.

Outputs:
  data/pilot.csv          single shared dataset
  reference/reference.json      reference estimates for the three tasks
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
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)
    eps = rng.normal(0.0, 1.0, size=n)
    y_lin = 1.0 + 0.5 * x1 - 0.3 * x2 + eps  # continuous outcome

    logit_p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x1 - 0.4 * x2)))
    y_bin_logit = (rng.uniform(size=n) < logit_p).astype(int)

    from scipy.stats import norm

    probit_p = norm.cdf(0.1 + 0.6 * x1 - 0.3 * x2)
    y_bin_probit = (rng.uniform(size=n) < probit_p).astype(int)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1, dtype=int),
            "x1": x1,
            "x2": x2,
            "y_lin": y_lin,
            "y_bin_logit": y_bin_logit,
            "y_bin_probit": y_bin_probit,
        }
    )
    return df


def fit_reference(df: pd.DataFrame) -> dict:
    X = sm.add_constant(df[["x1", "x2"]])

    ols = sm.OLS(df["y_lin"], X).fit(cov_type="HC1")
    logit = sm.Logit(df["y_bin_logit"], X).fit(disp=0)
    probit = sm.Probit(df["y_bin_probit"], X).fit(disp=0)

    def pack(res, term_order=("const", "x1", "x2")):
        return [
            {
                "term": t,
                "estimate": float(res.params[t]),
                "se": float(res.bse[t]),
                "n_obs": int(res.nobs),
            }
            for t in term_order
        ]

    return {
        "ols_robust": {"vcov": "HC1", "estimates": pack(ols)},
        "logit": {"vcov": "asymptotic", "estimates": pack(logit)},
        "probit": {"vcov": "asymptotic", "estimates": pack(probit)},
    }


def main() -> None:
    df = make_data(N)
    csv_path = HERE / "data" / "pilot.csv"
    df.to_csv(csv_path, index=False)

    reference = fit_reference(df)
    (HERE / "reference" / "reference.json").write_text(json.dumps(reference, indent=2))

    print(f"wrote {csv_path} ({len(df)} rows)")
    print(json.dumps(reference, indent=2))


if __name__ == "__main__":
    main()
