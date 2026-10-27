"""Generate the canonical dataset + reference for the bp_koenker task.

bp_koenker (Breusch-Pagan, Koenker studentized LM form) is an exploratory 23rd
task promoted from the team candidate pool
(`literature/task_references/candidate_tasks/01_linear_inference.md`). It is the
cleanest exhibit of the study's platform-default thesis: Stata `estat hettest`
(normal default), R `bptest` (studentized default), and Python
`het_breuschpagan` return different answers for the "same" named test. The prompt
pins exactly one convention: the Koenker studentized LM = n * R^2 of the
auxiliary regression of squared OLS residuals on the model regressors, chi-square
with 2 df, so a unique correct answer exists.

DGP (seed 20260713, n=600): x1, x2 ~ N(0,1); y_het = 0.5 + 1.5 x1 - 0.8 x2 + e
with Var(e | x) = exp(0.3 + 0.20 x1). The 0.20 loading is calibrated so bp_p lands
in the falsifiable band [0.02, 0.15] (measured ~0.05), well above the 5e-3
absolute tolerance floor. The reference is computed from the ROUND-TRIPPED CSV so
it matches exactly what models read as ./input.csv.

Run: uv run python code/benchmark/generate_bp_koenker.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

SEED = 20260713
N = 600
A1 = 0.20  # heteroskedasticity loading, calibrated for bp_p in [0.02, 0.15]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "benchmark" / "pilot_v6_bp_koenker.csv"
REF_PATH = PROJECT_ROOT / "data" / "references" / "reference_v6.json"


def generate_csv() -> None:
    rng = np.random.default_rng(SEED)
    x1 = rng.standard_normal(N)
    x2 = rng.standard_normal(N)
    sd = np.sqrt(np.exp(0.3 + A1 * x1))
    e = rng.standard_normal(N) * sd
    y_het = 0.5 + 1.5 * x1 - 0.8 * x2 + e
    df = pd.DataFrame(
        {"id": np.arange(1, N + 1), "x1": x1, "x2": x2, "y_het": y_het}
    )
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)


def bp_koenker_reference() -> dict:
    """Compute the pinned Koenker studentized LM from the round-tripped CSV."""

    df = pd.read_csv(DATA_PATH)
    X = sm.add_constant(df[["x1", "x2"]].to_numpy())
    ols = sm.OLS(df["y_het"].to_numpy(), X).fit()
    u2 = ols.resid ** 2
    aux = sm.OLS(u2, X).fit()  # squared residuals on x1, x2 + const
    lm = float(N * aux.rsquared)
    p = float(stats.chi2.sf(lm, 2))
    return {
        "estimand": "Breusch-Pagan heteroskedasticity test (Koenker studentized LM)",
        "estimates": [
            {"term": "bp_lm", "estimate": lm, "se": None, "n_obs": N},
            {"term": "bp_df", "estimate": 2.0, "se": None, "n_obs": N},
            {"term": "bp_p", "estimate": p, "se": None, "n_obs": N},
        ],
    }


def main() -> None:
    generate_csv()
    ref = bp_koenker_reference()
    payload = {"bp_koenker": ref}
    REF_PATH.write_text(json.dumps(payload, indent=1))
    lm = ref["estimates"][0]["estimate"]
    p = ref["estimates"][2]["estimate"]
    print(f"wrote {DATA_PATH.relative_to(PROJECT_ROOT)} (n={N})")
    print(f"wrote {REF_PATH.relative_to(PROJECT_ROOT)}")
    print(f"reference: bp_lm={lm:.6f}  bp_df=2  bp_p={p:.6f}")
    print(f"falsifiability: bp_p={p:.4f} in [0.02,0.15]? {0.02 <= p <= 0.15}; "
          f"above 5e-3 floor? {p > 5e-3}")


if __name__ == "__main__":
    main()
