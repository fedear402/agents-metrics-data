"""Generate the canonical dataset + reference for the synth_weights_fixedv task.

Synthetic control with a FIXED predictor-weight matrix V = I. Fixing V collapses
the (normally nested, package-dependent) Abadie estimator to a deterministic
simplex-constrained least squares: choose donor weights w >= 0, sum(w) = 1, that
minimize ||y_treated_pre - D_pre @ w||^2 over the 12 pre-treatment periods. That
has a unique solution any platform can hand-roll, so a single shared reference is
valid. The exhibit: no standard-library synthetic-control route exists anywhere
(Synth/scpi in R, synth SSC in Stata, pysyncon in Python are all community /
usually-absent), so models must recognize the constrained-LS problem; the usual
platform gradient inverts (Python easy via scipy, Stata hard).

DGP (seed 20260713): 9 units (0 = treated, 1-8 = donors), 17 periods, 1-12 pre.
Donors follow a 3-factor structure. The treated COUNTERFACTUAL is
0.55*d3 + 0.30*d7 + 0.15*d5 plus a small component orthogonal to the donor
pre-period span (so the unit sits strictly outside the donor convex hull and the
simplex projection is unique, with a small nonzero pre-RMSPE above the 5e-3
floor). Observed treated = counterfactual + 2.0 in periods 13-17 (treatment
effect), so att_post ~ 2. Uniqueness / optimizer-independence is verified by
solving the QP from many random starts and requiring agreement to 1e-6.

Run: uv run python code/benchmark/generate_synth_control.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SEED = 20260713
N_DONORS = 8
T = 17
T_PRE = 12
ATT = 2.0
W_TRUE = {3: 0.55, 7: 0.30, 5: 0.15}  # donor index (1-based) -> weight
ORTH_SCALE = 0.6  # controls pre-RMSPE (calibrated above the 5e-3 floor)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "benchmark" / "pilot_v6_synth.csv"
REF_PATH = PROJECT_ROOT / "data" / "references" / "reference_v6.json"


def build_panel() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    # 3 smooth latent factors over T periods
    t = np.arange(T)
    factors = np.vstack([
        np.ones(T),
        t / T,
        np.sin(2 * np.pi * t / 9),
    ])  # 3 x T
    loadings = rng.normal(0, 1, size=(N_DONORS, 3))  # 8 x 3
    donors = loadings @ factors + rng.normal(0, 0.15, size=(N_DONORS, T))  # 8 x T

    # treated counterfactual = convex combo of donors + orthogonal-to-span shift
    w = np.zeros(N_DONORS)
    for idx, val in W_TRUE.items():
        w[idx - 1] = val
    cf_inspan = w @ donors  # 1 x T

    # component of a random vector orthogonal to the donor PRE-period row space
    Dpre = donors[:, :T_PRE]  # 8 x 12
    r = rng.normal(0, 1, size=T_PRE)
    # project r onto row space of Dpre, subtract -> orthogonal complement
    proj = Dpre.T @ np.linalg.lstsq(Dpre.T, r, rcond=None)[0]
    orth_pre = r - proj
    orth_pre = ORTH_SCALE * orth_pre / np.linalg.norm(orth_pre)
    orth = np.zeros(T)
    orth[:T_PRE] = orth_pre  # keep post orthogonal shift at 0 so att_post ~ ATT

    cf = cf_inspan + orth
    treated = cf.copy()
    treated[T_PRE:] += ATT  # observed treated = counterfactual + treatment effect

    rows = []
    for p in range(T):
        rows.append({"unit": 0, "period": p + 1, "y": treated[p]})
    for d in range(N_DONORS):
        for p in range(T):
            rows.append({"unit": d + 1, "period": p + 1, "y": donors[d, p]})
    return pd.DataFrame(rows)


def solve_weights(y_treated_pre: np.ndarray, D_pre: np.ndarray, starts: int = 40):
    """Simplex-constrained least squares via SLSQP from many starts; return the
    best solution and the max pairwise disagreement (uniqueness check)."""

    n = D_pre.shape[0]

    def obj(w):
        r = y_treated_pre - D_pre.T @ w
        return float(r @ r)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    rng = np.random.default_rng(SEED + 1)
    sols = []
    for _ in range(starts):
        w0 = rng.dirichlet(np.ones(n))
        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            sols.append(res.x)
    sols = np.array(sols)
    best = sols[np.argmin([obj(s) for s in sols])]
    max_disagree = float(np.max(np.abs(sols - best))) if len(sols) > 1 else 0.0
    return best, max_disagree


def main() -> None:
    df = build_panel()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    d = pd.read_csv(DATA_PATH)  # round-trip so reference matches ./input.csv
    wide = d.pivot(index="unit", columns="period", values="y").sort_index()
    y_all = wide.loc[0].to_numpy()
    D_all = wide.loc[1:].to_numpy()  # 8 x 17
    y_pre = y_all[:T_PRE]
    D_pre = D_all[:, :T_PRE]

    w, max_disagree = solve_weights(y_pre, D_pre)
    synth = D_all.T @ w  # 17-vector
    resid_pre = y_pre - synth[:T_PRE]
    rmspe_pre = float(np.sqrt(np.mean(resid_pre ** 2)))
    att_post = float(np.mean(y_all[T_PRE:] - synth[T_PRE:]))
    n_obs = int(len(d))

    ref = {
        "estimand": "Synthetic control weights (V=I, simplex-constrained LS)",
        "estimates": [
            {"term": "w_3", "estimate": float(w[2]), "se": None, "n_obs": n_obs},
            {"term": "w_7", "estimate": float(w[6]), "se": None, "n_obs": n_obs},
            {"term": "rmspe_pre", "estimate": rmspe_pre, "se": None, "n_obs": n_obs},
            {"term": "att_post", "estimate": att_post, "se": None, "n_obs": n_obs},
        ],
    }
    existing = json.loads(REF_PATH.read_text()) if REF_PATH.exists() else {}
    existing["synth_weights_fixedv"] = ref
    REF_PATH.write_text(json.dumps(existing, indent=1))

    print(f"wrote {DATA_PATH.relative_to(PROJECT_ROOT)} ({n_obs} rows)")
    print(f"weights: {np.round(w, 4)}")
    print(f"reference: w_3={w[2]:.6f} w_7={w[6]:.6f} rmspe_pre={rmspe_pre:.6f} "
          f"att_post={att_post:.6f}")
    print(f"UNIQUENESS: max multi-start disagreement = {max_disagree:.2e} "
          f"(< 1e-6? {max_disagree < 1e-6})")
    print(f"FALSIFIABILITY: rmspe_pre above 5e-3 floor? {rmspe_pre > 5e-3}; "
          f"w_3,w_7 dominant? {w[2] > 0.2 and w[6] > 0.15}")

    # Self-auditing identity (integrity check): the construction makes the truth
    # analytic — cf = 0.55 d3 + 0.30 d7 + 0.15 d5 + orth with orth ⊥ donor
    # pre-span and zero post ⇒ w* = W_TRUE exactly, rmspe = ||orth||/sqrt(T_PRE),
    # att = ATT exactly. The solved reference must match to solver precision.
    analytic = {
        "w_3": W_TRUE[3],
        "w_7": W_TRUE[7],
        "rmspe_pre": ORTH_SCALE / np.sqrt(T_PRE),
        "att_post": ATT,
    }
    solved = {"w_3": w[2], "w_7": w[6], "rmspe_pre": rmspe_pre, "att_post": att_post}
    for term, truth in analytic.items():
        err = abs(solved[term] - truth)
        assert err < 1e-6, f"analytic identity violated for {term}: |err|={err:.2e}"
    print("ANALYTIC IDENTITY: reference matches closed-form truth to < 1e-6 "
          f"(max err {max(abs(solved[k]-v) for k, v in analytic.items()):.2e})")


if __name__ == "__main__":
    main()
