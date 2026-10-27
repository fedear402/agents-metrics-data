"""Pilot IV data: three new time-series tasks plus a simulated-DGP Cox PH task.

Datasets generated (each written to code/pilot/data/):
  - pilot_v4_arima.csv   T02-replacement: ARIMA(1,1,1) series of length 250
  - pilot_v4_cox.csv     T13 (simulated):  right-censored panel, n=2000
  - pilot_v4_var.csv     T21 / T22:        bivariate VAR(2) of length 200

References (also computed here for sanity-check; production manual references
live under code/pilot/manual_references/<task>/):
  - arima_forecast:  one-step-ahead point forecast from the fitted ARIMA(1,1,1)
                     plus 95% prediction interval. Sample-based reference.
  - cox_hazard:      Cox PH coefficient (log hazard ratio) for treatment and
                     for the continuous covariate, with model-based SE.
  - var_irf:         orthogonal IRF of y1 in response to a one-SD shock in y2
                     at horizon 4, Cholesky identification (y1 ordered first).
  - var_granger:     F-statistic, degrees of freedom, and p-value for
                     "y2 does not Granger-cause y1" in an OLS lag-augmented
                     regression with two lags.

SEED is fixed for replicability.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

SEED = 20260516
HERE = Path(__file__).parent

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# T02 (replacement): ARIMA(1,1,1) of length 250
# ---------------------------------------------------------------------------

def make_arima_data() -> pd.DataFrame:
    """Simulate an ARIMA(1,1,1) of length 250.

    Model in differenced form:  (1 - phi B) (y_t - y_{t-1}) = (1 + theta B) e_t,
    with phi = 0.5, theta = 0.3, e_t ~ N(0, 1).  Initial level y_0 = 0.
    """
    phi, theta, sigma = 0.5, 0.3, 1.0
    n = 250
    e = rng.normal(0.0, sigma, size=n + 1)
    diff = np.zeros(n)
    diff[0] = e[1]
    for t in range(1, n):
        diff[t] = phi * diff[t - 1] + e[t + 1] + theta * e[t]
    y = np.cumsum(diff)
    return pd.DataFrame({"t": np.arange(1, n + 1, dtype=int), "y": y})


def reference_arima_forecast(df: pd.DataFrame) -> dict:
    """Fit ARIMA(1,1,1) on the simulated series; report one-step-ahead forecast and 95% PI."""
    y = df["y"].values
    res = ARIMA(y, order=(1, 1, 1)).fit()
    fc = res.get_forecast(steps=1)
    mean = float(fc.predicted_mean[0])
    ci = fc.conf_int(alpha=0.05)[0]
    lo, hi = float(ci[0]), float(ci[1])
    return {
        "estimand": "ARIMA(1,1,1) one-step-ahead forecast with 95% prediction interval",
        "estimates": [
            {"term": "forecast",  "estimate": mean, "se": float(fc.se_mean[0]), "n_obs": int(len(y))},
            {"term": "lower_95",  "estimate": lo,   "se": float("nan"),         "n_obs": int(len(y))},
            {"term": "upper_95",  "estimate": hi,   "se": float("nan"),         "n_obs": int(len(y))},
        ],
        "n_obs": int(len(y)),
    }


# ---------------------------------------------------------------------------
# T13 (simulated): Cox PH on a right-censored panel
# ---------------------------------------------------------------------------

def make_cox_data() -> pd.DataFrame:
    """Simulate survival times under Cox PH with a Weibull baseline hazard.

    Subjects: n = 2000, binary treatment Bernoulli(0.5), continuous covariate x ~ N(0, 1).
    Hazard:   h(t | treat, x) = h_0(t) * exp(beta_treat * treat + beta_x * x),
              h_0(t) = shape * scale^(-shape) * t^(shape - 1) (Weibull).
    Coefficients:  beta_treat = -0.5  (HR = 0.6065),  beta_x = 0.3 (HR = 1.3499).
    Censoring:    C ~ Exp(rate = 0.30), independent of T.  Observed: t_obs = min(T, C),
                  event = 1{T <= C}; ~30% censoring at these parameter values.
    """
    n = 2000
    shape, scale = 1.5, 1.0
    beta_treat, beta_x = -0.5, 0.3

    treat = rng.binomial(1, 0.5, size=n).astype(int)
    x = rng.normal(0.0, 1.0, size=n)
    linpred = beta_treat * treat + beta_x * x
    u = rng.uniform(size=n)
    # T = scale * (-log(U) * exp(-linpred))**(1/shape)
    t_event = scale * (-np.log(u) * np.exp(-linpred)) ** (1.0 / shape)
    t_cens = rng.exponential(scale=1.0 / 0.30, size=n)
    t_obs = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)

    return pd.DataFrame(
        {
            "subject":  np.arange(n, dtype=int),
            "treat":    treat,
            "x":        x,
            "t_obs":    t_obs,
            "event":    event,
        }
    )


def reference_cox_hazard(df: pd.DataFrame) -> dict:
    """Cox PH coefficients via statsmodels PHReg. Reference: log HR for treat and x."""
    from statsmodels.duration.hazard_regression import PHReg

    Xmat = df[["treat", "x"]].values
    res = PHReg(df["t_obs"].values, Xmat, status=df["event"].values).fit(method="bfgs", disp=0)
    coef = res.params  # log hazard ratios
    se = res.bse
    return {
        "estimand": "Cox proportional hazards: log hazard ratios for treat and x; model-based SE",
        "estimates": [
            {"term": "treat", "estimate": float(coef[0]), "se": float(se[0]), "n_obs": int(len(df))},
            {"term": "x",     "estimate": float(coef[1]), "se": float(se[1]), "n_obs": int(len(df))},
        ],
        "events_total": int(df["event"].sum()),
        "n_obs": int(len(df)),
    }


# ---------------------------------------------------------------------------
# T21 / T22: bivariate VAR(2)
# ---------------------------------------------------------------------------

def make_var_data() -> pd.DataFrame:
    """Simulate a bivariate VAR(2) of length 200.

    y_t = A1 y_{t-1} + A2 y_{t-2} + e_t,  e_t ~ N(0, I).
        A1 = [[0.5, 0.2],
              [0.0, 0.4]],
        A2 = [[0.2, 0.0],
              [0.0, 0.1]].
    Burn 100 initial periods.
    """
    n = 200
    burn = 100
    A1 = np.array([[0.5, 0.2], [0.0, 0.4]])
    A2 = np.array([[0.2, 0.0], [0.0, 0.1]])
    y = np.zeros((n + burn + 2, 2))
    for t in range(2, n + burn + 2):
        y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + rng.normal(0.0, 1.0, size=2)
    y = y[burn + 2:]
    return pd.DataFrame({"t": np.arange(1, n + 1, dtype=int), "y1": y[:, 0], "y2": y[:, 1]})


def reference_var_irf(df: pd.DataFrame) -> dict:
    """Orthogonal IRF of y1 in response to a one-SD shock in y2 at horizon 4.

    Cholesky identification with y1 ordered first (so a shock to y2 has no
    contemporaneous effect on y1; the IRF at h=0 is zero by construction).
    """
    Y = df[["y1", "y2"]].values
    model = VAR(Y).fit(2)
    irf = model.irf(periods=10).orth_irfs  # shape (h+1, 2, 2): [h, response, shock]
    # Response of y1 (index 0) to shock in y2 (index 1) at h=4
    irf_y1_to_y2_h4 = float(irf[4, 0, 1])
    return {
        "estimand": "orthogonal IRF (Cholesky, y1 ordered first) of y1 response to one-SD shock in y2 at horizon 4",
        "estimates": [
            {"term": "irf_y1_to_y2_h4", "estimate": irf_y1_to_y2_h4, "se": float("nan"), "n_obs": int(len(df))},
        ],
        "lag_order": 2,
        "identification": "Cholesky, y1 first",
        "n_obs": int(len(df)),
    }


def reference_var_granger(df: pd.DataFrame) -> dict:
    """Granger F-test: does y2 Granger-cause y1 in a two-lag OLS regression?

    Restricted:  y1_t ~ y1_{t-1} + y1_{t-2}
    Unrestricted: y1_t ~ y1_{t-1} + y1_{t-2} + y2_{t-1} + y2_{t-2}
    Test the joint null that the two y2 coefficients are zero. Standard OLS F-test.
    """
    y1 = df["y1"].values
    y2 = df["y2"].values
    n = len(y1)
    # Build aligned design matrices for t = 3..n (using two lags)
    idx = np.arange(2, n)
    Y = y1[idx]
    lag1_y1, lag2_y1 = y1[idx - 1], y1[idx - 2]
    lag1_y2, lag2_y2 = y2[idx - 1], y2[idx - 2]

    X_res = sm.add_constant(np.column_stack([lag1_y1, lag2_y1]))
    X_unr = sm.add_constant(np.column_stack([lag1_y1, lag2_y1, lag1_y2, lag2_y2]))

    res_res = sm.OLS(Y, X_res).fit()
    res_unr = sm.OLS(Y, X_unr).fit()

    rss_res = float(np.sum(res_res.resid ** 2))
    rss_unr = float(np.sum(res_unr.resid ** 2))
    q = 2  # restrictions
    k_unr = X_unr.shape[1]
    n_used = len(Y)
    F = ((rss_res - rss_unr) / q) / (rss_unr / (n_used - k_unr))
    p_value = 1.0 - stats.f.cdf(F, q, n_used - k_unr)
    return {
        "estimand": "F-test of H0: y2 does not Granger-cause y1 with 2 lags; OLS-based",
        "estimates": [
            {"term": "F",         "estimate": float(F),       "se": float("nan"), "n_obs": int(n_used)},
            {"term": "df_num",    "estimate": float(q),       "se": float("nan"), "n_obs": int(n_used)},
            {"term": "df_den",    "estimate": float(n_used - k_unr), "se": float("nan"), "n_obs": int(n_used)},
            {"term": "p_value",   "estimate": float(p_value), "se": float("nan"), "n_obs": int(n_used)},
        ],
        "n_obs": int(n_used),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = HERE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = HERE / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    df_arima = make_arima_data()
    df_cox = make_cox_data()
    df_var = make_var_data()

    arima_csv = data_dir / "pilot_v4_arima.csv"
    cox_csv = data_dir / "pilot_v4_cox.csv"
    var_csv = data_dir / "pilot_v4_var.csv"

    df_arima.to_csv(arima_csv, index=False)
    df_cox.to_csv(cox_csv, index=False)
    df_var.to_csv(var_csv, index=False)

    references = {
        "arima_forecast": reference_arima_forecast(df_arima),
        "cox_hazard":     reference_cox_hazard(df_cox),
        "var_irf":        reference_var_irf(df_var),
        "var_granger":    reference_var_granger(df_var),
    }
    (ref_dir / "reference_v4.json").write_text(json.dumps(references, indent=2))

    for name, path, df in (
        ("arima", arima_csv, df_arima),
        ("cox",   cox_csv,   df_cox),
        ("var",   var_csv,   df_var),
    ):
        print(f"wrote {path}  ({len(df)} rows, {df.shape[1]} cols)")
    print()
    print(json.dumps(references, indent=2))


if __name__ == "__main__":
    main()
