"""Materialize runnable Monte Carlo datasets and references for supported tasks.

The broader Monte Carlo plan intentionally includes every active task. This
script promotes the subset whose DGP and reference output can be generated
deterministically from Python without hand adjudication.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent.parent
DEFAULT_REGISTRY = ROOT / "task_registry.json"
DEFAULT_PLAN = (
    PROJECT_ROOT / "data" / "generated" / "2026-05-19_monte_carlo_plan" / "monte_carlo_plan.json"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "data" / "generated" / "2026-05-19_monte_carlo_realizations"
)

SUPPORTED_TASKS = {
    "ols_robust",
    "arima_forecast",
    "logit",
    "probit",
    "cluster_ols",
    "two_way_cluster",
    "iv2sls_robust",
    "poisson_offset",
    "vif",
    "cooks_distance",
    "ttest_welch",
    "anova_oneway",
    "chisq_indep",
    "cox_hazard",
    "twoway_fe",
    "dynpanel_ab",
    "staggered_did",
    "rdd_sharp",
    "lp_irf",
    "wcb_few",
    "var_irf",
    "var_granger",
}


def safe_token(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()


def variant_id(task_id: str, axis: str, value: Any, seed: int) -> str:
    return f"{task_id}__mc__{safe_token(axis)}_{safe_token(value)}__seed_{seed}"


def seeded_rng(seed: int, variant: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{variant}".encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return np.random.default_rng(seed + offset)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_safe(inner) for inner in value]
    return value


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def pack_result(res: Any, terms: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "term": term,
            "estimate": float(res.params[term]),
            "se": float(res.bse[term]),
            "n_obs": int(res.nobs),
        }
        for term in terms
    ]


def base_binary_frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y_lin = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(0.0, 1.0, n)
    logit_p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x1 - 0.4 * x2)))
    y_bin_logit = (rng.uniform(size=n) < logit_p).astype(int)
    probit_p = stats.norm.cdf(0.1 + 0.6 * x1 - 0.3 * x2)
    y_bin_probit = (rng.uniform(size=n) < probit_p).astype(int)
    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1, dtype=int),
            "x1": x1,
            "x2": x2,
            "y_lin": y_lin,
            "y_bin_logit": y_bin_logit,
            "y_bin_probit": y_bin_probit,
        }
    )


def generate_ols(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"n": 2000, "signal_to_noise": 1.0, "heteroskedasticity_strength": 0.5}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n = int(params["n"])
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    signal_sd = np.std(0.5 * x1 - 0.3 * x2)
    eps_base = signal_sd / float(params["signal_to_noise"])
    eps_sd = eps_base * (1.0 + float(params["heteroskedasticity_strength"]) * np.abs(x1))
    y_lin = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(0.0, eps_sd, n)
    df = base_binary_frame(rng, n)
    df["x1"] = x1
    df["x2"] = x2
    df["y_lin"] = y_lin
    fit = sm.OLS(df["y_lin"], sm.add_constant(df[["x1", "x2"]])).fit(cov_type="HC1")
    ref = {"vcov": "HC1", "estimates": pack_result(fit, ["const", "x1", "x2"])}
    return df, ref, {"baseline_parameters": params}


def generate_binary(
    variant: dict[str, Any],
    model: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"base_rate": 0.5, "effect_size": 0.8, "separation_strength": "none"}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n = 2000
    df = base_binary_frame(rng, n)
    sep_scale = {"none": 1.0, "partial": 1.7, "near": 2.5}[params["separation_strength"]]
    effect = float(params["effect_size"]) * sep_scale
    index = effect * df["x1"].to_numpy() - 0.5 * effect * df["x2"].to_numpy()
    if model == "logit":
        intercept = math.log(float(params["base_rate"]) / (1.0 - float(params["base_rate"])))
        p = 1.0 / (1.0 + np.exp(-(intercept + index)))
        df["y_bin_logit"] = (rng.uniform(size=n) < p).astype(int)
        fit = sm.Logit(df["y_bin_logit"], sm.add_constant(df[["x1", "x2"]])).fit(
            disp=0,
            maxiter=200,
        )
        ref = {"vcov": "asymptotic", "estimates": pack_result(fit, ["const", "x1", "x2"])}
    else:
        intercept = stats.norm.ppf(float(params["base_rate"]))
        p = stats.norm.cdf(intercept + index)
        df["y_bin_probit"] = (rng.uniform(size=n) < p).astype(int)
        fit = sm.Probit(df["y_bin_probit"], sm.add_constant(df[["x1", "x2"]])).fit(
            disp=0,
            maxiter=200,
        )
        ref = {"vcov": "asymptotic", "estimates": pack_result(fit, ["const", "x1", "x2"])}
    return df, ref, {"baseline_parameters": params}


def generate_cluster(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"n_clusters": 50, "cluster_size": 40, "intraclass_correlation": 0.2}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n_clusters = int(params["n_clusters"])
    cluster_size = int(params["cluster_size"])
    n = n_clusters * cluster_size
    df = base_binary_frame(rng, n)
    cluster = np.repeat(np.arange(n_clusters), cluster_size)
    icc = float(params["intraclass_correlation"])
    cluster_sd = math.sqrt(max(icc, 0.0))
    eps_sd = math.sqrt(max(1.0 - icc, 0.05))
    y_panel = (
        0.5
        + 0.7 * df["x1"].to_numpy()
        - 0.2 * df["x2"].to_numpy()
        + rng.normal(0.0, cluster_sd, n_clusters)[cluster]
        + rng.normal(0.0, eps_sd, n)
    )
    df["cluster"] = cluster.astype(int)
    df["y_panel"] = y_panel
    df["z"] = rng.normal(0.0, 1.0, n)
    df["x1_iv"] = 0.6 * df["z"] + rng.normal(0.0, 1.0, n)
    df["y_iv"] = 1.0 + 0.8 * df["x1_iv"] + rng.normal(0.0, 1.0, n)
    df["exposure"] = rng.uniform(0.5, 3.0, n)
    df["y_count"] = rng.poisson(np.exp(0.3 + 0.4 * df["x1"] - 0.2 * df["x2"]) * df["exposure"])
    fit = sm.OLS(df["y_panel"], sm.add_constant(df[["x1", "x2"]])).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["cluster"].to_numpy()},
    )
    ref = {"vcov": "cluster (CR1)", "estimates": pack_result(fit, ["const", "x1", "x2"])}
    return df, ref, {"baseline_parameters": params}


def generate_iv(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    from linearmodels.iv import IV2SLS

    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"first_stage_strength": 10, "endogeneity_strength": 0.4, "n": 2000}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n = int(params["n"])
    df = base_binary_frame(rng, n)
    z = rng.normal(0.0, 1.0, n)
    v = rng.normal(0.0, 1.0, n)
    pi = {3: 0.18, 10: 0.38, 30: 0.75}[int(params["first_stage_strength"])]
    x1_iv = pi * z + v
    rho = float(params["endogeneity_strength"])
    eps = rho * v + math.sqrt(max(1.0 - rho**2, 0.0)) * rng.normal(0.0, 1.0, n)
    y_iv = 1.0 + 0.8 * x1_iv + eps
    df["z"] = z
    df["x1_iv"] = x1_iv
    df["y_iv"] = y_iv
    df["cluster"] = rng.integers(0, 50, n)
    df["y_panel"] = 0.5 + 0.7 * df["x1"] - 0.2 * df["x2"] + rng.normal(0.0, 1.0, n)
    df["exposure"] = rng.uniform(0.5, 3.0, n)
    df["y_count"] = rng.poisson(np.exp(0.3 + 0.4 * df["x1"] - 0.2 * df["x2"]) * df["exposure"])
    fit = IV2SLS(
        dependent=df["y_iv"],
        exog=sm.add_constant(pd.DataFrame(index=df.index)),
        endog=df[["x1_iv"]],
        instruments=df[["z"]],
    ).fit(cov_type="robust", debiased=True)
    ref = {
        "vcov": "robust (HC1-like)",
        "estimates": [
            {
                "term": "const",
                "estimate": float(fit.params["const"]),
                "se": float(fit.std_errors["const"]),
                "n_obs": int(fit.nobs),
            },
            {
                "term": "x1_iv",
                "estimate": float(fit.params["x1_iv"]),
                "se": float(fit.std_errors["x1_iv"]),
                "n_obs": int(fit.nobs),
            },
        ],
    }
    return df, ref, {"baseline_parameters": params}


def generate_poisson(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"exposure_variation": "baseline", "count_overdispersion": 0.0, "effect_size": 0.4}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n = 2000
    df = base_binary_frame(rng, n)
    exposure_bounds = {
        "low": (0.8, 1.2),
        "baseline": (0.5, 3.0),
        "high": (0.1, 8.0),
    }[params["exposure_variation"]]
    exposure = rng.uniform(*exposure_bounds, n)
    effect = float(params["effect_size"])
    log_mu = 0.3 + effect * df["x1"].to_numpy() - 0.5 * effect * df["x2"].to_numpy() + np.log(exposure)
    mu = np.exp(log_mu)
    overdispersion = float(params["count_overdispersion"])
    if overdispersion > 0:
        shock = rng.gamma(shape=1.0 / overdispersion, scale=overdispersion, size=n)
        mu = mu * shock
    df["exposure"] = exposure
    df["y_count"] = rng.poisson(mu)
    df["cluster"] = rng.integers(0, 50, n)
    df["z"] = rng.normal(0.0, 1.0, n)
    df["x1_iv"] = 0.6 * df["z"] + rng.normal(0.0, 1.0, n)
    df["y_iv"] = 1.0 + 0.8 * df["x1_iv"] + rng.normal(0.0, 1.0, n)
    df["y_panel"] = 0.5 + 0.7 * df["x1"] - 0.2 * df["x2"] + rng.normal(0.0, 1.0, n)
    fit = sm.GLM(
        df["y_count"],
        sm.add_constant(df[["x1", "x2"]]),
        family=sm.families.Poisson(),
        offset=np.log(df["exposure"].to_numpy()),
    ).fit()
    ref = {"vcov": "asymptotic", "estimates": pack_result(fit, ["const", "x1", "x2"])}
    return df, ref, {"baseline_parameters": params}


def generate_collinear(
    variant: dict[str, Any],
    cooks: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"collinearity_strength": 0.7, "n": 500, "number_of_regressors": 5}
    if cooks:
        params = {"outlier_leverage": 0.5, "outlier_residual_size": 2.5, "n": 500}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n = int(params["n"])
    k = int(params.get("number_of_regressors", 5))
    x1 = rng.normal(0.0, 1.0, n)
    data = {"x1": x1}
    col_strength = float(params.get("collinearity_strength", 0.7))
    for j in range(2, k + 1):
        data[f"x{j}"] = col_strength * x1 + math.sqrt(max(1.0 - col_strength**2, 0.01)) * rng.normal(0.0, 1.0, n)
    X = pd.DataFrame(data)
    y = 1.0 + sum((1.0 / j) * X[f"x{j}"] for j in range(1, k + 1)) + rng.normal(0.0, 1.0, n)
    if cooks:
        leverage = float(params["outlier_leverage"])
        residual = float(params["outlier_residual_size"])
        X.loc[0, "x1"] = X["x1"].mean() + leverage * 8.0
        y.iloc[0] = y.mean() + residual * y.std(ddof=1)
    df = pd.DataFrame({"y": y, **{col: X[col] for col in X.columns}})
    terms = [f"x{j}" for j in range(1, k + 1)]
    fit = sm.OLS(df["y"], sm.add_constant(df[terms])).fit()
    if cooks:
        cooks_d = fit.get_influence().cooks_distance[0]
        order = np.argsort(cooks_d)[::-1][:5]
        estimates = []
        for rank, obs_id in enumerate(order, start=1):
            estimates.extend(
                [
                    {"term": f"rank{rank}_obs_id", "estimate": int(obs_id), "se": None, "n_obs": int(n)},
                    {
                        "term": f"rank{rank}_cooks_d",
                        "estimate": float(cooks_d[obs_id]),
                        "se": None,
                        "n_obs": int(n),
                    },
                ]
            )
        ref = {"vcov": "OLS influence", "estimates": estimates}
        task_update = {"baseline_parameters": params}
    else:
        X_with_const = sm.add_constant(df[terms])
        estimates = [
            {
                "term": f"vif_x{j}",
                "estimate": float(variance_inflation_factor(X_with_const.to_numpy(), j)),
                "se": None,
                "n_obs": int(n),
            }
            for j in range(1, k + 1)
        ]
        ref = {"vcov": "VIF auxiliary regressions", "estimates": estimates}
        task_update = {
            "estimands": [f"vif_x{j}" for j in range(1, k + 1)],
            "prompt": (
                f"For the linear model y on {', '.join(terms)}, compute the VIF "
                f"for each regressor. Report rows "
                f"{', '.join(f'vif_x{j}' for j in range(1, k + 1))}."
            ),
            "data_description": (
                f"Dataset: pilot.csv. Columns: y, {', '.join(terms)}."
            ),
            "baseline_parameters": params,
        }
    return df, ref, task_update


def generate_ttest(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"mean_difference": 0.3, "variance_ratio": 2.25, "group_imbalance": "balanced"}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    ratio = {"balanced": (100, 100), "2_to_1": (100, 200), "5_to_1": (100, 500)}[params["group_imbalance"]]
    n0, n1 = ratio
    y0 = rng.normal(0.0, 1.0, n0)
    y1 = rng.normal(float(params["mean_difference"]), math.sqrt(float(params["variance_ratio"])), n1)
    df = pd.DataFrame({"group": np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)], "y": np.r_[y0, y1]})
    mean0, mean1 = float(y0.mean()), float(y1.mean())
    var0, var1 = float(y0.var(ddof=1)), float(y1.var(ddof=1))
    se = math.sqrt(var0 / n0 + var1 / n1)
    t_stat = (mean1 - mean0) / se
    df_welch = (var0 / n0 + var1 / n1) ** 2 / ((var0 / n0) ** 2 / (n0 - 1) + (var1 / n1) ** 2 / (n1 - 1))
    p_value = 2.0 * stats.t.sf(abs(t_stat), df_welch)
    ref = {
        "vcov": "Welch",
        "estimates": [
            {"term": "t", "estimate": float(t_stat), "se": None, "n_obs": int(n0 + n1)},
            {"term": "df", "estimate": float(df_welch), "se": None, "n_obs": int(n0 + n1)},
            {"term": "p_value", "estimate": float(p_value), "se": None, "n_obs": int(n0 + n1)},
        ],
    }
    return df, ref, {"baseline_parameters": params}


def generate_anova(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"between_group_spread": 0.5, "within_group_sd": 1.0, "n_per_group": 50}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    spread = float(params["between_group_spread"])
    means = [-1.5 * spread, -0.5 * spread, 0.5 * spread, 1.5 * spread]
    per = int(params["n_per_group"])
    groups = [rng.normal(mean, float(params["within_group_sd"]), per) for mean in means]
    df = pd.concat(
        [pd.DataFrame({"group": idx, "y": values}) for idx, values in enumerate(groups)],
        ignore_index=True,
    )
    f_stat, p_value = stats.f_oneway(*groups)
    ref = {
        "vcov": "one-way ANOVA",
        "estimates": [
            {"term": "F", "estimate": float(f_stat), "se": None, "n_obs": int(len(df))},
            {"term": "df_num", "estimate": 3.0, "se": None, "n_obs": int(len(df))},
            {"term": "df_den", "estimate": float(len(df) - 4), "se": None, "n_obs": int(len(df))},
            {"term": "p_value", "estimate": float(p_value), "se": None, "n_obs": int(len(df))},
        ],
    }
    return df, ref, {"baseline_parameters": params}


def generate_chisq(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"association_strength": 0.1, "cell_sparsity": "none", "n": 600}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n = int(params["n"])
    sparsity_probs = {
        "none": ([0.45, 0.35, 0.20], [0.35, 0.40, 0.25]),
        "moderate": ([0.70, 0.25, 0.05], [0.40, 0.50, 0.10]),
        "high": ([0.88, 0.10, 0.02], [0.55, 0.40, 0.05]),
    }
    p0, p1 = sparsity_probs[params["cell_sparsity"]]
    strength = float(params["association_strength"])
    p1 = np.array(p1)
    p0 = np.array(p0)
    p1 = (1.0 - strength) * p0 + strength * p1
    p1 = p1 / p1.sum()
    row = rng.choice([0, 1], size=n, p=[0.5, 0.5])
    col = np.array([rng.choice([0, 1, 2], p=p1 if item else p0) for item in row])
    df = pd.DataFrame({"row": row, "col": col})
    table = pd.crosstab(df["row"], df["col"]).reindex(
        index=[0, 1],
        columns=[0, 1, 2],
        fill_value=0,
    )
    chi2, p_value, dof, _ = stats.chi2_contingency(table, correction=False)
    ref = {
        "vcov": "Pearson chi-square",
        "estimates": [
            {"term": "chi2", "estimate": float(chi2), "se": None, "n_obs": int(n)},
            {"term": "df", "estimate": float(dof), "se": None, "n_obs": int(n)},
            {"term": "p_value", "estimate": float(p_value), "se": None, "n_obs": int(n)},
        ],
    }
    return df, ref, {"baseline_parameters": params}


def generate_arima(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"n": 250, "phi": 0.5, "theta": 0.3, "innovation_sigma": 1.0}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n = int(params["n"])
    phi = float(params["phi"])
    theta = float(params["theta"])
    sigma = float(params["innovation_sigma"])
    e = rng.normal(0.0, sigma, size=n + 1)
    diff = np.zeros(n)
    diff[0] = e[1]
    for t in range(1, n):
        diff[t] = phi * diff[t - 1] + e[t + 1] + theta * e[t]
    y = np.cumsum(diff)
    df = pd.DataFrame({"t": np.arange(1, n + 1, dtype=int), "y": y})
    fit = ARIMA(df["y"].to_numpy(), order=(1, 1, 1)).fit()
    fc = fit.get_forecast(steps=1)
    ci = fc.conf_int(alpha=0.05)[0]
    ref = {
        "estimand": "ARIMA(1,1,1) one-step-ahead forecast with 95% prediction interval",
        "estimates": [
            {
                "term": "forecast",
                "estimate": float(fc.predicted_mean[0]),
                "se": float(fc.se_mean[0]),
                "n_obs": n,
            },
            {"term": "lower_95", "estimate": float(ci[0]), "se": None, "n_obs": n},
            {"term": "upper_95", "estimate": float(ci[1]), "se": None, "n_obs": n},
        ],
        "n_obs": n,
    }
    return df, ref, {"baseline_parameters": params}


def generate_panel_data(
    rng: np.random.Generator,
    n_firms: int,
    n_years: int,
    *,
    within_variation_share: float = 0.4,
    firm_year_error_share: str = "balanced",
) -> pd.DataFrame:
    n = n_firms * n_years
    firm = np.repeat(np.arange(n_firms, dtype=int), n_years)
    year = np.tile(np.arange(n_years, dtype=int), n_firms)
    firm_component = rng.normal(0.0, 1.0, n_firms)[firm]
    within_component = rng.normal(0.0, 1.0, n)
    share = float(within_variation_share)
    x = math.sqrt(share) * within_component + math.sqrt(max(1.0 - share, 0.0)) * firm_component

    error_scales = {
        "mostly_firm": (1.0, 0.25, 0.5),
        "balanced": (0.7, 0.7, 0.5),
        "mostly_year": (0.25, 1.0, 0.5),
    }[firm_year_error_share]
    firm_sd, year_sd, eps_sd = error_scales
    firm_effect = rng.normal(0.0, 0.8, n_firms)[firm]
    year_effect = rng.normal(0.0, 0.4, n_years)[year]
    error = (
        rng.normal(0.0, firm_sd, n_firms)[firm]
        + rng.normal(0.0, year_sd, n_years)[year]
        + rng.normal(0.0, eps_sd, n)
    )
    y = 1.0 + 0.55 * x + firm_effect + year_effect + error
    return pd.DataFrame({"firm": firm, "year": year, "x": x, "y_twocluster": y})


def generate_two_way_cluster(
    variant: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"n_firms": 200, "n_years": 10, "firm_year_error_share": "balanced"}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    df = generate_panel_data(
        rng,
        int(params["n_firms"]),
        int(params["n_years"]),
        firm_year_error_share=str(params["firm_year_error_share"]),
    )
    fit = sm.OLS(df["y_twocluster"], sm.add_constant(df[["x"]])).fit()
    cov_both, _, _ = cov_cluster_2groups(
        fit,
        df["firm"].to_numpy(),
        df["year"].to_numpy(),
    )
    se = np.sqrt(np.diag(cov_both))
    ref = {
        "vcov": "two-way cluster (CGM) on (firm, year)",
        "estimates": [
            {
                "term": "const",
                "estimate": float(fit.params["const"]),
                "se": float(se[0]),
                "n_obs": int(fit.nobs),
            },
            {
                "term": "x",
                "estimate": float(fit.params["x"]),
                "se": float(se[1]),
                "n_obs": int(fit.nobs),
            },
        ],
    }
    return df, ref, {"baseline_parameters": params}


def generate_twoway_fe(
    variant: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"n_firms": 200, "n_years": 10, "within_variation_share": 0.4}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    df = generate_panel_data(
        rng,
        int(params["n_firms"]),
        int(params["n_years"]),
        within_variation_share=float(params["within_variation_share"]),
    )
    firm_dummies = pd.get_dummies(df["firm"], prefix="firm", drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(df["year"], prefix="year", drop_first=True, dtype=float)
    design = pd.concat([df[["x"]].astype(float), firm_dummies, year_dummies], axis=1)
    fit = sm.OLS(df["y_twocluster"], sm.add_constant(design)).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["firm"].to_numpy()},
    )
    ref = {
        "estimand": "Two-way fixed effects coefficient with firm-clustered standard error",
        "estimates": [
            {
                "term": "x",
                "estimate": float(fit.params["x"]),
                "se": float(fit.bse["x"]),
                "n_obs": int(fit.nobs),
            }
        ],
    }
    return df, ref, {"baseline_parameters": params}


def generate_cox(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    from statsmodels.duration.hazard_regression import PHReg

    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"censoring_rate": 0.3, "hazard_ratio": 0.6, "baseline_shape": 1.5}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n = 2000
    shape = float(params["baseline_shape"])
    beta_treat = math.log(float(params["hazard_ratio"]))
    beta_x = 0.3
    treat = rng.binomial(1, 0.5, n).astype(int)
    x = rng.normal(0.0, 1.0, n)
    linpred = beta_treat * treat + beta_x * x
    u = rng.uniform(size=n)
    t_event = (-np.log(u) * np.exp(-linpred)) ** (1.0 / shape)
    target_censor = float(params["censoring_rate"])
    censor_scale = np.quantile(t_event, max(0.05, min(0.95, 1.0 - target_censor)))
    t_cens = rng.exponential(scale=max(float(censor_scale), 0.05), size=n)
    t_obs = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)
    df = pd.DataFrame(
        {
            "subject": np.arange(n, dtype=int),
            "treat": treat,
            "x": x,
            "t_obs": t_obs,
            "event": event,
        }
    )
    fit = PHReg(
        df["t_obs"].to_numpy(),
        df[["treat", "x"]].to_numpy(),
        status=df["event"].to_numpy(),
    ).fit(method="bfgs", disp=0)
    ref = {
        "estimand": "Cox proportional hazards: log hazard ratios for treat and x; model-based SE",
        "estimates": [
            {
                "term": "treat",
                "estimate": float(fit.params[0]),
                "se": float(fit.bse[0]),
                "n_obs": n,
            },
            {
                "term": "x",
                "estimate": float(fit.params[1]),
                "se": float(fit.bse[1]),
                "n_obs": n,
            },
        ],
        "events_total": int(df["event"].sum()),
        "n_obs": n,
    }
    return df, ref, {"baseline_parameters": params}


def generate_dynpanel(
    variant: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"rho": 0.5, "n_firms": 100, "n_periods": 10}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n_firms = int(params["n_firms"])
    n_periods = int(params["n_periods"])
    rho = float(params["rho"])
    beta_x = 0.5
    alpha = rng.normal(0.0, 0.5, n_firms)
    x = rng.normal(0.0, 1.0, (n_firms, n_periods))
    y = np.zeros((n_firms, n_periods))
    y[:, 0] = alpha + rng.normal(0.0, 1.0, n_firms)
    for t in range(1, n_periods):
        y[:, t] = rho * y[:, t - 1] + beta_x * x[:, t] + alpha + rng.normal(0.0, 1.0, n_firms)
    rows = [
        {"firm": i, "year": t, "y": y[i, t], "x": x[i, t]}
        for i in range(n_firms)
        for t in range(n_periods)
    ]
    df = pd.DataFrame(rows)

    dy_list, dy_lag_list, dx_list, z_list = [], [], [], []
    for i in range(n_firms):
        yi = y[i]
        xi = x[i]
        for t in range(2, n_periods):
            dy_list.append(yi[t] - yi[t - 1])
            dy_lag_list.append(yi[t - 1] - yi[t - 2])
            dx = xi[t] - xi[t - 1]
            dx_list.append(dx)
            z_y = np.zeros(n_periods - 2)
            z_y[: t - 1] = yi[: t - 1]
            z_list.append(np.concatenate([z_y, [dx]]))
    d_y = np.array(dy_list)
    x_stack = np.column_stack([dy_lag_list, dx_list])
    z_stack = np.array(z_list)
    ztz_inv = np.linalg.pinv(z_stack.T @ z_stack)
    xz = x_stack.T @ z_stack
    beta = np.linalg.solve(xz @ ztz_inv @ xz.T, xz @ ztz_inv @ z_stack.T @ d_y)
    ref = {
        "estimand": "First-difference dynamic panel GMM coefficient reference",
        "estimates": [
            {"term": "rho", "estimate": float(beta[0]), "se": None, "n_obs": int(len(d_y))},
            {"term": "beta_x", "estimate": float(beta[1]), "se": None, "n_obs": int(len(d_y))},
        ],
    }
    return df, ref, {"baseline_parameters": params}


def generate_staggered_did(
    variant: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {
        "effect_heterogeneity": 1.0,
        "cohort_share": "balanced",
        "pretrend_slope": 0.0,
    }
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n_firms = 240
    n_years = 10
    cohort_probs = {
        "balanced": [0.25, 0.25, 0.25, 0.25],
        "early_heavy": [0.20, 0.45, 0.25, 0.10],
        "late_heavy": [0.20, 0.10, 0.25, 0.45],
    }[params["cohort_share"]]
    cohorts = rng.choice([-1, 4, 6, 8], size=n_firms, p=cohort_probs)
    firm_fe = rng.normal(0.0, 0.8, n_firms)
    year_fe = rng.normal(0.0, 0.3, n_years)
    het = float(params["effect_heterogeneity"])
    tau = {4: 1.5 - 0.5 * het, 6: 1.5 + 0.5 * het, 8: 1.5 + 1.5 * het}
    rows = []
    for i in range(n_firms):
        cohort = int(cohorts[i])
        for year in range(n_years):
            treated = cohort != -1 and year >= cohort
            pretrend = float(params["pretrend_slope"]) * year * (cohort != -1)
            y_did = (
                firm_fe[i]
                + year_fe[year]
                + pretrend
                + (tau.get(cohort, 0.0) if treated else 0.0)
                + rng.normal(0.0, 1.0)
            )
            rows.append(
                {
                    "firm": i,
                    "year": year,
                    "cohort": cohort,
                    "treat_post": int(treated),
                    "y_did": y_did,
                }
            )
    df = pd.DataFrame(rows)
    cells = []
    for cohort in (4, 6, 8):
        for year in range(cohort, n_years):
            base = cohort - 1
            treated_t = df.loc[(df["cohort"] == cohort) & (df["year"] == year), "y_did"].mean()
            treated_base = df.loc[(df["cohort"] == cohort) & (df["year"] == base), "y_did"].mean()
            control_t = df.loc[(df["cohort"] == -1) & (df["year"] == year), "y_did"].mean()
            control_base = df.loc[(df["cohort"] == -1) & (df["year"] == base), "y_did"].mean()
            att = float((treated_t - treated_base) - (control_t - control_base))
            cells.append({"g": cohort, "t": year, "att_gt": att})
    ref = {
        "estimand": "ATT_uniform = (1/|S|) * sum_{(g,t) in S} ATT(g,t)",
        "vcov": "not scored; task asks for direct cell means only",
        "estimates": [
            {
                "term": "att",
                "estimate": float(np.mean([row["att_gt"] for row in cells])),
                "se": None,
                "n_obs": int(len(df)),
            }
        ],
        "n_cells": len(cells),
        "cells": cells,
    }
    return df, ref, {"baseline_parameters": params}


def generate_rdd(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"bandwidth": 0.5, "running_variable_density": "uniform", "jump_size": 1.0}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n = 1000
    density = params["running_variable_density"]
    if density == "uniform":
        z = rng.uniform(-1.0, 1.0, n)
    elif density == "center_sparse":
        z = np.sign(rng.uniform(-1.0, 1.0, n)) * np.sqrt(rng.uniform(0.0, 1.0, n))
    else:
        z = rng.triangular(-1.0, 0.0, 1.0, n)
    d = (z >= 0).astype(int)
    y = 0.5 + float(params["jump_size"]) * d + 0.3 * z + rng.normal(0.0, 0.5, n)
    df = pd.DataFrame({"z": z, "d": d, "y": y})
    h = float(params["bandwidth"])
    window = df.loc[df["z"].abs() <= h].copy()
    weights = 1.0 - np.abs(window["z"].to_numpy()) / h
    left = window["z"] < 0
    right = window["z"] >= 0
    left_fit = sm.WLS(
        window.loc[left, "y"],
        sm.add_constant(window.loc[left, ["z"]]),
        weights=weights[left.to_numpy()],
    ).fit()
    right_fit = sm.WLS(
        window.loc[right, "y"],
        sm.add_constant(window.loc[right, ["z"]]),
        weights=weights[right.to_numpy()],
    ).fit()
    late = float(right_fit.params["const"] - left_fit.params["const"])
    n_window = int(len(window))
    ref = {
        "estimand": f"Sharp RD local-linear LATE with triangular kernel and h={h:g}",
        "estimates": [{"term": "late", "estimate": late, "se": None, "n_obs": n_window}],
    }
    task_update = {
        "baseline_parameters": params,
        "prompt": (
            f"Using bandwidth h = {h:g} and triangular kernel weights, fit local "
            "linear regressions on each side of cutoff z = 0 and report the "
            "difference in intercepts as row late."
        ),
    }
    return df, ref, task_update


def generate_lp_irf(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"shock_strength": 0.5, "persistence": 0.5, "horizon": 4}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n = 250
    h = int(params["horizon"])
    shock = rng.normal(0.0, 1.0, n)
    eps = rng.normal(0.0, 1.0, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = float(params["persistence"]) * y[t - 1] + float(params["shock_strength"]) * shock[t] + eps[t]
    df = pd.DataFrame({"t": np.arange(1, n + 1, dtype=int), "y": y, "shock": shock})
    y_future = y[h + 1 :]
    shock_t = shock[1 : n - h]
    y_lag = y[: n - h - 1]
    fit = sm.OLS(y_future, sm.add_constant(np.column_stack([shock_t, y_lag]))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": h},
    )
    term = f"irf_h{h}"
    ref = {
        "estimand": f"Local-projection IRF at horizon {h} with Newey-West lag {h} standard error",
        "estimates": [
            {
                "term": term,
                "estimate": float(fit.params[1]),
                "se": float(fit.bse[1]),
                "n_obs": int(len(y_future)),
            }
        ],
    }
    task_update = {
        "estimands": [term],
        "baseline_parameters": params,
        "prompt": (
            f"Estimate the local projection y_{{t+{h}}} on shock_t and y_{{t-1}}, "
            f"with an intercept, using Newey-West standard error with lag {h}. "
            f"Report row {term}."
        ),
    }
    return df, ref, task_update


def generate_wcb(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"cluster_count": 5, "cluster_imbalance": "none", "effect_size": 0.4}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    n_clusters = int(params["cluster_count"])
    if params["cluster_imbalance"] == "none":
        sizes = np.repeat(80, n_clusters)
    elif params["cluster_imbalance"] == "moderate":
        sizes = np.linspace(45, 115, n_clusters).round().astype(int)
    else:
        raw = np.geomspace(25, 180, n_clusters)
        sizes = raw.round().astype(int)
    cluster = np.repeat(np.arange(n_clusters, dtype=int), sizes)
    n = int(cluster.size)
    x = rng.normal(0.0, 1.0, n)
    cluster_re = rng.normal(0.0, 0.7, n_clusters)[cluster]
    y = float(params["effect_size"]) * x + cluster_re + rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"cluster": cluster, "x": x, "y": y})
    design = sm.add_constant(df[["x"]])
    fit = sm.OLS(df["y"], design).fit()
    cov = cov_cluster(fit, df["cluster"].to_numpy())
    se_x = float(np.sqrt(np.diag(cov))[1])
    beta_x = float(fit.params["x"])
    t_obs = beta_x / se_x
    if n_clusters > 12:
        raise ValueError(
            f"exact Rademacher enumeration has {2 ** n_clusters} sign patterns "
            f"for {n_clusters} clusters"
        )
    restricted = sm.OLS(df["y"], np.ones((n, 1))).fit()
    y_hat0 = restricted.fittedvalues
    resid0 = restricted.resid
    boot_t = []
    for pattern in itertools.product([-1.0, 1.0], repeat=n_clusters):
        weights = np.array(pattern)
        y_star = y_hat0 + resid0 * weights[cluster]
        fit_star = sm.OLS(y_star, design).fit()
        cov_star = cov_cluster(fit_star, df["cluster"].to_numpy())
        se_star = float(np.sqrt(np.diag(cov_star))[1])
        if se_star > 0:
            boot_t.append(float(fit_star.params["x"] / se_star))
    p_wcb = (
        float(np.mean(np.abs(boot_t) >= abs(t_obs) - 1e-9))
        if boot_t
        else None
    )
    ref = {
        "estimand": "OLS coefficient, observed t-statistic, and wild cluster bootstrap p-value",
        "estimates": [
            {"term": "x", "estimate": beta_x, "se": se_x, "n_obs": n},
            {"term": "t_obs", "estimate": float(t_obs), "se": None, "n_obs": n},
            {"term": "p_wcb", "estimate": p_wcb, "se": None, "n_obs": n},
        ],
    }
    task_update = {
        "baseline_parameters": params,
        "prompt": (
            "Estimate y on x with an intercept and a cluster-robust standard "
            "error clustered on cluster, using the finite-sample correction "
            "G/(G-1) x (N-1)/(N-K). Also compute the restricted wild cluster "
            "bootstrap p-value for H0: beta_x = 0 with Rademacher weights: impose "
            "the null by regressing y on the intercept only, and enumerate all "
            f"2^{n_clusters} = {2 ** n_clusters} cluster sign patterns instead of "
            "random resampling. The p-value is the fraction of the bootstrap "
            "t-statistics with |t*| >= |t_obs|, counting a pattern as extreme when "
            "|t*| >= |t_obs| - 1e-9. Report rows x, t_obs, and p_wcb."
        ),
    }
    return df, ref, task_update


def simulate_var_data(
    rng: np.random.Generator,
    *,
    n: int,
    lag_order: int,
    cross_lag_strength: float,
    innovation_correlation: float,
) -> pd.DataFrame:
    burn = 100
    total = n + burn + lag_order
    y = np.zeros((total, 2))
    corr = float(innovation_correlation)
    cov = np.array([[1.0, corr], [corr, 1.0]])
    innovations = rng.multivariate_normal([0.0, 0.0], cov, size=total)
    for t in range(lag_order, total):
        own = 0.45 * y[t - 1]
        cross = np.array([cross_lag_strength * y[t - 1, 1], 0.15 * y[t - 1, 0]])
        lag_sum = np.zeros(2)
        for lag in range(2, lag_order + 1):
            lag_sum += (0.15 / lag) * y[t - lag]
        y[t] = own + cross + lag_sum + innovations[t]
    y = y[burn + lag_order :]
    return pd.DataFrame({"t": np.arange(1, n + 1, dtype=int), "y1": y[:, 0], "y2": y[:, 1]})


def generate_var_irf(variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"lag_order": 2, "cross_lag_strength": 0.2, "innovation_correlation": 0.4}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    lag_order = int(params["lag_order"])
    df = simulate_var_data(
        rng,
        n=200,
        lag_order=lag_order,
        cross_lag_strength=float(params["cross_lag_strength"]),
        innovation_correlation=float(params["innovation_correlation"]),
    )
    fit = VAR(df[["y1", "y2"]].to_numpy()).fit(lag_order)
    irf = fit.irf(periods=10).orth_irfs
    ref = {
        "estimand": "orthogonal IRF (Cholesky, y1 ordered first) of y1 response to one-SD shock in y2 at horizon 4",
        "estimates": [
            {
                "term": "irf_y1_to_y2_h4",
                "estimate": float(irf[4, 0, 1]),
                "se": None,
                "n_obs": int(len(df)),
            }
        ],
        "lag_order": lag_order,
        "identification": "Cholesky, y1 first",
    }
    task_update = {
        "baseline_parameters": params,
        "prompt": (
            f"Fit a two-variable VAR({lag_order}) with a constant. Under Cholesky "
            "identification with y1 ordered first, report the orthogonal IRF of "
            "y1 to a one-standard-deviation shock in y2 at horizon 4 as row "
            "irf_y1_to_y2_h4."
        ),
    }
    return df, ref, task_update


def generate_var_granger(
    variant: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    axis, value, seed = variant["axis"], variant["value"], int(variant["seed"])
    params = {"lag_order": 2, "cross_lag_strength": 0.2, "n": 200}
    params[axis] = value
    rng = seeded_rng(seed, variant["variant_id"])
    lag_order = int(params["lag_order"])
    df = simulate_var_data(
        rng,
        n=int(params["n"]),
        lag_order=lag_order,
        cross_lag_strength=float(params["cross_lag_strength"]),
        innovation_correlation=0.4,
    )
    y1 = df["y1"].to_numpy()
    y2 = df["y2"].to_numpy()
    idx = np.arange(lag_order, len(df))
    y_target = y1[idx]
    x_res = [y1[idx - lag] for lag in range(1, lag_order + 1)]
    x_unr = x_res + [y2[idx - lag] for lag in range(1, lag_order + 1)]
    design_res = sm.add_constant(np.column_stack(x_res))
    design_unr = sm.add_constant(np.column_stack(x_unr))
    fit_res = sm.OLS(y_target, design_res).fit()
    fit_unr = sm.OLS(y_target, design_unr).fit()
    rss_res = float(np.sum(fit_res.resid**2))
    rss_unr = float(np.sum(fit_unr.resid**2))
    q = lag_order
    k_unr = design_unr.shape[1]
    n_used = len(y_target)
    f_stat = ((rss_res - rss_unr) / q) / (rss_unr / (n_used - k_unr))
    p_value = float(1.0 - stats.f.cdf(f_stat, q, n_used - k_unr))
    ref = {
        "estimand": f"F-test of H0: y2 does not Granger-cause y1 with {lag_order} lags; OLS-based",
        "estimates": [
            {"term": "F", "estimate": float(f_stat), "se": None, "n_obs": int(n_used)},
            {"term": "df_num", "estimate": float(q), "se": None, "n_obs": int(n_used)},
            {"term": "df_den", "estimate": float(n_used - k_unr), "se": None, "n_obs": int(n_used)},
            {"term": "p_value", "estimate": p_value, "se": None, "n_obs": int(n_used)},
        ],
    }
    task_update = {
        "baseline_parameters": params,
        "prompt": (
            f"Test whether y2 Granger-causes y1 with {lag_order} lags using "
            "the aligned OLS F-test. Report rows F, df_num, df_den, and p_value."
        ),
    }
    return df, ref, task_update


GENERATORS: dict[str, Callable[[dict[str, Any]], tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]]] = {
    "ols_robust": generate_ols,
    "arima_forecast": generate_arima,
    "logit": lambda variant: generate_binary(variant, "logit"),
    "probit": lambda variant: generate_binary(variant, "probit"),
    "cluster_ols": generate_cluster,
    "two_way_cluster": generate_two_way_cluster,
    "iv2sls_robust": generate_iv,
    "poisson_offset": generate_poisson,
    "vif": lambda variant: generate_collinear(variant, cooks=False),
    "cooks_distance": lambda variant: generate_collinear(variant, cooks=True),
    "ttest_welch": generate_ttest,
    "anova_oneway": generate_anova,
    "chisq_indep": generate_chisq,
    "cox_hazard": generate_cox,
    "twoway_fe": generate_twoway_fe,
    "dynpanel_ab": generate_dynpanel,
    "staggered_did": generate_staggered_did,
    "rdd_sharp": generate_rdd,
    "lp_irf": generate_lp_irf,
    "wcb_few": generate_wcb,
    "var_irf": generate_var_irf,
    "var_granger": generate_var_granger,
}


def materialize(
    registry: dict[str, Any],
    plan: dict[str, Any],
    out_dir: Path,
    max_variants: int | None = None,
) -> dict[str, Any]:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tasks_by_id = {task["task_id"]: task for task in registry["tasks"]}
    generated_tasks = []
    references = {}
    manifest = {
        "source_registry": str(DEFAULT_REGISTRY.relative_to(PROJECT_ROOT)),
        "source_plan": str(DEFAULT_PLAN.relative_to(PROJECT_ROOT)),
        "supported_tasks": sorted(SUPPORTED_TASKS),
        "generated": [],
        "skipped": [],
    }

    variants = plan["variants"]
    if max_variants is not None:
        variants = variants[:max_variants]

    for variant in variants:
        task_id = variant["task_id"]
        if task_id not in GENERATORS:
            manifest["skipped"].append(
                {
                    "variant_id": variant["variant_id"],
                    "task_id": task_id,
                    "reason": "generator_not_implemented",
                }
            )
            continue

        promoted_id = variant_id(task_id, variant["axis"], variant["value"], int(variant["seed"]))
        generator = GENERATORS[task_id]
        try:
            df, reference, task_update = generator({**variant, "variant_id": promoted_id})
        except Exception as exc:  # keep the promotion pass informative
            manifest["skipped"].append(
                {
                    "variant_id": promoted_id,
                    "task_id": task_id,
                    "reason": f"generation_failed:{exc!r}",
                }
            )
            continue

        data_path = data_dir / f"{promoted_id}.csv"
        df.to_csv(data_path, index=False)
        data_file = portable_path(data_path)
        references[promoted_id] = reference

        task_entry = deepcopy(tasks_by_id[task_id])
        task_entry.update(task_update)
        task_entry.update(
            {
                "task_id": promoted_id,
                "title": f"{task_entry['title']} [{variant['axis']}={variant['value']}, seed={variant['seed']}]",
                "reference_key": promoted_id,
                "data_file": data_file,
                "data_description": task_update.get(
                    "data_description",
                    f"Dataset: {data_path.name}. Monte Carlo variant of {task_id}; "
                    f"axis {variant['axis']}={variant['value']}; seed {variant['seed']}.",
                ),
                "monte_carlo_parent_task": task_id,
                "monte_carlo_axis": variant["axis"],
                "monte_carlo_value": variant["value"],
                "monte_carlo_seed": int(variant["seed"]),
                "monte_carlo_status": "dataset_and_reference_generated",
            }
        )
        generated_tasks.append(task_entry)
        manifest["generated"].append(
            {
                "variant_id": promoted_id,
                "parent_task_id": task_id,
                "axis": variant["axis"],
                "value": variant["value"],
                "seed": int(variant["seed"]),
                "data_file": data_file,
                "reference_key": promoted_id,
                "n_rows": int(len(df)),
            }
        )

    generated_registry = {
        "version": "2026-05-19-monte-carlo-realizations",
        "status": "generated_supported_monte_carlo_registry",
        "source_registry": str(DEFAULT_REGISTRY.relative_to(PROJECT_ROOT)),
        "source_plan": str(DEFAULT_PLAN.relative_to(PROJECT_ROOT)),
        "notes": [
            "Contains only variants whose DGP and reference output were generated.",
            "Run with code/benchmark/harness.py --registry-json <this file> --reference-json <references file>.",
            "The original full sweep is separate and remains under data/generated/2026-05-19_full_sweep/.",
        ],
        "platforms": registry["platforms"],
        "prompt_modes": registry["prompt_modes"],
        "execution_modes": registry["execution_modes"],
        "tasks": generated_tasks,
    }
    return {
        "registry": generated_registry,
        "references": references,
        "manifest": manifest,
    }


def markdown_manifest(manifest: dict[str, Any]) -> str:
    generated = manifest["generated"]
    skipped = manifest["skipped"]
    by_task: dict[str, int] = {}
    for row in generated:
        by_task[row["parent_task_id"]] = by_task.get(row["parent_task_id"], 0) + 1

    lines = [
        "# Monte Carlo Realization Manifest",
        "",
        f"- Generated variants: **{len(generated)}**",
        f"- Skipped variants: **{len(skipped)}**",
        f"- Supported parent tasks: **{', '.join(manifest['supported_tasks'])}**",
        "",
        "## Generated by Parent Task",
        "| task | variants |",
        "|---|---:|",
    ]
    for task_id, count in sorted(by_task.items()):
        lines.append(f"| `{task_id}` | {count} |")
    lines.extend(["", "## Skipped Reasons", "| reason | count |", "|---|---:|"])
    reasons: dict[str, int] = {}
    for row in skipped:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    for reason, count in sorted(reasons.items()):
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(
        [
            "",
            "## How to Validate",
            "",
            "```bash",
            "uv run python code/benchmark/harness.py \\",
            "  --registry-json data/generated/2026-05-19_monte_carlo_realizations/task_registry_monte_carlo_supported.json \\",
            "  --reference-json data/generated/2026-05-19_monte_carlo_realizations/reference_monte_carlo_supported.json \\",
            "  --task-ids <variant_id> --n-reps 1 --model sonnet",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-variants", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = json.loads(args.registry.expanduser().read_text())
    plan = json.loads(args.plan.expanduser().read_text())
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = materialize(registry, plan, out_dir, max_variants=args.max_variants)
    (out_dir / "task_registry_monte_carlo_supported.json").write_text(
        json.dumps(json_safe(payload["registry"]), indent=2, allow_nan=False) + "\n"
    )
    (out_dir / "reference_monte_carlo_supported.json").write_text(
        json.dumps(json_safe(payload["references"]), indent=2, allow_nan=False) + "\n"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(json_safe(payload["manifest"]), indent=2, allow_nan=False) + "\n"
    )
    (out_dir / "manifest.md").write_text(markdown_manifest(payload["manifest"]))
    print(out_dir / "manifest.md")


if __name__ == "__main__":
    main()
