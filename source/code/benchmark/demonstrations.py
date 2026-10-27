"""Task-family-matched worked demonstrations for the few-shot prompt condition.

Respecified 2026-07-25 (see
``quality_reports/decision_records/2026-07-25_fewshot-respecification.md`` and
``archive/quality_reports_2026-08-10/plans/2026-07-25_fewshot-demonstrations-spec.md``). The
earlier few-shot condition injected one fixed toy OLS demonstration per
platform, identical across all tasks. This registry replaces it with worked
demonstrations of a related econometric task drawn from the same task family
as the scored task.

Rules enforced by ``code/benchmark/tests/test_demonstrations.py``:

- Family matching: every scored task receives a donor demonstration from the
  same task family (core regression, time series, diagnostics, tests, GLM,
  survival, panels/endogeneity, treatment effects).
- Exclusion: a demonstration is never the scored task's own solution and never
  mentions the scored task's dataset file, task id, or reference outputs.
- Completeness: each demonstration carries a short task statement, complete
  runnable-style code on small inline data (no external files), and the shared
  ``result.json`` schema conventions.
- First collection wave uses n = 1 demonstration per prompt; the assignment
  registry holds ordered tuples so later waves can attach more donors per task
  without changing the accessor API.

Donor library:

======================  ==============================================  =========================
donor id                concept                                         assigned scored tasks
======================  ==============================================  =========================
wls_inline              weighted least squares, classical SEs           ols_robust, cluster_ols,
                                                                        two_way_cluster, wcb_few
ar1_ols                 AR(1) by OLS plus one-step-ahead forecast       arima_forecast, lp_irf,
                                                                        var_irf, var_granger
leverage_diag           OLS hat values, maximum leverage                vif, cooks_distance,
                                                                        bp_koenker
corr_test               Pearson correlation t test                      ttest_welch, anova_oneway,
                                                                        chisq_indep
cloglog_glm             binary GLM, complementary log-log link          logit, probit,
                                                                        poisson_offset
expo_hazard             exponential proportional-hazards regression     cox_hazard
panel_fe                static one-way fixed effects (within)           twoway_fe, dynpanel_ab
wald_iv                 Wald IV estimate with a binary instrument       iv2sls_robust
did_2x2                 canonical 2x2 difference-in-differences         staggered_did, rdd_sharp,
                                                                        synth_weights_fixedv
======================  ==============================================  =========================
"""

from __future__ import annotations

DEMO_PLATFORMS: tuple[str, ...] = ("python", "r", "stata")


_WLS_INLINE = {
    "python": """\
Demonstration task: with inline data on hourly wage and years of schooling,
estimate weighted least squares of wage on educ with an intercept, using
precision weights wt. Report rows const and educ with classical WLS standard errors.

```python
import json
import pandas as pd
import statsmodels.api as sm

df = pd.DataFrame({
    "wage": [9.5, 12.1, 13.0, 15.8, 18.4, 21.0],
    "educ": [8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
    "wt": [1.0, 2.0, 1.5, 1.0, 2.5, 1.0],
})
X = sm.add_constant(df[["educ"]])
fit = sm.WLS(df["wage"], X, weights=df["wt"]).fit()
out = {
    "task_id": "demo_wls",
    "platform": "python",
    "estimates": [
        {"term": "const", "estimate": float(fit.params["const"]),
         "se": float(fit.bse["const"]), "n_obs": int(fit.nobs)},
        {"term": "educ", "estimate": float(fit.params["educ"]),
         "se": float(fit.bse["educ"]), "n_obs": int(fit.nobs)},
    ],
    "method_metadata": {"estimator": "WLS", "vcov": "classical"},
    "diagnostics": {},
}
with open("result.json", "w") as f:
    json.dump(out, f)
```""",
    "r": """\
Demonstration task: with inline data on hourly wage and years of schooling,
estimate weighted least squares of wage on educ with an intercept, using
precision weights wt. Report rows const and educ with classical WLS standard errors.

```r
library(jsonlite)
df <- data.frame(
  wage = c(9.5, 12.1, 13.0, 15.8, 18.4, 21.0),
  educ = c(8, 10, 12, 14, 16, 18),
  wt = c(1.0, 2.0, 1.5, 1.0, 2.5, 1.0)
)
fit <- lm(wage ~ educ, data = df, weights = wt)
co <- coef(summary(fit))
out <- list(
  task_id = "demo_wls",
  platform = "r",
  estimates = list(
    list(term = "const", estimate = unname(co["(Intercept)", "Estimate"]),
         se = unname(co["(Intercept)", "Std. Error"]), n_obs = nrow(df)),
    list(term = "educ", estimate = unname(co["educ", "Estimate"]),
         se = unname(co["educ", "Std. Error"]), n_obs = nrow(df))
  ),
  method_metadata = list(estimator = "WLS", vcov = "classical"),
  diagnostics = structure(list(), names = character(0))
)
write_json(out, "result.json", auto_unbox = TRUE)
```""",
    "stata": """\
Demonstration task: with inline data on hourly wage and years of schooling,
estimate weighted least squares of wage on educ with an intercept, using
precision weights wt. Report rows const and educ with classical WLS standard errors.

```stata
clear
input wage educ wt
9.5 8 1.0
12.1 10 2.0
13.0 12 1.5
15.8 14 1.0
18.4 16 2.5
21.0 18 1.0
end
regress wage educ [aweight=wt]
local b0 = string(_b[_cons], "%21.12f")
local se0 = string(_se[_cons], "%21.12f")
local b1 = string(_b[educ], "%21.12f")
local se1 = string(_se[educ], "%21.12f")
local n = string(e(N), "%12.0f")
local q = char(34)
file open fh using "result.json", write replace
file write fh "{"
file write fh `"`q'task_id`q':`q'demo_wls`q',`q'platform`q':`q'stata`q',`q'estimates`q':["'
file write fh `"{`q'term`q':`q'const`q',`q'estimate`q':`b0',`q'se`q':`se0',`q'n_obs`q':`n'},"'
file write fh `"{`q'term`q':`q'educ`q',`q'estimate`q':`b1',`q'se`q':`se1',`q'n_obs`q':`n'}],"'
file write fh `"`q'method_metadata`q':{`q'estimator`q':`q'WLS`q',`q'vcov`q':`q'classical`q'},`q'diagnostics`q':{}}"'
file close fh
```""",
}


_AR1_OLS = {
    "python": """\
Demonstration task: for a short quarterly sales series provided inline, estimate
an AR(1) regression of sales on its first lag with an intercept by OLS. Report
the lag coefficient as row ar1 and the one-step-ahead forecast as row forecast_next.

```python
import json
import pandas as pd
import statsmodels.api as sm

sales = pd.Series([102.0, 104.5, 103.8, 106.2, 108.0, 107.1,
                   109.4, 111.0, 110.2, 112.5, 114.1, 113.6])
df = pd.DataFrame({"sales": sales, "sales_lag": sales.shift(1)}).dropna()
X = sm.add_constant(df[["sales_lag"]])
fit = sm.OLS(df["sales"], X).fit()
fc = float(fit.params["const"] + fit.params["sales_lag"] * sales.iloc[-1])
out = {
    "task_id": "demo_ar1",
    "platform": "python",
    "estimates": [
        {"term": "ar1", "estimate": float(fit.params["sales_lag"]),
         "se": float(fit.bse["sales_lag"]), "n_obs": int(fit.nobs)},
        {"term": "forecast_next", "estimate": fc, "se": None,
         "n_obs": int(fit.nobs)},
    ],
    "method_metadata": {"estimator": "AR(1) by OLS", "vcov": "classical"},
    "diagnostics": {},
}
with open("result.json", "w") as f:
    json.dump(out, f)
```""",
    "r": """\
Demonstration task: for a short quarterly sales series provided inline, estimate
an AR(1) regression of sales on its first lag with an intercept by OLS. Report
the lag coefficient as row ar1 and the one-step-ahead forecast as row forecast_next.

```r
library(jsonlite)
sales <- c(102.0, 104.5, 103.8, 106.2, 108.0, 107.1,
           109.4, 111.0, 110.2, 112.5, 114.1, 113.6)
df <- data.frame(sales = sales[-1], sales_lag = sales[-length(sales)])
fit <- lm(sales ~ sales_lag, data = df)
co <- coef(summary(fit))
fc <- unname(co["(Intercept)", "Estimate"] +
               co["sales_lag", "Estimate"] * sales[length(sales)])
out <- list(
  task_id = "demo_ar1",
  platform = "r",
  estimates = list(
    list(term = "ar1", estimate = unname(co["sales_lag", "Estimate"]),
         se = unname(co["sales_lag", "Std. Error"]), n_obs = nrow(df)),
    list(term = "forecast_next", estimate = fc, se = NA, n_obs = nrow(df))
  ),
  method_metadata = list(estimator = "AR(1) by OLS", vcov = "classical"),
  diagnostics = structure(list(), names = character(0))
)
write_json(out, "result.json", auto_unbox = TRUE)
```""",
    "stata": """\
Demonstration task: for a short quarterly sales series provided inline, estimate
an AR(1) regression of sales on its first lag with an intercept by OLS. Report
the lag coefficient as row ar1 and the one-step-ahead forecast as row forecast_next.

```stata
clear
input sales
102.0
104.5
103.8
106.2
108.0
107.1
109.4
111.0
110.2
112.5
114.1
113.6
end
gen sales_lag = sales[_n-1]
regress sales sales_lag
local last = sales[_N]
local fc = _b[_cons] + _b[sales_lag]*`last'
local b1 = string(_b[sales_lag], "%21.12f")
local se1 = string(_se[sales_lag], "%21.12f")
local fcs = string(`fc', "%21.12f")
local n = string(e(N), "%12.0f")
local q = char(34)
file open fh using "result.json", write replace
file write fh "{"
file write fh `"`q'task_id`q':`q'demo_ar1`q',`q'platform`q':`q'stata`q',`q'estimates`q':["'
file write fh `"{`q'term`q':`q'ar1`q',`q'estimate`q':`b1',`q'se`q':`se1',`q'n_obs`q':`n'},"'
file write fh `"{`q'term`q':`q'forecast_next`q',`q'estimate`q':`fcs',`q'se`q':null,`q'n_obs`q':`n'}],"'
file write fh `"`q'method_metadata`q':{`q'estimator`q':`q'AR(1) by OLS`q',`q'vcov`q':`q'classical`q'},`q'diagnostics`q':{}}"'
file close fh
```""",
}


_LEVERAGE_DIAG = {
    "python": """\
Demonstration task: fit OLS of resp on w1 and w2 with an intercept on inline
data and compute the leverage (hat) value of each observation. Report the
maximum leverage as row max_leverage and its zero-based row index as row max_leverage_obs.

```python
import json
import pandas as pd
import statsmodels.api as sm

df = pd.DataFrame({
    "resp": [3.1, 4.0, 4.8, 6.2, 5.9, 7.4, 8.1, 9.0],
    "w1": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 9.0],
    "w2": [0.5, 0.7, 0.6, 1.1, 0.9, 1.3, 1.2, 1.0],
})
X = sm.add_constant(df[["w1", "w2"]])
fit = sm.OLS(df["resp"], X).fit()
hat = fit.get_influence().hat_matrix_diag
idx = int(hat.argmax())
out = {
    "task_id": "demo_leverage",
    "platform": "python",
    "estimates": [
        {"term": "max_leverage", "estimate": float(hat[idx]), "se": None,
         "n_obs": int(fit.nobs)},
        {"term": "max_leverage_obs", "estimate": float(idx), "se": None,
         "n_obs": int(fit.nobs)},
    ],
    "method_metadata": {"estimator": "OLS hat values", "vcov": "none"},
    "diagnostics": {},
}
with open("result.json", "w") as f:
    json.dump(out, f)
```""",
    "r": """\
Demonstration task: fit OLS of resp on w1 and w2 with an intercept on inline
data and compute the leverage (hat) value of each observation. Report the
maximum leverage as row max_leverage and its zero-based row index as row max_leverage_obs.

```r
library(jsonlite)
df <- data.frame(
  resp = c(3.1, 4.0, 4.8, 6.2, 5.9, 7.4, 8.1, 9.0),
  w1 = c(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 9.0),
  w2 = c(0.5, 0.7, 0.6, 1.1, 0.9, 1.3, 1.2, 1.0)
)
fit <- lm(resp ~ w1 + w2, data = df)
hat <- hatvalues(fit)
idx <- which.max(hat)
out <- list(
  task_id = "demo_leverage",
  platform = "r",
  estimates = list(
    list(term = "max_leverage", estimate = unname(hat[idx]), se = NA,
         n_obs = nrow(df)),
    list(term = "max_leverage_obs", estimate = unname(idx) - 1, se = NA,
         n_obs = nrow(df))
  ),
  method_metadata = list(estimator = "OLS hat values", vcov = "none"),
  diagnostics = structure(list(), names = character(0))
)
write_json(out, "result.json", auto_unbox = TRUE)
```""",
    "stata": """\
Demonstration task: fit OLS of resp on w1 and w2 with an intercept on inline
data and compute the leverage (hat) value of each observation. Report the
maximum leverage as row max_leverage and its zero-based row index as row max_leverage_obs.

```stata
clear
input resp w1 w2
3.1 1.0 0.5
4.0 1.5 0.7
4.8 2.0 0.6
6.2 2.5 1.1
5.9 3.0 0.9
7.4 3.5 1.3
8.1 4.0 1.2
9.0 9.0 1.0
end
regress resp w1 w2
predict hat, hat
gen obs0 = _n - 1
gsort -hat
local maxh = string(hat[1], "%21.12f")
local maxi = string(obs0[1], "%12.0f")
local n = string(e(N), "%12.0f")
local q = char(34)
file open fh using "result.json", write replace
file write fh "{"
file write fh `"`q'task_id`q':`q'demo_leverage`q',`q'platform`q':`q'stata`q',`q'estimates`q':["'
file write fh `"{`q'term`q':`q'max_leverage`q',`q'estimate`q':`maxh',`q'se`q':null,`q'n_obs`q':`n'},"'
file write fh `"{`q'term`q':`q'max_leverage_obs`q',`q'estimate`q':`maxi',`q'se`q':null,`q'n_obs`q':`n'}],"'
file write fh `"`q'method_metadata`q':{`q'estimator`q':`q'OLS hat values`q',`q'vcov`q':`q'none`q'},`q'diagnostics`q':{}}"'
file close fh
```""",
}


_CORR_TEST = {
    "python": """\
Demonstration task: with inline data on study hours and exam score, compute the
Pearson correlation and the two-sided t test of zero correlation with n - 2
degrees of freedom. Report rows r, t_stat, and p_val.

```python
import json
import numpy as np
from scipy import stats

hours = np.array([2.0, 3.5, 5.0, 6.0, 7.5, 9.0, 10.0, 12.0])
score = np.array([55.0, 60.0, 62.0, 68.0, 70.0, 74.0, 79.0, 83.0])
n = len(hours)
r, p_val = stats.pearsonr(hours, score)
t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
out = {
    "task_id": "demo_corr",
    "platform": "python",
    "estimates": [
        {"term": "r", "estimate": float(r), "se": None, "n_obs": n},
        {"term": "t_stat", "estimate": float(t_stat), "se": None, "n_obs": n},
        {"term": "p_val", "estimate": float(p_val), "se": None, "n_obs": n},
    ],
    "method_metadata": {"estimator": "Pearson correlation t test",
                        "vcov": "none"},
    "diagnostics": {},
}
with open("result.json", "w") as f:
    json.dump(out, f)
```""",
    "r": """\
Demonstration task: with inline data on study hours and exam score, compute the
Pearson correlation and the two-sided t test of zero correlation with n - 2
degrees of freedom. Report rows r, t_stat, and p_val.

```r
library(jsonlite)
hours <- c(2.0, 3.5, 5.0, 6.0, 7.5, 9.0, 10.0, 12.0)
score <- c(55.0, 60.0, 62.0, 68.0, 70.0, 74.0, 79.0, 83.0)
ct <- cor.test(hours, score)
out <- list(
  task_id = "demo_corr",
  platform = "r",
  estimates = list(
    list(term = "r", estimate = unname(ct$estimate), se = NA,
         n_obs = length(hours)),
    list(term = "t_stat", estimate = unname(ct$statistic), se = NA,
         n_obs = length(hours)),
    list(term = "p_val", estimate = ct$p.value, se = NA,
         n_obs = length(hours))
  ),
  method_metadata = list(estimator = "Pearson correlation t test",
                         vcov = "none"),
  diagnostics = structure(list(), names = character(0))
)
write_json(out, "result.json", auto_unbox = TRUE)
```""",
    "stata": """\
Demonstration task: with inline data on study hours and exam score, compute the
Pearson correlation and the two-sided t test of zero correlation with n - 2
degrees of freedom. Report rows r, t_stat, and p_val.

```stata
clear
input hours score
2.0 55
3.5 60
5.0 62
6.0 68
7.5 70
9.0 74
10.0 79
12.0 83
end
correlate hours score
local r = r(rho)
local n = r(N)
local t = `r' * sqrt(`n' - 2) / sqrt(1 - `r'^2)
local p = 2 * ttail(`n' - 2, abs(`t'))
local rs = string(`r', "%21.12f")
local ts = string(`t', "%21.12f")
local ps = string(`p', "%21.12f")
local ns = string(`n', "%12.0f")
local q = char(34)
file open fh using "result.json", write replace
file write fh "{"
file write fh `"`q'task_id`q':`q'demo_corr`q',`q'platform`q':`q'stata`q',`q'estimates`q':["'
file write fh `"{`q'term`q':`q'r`q',`q'estimate`q':`rs',`q'se`q':null,`q'n_obs`q':`ns'},"'
file write fh `"{`q'term`q':`q't_stat`q',`q'estimate`q':`ts',`q'se`q':null,`q'n_obs`q':`ns'},"'
file write fh `"{`q'term`q':`q'p_val`q',`q'estimate`q':`ps',`q'se`q':null,`q'n_obs`q':`ns'}],"'
file write fh `"`q'method_metadata`q':{`q'estimator`q':`q'Pearson correlation t test`q',`q'vcov`q':`q'none`q'},`q'diagnostics`q':{}}"'
file close fh
```""",
}


_CLOGLOG_GLM = {
    "python": """\
Demonstration task: with inline data on a binary purchase outcome and household
income, estimate a binary GLM with a complementary log-log link and an
intercept. Report coefficient-scale rows const and income with asymptotic standard errors.

```python
import json
import pandas as pd
import statsmodels.api as sm

df = pd.DataFrame({
    "purchase": [0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
    "income": [1.2, 1.8, 2.0, 2.4, 2.6, 3.0, 3.5, 4.0, 4.4, 2.2],
})
X = sm.add_constant(df[["income"]])
family = sm.families.Binomial(link=sm.families.links.CLogLog())
fit = sm.GLM(df["purchase"], X, family=family).fit()
out = {
    "task_id": "demo_cloglog",
    "platform": "python",
    "estimates": [
        {"term": "const", "estimate": float(fit.params["const"]),
         "se": float(fit.bse["const"]), "n_obs": int(fit.nobs)},
        {"term": "income", "estimate": float(fit.params["income"]),
         "se": float(fit.bse["income"]), "n_obs": int(fit.nobs)},
    ],
    "method_metadata": {"estimator": "binomial GLM cloglog link",
                        "vcov": "asymptotic"},
    "diagnostics": {},
}
with open("result.json", "w") as f:
    json.dump(out, f)
```""",
    "r": """\
Demonstration task: with inline data on a binary purchase outcome and household
income, estimate a binary GLM with a complementary log-log link and an
intercept. Report coefficient-scale rows const and income with asymptotic standard errors.

```r
library(jsonlite)
df <- data.frame(
  purchase = c(0, 0, 0, 1, 0, 1, 1, 1, 1, 0),
  income = c(1.2, 1.8, 2.0, 2.4, 2.6, 3.0, 3.5, 4.0, 4.4, 2.2)
)
fit <- glm(purchase ~ income, data = df, family = binomial(link = "cloglog"))
co <- coef(summary(fit))
out <- list(
  task_id = "demo_cloglog",
  platform = "r",
  estimates = list(
    list(term = "const", estimate = unname(co["(Intercept)", "Estimate"]),
         se = unname(co["(Intercept)", "Std. Error"]), n_obs = nrow(df)),
    list(term = "income", estimate = unname(co["income", "Estimate"]),
         se = unname(co["income", "Std. Error"]), n_obs = nrow(df))
  ),
  method_metadata = list(estimator = "binomial GLM cloglog link",
                         vcov = "asymptotic"),
  diagnostics = structure(list(), names = character(0))
)
write_json(out, "result.json", auto_unbox = TRUE)
```""",
    "stata": """\
Demonstration task: with inline data on a binary purchase outcome and household
income, estimate a binary GLM with a complementary log-log link and an
intercept. Report coefficient-scale rows const and income with asymptotic standard errors.

```stata
clear
input purchase income
0 1.2
0 1.8
0 2.0
1 2.4
0 2.6
1 3.0
1 3.5
1 4.0
1 4.4
0 2.2
end
glm purchase income, family(binomial) link(cloglog)
local b0 = string(_b[_cons], "%21.12f")
local se0 = string(_se[_cons], "%21.12f")
local b1 = string(_b[income], "%21.12f")
local se1 = string(_se[income], "%21.12f")
local n = string(e(N), "%12.0f")
local q = char(34)
file open fh using "result.json", write replace
file write fh "{"
file write fh `"`q'task_id`q':`q'demo_cloglog`q',`q'platform`q':`q'stata`q',`q'estimates`q':["'
file write fh `"{`q'term`q':`q'const`q',`q'estimate`q':`b0',`q'se`q':`se0',`q'n_obs`q':`n'},"'
file write fh `"{`q'term`q':`q'income`q',`q'estimate`q':`b1',`q'se`q':`se1',`q'n_obs`q':`n'}],"'
file write fh `"`q'method_metadata`q':{`q'estimator`q':`q'binomial GLM cloglog link`q',`q'vcov`q':`q'asymptotic`q'},`q'diagnostics`q':{}}"'
file close fh
```""",
}


_EXPO_HAZARD = {
    "python": """\
Demonstration task: with inline duration data (dur, fail) and a binary
treatment arm, fit an exponential proportional-hazards regression of the hazard
on arm. Report the log-hazard-ratio row arm with its asymptotic standard error.

```python
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.DataFrame({
    "dur": [5.0, 8.0, 3.0, 12.0, 7.0, 4.0, 10.0, 6.0, 9.0, 2.0],
    "fail": [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    "arm": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
})
X = sm.add_constant(df[["arm"]])
# The exponential proportional-hazards MLE equals a Poisson GLM of the
# failure indicator with offset log(duration).
fit = sm.GLM(df["fail"], X, family=sm.families.Poisson(),
             offset=np.log(df["dur"])).fit()
out = {
    "task_id": "demo_expsurv",
    "platform": "python",
    "estimates": [
        {"term": "arm", "estimate": float(fit.params["arm"]),
         "se": float(fit.bse["arm"]), "n_obs": int(fit.nobs)},
    ],
    "method_metadata": {"estimator": "exponential proportional hazards",
                        "vcov": "asymptotic"},
    "diagnostics": {},
}
with open("result.json", "w") as f:
    json.dump(out, f)
```""",
    "r": """\
Demonstration task: with inline duration data (dur, fail) and a binary
treatment arm, fit an exponential proportional-hazards regression of the hazard
on arm. Report the log-hazard-ratio row arm with its asymptotic standard error.

```r
library(jsonlite)
library(survival)
df <- data.frame(
  dur = c(5, 8, 3, 12, 7, 4, 10, 6, 9, 2),
  fail = c(1, 1, 0, 1, 1, 1, 0, 1, 1, 1),
  arm = c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
)
fit <- survreg(Surv(dur, fail) ~ arm, data = df, dist = "exponential")
# survreg fits the AFT parameterization; for the exponential distribution the
# log hazard ratio is the negative of the AFT coefficient, same standard error.
loghr <- -unname(coef(fit)["arm"])
se_arm <- unname(sqrt(diag(vcov(fit)))["arm"])
out <- list(
  task_id = "demo_expsurv",
  platform = "r",
  estimates = list(
    list(term = "arm", estimate = loghr, se = se_arm, n_obs = nrow(df))
  ),
  method_metadata = list(estimator = "exponential proportional hazards",
                         vcov = "asymptotic"),
  diagnostics = structure(list(), names = character(0))
)
write_json(out, "result.json", auto_unbox = TRUE)
```""",
    "stata": """\
Demonstration task: with inline duration data (dur, fail) and a binary
treatment arm, fit an exponential proportional-hazards regression of the hazard
on arm. Report the log-hazard-ratio row arm with its asymptotic standard error.

```stata
clear
input dur fail arm
5 1 0
8 1 0
3 0 0
12 1 0
7 1 0
4 1 1
10 0 1
6 1 1
9 1 1
2 1 1
end
stset dur, failure(fail)
streg arm, distribution(exponential) nohr
local b = string(_b[arm], "%21.12f")
local se = string(_se[arm], "%21.12f")
local n = string(e(N), "%12.0f")
local q = char(34)
file open fh using "result.json", write replace
file write fh "{"
file write fh `"`q'task_id`q':`q'demo_expsurv`q',`q'platform`q':`q'stata`q',`q'estimates`q':["'
file write fh `"{`q'term`q':`q'arm`q',`q'estimate`q':`b',`q'se`q':`se',`q'n_obs`q':`n'}],"'
file write fh `"`q'method_metadata`q':{`q'estimator`q':`q'exponential proportional hazards`q',`q'vcov`q':`q'asymptotic`q'},`q'diagnostics`q':{}}"'
file close fh
```""",
}


_PANEL_FE = {
    "python": """\
Demonstration task: with an inline balanced panel of four plants over four
periods, estimate the within (fixed effects) regression of output q on input k
with plant fixed effects. Report row k with its conventional within standard error.

```python
import json
import pandas as pd
import statsmodels.formula.api as smf

df = pd.DataFrame({
    "plant": [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
    "yr": [1, 2, 3, 4] * 4,
    "k": [1.0, 1.2, 1.4, 1.6, 2.0, 2.1, 2.3, 2.6,
          1.5, 1.7, 1.6, 1.9, 2.5, 2.8, 3.0, 3.2],
    "q": [2.1, 2.3, 2.2, 2.5, 3.4, 3.6, 3.5, 3.9,
          4.0, 4.2, 4.1, 4.4, 5.6, 5.9, 6.0, 6.3],
})
fit = smf.ols("q ~ k + C(plant)", data=df).fit()
out = {
    "task_id": "demo_panelfe",
    "platform": "python",
    "estimates": [
        {"term": "k", "estimate": float(fit.params["k"]),
         "se": float(fit.bse["k"]), "n_obs": int(fit.nobs)},
    ],
    "method_metadata": {"estimator": "one-way FE (LSDV)",
                        "vcov": "classical"},
    "diagnostics": {},
}
with open("result.json", "w") as f:
    json.dump(out, f)
```""",
    "r": """\
Demonstration task: with an inline balanced panel of four plants over four
periods, estimate the within (fixed effects) regression of output q on input k
with plant fixed effects. Report row k with its conventional within standard error.

```r
library(jsonlite)
library(plm)
df <- data.frame(
  plant = rep(1:4, each = 4),
  yr = rep(1:4, times = 4),
  k = c(1.0, 1.2, 1.4, 1.6, 2.0, 2.1, 2.3, 2.6,
        1.5, 1.7, 1.6, 1.9, 2.5, 2.8, 3.0, 3.2),
  q = c(2.1, 2.3, 2.2, 2.5, 3.4, 3.6, 3.5, 3.9,
        4.0, 4.2, 4.1, 4.4, 5.6, 5.9, 6.0, 6.3)
)
pdf <- pdata.frame(df, index = c("plant", "yr"))
fit <- plm(q ~ k, data = pdf, model = "within")
co <- coef(summary(fit))
out <- list(
  task_id = "demo_panelfe",
  platform = "r",
  estimates = list(
    list(term = "k", estimate = unname(co["k", "Estimate"]),
         se = unname(co["k", "Std. Error"]), n_obs = nrow(df))
  ),
  method_metadata = list(estimator = "one-way FE (within)",
                         vcov = "classical"),
  diagnostics = structure(list(), names = character(0))
)
write_json(out, "result.json", auto_unbox = TRUE)
```""",
    "stata": """\
Demonstration task: with an inline balanced panel of four plants over four
periods, estimate the within (fixed effects) regression of output q on input k
with plant fixed effects. Report row k with its conventional within standard error.

```stata
clear
input plant yr k q
1 1 1.0 2.1
1 2 1.2 2.3
1 3 1.4 2.2
1 4 1.6 2.5
2 1 2.0 3.4
2 2 2.1 3.6
2 3 2.3 3.5
2 4 2.6 3.9
3 1 1.5 4.0
3 2 1.7 4.2
3 3 1.6 4.1
3 4 1.9 4.4
4 1 2.5 5.6
4 2 2.8 5.9
4 3 3.0 6.0
4 4 3.2 6.3
end
xtset plant yr
xtreg q k, fe
local b = string(_b[k], "%21.12f")
local se = string(_se[k], "%21.12f")
local n = string(e(N), "%12.0f")
local q = char(34)
file open fh using "result.json", write replace
file write fh "{"
file write fh `"`q'task_id`q':`q'demo_panelfe`q',`q'platform`q':`q'stata`q',`q'estimates`q':["'
file write fh `"{`q'term`q':`q'k`q',`q'estimate`q':`b',`q'se`q':`se',`q'n_obs`q':`n'}],"'
file write fh `"`q'method_metadata`q':{`q'estimator`q':`q'one-way FE (within)`q',`q'vcov`q':`q'classical`q'},`q'diagnostics`q':{}}"'
file close fh
```""",
}


_WALD_IV = {
    "python": """\
Demonstration task: with inline data on earnings, program attendance, and a
randomized binary encouragement, compute the Wald instrumental-variables
estimate as the reduced-form difference over the first-stage difference.
Report rows attend_effect and first_stage with se null.

```python
import json
import pandas as pd

df = pd.DataFrame({
    "encour": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    "attend": [0, 0, 1, 0, 0, 1, 1, 0, 1, 1],
    "earn": [10.0, 11.5, 14.0, 10.5, 12.0, 16.0, 15.5, 12.5, 17.0, 16.5],
})
g1 = df[df["encour"] == 1]
g0 = df[df["encour"] == 0]
reduced_form = g1["earn"].mean() - g0["earn"].mean()
first_stage = g1["attend"].mean() - g0["attend"].mean()
wald = reduced_form / first_stage
out = {
    "task_id": "demo_wald",
    "platform": "python",
    "estimates": [
        {"term": "attend_effect", "estimate": float(wald), "se": None,
         "n_obs": int(len(df))},
        {"term": "first_stage", "estimate": float(first_stage), "se": None,
         "n_obs": int(len(df))},
    ],
    "method_metadata": {"estimator": "Wald IV from group means",
                        "vcov": "none"},
    "diagnostics": {},
}
with open("result.json", "w") as f:
    json.dump(out, f)
```""",
    "r": """\
Demonstration task: with inline data on earnings, program attendance, and a
randomized binary encouragement, compute the Wald instrumental-variables
estimate as the reduced-form difference over the first-stage difference.
Report rows attend_effect and first_stage with se null.

```r
library(jsonlite)
df <- data.frame(
  encour = c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1),
  attend = c(0, 0, 1, 0, 0, 1, 1, 0, 1, 1),
  earn = c(10.0, 11.5, 14.0, 10.5, 12.0, 16.0, 15.5, 12.5, 17.0, 16.5)
)
reduced_form <- mean(df$earn[df$encour == 1]) - mean(df$earn[df$encour == 0])
first_stage <- mean(df$attend[df$encour == 1]) - mean(df$attend[df$encour == 0])
wald <- reduced_form / first_stage
out <- list(
  task_id = "demo_wald",
  platform = "r",
  estimates = list(
    list(term = "attend_effect", estimate = wald, se = NA, n_obs = nrow(df)),
    list(term = "first_stage", estimate = first_stage, se = NA,
         n_obs = nrow(df))
  ),
  method_metadata = list(estimator = "Wald IV from group means",
                         vcov = "none"),
  diagnostics = structure(list(), names = character(0))
)
write_json(out, "result.json", auto_unbox = TRUE)
```""",
    "stata": """\
Demonstration task: with inline data on earnings, program attendance, and a
randomized binary encouragement, compute the Wald instrumental-variables
estimate as the reduced-form difference over the first-stage difference.
Report rows attend_effect and first_stage with se null.

```stata
clear
input encour attend earn
0 0 10.0
0 0 11.5
0 1 14.0
0 0 10.5
0 0 12.0
1 1 16.0
1 1 15.5
1 0 12.5
1 1 17.0
1 1 16.5
end
sum earn if encour == 1, meanonly
local y1 = r(mean)
sum earn if encour == 0, meanonly
local y0 = r(mean)
sum attend if encour == 1, meanonly
local a1 = r(mean)
sum attend if encour == 0, meanonly
local a0 = r(mean)
local first = `a1' - `a0'
local wald = (`y1' - `y0') / `first'
local ws = string(`wald', "%21.12f")
local fss = string(`first', "%21.12f")
local ns = string(_N, "%12.0f")
local q = char(34)
file open fh using "result.json", write replace
file write fh "{"
file write fh `"`q'task_id`q':`q'demo_wald`q',`q'platform`q':`q'stata`q',`q'estimates`q':["'
file write fh `"{`q'term`q':`q'attend_effect`q',`q'estimate`q':`ws',`q'se`q':null,`q'n_obs`q':`ns'},"'
file write fh `"{`q'term`q':`q'first_stage`q',`q'estimate`q':`fss',`q'se`q':null,`q'n_obs`q':`ns'}],"'
file write fh `"`q'method_metadata`q':{`q'estimator`q':`q'Wald IV from group means`q',`q'vcov`q':`q'none`q'},`q'diagnostics`q':{}}"'
file close fh
```""",
}


_DID_2X2 = {
    "python": """\
Demonstration task: with an inline two-period dataset of treated and control
stores, compute the canonical 2x2 difference-in-differences estimate from the
four group-by-period means of sales. Report row did with se null.

```python
import json
import pandas as pd

df = pd.DataFrame({
    "store": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    "treated": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "post": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "sales": [20.0, 26.0, 22.0, 27.5, 21.0, 26.5,
              19.5, 21.0, 20.5, 22.5, 21.5, 23.0],
})
means = df.groupby(["treated", "post"])["sales"].mean()
did = (means[1, 1] - means[1, 0]) - (means[0, 1] - means[0, 0])
out = {
    "task_id": "demo_did",
    "platform": "python",
    "estimates": [
        {"term": "did", "estimate": float(did), "se": None,
         "n_obs": int(len(df))},
    ],
    "method_metadata": {"estimator": "2x2 DiD from cell means",
                        "vcov": "none"},
    "diagnostics": {},
}
with open("result.json", "w") as f:
    json.dump(out, f)
```""",
    "r": """\
Demonstration task: with an inline two-period dataset of treated and control
stores, compute the canonical 2x2 difference-in-differences estimate from the
four group-by-period means of sales. Report row did with se null.

```r
library(jsonlite)
df <- data.frame(
  store = c(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6),
  treated = c(1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
  post = c(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
  sales = c(20.0, 26.0, 22.0, 27.5, 21.0, 26.5,
            19.5, 21.0, 20.5, 22.5, 21.5, 23.0)
)
m11 <- mean(df$sales[df$treated == 1 & df$post == 1])
m10 <- mean(df$sales[df$treated == 1 & df$post == 0])
m01 <- mean(df$sales[df$treated == 0 & df$post == 1])
m00 <- mean(df$sales[df$treated == 0 & df$post == 0])
did <- (m11 - m10) - (m01 - m00)
out <- list(
  task_id = "demo_did",
  platform = "r",
  estimates = list(
    list(term = "did", estimate = did, se = NA, n_obs = nrow(df))
  ),
  method_metadata = list(estimator = "2x2 DiD from cell means",
                         vcov = "none"),
  diagnostics = structure(list(), names = character(0))
)
write_json(out, "result.json", auto_unbox = TRUE)
```""",
    "stata": """\
Demonstration task: with an inline two-period dataset of treated and control
stores, compute the canonical 2x2 difference-in-differences estimate from the
four group-by-period means of sales. Report row did with se null.

```stata
clear
input store treated post sales
1 1 0 20.0
1 1 1 26.0
2 1 0 22.0
2 1 1 27.5
3 1 0 21.0
3 1 1 26.5
4 0 0 19.5
4 0 1 21.0
5 0 0 20.5
5 0 1 22.5
6 0 0 21.5
6 0 1 23.0
end
sum sales if treated == 1 & post == 1, meanonly
local m11 = r(mean)
sum sales if treated == 1 & post == 0, meanonly
local m10 = r(mean)
sum sales if treated == 0 & post == 1, meanonly
local m01 = r(mean)
sum sales if treated == 0 & post == 0, meanonly
local m00 = r(mean)
local did = (`m11' - `m10') - (`m01' - `m00')
local ds = string(`did', "%21.12f")
local ns = string(_N, "%12.0f")
local q = char(34)
file open fh using "result.json", write replace
file write fh "{"
file write fh `"`q'task_id`q':`q'demo_did`q',`q'platform`q':`q'stata`q',`q'estimates`q':["'
file write fh `"{`q'term`q':`q'did`q',`q'estimate`q':`ds',`q'se`q':null,`q'n_obs`q':`ns'}],"'
file write fh `"`q'method_metadata`q':{`q'estimator`q':`q'2x2 DiD from cell means`q',`q'vcov`q':`q'none`q'},`q'diagnostics`q':{}}"'
file close fh
```""",
}


DEMO_LIBRARY: dict[str, dict[str, str]] = {
    "wls_inline": _WLS_INLINE,
    "ar1_ols": _AR1_OLS,
    "leverage_diag": _LEVERAGE_DIAG,
    "corr_test": _CORR_TEST,
    "cloglog_glm": _CLOGLOG_GLM,
    "expo_hazard": _EXPO_HAZARD,
    "panel_fe": _PANEL_FE,
    "wald_iv": _WALD_IV,
    "did_2x2": _DID_2X2,
}


# Ordered donor tuples per scored task. The first collection wave uses n = 1
# demonstration per prompt; append donor ids to a tuple to raise n for a later
# wave (the joined text changes and prompt_sha256 stamps the new wave).
DEMO_ASSIGNMENTS: dict[str, tuple[str, ...]] = {
    # Core regression and cluster inference
    "ols_robust": ("wls_inline",),
    "cluster_ols": ("wls_inline",),
    "two_way_cluster": ("wls_inline",),
    "wcb_few": ("wls_inline",),
    # Time series
    "arima_forecast": ("ar1_ols",),
    "lp_irf": ("ar1_ols",),
    "var_irf": ("ar1_ols",),
    "var_granger": ("ar1_ols",),
    # Regression diagnostics
    "vif": ("leverage_diag",),
    "cooks_distance": ("leverage_diag",),
    # Classical tests
    "ttest_welch": ("corr_test",),
    "anova_oneway": ("corr_test",),
    "chisq_indep": ("corr_test",),
    # GLM
    "logit": ("cloglog_glm",),
    "probit": ("cloglog_glm",),
    "poisson_offset": ("cloglog_glm",),
    # Survival
    "cox_hazard": ("expo_hazard",),
    # Panels and endogeneity
    "twoway_fe": ("panel_fe",),
    "dynpanel_ab": ("panel_fe",),
    "iv2sls_robust": ("wald_iv",),
    # Treatment effects
    "staggered_did": ("did_2x2",),
    "rdd_sharp": ("did_2x2",),
    # Exploratory tasks (T23, T24), covered so a few-shot cell cannot crash
    "bp_koenker": ("leverage_diag",),
    "synth_weights_fixedv": ("did_2x2",),
}


DEMONSTRATIONS: dict[str, dict[str, str]] = {
    task_id: {
        platform: "\n\n".join(DEMO_LIBRARY[donor][platform] for donor in donors)
        for platform in DEMO_PLATFORMS
    }
    for task_id, donors in DEMO_ASSIGNMENTS.items()
}


def resolve_demo_task_id(task_id: str) -> str | None:
    """Map a task id (or a Monte Carlo variant id) to its demonstration key.

    Monte Carlo variant ids follow ``<parent>__mc__<axis>__<seed>``; the parent
    task's demonstration applies to every variant. Returns None when no
    demonstration is registered.
    """
    if task_id in DEMONSTRATIONS:
        return task_id
    parent = task_id.split("__mc__", 1)[0]
    if parent in DEMONSTRATIONS:
        return parent
    return None


def demo_for(task_id: str, platform: str) -> str:
    """The joined demonstration text for one scored task on one platform."""
    if platform not in DEMO_PLATFORMS:
        raise ValueError(f"unknown platform: {platform!r}")
    resolved = resolve_demo_task_id(task_id)
    if resolved is None:
        raise KeyError(
            f"no demonstration registered for task_id {task_id!r}; "
            "add it to demonstrations.DEMO_ASSIGNMENTS before running a "
            "few-shot cell"
        )
    return DEMONSTRATIONS[resolved][platform]
