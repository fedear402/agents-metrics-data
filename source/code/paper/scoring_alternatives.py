"""Wrong-method alternatives of the scored values, and the check that uses them (PNAS analysis only).

Rule (decision record quality_reports/decision_records/2026-10-29_wrong-method-alternative-check.md):
a scored value passes only if it is inside its tolerance and it is not closer to a listed
alternative of its term than to the nearest accepted value of that term. The accepted values
are the stored reference and the accepted conventions of data/analysis/rescore_v2/scoring_spec.json.
An alternative is tested only when it and every accepted value differ after rounding to the
decimals of the reported value. A correct value that a run rounds therefore never fails.

The alternatives come from known wrong methods or conventions that the task statement excludes.
build() computes them once from the task data and writes data/analysis/six_models/data/
scoring_alternatives.json. For every task it also recomputes the stored reference with the same
code and stops when the recomputation differs. code/paper/pnas_analysis.py calls apply_check()
after the raw scoring. The two-model analysis does not use this module.

Run from the repository root:

    uv run --no-sync python code/paper/scoring_alternatives.py
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'data/analysis/six_models/data/catalog.json'
SPEC = ROOT / 'data/analysis/rescore_v2/scoring_spec.json'
OUT = ROOT / 'data/analysis/six_models/data/scoring_alternatives.json'
# A recomputed reference must equal the stored value to this relative difference.
REFERENCE_MATCH = 1e-6
# An alternative closer than this to an accepted value is that accepted value. It is not listed.
SAME_VALUE = 1e-6


# ---------------------------------------------------------------- shared algebra
def data_path(task: dict) -> Path:
    """The task input file. Catalog paths are absolute on the build machine; elsewhere they resolve from data/."""
    path = Path(task['data_path'])
    if path.is_file():
        return path
    return ROOT.joinpath(*path.parts[path.parts.index('data'):])


def ols(X, y, w=None):
    """Coefficients, residuals, and the bread (X'WX)^-1 of (weighted) least squares."""
    W = np.ones(len(y)) if w is None else w
    bread = np.linalg.inv((X * W[:, None]).T @ X)
    b = bread @ (X * W[:, None]).T @ y
    return b, y - X @ b, bread


def sandwich(X, e, bread, w=None, scale=None):
    """Heteroskedasticity-robust covariance with optional observation scale factors on e^2."""
    W = np.ones(len(e)) if w is None else w
    s = e ** 2 * (1.0 if scale is None else scale)
    meat = (X * (W ** 2 * s)[:, None]).T @ X
    return bread @ meat @ bread


def hc_family(X, e, bread, leverage_X=None):
    """HC0 to HC3, the classical covariance, and the HC1 factor n/(n-k) applied on top of HC1, HC2, and HC3."""
    n, k = X.shape
    L = X if leverage_X is None else leverage_X
    h = np.einsum('ij,jk,ik->i', L, bread, L)
    v0 = sandwich(X, e, bread)
    v = dict(HC0=v0, HC1=v0 * n / (n - k), HC2=sandwich(X, e, bread, scale=1 / (1 - h)),
             HC3=sandwich(X, e, bread, scale=1 / (1 - h) ** 2), classical=bread * (e @ e) / (n - k))
    for name in ('HC1', 'HC2', 'HC3'):
        v[name + '_times_hc1_factor'] = v[name] * n / (n - k)
    return v


HC_NAMED_HC1 = [
    ('HC0', 'sandwich without the n/(n-k) factor'),
    ('HC2', 'sandwich with e^2/(1-h)'),
    ('HC3', 'sandwich with e^2/(1-h)^2'),
    ('HC1_times_hc1_factor', 'n/(n-k) applied on top of HC1 (the factor twice)'),
    ('HC2_times_hc1_factor', 'HC2 multiplied by n/(n-k)'),
    ('HC3_times_hc1_factor', 'HC3 multiplied by n/(n-k)'),
    ('classical', 'homoskedastic standard error')]


def cluster_meat(X, e, groups):
    scores = pd.DataFrame(X * e[:, None]).groupby(np.asarray(groups)).sum().to_numpy()
    return scores.T @ scores, len(scores)


def cluster_cov(X, e, bread, groups, k=None, correction=True):
    """One-way cluster covariance. CR1 uses G/(G-1)*(N-1)/(N-K)."""
    meat, g = cluster_meat(X, e, groups)
    n = len(e)
    k = X.shape[1] if k is None else k
    factor = g / (g - 1) * (n - 1) / (n - k) if correction else 1.0
    return factor * bread @ meat @ bread


def newey_west(X, e, bread, lags, weight_lags=None, correction=False):
    """Newey-West HAC covariance. Bartlett weights 1-j/(weight_lags+1); weight_lags defaults to lags."""
    n, k = X.shape
    u = X * e[:, None]
    meat = u.T @ u
    m = lags if weight_lags is None else weight_lags
    for j in range(1, lags + 1):
        gamma = u[j:].T @ u[:-j]
        meat += (1 - j / (m + 1)) * (gamma + gamma.T)
    v = bread @ meat @ bread
    return v * n / (n - k) if correction else v


def se(v, i):
    return float(math.sqrt(v[i, i]))


def probit_fit(X, y, score=None, start=None):
    """Probit by maximum likelihood (score=None) or the root of a supplied estimating equation."""
    start = np.zeros(X.shape[1]) if start is None else start
    if score is not None:
        return optimize.root(lambda b: score(b), start, method='hybr', tol=1e-14).x

    def nll(b):
        q = 2 * y - 1
        return -stats.norm.logcdf(q * (X @ b)).sum()

    def grad(b):
        q = 2 * y - 1
        idx = q * (X @ b)
        return -X.T @ (q * np.exp(stats.norm.logpdf(idx) - stats.norm.logcdf(idx)))
    return optimize.minimize(nll, start, jac=grad, method='BFGS', options=dict(gtol=1e-12, maxiter=10000)).x


def entry(term, kind, method, value, how):
    return dict(term=term, kind=kind, method=method, value=float(value), how=how)


# ---------------------------------------------------------------- tasks with a scored standard error
def ols_robust(task, df):
    X = np.column_stack([np.ones(len(df)), df.x1, df.x2])
    b, e, bread = ols(X, df.y_lin.to_numpy())
    v = hc_family(X, e, bread)
    terms = ['const', 'x1', 'x2']
    ref = {(t, 'estimate'): b[i] for i, t in enumerate(terms)} | {(t, 'se'): se(v['HC1'], i) for i, t in enumerate(terms)}
    alts = [entry(t, 'se', m, se(v[m], i), 'OLS ' + how) for i, t in enumerate(terms) for m, how in HC_NAMED_HC1]
    return ref, alts, 'The task names HC1.'


def iv2sls_robust(task, df):
    y = df.y_iv.to_numpy()
    X = np.column_stack([np.ones(len(df)), df.x1_iv])
    Z = np.column_stack([np.ones(len(df)), df.z])
    n, k = X.shape
    Xh = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)
    A = np.linalg.inv(Xh.T @ X)
    b = A @ Xh.T @ y
    e = y - X @ b
    # 2SLS sandwich: projected regressors, structural residuals, leverage of the projected regressors.
    v = hc_family(Xh, e, A)
    # Manual second stage: OLS of y on the fitted regressors, with that regression's residuals.
    b2, u, A2 = ols(Xh, y)
    v['two_step_HC1'] = sandwich(Xh, u, A2) * n / (n - k)
    bo, _, _ = ols(X, y)
    ref = {('const', 'estimate'): b[0], ('x1_iv', 'estimate'): b[1], ('x1_iv', 'se'): se(v['HC1'], 1)}
    alts = [entry('x1_iv', 'se', m, se(v[m], 1), '2SLS ' + how + (
        ' (ivregress without small, linearmodels robust without debiased)' if m == 'HC0' else ''))
        for m, how in HC_NAMED_HC1]
    alts.append(entry('x1_iv', 'se', 'two_step_HC1', se(v['two_step_HC1'], 1),
                      'HC1 of a manual second-stage OLS on the fitted regressor'))
    alts += [entry(t, 'estimate', 'OLS', bo[i], 'OLS of y_iv on x1_iv without the instrument')
             for i, t in enumerate(['const', 'x1_iv'])]
    return ref, alts, 'The task names HC1 and a proper 2SLS routine.'


def cluster_ols(task, df):
    X = np.column_stack([np.ones(len(df)), df.x1, df.x2])
    b, e, bread = ols(X, df.y_panel.to_numpy())
    terms = ['const', 'x1', 'x2']
    v1 = cluster_cov(X, e, bread, df.cluster)
    v0 = cluster_cov(X, e, bread, df.cluster, correction=False)
    hc = hc_family(X, e, bread)
    ref = {(t, 'estimate'): b[i] for i, t in enumerate(terms)} | {(t, 'se'): se(v1, i) for i, t in enumerate(terms)}
    alts = [entry(t, 'se', m, se(vv, i), how) for i, t in enumerate(terms) for m, vv, how in [
        ('CR0', v0, 'cluster sandwich without small-sample factors'),
        ('HC1', hc['HC1'], 'heteroskedasticity-robust standard error without clustering'),
        ('classical', hc['classical'], 'homoskedastic OLS standard error')]]
    return ref, alts, 'The task asks for cluster-robust standard errors on cluster.'


def two_way_cluster(task, df):
    X = np.column_stack([np.ones(len(df)), df.x])
    b, e, bread = ols(X, df.y_twocluster.to_numpy())
    cell = df.firm * 100 + df.year
    vf, vy, vi = (cluster_cov(X, e, bread, g) for g in (df.firm, df.year, cell))
    vf0, vy0, vi0 = (cluster_cov(X, e, bread, g, correction=False) for g in (df.firm, df.year, cell))
    hc = hc_family(X, e, bread)
    terms = ['const', 'x']
    ref = {(t, 'estimate'): b[i] for i, t in enumerate(terms)} | {(t, 'se'): se(vf + vy - vi, i) for i, t in enumerate(terms)}
    alts = [entry(t, 'se', m, se(vv, i), how) for i, t in enumerate(terms) for m, vv, how in [
        ('CR0', vf0 + vy0 - vi0, 'Cameron-Gelbach-Miller without small-sample factors'),
        ('firm_only', vf, 'one-way CR1 clustered on firm'),
        ('year_only', vy, 'one-way CR1 clustered on year'),
        ('no_intersection', vf + vy, 'firm plus year covariance without subtracting the intersection term'),
        ('HC1', hc['HC1'], 'heteroskedasticity-robust standard error without clustering')]]
    return ref, alts, 'The task asks for two-way clustering on firm and year (Cameron-Gelbach-Miller).'


def twoway_fe(task, df):
    y = df.y_twocluster.to_numpy()
    firm = pd.get_dummies(df.firm, drop_first=True).to_numpy(float)
    year = pd.get_dummies(df.year, drop_first=True).to_numpy(float)
    X = np.column_stack([df.x, np.ones(len(df)), firm, year])
    b, e, bread = ols(X, y)
    v211 = cluster_cov(X, e, bread, df.firm, k=211)
    v0 = cluster_cov(X, e, bread, df.firm, correction=False)
    vyear = cluster_cov(X, e, bread, df.year, k=211)
    hc1 = sandwich(X, e, bread) * len(y) / (len(y) - 211)
    # The design of the hand-read run twoway_fe__luna6__python__zero_shot__agent__2: firm and year
    # dummies with one base category each and no intercept, so the base firm-year level is forced to zero.
    Xn = np.column_stack([df.x, firm, year])
    bn, en, breadn = ols(Xn, y)
    vn = cluster_cov(Xn, en, breadn, df.firm)
    ref = {('x', 'estimate'): b[0], ('x', 'se'): se(v211, 0)}
    alts = [entry('x', 'se', 'cluster_year', se(vyear, 0), 'CR1 clustered on year'),
            entry('x', 'se', 'HC1', se(hc1, 0), 'heteroskedasticity-robust standard error without clustering'),
            entry('x', 'estimate', 'fe_without_intercept', bn[0],
                  'firm and year dummies with one base category each and no intercept'),
            entry('x', 'se', 'fe_without_intercept', se(vn, 0),
                  'CR1 standard error of that design (statsmodels use_correction=True)')]
    notes = ('The task asks for firm and year fixed effects and a firm-clustered standard error. It names no '
             f'small-sample correction, so CR0 ({se(v0, 0):.7f}) is not listed: the scoring accepts CR0 in '
             'cluster_ols and two_way_cluster.')
    return ref, alts, notes


def lp_irf(task, df):
    df = df.sort_values('t').reset_index(drop=True)
    y, shock, h = df.y.to_numpy(), df.shock.to_numpy(), 4
    T = len(df)
    X = np.column_stack([np.ones(T - h - 1), shock[1:T - h], y[:T - h - 1]])
    b, e, bread = ols(X, y[h + 1:])
    hc = hc_family(X, e, bread)
    ref = {('irf_h4', 'estimate'): b[1], ('irf_h4', 'se'): se(newey_west(X, e, bread, 4), 1)}
    alts = []
    for lags in (3, 5):
        for corr, label in ((False, ''), (True, ', with n/(n-k)')):
            # Lag 4 with Bartlett weights 1-j/4 gives the lag-3 value, because the weight of lag 4 is zero.
            extra = ' (also lag 4 with weights 1-j/4)' if lags == 3 else ''
            alts.append(entry('irf_h4', 'se', f'nw_lag{lags}' + ('_dfk' if corr else ''),
                              se(newey_west(X, e, bread, lags, correction=corr), 1),
                              f'Newey-West with lag {lags}{label}{extra}'))
    alts += [entry('irf_h4', 'se', 'HC1', se(hc['HC1'], 1), 'heteroskedasticity-robust standard error without HAC'),
             entry('irf_h4', 'se', 'classical', se(hc['classical'], 1), 'homoskedastic OLS standard error')]
    return ref, alts, 'The task names Newey-West with lag 4.'


def applied_cumulative_hac(task, df):
    df = df.sort_values('date').reset_index(drop=True)
    dlv = np.log(df.vix).diff().to_numpy()
    lags = np.column_stack([np.roll(dlv, j) for j in range(1, 16)])
    keep = np.arange(len(df)) >= 16
    X = np.column_stack([np.ones(keep.sum()), lags[keep]])
    y = np.log(df.sales.to_numpy())[keep]
    b, e, bread = ols(X, y)
    n = len(y)
    v = newey_west(X, e, bread, 15) * n / (n - 16)
    c = np.r_[0, np.ones(15)]
    ref = {('cumulative_response', 'estimate'): c @ b, ('lag_1', 'estimate'): b[1], ('lag_15', 'estimate'): b[15],
           ('cumulative_response', 'se'): math.sqrt(c @ v @ c)}
    vw = newey_west(X, e, bread, 15, weight_lags=14) * n / (n - 16)
    alts = [entry('cumulative_response', 'se', 'no_multiplier', math.sqrt(c @ v @ c * (n - 16) / n),
                  'Newey-West lag 15 without the N/(N-16) multiplier'),
            entry('cumulative_response', 'se', 'multiplier_twice', math.sqrt(c @ v @ c * n / (n - 16)),
                  'N/(N-16) applied on top of a covariance that already includes it'),
            entry('cumulative_response', 'se', 'weights_1_minus_j_over_15', math.sqrt(c @ vw @ c),
                  'Bartlett weights 1-j/15 in place of 1-j/16')]
    return ref, alts, 'The task names Newey-West lag 15, weights 1-j/16, and N/(N-16).'


def applied_fuzzy_rd(task, df):
    d = df[df.score.abs() < 1]
    s = d.score.to_numpy()
    right = (s >= 0).astype(float)
    w = 1 - np.abs(s)
    X = np.column_stack([np.ones(len(d)), right, s, right * s])
    n, k = X.shape
    out, res = {}, {}
    for name, col in (('reduced_form', 'earnings'), ('first_stage', 'participated')):
        b, e, bread = ols(X, d[col].to_numpy(float), w)
        out[name], res[name] = b[1], e
    _, _, bread = ols(X, d.earnings.to_numpy(float), w)
    hmat = np.einsum('ij,jk,ik->i', X * w[:, None], bread, X)

    def joint(scale):
        ee = {(a, c): bread @ (X * (w ** 2 * res[a] * res[c] * scale)[:, None]).T @ X @ bread
              for a in res for c in res}
        return ee[('reduced_form', 'reduced_form')][1, 1], ee[('first_stage', 'first_stage')][1, 1], \
            ee[('reduced_form', 'first_stage')][1, 1]

    rf, fs = out['reduced_form'], out['first_stage']

    def delta(v, cov=True):
        vr, vf, c = v
        return math.sqrt(vr / fs ** 2 + rf ** 2 * vf / fs ** 4 - (2 * rf * c / fs ** 3 if cov else 0))
    v0 = joint(1.0)
    ref = {('reduced_form', 'estimate'): rf, ('first_stage', 'estimate'): fs, ('treatment_effect', 'estimate'): rf / fs,
           ('treatment_effect', 'se'): delta(v0)}
    alts = [entry('treatment_effect', 'se', 'HC1', delta(joint(n / (n - k))), 'HC0 times n/(n-4)'),
            entry('treatment_effect', 'se', 'HC2', delta(joint(1 / (1 - hmat))), 'weighted e^2/(1-h)'),
            entry('treatment_effect', 'se', 'HC3', delta(joint(1 / (1 - hmat) ** 2)), 'weighted e^2/(1-h)^2'),
            entry('treatment_effect', 'se', 'delta_without_covariance', delta(v0, cov=False),
                  'delta method that drops the covariance of the two discontinuities')]
    return ref, alts, 'The task names HC0 and the delta method with the joint covariance.'


def applied_shift_share(task, df):
    reg = df.groupby('region_id').agg(region_weight=('region_weight', 'first'), x=('x', 'first'), y=('y', 'first'))
    e_r = reg.region_weight / reg.region_weight.sum()
    df = df.assign(e=df.region_id.map(e_r))
    z = (df.share * df.shock).groupby(df.region_id).sum().reindex(reg.index)
    v0 = {}
    for name in ('y', 'x'):
        v0[name] = reg[name] - (e_r * reg[name]).sum()
    v0['z'] = z - (e_r * z).sum()
    s_n = (df.e * df.share).groupby(df.industry_id).sum()
    shock = df.groupby('industry_id').shock.first().reindex(s_n.index)
    bar = {k: (df.e * df.share * df.region_id.map(v)).groupby(df.industry_id).sum().reindex(s_n.index) / s_n
           for k, v in v0.items()}
    N = len(s_n)

    def shock_iv(yv, xv):
        Xs = np.column_stack([np.ones(N), xv])
        Zs = np.column_stack([np.ones(N), shock])
        W = s_n.to_numpy()
        A = np.linalg.inv((Zs * W[:, None]).T @ Xs)
        b = A @ (Zs * W[:, None]).T @ yv
        e = yv - Xs @ b
        meat = (Zs * (W ** 2 * e ** 2)[:, None]).T @ Zs
        return b[1], math.sqrt((A @ meat @ A.T)[1, 1] * N / (N - 2))
    eff, se_eff = shock_iv(bar['y'].to_numpy(), bar['x'].to_numpy())
    fs, se_fs = shock_iv(bar['x'].to_numpy(), bar['z'].to_numpy())
    ref = {('effect', 'estimate'): eff, ('first_stage', 'estimate'): fs,
           ('effective_shocks', 'estimate'): 1 / (s_n ** 2).sum(), ('effect', 'se'): se_eff, ('first_stage', 'se'): se_fs}
    alts = [entry(t, 'se', 'HC0', v * math.sqrt((N - 2) / N), 'shock-level sandwich without N/(N-2)')
            for t, v in (('effect', se_eff), ('first_stage', se_fs))]
    return ref, alts, 'The task names the shock-level HC1 with N/(N-2). The tolerance 1e-5 already rejects HC0.'


# ---------------------------------------------------------------- estimate-only tasks
def vif(task, df):
    names = ['x1', 'x2', 'x3', 'x4', 'x5']
    ref, alts = {}, []
    for j, name in enumerate(names):
        xj = df[name].to_numpy()
        others = df[[c for c in names if c != name]].to_numpy()
        _, e, _ = ols(np.column_stack([np.ones(len(df)), others]), xj)
        ref[(f'vif_{name}', 'estimate')] = ((xj - xj.mean()) ** 2).sum() / (e @ e)
        _, eu, _ = ols(others, xj)
        alts.append(entry(f'vif_{name}', 'estimate', 'uncentered', (xj @ xj) / (eu @ eu),
                          'auxiliary regression without an intercept and uncentered R^2 '
                          '(statsmodels variance_inflation_factor on a design without a constant)'))
    return ref, alts, ('The task asks for the VIF of each regressor of the linear model, which has an intercept. '
                       'Federico named the uncentered VIF a wrong method on 2026-10-29.')


def var_system(df):
    Y = df[['y1', 'y2']].to_numpy()
    T = len(Y) - 2
    X = np.column_stack([Y[1:-1], Y[:-2], np.ones(T)])  # L1.y1 L1.y2 L2.y1 L2.y2 const
    B = np.linalg.lstsq(X, Y[2:], rcond=None)[0]
    U = Y[2:] - X @ B
    A1, A2 = B[0:2].T, B[2:4].T
    return A1, A2, U, T, B


def ma_coefficients(A1, A2, horizons):
    phi = [np.eye(2), A1]
    for _ in range(2, horizons + 1):
        phi.append(A1 @ phi[-1] + A2 @ phi[-2])
    return phi


def var_irf(task, df):
    A1, A2, U, T, B = var_system(df)
    sig = {'df': U.T @ U / (T - 5), 'T': U.T @ U / T}
    phi = ma_coefficients(A1, A2, 6)
    ref = {('irf_y1_to_y2_h4', 'estimate'): (phi[4] @ np.linalg.cholesky(sig['df']))[0, 1]}
    alts = []
    for label, S in sig.items():
        P = np.linalg.cholesky(S)
        tag = '' if label == 'df' else '_sigma_T'
        how = '' if label == 'df' else ' (Sigma divided by T)'
        for hh in (3, 5):
            alts.append(entry('irf_y1_to_y2_h4', 'estimate', f'horizon_{hh}{tag}', (phi[hh] @ P)[0, 1],
                              f'orthogonal response at horizon {hh}{how}'))
        alts.append(entry('irf_y1_to_y2_h4', 'estimate', f'cumulative{tag}', sum(phi[i] @ P for i in range(5))[0, 1],
                          f'cumulative orthogonal response to horizon 4{how}'))
        perm = np.array([[0, 1], [1, 0]])
        Pr = perm @ np.linalg.cholesky(perm @ S @ perm) @ perm
        alts.append(entry('irf_y1_to_y2_h4', 'estimate', f'y2_first{tag}', (phi[4] @ Pr)[0, 1],
                          f'Cholesky with y2 ordered first{how}'))
        alts.append(entry('irf_y1_to_y2_h4', 'estimate', f'upper_factor{tag}', (phi[4] @ P.T)[0, 1],
                          f'transposed Cholesky factor{how}'))
        # Stata e(b) per equation: L.y1 L2.y1 L.y2 L2.y2 _cons. The hand-read run
        # var_irf__luna__stata__zero_shot__chatbot__8 fills A1 from b[1,2,5,6] and A2 from b[3,4,7,8].
        b = np.r_[B[[0, 2, 1, 3, 4], 0], B[[0, 2, 1, 3, 4], 1]]
        M1 = np.array([[b[0], b[1]], [b[4], b[5]]])
        M2 = np.array([[b[2], b[3]], [b[6], b[7]]])
        alts.append(entry('irf_y1_to_y2_h4', 'estimate', f'misread_coefficients{tag}',
                          (ma_coefficients(M1, M2, 4)[4] @ P)[0, 1],
                          f'companion matrix filled from e(b) in lag-major order with the constant skipped{how}'))
    alts.append(entry('irf_y1_to_y2_h4', 'estimate', 'non_orthogonal', phi[4][0, 1], 'unit-shock (non-orthogonal) response'))
    return ref, alts, 'The task names the orthogonal response of y1 to a y2 shock at horizon 4 with y1 first.'


def arima_forecast(task, df):
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    y = df.y.to_numpy()

    def fc(order, trend='n'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = ARIMA(y, order=order, trend=trend).fit()
        f = res.get_forecast(1)
        lo, hi = f.conf_int(alpha=0.05)[0]
        return float(f.predicted_mean[0]), float(lo), float(hi), res
    m, lo, hi, res = fc((1, 1, 1))
    ref = {('forecast', 'estimate'): m, ('lower_95', 'estimate'): lo, ('upper_95', 'estimate'): hi}
    # Only the forecast carries the model choice. The interval bounds also depend on the forecast-variance
    # convention, which differs across correct implementations (R arima intervals are 0.4 percent wider).
    alts = [entry('forecast', 'estimate', label, fc(order, trend)[0], how)
            for label, order, trend, how in [('drift', (1, 1, 1), 't', 'ARIMA(1,1,1) with drift'),
                                             ('arima_011', (0, 1, 1), 'n', 'ARIMA(0,1,1)'),
                                             ('arima_110', (1, 1, 0), 'n', 'ARIMA(1,1,0)'),
                                             ('arima_211', (2, 1, 1), 'n', 'ARIMA(2,1,1)'),
                                             ('arima_112', (1, 1, 2), 'n', 'ARIMA(1,1,2)')]]
    phi, theta = res.params[0], res.params[1]
    alts.append(entry('forecast', 'estimate', 'wrong_formula', y[-1] + (phi + theta) * (y[-1] - y[-2]),
                      'y_T+(phi+theta)*(y_T-y_{T-1}), as in arima_forecast__luna6__stata__zero_shot__agent__7'))
    return ref, alts, ('The task names ARIMA(1,1,1) by maximum likelihood with no constant or drift. '
                       'The bounds carry no alternative: their variance convention differs across correct implementations.')


def pilot_probit_interaction_ame(task, df):
    y = df.y.to_numpy(float)
    x, d, z = df.x.to_numpy(), df.d.to_numpy(), df.z.to_numpy()

    def design(xx, dd):
        return np.column_stack([np.ones(len(xx)), xx, dd, xx * dd, z, xx ** 2])
    X = design(x, d)
    t = df.target.to_numpy() == 1

    def effects(b, mask=t, deriv_d_for_x=False):
        idx = X @ b
        slope = b[1] + b[3] * (x if deriv_d_for_x else d) + 2 * b[5] * x
        ame = (stats.norm.pdf(idx) * slope)[mask].mean()
        eff = (stats.norm.cdf(design(x, np.ones_like(d)) @ b) - stats.norm.cdf(design(x, np.zeros_like(d)) @ b))[mask].mean()
        return ame, eff
    b = probit_fit(X, y)
    ame, eff = effects(b)
    ref = {('ame_x', 'estimate'): ame, ('effect_d', 'estimate'): eff}
    bq = probit_fit(X, y, score=lambda c: X.T @ (y - stats.norm.cdf(X @ c)), start=b)
    aq, eq = effects(bq)
    af, ef = effects(b, mask=np.ones(len(y), bool))
    idx_d = X @ b
    deriv_d = (stats.norm.pdf(idx_d) * (b[2] + b[3] * x))[t].mean()
    xm, dm, zm = x[t].mean(), d[t].mean(), z[t].mean()
    row = np.array([1, xm, dm, xm * dm, zm, xm ** 2])
    at_mean = stats.norm.pdf(row @ b) * (b[1] + b[3] * dm + 2 * b[5] * xm)
    alts = [entry('ame_x', 'estimate', 'logit_score', aq, 'root of X\'(y-Phi(Xb))=0 (logit score used as the probit gradient)'),
            entry('effect_d', 'estimate', 'logit_score', eq, 'root of X\'(y-Phi(Xb))=0 (logit score used as the probit gradient)'),
            entry('ame_x', 'estimate', 'interaction_derivative_in_x', effects(b, deriv_d_for_x=True)[0],
                  'x*d derivative taken as b_xd*x in place of b_xd*d'),
            entry('ame_x', 'estimate', 'full_sample', af, 'average over all rows in place of target==1'),
            entry('effect_d', 'estimate', 'full_sample', ef, 'average over all rows in place of target==1'),
            entry('effect_d', 'estimate', 'derivative', deriv_d, 'derivative in d in place of the discrete change'),
            entry('ame_x', 'estimate', 'at_means', at_mean, 'marginal effect at the target means')]
    return ref, alts, 'The task names the probit MLE, target-subset averages, and the discrete change in d.'


def hard_heteroskedastic_probit(task, df):
    y = df.y.to_numpy(float)
    X = np.column_stack([np.ones(len(df)), df.x, df.d])
    z = df.z.to_numpy()

    def index(p):
        return (X @ p[:3]) / np.exp(p[3] * z)

    def nll(p):
        return -stats.norm.logcdf((2 * y - 1) * index(p)).sum()
    p = optimize.minimize(nll, np.zeros(4), method='BFGS', options=dict(gtol=1e-11, maxiter=20000)).x
    p = optimize.minimize(nll, p, method='Nelder-Mead', options=dict(xatol=1e-12, fatol=1e-12, maxiter=40000)).x

    def effects(q):
        s = np.exp(q[3] * z)
        ame = (stats.norm.pdf(index(q)) * q[1] / s).mean()
        X1, X0 = X.copy(), X.copy()
        X1[:, 2], X0[:, 2] = 1, 0
        eff = (stats.norm.cdf(X1 @ q[:3] / s) - stats.norm.cdf(X0 @ q[:3] / s)).mean()
        return q[3], ame, eff

    def quasi(q):
        s = np.exp(q[3] * z)
        r = y - stats.norm.cdf(index(q))
        dmu = X / s[:, None]
        dg = (-(X @ q[:3]) * z / s)[:, None]
        return np.column_stack([dmu, dg]).T @ r
    pq = optimize.root(quasi, p, method='hybr', tol=1e-14).x
    ref = dict(zip([('gamma_z', 'estimate'), ('ame_x', 'estimate'), ('effect_d', 'estimate')], effects(p)))
    alts = [entry(t, 'estimate', 'logit_score', v, 'root of the logit-type score sum (y-Phi)*d(index)/d(theta)')
            for t, v in zip(('gamma_z', 'ame_x', 'effect_d'), effects(pq))]
    return ref, alts, ('The task names the heteroskedastic probit MLE. A scale constant leaves the fitted '
                       'probabilities unidentified at the same optimum, so it has no distinct value to list.')


def applied_nonrecipient_population_iv(task, df):
    folder = data_path(task).parent
    h = pd.read_csv(folder / 'households.csv')
    census = pd.read_csv(folder / 'census.csv').set_index('eligible').nonrecipient_population
    h = h[h.recipient.eq(0)]
    el = h.eligible.to_numpy(float)
    ne = 1 - el
    dose = h.dose.to_numpy()
    X = np.column_stack([dose * ne, dose * el, ne, el, h.baseline * ne, h.baseline * el])
    Z = np.column_stack([ne, el, h.baseline * ne, h.baseline * el, h.instrument * ne, h.instrument * el])
    w = h.weight.to_numpy()
    Xh = Z @ np.linalg.solve((Z * w[:, None]).T @ Z, (Z * w[:, None]).T @ X)
    b = np.linalg.solve((Xh * w[:, None]).T @ X, (Xh * w[:, None]).T @ h.outcome.to_numpy())

    def effects(weighted=True, pop=census):
        m0 = np.average(dose[el == 0], weights=w[el == 0] if weighted else None)
        m1 = np.average(dose[el == 1], weights=w[el == 1] if weighted else None)
        e0, e1 = b[0] * m0, b[1] * m1
        return e0, e1, (pop[0] * e0 + pop[1] * e1) / (pop[0] + pop[1])
    terms = ('ineligible_effect', 'eligible_nonrecipient_effect', 'population_effect')
    ref = dict(zip([(t, 'estimate') for t in terms], effects()))
    counts = pd.Series({0: (el == 0).sum(), 1: (el == 1).sum()})
    wsum = pd.Series({0: w[el == 0].sum(), 1: w[el == 1].sum()})
    alts = [entry(t, 'estimate', 'unweighted_mean_dose', v, 'unweighted mean dose in each group')
            for t, v in zip(terms, effects(weighted=False))]
    alts += [entry('population_effect', 'estimate', 'household_count_weights', effects(pop=counts)[2],
                   'group effects averaged with retained household counts'),
             entry('population_effect', 'estimate', 'household_weight_sums', effects(pop=wsum)[2],
                   'group effects averaged with the sums of household weights')]
    return ref, alts, 'The task names household-weighted mean doses and census population weights.'


def hard_conditional_logit_welfare(task, df):
    df = df.sort_values(['person', 'alternative'])
    P = df.person.nunique()
    cost = df.cost.to_numpy().reshape(P, 3)
    qual = df.quality.to_numpy().reshape(P, 3)
    chosen = df.chosen.to_numpy().reshape(P, 3).argmax(axis=1)
    wt = df.weight.to_numpy().reshape(P, 3)[:, 0]
    asc = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])

    def util(p, c=cost):
        return p[0] * c + p[1] * qual + p[2] * asc[1] + p[3] * asc[2]

    def nll(p, weights=None):
        u = util(p)
        ll = u[np.arange(P), chosen] - np.log(np.exp(u).sum(axis=1))
        return -(ll if weights is None else weights * ll).sum()

    def fit(weights=None):
        return optimize.minimize(nll, np.zeros(4), args=(weights,), method='BFGS', options=dict(gtol=1e-11)).x

    def results(p, weights=wt):
        policy = cost.copy()
        policy[:, 1] *= 0.7
        lse = lambda u: np.log(np.exp(u).sum(axis=1))  # noqa: E731
        welfare = (lse(util(p, policy)) - lse(util(p))) / (-p[0])
        prob = lambda u: np.exp(u[:, 1]) / np.exp(u).sum(axis=1)  # noqa: E731
        change = prob(util(p, policy)) - prob(util(p))
        return np.average(welfare, weights=weights), np.average(change, weights=weights), p[0]
    p = fit()
    terms = ('welfare_gain', 'share_change_alt1', 'cost_coefficient')
    ref = dict(zip([(t, 'estimate') for t in terms], results(p)))
    alts = [entry(t, 'estimate', 'unweighted_means', v, 'post-estimation averages without the person weights')
            for t, v in zip(terms[:2], results(p, weights=None)[:2])]
    alts += [entry(t, 'estimate', 'weighted_estimation', v, 'conditional logit estimated with person weights')
             for t, v in zip(terms, results(fit(wt)))]
    return ref, alts, 'The task names unweighted estimation and person-weighted post-estimation averages.'


def ttest_welch(task, df):
    a, b = df.y[df.group == 1].to_numpy(), df.y[df.group == 0].to_numpy()
    w = stats.ttest_ind(a, b, equal_var=False)
    s = stats.ttest_ind(a, b, equal_var=True)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    ref = {('t', 'estimate'): w.statistic, ('df', 'estimate'): dfw, ('p_value', 'estimate'): w.pvalue}
    alts = [entry('t', 'estimate', 'pooled_variance', s.statistic, 'Student t-test with a pooled variance'),
            entry('df', 'estimate', 'pooled_variance', len(a) + len(b) - 2, 'Student t-test degrees of freedom'),
            entry('p_value', 'estimate', 'pooled_variance', s.pvalue, 'Student t-test p-value')]
    return ref, alts, 'The task names the Welch test.'


def anova_oneway(task, df):
    groups = [g.y.to_numpy() for _, g in df.groupby('group')]
    f = stats.f_oneway(*groups)
    k, n = len(groups), len(df)
    wts = np.array([len(g) / g.var(ddof=1) for g in groups])
    means = np.array([g.mean() for g in groups])
    mw = (wts * means).sum() / wts.sum()
    lam = (((1 - wts / wts.sum()) ** 2) / (np.array([len(g) for g in groups]) - 1)).sum()
    fw = ((wts * (means - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) * lam / (k ** 2 - 1))
    dfw = (k ** 2 - 1) / (3 * lam)
    ref = {('F', 'estimate'): f.statistic, ('df_num', 'estimate'): k - 1, ('df_den', 'estimate'): n - k,
           ('p_value', 'estimate'): f.pvalue}
    alts = [entry('F', 'estimate', 'welch_anova', fw, 'Welch heteroskedastic one-way ANOVA'),
            entry('df_den', 'estimate', 'welch_anova', dfw, 'Welch ANOVA denominator degrees of freedom'),
            entry('p_value', 'estimate', 'welch_anova', stats.f.sf(fw, k - 1, dfw), 'Welch ANOVA p-value')]
    return ref, alts, 'The task names the one-way ANOVA omnibus F test.'


def chisq_indep(task, df):
    table = pd.crosstab(df['row'], df['col']).to_numpy()
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    g, pg, _, _ = stats.chi2_contingency(table, correction=False, lambda_='log-likelihood')
    ref = {('chi2', 'estimate'): chi2, ('df', 'estimate'): dof, ('p_value', 'estimate'): p}
    alts = [entry('chi2', 'estimate', 'likelihood_ratio', g, 'likelihood-ratio G statistic'),
            entry('p_value', 'estimate', 'likelihood_ratio', pg, 'likelihood-ratio p-value')]
    return ref, alts, 'The task names the Pearson statistic without Yates correction.'


def rdd_sharp(task, df):
    h = 0.5
    d = df[df.z.abs() < h]
    z, y = d.z.to_numpy(), d.y.to_numpy()

    def late(w):
        out = []
        for side in (z >= 0, z < 0):
            b, _, _ = ols(np.column_stack([np.ones(side.sum()), z[side]]), y[side], w[side])
            out.append(b[0])
        return out[0] - out[1]
    ref = {('late', 'estimate'): late(1 - np.abs(z) / h)}
    alts = [entry('late', 'estimate', 'uniform_kernel', late(np.ones(len(z))), 'uniform kernel weights'),
            entry('late', 'estimate', 'weights_1_minus_abs_z', late(1 - np.abs(z)),
                  'triangular weights 1-|z| that ignore the bandwidth')]
    return ref, alts, 'The task names a triangular kernel with h = 0.5.'


def staggered_did(task, df):
    m = df.groupby(['cohort', 'year']).y_did.mean()
    cells = [(g, t) for g in (4, 6, 8) for t in range(g, 10)]
    att = {c: (m[c] - m[(c[0], c[0] - 1)]) - (m[(-1, c[1])] - m[(-1, c[0] - 1)]) for c in cells}
    sizes = df[df.year.eq(0)].groupby('cohort').size()
    ref = {('att', 'estimate'): np.mean(list(att.values()))}
    weighted = sum(att[c] * sizes[c[0]] for c in cells) / sum(sizes[c[0]] for c in cells)
    X = np.column_stack([df.treat_post, pd.get_dummies(df.firm, drop_first=True).to_numpy(float),
                         pd.get_dummies(df.year, drop_first=True).to_numpy(float), np.ones(len(df))])
    twfe = ols(X, df.y_did.to_numpy())[0][0]
    alts = [entry('att', 'estimate', 'cohort_size_weights', weighted, 'cells weighted by cohort size'),
            entry('att', 'estimate', 'twfe', twfe, 'two-way fixed-effects regression on treat_post')]
    return ref, alts, 'The task names the equal-weight average of the group-time cells and excludes TWFE.'


def broad_cohort_stack(task, df):
    atts, treated, rows = {}, {}, {}
    for c in (3, 5, 7):
        q = df[df.time.between(c - 2, c + 2) & (df.g.eq(c) | df.g.eq(0) | df.g.gt(c + 2))]
        full = q.groupby('id').time.transform('size').eq(5)
        q = q[full]
        did = (q.g.eq(c) & q.time.ge(c)).astype(float).to_numpy()
        X = np.column_stack([did, pd.get_dummies(q.id, drop_first=True).to_numpy(float),
                             pd.get_dummies(q.time, drop_first=True).to_numpy(float), np.ones(len(q))])
        atts[c] = ols(X, q.y.to_numpy())[0][0]
        treated[c], rows[c] = q[q.g.eq(c)].id.nunique(), len(q)
    ref = {(f'att_g{c}', 'estimate'): atts[c] for c in atts}
    ref[('att_weighted', 'estimate')] = sum(atts[c] * treated[c] for c in atts) / sum(treated.values())
    alts = [entry('att_weighted', 'estimate', 'row_count_weights', sum(atts[c] * rows[c] for c in atts) / sum(rows.values()),
                  'cohort coefficients weighted by retained rows'),
            entry('att_weighted', 'estimate', 'equal_weights', np.mean(list(atts.values())), 'unweighted mean of the three coefficients')]
    return ref, alts, 'The task names treated-unit-count weights.'


def broad_oaxaca(task, df):
    def fit(g):
        q = df[df.group.eq(g)]
        X = np.column_stack([np.ones(len(q)), q.x, q.z])
        return ols(X, q.y.to_numpy())[0], X.mean(axis=0)
    b1, m1 = fit(1)
    b0, m0 = fit(0)
    gap = df.y[df.group.eq(1)].mean() - df.y[df.group.eq(0)].mean()
    ref = {('gap', 'estimate'): gap, ('explained', 'estimate'): (m1 - m0) @ b0, ('unexplained', 'estimate'): m1 @ (b1 - b0),
           ('explained_x', 'estimate'): (m1[1] - m0[1]) * b0[1]}
    alts = [entry('explained', 'estimate', 'group1_reference', (m1 - m0) @ b1, 'group 1 coefficients as the reference'),
            entry('unexplained', 'estimate', 'group1_reference', m0 @ (b1 - b0), 'group 1 coefficients as the reference'),
            entry('explained_x', 'estimate', 'group1_reference', (m1[1] - m0[1]) * b1[1], 'group 1 coefficients as the reference')]
    return ref, alts, 'The task names group 0 coefficients as the reference.'


def hard_rolling(task, df):
    y = df.sort_values('time').y.to_numpy()

    def forecasts(pairs):
        out = {}
        for f in range(101, 251):
            i = f - 1  # zero-based index of y_f
            ys, yl = y[i - pairs:i], y[i - pairs - 1:i - 1]
            b, _, _ = ols(np.column_stack([np.ones(pairs), yl]), ys)
            out[f] = b[0] + b[1] * y[i - 1]
        return out
    fr = forecasts(99)
    rmse = math.sqrt(np.mean([(y[f - 1] - fr[f]) ** 2 for f in fr]))
    ref = {('rmse', 'estimate'): rmse} | {(f'forecast_{f}', 'estimate'): fr[f] for f in (101, 150, 200, 250)}
    fa = forecasts(98)
    alts = [entry(f'forecast_{f}', 'estimate', 'window_98_pairs', fa[f], 'rolling window of 98 regression pairs')
            for f in (101, 150, 200, 250)]
    alts.append(entry('rmse', 'estimate', 'window_98_pairs', math.sqrt(np.mean([(y[f - 1] - fa[f]) ** 2 for f in fa])),
                      'rolling window of 98 regression pairs'))
    return ref, alts, 'The task names 99 regression pairs from the previous 100 levels.'


def cooks_distance(task, df):
    names = ['x1', 'x2', 'x3', 'x4', 'x5']
    X = np.column_stack([np.ones(len(df)), df[names]])
    b, e, bread = ols(X, df.y.to_numpy())
    n, p = X.shape
    h = np.einsum('ij,jk,ik->i', X, bread, X)
    s2 = e @ e / (n - p)
    d = e ** 2 * h / (p * s2 * (1 - h) ** 2)
    order = np.argsort(-d)[:5]
    ref = {}
    for r, i in enumerate(order, 1):
        ref[(f'rank{r}_obs_id', 'estimate')] = float(i)
        ref[(f'rank{r}_cooks_d', 'estimate')] = d[i]
    alts = [entry(f'rank{r}_cooks_d', 'estimate', 'slopes_only_p', d[i] * p / (p - 1),
                  'Cook distance divided by the slope count 5 in place of 6 parameters')
            for r, i in enumerate(order, 1)]
    return ref, alts, 'The model has an intercept, so Cook distance uses p = 6.'


TASKS = {f.__name__: f for f in [
    ols_robust, iv2sls_robust, cluster_ols, two_way_cluster, twoway_fe, lp_irf, applied_cumulative_hac,
    applied_fuzzy_rd, applied_shift_share, vif, var_irf, arima_forecast, pilot_probit_interaction_ame,
    hard_heteroskedastic_probit, applied_nonrecipient_population_iv, hard_conditional_logit_welfare,
    ttest_welch, anova_oneway, chisq_indep, rdd_sharp, staggered_did, broad_cohort_stack, broad_oaxaca,
    hard_rolling, cooks_distance]}

# Tasks with no listed alternative, and the reason.
NOT_LISTED = {
    'applied_multivintage_trade_iv': 'The reviewed wrong constructions (duplicate crosswalk codes, base employment) move the values far outside the tolerance.',
    'broad_ar_confidence_set': 'No success cluster with a distinct method; the scan found only rounded values.',
    'broad_multinomial_logit': 'The scan found rounded values only; the alternatives (another base outcome) are far outside the tolerance.',
    'broad_ordered_logit': 'The Python cluster differs by 2e-5 to 5e-5 through optimizer tolerance; no excluded method lies inside the tolerance.',
    'broad_tobit': 'The scan found rounded values only.',
    'cox_hazard': 'The task names no tie method; the scan cluster differs by 4e-6 in every language (optimizer tolerance).',
    'hard_panel_lp_iv': 'No success cluster with a distinct value.',
    'hard_poisson_hurdle_effects': 'The scan clusters differ by 1e-6 to 9e-5 (optimizer tolerance).',
    'hard_truncated_normal_effects': 'The scan clusters differ by 1e-6 to 3e-6 (optimizer tolerance).',
    'logit': 'No success cluster with a distinct value.',
    'lp04_ppml': 'The Stata cluster differs by 1e-5 (convergence tolerance of ppmlhdfe).',
    'lp11_duration': 'The R cluster differs by 2e-5 (optimizer tolerance); the log-probability check guards the probability term.',
    'poisson_offset': 'No success cluster with a distinct value.',
    'probit': 'The scan found rounded values only.',
    'var_granger': 'No success cluster with a distinct value; the F and exact df checks guard the p-value.',
}


def accepted_values(spec: dict, catalog: dict) -> dict:
    """{task: {(term, kind): [accepted values]}} from the reference and scoring_spec.json."""
    out = {}
    for task, cat in catalog.items():
        rule = spec['tasks'].get(task, {})
        values = {}
        for row in cat['reference']['estimates']:
            term = row['term']
            values[(term, 'estimate')] = [float(row['estimate'])]
            extra = rule.get('terms', {}).get(term, {}).get('accepted_values', [])
            values[(term, 'estimate')] += [float(v) for v in extra
                                           if abs(v - row['estimate']) > SAME_VALUE * abs(row['estimate'])]
            if rule.get('se_scored') and term in rule.get('se_terms', {}):
                values[(term, 'se')] = [float(v) for v in rule['se_terms'][term]['accepted_values']]
        out[task] = values
    return out


def build(out: Path = OUT) -> dict:
    catalog = json.loads(CATALOG.read_text())
    spec = json.loads(SPEC.read_text())
    accepted = accepted_values(spec, catalog)
    tasks, dropped = {}, []
    for name, func in TASKS.items():
        cat = catalog[name]
        path = data_path(cat)
        ref, alts, notes = func(cat, pd.read_csv(path))
        checks = []
        for (term, kind), value in ref.items():
            stored = accepted[name][(term, kind)][0]
            rel = abs(value - stored) / max(abs(stored), 1e-12)
            checks.append(dict(term=term, kind=kind, stored=stored, recomputed=float(value), relative_difference=rel))
            if rel > REFERENCE_MATCH:
                raise ValueError(f'{name} {term} {kind}: recomputed reference {value} differs from {stored}')
        listed = []
        for alt in alts:
            acc = accepted[name][(alt['term'], alt['kind'])]
            nearest = min(acc, key=lambda a: abs(alt['value'] - a))
            alt['relative_gap'] = abs(alt['value'] - nearest) / abs(nearest)
            if alt['relative_gap'] <= SAME_VALUE:
                dropped.append(dict(task=name, **alt, reason='equals an accepted value'))
            else:
                listed.append(alt)
        tasks[name] = dict(data=str(path.relative_to(ROOT)), data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           notes=notes, reference_checks=checks, alternatives=listed)
    result = dict(
        rule=('A scored value passes only if it is inside its tolerance and not closer to a listed alternative of its '
              'term than to the nearest accepted value of that term. An alternative is tested only when it and every '
              'accepted value differ after rounding to the decimals of the reported value.'),
        decision_record='quality_reports/decision_records/2026-10-29_wrong-method-alternative-check.md',
        generated_by='code/paper/scoring_alternatives.py',
        scoring_spec_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        accepted_estimate_conventions={t: {term: v[1:] for (term, kind), v in values.items() if kind == 'estimate' and len(v) > 1}
                                       for t, values in accepted.items()
                                       if any(kind == 'estimate' and len(v) > 1 for (term, kind), v in values.items())},
        tasks=tasks, not_listed=NOT_LISTED, dropped_as_accepted=dropped)
    out.write_text(json.dumps(result, indent=2) + '\n')
    return result


# ---------------------------------------------------------------- the check
def decimals(value: float, token=None) -> int:
    """Decimal places of the reported value: its token when given, otherwise the shortest repr."""
    text = str(token).strip() if token is not None and str(token).strip() not in ('', 'nan', 'None') else repr(float(value))
    try:
        exponent = Decimal(text).as_tuple().exponent
    except Exception:  # noqa: BLE001  an unparsable token falls back to the float
        exponent = Decimal(repr(float(value))).as_tuple().exponent
    return -int(exponent)


def closer_alternative(got: float, accepted: list[float], alternatives: list[dict], places: int):
    """The alternative that the reported value is closer to than to every accepted value, or None."""
    if got is None or not np.isfinite(got):
        return None
    d_ref = min(abs(got - a) for a in accepted)
    best = None
    for alt in alternatives:
        if any(round(alt['value'], places) == round(a, places) for a in accepted):
            continue  # the reported precision cannot separate this alternative from an accepted value
        d_alt = abs(got - alt['value'])
        if d_alt < d_ref and (best is None or d_alt < abs(got - best['value'])):
            best = alt
    return best


def load(path: Path = OUT) -> dict:
    """The alternatives file, after a check that its task data and scoring_spec.json are unchanged."""
    table = json.loads(path.read_text())
    if table['scoring_spec_sha256'] != hashlib.sha256(SPEC.read_bytes()).hexdigest():
        raise ValueError('scoring_spec.json changed; rebuild code/paper/scoring_alternatives.py')
    for task, info in table['tasks'].items():
        if hashlib.sha256((ROOT / info['data']).read_bytes()).hexdigest() != info['data_sha256']:
            raise ValueError(f'The {task} data changed; rebuild code/paper/scoring_alternatives.py')
    return table


def apply_check(runs: pd.DataFrame, terms: pd.DataFrame, ses: pd.DataFrame, catalog: dict, table: dict | None = None):
    """Apply the rule to raw PNAS scoring frames. Returns (runs, terms, ses, flagged rows).

    Only passing values can change, and only from pass to fail. term_checks and se_checks gain the
    columns alternative_method and alternative_value. A run with a changed value becomes a silent
    failure, because it executed and delivered every value.
    """
    table = load() if table is None else table
    spec = json.loads(SPEC.read_text())
    accepted = accepted_values(spec, catalog)
    alts = {}
    for task, info in table['tasks'].items():
        for alt in info['alternatives']:
            alts.setdefault((task, alt['term'], alt['kind']), []).append(alt)
    terms, ses, runs = terms.copy(), ses.copy(), runs.copy()
    flagged = []
    for frame, kind, got_col, pass_col, token_col in ((terms, 'estimate', 'got_estimate', 'coef_pass', None),
                                                      (ses, 'se', 'se_got', 'se_pass', 'se_token')):
        frame['alternative_method'] = ''
        frame['alternative_value'] = np.nan
        passed = frame[pass_col].eq(True)
        for i in frame.index[passed]:
            row = frame.loc[i]
            options = alts.get((row.task, row.term, kind))
            if not options:
                continue
            token = row[token_col] if token_col else None
            hit = closer_alternative(float(row[got_col]), accepted[row.task][(row.term, kind)], options,
                                     decimals(row[got_col], token))
            if hit is not None:
                frame.loc[i, ['alternative_method', 'alternative_value']] = [hit['method'], hit['value']]
                frame.loc[i, pass_col] = False
                flagged.append(dict(position_id=row.position_id, task=row.task, term=row.term, kind=kind,
                                    got=float(row[got_col]), reference=accepted[row.task][(row.term, kind)][0],
                                    alternative_method=hit['method'], alternative_value=hit['value']))
        if 'se_reason' in frame:
            frame['se_reason'] = frame.se_reason.astype(object)
            frame.loc[frame.alternative_method.ne(''), 'se_reason'] = 'closer_to_alternative'
    flagged = pd.DataFrame(flagged, columns=['position_id', 'task', 'term', 'kind', 'got', 'reference',
                                             'alternative_method', 'alternative_value'])
    changed = set(flagged.position_id)
    turn = runs.position_id.isin(changed) & runs.primary.astype(bool)
    notes = flagged.assign(note=[f"{t}: {'standard error' if k == 'se' else 'estimate'} closer to the {m} value"
                                 for t, k, m in zip(flagged.term, flagged.kind, flagged.alternative_method)]
                           ).groupby('position_id').note.apply(' | '.join)
    old = runs.scoring_notes.fillna('').astype(str)
    add = runs.position_id.map(notes).fillna('')
    runs['scoring_notes'] = [(' | '.join(x for x in (o, a) if x) or np.nan) for o, a in zip(old, add)]
    runs.loc[turn, ['primary', 'y']] = [False, 0]
    runs.loc[turn, 'failure_category'] = 'Silent failure'
    flagged['run_changed'] = flagged.position_id.isin(set(runs.position_id[turn]))
    return runs, terms, ses, flagged


if __name__ == '__main__':
    result = build()
    n = sum(len(t['alternatives']) for t in result['tasks'].values())
    print(json.dumps(dict(tasks=len(result['tasks']), alternatives=n, dropped=len(result['dropped_as_accepted']),
                          out=str(OUT.relative_to(ROOT)))))
