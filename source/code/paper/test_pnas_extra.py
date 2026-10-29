"""Checks for the extra PNAS analyses (code/paper/pnas_extra.py).

Run: uv run --no-sync pytest -q code/paper/test_pnas_extra.py
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnas_extra as x  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HAVE_DATA = (x.DATA / 'data/runs.csv').is_file() and (x.ORIGINAL / 'data/runs.csv').is_file()


# ---------------------------------------------------------------------------
# 1. Wild cluster bootstrap as a sign-flip test
# ---------------------------------------------------------------------------
def toy_design(groups=7, seed=3):
    """A balanced design: tasks, three cells, two runs per task and cell, binary outcomes."""
    rng = np.random.default_rng(seed)
    rows = [dict(task=f't{g}', cell=c, y=float(rng.random() < .3 + .1 * c + .05 * g))
            for g in range(groups) for c in range(3) for _ in range(2)]
    return pd.DataFrame(rows)


def test_sign_flip_equals_the_null_imposed_wild_cluster_bootstrap():
    data = toy_design()
    tasks = sorted(data.task.unique())
    weights = x.rademacher(len(tasks), draws=25, seed=11)
    # Contrast: cell 1 minus cell 0. Task-level contrast values.
    means = data.groupby(['task', 'cell']).y.mean().unstack()
    values = (means[1] - means[0]).reindex(tasks).to_numpy()
    mine = x.sign_flip_test(values[:, None], weights=weights)
    # Brute force: restricted fit (cell 1 effect zero), Rademacher-flipped residuals, CR1 t.
    full = smf.ols('y ~ C(task) + C(cell)', data).fit(cov_type='cluster', cov_kwds=dict(groups=data.task.astype('category').cat.codes))
    t_obs = full.params['C(cell)[T.1]'] / full.bse['C(cell)[T.1]']
    restricted = smf.ols('y ~ C(task) + I(cell == 2)', data).fit()
    index = data.task.map({t: i for i, t in enumerate(tasks)}).to_numpy()
    for draw, w in enumerate(weights):
        star = data.assign(y=restricted.fittedvalues + restricted.resid * w[index])
        fit = smf.ols('y ~ C(task) + C(cell)', star).fit(cov_type='cluster', cov_kwds=dict(groups=index))
        t_star = fit.params['C(cell)[T.1]'] / fit.bse['C(cell)[T.1]']
        mine_star = (weights[draw] @ values / len(values)) / (
            np.sqrt((values ** 2).sum() - len(values) * (weights[draw] @ values / len(values)) ** 2) / len(values))
        mine_t = values.mean() / (np.sqrt(((values - values.mean()) ** 2).sum()) / len(values))
        # The CR1 factor is common to t and t*, so the ratios agree exactly.
        assert t_star / t_obs == pytest.approx(mine_star / mine_t, rel=1e-8)
    assert np.isfinite(mine['p'][0])


def test_sign_flip_counts_ties_and_matches_enumeration():
    values = np.array([10., -10., 10., 0., 5., 5., 0., -5.])
    weights = np.array(list(itertools.product([-1., 1.], repeat=len(values))))
    result = x.sign_flip_test(values[:, None], weights=weights)
    t = values.mean() / np.sqrt(((values - values.mean()) ** 2).sum())
    star = weights @ values / len(values)
    t_star = star / np.sqrt(np.maximum((values ** 2).sum() - len(values) * star ** 2, 0))
    share = np.mean(np.abs(t_star) >= abs(t) * (1 - 1e-9))
    assert result['p_raw'][0] == pytest.approx(share)
    assert result['count_ge'][0] > result['count_gt'][0]  # exact ties exist with discrete values
    assert result['p'][0] == pytest.approx((result['count_ge'][0] + 1) / (len(weights) + 1))


def test_one_informative_task_gives_p_one_and_none_gives_missing():
    one = np.zeros((40, 1))
    one[3] = 7.5
    assert x.sign_flip_test(one)['p'][0] == pytest.approx(1.0)
    assert np.isnan(x.sign_flip_test(np.zeros((40, 1)))['p'][0])


@pytest.mark.parametrize('method,ours', [('holm', x.holm), ('fdr_bh', x.benjamini_hochberg)])
def test_adjustments_match_statsmodels(method, ours):
    p = np.array([.001, .04, .03, .2, .0001, .5, .049, .012, .9, .3])
    assert ours(p) == pytest.approx(multipletests(p, method=method)[1])


def test_missing_p_enters_the_family_as_one():
    assert x.holm(np.array([.01, np.nan])) == pytest.approx([.02, 1.0])


def test_table1_rows_and_families_follow_the_pnas_table():
    spec = importlib.util.spec_from_file_location('pnas_build_for_extra', ROOT / 'paper/pnas/build_pnas.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    assert tuple(x.PNAS_PROMPT_ROWS) == tuple(build.PROMPT_ROWS)
    rows = x.table1_rows()
    assert [r['contrast'] for r in rows][:3] == ['H3_self_repair', 'H3_tool_loop', 'H2_fewshot']
    sizes = pd.Series([r['family'] for r in rows]).value_counts().to_dict()
    assert sizes == {'primary': 2, 'secondary': 10}


# ---------------------------------------------------------------------------
# 2. Silent failure kinds
# ---------------------------------------------------------------------------
def test_wrong_method_rows_inside_tolerance_are_not_estimate_failures():
    terms = pd.DataFrame(dict(position_id=['a', 'b', 'c'], coef_err=[.001, .02, .001], coef_tol=[.01, .01, .01],
                              coef_pass=[False, False, True], exact_estimate_pass=[np.nan] * 3,
                              got_estimate=[1., 1., 1.], ref_estimate=[1., 1., 1.],
                              alternative_method=['HC0', np.nan, np.nan]))
    assert x.outside_estimates(terms).tolist() == [False, True, False]


def test_silent_kinds_separate_estimate_se_and_wrong_method():
    runs = pd.DataFrame(dict(position_id=['e', 's', 'w'], failure_category=['Silent failure'] * 3,
                             model='luna', task='t', platform='r', prompt='zero_shot', agency='agent'))
    terms = pd.DataFrame(dict(position_id=['e', 's', 'w'], coef_err=[.5, 0, 0], coef_tol=[.01] * 3,
                              coef_pass=[False, True, True], exact_estimate_pass=[np.nan] * 3,
                              got_estimate=[1.] * 3, ref_estimate=[1.] * 3, alternative_method=[np.nan] * 3))
    ses = pd.DataFrame(dict(position_id=['s', 'w'], se_pass=[False, False], se_reason=['se_outside_tolerance', 'closer_to_alternative'],
                            se_err=[.03, .001], se_ref=[1., 1.], se_got=[1.03, 1.001]))
    kinds = x.silent_kinds(runs, terms, ses, changed={'w'}).set_index('position_id')
    assert kinds.kind.to_dict() == dict(e='estimate', s='se_only', w='wrong_method')
    assert kinds.loc['s', 'largest_se_deviation'] == pytest.approx(.03)


# ---------------------------------------------------------------------------
# 3. Majority vote
# ---------------------------------------------------------------------------
def brute_vote(answers, success, k):
    """Expected success of the plurality answer over all k-subsets (None marks no answer)."""
    values = []
    for subset in itertools.combinations(range(len(answers)), k):
        got = [answers[i] for i in subset if answers[i] is not None]
        if not got:
            values.append(0.0)
            continue
        counts = pd.Series(got).value_counts()
        top = counts[counts == counts.max()].index
        picks = [i for i in subset if answers[i] in top]
        values.append(np.mean([success[i] for i in picks]))
    return np.mean(values)


@pytest.mark.parametrize('k', [1, 3, 5, 9])
def test_vote_matches_brute_force(k):
    answers = ['right'] * 4 + ['wrong'] * 3 + ['other'] + [None] * 2
    success = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    answered = np.array([a is not None for a in answers])
    agree = np.array([[a == b and a is not None for b in answers] for a in answers])
    among, _ = x.vote(agree, answered, success, x.subsets(10, k))
    assert among == pytest.approx(brute_vote(answers, success, k))


def test_vote_with_no_answer_as_a_candidate():
    answers = ['right'] * 3 + [None] * 7
    success = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    answered = np.array([a is not None for a in answers])
    agree = np.array([[a == b and a is not None for b in answers] for a in answers])
    among, with_none = x.vote(agree, answered, success, x.subsets(10, 3))
    assert among > with_none
    assert x.vote(agree, answered, success, x.subsets(10, 10))[1] == 0.0


def test_agreement_uses_term_tolerance_and_missing_se():
    est = np.array([[1.0], [1.004], [1.02]])
    se = np.array([[np.nan], [np.nan], [.5]])
    agree = x.agreement_matrix(est, np.array([.005]), se, np.zeros_like(se))
    assert agree[0, 1] and not agree[0, 2] and not agree[1, 2]


# ---------------------------------------------------------------------------
# 4. Shared wrong values
# ---------------------------------------------------------------------------
def test_cluster_values_groups_within_tolerance():
    labels = x.cluster_values(np.array([1.0, 1.001, 1.002, 5.0, 5.0005, 9.0]), .002)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] != labels[0]
    assert len(set(labels)) == 3


# ---------------------------------------------------------------------------
# 5. Self-check keyword rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('text,label', [
    ('Independent check: estimates match to 1e-12. No discrepancies found.', 'agreement'),
    ('Discrepancies found: none.', 'agreement'),
    ('The independent check disagreed with Stata. This discrepancy is unresolved.', 'discrepancy'),
    ('I did not resolve that discrepancy, so the estimates are not verified.', 'discrepancy'),
    ('An earlier version of the check omitted a weight and disagreed; it was corrected.', 'not discrepancy'),
    ('The small SE discrepancy (~0.05%) is due to the df convention.', 'agreement'),
    ('No discrepancy requires correction.', 'agreement'),
    ('Ran the script.', 'neither'),
    ('', 'neither'),
])
def test_classify_check(text, label):
    got = x.classify_check(text)['label']
    assert got != 'discrepancy' if label == 'not discrepancy' else got == label


# ---------------------------------------------------------------------------
# 6. Lenient reader
# ---------------------------------------------------------------------------
def test_lenient_values_reads_unread_shapes():
    log = 'x1 se=0.1 estimate=0.5 n_obs=10\nANSWER x2 estimate=-1e-3\n'
    assert x.lenient_values(log, 'stata', ['x1', 'x2']) == {'x1': 0.5, 'x2': -0.001}
    keyed = '{"estimates": {"x1": {"estimate": 0.5}, "x2": [2.0]}}'
    assert x.lenient_values(keyed, 'python', ['x1', 'x2']) == {'x1': 0.5, 'x2': 2.0}
    prose = 'Result:\n{"estimates": [{"term": "x1", "estimate": 0.25}]}\nDone.'
    assert x.lenient_values(prose, 'r', ['x1']) == {'x1': 0.25}


# ---------------------------------------------------------------------------
# On the current data
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_DATA, reason='analysis data not present')
def test_contrasts_reproduce_the_current_table_and_the_stored_bootstrap():
    data = x.load()
    stored = pd.read_csv(x.DATA / 'tables/regression_contrasts.csv').set_index(['model', 'contrast'])
    for column in ('sonnet', 'opus', 'pooled'):
        frame = x.column_inference(data['runs'], column, x.GROUPS[column]).set_index('contrast')
        for contrast, rec in frame.iterrows():
            ref = stored.loc[(column, contrast)]
            assert rec.estimate_pp == pytest.approx(ref.estimate_pp, abs=1e-9)
            assert rec.se_cr1_pp == pytest.approx(ref.se_cr1_pp, abs=1e-9)
            assert rec.informative_tasks == ref.informative_tasks
    check = x.wcb_check()
    assert not check.beyond_mc_error.any()
    assert check.estimate_diff.abs().max() < 1e-9


@pytest.mark.skipif(not HAVE_DATA, reason='analysis data not present')
def test_silent_kinds_cover_every_silent_failure():
    data = x.load()
    kinds = x.silent_kinds(data['runs'], data['terms'], data['ses'], x.changed_by_alternative(x.DATA))
    assert len(kinds) == int(data['runs'].failure_category.eq('Silent failure').sum())
    assert set(kinds.kind) <= {'estimate', 'se_only', 'wrong_method', 'other'}
