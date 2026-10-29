"""Checks of the wrong-method alternative rule (code/paper/scoring_alternatives.py)."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring_alternatives as sa  # noqa: E402

HC1_REF = 0.02195426461854331
HC3 = 0.02197106718


def alt(value, method='HC3'):
    return dict(term='const', kind='se', method=method, value=value)


def test_decimals_follow_the_token_then_the_shortest_repr():
    assert sa.decimals(0.022) == 3
    assert sa.decimals(1.2e-05) == 6
    assert sa.decimals(0.022, '0.02200') == 5
    assert sa.decimals(0.043180421974, None) == 12


def test_reference_passes_and_the_alternative_fails():
    assert sa.closer_alternative(HC1_REF, [HC1_REF], [alt(HC3)], 17) is None
    assert sa.closer_alternative(HC3, [HC1_REF], [alt(HC3)], 17)['method'] == 'HC3'
    # A value at the reference side of the midpoint passes.
    assert sa.closer_alternative(HC1_REF + 0.4 * (HC3 - HC1_REF), [HC1_REF], [alt(HC3)], 17) is None


def test_rounded_values_that_cannot_separate_the_alternative_pass():
    # 0.022 is closer to HC3 than to HC1, but both round to 0.022 at three decimals.
    assert sa.closer_alternative(0.022, [HC1_REF], [alt(HC3)], sa.decimals(0.022)) is None
    # At five decimals they differ (0.02195 against 0.02197), and the rounded HC1 value passes.
    assert sa.closer_alternative(round(HC1_REF, 5), [HC1_REF], [alt(HC3)], 5) is None
    assert sa.closer_alternative(round(HC3, 5), [HC1_REF], [alt(HC3)], 5)['method'] == 'HC3'


def test_every_accepted_value_passes_and_the_rounded_reference_passes():
    table = sa.load()
    catalog = json.loads(sa.CATALOG.read_text())
    accepted = sa.accepted_values(json.loads(sa.SPEC.read_text()), catalog)
    for task, info in table['tasks'].items():
        by_term = {}
        for a in info['alternatives']:
            by_term.setdefault((a['term'], a['kind']), []).append(a)
        for key, alts in by_term.items():
            for value in accepted[task][key]:
                for places in range(2, 18):
                    assert sa.closer_alternative(round(value, places), accepted[task][key], alts, places) is None, \
                        (task, key, value, places)


def test_listed_alternatives_differ_from_accepted_values_and_references_reproduce():
    table = sa.load()
    for task, info in table['tasks'].items():
        assert all(c['relative_difference'] <= sa.REFERENCE_MATCH for c in info['reference_checks']), task
        assert all(a['relative_gap'] > sa.SAME_VALUE for a in info['alternatives']), task
    listed = {(t, a['term'], a['method']): a['value'] for t, i in table['tasks'].items() for a in i['alternatives']}
    # The measured wrong-method values of the 2026-10-28 review.
    assert listed[('iv2sls_robust', 'x1_iv', 'HC0')] == pytest.approx(0.043180421974, rel=1e-8)
    assert listed[('vif', 'vif_x1', 'uncentered')] == pytest.approx(3.990921, rel=1e-6)
    assert listed[('var_irf', 'irf_y1_to_y2_h4', 'misread_coefficients_sigma_T')] == pytest.approx(0.084852, rel=1e-5)
    # Accepted conventions are never listed.
    assert not any(t == 'var_irf' and abs(v - 0.0807026210) < 1e-8 for (t, _, _), v in listed.items())
    assert ('cluster_ols', 'const', 'CR0') not in listed


def test_hc_family_matches_statsmodels():
    df = pd.read_csv(sa.data_path(json.loads(sa.CATALOG.read_text())['ols_robust']))
    X = np.column_stack([np.ones(len(df)), df.x1, df.x2])
    b, e, bread = sa.ols(X, df.y_lin.to_numpy())
    v = sa.hc_family(X, e, bread)
    fit = sm.OLS(df.y_lin.to_numpy(), X).fit()
    for name in ('HC0', 'HC1', 'HC2', 'HC3'):
        np.testing.assert_allclose(v[name], getattr(fit, f'cov_{name}'), rtol=1e-10)


def frames():
    runs = pd.DataFrame(dict(position_id=['a', 'b', 'c'], task='ols_robust', primary=[True, False, True],
                             y=[1, 0, 1], failure_category=['Success', 'Silent failure', 'Success'],
                             scoring_notes=[np.nan, 'x1: estimate outside tolerance', np.nan]))
    terms = pd.DataFrame(dict(position_id=['a', 'b', 'c'], task='ols_robust', term='const',
                              got_estimate=1.0115765173916533, coef_pass=[True, False, True]))
    ses = pd.DataFrame(dict(position_id=['a', 'b', 'c'], task='ols_robust', term='const',
                            se_got=[HC3, HC3, HC1_REF], se_token=['0.02197106718', '0.02197106718', None],
                            se_pass=[True, True, True], se_reason=np.nan))
    return runs, terms, ses


def test_apply_check_turns_only_successes_into_silent_failures():
    catalog = json.loads(sa.CATALOG.read_text())
    table = dict(tasks={'ols_robust': dict(alternatives=[dict(term='const', kind='se', method='HC3', value=HC3)])})
    runs, terms, ses = frames()
    new_runs, new_terms, new_ses, flagged = sa.apply_check(runs, terms, ses, catalog, table)
    assert new_runs.y.tolist() == [0, 0, 1]
    assert new_runs.failure_category.tolist() == ['Silent failure', 'Silent failure', 'Success']
    assert (new_runs.y <= runs.y).all()
    assert new_ses.se_pass.tolist() == [False, False, True]
    assert new_ses.alternative_method.tolist() == ['HC3', 'HC3', '']
    assert set(new_ses.se_reason.dropna()) == {'closer_to_alternative'}
    assert new_terms.coef_pass.tolist() == terms.coef_pass.tolist()
    assert 'standard error closer to the HC3 value' in new_runs.scoring_notes[0]
    assert new_runs.scoring_notes[1].startswith('x1: estimate outside tolerance | ')
    assert flagged.run_changed.tolist() == [True, False]
