"""Build and check the decision versions of the final bank (decision record 2026-09-29_decision-versions).

A decision version <task>__decision uses the data and the reference of the specified task <task>. Its prompt states
the facts of the data and the design and the output conventions. It does not name the scored decision. Source of the
design: quality_reports/audits/2026-09-29_bank-final/out/construct-design.md (Sections 1.2 to 1.4).

Folder: data/task_candidates/2026-09-29_decision/<task>__decision/
    data/                  byte copies of the specified task's data files
    task.json              task_id <task>__decision, task_type "decision", specified_task <task>, the decision text,
                           prompt_levels (zero_shot only where the few-shot example answers the
                           decision), few_shot_note
    reference.json         the specified task's reference.json with the new task_id and specified_task
    scoring.json           the specified task's checks for the kept output rows, and "conventions": the accepted list
    variants.json          the specified task's variants (same data, same values)
    routines.json          the specified task's routines.json with the new task_id
    decision_checks.json   the three checks below, written by build
    prompts/, output_instructions/   rendered by final_bank_prompts.py

The three checks (construct-design Section 1.2):
    fact check   every wrong-by-fact variant fails the scorer, with a worst ratio above FACT_MIN_RATIO
    class check  every legitimate variant inside the correct class passes, under the main references or under one
                 accepted convention; the accepted list has at most MAX_ACCEPTED items
    word check   no prompt of the package matches a word that names the decision
A task whose class check needs a longer list, or whose choice stays a matter of taste, is not built (DROPPED).
The 2026-09-30 redraws of the hand audit (decision record 2026-09-30_hand-audit-fixes) dropped arima_forecast and
ttest_welch: on the new data their legitimate conventions fail the rule. Their specs stay in DROPPED_SPECS, and their
last packages are in data/task_candidates/2026-09-29_dropped/.

Command line (repository root):
    uv run python code/benchmark/decision_versions.py build [--task ID ...]
    uv run python code/benchmark/decision_versions.py check
No provider calls.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_bank_prompts as fp  # noqa: E402
import final_bank_scorer as fs  # noqa: E402

ROOT = fp.ROOT
DECISION_DIR = fp.DECISION_DIRS[0]
SUFFIX = '__decision'
FACT_MIN_RATIO = 2.0   # chosen limit (Federico, 2026-09-29): a wrong-by-fact answer misses by more than 2 tolerances
MAX_ACCEPTED = 2       # chosen limit (Federico, 2026-09-29): at most 2 accepted conventions per task
# zero_shot only: heldout_few_shot left the bank on 2026-09-30 (decision record 2026-09-30_drop-heldout-level).
LEVELS_NO_FEW_SHOT = ['zero_shot']
FEW_SHOT_NOTE = ('The few-shot example of the specified task solves the same calculation and states the decision '
                 '({what}). It would answer the decision version, so this version has no few_shot prompt.')

SPECS: dict[str, dict[str, Any]] = {
    'iv2sls_robust': dict(
        file_descriptions=['input.csv: one row per observation. Columns: id (row identifier), y (outcome), x '
                           '(regressor), z (a variable that shifts x), w (covariate).'],
        task_text=('The equation of interest is y = b0 + b1 x + error. x is correlated with the error of this '
                   'equation. z shifts x, and z is unrelated to the error of this equation. The observations are '
                   'independent, and the error variance may differ across observations. w is not part of the '
                   'equation of interest, and this task does not use w. Estimate b0 and b1 consistently. Report b0 as '
                   'row const and b1 as row x, each with a standard error that is valid under these facts. For every '
                   'row, n_obs is the number of observations used in the estimation.'),
        words=[r'two[- ]stage', r'2sls', r'instrument', r'endogen', r'\biv\b', r'\bhc\d?\b', r'robust',
               r'heteroskedastic', r'sandwich', r'\bwhite\b', r'huber', r'control function'],
        fact=['OLS ignoring endogeneity', '2SLS with homoskedastic SE', '2SLS with cluster SE by blocks of 40 ids',
              'w as an extra instrument (overidentified)', '2SLS with w added as a control'],
        class_=['2SLS with HC0 SE (Stata ivregress vce(robust) without small)', '2SLS with HC2 SEs',
                '2SLS with HC3 SEs'],
        accepted=[], example='two-stage least squares with the excluded instrument and HC1 standard errors'),
    'broad_tobit': dict(
        task_text=('y is weekly hours of work. Desired hours are linear in x and z with an intercept and a normal '
                   'error of constant variance. A person whose desired hours are zero or negative works 0 hours, is '
                   'recorded with y = 0, and stays in the file. A person whose desired hours are positive works '
                   'those hours. Estimate the intercept and the coefficients of x and z of the desired-hours '
                   'equation, and the standard deviation of its error, by maximum likelihood on all rows. Report the '
                   'coefficients as rows const, x, and z, each with its standard error from the inverse of the '
                   'observed information matrix of the joint likelihood. Report the error standard deviation as row '
                   'sigma. For every row, n_obs is the number of observations used in the estimation.'),
        words=[r'tobit', r'censor', r'truncat', r'limited dependent', r'survreg', r'corner'],
        fact=['OLS on all rows', 'OLS on y > 0 only', 'truncated regression on y > 0',
              'Tobit point estimates with OLS standard errors'],
        # The two sigma-factor variants left the specified task on 2026-09-30 (decision record
        # 2026-09-30_hand-audit-fixes): their ratio depends only on n, and the rule cannot resolve them at n = 400.
        class_=['censoring at 0 with y <= 0 rows as censored (same as reference)'],
        accepted=[], example='a Tobit model with lower censoring point 0'),
    'hard_truncated_normal_effects': dict(
        task_text=('The file holds only units whose outcome y is positive. Units with y at or below zero were never '
                   'recorded. No row has y equal to zero, and the file has no information about the missing units. '
                   'In the population, the latent outcome y* is linear in x and z with an intercept and a normal '
                   'error with constant variance, and the recorded y equals y* when y* > 0. Fit this model by '
                   'maximum likelihood on all rows. Report row latent_x with the coefficient on x in the latent '
                   'linear model. Report row sigma with the estimated standard deviation of the latent normal error. '
                   'Report row ame_x with the derivative with respect to x of the expected outcome among recorded '
                   'units, E[y | y > 0, x, z], evaluated at each row\'s own x and z and averaged over all rows. Report '
                   'row prob_positive with the fitted probability that the latent outcome is positive, '
                   'Pr(y* > 0 | x, z), averaged over all rows. Use equal weights. For every row, n_obs is the number '
                   'of rows used in estimation.'),
        words=[r'truncat', r'censor', r'tobit', r'truncreg'],
        fact=['OLS on the truncated sample (all targets from OLS)',
              'Gaussian censored (tobit) model on the supplied rows: no censored rows, so OLS slopes',
              'likelihood without the -log Phi(mu/sigma) term (equals OLS)',
              'prob_positive reported as 1 (every supplied row is positive)'],
        class_=[], accepted=[], example='a normal regression truncated from below at zero'),
    'tobit_twolimit_ame': dict(
        task_text=('Desired weekly hours follow a linear model in an intercept, educ, kids, exper, and nonlabor with a '
                   'normal error of constant variance. A woman whose desired hours are zero or negative does not work '
                   'and reports 0 hours. A woman whose desired hours are positive works those hours, and the survey '
                   'records any value above 60 as 60. Fit the model of desired hours by maximum likelihood on all '
                   'women. Report the average marginal effects of educ and of kids on expected recorded hours '
                   'E[hours | x] (the recorded outcome, which includes the zeros and the values recorded as 60), '
                   'averaged over all women in the sample. Their standard errors use the delta method with the '
                   'inverse of the negative Hessian of the log likelihood at the estimates. Also report the '
                   'coefficient of educ in the desired-hours equation with its standard error from the same inverse '
                   'Hessian, and sigma, the standard deviation of the error of that equation. Report rows named '
                   'ame_educ and ame_kids with the average marginal effects and their standard errors, a row named '
                   'beta_educ with the coefficient and its standard error, and a row named sigma with its estimate '
                   'and a null standard error. For every row, n_obs is the number of women in the estimation sample.'),
        words=[r'tobit', r'censor', r'truncat', r'two-limit', r'lower limit', r'upper limit'],
        fact=['left_censoring_only', 'one_limit_formula', 'ols_all', 'ols_uncensored', 'truncated_mean',
              'latent_coefficients'],
        class_=['sigma_fixed'], accepted=[], example='a Tobit model with a floor at 0 and a top code at 100'),
    'weibull_stock_entry': dict(
        task_text=('The file is a stock sample of unemployment spells. On the survey date, the survey sampled persons '
                   'who were unemployed on that date and then followed each of them for some months. Spells that had '
                   'ended before the survey date could not be sampled. Duration is measured in months from the start '
                   'of the spell. Fit a Weibull proportional hazards model by maximum likelihood on all rows, with an '
                   'intercept, treat, and age: the hazard of leaving unemployment at duration t is '
                   'p t^(p-1) exp(b0 + b_treat treat + b_age age), with shape p > 0. Report rows treat and age with '
                   'the proportional-hazards coefficients b_treat and b_age (log hazard ratios) and their standard '
                   'errors from the inverse of the observed information matrix of the maximum likelihood fit. Report '
                   'a row shape with the Weibull shape p as the estimate and a null standard error. For every row, '
                   'n_obs is the number of spells used.'),
        words=[r'truncat', r'delayed entry', r'late entry', r'at risk from', r'entry onward',
               r'conditional on the spell', r'left[- ]trunc'],
        fact=['ignore_entry', 'ignore_entry_aft', 'clock_reset', 'events_only'],
        class_=[], accepted=[], example='each spell at risk from its elapsed duration on the audit date onward'),
    'applied_fuzzy_rd': dict(
        task_text=('Applicants with an eligibility score at or above the cutoff 0 are more likely to participate in '
                   'the program. The chance of participation jumps at the cutoff, but participation is not automatic '
                   'on either side of the cutoff: some applicants below the cutoff participate, and some applicants '
                   'above it do not. Estimate the effect of participation on earnings for applicants at the cutoff. '
                   'Use local-linear regression on each side of the cutoff with a triangular kernel and bandwidth 1: '
                   'keep the applicants with abs(score) < 1, and treat score >= 0 as the right side. Use the same '
                   'applicants and kernel weights in every regression. Do not apply bias correction. Report the '
                   'effect as row treatment_effect, with its HC0 standard error (no degrees-of-freedom correction, '
                   'bandwidth and kernel weights treated as fixed). For every row, n_obs is the number of applicants '
                   'inside the bandwidth.'),
        output_rows=['treatment_effect'],
        words=[r'fuzzy', r'two[- ]stage', r'2sls', r'instrument', r'\bwald\b', r'\bratio\b', r'first[_ ]stage',
               r'reduced[_ ]form', r'\biv\b', r'discontinuit'],
        fact=['reduced_form_as_effect', 'naive_wls_on_participation', 'ignoring_covariance_se', 'N2_se_rf_over_fs'],
        class_=['hc1_se_k4', 'N1_cluster_by_id_factor', 'strict_cutoff_inclusive_edge'], accepted=[],
        example='the two discontinuities and the effect from weighted two-stage least squares'),
    'hard_panel_lp_iv': dict(
        file_descriptions=['input.csv: one row per observed unit-period. Columns: unit (unit identifier), period '
                           '(time period, integer), y (outcome), x (endogenous regressor), z (excluded instrument).'],
        task_text=('input.csv is an unbalanced panel: some unit-periods are missing, and the rows are not sorted. '
                   'Estimate the responses of y at horizons h = 0, 1, 2 to x with two-stage least squares local '
                   'projections. For horizon h and origin (unit i, period t), the outcome is y of unit i h periods '
                   'after period t. The included exogenous regressors are y of unit i one period before period t, '
                   'unit fixed effects, and period fixed effects for the origin period t. Include an intercept and '
                   'omit one category from each fixed-effect set. x at (i, t) is the only endogenous regressor, and z '
                   'at (i, t) is its only excluded instrument. Use one common sample for all three horizons: the '
                   'origins whose unit is observed in periods t-1, t, t+1, and t+2. Report the x coefficients as rows '
                   'lp_h0, lp_h1, and lp_h2. Report the number of origins in the common sample as row n_origins. No '
                   'row has a standard error. For every row, n_obs is the number of origins in the common sample.'),
        words=[r'calendar', r'adjacent', r'stored rows', r'row shift', r'shift\('],
        fact=['row_shift_leads_and_lag', 'lag_row_shift_leads_calendar', 'leads_row_shift_lag_calendar',
              'keep_then_operators'],
        class_=['target_period_fe', 'long_difference_with_lag'], accepted=[],
        example='leads and the lag of output built by quarter for firms observed at quarters t-1 through t+2'),
}

# Specs of decision versions that the 2026-09-30 redraws dropped (see DROPPED). Kept for the record and for
# the neutral output ids; the command line builds and checks only SPECS.
DROPPED_SPECS: dict[str, dict[str, Any]] = {
    'arima_forecast': dict(
        task_text=('The expected change of y from one period to the next is not zero. Fit an ARIMA(1,1,1) model to '
                   'the level series y by exact maximum likelihood. Use the maximum-likelihood estimate of the '
                   'innovation variance. Report the one-step-ahead forecast of the level of y for the period after '
                   'the last observation as row forecast, with its forecast standard error as the se of that row. '
                   'Report the bounds of the 95 percent prediction interval, based on the normal distribution, as '
                   'rows lower_95 and upper_95. For every row, n_obs is the number of observations of y used in '
                   'estimation.'),
        words=[r'drift', r'constant', r'trend', r'intercept', r'nonzero mean', r'non-zero mean'],
        fact=['no_drift', 'r_stats_arima_default_no_drift', 'r_forecast_Arima_default_no_drift'],
        class_=['correct_sarimax_on_difference_constant', 'css_with_drift_r', 'interval_t_quantile_ml_variance',
                'correct_df_variance_forecast_pkg_convention'],
        accepted=[], example='an ARIMA(1,1,1) model with a drift term'),
    'ttest_welch': dict(
        task_text=('Test whether the mean of y differs between group 1 and group 0. The two groups are independent '
                   'samples of different sizes, and y is much more dispersed in group 1 than in group 0. Use a '
                   'two-sample t test that is valid under these facts. Report the t statistic for the difference '
                   'defined as the mean of group 1 minus the mean of group 0 as row t, and the degrees of freedom of '
                   'the test as row df. For every row, n_obs is the number of rows used in the test.'),
        words=[r'welch', r'satterthwaite', r'equal variance', r'unequal', r'pooled', r'heteroskedastic'],
        fact=['pooled_student_test', 'pooled_t_with_satterthwaite_df', 'welch_t_with_df_n_minus_2',
              'pooled_student_sign_flipped'],
        class_=['stata_welch_option_1947_df', 'df_rounded_to_integer', 'df_floor_integer',
                'welch_t_with_df_min_n_minus_1'],
        accepted=['welch_t_with_df_min_n_minus_1'],
        example='a two-sample t test without equal variances and with Welch-Satterthwaite degrees of freedom'),
}

DROPPED = {
    'ols_robust': ('Class check fails. After the 2026-09-29 redraw, HC0 (2.53), HC2 (5.95), and HC3 (15.3) fail '
                   'against the HC1 reference. The facts (independent errors, variance that grows with x1) select the '
                   'robust class but no member of it. The accepted list would need 3 items, and a sentence that '
                   'fixes HC1 names the decision.'),
    'cluster_ols': ('Class check fails. CR0 (1.85), the factor (N-1)/(N-K) only (1.68), CR2 (5.34), and CR3 (13.3) '
                    'fail against the CR1 reference with 30 groups. The facts select clustering by group but no '
                    'member of the cluster-robust class. The accepted list would need 4 items, and a sentence that '
                    'fixes CR1 names the grouping and the decision.'),
    'arima_forecast': ('Class check fails after the 2026-09-30 redraw (level Y0 -167.8, decision record '
                       '2026-09-30_hand-audit-fixes). css_with_drift_r (4.13), interval_t_quantile_ml_variance (6.65), '
                       'and correct_df_variance_forecast_pkg_convention (8.17) fail against the reference. The facts '
                       'do not select one of these conventions, so the accepted list would need 3 items.'),
    'ttest_welch': ('Class check fails after the 2026-09-30 redraw (12 and 30 rows, df 14.5, decision record '
                    '2026-09-30_hand-audit-fixes). stata_welch_option_1947_df (4.28), df_rounded_to_integer (3.44), '
                    'and df_floor_integer (3.45) fail, and welch_t_with_df_min_n_minus_1 (24.1) is already on the '
                    'accepted list. The accepted list would need 4 items.'),
}
# The neutral output ids keep the order of the nine versions of 2026-09-29, so a drop does not renumber the others.
OUTPUT_ORDER = ['iv2sls_robust', 'arima_forecast', 'broad_tobit', 'hard_truncated_normal_effects',
                'tobit_twolimit_ame', 'weibull_stock_entry', 'applied_fuzzy_rd', 'hard_panel_lp_iv', 'ttest_welch']


def decision_id(task: str) -> str:
    return task + SUFFIX


def output_task_id(task: str) -> str:
    """The task_id that the output format shows: neutral, because <task>__decision names the method."""
    return f'task_d{OUTPUT_ORDER.index(task) + 1:02d}'


def specified_dir(task: str) -> Path:
    folder = fp.package_dir(task)
    if folder.parent in fp.DECISION_DIRS:
        raise ValueError(f'{task} is a decision version')
    return folder


def _variant_rows(values: dict) -> dict:
    return {term: dict(row) for term, row in (values or {}).items()}


def worst_ratio(rows: dict, spec: dict) -> float:
    """Largest error in tolerances over the checks of spec, against the main references."""
    worst = 0.0
    for check in spec['checks']:
        value = (rows.get(check['term']) or {}).get(check['field'])
        if check['kind'] == 'integer':
            ok = fs._finite(value) and float(value) == float(check['reference'])
            worst = max(worst, 0.0 if ok else float('inf'))
        else:
            worst = max(worst, fs.ratio(value, check['reference'], spec['rel_tol'], spec['abs_tol']))
    return worst


def conventions_for(names: list[str], variants: dict, spec: dict) -> list[dict]:
    out = []
    for name in names:
        values = variants[name]['values']
        refs = [dict(term=c['term'], field=c['field'], reference=values[c['term']][c['field']])
                for c in spec['checks']]
        out.append(dict(name=name, description=variants[name].get('description', name), references=refs,
                        source='decision record 2026-09-29_decision-versions (accepted list)'))
    return out


def build(task: str) -> Path:
    spec_d = SPECS[task]
    src = specified_dir(task)
    dst = DECISION_DIR / decision_id(task)
    if dst.exists():
        if dst.parent != DECISION_DIR:
            raise ValueError(f'refusing to remove {dst}')
        shutil.rmtree(dst)
    (dst / 'data').mkdir(parents=True)
    src_task = json.loads((src / 'task.json').read_text())
    for name in src_task['files']:
        shutil.copyfile(src / 'data' / name, dst / 'data' / name)
    keep = spec_d.get('output_rows')
    rows = [r for r in src_task['output_rows'] if keep is None or r['term'] in keep]
    targets = [t for t in src_task.get('targets', []) if keep is None or t['term'] in keep]
    task_json = dict(
        task_id=decision_id(task), task_type='decision', specified_task=task,
        output_task_id=output_task_id(task),
        specified_folder=str(src.relative_to(ROOT)),
        source_of_difficulty=src_task.get('source_of_difficulty') or src_task.get('decision'),
        files=src_task['files'],
        file_descriptions=spec_d.get('file_descriptions') or src_task['file_descriptions'],
        task_text=spec_d['task_text'], output_rows=rows,
        scoring=src_task.get('scoring'),
        prompt_levels=LEVELS_NO_FEW_SHOT,
        few_shot_note=FEW_SHOT_NOTE.format(what=spec_d['example']),
        decision_words=spec_d['words'],
        decision_record='quality_reports/decision_records/2026-09-29_decision-versions.md')
    if targets:
        task_json['targets'] = targets
    if src_task.get('title'):
        task_json['title'] = src_task['title'] + ' (decision version)'
    fp_save(dst / 'task.json', task_json)
    reference = json.loads((src / 'reference.json').read_text())
    reference.update(task_id=decision_id(task), specified_task=task)
    fp_save(dst / 'reference.json', reference)
    variants = json.loads((src / 'variants.json').read_text())
    fp_save(dst / 'variants.json', variants)
    routines = json.loads((src / 'routines.json').read_text())
    routines['task_id'] = decision_id(task)
    fp_save(dst / 'routines.json', routines)
    scoring = json.loads((src / 'scoring.json').read_text())
    scoring['task_id'] = decision_id(task)
    scoring['specified_task'] = task
    if keep is not None:
        scoring['scored'] = [c for c in scoring['scored'] if c['term'] in keep]
        scoring['diagnostic'] = [c for c in scoring.get('diagnostic', []) if c.get('term') in keep]
    spec = fs.load_spec(scoring)
    scoring['conventions'] = conventions_for(spec_d['accepted'], variants, spec)
    fp_save(dst / 'scoring.json', scoring)
    fp.write_package(dst)
    fp_save(dst / 'decision_checks.json', checks(task))
    return dst


def fp_save(path: Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(fs.jsonable(value), indent=2) + '\n')


def checks(task: str) -> dict:
    """The fact, class, and word checks of one decision version (its package must exist)."""
    spec_d = SPECS[task]
    dst = DECISION_DIR / decision_id(task)
    spec = fs.load_spec(dst)
    variants = json.loads((dst / 'variants.json').read_text())
    fact = []
    for name in spec_d['fact']:
        rows = _variant_rows(variants[name]['values'])
        res = fs.score_rows(rows, spec)
        ratio = worst_ratio(rows, spec)
        fact.append(dict(variant=name, passed_scorer=res['passed'], worst_ratio=ratio,
                         ok=not res['passed'] and ratio > FACT_MIN_RATIO))
    klass = []
    for name in spec_d['class_']:
        rows = _variant_rows(variants[name]['values'])
        res = fs.score_rows(rows, spec)
        klass.append(dict(variant=name, passed_scorer=res['passed'], convention=res['convention'],
                          worst_ratio_main=worst_ratio(rows, spec),
                          ok=res['passed'] and (res['convention'] == 'main' or res['convention'] in spec_d['accepted'])))
    words = [re.compile(w, re.I) for w in spec_d['words']]
    hits = []
    for path in sorted((dst / 'prompts').glob('*.txt')):
        text = path.read_text()
        hits += [dict(prompt=path.name, word=m.group(0)) for w in words for m in [w.search(text)] if m]
    src_text = fp.task_information(fp.load_task(specified_dir(task)))
    specified_hits = sorted({m.group(0).lower() for w in words for m in w.finditer(src_text)})
    reference_rows = json.loads((dst / 'reference.json').read_text()).get('route_a_values') or {}
    same_data = all((dst / 'data' / n).read_bytes() == (specified_dir(task) / 'data' / n).read_bytes()
                    for n in fp.load_task(dst)['files'])
    out = dict(
        task_id=decision_id(task), specified_task=task, same_data=same_data,
        reference_passes=fs.score_rows(reference_rows, spec)['passed'],
        fact_check=dict(min_ratio=FACT_MIN_RATIO, variants=fact, ok=bool(fact) and all(f['ok'] for f in fact)),
        class_check=dict(max_accepted=MAX_ACCEPTED, accepted=spec_d['accepted'], variants=klass,
                         ok=len(spec_d['accepted']) <= MAX_ACCEPTED and all(c['ok'] for c in klass)),
        word_check=dict(words=spec_d['words'], hits=hits, specified_prompt_hits=specified_hits, ok=not hits),
        prompt_levels=LEVELS_NO_FEW_SHOT)
    out['ok'] = (out['same_data'] and out['reference_passes'] and out['fact_check']['ok']
                 and out['class_check']['ok'] and out['word_check']['ok'])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--task', action='append')
    sub.add_parser('check')
    args = parser.parse_args(argv)
    tasks = args.task if args.command == 'build' and args.task else list(SPECS)
    bad = []
    for task in tasks:
        if args.command == 'build':
            build(task)
        result = checks(task)
        stored = DECISION_DIR / decision_id(task) / 'decision_checks.json'
        if args.command == 'check' and json.loads(stored.read_text()) != json.loads(json.dumps(fs.jsonable(result))):
            bad.append(f'{task}: decision_checks.json is stale')
        if not result['ok']:
            bad.append(f'{task}: checks fail')
        print(f"{decision_id(task)}: fact {result['fact_check']['ok']} class {result['class_check']['ok']} "
              f"word {result['word_check']['ok']} same_data {result['same_data']}")
    print(f'{len(tasks)} decision versions, {len(bad)} problems' + ''.join(f'\n  {p}' for p in bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
