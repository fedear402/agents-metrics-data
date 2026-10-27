"""Reference-output validation: one row per task, every route compared with the stored reference.

Usage (from the repository root):

    uv run --no-sync python code/benchmark/paper_reference_validation.py recompute
    uv run --no-sync python code/benchmark/paper_reference_validation.py build

``recompute`` runs, offline and in a temporary folder, the manual Stata, R, and Python
programs of the original 20 tasks (reference_implementations/manual/), two Stata
variants that set the reference convention, and the supplementary R routes in
code/benchmark/reference_validation_supplement.R. It saves the parsed values in
RECOMPUTED. It needs Stata, R, and the project Python environment. It never writes into
reference_implementations/ or data/.

``build`` makes no external calls. It reads the catalog references, the recomputed
routes, and the stored native-route files listed below. It compares every route value
with the stored reference and writes

    data/analysis/main/data/reference_validation.csv          one row per task
    data/analysis/main/data/reference_validation_routes.csv   one row per route and term
    data/analysis/main/tables/appendix_reference_validation.tex  the table (not in the manuscript)

Stored native-route evidence (read only). A file counts only when its input files have
the same SHA-256 as the catalog input:

    data/task_candidates/2026-09-23_v2/<task>/native_target_validation.json
    data/task_candidates/2026-09-19_six_expansion/<task>/native_target_validation.json
    data/tasks/paper22_2026-09-18/<task>/work/native_validation.json
    data/task_candidates/2026-09-21_welfare_inputs/hard_conditional_logit_welfare/native_validation.json
    catalog field validation (independent Python routes and recorded R and Stata values)

Definitions:

    scored terms     the catalog estimands, plus the standard errors that the primary outcome
                     scores (se_terms of data/analysis/rescore_v2/scoring_spec.json).
    relative diff    |route - reference| / |reference| (absolute difference when the reference is 0).
    independent      a route in another language, or a Python route with another library or
                     algorithm. Reruns of the reference program itself are listed as reproductions.
    convention       a route that uses a different documented convention (for example, a
                     degrees-of-freedom factor) is rescaled by the exact factor when one exists,
                     or else excluded for that term. Every such case is listed in CONVENTIONS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_labels import TASK_LABELS, tex_label  # noqa: E402

CATALOG = ROOT / 'data/analysis/main/data/catalog.json'
SCORING_SPEC = ROOT / 'data/analysis/rescore_v2/scoring_spec.json'
OUT_CSV = ROOT / 'data/analysis/main/data/reference_validation.csv'
OUT_ROUTES = ROOT / 'quality_reports/audits/2026-10-01_reference-validation/work/reference_validation_routes.csv'
OUT_TEX = ROOT / 'data/analysis/main/tables/appendix_reference_validation.tex'
AUDIT = ROOT / 'quality_reports/audits/2026-10-01_reference-validation'
RECOMPUTED = AUDIT / 'work/recomputed_routes.json'
MANUAL = ROOT / 'reference_implementations/manual'
SUPPLEMENT_R = ROOT / 'code/benchmark/reference_validation_supplement.R'
V2 = ROOT / 'data/task_candidates/2026-09-23_v2'
SIX = ROOT / 'data/task_candidates/2026-09-19_six_expansion'
PAPER22 = ROOT / 'data/tasks/paper22_2026-09-18'
WELFARE = ROOT / 'data/task_candidates/2026-09-21_welfare_inputs/hard_conditional_logit_welfare'
STATA = '/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se'
AGREEMENT = 1e-6
LANGUAGES = ('Python', 'R', 'Stata')

# Reference route of each task: language and library or formula.
REFERENCE = {
    'anova_oneway': ('Python', 'SciPy f_oneway'),
    'applied_cumulative_hac': ('Python', 'statsmodels OLS, HAC'),
    'applied_fuzzy_rd': ('Python', 'NumPy WLS, delta method'),
    'applied_multivintage_trade_iv': ('Python', 'NumPy weighted sums, projected 2SLS'),
    'applied_nonrecipient_population_iv': ('Python', 'NumPy interacted weighted 2SLS'),
    'applied_shift_share': ('Python', 'linearmodels IV2SLS, shock level'),
    'arima_forecast': ('Python', 'statsmodels ARIMA'),
    'broad_ar_confidence_set': ('Python', 'closed-form F inversion'),
    'broad_cohort_stack': ('Python', 'pandas double demeaning by stack'),
    'broad_multinomial_logit': ('Python', 'statsmodels MNLogit'),
    'broad_oaxaca': ('Python', 'NumPy least squares by group'),
    'broad_ordered_logit': ('Python', 'statsmodels OrderedModel'),
    'broad_tobit': ('Python', 'statsmodels likelihood model'),
    'chisq_indep': ('Python', 'SciPy chi2_contingency'),
    'cluster_ols': ('Python', 'statsmodels OLS, CR1'),
    'cooks_distance': ('Python', 'NumPy hat matrix'),
    'cox_hazard': ('Python', 'statsmodels PHReg'),
    'hard_conditional_logit_welfare': ('Python', 'SciPy grouped likelihood'),
    'hard_heteroskedastic_probit': ('Python', 'SciPy likelihood, analytic gradient'),
    'hard_panel_lp_iv': ('Python', 'NumPy 2SLS after partialling out'),
    'hard_poisson_hurdle_effects': ('Python', 'statsmodels Logit, SciPy truncated Poisson'),
    'hard_rolling': ('Python', 'NumPy rolling least squares'),
    'hard_truncated_normal_effects': ('Python', 'SciPy likelihood, analytic gradient'),
    'iv2sls_robust': ('Python', 'linearmodels IV2SLS'),
    'logit': ('Python', 'statsmodels Logit'),
    'lp04_ppml': ('Python', 'SciPy separation LP, Newton'),
    'lp11_duration': ('Python', 'SciPy likelihood, polynomial recursion'),
    'lp_irf': ('Python', 'statsmodels OLS, HAC'),
    'ols_robust': ('Python', 'statsmodels OLS, HC1'),
    'pilot_probit_interaction_ame': ('Python', 'statsmodels Probit, analytic effects'),
    'poisson_offset': ('Python', 'statsmodels GLM'),
    'probit': ('Python', 'statsmodels Probit'),
    'rdd_sharp': ('Python', 'NumPy weighted least squares'),
    'staggered_did': ('Python', 'pandas group means'),
    'ttest_welch': ('Python', 'NumPy Welch formula, SciPy t'),
    'two_way_cluster': ('Python', 'NumPy CGM formula'),
    'twoway_fe': ('Python', 'NumPy within transformation, CR1'),
    'var_granger': ('Python', 'statsmodels OLS F test'),
    'var_irf': ('Python', 'statsmodels VAR'),
    'vif': ('Python', 'NumPy auxiliary regressions'),
}

# Library or command of each recomputed route.
MANUAL_METHOD = {
    ('anova_oneway', 'r'): 'aov', ('anova_oneway', 'stata'): 'oneway',
    ('arima_forecast', 'r'): 'forecast::Arima', ('arima_forecast', 'stata'): 'arima',
    ('chisq_indep', 'r'): 'chisq.test', ('chisq_indep', 'stata'): 'tabulate, chi2',
    ('cluster_ols', 'stata'): 'regress, vce(cluster)',
    ('cooks_distance', 'r'): 'cooks.distance', ('cooks_distance', 'stata'): 'predict, cooksd',
    ('cox_hazard', 'r'): 'survival::coxph', ('cox_hazard', 'stata'): 'stcox, breslow',
    ('iv2sls_robust', 'stata'): 'ivregress 2sls, vce(robust)',
    ('logit', 'stata'): 'logit', ('lp_irf', 'r'): 'sandwich::NeweyWest', ('lp_irf', 'stata'): 'newey',
    ('ols_robust', 'stata'): 'regress, vce(robust)', ('poisson_offset', 'stata'): 'poisson, offset()',
    ('probit', 'stata'): 'probit', ('rdd_sharp', 'r'): 'weighted lm', ('rdd_sharp', 'stata'): 'regress [aweight]',
    ('staggered_did', 'r'): 'base R means', ('staggered_did', 'stata'): 'group means',
    ('ttest_welch', 'r'): 't.test', ('ttest_welch', 'stata'): 'ttest, unequal',
    ('two_way_cluster', 'stata'): 'three regress, vce(cluster)',
    ('twoway_fe', 'r'): 'lm dummies, hand CR1', ('twoway_fe', 'stata'): 'areg, vce(cluster)',
    ('var_granger', 'r'): 'lm F test', ('var_granger', 'stata'): 'regress F test',
    ('var_irf', 'r'): 'vars::irf', ('var_irf', 'stata'): 'var, irf',
    ('vif', 'r'): 'auxiliary lm', ('vif', 'stata'): 'auxiliary regress',
}
SUPPLEMENT_METHOD = {
    'ols_robust': 'sandwich::vcovHC HC1', 'cluster_ols': 'sandwich::vcovCL',
    'two_way_cluster': 'sandwich::vcovCL two-way', 'arima_forecast': 'stats::arima',
}
# Stata variants: the manual program with one option that sets the reference convention.
STATA_VARIANTS = {
    'var_irf': ('var y1 y2, lags(1/2)', 'var y1 y2, lags(1/2) dfk', 'var, dfk, irf'),
    'iv2sls_robust': ('vce(robust)', 'vce(robust) small', 'ivregress 2sls, vce(robust) small'),
}

# Python routes in the stored files that rerun the reference program (not independent).
PYTHON_REPRODUCTIONS = {
    ('applied_fuzzy_rd', 'python_delta_joint_covariance'), ('applied_shift_share', 'python_linearmodels'),
    ('broad_ar_confidence_set', 'python_closed_form_quadratic'), ('hard_panel_lp_iv', 'python_fwl'),
}

# Routes that use a different documented convention. A factor rescales the route value
# to the reference convention. None excludes that route and term from the comparison.
_K211 = 'Uses K = 210 in (n - 1)/(n - K). The reference counts K = 211.'
_NW = 'Newey-West with the factor n/(n - k). The reference has no factor.'
_HC = 'Robust SE without the factor n/(n - k). The reference has the factor.'
_ARIMA = 'forecast::Arima innovation variance with divisor n - 3. The reference uses the ML divisor.'
_GLM = 'R glm SE from the expected information. The reference uses the observed information. SE not scored.'
CONVENTIONS = {
    ('twoway_fe', 'manual:r', 'x:se'): (math.sqrt((2000 - 210) / (2000 - 211)), _K211),
    ('twoway_fe', 'manual:stata', 'x:se'): (math.sqrt((2000 - 210) / (2000 - 211)), _K211),
    ('lp_irf', 'manual:r', 'irf_h4:se'): (math.sqrt((245 - 3) / 245), _NW),
    ('lp_irf', 'manual:stata', 'irf_h4:se'): (math.sqrt((245 - 3) / 245), _NW),
    ('iv2sls_robust', 'manual:stata', 'const:se'): (math.sqrt(2000 / 1998), _HC),
    ('iv2sls_robust', 'manual:stata', 'x1_iv:se'): (math.sqrt(2000 / 1998), _HC),
    ('var_irf', 'manual:stata', 'irf_y1_to_y2_h4'): (
        None, 'Residual covariance with divisor T. The reference uses T - 5 (the variant:stata route adds dfk).'),
    ('arima_forecast', 'manual:r', 'forecast:se'): (None, _ARIMA),
    ('arima_forecast', 'manual:r', 'lower_95'): (None, _ARIMA),
    ('arima_forecast', 'manual:r', 'upper_95'): (None, _ARIMA),
    ('probit', 'v2:r', 'const:se'): (None, _GLM),
    ('probit', 'v2:r', 'x1:se'): (None, _GLM),
    ('probit', 'v2:r', 'x2:se'): (None, _GLM),
}

# Recorded explanations for differences above AGREEMENT that remain after conventions.
# build() fails when another task exceeds AGREEMENT without an entry here.
EXPLANATIONS = {
    'cox_hazard': ('The stored reference comes from statsmodels PHReg with method bfgs and its default stop. '
                   'R coxph, Stata stcox, and statsmodels PHReg with method newton agree with each other '
                   'within 1e-11 and lie 3.5e-6 relative from the stored reference.'),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_catalog() -> dict:
    return json.loads(CATALOG.read_text())


def references(entry: dict) -> dict:
    """Return {term: value, 'term:se': se} from one catalog entry."""
    values = {}
    for row in entry['reference']['estimates']:
        values[row['term']] = float(row['estimate'])
        if row.get('se') is not None:
            values[row['term'] + ':se'] = float(row['se'])
    return values


def scored_se_terms() -> dict:
    tasks = json.loads(SCORING_SPEC.read_text())['tasks']
    return {task: sorted(rule['se_terms']) for task, rule in tasks.items() if rule.get('se_scored')}


def catalog_inputs(entry: dict) -> dict:
    """Return {file name: sha256} of the catalog input files of one task."""
    data = Path(entry['data_path'])
    names = entry.get('input_files') or []
    names = [n for n in names if isinstance(n, str)]
    if names:
        return {name: sha256(data.parent / name) for name in names}
    return {'input.csv': sha256(data)}


def language_of(name: str) -> str | None:
    head = name.split('_')[0].lower()
    return {'python': 'Python', 'r': 'R', 'stata': 'Stata', 'statanow': 'Stata'}.get(head)


def method_of(name: str) -> str:
    parts = name.split('_')[1:]
    return ' '.join(parts) if parts else 'native script'


def route(task, source, name, language, method, values, independent=True, path=None):
    return {'task': task, 'route_id': f'{source}:{name}', 'source': source, 'name': name,
            'language': language, 'method': method, 'values': values, 'independent': independent,
            'path': str(path.relative_to(ROOT)) if path else ''}


def _term_key(raw: str, terms: set) -> str | None:
    if raw in terms:
        return raw
    for suffix in ('_se', '__se', ':se'):
        if raw.endswith(suffix) and raw[: -len(suffix)] + ':se' in terms:
            return raw[: -len(suffix)] + ':se'
    return None


def v2_routes(task: str, entry: dict, terms: set) -> list:
    """Routes in the 2026-09-23 v2 package, used only when its inputs equal the catalog inputs."""
    folder = V2 / task
    path = folder / 'native_target_validation.json'
    if not path.exists():
        return []
    for name, digest in catalog_inputs(entry).items():
        candidate = folder / name
        if not candidate.exists() or sha256(candidate) != digest:
            return []
    doc = json.loads(path.read_text())
    out = []
    routes = doc.get('routes')
    if 'checks' in doc:  # one row per language and term
        by_lang: dict = {}
        for check in doc['checks']:
            key = _term_key(check['term'], terms)
            if key is not None:
                by_lang.setdefault(check['language'], {})[key] = float(check['estimate'])
        for lang, values in by_lang.items():
            out.append(route(task, 'v2', lang, language_of(lang), 'native script', values, path=path))
        return out
    if not isinstance(routes, dict):
        return []
    for name, body in routes.items():
        language = language_of(name)
        if language is None:
            continue
        values = {}
        if 'primary_route' in doc or 'errors' in doc:  # {term: {estimate, se}}
            for term, cell in body.items():
                if term in terms and isinstance(cell, dict):
                    values[term] = float(cell['estimate'])
                    if cell.get('se') is not None and term + ':se' in terms:
                        values[term + ':se'] = float(cell['se'])
        elif 'terms' in body:  # {terms: {term: {value}}}
            for raw, cell in body['terms'].items():
                key = _term_key(raw, terms)
                if key is not None:
                    values[key] = float(cell['value'])
        else:  # {values: {...}} or {term: value}
            flat = body.get('values', body)
            for raw, value in flat.items():
                key = _term_key(raw, terms)
                if key is not None and isinstance(value, (int, float)):
                    values[key] = float(value)
        if name in ('python', 'r', 'stata'):
            independent = name != 'python'
            method = 'native script'
        else:
            independent = not (language == 'Python' and ((task, name) in PYTHON_REPRODUCTIONS))
            method = method_of(name)
        out.append(route(task, 'v2', name, language, method, values, independent, path))
    return out


def six_routes(task: str, entry: dict, terms: set) -> list:
    path = SIX / task / 'native_target_validation.json'
    if not path.exists() or Path(entry['data_path']).parent != SIX / task:
        return []
    doc = json.loads(path.read_text())
    out = []
    for lang in ('r', 'stata'):
        values = {k: float(v) for k, v in doc.get(lang, {}).items() if k in terms}
        if values:
            out.append(route(task, 'six', lang, language_of(lang), 'native script', values, path=path))
    return out


def paper22_routes(task: str, entry: dict, terms: set) -> list:
    path = PAPER22 / task / 'work/native_validation.json'
    if not path.exists():
        return []
    doc = json.loads(path.read_text())
    if doc.get('source_input_sha256') != sha256(entry['data_path']):
        return []
    by_lang: dict = {}
    for check in doc['checks']:
        if check['term'] in terms:
            by_lang.setdefault(check['language'], {})[check['term']] = float(check['value'])
    return [route(task, 'paper22', lang, language_of(lang), 'native script', values, path=path)
            for lang, values in by_lang.items()]


def welfare_routes(task: str, entry: dict, terms: set) -> list:
    path = WELFARE / 'native_validation.json'
    if task != 'hard_conditional_logit_welfare' or Path(entry['data_path']).parent != WELFARE:
        return []
    doc = json.loads(path.read_text())
    methods = {'r': 'survival::clogit', 'stata': 'clogit'}
    return [route(task, 'welfare', lang, language_of(lang), methods[lang],
                  {k: float(v) for k, v in body['values'].items() if k in terms}, path=path)
            for lang, body in doc['references'].items()]


def catalog_routes(task: str, entry: dict, terms: set) -> list:
    """Independent routes recorded in the catalog validation field."""
    validation = entry.get('validation') or {}
    out = []
    independent = validation.get('independent')
    if isinstance(independent, dict):
        values = {k: float(v) for k, v in independent.items() if k in terms and isinstance(v, (int, float))}
        details = validation.get('independent_details') or {}
        routes = validation.get('reference_routes') or (validation.get('original_validation') or {}).get('reference_routes')
        method = details.get('route') or validation.get('independent_route') or (routes[1] if routes else 'independent route')
        out.append(route(task, 'catalog', 'python_independent', 'Python', method, values, path=CATALOG))
    for lang, body in (validation.get('cross_language') or {}).items():
        values = {k: float(v) for k, v in body.get('values', {}).items() if k in terms}
        out.append(route(task, 'catalog', lang, language_of(lang), 'native script', values, path=CATALOG))
    return out


def recomputed_routes(task: str, terms: set, recomputed: dict) -> list:
    out = []
    for name, body in recomputed.get(task, {}).items():
        values = {k: v for k, v in body['values'].items() if k in terms}
        source, lang = name.split(':', 1)
        language = language_of(lang)
        if source == 'manual':
            method = MANUAL_METHOD.get((task, lang.split('_')[0]), 'manual program')
            independent = language != 'Python'
            path = MANUAL / task
        elif source == 'variant':
            method = STATA_VARIANTS[task][2]
            independent = True
            path = MANUAL / task
        elif source == 'check':
            method = 'statsmodels PHReg, method newton'
            independent = False
            path = Path(__file__).resolve()
        else:
            method = SUPPLEMENT_METHOD[task]
            independent = True
            path = SUPPLEMENT_R
        rec = route(task, source, lang, language, method, values, independent, path)
        rec['route_id'] = name
        out.append(rec)
    return out


def map_manual(task: str, raw: dict) -> dict:
    """Map the printed keys of a manual program to catalog terms."""
    out = {}
    for key, value in raw.items():
        if isinstance(value, list):
            continue
        if not isinstance(value, (int, float)):
            continue
        if key.startswith('se_'):
            base = key[3:]
            base = {'x_cluster_firm': 'x', 'irf_h4': 'irf_h4', 'treat': 'treat'}.get(base, base)
            out[base + ':se'] = float(value)
        elif key.startswith('coef_'):
            out[key[5:]] = float(value)
        elif key == 'att_uniform':
            out['att'] = float(value)
        elif key == 'se':
            out['forecast:se'] = float(value)
        else:
            out[key] = float(value)
    if task == 'cooks_distance':
        for i, (row, value) in enumerate(zip(raw['top5_obs_ids'], raw['top5_cooks_d']), start=1):
            out[f'rank{i}_obs_id'] = float(row)
            out[f'rank{i}_cooks_d'] = float(value)
    return out


# ---------------------------------------------------------------- recompute
def _patch_python(text: str, data: str) -> str:
    return re.sub(r'^DATA = .*$', f'DATA = Path({data!r})', text, count=1, flags=re.M)


def _patch_r(text: str, data: str) -> str:
    text = re.sub(r'read\.csv\(data_path\)', f'read.csv({json.dumps(data)})', text)
    return text.replace('toJSON(', 'toJSON(digits = NA, ')


def _patch_stata(text: str, data: str) -> str:
    """Read the catalog input in double precision and print 15 significant digits.

    The stored programs import the CSV as float and print 4 to 6 decimals, which
    hides agreement beyond about 1e-7 relative.
    """
    text = text.replace('import delimited "`data_path\'", clear', f'import delimited "{data}", clear asdouble')
    text = re.sub(r'^(clear all\s*)$', r'\1\nset type double', text, count=1, flags=re.M)
    return re.sub(r'%\d+\.\d+[fg]', '%21.15g', text)


def _json_blob(stdout: str) -> dict:
    return json.loads(stdout[stdout.find('{'): stdout.rfind('}') + 1])


def _stata_values(log: str) -> dict:
    lines = log.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('JSON_RESULT:')]
    if not starts:
        return {}
    i = starts[-1]
    text = lines[i]
    while i + 1 < len(lines) and lines[i + 1].startswith('> '):
        i += 1
        text += lines[i][2:]
    number = r'(-?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    values: dict = {key: float(v) for key, v in re.findall(r'"([A-Za-z0-9_]+)":"?\s*' + number, text)}
    for key, inner in re.findall(r'"([A-Za-z0-9_]+)":\[([^\]]*)\]', text):
        values[key] = [float(v.strip().strip('"')) for v in inner.split(',') if v.strip().strip('"')]
    return values


def _run_stata(do_text: str, folder: Path) -> tuple[dict, str | None]:
    (folder / 'route.do').write_text(do_text)
    subprocess.run([STATA, '-b', 'do', 'route.do'], cwd=folder, capture_output=True, text=True, timeout=900)
    log = (folder / 'route.log').read_text(errors='replace') if (folder / 'route.log').exists() else ''
    error = re.search(r'^r\(\d+\);', log, re.M)
    return _stata_values(log), (error.group(0) if error else None)


def cox_newton(data: str) -> dict:
    """Refit the Cox reference model with Newton steps, the reference library, and Breslow ties."""
    import pandas as pd
    from statsmodels.duration.hazard_regression import PHReg

    frame = pd.read_csv(data)
    fit = PHReg(frame['t_obs'], frame[['treat', 'x']], status=frame['event'], ties='breslow').fit(method='newton')
    params, bse = [float(v) for v in fit.params], [float(v) for v in fit.bse]
    return {'treat': params[0], 'x': params[1], 'treat:se': bse[0], 'x:se': bse[1]}


def recompute(out: Path = RECOMPUTED) -> dict:
    catalog = load_catalog()
    results: dict = {}
    for task in sorted(catalog):
        data = catalog[task]['data_path']
        folder = MANUAL / task
        entries: dict = {}
        if folder.is_dir():
            for script in sorted(folder.iterdir()):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    text = script.read_text()
                    if script.name == 'manual_python.py':
                        (tmp / 'route.py').write_text(_patch_python(text, data))
                        proc = subprocess.run(['uv', 'run', '--no-sync', '--project', str(ROOT), 'python', 'route.py'],
                                              cwd=tmp, capture_output=True, text=True, timeout=600)
                        entries['manual:python'] = {'returncode': proc.returncode, 'raw': _json_blob(proc.stdout)}
                    elif script.name == 'manual_r.R':
                        (tmp / 'route.R').write_text(_patch_r(text, data))
                        proc = subprocess.run(['Rscript', 'route.R'], cwd=tmp, capture_output=True, text=True, timeout=600)
                        entries['manual:r'] = {'returncode': proc.returncode, 'raw': _json_blob(proc.stdout)}
                    elif script.name == 'manual_stata.do':
                        values, error = _run_stata(_patch_stata(text, data), tmp)
                        entries['manual:stata'] = {'returncode': 0 if error is None else 1, 'error': error, 'raw': values}
                        if task in STATA_VARIANTS:
                            old, new, _ = STATA_VARIANTS[task]
                            if old not in text:
                                raise ValueError(f'{task}: variant anchor {old!r} not in the manual do file')
                            values, error = _run_stata(_patch_stata(text.replace(old, new), data), tmp)
                            entries['variant:stata'] = {'returncode': 0 if error is None else 1, 'error': error,
                                                        'raw': values}
        if task == 'cox_hazard':
            entries['check:python'] = {'returncode': 0, 'raw': cox_newton(data)}
        if task in SUPPLEMENT_METHOD:
            proc = subprocess.run(['Rscript', str(SUPPLEMENT_R), task, data], capture_output=True, text=True, timeout=600)
            entries['supplement:r'] = {'returncode': proc.returncode, 'raw': _json_blob(proc.stdout)}
        for name, body in entries.items():
            direct = name.startswith(('supplement', 'check'))
            body['values'] = body['raw'] if direct else map_manual(task, body['raw'])
            if body['returncode'] != 0:
                raise RuntimeError(f'{task} {name} failed: {body}')
        if entries:
            results[task] = entries
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'input_sha256': {t: sha256(catalog[t]['data_path']) for t in results},
                               'routes': results}, indent=1) + '\n')
    return results


# ---------------------------------------------------------------- build
def relative(value: float, ref: float) -> float:
    return abs(value - ref) / abs(ref) if ref != 0 else abs(value - ref)


def collect_routes(task: str, entry: dict, terms: set, recomputed: dict) -> list:
    routes = []
    routes += v2_routes(task, entry, terms)
    routes += six_routes(task, entry, terms)
    routes += paper22_routes(task, entry, terms)
    routes += welfare_routes(task, entry, terms)
    routes += catalog_routes(task, entry, terms)
    routes += recomputed_routes(task, terms, recomputed)
    return [r for r in routes if r['values']]


def compare(task: str, entry: dict, se_terms: dict, recomputed: dict) -> tuple[dict, list]:
    ref = references(entry)
    scored = list(entry['estimands']) + [t + ':se' for t in se_terms.get(task, [])]
    terms = set(ref)
    rows = []
    for rec in collect_routes(task, entry, terms, recomputed):
        for term, value in sorted(rec['values'].items()):
            if term not in ref:
                continue
            factor, note = CONVENTIONS.get((task, rec['route_id'], term), (1.0, ''))
            used = factor is not None
            adjusted = value * factor if used else value
            rows.append({
                'task': task, 'route_id': rec['route_id'], 'language': rec['language'], 'method': rec['method'],
                'independent': rec['independent'], 'source_path': rec['path'], 'term': term,
                'scored': term in scored, 'reference': ref[term], 'route_value': value,
                'convention_factor': '' if factor in (None, 1.0) else factor, 'convention_note': note,
                'compared': used, 'abs_diff': abs(adjusted - ref[term]), 'rel_diff': relative(adjusted, ref[term]),
            })
    checks = [r for r in rows if r['independent'] and r['compared'] and r['scored']]
    covered = {r['term'] for r in checks}
    other_language = {r['term'] for r in checks if r['language'] != REFERENCE[task][0]}
    worst = max(checks, key=lambda r: r['rel_diff'], default=None)
    se_checks = [r for r in checks if r['term'].endswith(':se')]
    unscored_se = [r for r in rows if r['independent'] and r['compared'] and not r['scored'] and r['term'].endswith(':se')]
    languages = [lang for lang in LANGUAGES if any(r['language'] == lang for r in checks)]
    routes = sorted({(r['language'], r['method'], r['route_id']) for r in checks})
    missing = [t for t in scored if t not in covered]
    missing_other = [t for t in scored if t not in other_language]
    max_rel = worst['rel_diff'] if worst else None
    if missing:
        status = 'no independent route for ' + ', '.join(missing)
    elif max_rel > AGREEMENT:
        status = 'difference above 1e-6'
    elif missing_other:
        status = 'agree, Python routes only for ' + ', '.join(missing_other)
    else:
        status = 'agree'
    summary = {
        'task': task, 'task_label': TASK_LABELS[task],
        'task_group': 'original' if 'validation' not in entry else 'added',
        'reference_language': REFERENCE[task][0], 'reference_method': REFERENCE[task][1],
        'scored_terms': ';'.join(scored), 'n_scored_terms': len(scored),
        'independent_languages': ', '.join(languages),
        'independent_routes': '; '.join(f'{lang}: {method} [{rid}]' for lang, method, rid in routes),
        'n_independent_route_records': len(routes),
        'terms_without_independent_route': ';'.join(missing),
        'terms_without_other_language_route': ';'.join(missing_other),
        'max_rel_diff_scored': max_rel,
        'max_abs_diff_scored': max((r['abs_diff'] for r in checks), default=None),
        'max_rel_diff_scored_estimates': max((r['rel_diff'] for r in checks if not r['term'].endswith(':se')), default=None),
        'max_rel_diff_scored_se': max((r['rel_diff'] for r in se_checks), default=None),
        'max_rel_diff_unscored_se': max((r['rel_diff'] for r in unscored_se), default=None),
        'worst_term': worst['term'] if worst else '', 'worst_route': worst['route_id'] if worst else '',
        'convention_notes': '; '.join(sorted({f"{r['route_id']} {r['term']}: {r['convention_note']}"
                                              for r in rows if r['convention_note']})),
        'status': status, 'explanation': EXPLANATIONS.get(task, ''),
    }
    return summary, rows


def build_rows(recomputed_path: Path = RECOMPUTED) -> tuple[list, list]:
    catalog = load_catalog()
    se_terms = scored_se_terms()
    recomputed = json.loads(recomputed_path.read_text()) if recomputed_path.exists() else {'routes': {}}
    for task, digest in recomputed.get('input_sha256', {}).items():
        if digest != sha256(catalog[task]['data_path']):
            raise ValueError(f'{task}: recomputed routes used another input; rerun recompute')
    summaries, details = [], []
    for task in sorted(catalog, key=lambda t: TASK_LABELS[t].lower()):
        summary, rows = compare(task, catalog[task], se_terms, recomputed['routes'])
        summaries.append(summary)
        details.extend(rows)
    return summaries, details


def tex_number(value: float | None) -> str:
    if value is None:
        return '--'
    if value == 0:
        return '$0$'
    if value < 1e-15:
        return r'${<}10^{-15}$'
    mantissa, exponent = f'{value:.1e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'


def tex_routes(summary: dict) -> str:
    return summary['independent_languages'] or 'None'


HEADER = (r'Task & Reference route (Python) & \begin{tabular}[b]{@{}l@{}}Independent\\routes\end{tabular} & '
          r'\begin{tabular}[b]{@{}r@{}}Largest relative\\difference\end{tabular} \\')


def render_tex(summaries: list) -> str:
    """Return the appendix longtable. Rows follow the alphabetical order of Table A2."""
    unexplained = [s['task'] for s in summaries
                   if s['status'] == 'difference above 1e-6' and s['task'] not in EXPLANATIONS]
    if unexplained:
        raise ValueError(f'Differences above {AGREEMENT} without a recorded explanation: {unexplained}')
    lines = [
        '% Appendix reference-validation table. Generated by code/benchmark/paper_reference_validation.py.',
        '% Do not edit by hand. Source: data/analysis/main/data/reference_validation.csv.',
        r'\begin{footnotesize}',
        r'\setstretch{1}',
        r'\renewcommand{\arraystretch}{1.1}',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{longtable}{@{}>{\raggedright\arraybackslash}p{4.1cm}>{\raggedright\arraybackslash}p{4.2cm}'
        r'>{\raggedright\arraybackslash}p{2.2cm}r@{}}',
        r'\caption{Validation of the Reference Outputs\label{tab:reference_validation}}\\',
        r'\toprule', HEADER, r'\midrule', r'\endfirsthead',
        r'\multicolumn{4}{c}{\textit{(continued)}} \\',
        r'\toprule', HEADER, r'\midrule', r'\endhead',
    ]
    for i, s in enumerate(summaries):
        end = r' \\*' if i == len(summaries) - 1 else r' \\'
        if s['reference_language'] != 'Python':
            raise ValueError(f"{s['task']}: the table header assumes Python reference routes")
        reference = s['reference_method'].replace('&', r'\&').replace('_', r'\_')
        lines.append(f"{tex_label(s['task'])} & {reference} & {tex_routes(s)} & "
                     f"{tex_number(s['max_rel_diff_scored'])}{end}")
    n_tasks = len(summaries)
    n_other = sum(1 for s in summaries if not s['terms_without_other_language_route'])
    n_below = sum(1 for s in summaries if s['max_rel_diff_scored'] is not None and s['max_rel_diff_scored'] <= AGREEMENT)
    coverage = (f'In all {n_tasks} tasks' if n_other == n_tasks else f'In {n_other} of the {n_tasks} tasks')
    notes = [
        r'\textit{Notes:} The reference route is the Python program that computed the stored reference output.',
        r'An independent route is a program in R or Stata, or a Python program with another library or algorithm.',
        r'The last column reports the largest relative difference between the stored reference and an independent route.',
        r'It covers the scored estimates and, in the nine tasks that score them, the scored standard errors.',
        coverage + r', at least one R or Stata route checks every scored term.',
        r'When a route uses another documented convention, such as a degrees-of-freedom factor, '
        r'the comparison rescales the route value by the exact factor or omits that term.',
        rf'The difference is at most $10^{{-6}}$ in {n_below} tasks.',
    ]
    for s in summaries:
        if s['status'] == 'difference above 1e-6':
            notes.append(rf'For {tex_label(s["task"])}, the stored reference stopped {tex_number(s["max_rel_diff_scored"])} '
                         r'relative from the maximum of the likelihood. '
                         r'R, Stata, and a converged statsmodels fit agree with each other within $10^{-11}$.')
    notes.append(r'The scoring tolerance is $10^{-2}$ relative.')
    lines += [r'\bottomrule',
              r'\multicolumn{4}{@{}p{0.96\linewidth}@{}}{\setstretch{1}' + ' '.join(notes) + '}',
              r'\end{longtable}', r'\end{footnotesize}']
    return '\n'.join(lines) + '\n'


def write_csv(path: Path, rows: list) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if v is None else (repr(v) if isinstance(v, float) else v)) for k, v in row.items()})


def build(out_csv: Path = OUT_CSV, out_routes: Path = OUT_ROUTES, out_tex: Path = OUT_TEX,
          recomputed_path: Path = RECOMPUTED) -> list:
    summaries, details = build_rows(recomputed_path)
    write_csv(out_csv, summaries)
    write_csv(out_routes, details)
    out_tex.write_text(render_tex(summaries))
    return summaries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=['recompute', 'build'])
    args = parser.parse_args(argv)
    if args.command == 'recompute':
        recompute()
    else:
        for s in build():
            print(f"{s['task']:36s} {s['independent_languages']:20s} {s['max_rel_diff_scored']!s:24s} {s['status']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
