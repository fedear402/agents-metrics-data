"""Tests of the standard-routine lists (routines.json) and the prompt source of the final task bank."""
from __future__ import annotations

import filecmp
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_bank_prompts as fp  # noqa: E402

HARD = ROOT / 'data/task_candidates/2026-09-28_final_hard'
BUILD = HARD / '_build'
ENV_MANIFEST = HERE / 'collection_environment/environment_manifest.json'


def routines(task: str) -> dict:
    return json.loads((fp.package_dir(task) / 'routines.json').read_text())['languages']


def matches(task: str, lang: str, code: str) -> list[str]:
    return [label for pattern, label in routines(task)[lang]['patterns'].items() if re.search(pattern, code)]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('folder', fp.package_dirs(), ids=lambda p: p.name)
def test_routine_entries_are_consistent(folder):
    data = json.loads((folder / 'routines.json').read_text())
    assert set(data['languages']) == set(fp.LANGS)
    for lang, entry in data['languages'].items():
        for pattern in entry.get('patterns') or {}:
            re.compile(pattern)
        # A language has a standard routine exactly when it has detection patterns.
        assert (entry.get('routine') is None) == (not entry.get('patterns')), (lang, entry)
        if entry.get('routine') is None:
            assert entry.get('covers') in ([], None), (lang, entry)


def test_notes_do_not_call_installed_packages_missing():
    manifest = ENV_MANIFEST.read_text()
    installed = [name for name in ('clubSandwich', 'ivmodel', 'gmm', 'momentfit', 'censReg', 'weakiv', 'twostepweakiv')
                 if f'"{name}"' in manifest or name in manifest]
    assert {'clubSandwich', 'ivmodel', 'gmm', 'momentfit', 'censReg', 'weakiv'} <= set(installed)
    for folder in fp.package_dirs():
        text = (folder / 'routines.json').read_text()
        for name in installed:
            assert not re.search(rf'\b{name}\b[^."]*\b(is|are) not installed', text), (folder.name, name)


@pytest.mark.parametrize('task', sorted(p.name for p in HARD.iterdir() if (p / 'route_a.py').is_file()))
def test_hard_routines_json_equals_the_build_source(task):
    """build.py writes routines.json from route_a.ROUTINES, so a rebuild keeps the file."""
    module = load(HARD / task / 'route_a.py', f'route_a_{task}')
    assert json.loads(json.dumps(module.ROUTINES)) == routines(task)


def test_new_r_routines_are_listed():
    assert matches('cr2_bm_df', 'r', 'coef_test(fit, vcov = "CR2", cluster = d$school, test = "Satterthwaite")')
    assert routines('cr2_bm_df')['r']['covers'] == 'all'
    assert matches('ar_weakiv_invert', 'r', 'm <- ivmodel(Y = d$y, D = d$x, Z = Z, X = W); AR.test(m)')
    assert routines('ar_weakiv_invert')['r']['covers'] == 'all'
    assert matches('liml_many_iv', 'r', 'LIML(ivmodel(Y = y, D = x, Z = Z, X = W))')
    assert matches('gmm_twostep_j', 'r', 'library(gmm)\ng <- gmm(y ~ x + w, ~ w + z1 + z2 + z3 + z4)')
    assert matches('generated_regressor_mt', 'r', 'library(momentfit)')
    assert matches('tobit_twolimit_ame', 'r', 'fit <- censReg(hours ~ educ, left = 0, right = 60, data = d)')
    assert matches('ar_weakiv_invert', 'stata', 'ivregress 2sls y w1 w2 (x = z1-z4)\nweakiv')


def test_ar_weakiv_python_pattern_matches_statsmodels_only():
    assert matches('ar_weakiv_invert', 'python', 'fit.f_test(R)') == ['statsmodels f_test or compare_f_test']
    assert matches('ar_weakiv_invert', 'python', 'u.compare_f_test(r)[0]')
    assert matches('ar_weakiv_invert', 'python', 'res.anderson_rubin') == []


@pytest.mark.parametrize('code,expected', [
    ('smf.ols("y ~ x + C(firm) + C(year)", d)', True),
    ('smf.ols("y ~ x + C(firm_id) + C(period)", data=df)', True),
    ('smf.ols("y ~ x + C( firm ) + C( year )", d)', True),
    ('smf.ols("y ~ x + C(firm) + year", d)', False),
    ('smf.ols("y ~ x", d)', False),
])
def test_twoway_fe_python_pattern_needs_two_categorical_terms(code, expected):
    labels = matches('twoway_fe', 'python', code)
    assert ('statsmodels ols with firm and year dummies' in labels) is expected


@pytest.mark.parametrize('code,label', [
    ('from scipy import stats\nstats.ttest_ind(a, b, equal_var=False)', 'scipy.stats.ttest_ind'),
    ('from scipy.stats import norm, ttest_ind\nttest_ind(a, b, equal_var=False)', 'scipy.stats.ttest_ind'),
    ('scipy.stats.ttest_ind(a, b, equal_var=False)', 'scipy.stats.ttest_ind'),
    ('from statsmodels.stats.weightstats import ttest_ind\nttest_ind(a, b, usevar="unequal")', 'statsmodels ttest_ind'),
    ('import statsmodels.stats.api as sms\nsms.CompareMeans.from_data(a, b)', 'statsmodels ttest_ind'),
])
def test_ttest_welch_python_labels_separate_scipy_and_statsmodels(code, label):
    assert matches('ttest_welch', 'python', code) == [label]


def test_build_writes_prompts_only_through_the_renderer():
    source = (BUILD / 'build.py').read_text()
    assert 'final_bank_prompts.write_package(folder)' in source
    kit_source = (BUILD / 'kit.py').read_text()
    for stale in ('def build_prompt', 'The data are synthetic', 'LANG_LINE', 'shutil.rmtree(folder'):
        assert stale not in kit_source, stale


def test_kit_write_prompts_equals_the_renderer(tmp_path):
    sys.path.insert(0, str(BUILD))
    try:
        kit = load(BUILD / 'kit.py', 'final_hard_kit')
    finally:
        sys.path.remove(str(BUILD))
    copy = tmp_path / 'ar_weakiv_invert'
    shutil.copytree(HARD / 'ar_weakiv_invert', copy, ignore=shutil.ignore_patterns('prompts'))
    task = json.loads((copy / 'task.json').read_text())
    written = kit.write_prompts(task, copy)
    assert len(written) == 18  # 3 languages x 2 prompt levels x 3 conditions (decision record 2026-09-30_drop-heldout-level)
    compare = filecmp.dircmp(copy / 'prompts', HARD / 'ar_weakiv_invert/prompts')
    assert compare.left_only == compare.right_only == compare.diff_files == []
    with pytest.raises(ValueError):
        kit.write_prompts(dict(task, task_text='changed'), copy)


def test_verify_checks_prompts_against_the_renderer():
    sys.path.insert(0, str(BUILD))
    try:
        verify = load(BUILD / 'verify.py', 'final_hard_verify')
    finally:
        sys.path.remove(str(BUILD))
    folder = HARD / 'cr2_bm_df'
    result = verify.check_prompts(folder, json.loads((folder / 'task.json').read_text()))
    assert result == dict(files=18, identical_information_block=True, problems=[])


# Routine classes and the two packages installed on 2026-09-29 (decision record 2026-09-29_routine-classes).
ROUTINE_CLASSES = {'builtin', 'major_package', 'method_author_package', 'small_package', 'generic', 'none'}
UNNAMED = re.compile(r'\b(py4etrics|ivmodels)\b')


@pytest.mark.parametrize('folder', fp.package_dirs(), ids=lambda p: p.name)
def test_every_routine_has_a_class(folder):
    for lang, entry in json.loads((folder / 'routines.json').read_text())['languages'].items():
        assert entry['routine_class'] in ROUTINE_CLASSES, (lang, entry)
        labels = set((entry.get('patterns') or {}).values())
        assert set(entry['pattern_classes']) == labels, (lang, entry)
        assert set(entry['pattern_classes'].values()) <= ROUTINE_CLASSES - {'none'}
        assert (entry['routine_class'] == 'none') == (entry.get('routine') is None), (lang, entry)
        if entry['routine_class'] == 'method_author_package':
            assert entry['routine_citation'], (lang, entry)
        for label, cls in entry['pattern_classes'].items():
            if cls == 'method_author_package':
                assert entry['citations'][label], (lang, label)


def test_new_python_packages_are_detected_and_classified():
    heck = routines('heckman_twostep')['python']
    assert heck['routine_class'] == 'small_package' and heck['covers'] == 'all'
    assert matches('heckman_twostep', 'python', 'from py4etrics.heckit import Heckit\nres = Heckit(y, X, Z).fit()')
    assert matches('heckman_twostep', 'python', 'res = Heckit(y, X, Z).fit()') == []
    assert matches('tobit_twolimit_ame', 'python', 'from py4etrics.tobit import Tobit\nTobit(y, X, cens=c, left=0)')
    assert routines('tobit_twolimit_ame')['python']['routine_class'] == 'small_package'
    assert matches('broad_tobit', 'python', 'import py4etrics.tobit as pt\npt.Tobit(y, X, cens=c, left=0)')
    assert matches('hard_truncated_normal_effects', 'python', 'from py4etrics.truncreg import Truncreg')
    ar = routines('ar_weakiv_invert')['python']
    assert ar['routine_class'] == 'method_author_package' and ar['covers'] == 'all'
    assert 'Journal of Econometrics, 106239' in ar['routine_citation']
    code = 'from ivmodels.tests import inverse_anderson_rubin_test\ninverse_anderson_rubin_test(Z, X, y, C=W)'
    assert 'ivmodels Anderson-Rubin test' in matches('ar_weakiv_invert', 'python', code)
    assert matches('liml_many_iv', 'python', 'from ivmodels import KClass\nKClass(kappa="liml").fit(X, y, Z=Z, C=C)')
    assert routines('liml_many_iv')['r']['covers'] == 'all'
    assert routines('cr2_bm_df')['python']['routine_class'] == 'none'
    assert routines('generated_regressor_mt')['python']['routine_class'] == 'generic'
    assert matches('ar_weakiv_invert', 'stata', 'ivregress 2sls y w1 w2 (x = z1-z4), small\nestat weakrobust, ar')


def test_no_prompt_or_example_names_the_unnamed_packages():
    for folder in fp.package_dirs():
        for path in list((folder / 'prompts').rglob('*')) + list((folder / 'examples').glob('*')):
            if path.is_file() and path.suffix in ('.md', '.txt', '.json', '.py', '.R', '.do'):
                if path.name.startswith(('expected_', 'validation')):
                    continue
                assert not UNNAMED.search(path.read_text()), path
        assert not UNNAMED.search((folder / 'task.json').read_text()), folder.name


def test_environment_installs_the_unnamed_packages_with_statsmodels_0_14():
    spec = json.loads((HERE / 'collection_environment/packages.json').read_text())
    assert {'py4etrics', 'ivmodels'} <= set(spec['python']['imports'])
    assert spec['must_not_change']['python']['statsmodels'] == '0.14.6'
    manifest = json.loads(ENV_MANIFEST.read_text())
    checks = manifest['python']['checks']
    assert checks['py4etrics']['ok'] and checks['py4etrics']['version'] == '0.1.9'
    assert checks['ivmodels']['ok'] and checks['ivmodels']['version'] == '0.10.0'
    assert manifest['python']['distributions']['statsmodels'] == '0.14.6'
    pyproject = (ROOT / 'pyproject.toml').read_text()
    assert '"py4etrics==0.1.9"' in pyproject and '"ivmodels==0.10.0"' in pyproject
