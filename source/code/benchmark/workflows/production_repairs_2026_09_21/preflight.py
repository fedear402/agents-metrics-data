"""Check unchanged references and examples in Python, R, and Stata."""
import ast
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.optimize import linprog
from linearmodels.iv import IV2SLS
import prepare as p

HERE = Path(__file__).resolve().parent
WORK = p.META / 'preflight'

def run(argv, name, marker=None):
    result = subprocess.run(argv, cwd=WORK, capture_output=True, text=True)
    p.c.save(WORK / (name + '.json'), dict(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    assert result.returncode == 0, result.stderr
    if marker:
        assert marker in result.stdout
    return result

def example(task):
    source = p.ROOT / 'data/task_candidates/2026-09-19_revised' / task / 'example.py'
    stream = io.StringIO()
    namespace = {'__name__': '__main__'}
    with contextlib.redirect_stdout(stream):
        exec(compile(source.read_text(), str(source), 'exec'), namespace)
    expected = ast.literal_eval(source.with_name('example_output.txt').read_text().strip())
    actual = ast.literal_eval(stream.getvalue().strip())
    assert actual.keys() == expected.keys()
    assert all(np.isclose(actual[key], expected[key], rtol=1e-9, atol=1e-9) for key in actual)
    return namespace['d'].copy(), dict(path=str(source), sha256=p.c.sha(source), expected=expected, actual=actual)

def ppml_design(d, kind):
    groups = ['pair', 'exporter_year', 'importer_year'] if kind == 'target' else ['pair', 'time']
    Z = np.column_stack([pd.get_dummies(d[g], dtype=float) for g in groups])
    X = np.column_stack([d.agreement, Z])
    positive = d.trade.to_numpy() > 0
    separated = np.zeros(len(d), dtype=bool)
    for i in np.flatnonzero(~positive):
        result = linprog(X[i], A_ub=X[~positive], b_ub=np.zeros((~positive).sum()),
            A_eq=X[positive], b_eq=np.zeros(positive.sum()), bounds=[(-1,1)]*X.shape[1], method='highs')
        assert result.success
        separated[i] = result.fun < -1e-7
    Z = Z[~separated]
    _, _, piv = qr(Z, mode='economic', pivoting=True)
    Z = Z[:, piv[:np.linalg.matrix_rank(Z)]]
    X = np.column_stack([d.agreement.to_numpy()[~separated], Z])
    assert np.linalg.matrix_rank(X) == X.shape[1]
    frame = d.loc[~separated, ['trade','pair']].reset_index(drop=True)
    frame['row_id'] = np.flatnonzero(~separated)
    frame = pd.concat([frame, pd.DataFrame(X, columns=[f'v{i}' for i in range(X.shape[1])])], axis=1)
    frame.to_csv(WORK / f'ppml_design_{kind}.csv', index=False)
    return dict(retained=len(frame), separated=int(separated.sum()), rank=X.shape[1], clusters=frame.pair.nunique())

def main():
    WORK.mkdir(parents=True, exist_ok=True)
    rows = p.source_rows()
    _, specs, _ = p.source_material(rows)
    evidence = dict(passed=False, examples={}, python={}, native={}, input_hashes={})
    for task, short in zip(p.TASKS, ['lp','ppml']):
        source = p.c.task_input_files(specs[task])['input.csv']
        data = pd.read_csv(source)
        data.to_csv(WORK / f'{short}_target.csv', index=False)
        evidence['input_hashes'][task] = p.c.sha(source)
        example_data, receipt = example(task)
        example_data.to_csv(WORK / f'{short}_example.csv', index=False)
        evidence['examples'][task] = receipt
    d = pd.read_csv(WORK / 'lp_target.csv')
    assert list(d.columns) == ['unit','period','y','x','z']
    sample = d.copy()
    for h in [-1,1,2]:
        q = d[['unit','period','y']].copy()
        q.period -= h
        sample = sample.merge(q.rename(columns={'y':f'y{h}'}), on=['unit','period'], validate='one_to_one')
    C = pd.concat([pd.Series(1., index=sample.index, name='intercept'), sample['y-1'].rename('lag_y'),
        pd.get_dummies(sample.unit,prefix='unit',drop_first=True,dtype=float),
        pd.get_dummies(sample.period,prefix='period',drop_first=True,dtype=float)],axis=1)
    lp = {f'lp_h{h}':float(IV2SLS(sample['y' if h==0 else f'y{h}'],C,sample.x,sample.z).fit().params['x']) for h in range(3)}
    evidence['python']['hard_panel_lp_iv'] = dict(estimates=lp, rows=len(sample), route='Calendar merges and linearmodels IV2SLS')
    source = p.ROOT / 'data/task_candidates/2026-09-19/work/applied/references.py'
    spec = importlib.util.spec_from_file_location('ppml_reference_preflight', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result, details = module.independent('lp04_ppml', Path(p.c.task_input_files(specs['lp04_ppml'])['input.csv']).parent)
    evidence['python']['lp04_ppml'] = dict(estimates=result, details=details, reference_check_source=str(source), source_sha256=p.c.sha(source))
    expected = {}
    for task, short in zip(p.TASKS,['lp','ppml']):
        expected[(short,'target')] = {x['term']: x['estimate'] for x in specs[task]['reference']['estimates']}
        actual = evidence['python'][task]['estimates']
        for term, value in expected[(short,'target')].items():
            assert np.isclose(actual[term],value,rtol=1e-7,atol=1e-8), (task,term,actual[term],value)
    expected[('lp','example')] = evidence['examples'][p.TASKS[0]]['expected']
    ex = evidence['examples'][p.TASKS[1]]['expected']
    expected[('ppml','example')] = dict(agreement_coefficient=ex['coefficient'],proportional_mean_change=np.expm1(ex['coefficient']),
        pair_cluster_se=ex['cluster_se'],retained_observations=ex['retained'],separated_observations=5,retained_zero_observations=ex['ordinary_zeros'])
    for kind in ['target','example']:
        evidence['python'][f'ppml_design_{kind}'] = ppml_design(pd.read_csv(WORK/f'ppml_{kind}.csv'),kind)
    run(['Rscript','--vanilla',str(HERE/'preflight.R'),str(WORK)],'r_process','R_PREFLIGHT_COMPLETE')
    run(['/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se','-b','do',str(HERE/'preflight.do')],'stata_process')
    assert 'STATA_PREFLIGHT_COMPLETE' in (WORK/'preflight_stata.log').read_text()
    for platform in ['r','stata']:
        evidence['native'][platform] = {}
        for short in ['lp','ppml']:
            for kind in ['target','example']:
                frame = pd.read_csv(WORK/f'{platform}_{short}_{kind}.csv')
                actual = frame.set_index('term').estimate.to_dict()
                for term,value in expected[(short,kind)].items():
                    assert np.isclose(actual[term],value,rtol=1e-6,atol=1e-8), (platform,short,kind,term,actual[term],value)
                evidence['native'][platform][f'{short}_{kind}'] = actual
    native_log = (p.META/'native_check.log').read_text()
    assert 'NATIVE_PACKAGES_COMPLETE' in native_log
    evidence.update(passed=True, native_package_check=str(p.META/'native_check.log'), methods_unchanged=True,
        examples_unchanged=True, reference_outputs_unchanged=True,
        note='Native PPML covariance uses all retained design rank, including fixed effects. Stata native ppmlhdfe validates separation and coefficients.')
    p.c.save(p.META/'preflight.json',evidence)
    print('Passed unchanged target references and examples in Python, R, and Stata.')

if __name__ == '__main__':
    main()
