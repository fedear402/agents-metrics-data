"""Six-model PNAS numerical and missing-cost regression checks."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pnas_analysis as p


def test_cost_per_success_uses_same_priced_subset():
    frame=pd.DataFrame({'cost_usd':[1.0,3.0,np.nan], 'primary':[True,False,True]})
    assert p.cost_per_success(frame)==4.0
    assert frame.primary.mean()==pytest.approx(2/3)


@pytest.mark.parametrize('resource,values,mean,x_value',[
    ('cost_usd',[.1,.3,np.nan],.2,20.0),
    ('total_main_tokens',[100.0,300.0,np.nan],200.0,.2),
])
def test_resource_points_keep_unobserved_failed_runs_in_accuracy(monkeypatch,resource,values,mean,x_value):
    frame=pd.DataFrame({'model':['luna6']*3,'platform':['stata']*3,'agency':['agent']*3,
                        'primary':[True,False,False],resource:values})
    monkeypatch.setattr(p.six.ppc,'MODELS',['luna6'])
    monkeypatch.setattr(p.six.ppc,'PLATFORMS',['stata'])
    monkeypatch.setattr(p.six.ppc,'AGENCIES',['agent'])
    point=p.six.resource_points(frame,resource).iloc[0]
    assert point.runs==3
    assert point.observed==2
    assert point.success_pct==pytest.approx(100/3)
    assert point.mean_resource==pytest.approx(mean)
    assert point.x_value==pytest.approx(x_value)


def test_balanced_task_contrast_cr1_matches_ols():
    rng=np.random.default_rng(6081)
    frames=[]
    for model in ('a','b','c'):
        for task in range(5):
            for cell in range(18):
                for rep in range(10):
                    frames.append((model,str(task),cell,rng.binomial(1,.55+.03*(cell%3)),rep))
    data=pd.DataFrame(frames,columns=['model','task','cell','success','rep'])
    data['model_task']=data.model+'__'+data.task
    fit=smf.ols('success ~ C(model_task) + C(cell, Treatment(0))',data).fit(
        cov_type='cluster',cov_kwds={'groups':data.task})
    # Compare two cell levels. The task-level difference averages equally over models.
    table=data.groupby(['task','cell']).success.mean().unstack()*100
    values=(table[2]-table[0]).to_numpy()
    estimate,se=p.contrast_se(values,models=3)
    position=list(fit.params.index).index('C(cell, Treatment(0))[T.2]')
    assert estimate==pytest.approx(100*fit.params.iloc[position],abs=1e-10)
    assert se==pytest.approx(100*np.sqrt(fit.cov_params().iloc[position,position]),abs=1e-10)


def test_joint_shared_task_model_test_matches_interacted_ols():
    rng=np.random.default_rng(481)
    rows=[]
    for model in ('a','b','c'):
        for task in range(6):
            for cell in range(18):
                for rep in range(10):
                    rows.append((model,str(task),cell,rng.binomial(1,.5+.04*(cell%3))))
    data=pd.DataFrame(rows,columns=['model','task','cell','success'])
    data['model_task']=data.model+'__'+data.task
    import statsmodels.api as sm
    fe=pd.get_dummies(data.model_task,dtype=float).to_numpy()
    cell_columns=np.column_stack([(data.model.eq(m)&data.cell.eq(c)).astype(float)
                                  for m in ('a','b','c') for c in range(1,18)])
    design=np.column_stack([fe,cell_columns])
    assert np.linalg.matrix_rank(design)==design.shape[1]
    fit=sm.OLS(data.success,design).fit(cov_type='cluster',cov_kwds={'groups':data.task})
    vectors=np.zeros((2,design.shape[1]))
    vectors[:,fe.shape[1]+1]=-1
    vectors[0,fe.shape[1]+17+1]=1
    vectors[1,fe.shape[1]+34+1]=1
    tested=fit.f_test(vectors)
    means=data.groupby(['task','model','cell']).success.mean().unstack()
    values=(means[2]-means[0]).unstack().to_numpy()*100
    direct=p.joint_model_test(values)
    assert direct['F']==pytest.approx(float(tested.fvalue),abs=1e-9)
    assert direct['p_cluster_F']==pytest.approx(float(tested.pvalue),abs=1e-10)


@pytest.fixture(scope='module')
def actual():
    if not (p.OUT/'data/numbers.json').is_file():
        pytest.fail('Build code/paper/pnas_analysis.py before the integration checks')
    runs=pd.read_csv(p.OUT/'data/runs.csv',low_memory=False)
    reg=pd.read_csv(p.OUT/'tables/regression_contrasts.csv')
    return runs,reg


def test_complete_six_model_design(actual):
    runs,_=actual
    p.validate_design(runs)
    assert runs.groupby('model').size().eq(7200).all()
    assert runs.groupby(['model','agency']).size().eq(2400).all()
    assert runs.y.sum()==39900
    broken=runs.copy()
    broken.loc[0,'rep']=11
    with pytest.raises(ValueError,match='repetitions'):
        p.validate_design(broken)


def test_all_contrasts_equal_direct_balanced_cells(actual):
    runs,reg=actual
    ws=p.statistics.weights()
    for row in reg.itertuples():
        members=row.member_models.split(',')
        frame=runs[runs.model.isin(members)]
        index=pd.MultiIndex.from_product([p.six.PLATFORMS,['zero_shot','few_shot'],p.six.AGENCIES])
        means=frame.groupby(['platform','prompt','agency']).y.mean().reindex(index).to_numpy()*100
        assert row.estimate_pp==pytest.approx(means@ws[row.contrast],abs=1e-10)
    pooled=reg[reg.model.eq('pooled')].set_index('contrast').estimate_pp
    original=reg[reg.model.isin(p.six.ORDER)].groupby('contrast').estimate_pp.mean()
    np.testing.assert_allclose(pooled.sort_index(),original.sort_index(),atol=1e-10)


def test_original_models_match_established_analysis(actual):
    runs,reg=actual
    old_runs=pd.read_csv(p.OLD/'data/runs.csv',usecols=['position_id','record_path','y','failure_category'])
    selected=runs[runs.model.isin(['sonnet','luna'])].set_index('position_id')
    for row in old_runs.itertuples():
        assert selected.loc[row.position_id,'record_path']==row.record_path
        assert selected.loc[row.position_id,'y']==row.y
        assert selected.loc[row.position_id,'failure_category']==row.failure_category
    old=pd.read_csv(p.OLD/'tables/regression_contrasts.csv').set_index(['model','contrast'])
    new=reg[reg.model.isin(['sonnet','luna'])].set_index(['model','contrast'])
    for col in ('estimate_pp','se_cr1_pp','p_wcb'):
        np.testing.assert_allclose(new[col],old.loc[new.index,col],atol=1e-6,equal_nan=True)


def test_current_cost_missingness_and_macro_totals(actual):
    import json
    runs,_=actual
    pricing=pd.read_csv(p.OUT/'data/run_pricing.csv')
    data=runs.merge(pricing[['record_path','cost_usd']],on='record_path',validate='one_to_one')
    missing=data[data.cost_usd.isna()]
    assert len(missing)==1
    assert missing.model.iloc[0]=='luna6'
    assert data.groupby('agency').size().eq(14400).all()
    n=json.loads((p.OUT/'data/numbers.json').read_text())['macros']
    assert n['numRunsTotal']=='43,200'
    assert n['numRunsPerArm']=='7,200'
    assert n['numModels']=='6'
    assert n['numPricedRuns']=='43,199'
    for agency,label in p.AGENCY_TAGS.items():
        q=data[data.agency.eq(agency)]
        assert n['numAll'+label]==p.fixed(100*q.primary.mean(),1)
        assert n['numAllCps'+label]==p.fixed(p.cost_per_success(q),4)


def test_luna_six_stata_agent_resource_accuracy_uses_all_800_runs(actual,monkeypatch):
    runs,_=actual
    pricing=pd.read_csv(p.OUT/'data/run_pricing.csv')
    data=runs.merge(pricing[['record_path','cost_usd']],on='record_path',validate='one_to_one')
    monkeypatch.setattr(p.six.ppc,'MODELS',['luna6'])
    monkeypatch.setattr(p.six.ppc,'PLATFORMS',['stata'])
    monkeypatch.setattr(p.six.ppc,'AGENCIES',['agent'])
    point=p.six.resource_points(data,'cost_usd').iloc[0]
    assert point.runs==800
    assert point.observed==799
    assert point.success_pct==88.75
