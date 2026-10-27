"""Refresh the manuscript numbers, tables, and figures from a complete 40-task analysis.

By default the renderer writes only numbers.tex, numbers_pending.tex, the tables, and the figures.
Table 1 (tables/main_results.tex) is the output of code/benchmark/paper_table1.py for the analysis folder.
Table D1 and the numRob* macros come from code/benchmark/paper_robustness.py (run by rebuild.py).
The section and appendix prose is edited by hand and stays as it is.
--prose rebuilds the prose from the immutable 742d543 baseline and overwrites every hand edit.
No provider call, git commit, or remote write occurs in this renderer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
from task_labels import tex_label
from paper_numbers import fixed
import paper_figure_style
import paper_table1
MANUSCRIPT = ROOT / 'paper/manuscript'
BASE = ROOT / 'data/production/main/manuscript_baseline'
REPORT = ROOT / 'quality_reports/manuscript_changes.md'
CHANGES = []

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def number_definition(name, value):
    """Count each rendered numerical macro as one word in TeXcount."""
    return '%TC:macroword \\' + name + ' 1\n' + '\\newcommand{\\' + name + '}{' + value + '}\n'

def baseline():
    if (BASE / 'receipt.json').exists():
        receipt = json.loads((BASE / 'receipt.json').read_text())
        for name, hashed in receipt['files'].items():
            assert digest(BASE / name) == hashed, name
        return
    BASE.mkdir(parents=True, exist_ok=True)
    files = subprocess.check_output(['git','-C',str(MANUSCRIPT),'ls-files'],text=True).splitlines()
    selected = [name for name in files if name.endswith(('.tex','.bib','.cls','.sty')) and not name.startswith('VoxEU/')]
    for name in selected:
        dest = BASE / name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(MANUSCRIPT / name,dest)
    receipt = dict(commit=subprocess.check_output(['git','-C',str(MANUSCRIPT),'rev-parse','HEAD'],text=True).strip(),
                   files={name:digest(BASE / name) for name in selected})
    (BASE / 'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

def text(path):
    return (BASE/path).read_text()

def replace(content, old, new, reason='Updated sample, outcome, or cost definition'):
    if old not in content:
        raise ValueError('Missing baseline text: '+old[:150])
    if old != new:
        CHANGES.append((old,new,reason))
    return content.replace(old,new)

def line(content, beginning, new, reason='Unsupported historical statement replaced with a current-data statement'):
    matches=[s for s in content.splitlines() if s.startswith(beginning)]
    assert len(matches)==1,(beginning,len(matches))
    return replace(content,matches[0],new,reason)

def fmt(value, decimals=1):
    if value is None or not np.isfinite(value):
        return 'not available'
    # Half-up after noise removal, the same rule as the generated tables (paper_numbers.py).
    return fixed(value, decimals, thousands=True)

# Decimals of the cost macros, in US dollars. Sonnet costs print with three decimals. Two groups
# print with four, so that the differences and ratios stated next to them follow from the printed
# values: the mean cost and the cost per success of each agency condition (both models; for
# Sonnet, $0.0455 to $0.1405 gives the stated 9.5 cents and 209 percent, while $0.045 to $0.141
# would give 9.6 cents and 213 percent), and every GPT-5.6 Luna cost ($0.0018 to $0.0089).
USD_DECIMALS = {'sonnet': 3, 'luna': 4}

# Fixed classifications that feed macros (data/production/main/README.md).
EXAMPLE_RELATION = ROOT / 'data/production/main/fewshot_example_relation.csv'
SPURIOUS_REPAIRS = ROOT / 'data/production/main/self_repair_spurious_requests.csv'


def silent_change_intervals(d, draws, groups):
    """95 percent task-bootstrap intervals of the chatbot-to-agent change in silent failures per 100 runs.

    `draws` holds task positions per draw (paper_figure_style.task_draws). All groups use the same
    draws, and the chatbot and agent rates of one draw use the same tasks, as in Figure 2.
    """
    tasks=sorted(d.task.unique())
    out={}
    for name,frame in groups.items():
        rates={}
        for agency in ('chatbot','agent'):
            q=frame[frame.agency.eq(agency)]
            counts=q.assign(silent=q.failure_category.eq('Silent failure')).groupby('task').agg(n=('silent','size'),silent=('silent','sum')).reindex(tasks,fill_value=0)
            rates[agency]=100*counts.silent.to_numpy()[draws].sum(axis=1)/counts.n.to_numpy()[draws].sum(axis=1)
        out[name]=np.percentile(rates['agent']-rates['chatbot'],[2.5,97.5])
    return out


def calculations(analysis):
    d=pd.read_csv(analysis/'data/runs.csv')
    pricing=pd.read_csv(analysis/'data/run_pricing.csv')
    assert len(d)==14400 and d.task.nunique()==40
    assert d.groupby(['task','model','platform','prompt','agency']).size().eq(10).all()
    keys=['record_path']
    assert not pricing.duplicated(keys).any()
    cols=['cost_usd','cost_pricing_complete','priced_cached_input_tokens','priced_cache_write_input_tokens']
    d=d.drop(columns=[c for c in cols if c in d]).merge(pricing[keys+cols],on=keys,how='left',validate='one_to_one')
    assert d.cost_pricing_complete.notna().all()
    reg=pd.read_csv(analysis/'tables/regression_contrasts.csv').set_index(['model','contrast'])
    passes=pd.read_csv(analysis/'data/appendix_pass_at_k.csv')
    cat=json.loads((analysis/'data/catalog.json').read_text())
    n={}
    def put(name,value,decimals=1,absolute=False): n['num'+name]=fmt(abs(value) if absolute else value,decimals)
    def select(model=None,agency=None,prompt=None,platform=None,task=None):
        q=d
        for key,value in [('model',model),('agency',agency),('prompt',prompt),('platform',platform),('task',task)]:
            if value is not None:q=q[q[key].eq(value)]
        return q
    def success(**kw):return select(**kw).primary.mean()
    def cost(stat='mean',**kw):return getattr(select(**kw).cost_usd,stat)()
    def cps(**kw):
        q=select(**kw);q=q[q.cost_pricing_complete];return q.cost_usd.sum()/q.primary.sum()
    def incremental(a,b,**kw):return (cost(agency=b,**kw)-cost(agency=a,**kw))/(success(agency=b,**kw)-success(agency=a,**kw))
    for name,value in [('RunsTotal',len(d)),('RunsPerArm',len(d)//2),('RunsPerPrompt',len(d)//4)]:put(name,value,0)
    put('PooledSuccess',100*d.primary.mean())
    for agency,label in [('chatbot','Chatbot'),('agent','Agent')]:put('Pooled'+label+'Rounded',100*success(agency=agency),0)
    for model,suffix in [('sonnet','Sonnet'),('luna','Codex'),('pooled','Pooled')]:
        if model != 'pooled':
            usd=USD_DECIMALS[model]
            put(suffix+'Overall',100*success(model=model))
            for agency,label in [('chatbot','Chatbot'),('self_repair','OneRepair'),('agent','Agent')]:
                kw=dict(model=model,agency=agency);q=select(**kw)
                put(suffix+label,100*success(**kw))
                gain=100*(success(prompt='few_shot',**kw)-success(prompt='zero_shot',**kw))
                put('PromptGain'+label+suffix,gain,absolute=not(label=='OneRepair' and suffix=='Sonnet'))
                # Mean cost and cost per success of an agency condition: four decimals (see USD_DECIMALS).
                put(suffix+'MeanCost'+label,cost(**kw),4)
                put(suffix+'MedianCost'+label,cost(stat='median',**kw),usd)
                put(suffix+'Cps'+label,cps(**kw),4)
                put('Runtime'+label+'Median'+suffix,q.provider_seconds.median())
                loud=q.failure_category.eq('Loud failure');silent=q.failure_category.eq('Silent failure')
                put('Loud'+label+suffix,100*loud.mean(),2 if agency=='agent' else 1)
                put('SilentPerHundred'+label+suffix,100*silent.mean())
                put('Silent'+label+suffix,100*silent.sum()/(q.execution_ok & q.output_available).sum())
                put('SilentShare'+label+suffix,100*silent.sum()/(~q.primary).sum(),0)
                for k,kl in [(1,'One'),(10,'Twelve')]:
                    put('Pass'+kl+label+suffix,float(passes[(passes.model==model)&(passes.agency==agency)&(passes.k==k)].pass_at_k_pct.iloc[0]))
                for prompt,pl in [('zero_shot','ZS'),('few_shot','FS')]:
                    put(suffix+'MedianCost'+label+pl,cost(stat='median',prompt=prompt,**kw),usd)
                    put(suffix+'Cps'+label+pl,cps(prompt=prompt,**kw),usd)
            for k,kl in [(1,'One'),(10,'Twelve')]:put('Pass'+kl+suffix,passes[(passes.model==model)&(passes.k==k)].pass_at_k_pct.mean())
            for platform,pl in [('python','Python'),('r','R'),('stata','Stata')]:
                put(suffix+pl,100*success(model=model,platform=platform))
                put(suffix+'MedianCost'+pl,cost(stat='median',model=model,platform=platform),usd)
                put(suffix+'Cps'+pl,cps(model=model,platform=platform),usd)
                q=select(model=model,platform=platform,agency='chatbot')
                put('LoudShareChatbot'+pl+suffix,100*q.failure_category.eq('Loud failure').mean())
            for agency,label in [('chatbot','Chatbot'),('agent','Agent')]:put(suffix+'Stata'+label+'ZS',100*success(model=model,platform='stata',agency=agency,prompt='zero_shot'))
            put('ProviderHours'+suffix,select(model=model).provider_seconds.sum()/3600)
            put('RuntimeRatio'+suffix,select(model=model,agency='agent').provider_seconds.median()/select(model=model,agency='chatbot').provider_seconds.median())
            put('RuntimeRepairPct'+suffix,100*(select(model=model,agency='self_repair').provider_seconds.median()/select(model=model,agency='chatbot').provider_seconds.median()-1),0)
            put('RevShareGtOne'+suffix,100*select(model=model,agency='self_repair').calls_count.gt(2).mean())
            put('RevMax'+suffix,select(model=model,agency='self_repair').calls_count.max()-1,0)
            put(suffix+'FewShotCents',100*(cost(model=model,prompt='few_shot')-cost(model=model,prompt='zero_shot')),absolute=True)
            put(suffix+'FewShotPct',100*(cost(model=model,prompt='few_shot')/cost(model=model,prompt='zero_shot')-1),0,True)
            put(suffix+'IncrementalCpsRepair',incremental('chatbot','self_repair',model=model),usd)
            put(suffix+'IncrementalCpsRepairToAgent',incremental('self_repair','agent',model=model),usd)
        for prefix,contrast in [('HOneR','H1_r'),('HOneStata','H1_stata'),('HTwo','H2_fewshot'),('HThreeARepair','H3_self_repair'),('HThreeBAgent','H3_tool_loop'),('HFour','H4_tool_loop'),('StataGapOneRepair','GAP_stata_repair_zs'),('StataGapAgent','GAP_stata_agent_zs'),('StataGapAvg','GAP_stata_avg')]:
            if (model,contrast) not in reg.index:continue
            row=reg.loc[(model,contrast)]
            put(prefix+suffix,row.estimate_pp,absolute=True);put(prefix+suffix+'SE',row.se_cr1_pp)
    put('ProviderHoursTotal',d.provider_seconds.sum()/3600)
    put('SonnetPricedCoveragePct',100*select(model='sonnet').cost_pricing_complete.mean(),2)
    put('CostPerSuccessAgentZS',cps(model='sonnet',agency='agent',prompt='zero_shot'),USD_DECIMALS['sonnet'])
    put('SonnetIncrementalCps',incremental('chatbot','agent',model='sonnet'),USD_DECIMALS['sonnet'])
    for name,a,b in [('MeanCostDeltaCents','chatbot','agent'),('MeanCostDeltaRepairCents','chatbot','self_repair'),('MeanCostDeltaAgentCents','self_repair','agent')]:put('Sonnet'+name,100*(cost(model='sonnet',agency=b)-cost(model='sonnet',agency=a)))
    put('SonnetGainRepairToAgent',100*(success(model='sonnet',agency='agent')-success(model='sonnet',agency='self_repair')))
    for name,fun in [('MeanCost',cost),('Cps',cps)]:put('Sonnet'+name+'AgentPct',100*(fun(model='sonnet',agency='agent')/fun(model='sonnet',agency='chatbot')-1),0)
    put('SonnetMedianCostAgentPct',100*(cost(stat='median',model='sonnet',agency='agent')/cost(stat='median',model='sonnet',agency='chatbot')-1),0)
    for agency,label in [('chatbot','Chatbot'),('agent','Agent')]:put('CostRatioSonnetCodex'+label,cost(model='sonnet',agency=agency)/cost(model='luna',agency=agency))
    for name,fun in [('Mean',cost),('Cps',cps)]:
        put('StataVsPython'+name+'PctSonnet',100*(fun(model='sonnet',platform='stata')/fun(model='sonnet',platform='python')-1),0)
        put('StataVsPython'+name+'RatioCodex',fun(model='luna',platform='stata')/fun(model='luna',platform='python'))
    for prompt,label in [('zero_shot','ZS'),('few_shot','FS')]:
        put('ConvDelta'+label,100*(success(model='sonnet',agency='agent',prompt=prompt)-success(model='sonnet',agency='chatbot',prompt=prompt)))
        put('ConvPrice'+label,incremental('chatbot','agent',model='sonnet',prompt=prompt),USD_DECIMALS['sonnet'])
    price=json.loads((analysis/'data/pricing.json').read_text())['usd_per_million_tokens']['luna']
    for name,key in [('Input','input'),('CachedInput','cached_input'),('Output','output')]:put('GptPrice'+name,price[key],2)
    q=select(model='luna',agency='agent',prompt='zero_shot')
    for name,col in [('Input','main_input_tokens'),('Cached','priced_cached_input_tokens'),('Output','main_output_tokens')]:put('CodexAgentZS'+name+'Tokens',q[col].mean(),0)
    for label,func in [('Min',min),('Max',max)]:put('SilentPerHundred'+label+'Codex',func(100*select(model='luna',agency=a).failure_category.eq('Silent failure').mean() for a in ['chatbot','self_repair','agent']))
    for t,label in [('anova_oneway','Anova'),('staggered_did','Staggered')]:
        if t in cat:put('Task'+label+'Sonnet',100*success(model='sonnet',task=t))
    for a,al in [('chatbot','Chatbot'),('agent','Agent')]:
        if 'two_way_cluster' in cat:put('TwoWayCluster'+al+'Sonnet',100*success(model='sonnet',task='two_way_cluster',agency=a))
    # Luna costs are below one cent per run, so every Luna cost macro has four decimals (USD_DECIMALS).
    put('CodexAgentZSMeanCost',cost(model='luna',agency='agent',prompt='zero_shot'),USD_DECIMALS['luna'])
    # Cache reads as a share of all Sonnet input tokens (uncached input, cache writes, and cache reads).
    q=select(model='sonnet');q=q[q.cost_pricing_complete]
    put('SonnetCacheReadSharePct',100*q.priced_cached_input_tokens.sum()/q.main_input_tokens.sum(),0)
    # Restore the original cross-model comparison with the same CR1 specification.
    sys.path.insert(0, str(ROOT / 'code/benchmark'))
    import factorial_fiverep as ff
    frames, payloads = {}, {}
    for model in ('sonnet', 'luna'):
        frame = d[d.model.eq(model)].copy()
        frame['success'] = frame.primary.astype(float)
        frame['cell'] = [ff.fa.cell_index(p, q, {'chatbot':'no_feedback', 'self_repair':'self_repair', 'agent':'tool_loop'}[a])
                         for p, q, a in zip(frame.platform, frame.prompt, frame.agency)]
        frames[model] = frame
        payloads[model] = {'contrasts': {name: {'estimate': reg.loc[(model, name), 'estimate_pp'] / 100}
                                        for name in ff.CROSS_MODEL_CONTRASTS}}
    cross = ff.cross_model_contrast_differences(frames, payloads, pairs=[('sonnet', 'luna')])
    cross['runs_sha256'] = digest(analysis / 'data/runs.csv')
    cross['source_sha256'] = digest(ff.__file__)
    (analysis / 'data/cross_model.json').write_text(json.dumps(cross, indent=2) + '\n')
    difference = cross['differences']['H4_tool_loop']['sonnet_minus_luna']
    put('HFourDiffSonnetCodex', 100 * difference['estimate'], absolute=True)
    put('HFourDiffSonnetCodexP', difference['p_t20'], 3)
    for model, label in [('sonnet', 'Sonnet'), ('luna', 'Codex')]:
        rates=select(model=model).groupby('agency').primary.mean()
        put('RepairGainShare'+label, 100*(rates.self_repair-rates.chatbot)/(rates.agent-rates.chatbot), 0)
        put('MeanCostAgencyRatio'+label, cost(model=model,agency='agent')/cost(model=model,agency='chatbot'))
        for kind in ('Loud', 'Silent'):
            base=select(model=model,agency='chatbot').failure_category.eq(kind+' failure').mean()
            agent=select(model=model,agency='agent').failure_category.eq(kind+' failure').mean()
            put(kind+'Reduction'+label,100*(1-agent/base),0)
    put('SonnetRepairCostShare',100*(cost(model='sonnet',agency='self_repair')-cost(model='sonnet',agency='chatbot'))/(cost(model='sonnet',agency='agent')-cost(model='sonnet',agency='chatbot')),0)
    put('SonnetSalaryMinutesLow',60*incremental('chatbot','agent',model='sonnet')/50,0)
    put('SonnetSalaryMinutesHigh',60*incremental('chatbot','agent',model='sonnet')/15,0)
    put('PooledPromptGainChatbot',100*(success(agency='chatbot',prompt='few_shot')-success(agency='chatbot',prompt='zero_shot')))
    for task,label in [('hard_truncated_normal_effects','Truncated'),('anova_oneway','Anova')]:
        for model,suffix in [('sonnet','Sonnet'),('luna','Codex')]:
            put('Task'+label+suffix,100*success(task=task,model=model))
            put(label+'Agent'+suffix,100*success(task=task,model=model,agency='agent'))
    # Macros added in the 2026-09-25 revision round for the smaller review items (S36 to S44).
    put('SonnetPythonAgentZS',100*success(model='sonnet',platform='python',agency='agent',prompt='zero_shot'))
    for model,suffix in [('sonnet','Sonnet'),('luna','Codex')]:
        q=select(model=model,agency='chatbot')
        for prompt,pl in [('zero_shot','ZS'),('few_shot','FS')]:
            put('SilentPerHundredChatbot'+pl+suffix,100*q[q.prompt.eq(prompt)].failure_category.eq('Silent failure').mean())
            put('LoudChatbot'+pl+suffix,100*q[q.prompt.eq(prompt)].failure_category.eq('Loud failure').mean())
        # Within-cell intraclass correlation of task success: one-way ANOVA estimator on the 720 cells of ten runs.
        p=select(model=model).groupby(['task','platform','prompt','agency']).primary.mean()
        within=(p*(1-p)).mean()*10/9
        between=p.var()-within/10
        put('Icc'+suffix,between/(between+within),2)
    q=select(model='sonnet',agency='chatbot');q=q[q.cost_pricing_complete]
    for prompt,pl in [('zero_shot','ZS'),('few_shot','FS')]:put('SonnetChatbotOutputTokens'+pl,q[q.prompt.eq(prompt)].main_output_tokens.mean(),0)
    agent=select(agency='agent');failed=agent[~agent.primary]
    by_task=failed.groupby('task').size().sort_values(ascending=False)
    put('AgentTasksNoFailure',agent.task.nunique()-len(by_task),0)
    put('AgentTopFiveFailureShare',100*by_task.head(5).sum()/len(failed),0)
    put('AgentFailureShareCodex',100*failed.model.eq('luna').mean(),0)
    put('TradeIvAgentCodex',100*success(model='luna',agency='agent',task='applied_multivintage_trade_iv'))
    # Pooled failure mix (introduction, Section 4.3, conclusion, Appendix B), per 100 runs.
    for agency,label in [('chatbot','Chatbot'),('agent','Agent')]:
        q=select(agency=agency)
        put('LoudPerHundred'+label+'Pooled',100*q.failure_category.eq('Loud failure').mean())
        put('SilentPerHundred'+label+'Pooled',100*q.failure_category.eq('Silent failure').mean())
        put('FailuresPerHundred'+label+'Pooled',100*(~q.primary).mean())
    # Appendix B: intervals of the chatbot-to-agent change in silent failures, with the Figure 2 draws.
    draws=paper_figure_style.task_draws(sorted(d.task.unique()))
    for name,(low,high) in silent_change_intervals(d,draws,{'Sonnet':select(model='sonnet'),'Codex':select(model='luna'),'Pooled':d}).items():
        put('SilentChangeLow'+name,low);put('SilentChangeHigh'+name,high)
    # Section 4.2: chatbot few-shot effect in the tasks whose example shows the same calculation, or part
    # of it, and in the tasks whose example shows a related method (fewshot_example_relation.csv).
    relation=pd.read_csv(EXAMPLE_RELATION).set_index('task').relation
    if set(relation.index)!=set(d.task):raise ValueError('fewshot_example_relation.csv does not list the analysis tasks')
    for group,label in [(['same','partial'],'Same'),(['family','related'],'Related')]:
        tasks=relation[relation.isin(group)].index
        q=select(agency='chatbot');q=q[q.task.isin(tasks)]
        put('FewShotTasks'+label,len(tasks),0)
        put('FewShotGainChatbot'+label,100*(q[q.prompt.eq('few_shot')].primary.mean()-q[q.prompt.eq('zero_shot')].primary.mean()))
    # Appendix A1: self-repair runs of the three Stata tasks without standard-error fields in the
    # prompt that fail after a repair request that followed output already passing the check.
    spurious=pd.read_csv(SPURIOUS_REPAIRS)
    repair=select(agency='self_repair')
    if set(spurious.position_id)!=set(repair.position_id):raise ValueError('self_repair_spurious_requests.csv does not list the self-repair runs')
    no_se=spurious.platform.eq('stata')&spurious.contract_has_se.astype(str).eq('False')
    if spurious[no_se].task.nunique()!=3:raise ValueError('Appendix A1 names three Stata tasks without standard-error fields')
    flagged=spurious[no_se&spurious.extra_repair_after_pass.astype(bool)].position_id
    put('SpuriousRepairFailurePct',100*(repair.position_id.isin(flagged)&~repair.primary).sum()/len(repair))
    return d,n,cat,reg

def escape(s):
    return ''.join({'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','\\':r'\textbackslash{}'}.get(c,c) for c in str(s))

def common(s):
    for old,new in [('GPT-5.4','GPT-5.6 Luna'),('gpt-5.4','gpt-5.6-luna'),('21 econometric tasks','40 econometric tasks'),('21 tasks','40 tasks'),('twelve','ten'),('pass@12','pass@10'),('1,512','2,400'),('504 runs','800 runs'),('144 runs','120 runs'),('20 degrees of freedom','39 degrees of freedom'),('21 clusters','40 clusters'),('21 task clusters','40 task clusters')]:
        if old in s:s=replace(s,old,new)
    return s

def render_prose(d,n,cat,reg):
    files={str(p.relative_to(BASE)):common(p.read_text()) for folder in ('sections','appendix') for p in (BASE/folder).glob('*.tex')}
    def edit(f,start,new): files[f]=line(files[f],start,new,'Minimum factual change required by the current sample')
    def phrase(f,old,new): files[f]=replace(files[f],old,new,'Minimum factual change required by the current sample')
    a='sections/00_frontmatter.tex'
    edit(a,'On 40 tasks',r'On 40 tasks in Stata, R, and Python, we vary the \emph{prompt} (one version with a simpler statement and another with examples) and the \emph{agency} (from a chatbot that only writes the script to an agent that also runs and revises it).')
    edit(a,'For Claude Sonnet',r'For Claude Sonnet 4.6 and GPT-5.6 Luna, the agent raises task success from \numPooledChatbotRounded{} to \numPooledAgentRounded{} percent.'+'\n'+r'That comes at additional token costs.')
    edit(a,'The example and the software',r'The example and the software matter more under the chatbot than under the agent, which reduces execution failures and errors in the reported estimates.')
    f='sections/01_introduction.tex'
    phrase(f, r'\footnote{\textcolor{red}{Each task, with the prompts it uses in the three software environments, and individual runs behind the results can be browsed at \url{[URL]}. We will release the task definitions, the simulated datasets, the reference outputs, the harness and scoring code, and the run-level analysis file in a replication package.}}', '')
    phrase(f,'fails or produces no result file','fails or produces no readable output')
    phrase(f,'dynamic panel estimation, survival analysis, and time series','survival analysis, time series, decomposition, and discrete choice')
    phrase(f,'and the gap all but disappears for GPT-5.6 Luna as well','and the gap narrows for GPT-5.6 Luna as well')
    phrase(f,'the script does not run or writes no result file','the script does not run or returns no complete output')
    phrase(f,'why the Stata gap closes','why the Stata gap narrows')
    phrase(f,'The failures that remain under the constrained agent are silent','Most failures that remain under the constrained agent are silent')
    phrase(f,'the script runs and writes a result file whose numbers are wrong, and nothing in the result file reveals the error','the script runs and returns numbers that are wrong, and nothing in the output reveals the error')
    f='sections/03_experimental_design.tex'
    phrase(f,'a required result-file structure','a required output format')
    phrase(f,'its result file agrees','its reported estimates agree')
    phrase(f,'leaves no readable result file','leaves no readable output')
    phrase(f,'and within a session time limit it writes','and it writes')
    edit(f,'In every condition, script executions',r'Earlier retained collections applied time limits to script executions and model calls, while later collections have no such limits; a run that reaches a limit without a scored output counts as a failure.')
    phrase(f,'dynamic panel estimation, survival analysis, and time series','survival analysis, time series, decomposition, and discrete choice')
    phrase(f,'the required structure and location of the result file','the required format and location of the output')
    phrase(f,'on small inline data, and the result-file conventions','on separate data, and the output conventions')
    phrase(f,'does not mention its dataset, variables, or reference values','does not supply its dataset or reference values')
    edit(f,'A run succeeds when',r'A run succeeds when its submitted output reports all selected estimands within the stored reference tolerances; auxiliary standard errors and sample sizes do not enter this comparison.')
    phrase(f,'the final result file the run saved','the final output the run saved')
    edit(f,'Before the comparison',r'Before the comparison, it applies a formatting fix to JSON files that changes no reported digit, leaves a truncated or empty file as a failure, and normalizes the conventions with which Python and R write numbers and empty fields; Stata estimates are read from named printed lines (\Cref{app:method_checks}).')
    phrase(f,'the scoring system stores five checks for every run','the saved records contain five checks, where available')
    edit(f,'Task success requires',r'Task success requires numeric agreement for all selected estimands; the other checks are recorded as diagnostics.')
    edit(f,'We call the verdict',r'\Cref{sec:results} and \Cref{app:failure_anatomy} use execution and output availability to describe how unsuccessful runs fail.')
    # Five byte-identical input datasets still serve thirteen current tasks.
    # Keep the original clustering caveat and the restored cross-model specification.
    f='sections/04_preliminary_results.tex'
    phrase(f,'Most of the gain from agency therefore requires no more than one revision','Most of the gain from agency therefore requires no more than three revisions')
    edit(f,'Self-repair alone achieves',r"Self-repair alone achieves \numRepairGainShareSonnet{} percent of the constrained agent's improvement for Claude Sonnet 4.6 and \numRepairGainShareCodex{} percent for GPT-5.6 Luna.")
    phrase(f,'that probability is at least as high as','that probability is below')
    phrase(f,'about nine points above',r'\numPooledPromptGainChatbot{} points above')
    # The interaction signs survive. The original significance levels do not.
    phrase(f,'significant at the 1 percent level for GPT-5.6 Luna and for the pooled sample and at the 10 percent level for Claude Sonnet 4.6',r'with inference reported in \Cref{tab:estimates_three_models}')
    phrase(f,'is not statistically significant ($p = \\numHFourDiffSonnetCodexP$), so the substitution has the same sign and, within sampling error, the same size for the two models','is not statistically significant; the substitution has the same sign for the two models')
    phrase(f,'about half of the baseline gap','about two fifths of the baseline gap')
    phrase(f,'the gap is closed for both models','the gap is closed for Claude Sonnet 4.6')
    phrase(f,'and neither difference is statistically significant','and only the first difference is not statistically significant')
    phrase(f,'to write their result file','to return readable output')
    edit(f,'We classify every run',r'We classify every run that misses the primary outcome by whether it executes and returns every selected estimand (\Cref{app:failure_anatomy}).')
    phrase(f,'the script does not run or writes no result file, so that the user sees an error or a missing file','the script does not run or returns no complete output, so that the user sees an error or a missing result')
    edit(f,'We call a failure silent',r'We call a failure silent when the script runs and returns all selected estimands but their values fail the numeric check, so that nothing in the output reveals the error.')
    edit(f,'Agency therefore removes',r'Agency therefore removes almost all loud failures and \numSilentReductionSonnet{} percent of silent failures for Claude Sonnet 4.6, but less than half of silent failures for GPT-5.6 Luna.')
    phrase(f,'silent failures fall only from','silent failures fall from')
    phrase(f,'and the number of numeric failures does not fall','and the rate of silent failures does not fall for Claude Sonnet 4.6 but falls for GPT-5.6 Luna')
    edit(f,'The example therefore raises',r'The example therefore raises task success mainly by making more scripts run and return their output; for GPT-5.6 Luna, it also makes the reported estimates more accurate.')
    phrase(f,'Unlike the example, the constrained agent also removes some silent failures, although less than half of them.','The constrained agent also removes silent failures, although less than half of them for GPT-5.6 Luna.')
    edit(f,'Two-way clustered OLS is the only',r'Two-way clustered OLS improves under the constrained agent, and for Claude Sonnet 4.6 its success rate rises from \numTwoWayClusterChatbotSonnet{} percent under the chatbot to \numTwoWayClusterAgentSonnet{} percent under the constrained agent.')
    edit(f,'Its failures under the constrained',r'No failures remain for this task under the constrained agent.')
    edit(f,'For Claude Sonnet 4.6, the two tasks',r'For Claude Sonnet 4.6, the two tasks with the lowest success rates across all conditions are truncated normal regression, at \numTaskTruncatedSonnet{} percent, and one-way ANOVA, at \numTaskAnovaSonnet{} percent.')
    edit(f,'Arellano--Bond GMM has a low',r'Truncated normal regression reaches \numTruncatedAgentSonnet{} percent under the constrained agent for Claude Sonnet 4.6 and \numTruncatedAgentCodex{} percent for GPT-5.6 Luna.')
    edit(f,'For Claude Sonnet 4.6, Arellano--Bond',r'For example, the truncated normal task asks for a latent coefficient, the fitted mean conditional on observation, and the average derivative of that conditional mean.')
    edit(f,'Such a script runs',r'A script can run without error and return estimates outside tolerance, so a researcher still needs to review the specification that the model chose.')
    phrase(f,' The horizontal axis begins at 40 percent.','')
    edit(f,'Over the whole sample, the strict',r'The primary outcome compares the selected estimands; the remaining recorded checks are diagnostics.')
    edit(f,'\\Cref{app:method_checks} details',r'\Cref{app:method_checks} details the scoring rules and the kinds of error that can survive them.')
    phrase(f,', and under the strict diagnostic as well when the required method appears only in the script\'s comments','')
    f='sections/05_cost_efficiency.tex'
    phrase(f,"at the list prices of the model's API",'at fixed API-equivalent token rates')
    phrase(f,'writes no result file','returns no readable output')
    phrase(f,'the harness records the cost of each model call',r'we price every input token, including cached tokens, at \$3 per million and output tokens at \$15 per million')
    phrase(f,'at the list prices that OpenAI publishes','at the fixed API token rates')
    phrase(f,'Nine of the', 'Two of the')
    phrase(f,'carry no cost, because a model call timed out or a harness error ended the run before any model call was recorded','lack complete token records after provider timeouts')
    edit(f,'The median rises less',r'The median rises more than the mean, by \numSonnetMedianCostAgentPct{} percent.')
    edit(f,'Self-repair accounts for about',r'Self-repair accounts for about \numSonnetRepairCostShare{} percent of the increase in cost and, as \Cref{sec:results_interfaces} shows, for \numRepairGainShareSonnet{} percent of the gain in task success.')
    phrase(f,'the mean cost of a GPT-5.6 Luna run triples with agency',r'the mean cost of a GPT-5.6 Luna run increases by a factor of \numMeanCostAgencyRatioCodex{} with agency')
    phrase(f,'that of a Claude Sonnet 4.6 run rises by half',r'that of a Claude Sonnet 4.6 run increases by a factor of \numMeanCostAgencyRatioSonnet{}')
    phrase(f,'The few-shot example raises the mean cost of a Claude Sonnet 4.6 run','The few-shot example lowers the mean cost of a Claude Sonnet 4.6 run')
    phrase(f,'The opposite signs are consistent with two ways','These changes are consistent with two ways')
    phrase(f,'which raises the cost of Claude Sonnet 4.6 runs in every condition','which adds input tokens in every condition')
    phrase(f,'far more often under zero-shot prompts','more often under zero-shot prompts')
    phrase(f,'The Stata difference in cost per run already exists under the chatbot and widens as agency increases','For Claude Sonnet 4.6, the Stata difference in cost per run already exists under the chatbot and widens as agency increases')
    phrase(f,'the example has already removed most loud failures','the example has already removed some loud failures')
    phrase(f,'between one and three minutes',r'between about \numSonnetSalaryMinutesLow{} and \numSonnetSalaryMinutesHigh{} minutes')
    f='sections/06_conclusion.tex'
    phrase(f,'much of that gain requires no more than one revision','much of that gain requires no more than three revisions')
    phrase(f,'the two large language models that lead the coding market','two large language models from providers that lead the coding market')
    phrase(f,'the script does not run or writes no result file','the script does not run or returns no complete output')
    phrase(f,'Because the failures it removes are the loud ones','Because most of the failures it removes are loud')
    phrase(f,'the failures that remain once the model runs its code are silent','most failures that remain once the model runs its code are silent')
    phrase(f,'because it removes those that are cheapest to detect and leaves those','because it removes most of those that are cheapest to detect and leaves some of those')
    return files

def appendices(files,d,n,cat):
    f='appendix/A_tasks_scoring.tex';s=files[f]
    s=line(s,'The two harnesses are',r'The two harnesses are Claude Code, which runs Claude Sonnet 4.6 (model identifier claude-sonnet-4-6) at medium effort, and the Codex command-line interface, which runs GPT-5.6 Luna (gpt-5.6-luna) at its low reasoning setting.')
    s=line(s,'Under self-repair, the harness',r'Under self-repair, the harness runs the script in the same way and, when the script fails or leaves no readable output, returns the captured output and the output error to the model, which may submit up to three revised scripts.')
    s=line(s,'The final saved result file',r'The final saved output determines success, and the cost of the run includes every attempt.')
    environment=next(x for x in s.splitlines() if x.startswith('The collection ran'))
    s=replace(s,environment,environment.replace('The collection ran','The recorded initial environment used')+'\n'+'Later collections retain their own environment records.')
    s=line(s,r'\Cref{tab:tasks} lists',r'\Cref{tab:tasks} lists the tasks and their selected estimates, and \Cref{tab:statements} gives the statement of each task as the model receives it.')
    for old,new in [('Model call & 180 seconds & 900 seconds', 'Model call & Recorded by collection & Recorded by collection'),('Model call & 180 seconds per call & 900 seconds per call','Model call & Recorded by collection & Recorded by collection'),('Script execution & 180 seconds, run by the harness & 180 seconds, run by the harness','Script execution & Run by the harness & Run by the harness'),('Script execution & 180 seconds per attempt, run by the harness & 180 seconds per attempt, run by the harness','Script execution & Run by the harness & Run by the harness'),('Session & 600 seconds & 1,500 seconds','Session & Recorded by collection & Recorded by collection')]:
        if old in s:s=replace(s,old,new)
    s=line(s,r'\textit{Notes:} A run that reaches',r'\textit{Notes:} Earlier retained collections used finite time limits. Later collections have no model-call, session, or script time limit. Run records preserve these settings. Reference-access and method diagnostics remain available for review.')
    s=line(s,'The 40 tasks draw',r'The 40 tasks draw on 32 simulated datasets.')
    s=line(s,'The other eight tasks',r'The other 27 tasks use separate simulated inputs.')
    # Preserve both longtable environments and labels; regenerate their factual rows.
    tables=list(re.finditer(r'\\begin\{longtable\}.*?\\end\{longtable\}',s,re.S))
    assert len(tables)==2
    for match,kind in reversed(list(zip(tables,['outputs','statements']))):
        block=match.group();endhead=block.index(r'\endhead')+len(r'\endhead');bottom=block.index(r'\bottomrule',endhead)
        rows=[]
        for task,spec in sorted(cat.items(),key=lambda item:tex_label(item[0]).lower()):
            target=', '.join(spec['estimands']) if kind=='outputs' else spec['prompt']
            rendered=escape(target)
            if kind=='statements':
                rendered=rendered.replace('+',r'+\allowbreak{}').replace('*',r'*\allowbreak{}').replace('=',r'=\allowbreak{}')
            rows.append(escape(tex_label(task))+' & '+rendered+r' \\')
        replacement=block[:endhead]+'\n'+'\n'.join(rows)+'\n'+block[bottom:]
        s=replace(s,block,replacement,'Regenerated 40 task names, primary targets, and statements from the analysis catalog; retained longtable layout and labels')
    opening=next(x for x in s.splitlines() if x.startswith('A prompt is assembled'))
    s=replace(s,opening,opening.replace('A prompt is assembled','An original-task prompt is assembled')+'\n'+'Added tasks use their saved task-specific prompts.')
    s=line(s,'The task statement, the dataset block',r'Within each retained task version, the task statement and dataset block are identical across prompt and agency conditions.')
    s=line(s,'The prompt below is the complete',r'The prompt below reproduces the original zero-shot chatbot prompt for OLS with robust standard errors in Python.')
    example_line=next(x for x in s.splitlines() if x.startswith(('Nine examples serve','Nine worked examples serve')))
    s=replace(s,example_line,'The original tasks share examples by method family, and added tasks have the examples saved with their task definitions.')
    s=line(s,'Use Stata syntax in a do-file.',r'Use a Stata do-file. Read ./input.csv with import delimited.')
    s=line(s,'For R and for Stata it reads:',r'For R and for Stata it reads:'+'\n'+r'Stata uses a separate reporting block that requests one labelled result line per required term in the log, beginning with \texttt{ANSWER:}.')
    s=line(s,'Under Codex, the chatbot prompt',r'Under Codex, the original chatbot prompt and each repair request end with one added paragraph. Agent prompts include the instructions saved for that collection.')
    agent_line=next(x for x in s.splitlines() if x.startswith('Under the constrained agent, the opening'))
    s=replace(s,agent_line,agent_line+'\n'+'The following original-task agent prompt illustrates the execution setting. Later agent collections additionally request an independent numerical check before completion.')
    s=line(s,'Each reported quantity carries',"Original reference outputs carry each quantity's name, estimate, standard error, and effective sample size. Added tasks retain their task-specific reference fields.")
    s=line(s,'Standard errors are null',r'The primary outcome checks selected estimands; auxiliary standard errors and sample-size fields remain diagnostic.')
    s=line(s,'The scoring rules do not compare sample sizes',r'Standalone requested statistics remain in the task definition; auxiliary sample-size fields do not enter the primary outcome.')
    cox=pd.read_csv(cat['cox_hazard']['data_path'])
    s=line(s,'The Cox data, for instance',f'The Cox data, for instance, contain {len(cox):,} subjects and {int(cox.event.sum()):,} events.')
    s=line(s,'Two tasks fix conventions',r'The staggered-DiD task fixes conventions that several implementations would otherwise resolve differently.')
    for start in ['The Arellano--Bond task','The stored values come']:
        s=line(s,start,'')
    s=line(s,'The scoring system records five checks',r'The saved collection records contain the five diagnostics listed in \Cref{tab:five_checks}; availability and historical definitions differ across collections.')
    s=line(s,'Task success, the primary outcome',r'Task success requires all selected estimands to match their references within tolerance.')
    s=line(s,'The strict five-check diagnostic requires',r'The analysis does not combine the five diagnostics into a comparable full-sample success measure.')
    s=line(s,'\\Cref{app:failure_anatomy} distinguishes',r'\Cref{app:failure_anatomy} classifies primary failures using execution and output availability.')
    s=s.replace('Submission validity & The run read the task\'s input data and did not copy the reference & The input file compared with the task data; the script, searched for reads of the reference outputs and for reference values & Yes', 'Submission validity & The run read the task\'s input data and did not copy the reference & The input file compared with the task data; the script, searched for reads of the reference outputs and for reference values & No')
    s=line(s,r'\textit{Notes:} The numeric check',r'\textit{Notes:} The primary outcome compares selected estimands in the final output. Auxiliary fields and method checks are diagnostics. \Cref{tab:app_failure_gates} and \Cref{tab:gate_by_software} describe primary failures.')
    s=line(s,'A coefficient matches','A coefficient matches within 1 percent relative tolerance or 0.005 absolute tolerance.')
    s=line(s,'A standard error matches',r'Auxiliary standard errors do not enter this comparison.')
    s=line(s,'Except for the tasks listed',r'Auxiliary sample sizes do not enter the primary comparison.')
    # Historical planted checks describe a different outcome and are not a current-sample result.
    start=s.index('We tested the scoring rules on planted submissions.')
    end=s.index(r'\end{document}',start)
    s=replace(s,s[start:end],r'Numerical agreement alone can accept an incorrect implementation whose selected estimates fall inside the tolerance.'+'\n'+r'The method and input-integrity records support a separate review of that limitation.'+'\n\n','Removed historical planted-score and tolerance claims that have not been recomputed under the selected-estimands rule')
    s=s.replace('six, eight, ten, or ten significant digits', 'six, eight, ten, or twelve significant digits')
    s=s.replace('Italic rows name the type of method, and tasks appear in the analysis reading order.', 'Tasks appear in alphabetical order.')
    s=s.replace('The result file after any formatting correction, the reference output, the tolerance and row-matching rules', 'The saved output after formatting correction or printed-value extraction, the reference output, and the tolerance and row-matching rules')
    files[f]=s
    f='appendix/B_failure_types.tex';s=files[f]
    replacements={r'\Cref{tab:app_failure_gates} classifies':r'\Cref{tab:app_failure_gates} classifies primary outcomes as success, loud failure, or silent failure.',
        'The last column contains':'The final column reports the silent share of primary failures.',
        'A loud failure fails':'A loud failure has failed execution or incomplete output.',
        'A silent failure passes':'A silent failure executes and supplies every selected estimand, with at least one outside tolerance.',
        'This classification differs':'The classification uses the same primary success measure as the main results.',
        'A recoverable file can':'A run with a recoverable output can pass the primary comparison despite an execution error.',
        'The first-failure classification':'Such a run remains a primary success.',
        'The execution check can':'Execution diagnostics retain the saved process outcome.',
        'Execution and result-file failures':'Loud failures decline from chatbot to constrained agent for both models.',
        'Numeric failures fall':r'Silent failures fall from \numSilentPerHundredChatbotSonnet{} to \numSilentPerHundredAgentSonnet{} percent for Claude Sonnet 4.6 and from \numSilentPerHundredChatbotCodex{} to \numSilentPerHundredAgentCodex{} percent for GPT-5.6 Luna.',
        'These counts describe':'These rates use every run in the corresponding agency condition.',
        'No run fails submission':'Submission-validity records remain separate diagnostics.',
        'Under the chatbot, Stata accounts':r'Under the chatbot, loud failures affect \numLoudShareChatbotStataSonnet{} percent of Claude Sonnet 4.6 Stata runs and \numLoudShareChatbotStataCodex{} percent of GPT-5.6 Luna Stata runs.',
        'Under the constrained agent, one':r'Under the constrained agent, loud failures affect \numLoudAgentSonnet{} percent of Claude Sonnet 4.6 runs and \numLoudAgentCodex{} percent of GPT-5.6 Luna runs.',
        'Arellano--Bond GMM keeps':'',
        'For Claude Sonnet 4.6, its success':''}
    for start,new in replacements.items():s=line(s,start,new)
    s=s.replace('numeric and method failures (silent)','incorrect selected estimates (silent)').replace('diagnostic failures','primary failures')
    s=s.replace('runs that execute and return a valid file','runs that execute and return every selected estimand')
    s=s.replace('execution and result-file failures (loud)','execution and output failures (loud)')
    files[f]=s
    f='appendix/C_costs.tex';s=files[f]
    med=d[d.model.eq('sonnet')].groupby(['agency','prompt']).cost_usd.median()
    repair_delta=[100*(med.loc['self_repair',p]-med.loc['chatbot',p]) for p in ['zero_shot','few_shot']]
    agent_delta=[100*(med.loc['agent',p]-med.loc['chatbot',p]) for p in ['zero_shot','few_shot']]
    replacements={"Claude Sonnet 4.6's records report":r'Claude Sonnet 4.6 records input, output, cache-creation, and cache-read tokens; we price all input tokens at USD 3 per million and output at USD 15 per million.',
        'The cost comparisons reflect':'The cost comparisons use fixed API-equivalent token values across collection dates, with no allocation of subscription fees across runs.',
        'For GPT-5.6 Luna, we price':r'For GPT-5.6 Luna, we price input, cached-input, and output tokens at the API list prices read on September 18, 2026.',
        'The GPT-5.6 Luna records count':r'The GPT-5.6 Luna records count the cached tokens inside the input tokens, so the example below prices the cached count at the cached rate and the rest of the input count at the input rate. Recorded cache writes are priced separately at \$0.25 per million tokens.',
        'For Claude Sonnet 4.6, few-shot prompts raise':r'For Claude Sonnet 4.6, median cost changes from \$\numSonnetMedianCostChatbotZS{} to \$\numSonnetMedianCostChatbotFS{} under chatbot, from \$\numSonnetMedianCostOneRepairZS{} to \$\numSonnetMedianCostOneRepairFS{} under self-repair, and from \$\numSonnetMedianCostAgentZS{} to \$\numSonnetMedianCostAgentFS{} under the constrained agent.',
        'Self-repair adds about':r'Self-repair adds about \numSonnetMedianRepairAddZS{} cents to the zero-shot chatbot median and \numSonnetMedianRepairAddFS{} cents to the few-shot median.',
        'The constrained agent adds about':r'The constrained agent adds about \numSonnetMedianAgentAddZS{} and \numSonnetMedianAgentAddFS{} cents, respectively.'}
    n['numSonnetMedianRepairAddZS'],n['numSonnetMedianRepairAddFS']=(fmt(v) for v in repair_delta)
    n['numSonnetMedianAgentAddZS'],n['numSonnetMedianAgentAddFS']=(fmt(v) for v in agent_delta)
    for start,new in replacements.items():s=line(s,start,new)
    s=s.replace('The vertical axis begins at 70 percent, and the panels share both axes.','The panels share both axes.')
    # Restore exact quoted Python prompts, which require no sample update.
    original=text('appendix/A_tasks_scoring.tex')
    quoted=list(re.finditer(r'\\begin\{promptbox\}.*?\\end\{promptbox\}',original,re.S))
    current=list(re.finditer(r'\\begin\{promptbox\}.*?\\end\{promptbox\}',files['appendix/A_tasks_scoring.tex'],re.S))
    assert len(quoted)==len(current)
    a=files['appendix/A_tasks_scoring.tex']
    for old,new in reversed(list(zip(current,quoted))):
        if 'Use Stata syntax' not in new.group():
            a=a[:old.start()]+new.group()+a[old.end():]
    files['appendix/A_tasks_scoring.tex']=a
    files[f]=s
    return files

def position_figures(files):
    """Apply the September 22 figure moves without rewriting their captions."""
    def move(source, target, label, anchor):
        blocks = re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', files[source], re.S)
        matches = [block for block in blocks if r'\label{' + label + '}' in block]
        assert len(matches) == 1, (source, label)
        block = matches[0]
        files[source] = files[source].replace(block + '\n\n', '', 1) if block + '\n\n' in files[source] else files[source].replace(block + '\n', '', 1)
        assert files[target].count(anchor) == 1, (target, anchor)
        if label == 'fig:benchmark_design':
            # Keep the diagram below the appendix heading on the rendered page.
            block = block.replace(r'\begin{figure}[t]', r'\begin{figure}[!hbp]', 1)
        files[target] = files[target].replace(anchor, block + '\n\n' + anchor, 1)
        CHANGES.append((source + ': ' + label, target + ': ' + label,
                        'Author-approved figure relocation; caption preserved'))

    move('sections/03_experimental_design.tex', 'appendix/A_tasks_scoring.tex',
         'fig:benchmark_design', r'\subsection{Execution protocol}')
    move('appendix/B_failure_types.tex', 'sections/04_preliminary_results.tex',
         'fig:app_silent_loud', 'This accounting helps explain')
    path = 'sections/04_preliminary_results.tex'
    files[path] = replace(files[path],
        r'every selected estimand (\Cref{app:failure_anatomy}).',
        r'every selected estimand (\Cref{fig:app_silent_loud,app:failure_anatomy}).',
        'Cite the failure figure now displayed in the main text')
    return files


# Appendix tables B1 and B2: label, caption, the layout lines that follow \footnotesize, and the note.
# The tidy layouts of the 2026-09-30 table-design audit: grouped rows, a note, and an AER: Insights
# caption in title case without a final period.
APPENDIX_TABLES = {
    # One Appendix B table since the 2026-10-01 appendix reductions (proposal B5): its "All languages"
    # rows replace the former Table B1, so the table carries both labels.
    'appendix_gate_by_software': (('tab:gate_by_software', 'tab:app_failure_gates'),
        'Task Success and Failure Types by Model, Language, and Agency Condition',
        [r'\setlength{\tabcolsep}{6pt}'],
        [r'\textit{Notes:} Percent of runs in each cell.',
         'Success, loud failure, and silent failure sum to 100 up to rounding.',
         'A failure is loud when the script does not run or returns no complete output, and silent when the script runs and returns its output but at least one scored value fails the numeric check.',
         'The last panel pools the three programming languages.']),
}


def appendix_table(name, body):
    """Wrap a generated tabular (production_paper_assets.py) in the installed table float with its note."""
    labels, caption, layout, note = APPENDIX_TABLES[name]
    labels = (labels,) if isinstance(labels, str) else labels
    head = [r'\begin{table}[!htbp]', r'\setstretch{1}', r'\centering',
            r'\caption{'+caption+'}'+''.join(r'\label{'+label+'}' for label in labels),
            r'\footnotesize', *layout]
    tail = [r'\begin{flushleft}\footnotesize', *note, r'\end{flushleft}', r'\end{table}']
    return '\n'.join(head)+'\n'+body+'\n'.join(tail)+'\n'


def current_prose():
    return {str(p.relative_to(MANUSCRIPT)): p.read_text() for folder in ('sections', 'appendix')
            for p in sorted((MANUSCRIPT / folder).glob('*.tex'))}


def render(analysis, prose=False):
    CHANGES.clear()
    baseline()
    analysis=Path(analysis).resolve()
    snapshot=json.loads((analysis/'data/snapshot.json').read_text())
    repairs_complete=snapshot.get('production_repairs_complete') is True
    d,n,cat,reg=calculations(analysis)
    # Appendix D: the numRob* macros and the table come from code/benchmark/paper_robustness.py.
    robustness=analysis/'data/robustness.json'
    if not robustness.is_file():raise ValueError('Run code/benchmark/paper_robustness.py on the analysis folder first')
    n.update(json.loads(robustness.read_text())['macros'])
    # Appendices B and C: the num* macros of code/benchmark/paper_appendix_numbers.py.
    appendix_numbers=analysis/'data/appendix_numbers.json'
    if not appendix_numbers.is_file():raise ValueError('Run code/benchmark/paper_appendix_numbers.py on the analysis folder first')
    n.update(json.loads(appendix_numbers.read_text())['macros'])
    files=position_figures(appendices(render_prose(d,n,cat,reg),d,n,cat))
    if not prose:
        # Keep the hand-edited prose; the macros it uses are checked against the current files below.
        files={name:s for name,s in files.items() if not name.startswith(('sections/','appendix/'))}
    # Table 1 is generated whole by code/benchmark/paper_table1.py (effect over standard error).
    files['tables/main_results.tex']=paper_table1.render(analysis)
    files['tables/appendix_robustness.tex']=(analysis/'tables/appendix_robustness.tex').read_text()
    # Retain original appendix table floats/labels while updating their contents.
    for name in APPENDIX_TABLES:
        files[f'tables/{name}.tex']=appendix_table(name,(analysis/f'tables/{name}.tex').read_text())
    used=set(re.findall(r'\\(num[A-Za-z]+)', '\n'.join([*files.values(), *([] if prose else current_prose().values())])))
    missing=sorted(used-set(n))
    if missing:raise ValueError('Unmapped active numerical macros: '+', '.join(missing))
    originals=set(re.findall(r'\\(?:newcommand|providecommand)\{\\(num[A-Za-z]+)\}',text('numbers.tex')+text('numbers_pending.tex')))
    unused=sorted(originals-set(n))
    numbers='% Generated by code/paper/manuscript.py.\n% Sources: '+str(analysis.relative_to(ROOT))+'; fixed token-pricing ledger.\n'
    for name,value in sorted(n.items()):numbers+=number_definition(name,value)
    # The retired macro names go to the render receipt only.
    files['numbers.tex']=numbers
    files['numbers_pending.tex']='% All active numerical values are generated in numbers.tex.\n'
    for name,s in files.items():
        dest=MANUSCRIPT/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(s)
    for figure in (analysis/'figures').glob('*.pdf'):shutil.copyfile(figure,MANUSCRIPT/'figures'/figure.name)
    receipt=dict(analysis=str(analysis),rows=len(d),tasks=40,baseline_commit=json.loads((BASE/'receipt.json').read_text())['commit'],
        production_repairs_complete=repairs_complete,provisional=not repairs_complete,pdf_compilation_required=True,
        active_macros=len(used),generated_macros=len(n),unmapped_active_macros=missing,retired_historical_macros=unused,
        input_hashes={str(p.relative_to(ROOT)):digest(p) for p in [analysis/'data/runs.csv',analysis/'data/run_pricing.csv',analysis/'data/catalog.json',analysis/'tables/regression_contrasts.csv',analysis/'data/robustness.json',*[analysis/'data'/folder/m/'bootstrap_results.json' for folder in ('inference','inference_dataset') for m in paper_table1.MODELS],Path(__file__),Path(paper_table1.__file__)]},
        manuscript_hashes={name:digest(MANUSCRIPT/name) for name in files})
    (BASE/'latest_render.json').write_text(json.dumps(receipt,indent=2)+'\n')
    status='The authorized task replacements are complete. The rendered sources use the completed production sample. Compilation and visual verification remain separate steps.' if repairs_complete else 'Current results are provisional while the authorized task replacements collect.'
    report=['# Production 40-task manuscript changes','',('By default this rendering writes only numbers.tex, the tables, and the figures, and keeps the hand-edited prose. With --prose it rebuilds the prose from the original Overleaf source. No remote write occurred.' if not prose else 'This --prose rendering starts from the original Overleaf source and overwrites hand edits to the section and appendix prose. No remote write occurred.'),'','## Sources','',f'Analysis: `{analysis.relative_to(ROOT)}`. Baseline: `{receipt["baseline_commit"]}`. All costs use `data/run_pricing.csv`.','','## Rebuild and verification','','Run `uv run --no-sync python code/paper/manuscript.py --analysis data/analysis/main`, then run `latexmk -pdf -interaction=nonstopmode` inside `paper/manuscript/`. The default render keeps the section and appendix prose; --prose rebuilds it from the recorded baseline. `data/production/main/manuscript_baseline/verification.json` records the last compilation and idempotence checks. The `visual/` subfolder holds inspected PDF pages and their contact sheet. Verification receipts describe their recorded PDF hashes; compile and inspect again after a data refresh.','','## Limitations','','The strict five-check outcome and tolerance sensitivity are not recomputed in this analysis. Their historical numerical claims were removed. The original cross-model specification is recomputed with 40 task clusters and 39 degrees of freedom; data/cross_model.json records its results. '+status,'','## Changes','']
    for old,new,reason in CHANGES:
        report.extend(['- '+reason+'. Original: `'+old.replace('`','')+'` Replacement: `'+new.replace('`','')+'`',''])
    REPORT.write_text('\n'.join(report))
    print(json.dumps({k:receipt[k] for k in ('rows','tasks','active_macros','generated_macros','unmapped_active_macros')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=ROOT/'data/analysis/main')
    parser.add_argument('--prose',action='store_true',help='Also rebuild section and appendix prose from the 742d543 baseline; overwrites hand edits')
    args=parser.parse_args()
    render(args.analysis,args.prose)
