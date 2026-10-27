"""Prospective expanded evaluation. Preparation and adjudication make no provider calls."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import fcntl
import importlib.util
from ._loader import load_sibling
import importlib.metadata
import itertools
import json
from pathlib import Path
import random
import re
import signal
import sys

ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'quality_reports/reports/2026-09-16_expanded-prompt-evaluation'
WORK=REPORT/'work/collector'
PROMPTS=REPORT/'work/prompts'
PREVIOUS=ROOT/'quality_reports/reports/2026-09-16_prompt-variation-reps'
SOURCE = Path(__file__).with_name('prompt_variation.py')
previous = load_sibling(__package__, 'prompt_variation')
c=previous.c
RAW_RUN_JOB=c.run_job
c.WORK=WORK
POLICY=dict(c.POLICY,version='expanded_prompt_evaluation_v2_cli_version_normalization',startup=8,repetitions=list(range(6)),
            prompt_variants=['zero','few_shot'],shuffle_seed=2026091603,
            primary_comparison='All planned requests. Numerical agreement and data/script integrity, with confirmed boundary violations classified as operational failures.',
            pending_rule='Unresolved protocol or infrastructure holds have null primary outcomes. No silent exclusions.',
            numerical_rule='Null when collector scoring did not occur. Delivery failures remain primary failures.',
            retry_rule='One retry on root-driven resume after a recognized pre-response transport outage. Preserve attempts. Other model failures are terminal.',
            continuation='Initial matching eight reviewed for protocol compliance. Remaining fixed batch does not depend on effect signs.',
            accounting={'sonnet_input_usd_per_million':3,'sonnet_output_usd_per_million':15,
                        'sonnet_input_fields':['input_tokens','cache_creation_input_tokens','cache_read_input_tokens'],
                        'output_includes_reasoning':True,'nested_iterations_added':False,'helper_usage':'separate',
                        'basis':'Common all-input assumption, not the provider invoice.'})
c.POLICY=POLICY


def version_identity(result):
    return dict(argv=result['argv'],returncode=result['returncode'],stdout=result['stdout'].strip())


def preflight_jobs(manifest):
    return [j for j in manifest['jobs'] if j['task']=='applied_matching' and j['rep']==0]
c.preflight_jobs=preflight_jobs

def transport_retry_eligible(record):
    """Only visible transport failures before any model work qualify."""
    calls=record.get('calls',[])
    if len(calls)!=1:return False
    call=calls[0]
    if not call.get('infrastructure_failure') or call.get('ok'):return False
    if record.get('executions') or record.get('access_findings'):return False
    if call.get('text','').strip() or call.get('num_tool_calls',0):return False
    usage=call.get('usage') or {}
    if any(isinstance(value,(int,float)) and value>0 for value in usage.values()):return False
    message=call.get('error_message','')
    return bool(re.search(r'connection (?:reset|refused)|network (?:unreachable|error)|APIConnectionError|\b(?:502|503|504|429)\b|temporarily unavailable|overloaded|rate.limit',message,re.I))


def adapt_record(raw,destination):
    """Separate skipped scoring, pending review, and operational primary failure."""
    record=copy.deepcopy(raw);score=record['scoring'];executed=bool(record.get('executions'))
    numerical=bool(score['numeric_success']) if executed else None
    access=bool(record.get('access_findings') or record.get('protocol_violation'))
    technical=transport_retry_eligible(record)
    driver=bool(record.get('driver_error'))
    pre_response_infra=any(call.get('infrastructure_failure') and not call.get('text','').strip()
                          and not call.get('num_tool_calls',0) for call in record.get('calls',[])) and not executed
    attempt=int(destination.name.split('_')[-1])
    if technical:
        record['status']='technical_retry_available' if attempt==1 else 'needs_review'
        state='retry_available' if attempt==1 else 'retry_exhausted'
        primary=None;protocol='hold'
    elif access or driver or (pre_response_infra and not executed):
        state='protocol_hold' if access else 'infrastructure_hold'
        primary=None;protocol='hold'
    else:
        state='scored' if executed else 'delivery_failure'
        primary=bool(score['success']);protocol='clear'
    record['outcome']=dict(primary_success=primary,numerical_success=numerical,
                           data_script_integrity=bool(score.get('integrity_pass')) if executed else None,
                           protocol_status=protocol,measurement_status=state,
                           provider_completed=bool(record.get('provider_ok')),
                           denominator_rule='Retain every planned slot. Resolve holds explicitly.',
                           retry_eligible=technical and attempt==1,retries_used=max(0,attempt-1))
    record['scoring']=dict(score,success=primary,numeric_success=numerical,
                           integrity_pass=bool(score.get('integrity_pass')) if executed else None)
    model=record.get('job',{}).get('model')
    observations=[]
    for call in record.get('calls',[]):
        usage=call.get('usage');fields=POLICY['accounting']['sonnet_input_fields'] if model=='sonnet' else ['input_tokens']
        complete=isinstance(usage,dict) and all(isinstance(usage.get(k),(int,float)) for k in fields+['output_tokens'])
        normalized_input=sum(usage[k] for k in fields) if complete else None
        output=usage['output_tokens'] if complete else None
        returned=[e.get('model') or e.get('payload',{}).get('model') for e in call.get('observable_runtime',{}).get('initialization_events',[])]
        observations.append(dict(raw_usage=usage,normalized_main_input_tokens=normalized_input,main_output_tokens=output,
                                 usage_complete=complete,requested_model=c.MODELS[model][0] if model in c.MODELS else None,
                                 requested_effort=c.MODELS[model][1] if model in c.MODELS else None,
                                 observed_initialization_models=[x for x in returned if x],
                                 model_attestation='Only saved initialization fields. Empty means unobserved.',
                                 main_common_basis_cost_usd=(3*normalized_input+15*output)/1e6 if complete and model=='sonnet' else None,
                                 provider_reported_cost_usd=call.get('total_cost_usd'),
                                 separate_model_usage=call.get('raw_meta',{}).get('modelUsage')))
    record['accounting_observations']=observations
    record['inherited_runtime_record']=str(destination/'inherited_runtime_record.json')
    return record


def receipt_path(record_path):return Path(record_path).with_name('protocol_review.json')

def effective_record(record_path,protocol_hash):
    path=Path(record_path);record=json.loads(path.read_text())
    if record['protocol_sha256']!=protocol_hash:raise ValueError('Mixed protocol records')
    receipt=receipt_path(path)
    if receipt.exists():
        review=json.loads(receipt.read_text())
        if review.get('record_sha256')!=c.base.sha(path) or review.get('protocol_sha256')!=protocol_hash:raise ValueError('Review receipt hash mismatch')
        if review.get('status')!='confirmed_protocol_violation' or review.get('retain_primary_failure') is not True:raise ValueError('Unsupported review disposition')
        record=copy.deepcopy(record);record['outcome'].update(primary_success=False,protocol_status='confirmed_violation',measurement_status='resolved_operational_failure')
        record['scoring']['success']=False;record['status']='confirmed_protocol_violation';record['review_receipt']=str(receipt)
    return record


def completed_record(folder,protocol_hash):
    records=[effective_record(p,protocol_hash) for p in sorted(Path(folder).glob('attempt_*/run_record.json'))]
    terminal=[r for r in records if r.get('outcome',{}).get('primary_success') is not None]
    if len(terminal)>1:raise ValueError('Multiple evaluated records for one slot')
    return terminal[0] if terminal else None
c.completed_record=completed_record


def run_job(job,manifest,run_root,model_call=None,execute_call=None):
    kwargs={}
    if model_call is not None:kwargs['model_call']=model_call
    if execute_call is not None:kwargs['execute_call']=execute_call
    folder=Path(run_root)/c.job_id(job)
    if completed_record(folder,manifest['protocol_sha256']) is None:validate_resume(folder,manifest['protocol_sha256'])
    raw=RAW_RUN_JOB(job,manifest,run_root,**kwargs)
    if raw is None or 'outcome' in raw:return raw
    destination=Path(raw['attempt_directory']);raw_path=destination/'inherited_runtime_record.json'
    if raw_path.exists():raise ValueError('Refuse to overwrite an inherited runtime record')
    c.save(raw_path,raw)
    record=adapt_record(raw,destination);c.save(destination/'run_record.json',record)
    c.checkpoint('adapted_outcome',id=c.job_id(job),status=record['status'],outcome=record['outcome'])
    return record


def validate_resume(folder,protocol_hash):
    paths=sorted(Path(folder).glob('attempt_*/run_record.json'))
    attempts=list(Path(folder).glob('attempt_*'))
    if not attempts:return
    if len(paths)!=1 or len(attempts)!=1:raise ValueError('Review unresolved or exhausted attempts before collection')
    record=effective_record(paths[0],protocol_hash)
    if record.get('status')!='technical_retry_available' or not record.get('outcome',{}).get('retry_eligible'):raise ValueError('Only one verified pre-response transport retry is allowed')
    if not transport_retry_eligible(json.loads(Path(record['inherited_runtime_record']).read_text())):raise ValueError('Retry evidence does not qualify')


def make_manifest(selection):
    selections=json.loads((PROMPTS/'selections.json').read_text());ids=selections[selection]
    catalog=json.loads((PROMPTS/'catalog.json').read_text());specs={key:catalog[key] for key in ids}
    assert len(ids)==(8 if selection=='applied8' else 18) and len(set(ids))==len(ids)
    pairs=json.loads((PROMPTS/'prompt_pairs.json').read_text())['pairs']
    previous_path=PREVIOUS/'work/collector/manifest.json';old=json.loads(previous_path.read_text())
    pins={str(p):c.base.sha(p) for p in [Path(__file__),SOURCE,previous.SOURCE,c.PRIOR,*c.prior.dependency_paths(),previous_path]}
    for name in ['catalog.json','bundles.json','prompt_pairs.json','selections.json','scoring_fields.json','environment.json','validation.json','provenance.json','sample_count_checks.json']:
        path=PROMPTS/name;pins[str(path)]=c.base.sha(path)
    for path in [REPORT/'src/prompts/build_bank.py',ROOT/'quality_reports/decision_records/2026-09-16_expanded-prompt-evaluation.md',ROOT/'pyproject.toml',ROOT/'uv.lock']:
        pins[str(path)]=c.base.sha(path)
    scoring_fields=json.loads((PROMPTS/'scoring_fields.json').read_text())
    assert scoring_fields['exact_integer_n_obs'] is True
    assert {k:set(v) for k,v in scoring_fields['exact_integer_estimates'].items()}==c.EXACT_TERMS,'Discrete scoring map changed'
    bank_environment=json.loads((PROMPTS/'environment.json').read_text())
    assert bank_environment['executable']==sys.executable and bank_environment['python']==sys.version
    for package,version in bank_environment['packages'].items():assert importlib.metadata.version(package)==version,'Package environment changed'
    jobs=[dict(block='expanded20260916',task=t,model=m,platform='python',prompt=v,condition=a,output_format='json',rep=r)
          for t,m,v,a,r in itertools.product(ids,['luna','sonnet'],['zero','few_shot'],['chatbot','agent'],range(6))]
    texts={};runtime_diffs={}
    for j in jobs:
        key=c.prompt_key(j)
        if key in texts:continue
        pair=next(p for p in pairs if p['task_id']==j['task'] and p['condition']==j['condition'])
        path=Path(pair['zero_path' if j['prompt']=='zero' else 'few_path']);pins[str(path)]=c.base.sha(path)
        original=path.read_text();text=original
        if j['condition']=='agent':text=text.replace('Run the script with python3 run.py.',f'Run the script with {sys.executable} run.py.')
        if text!=original:runtime_diffs[key]=''.join(c.difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(path),tofile='same_pinned_runtime_interpreter'))
        texts[key]=text
    for task,condition in itertools.product(ids,['chatbot','agent']):
        template=dict(task=task,platform='python',condition=condition,output_format='json')
        zero=texts[c.prompt_key(dict(template,prompt='zero'))];few=texts[c.prompt_key(dict(template,prompt='few_shot'))]
        assert zero.split('\nTask:',1)[1]==few.split('\nTask:',1)[1],'Paired target or reporting mismatch'
    for spec in specs.values():pins[spec['data_path']]=c.base.sha(Path(spec['data_path']))
    commands={f'{m}/{a}':c.command(dict(model=m,condition=a),Path('<isolated-cwd>'),Path('<last-message>')) for m,a in itertools.product(['luna','sonnet'],['chatbot','agent'])}
    assert c.digest(commands)==c.digest(old['protocol']['commands']),'Provider argv or deadlines changed'
    cli=c.prior.check_tools()
    versions={k:{f:v[f] for f in ['argv','returncode','stdout','stderr']} for k,v in cli.items() if k in ['codex_version','sonnet_version']}
    protocol=dict(block='expanded20260916',selection=selection,policy=POLICY,tasks=specs,prompts=texts,
                  runtime_prompt_diffs=runtime_diffs,commands=commands,source_hashes=pins,
                  environment=old['protocol']['environment'],package_environment=bank_environment,scoring_fields=scoring_fields,cli_versions=versions)
    random.Random(POLICY['shuffle_seed']).shuffle(jobs)
    assert len(jobs)==len(set(c.job_id(j) for j in jobs))==len(ids)*48
    return dict(created_at=c.now(),protocol_sha256=c.digest(protocol),protocol=protocol,jobs=jobs)


def require_review(manifest,selected,run_root,path):
    c.require_preflight_review(manifest,selected,run_root,path)
    if {c.job_id(j) for j in selected}<={c.job_id(j) for j in preflight_jobs(manifest)}:return
    review=json.loads(Path(path).read_text())
    for key in ['initial8_protocol_reviewed','outcome_rule_confirmed','continuation_independent_of_effect_sign']:
        if review.get(key) is not True:raise ValueError('Missing fixed startup review: '+key)


def status(manifest,run_root):
    rows=[]
    for job in manifest['jobs']:
        folder=Path(run_root)/c.job_id(job);paths=sorted(folder.glob('attempt_*/run_record.json'))
        record=completed_record(folder,manifest['protocol_sha256'])
        if record is None and paths:record=effective_record(paths[-1],manifest['protocol_sha256'])
        rows.append(dict(id=c.job_id(job),status=record['status'] if record else 'unfinished' if list(folder.glob('attempt_*')) else 'pending',outcome=record.get('outcome') if record else None))
    return dict(total=len(rows),counts=dict(Counter(r['status'] for r in rows)),slots=rows)


def adjudicate(path,note):
    path=Path(path).resolve();r=json.loads(path.read_text())
    if not r.get('access_findings') or r.get('outcome',{}).get('protocol_status')!='hold':raise ValueError('Only observed boundary holds qualify')
    target=receipt_path(path)
    if target.exists():raise ValueError('Preserve existing review receipt')
    c.save(target,dict(status='confirmed_protocol_violation',protocol_sha256=r['protocol_sha256'],record_path=str(path),record_sha256=c.base.sha(path),retain_primary_failure=True,numerical_success=r['outcome']['numerical_success'],reviewed_at=c.now(),reason=note))
    print(str(target))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prep','check','status','collect','adjudicate'])
    parser.add_argument('--selection',choices=['applied8','all18'],default='applied8')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--authorization',type=Path);parser.add_argument('--jobs-file',type=Path);parser.add_argument('--preflight-review',type=Path)
    parser.add_argument('--record',type=Path);parser.add_argument('--reason');parser.add_argument('--workers',type=int,choices=range(1,7),default=6)
    args=parser.parse_args(argv)
    if args.action=='adjudicate':
        if not args.record or not args.reason:parser.error('--record and --reason are required')
        adjudicate(args.record,args.reason);return
    if args.manifest is None:args.manifest=WORK/(args.selection+'_v2')/'manifest.json'
    if args.action=='prep':
        manifest=make_manifest(args.selection)
        if args.manifest.exists() and json.loads(args.manifest.read_text())['protocol_sha256']!=manifest['protocol_sha256']:raise ValueError('Use a new manifest path after changes')
        c.save(args.manifest,manifest);c.save(args.manifest.parent/'preflight_jobs.json',preflight_jobs(manifest));c.save(args.manifest.parent/'all_jobs.json',manifest['jobs'])
        print(json.dumps(dict(jobs=len(manifest['jobs']),startup=len(preflight_jobs(manifest)),protocol_sha256=manifest['protocol_sha256'])));return
    manifest=json.loads(args.manifest.read_text());run_root=args.manifest.parent/'runs'/manifest['protocol_sha256']
    if args.action=='status':print(json.dumps(status(manifest,run_root),indent=2));return
    c.verify_manifest(manifest)
    if args.action=='check':
        checks=c.prior.check_tools();assert '--disable-slash-commands' in checks['sonnet_help']['stdout']
        for k,v in manifest['protocol']['cli_versions'].items():
            if version_identity(checks[k])!=version_identity(v):raise ValueError('CLI version changed after preparation')
        c.save(args.manifest.parent/'offline_check.json',dict(provider_calls=0,tools=checks));print('Offline source and CLI checks passed.');return
    current=c.prior.check_tools()
    for k,v in manifest['protocol']['cli_versions'].items():
        if version_identity(current[k])!=version_identity(v):raise ValueError('CLI version changed after preparation')
    for package,version in manifest['protocol']['package_environment']['packages'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Package environment changed after preparation')
    c.require_authorization(manifest,args.authorization)
    selected=c.select_jobs(manifest,args.jobs_file);require_review(manifest,selected,run_root,args.preflight_review)
    for signum in [signal.SIGINT,signal.SIGTERM]:signal.signal(signum,c.prior.stop_children)
    with (args.manifest.parent/'collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[]
        for job in selected:
            folder=run_root/c.job_id(job)
            if completed_record(folder,manifest['protocol_sha256']):continue
            validate_resume(folder,manifest['protocol_sha256']);pending.append(job)
        c.collect_queue(pending,manifest,run_root,args.workers,run=run_job)

if __name__=='__main__':main()
