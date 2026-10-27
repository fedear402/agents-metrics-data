"""Fresh 108-slot prompt comparison using the immutable corrected runtime."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import importlib.util
from ._loader import load_sibling
import itertools
import json
from pathlib import Path
import random
import signal
import sys

ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'quality_reports/reports/2026-09-16_prompt-variation-reps'
WORK=REPORT/'work/collector'
PROMPTS=REPORT/'work/prompts'
PREVIOUS=ROOT/'quality_reports/reports/2026-09-16_format-prompt-experiments'
SOURCE = Path(__file__).with_name('corrected.py')
c = load_sibling(__package__, 'corrected')
c.WORK=WORK
POLICY=dict(c.POLICY,version='prompt_variation_fresh108_v1',startup=12,
            repetitions=[2,3,4],prompt_variants=['zero','current_example','revised_example'],
            shuffle_seed=2026091602,output='Python result.json with identical numeric contract',
            primary_comparison='Fresh three repetitions only. Older runs remain a separate supplement.',
            continuation='Root review of initial12, including live Sonnet Bash argv and action metadata.')
c.POLICY=POLICY
TASKS=['applied_fuzzy_rd','applied_matching','broad_ar_confidence_set']
VARIANTS=['zero','current_example','revised_example']

def preflight_jobs(manifest):
    return [j for j in manifest['jobs'] if j['task']=='applied_fuzzy_rd' and j['rep']==2]
c.preflight_jobs=preflight_jobs

def make_manifest():
    previous_path=PREVIOUS/'work/collector_correction/prompt24/manifest.json'
    previous=json.loads(previous_path.read_text())
    catalog=json.loads((PROMPTS/'catalog.json').read_text())
    specs={key:catalog[key] for key in TASKS}
    assert specs==previous['protocol']['tasks'],'Task definitions or references changed'
    pairs=json.loads((PROMPTS/'prompt_variants.json').read_text())['pairs']
    pins={str(p):c.base.sha(p) for p in [Path(__file__),SOURCE,c.PRIOR,*c.prior.dependency_paths(),previous_path]}
    for name in ['catalog.json','bundles.json','prompt_variants.json']:
        p=PROMPTS/name;pins[str(p)]=c.base.sha(p)
    jobs=[dict(block='prompt108',task=t,model=m,platform='python',prompt=v,condition=a,output_format='json',rep=r)
          for t,m,v,a,r in itertools.product(TASKS,['luna','sonnet'],VARIANTS,['chatbot','agent'],[2,3,4])]
    texts={};runtime_diffs={}
    for j in jobs:
        key=c.prompt_key(j)
        if key in texts:continue
        pair=next(p for p in pairs if p['task_id']==j['task'] and p['condition']==j['condition'])
        path=Path(pair[j['prompt']+'_path']);text=path.read_text();pins[str(path)]=c.base.sha(path)
        original=text
        if j['condition']=='agent':text=text.replace('Run the script with python3 run.py.',f'Run the script with {sys.executable} run.py.')
        if text!=original:
            runtime_diffs[key]=''.join(c.difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(path),tofile='identical_prior_interpreter_instruction'))
        if j['prompt'] in {'zero','current_example'}:
            oldjob=dict(j,prompt='zero_shot' if j['prompt']=='zero' else 'few_shot')
            assert text==previous['protocol']['prompts'][c.prompt_key(oldjob)],'Baseline prompt changed'
        baseline=c.prompt_key(dict(j,prompt='zero_shot'))
        target=previous['protocol']['prompts'][baseline].split('\nTask:',1)[1]
        assert text.split('\nTask:',1)[1]==target,'Variant changed target instructions or reporting contract'
        texts[key]=text
    for spec in specs.values():pins[spec['data_path']]=c.base.sha(Path(spec['data_path']))
    commands={f'{m}/{a}':c.command(dict(model=m,condition=a),Path('<isolated-cwd>'),Path('<last-message>'))
              for m,a in itertools.product(['luna','sonnet'],['chatbot','agent'])}
    protocol=dict(block='prompt108',policy=POLICY,tasks=specs,prompts=texts,runtime_prompt_diffs=runtime_diffs,
                  commands=commands,source_hashes=pins,environment=previous['protocol']['environment'])
    random.Random(POLICY['shuffle_seed']).shuffle(jobs)
    assert len(jobs)==len({c.job_id(j) for j in jobs})==108
    return dict(created_at=c.now(),protocol_sha256=c.digest(protocol),protocol=protocol,jobs=jobs)

def require_review(manifest,selected,run_root,path):
    c.require_preflight_review(manifest,selected,run_root,path)
    if {c.job_id(j) for j in selected}<={c.job_id(j) for j in preflight_jobs(manifest)}:return
    review=json.loads(Path(path).read_text())
    for key in ['sonnet_agent_argv_verified','sonnet_agent_action_metadata_verified','initial12_integrity_reviewed']:
        if review.get(key) is not True:raise ValueError('Missing live startup review: '+key)

def status(manifest,run_root):
    counts=Counter();rows=[]
    for job in manifest['jobs']:
        folder=run_root/c.job_id(job);paths=sorted(folder.glob('attempt_*/run_record.json'))
        record=c.completed_record(folder,manifest['protocol_sha256'])
        state=record['status'] if record else json.loads(paths[-1].read_text())['status'] if paths else 'unfinished' if list(folder.glob('attempt_*')) else 'pending'
        counts[state]+=1;rows.append(dict(id=c.job_id(job),status=state))
    return dict(total=len(manifest['jobs']),counts=dict(counts),slots=rows,
                unfinished_note='An unfinished attempt may be active. Inspect the live collector before diagnosing interruption.')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prep','check','status','collect'])
    parser.add_argument('--manifest',type=Path,default=WORK/'manifest.json')
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--jobs-file',type=Path)
    parser.add_argument('--preflight-review',type=Path)
    parser.add_argument('--workers',type=int,choices=range(1,7),default=6)
    args=parser.parse_args(argv)
    if args.action=='prep':
        manifest=make_manifest()
        if args.manifest.exists() and json.loads(args.manifest.read_text())['protocol_sha256']!=manifest['protocol_sha256']:raise ValueError('Use a new manifest path after source changes')
        c.save(args.manifest,manifest);c.save(args.manifest.parent/'preflight_jobs.json',preflight_jobs(manifest));c.save(args.manifest.parent/'all_jobs.json',manifest['jobs'])
        print(json.dumps(dict(jobs=108,startup=12,protocol_sha256=manifest['protocol_sha256'])));return
    manifest=json.loads(args.manifest.read_text());run_root=args.manifest.parent/'runs'/manifest['protocol_sha256']
    if args.action=='status':print(json.dumps(status(manifest,run_root),indent=2));return
    c.verify_manifest(manifest)
    if args.action=='check':
        checks=c.prior.check_tools();assert '--disable-slash-commands' in checks['sonnet_help']['stdout']
        c.save(args.manifest.parent/'offline_check.json',dict(provider_calls=0,tools=checks))
        print('Offline source and CLI checks passed.');return
    c.require_authorization(manifest,args.authorization)
    selected=c.select_jobs(manifest,args.jobs_file);require_review(manifest,selected,run_root,args.preflight_review)
    for signum in [signal.SIGINT,signal.SIGTERM]:signal.signal(signum,c.prior.stop_children)
    with (args.manifest.parent/'collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[]
        for job in selected:
            folder=run_root/c.job_id(job)
            if c.completed_record(folder,manifest['protocol_sha256']):continue
            if list(folder.glob('attempt_*')):raise ValueError('Review unresolved attempt: '+str(folder))
            pending.append(job)
        c.collect_queue(pending,manifest,run_root,args.workers)

if __name__=='__main__':main()
