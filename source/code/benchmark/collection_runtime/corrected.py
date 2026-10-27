"""Two bounded 24-slot experiments. Preparing never calls a provider."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import importlib.util
from ._loader import load_sibling
import itertools
import json
import os
from pathlib import Path
import re
import random
import difflib
import shutil
import signal
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'quality_reports/reports/2026-09-16_format-prompt-experiments'
WORK=REPORT/'work/collector_correction'
ORIGINAL_PATH=os.environ.get('PATH','')
PRIOR = Path(__file__).with_name('collector.py')
prior = load_sibling(__package__, 'collector')
base,harness,sp,codex=prior.base,prior.harness,prior.sp,prior.codex
STOP=prior.STOP
MODELS=prior.MODELS
REVISED=ROOT/'quality_reports/reports/2026-09-16_python-screen/work/prompts_revised'
POLICY=dict(prior.POLICY,version='format_prompt_24_each_v2_quoted_command_fix',workers=6,
            score='Fresh collector execution numeric agreement and integrity. Provider completion is separate.',
            exact_fields='All n_obs and explicit count/binary estimands are exact. Continuous tolerances unchanged.',
            startup={'format24':6,'prompt24':4},sonnet_skills='--disable-slash-commands',
            environment='Explicit pinned interpreter in both agent prompt levels and collector execution; PATH is also prefixed once',
            shuffle_seed=20260916,
            output={'format24':'JSON estimates rows or ANSWER lines in each language','prompt24':'Python result.json with reviewed schema'},
            scanner_limit='Prospective visible commands, paths, and final source checks. Not a proof against indirect or hidden access.')
TERMINAL={'completed','provider_timeout','execution_timeout','provider_error','missing_script'}
EXACT_TERMS={'applied_matching':{'matched_treated','unique_controls'},
             'applied_dynamic_did':{'eligible_regions'},'broad_scm_placebo':{'eligible_placebos'},
             'broad_ar_confidence_set':{'accepts_zero','unbounded'}}
save,now,digest=prior.save,prior.now,prior.digest

def checkpoint(stage,**values):
    WORK.mkdir(parents=True,exist_ok=True)
    with prior.JOURNAL_LOCK,(WORK/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=now(),stage=stage,**values))+'\n')

def command(job,cwd,last):
    argv,timeout=prior.command(job,cwd,last)
    if job['model']=='sonnet':argv+=['--disable-slash-commands']
    assert '--max-turns' not in argv
    return argv,timeout

def environment():
    env=dict(os.environ)
    env['PATH']=str(ROOT/'.venv/bin')+os.pathsep+ORIGINAL_PATH
    return env

def prompt_key(job):
    return '__'.join(job[k] for k in ('task','platform','prompt','condition','output_format'))

def job_id(job):
    return '__'.join(str(job[k]) for k in ('block','task','model','platform','prompt','condition','output_format'))+f"__rep{job['rep']:02d}"

def format_prompt(spec,job):
    if job['output_format']=='json':
        template={'estimates':[dict(term=t,estimate='<number>',se='<number or null>',n_obs='<integer>') for t in spec['estimands']]}
        contract='Write result.json containing only the following numeric result rows:\n'+json.dumps(template,indent=2)+'\nUse at least 12 significant digits. Use null for an unrequested standard error. Do not add estimator or platform metadata.'
    else:
        rows='\n'.join(f'ANSWER: {term} estimate=<number> se=<number or NA> n_obs=<integer>' for term in spec['estimands'])
        contract='Print one labelled line for each required result to standard output (or the Stata log):\n'+rows+'\nUse at least 12 significant digits. Use NA or . for an unrequested standard error.'
    language=harness.script_language(job['platform'])
    return (f"You are a careful econometrician completing a benchmark task.\nThe dataset is provided as ./input.csv.\n{spec['data_description']}\n\n"
            f"Task: {spec['prompt']}\n\nUse {language} and installed statistical packages.\n\n"
            f"Required reporting contract:\n{contract}\n\n"
            f"The evaluator executes your script later in a directory containing input.csv.\n"
            f"Act only as a code generator. Do not use tools, inspect files, or execute code.\nReturn exactly one fenced {language} code block and nothing else.\n")

def make_manifest(block):
    import tasks as task_module
    pins={str(p):base.sha(p) for p in [Path(__file__),PRIOR,*prior.dependency_paths()]}
    recovery_source=Path(__file__).with_name('recover.py')
    if recovery_source.exists():pins[str(recovery_source)]=base.sha(recovery_source)
    runtime_diffs={}
    if block=='format24':
        canonical={t.task_id:t for t in task_module.TASKS}
        specs={}
        for key in ['ols_robust','iv2sls_robust']:
            t=canonical[key]
            specs[key]=dict(task_id=key,title=t.title,prompt=t.prompt,estimands=t.estimands,
                            data_path=str(harness.resolve_data_source(t.data_file)),data_description=t.data_description,
                            reference=harness.REFERENCE[t.reference_key])
        for path in harness.REFERENCE_DIR.glob('reference*.json'):pins[str(path)]=base.sha(path)
        jobs=[dict(block=block,task=t,model='sonnet',platform=l,prompt='zero_shot',condition='chatbot',output_format=o,rep=r)
              for t,l,o,r in itertools.product(specs,['python','r','stata'],['json','answer'],range(2))]
        texts={prompt_key(j):format_prompt(specs[j['task']],j) for j in jobs}
    elif block=='prompt24':
        catalog=json.loads((REVISED/'catalog.json').read_text())
        specs={k:catalog[k] for k in ['applied_fuzzy_rd','applied_matching','broad_ar_confidence_set']}
        pairs=json.loads((REVISED/'prompt_pairs.json').read_text())['pairs']
        jobs=[dict(block=block,task=t,model='luna',platform='python',prompt=p,condition=c,output_format='json',rep=r)
              for t,p,c,r in itertools.product(specs,['zero_shot','few_shot'],['chatbot','agent'],range(2))]
        texts={}
        for j in jobs:
            pair=next(p for p in pairs if p['task_id']==j['task'] and p['condition']==j['condition'])
            path=Path(pair['zero_path' if j['prompt']=='zero_shot' else 'few_path'])
            pins[str(path)]=base.sha(path)
            text=path.read_text()
            if j['condition']=='agent':
                original=text
                text=text.replace('Run the script with python3 run.py.',f'Run the script with {sys.executable} run.py.')
                assert text!=original
                runtime_diffs[prompt_key(j)]=''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(path),tofile='new_runtime_prompt'))
            texts[prompt_key(j)]=text
        for name in ['catalog.json','prompt_pairs.json','bundles.json','independent_validation.json']:
            path=REVISED/name;pins[str(path)]=base.sha(path)
    else:raise ValueError(block)
    for spec in specs.values():
        pins[spec['data_path']]=base.sha(Path(spec['data_path']))
    commands={f'{m}/{c}':command(dict(model=m,condition=c),Path('<isolated-cwd>'),Path('<last-message>'))
              for m,c in sorted({(j['model'],j['condition']) for j in jobs})}
    protocol=dict(block=block,policy=POLICY,tasks=specs,prompts=texts,runtime_prompt_diffs=runtime_diffs,commands=commands,source_hashes=pins,
                  environment=dict(python=str(sys.executable),path_prefix=str(ROOT/'.venv/bin')))
    assert len(jobs)==len({job_id(j) for j in jobs})==24
    random.Random(POLICY['shuffle_seed']).shuffle(jobs)
    return dict(created_at=now(),protocol_sha256=digest(protocol),protocol=protocol,jobs=jobs)

verify_manifest=prior.verify_manifest
require_authorization=prior.require_authorization

def access_findings(text,cwd):
    """Review visible paths and common external-access constructs conservatively."""
    text=str(text);findings=[]
    patterns={'external_skill':r'\.agents[/\\]skills|\.codex[/\\](?:skills|memories)|\.claude[/\\](?:skills|commands)',
              'project_or_reference':r'\b(?:AGENTS|CLAUDE)\.md\b|reference_implementations|quality_reports|catalog\.json|reference_outputs?\.json',
              'network_or_git':r'https?://|\b(?:curl|wget|git)\s+|\b(?:requests|urllib|httpx|socket)\b',
              'parent_or_home':r'\.\.[/\\]|~/|\$(?:HOME|USERPROFILE)\b|Path\.home\(|expanduser\(|getenv\([\"\x27]HOME'}
    for kind,pattern in patterns.items():
        if re.search(pattern,text,re.I):findings.append(kind)
    # Include constant string concatenations used to construct Python paths.
    try:
        tree=ast.parse(text)
        def constant(node):
            if isinstance(node,ast.Constant) and isinstance(node.value,str):return node.value
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
                a,b=constant(node.left),constant(node.right)
                if a is not None and b is not None:return a+b
        text+='\n'+'\n'.join(v for n in ast.walk(tree) if (v:=constant(n)) is not None)
    except SyntaxError:pass
    allowed_executables={str(sys.executable),str(harness.STATA_BIN),'/bin/sh','/bin/bash','/bin/zsh','/usr/bin/env','/dev/null'}
    resource_roots=[str(ROOT/'.venv/lib'),str(Path(sys.base_prefix)/'lib'),'/usr/lib/','/System/Library/','/Library/Frameworks/']
    # Scan single and double quotes separately. A quoted shell program may
    # contain the other quote type around its arguments.
    quoted_paths=set(re.findall(r"'(/[^'\n]+)'",text))|set(re.findall(r'"(/[^"\n]+)"',text))
    # An approved executable plus arguments is a command, not one long path.
    # Its actual path tokens and independently quoted arguments remain scanned.
    quoted_paths={value for value in quoted_paths if not any(value.startswith(exe+' ') for exe in allowed_executables)}
    if text.strip().startswith('/') and not re.search(r'\s',text.strip()):quoted_paths.add(text.strip())
    for raw in set(re.findall(r'(?<![A-Za-z0-9_])/(?:[^\s\"\x27`<>|;,(){}\[\]]+)',text))|quoted_paths:
        path=raw.rstrip(':')
        if path in allowed_executables or path.startswith(tuple(resource_roots)):continue
        if path==str(cwd) or path.startswith(str(cwd)+'/'):continue
        # Shell switches and statistical division are not filesystem paths.
        if path not in quoted_paths and (path.startswith('//') or not re.match(r'/(?:Users|home|private|tmp|var|etc|opt|Applications|Volumes|mnt|root|usr|data|workspace)/',path)):continue
        findings.append('external_absolute_path:'+path)
    return sorted(set(findings))

def visible_inputs(item):
    found=[]
    if isinstance(item,dict):
        for key,value in item.items():
            if key in {'command','code','file_path','path','filename','paths','arguments','input','changes'}:
                if isinstance(value,str):found.append(value)
                else:found.extend(visible_inputs(value))
            elif isinstance(value,(dict,list)):found.extend(visible_inputs(value))
    elif isinstance(item,list):
        for value in item:found.extend(visible_inputs(value))
    return found

def tool_violations(job,calls,cwd):
    findings=prior.tool_violations(job,calls)
    for attempt,call in enumerate(calls,1):
        items=call.get('observable_runtime',{}).get('all_tool_items',call.get('observable_runtime',{}).get('completed_tool_items',[]))
        allowed=set(POLICY['allowed_sonnet_agent_tools'] if job['model']=='sonnet' else POLICY['allowed_agent_tool_item_types'])
        observed={item.get('name' if job['model']=='sonnet' else 'type','<unknown>') for item in items}
        if observed-allowed:findings.append(dict(attempt=attempt,unapproved_observed_tools=sorted(observed-allowed)))
        if items and job['condition']!='agent':findings.append(dict(attempt=attempt,generation_tool_items=True))
        for item in items:
            for value in visible_inputs(item):
                matches=access_findings(value,cwd)
                if matches:findings.append(dict(attempt=attempt,visible_access_findings=matches,visible_input=value))
    return findings

def parse_call(model,raw,last_text,returncode,timed_out,stderr=''):
    call=prior.parse_call(model,raw,last_text,returncode,timed_out,stderr)
    events=harness.load_json_lines(raw);items=[]
    for event in events:
        if event.get('type') in {'item.started','item.completed','item.updated'}:
            item=event.get('item',{})
            if item.get('type') not in codex.NON_TOOL_ITEM_TYPES:items.append(item)
        if event.get('type')=='assistant':
            content=event.get('message',{}).get('content',[])
            items.extend(v for v in content if isinstance(v,dict) and v.get('type') in {'tool_use','server_tool_use','mcp_tool_use'})
    call['observable_runtime']['all_tool_items']=items
    if not call.get('text'):
        chunks=[block.get('text','') for event in events if event.get('type')=='assistant'
                for block in event.get('message',{}).get('content',[]) if isinstance(block,dict) and block.get('type')=='text']
        call['text']='\n'.join(chunks)
    return call

def call_model(text,cwd,job,destination,attempt=1):
    last=destination/f'last_attempt{attempt}.txt'
    argv,timeout=command(job,cwd,last)
    (destination/f'prompt_attempt{attempt}.txt').write_text(text)
    save(destination/f'command_attempt{attempt}.json',dict(argv=argv,cwd=str(cwd),timeout_seconds=timeout,path_prefix=str(ROOT/'.venv/bin')))
    # This collector process uses one pinned PATH for every subprocess and job.
    os.environ['PATH']=environment()['PATH']
    run=prior.subprocess_capture(argv,cwd,destination,f'provider_attempt{attempt}',timeout,text)
    call=parse_call(job['model'],run['stdout'],last.read_text() if last.exists() else '',run['returncode'],run['timed_out'],run['stderr'])
    call.update(wall_seconds=run['wall_seconds'],raw_stream_file=f'provider_attempt{attempt}.stdout',stderr_file=f'provider_attempt{attempt}.stderr')
    last.write_text(call.get('text',''));save(destination/f'call_attempt{attempt}.json',call)
    return call

def execute(script,platform,cwd,destination,attempt=1):
    os.environ['PATH']=environment()['PATH']
    if platform=='stata':
        with open('/private/tmp/agents_metrics_native_stata.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return prior.execute(script,platform,cwd,destination,attempt)
    return prior.execute(script,platform,cwd,destination,attempt)

def load_output(cwd,spec,job,execution):
    if job['output_format']=='answer':
        log=(cwd/'run.log').read_text(errors='replace') if (cwd/'run.log').exists() else ''
        text=log if job['platform']=='stata' else execution.get('stdout','')
        (cwd/'printed_output.txt').write_text(text)
        estimates,notes=base.printed_estimates(text)
        return estimates,'printed_answer',notes
    path=cwd/'result.json'
    data,rule=base.repair_result_text(path.read_text(errors='replace') if path.exists() else '')
    estimates={};notes=[]
    if not isinstance(data,dict) or not isinstance(data.get('estimates'),list):
        return estimates,rule,['Missing estimates list']
    for row in data['estimates']:
        if not isinstance(row,dict) or not isinstance(row.get('term'),str):
            notes.append('Invalid estimate row');continue
        if row['term'] in estimates:notes.append('Repeated output term: '+row['term'])
        estimates[row['term']]={key:row.get(key) for key in ['estimate','se','n_obs']}
    return estimates,rule,notes

def compare_numeric(estimates,spec):
    good,terms,notes=sp.compare_numeric(estimates,base.as_task(spec),{spec['task_id']:spec['reference']})
    for ref in spec['reference']['estimates']:
        term=ref['term'];row=estimates.get(term,{})
        detail=terms.setdefault(term,{})
        for field in ['n_obs']+(['estimate'] if term in EXACT_TERMS.get(spec['task_id'],set()) else []):
            wanted=sp.number_or_none(ref.get(field));got=sp.number_or_none(row.get(field))
            if wanted is not None:
                passed=got is not None and got==wanted and float(got).is_integer()
                detail['exact_'+field+'_pass']=passed
                if not passed:good=False;notes.append(f'{term}: exact {field} mismatch')
    return bool(good),terms,notes

def completed_record(folder,protocol_hash):
    terminal=[]
    for path in sorted(Path(folder).glob('attempt_*/run_record.json')):
        record=json.loads(path.read_text())
        if record['protocol_sha256']!=protocol_hash:raise ValueError('Mixed protocol attempt')
        if record['status'] in TERMINAL:terminal.append(record)
    if len(terminal)>1:raise ValueError('Multiple terminal records for one slot')
    return terminal[0] if terminal else None

def run_job(job,manifest,run_root,model_call=call_model,execute_call=execute):
    folder=Path(run_root)/job_id(job);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=completed_record(folder,manifest['protocol_sha256'])
        if old:return old
        if STOP.is_set():return None
        verify_manifest(manifest)
        destination=folder/f'attempt_{len(list(folder.glob("attempt_*")))+1:04d}';destination.mkdir()
        spec=manifest['protocol']['tasks'][job['task']]
        record=dict(job=job,protocol_sha256=manifest['protocol_sha256'],started_at=now(),status='in_progress',calls=[],executions=[],access_findings=[],attempt_directory=str(destination))
        save(destination/'started.json',record)
        execution={};estimates={};integrity=False;numeric=False;notes=[];terms={}
        try:
            with tempfile.TemporaryDirectory(prefix='format_prompt_',dir='/private/tmp') as tmp:
                cwd=Path(tmp);shutil.copyfile(spec['data_path'],cwd/'input.csv')
                text=manifest['protocol']['prompts'][prompt_key(job)]
                (destination/'prompt_attempt1.txt').write_text(text)
                if job['condition']=='agent':
                    call=model_call(text,cwd,job,destination,1)
                else:
                    with tempfile.TemporaryDirectory(prefix='format_generate_',dir='/private/tmp') as generated:
                        call=model_call(text,Path(generated),job,destination,1)
                record['calls'].append(call)
                record['stop_after_record']=bool(call.get('infrastructure_failure') or call.get('provider_limit'))
                record['access_findings']=tool_violations(job,[call],cwd)
                if job['condition']=='chatbot':
                    code=harness.extract_code(call.get('text',''))
                    if code is not None:(cwd/harness.script_name(job['platform'])).write_text(code)
                script=sp.find_script(cwd,job['platform'])
                for path in cwd.rglob('*'):
                    if path.is_symlink():record['access_findings'].append(dict(symlink=str(path)))
                    elif path.is_file() and path.suffix.lower() in {'.py','.r','.do','.sh','.bash'}:
                        matches=access_findings(path.read_text(errors='replace'),cwd)
                        if matches:record['access_findings'].append(dict(script=str(path.relative_to(cwd)),findings=matches))
                input_ok=(cwd/'input.csv').is_file() and base.sha(cwd/'input.csv')==base.sha(Path(spec['data_path']))
                prior.snapshot(cwd,destination/'submitted_workspace')
                if not input_ok:record['access_findings'].append(dict(input_modified=True))
                if record['access_findings']:
                    record['status']='needs_review';STOP.set()
                elif script is None:
                    record['status']='provider_timeout' if call.get('timed_out') else 'provider_error' if not call.get('ok') else 'missing_script'
                else:
                    # Keep the task cwd valid for absolute local paths. Clear old outputs before rerunning.
                    with nullcontext(cwd) as fresh:
                        for name in ['result.json','run.log','printed_output.txt']:(fresh/name).unlink(missing_ok=True)
                        shutil.copyfile(spec['data_path'],fresh/'input.csv')
                        fresh_script=script
                        execution=execute_call(fresh_script,job['platform'],fresh,destination,1)
                        record['executions'].append(execution)
                        estimates,rule,notes=load_output(fresh,spec,job,execution)
                        numeric,terms,numeric_notes=compare_numeric(estimates,spec);notes+=numeric_notes
                        integrity,integrity_notes=sp.score_integrity(base.as_task(spec),fresh,fresh_script,{spec['task_id']:spec['reference']})
                        integrity=bool(integrity and (fresh/'input.csv').is_file() and base.sha(fresh/'input.csv')==base.sha(Path(spec['data_path'])))
                        notes+=integrity_notes
                        prior.snapshot(fresh,destination/'final_workspace')
                        record['primary_output_path']=str(destination/'final_workspace'/('result.json' if job['output_format']=='json' else 'printed_output.txt'))
                        record['status']='needs_review' if not integrity else 'execution_timeout' if execution.get('timed_out') else 'provider_timeout' if call.get('timed_out') else 'provider_error' if not call.get('ok') else 'completed'
                        if not integrity:STOP.set()
        except Exception as error:
            record.update(driver_error=repr(error),status='needs_review');STOP.set()
        record.update(provider_ok=bool(record['calls']) and all(c.get('ok') for c in record['calls']),
                      task_censored=any(c.get('task_censored') for c in record['calls']),
                      execution_timed_out=bool(execution.get('timed_out')),protocol_violation=bool(record['access_findings']),
                      completed_at=now())
        record['scoring']=dict(success=bool(numeric and integrity and not record['access_findings']),numeric_success=numeric,
                               integrity_pass=integrity,estimates=estimates,per_term=terms,notes=notes,
                               primary_source='fresh collector reexecution')
        save(destination/'run_record.json',record)
        if record.get('stop_after_record'):STOP.set()
        checkpoint('job',id=job_id(job),status=record['status'],success=record['scoring']['success'])
        return record

def preflight_jobs(manifest):
    if manifest['protocol']['block']=='format24':
        return [j for j in manifest['jobs'] if j['task']=='ols_robust' and j['rep']==0]
    return [j for j in manifest['jobs'] if j['task']=='applied_fuzzy_rd' and j['rep']==0]

def select_jobs(manifest,path=None):
    lookup={job_id(j):j for j in manifest['jobs']}
    if path is None:return preflight_jobs(manifest)
    entries=json.loads(Path(path).read_text());ids=[job_id(j) if isinstance(j,dict) else j for j in entries]
    if len(ids)!=len(set(ids)) or not set(ids)<=set(lookup):raise ValueError('Invalid selected jobs')
    return [lookup[k] for k in ids]

def require_preflight_review(manifest,selected,run_root,path):
    initial=preflight_jobs(manifest)
    if {job_id(j) for j in selected}<={job_id(j) for j in initial}:return
    if path is None:raise ValueError('Remaining slots require root startup review')
    review=json.loads(Path(path).read_text())
    if review.get('protocol_sha256')!=manifest['protocol_sha256'] or review.get('approved_to_continue') is not True:raise ValueError('Review does not approve this protocol')
    if any(completed_record(Path(run_root)/job_id(j),manifest['protocol_sha256']) is None for j in initial):raise ValueError('Startup subset unresolved')

def status(manifest,run_root):
    counts=Counter();rows=[]
    for job in manifest['jobs']:
        folder=Path(run_root)/job_id(job);paths=sorted(folder.glob('attempt_*/run_record.json'))
        record=completed_record(folder,manifest['protocol_sha256'])
        state=record['status'] if record else json.loads(paths[-1].read_text())['status'] if paths else 'interrupted' if list(folder.glob('attempt_*')) else 'pending'
        counts[state]+=1;rows.append(dict(id=job_id(job),status=state))
    return dict(total=24,counts=dict(counts),slots=rows)

def collect_queue(pending,manifest,run_root,workers=6,run=run_job):
    iterator=iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active={}
        def refill():
            while len(active)<workers and not STOP.is_set():
                job=next(iterator,None)
                if job is None:return
                active[pool.submit(run,job,manifest,run_root)]=job
        refill()
        while active:
            finished,_=wait(active,return_when=FIRST_COMPLETED)
            for future in finished:
                active.pop(future)
                try:result=future.result()
                except BaseException:STOP.set();raise
                if result is not None:print(json.dumps(dict(id=job_id(result['job']),status=result['status'],success=result['scoring']['success'])),flush=True)
            refill()

def inherit_startup(manifest,original_path):
    original_path=Path(original_path).resolve()
    old=json.loads(original_path.read_text())
    if old['protocol']['block']!='prompt24' or manifest['protocol']['block']!='prompt24':raise ValueError('This recovery is restricted to prompt24')
    for key in ['tasks','prompts','commands','environment']:
        if digest(manifest['protocol'][key])!=digest(old['protocol'][key]):raise ValueError(f'Provider treatment changed: {key}')
    if manifest['jobs']!=old['jobs']:raise ValueError('Job assignment changed')
    pins=manifest['protocol']['source_hashes'];pins[str(original_path)]=base.sha(original_path)
    inherited=[]
    for job in preflight_jobs(old):
        folder=original_path.parent/'runs'/old['protocol_sha256']/job_id(job)
        paths=list(folder.glob('attempt_*/run_record.json'))
        if len(paths)!=1:raise ValueError('Each startup slot must have one preserved attempt')
        path=paths[0];record=json.loads(path.read_text())
        if record['job']!=job or record['protocol_sha256']!=old['protocol_sha256']:raise ValueError('Original record mismatch')
        if record.get('driver_error'):raise ValueError('Driver errors are outside this bounded recovery')
        for item in path.parent.rglob('*'):
            if item.is_symlink():raise ValueError('Original symlink requires separate review')
            if item.is_file():pins[str(item)]=base.sha(item)
        inherited.append(dict(job_id=job_id(job),original_record=str(path),original_record_sha256=base.sha(path),original_protocol_sha256=old['protocol_sha256']))
    if len(inherited)!=4:raise ValueError('Expected four startup records')
    manifest['protocol']['inherited_startup']=inherited
    manifest['protocol']['recovery_rule']='Reuse identical saved provider calls. Correct access adjudication and rerun relative-path submissions locally. Preserve originals.'
    manifest['protocol_sha256']=digest(manifest['protocol'])
    return manifest

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prep','check','status','collect'])
    parser.add_argument('--block',choices=['format24','prompt24'])
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--inherit-from',type=Path,help='Reuse four saved startup provider calls through separate local adjudication')
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--jobs-file',type=Path)
    parser.add_argument('--preflight-review',type=Path)
    parser.add_argument('--workers',type=int,choices=range(1,7),default=6)
    args=parser.parse_args(argv)
    if args.manifest is None:
        if args.block is None:parser.error('--block or --manifest is required')
        args.manifest=WORK/args.block/'manifest.json'
    if args.action=='prep':
        if not args.block:parser.error('--block is required for prep')
        manifest=make_manifest(args.block)
        if args.inherit_from is not None:manifest=inherit_startup(manifest,args.inherit_from)
        if args.manifest.exists() and json.loads(args.manifest.read_text())['protocol_sha256']!=manifest['protocol_sha256']:raise ValueError('Use a new manifest path after changes')
        save(args.manifest,manifest);save(args.manifest.parent/'preflight_jobs.json',preflight_jobs(manifest))
        save(args.manifest.parent/'all_jobs.json',manifest['jobs'])
        print(json.dumps(dict(block=args.block,jobs=24,startup=len(preflight_jobs(manifest)),protocol_sha256=manifest['protocol_sha256'])));return
    manifest=json.loads(args.manifest.read_text());run_root=args.manifest.parent/'runs'/manifest['protocol_sha256']
    if args.action=='status':print(json.dumps(status(manifest,run_root),indent=2));return
    verify_manifest(manifest)
    if args.action=='check':
        checks=prior.check_tools()
        assert '--disable-slash-commands' in checks['sonnet_help']['stdout']
        save(args.manifest.parent/'offline_check.json',dict(provider_calls=0,tools=checks))
        print('Offline source and CLI checks passed.');return
    require_authorization(manifest,args.authorization)
    selected=select_jobs(manifest,args.jobs_file)
    require_preflight_review(manifest,selected,run_root,args.preflight_review)
    for signum in [signal.SIGINT,signal.SIGTERM]:signal.signal(signum,prior.stop_children)
    with (args.manifest.parent/'collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[]
        for job in selected:
            folder=run_root/job_id(job)
            if completed_record(folder,manifest['protocol_sha256']):continue
            if list(folder.glob('attempt_*')):raise ValueError(f'Review unresolved attempt before any new call: {folder}')
            pending.append(job)
        collect_queue(pending,manifest,run_root,args.workers)

if __name__=='__main__':main()
