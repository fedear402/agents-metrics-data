"""Isolated seven-repetition Luna collection with complete runtime evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'quality_reports/reports/2026-09-14_luna-hard-five-seven'
RUNS = ROOT / 'data/pilots/2026-09-14_luna_hard_five_seven'
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import codex_backend
import demonstrations
import harness
import strict_posthoc as sp
from artifact_repair import repair_result_text
from tasks import Task

MODEL = 'gpt-5.6-luna'
EFFORT = 'low'
CONDITIONS = ('chatbot', 'self_repair', 'agent')
PLATFORMS = ('python', 'r', 'stata')
TASK_MODULES = ('heckman', 'gmm', 'conley', 'rolling', 'dml')
LOCK = threading.Lock()
ACTIVE = 0
STOP = threading.Event()
STATA_EXEC = threading.Semaphore(2)
STATA_AGENT = threading.Semaphore(3)
OTHER_EXEC = threading.Semaphore(6)
class _AnswerLine:
    """Match one printed ``ANSWER:`` line of the output contract.

    The prompts state ``estimate=`` and optionally ``se=`` and ``n_obs=``.
    The fields may come in any order. Each field may occur once. A field that
    the line omits is returned as ``.`` (missing). ``printed_estimates`` joins
    Stata ``> `` continuation lines before matching. This object keeps the
    ``ANSWER.match(line).groups()`` interface, so the function bodies stay
    identical to the migration baseline in tests/fixtures.
    """
    LINE = re.compile(r'^ANSWER:\s*([A-Za-z0-9_]+)((?:\s+(?:estimate|se|n_obs)\s*=\s*\S+)+)\s*$')
    FIELD = re.compile(r'(estimate|se|n_obs)\s*=\s*(\S+)')

    class _Fields:
        def __init__(self, values):
            self.values = values

        def groups(self):
            return self.values

    def match(self, line):
        found = self.LINE.match(line)
        if not found:
            return None
        fields = self.FIELD.findall(found.group(2))
        names = [name for name, _ in fields]
        if 'estimate' not in names or len(names) != len(set(names)):
            return None
        values = dict(fields)
        return self._Fields((found.group(1), values['estimate'], values.get('se', '.'), values.get('n_obs', '.')))


ANSWER = _AnswerLine()

def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)

def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_module(name):
    spec = importlib.util.spec_from_file_location('hard_' + name, OUT/'src/tasks'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def as_task(spec):
    return Task(task_id=spec['task_id'], title=spec['title'], prompt=spec['prompt'],
                estimands=spec['estimands'], reference_key=spec['task_id'],
                data_file=spec['data_path'], data_description=spec['data_description'])

def stata_demo(task_id):
    text = demonstrations.demo_for(task_id, 'stata')
    head = text.split('file open', 1)[0]
    pattern = re.compile(r"`q'term`q':`q'([^`]+)`q',`q'estimate`q':([^,]+),`q'se`q':([^,]+),`q'n_obs`q':([^}]+)")
    lines = []
    for term, est, se, n in pattern.findall(text):
        se = '.' if se == 'null' else se
        lines.append(f'display "ANSWER: {term} estimate={est} se={se} n_obs={n}"')
    assert lines, task_id
    return head + '\n'.join(lines) + '\n```'

def demo(spec, platform):
    return stata_demo(spec['demo_task_id']) if platform == 'stata' else demonstrations.demo_for(spec['demo_task_id'], platform)

def output_contract(spec, platform):
    if platform != 'stata':
        return harness.output_schema_blurb(as_task(spec))
    rows = '\n'.join(f'ANSWER: {t} estimate=<number> se=<number or .> n_obs=<integer>' for t in spec['estimands'])
    return f'''Print one labelled result line per required term to the Stata log:
{rows}
Use valid Stata formats. This template illustrates the reporting syntax:
display "ANSWER: term_name estimate=" %21.12f estimate_value " se=" %21.12f se_value " n_obs=" %12.0f n_value
Replace the example names with computed scalars or expressions. Use . for an unrequested standard error.
Do not use %0.0f. Set linesize 255 to reduce wrapping. A JSON result file is not required.'''

def prompt(spec, j, previous=None, feedback=None):
    platform = j['platform']
    task_text = spec['prompt'].replace('null', 'missing (.)') if platform == 'stata' else spec['prompt']
    example = '' if j['prompt'] == 'zero_shot' else 'Related example:\n' + demo(spec, platform)
    hint = harness.platform_lang_hint(platform) if platform != 'stata' else 'Use a Stata do-file. Read ./input.csv with import delimited.'
    common = f'''You are a careful econometrician completing a benchmark task.
The dataset is provided as ./input.csv.
{spec['data_description']}

{example}

Task: {task_text}

{hint}

{output_contract(spec, platform)}
'''
    if j['condition'] == 'agent':
        if platform == 'stata':
            finish = 'Write run.do and execute it in batch mode. Leave run.log containing all required ANSWER lines.'
        else:
            finish = 'Write and run a script. Leave result.json and the script in this directory.'
        return common + f'''You may use the shell to write, execute, inspect errors, and revise your program.
{harness.execution_hint(platform)}
{finish}
When the requested output is present, finish with DONE.
'''
    if previous is not None:
        common += f'\nPrevious script:\n```{harness.script_language(platform)}\n{previous}\n```\nExecution and output feedback:\n{feedback}\nReturn a corrected complete script.\n'
    return common + f'''The evaluator executes your script later in a directory containing input.csv.
Act only as a code generator. Do not use tools, inspect files, or execute code.
Return exactly one fenced {harness.script_language(platform)} code block and nothing else.
'''

def printed_estimates(text):
    joined = []
    for line in text.splitlines():
        if line.startswith('> ') and joined:
            joined[-1] += line[2:]
        else:
            joined.append(line)
    estimates = {}
    notes = []
    for line in joined:
        match = ANSWER.match(line.strip())
        if not match:
            continue
        term, est, se, n = match.groups()
        if term in estimates:
            notes.append('Repeated output term: ' + term)
        try:
            estimates[term] = {k: None if v.lower() in ('.','na','null','nan') else float(v)
                               for k, v in (('estimate',est),('se',se),('n_obs',n))}
        except ValueError:
            notes.append('Unparseable numeric value: ' + term)
    return estimates, notes

def load_output(dest, spec, platform):
    if platform == 'stata':
        path = dest/'run.log'
        estimates, notes = printed_estimates(path.read_text(errors='replace') if path.exists() else '')
        return estimates, 'printed_stata', notes
    path = dest/'result.json'
    data, rule = repair_result_text(path.read_text(errors='replace') if path.exists() else '')
    _, estimates, notes = sp.score_artifact(data, as_task(spec), platform)
    return estimates, rule, notes

def feedback_output(dest, spec, platform):
    estimates, _, notes = load_output(dest, spec, platform)
    missing = [term for term in spec['estimands'] if term not in estimates]
    if missing:
        return 'Missing required output rows: ' + ', '.join(missing) + '. ' + '; '.join(notes)
    problems = []
    for ref in spec['reference']['estimates']:
        row = estimates[ref['term']]
        for key in ('estimate', 'se', 'n_obs'):
            # The output contract specifies which fields are required. Values
            # are never compared with the reference during model feedback.
            if sp.number_or_none(ref.get(key)) is not None:
                value = sp.number_or_none(row.get(key))
                if value is None:
                    problems.append(ref['term'] + ': missing finite ' + key)
    return '; '.join(problems) or None

def codex_completed(parsed, returncode, timed_out):
    """Use terminal completion, retaining recovered stream errors as diagnostics."""
    return returncode == 0 and not timed_out and parsed.get('event_types', {}).get('turn.completed', 0) > 0


def call_model(text, cwd, j, dest, attempt):
    events = dest/f'codex_events_attempt{attempt}.jsonl'
    stderr = dest/f'codex_stderr_attempt{attempt}.txt'
    last = dest/f'codex_last_attempt{attempt}.txt'
    (dest/f'prompt_attempt{attempt}.txt').write_text(text)
    timeout = 1500 if j['condition'] == 'agent' else 900
    sandbox = 'workspace-write' if j['condition'] == 'agent' else 'read-only'
    argv = [codex_backend.CODEX_BIN, 'exec', '--json', '--ignore-user-config', '--ephemeral',
            '--skip-git-repo-check', '-m', MODEL, '-c', 'model_reasoning_effort="low"',
            '--sandbox', sandbox, '-C', str(cwd), '-o', str(last)]
    for config in ('features.hooks=false','features.apps=false','features.memories=false',
                   'features.multi_agent=false','web_search="disabled"'):
        argv += ['-c', config]
    if j['condition'] != 'agent':
        argv += ['-c', 'features.shell_tool=false']
    save(dest/f'command_attempt{attempt}.json', argv + ['<prompt file>'])
    start = time.monotonic()
    with events.open('w') as output, stderr.open('w') as errors:
        process = subprocess.Popen(argv + [text], cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=errors, start_new_session=True)
        save(dest/'active_call.json', dict(pid=process.pid, attempt=attempt, timeout=timeout, started_at=datetime.now(timezone.utc).isoformat()))
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
    parsed = codex_backend.parse_events(events.read_text(errors='replace'))
    errors = stderr.read_text(errors='replace')
    text = last.read_text(errors='replace') if last.exists() else parsed['agent_text']
    completed = codex_completed(parsed, returncode, timed_out)
    limit = not completed and (codex_backend.is_usage_limit(parsed['error_message']) or codex_backend.is_usage_limit(errors))
    infrastructure_failure = not timed_out and not completed
    if limit or infrastructure_failure:
        STOP.set()
    result = dict(parsed, text=text, model=MODEL, reasoning_effort=EFFORT,
                  returncode=returncode, timed_out=timed_out, provider_limit=limit,
                  infrastructure_failure=infrastructure_failure,
                  ok=completed, sandbox=sandbox,
                  timeout_seconds=timeout, wall_seconds=time.monotonic()-start,
                  codex_version=codex_backend.codex_version(), stderr_tail=errors[-1500:])
    save(dest/f'call_attempt{attempt}.json', result)
    return result

@contextmanager
def active_slot():
    global ACTIVE
    while not STOP.is_set():
        limits = json.loads((OUT/'results/limits.json').read_text())
        with LOCK:
            if ACTIVE < limits['workers']:
                ACTIVE += 1
                break
        time.sleep(0.5)
    else:
        yield False
        return
    try:
        yield True
    finally:
        with LOCK:
            ACTIVE -= 1

def run_id(j):
    return f"{j['task']}__{j['platform']}__{j['prompt']}__{j['condition']}__rep{j['rep']:02d}"

def run(j, specs):
    dest = RUNS/run_id(j)
    if (dest/'run_record.json').exists():
        previous_record = json.loads((dest/'run_record.json').read_text())
        if not any(previous_record.get(k) for k in ('driver_error','provider_limit','infrastructure_failure','protocol_violation')):
            return previous_record
    with active_slot() as allowed:
        if not allowed:
            return None
        if dest.exists():
            archive = RUNS.parent/(RUNS.name + '_interrupted')
            archive.mkdir(exist_ok=True)
            dest.rename(archive/(dest.name+'_'+str(time.time_ns())))
        dest.mkdir(parents=True)
        spec = specs[j['task']]
        shutil.copyfile(spec['data_path'], dest/'input.csv')
        started = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()
        record = dict(j, model=MODEL, reasoning_effort=EFFORT, started_at=started,
                      calls=[], executions=[], output_contract='printed_stata' if j['platform']=='stata' else 'json',
                      source_data_sha256=sha(Path(spec['data_path'])), run_dir=str(dest))
        try:
            if j['condition'] == 'agent':
                with tempfile.TemporaryDirectory(prefix='hard_luna_agent_', dir='/private/tmp') as folder:
                    work = Path(folder)
                    shutil.copyfile(spec['data_path'], work/'input.csv')
                    if j['platform'] == 'stata':
                        with STATA_AGENT:
                            result = call_model(prompt(spec,j), work, j, dest, 1)
                    else:
                        result = call_model(prompt(spec,j), work, j, dest, 1)
                    record['calls'].append(result)
                    # Preserve all model-created files and the final input for integrity checks.
                    shutil.copytree(work, dest/'agent_workspace', dirs_exist_ok=True)
                    harness.copy_back(work, dest, platform=j['platform'])
                    if (work/'input.csv').exists():
                        shutil.copyfile(work/'input.csv', dest/'input.csv')
                    else:
                        (dest/'input.csv').unlink()
                    if j['platform'] == 'stata':
                        if (work/'run.log').exists():
                            shutil.copyfile(work/'run.log', dest/'run.log')
                record['output_delivered'] = feedback_output(dest,spec,j['platform']) is None
                record['execution_success'] = None  # Native tool traces require a separate execution audit.
            else:
                previous = feedback = None
                for attempt in range(1, 5 if j['condition']=='self_repair' else 2):
                    with tempfile.TemporaryDirectory(prefix='hard_luna_generate_',dir='/private/tmp') as folder:
                        result = call_model(prompt(spec,j,previous,feedback),Path(folder),j,dest,attempt)
                    record['calls'].append(result)
                    code = harness.extract_code(result['text']) if result['ok'] else None
                    if not code:
                        record['execution_success'] = False
                        break
                    script = dest/harness.script_name(j['platform'])
                    script.write_text(code)
                    (dest/f'attempt{attempt}_{script.name}').write_text(code)
                    output_name = 'run.log' if j['platform']=='stata' else 'result.json'
                    existing = dest/output_name
                    if existing.exists():
                        shutil.copyfile(existing,dest/f'before_attempt{attempt}_{output_name}')
                    # Clear the previous output so a failed revision cannot reuse it.
                    existing.unlink(missing_ok=True)
                    sem = STATA_EXEC if j['platform']=='stata' else OTHER_EXEC
                    with sem:
                        execution = harness.execute_script(script,j['platform'],dest,180)
                    record['executions'].append(execution)
                    save(dest/f'execution_attempt{attempt}.json',execution)
                    if existing.exists():
                        shutil.copyfile(existing,dest/f'after_attempt{attempt}_{output_name}')
                    problem = feedback_output(dest,spec,j['platform'])
                    record['execution_success'] = execution['ok'] and problem is None
                    if record['execution_success']:
                        break
                    previous = code
                    feedback = json.dumps(execution) + '\n' + str(problem or '')
            estimates, repair, notes = load_output(dest,spec,j['platform'])
            refs = {spec['task_id']:spec['reference']}
            numeric, terms, numeric_notes = sp.compare_numeric(estimates,as_task(spec),refs)
            integrity, integrity_notes = sp.score_integrity(as_task(spec),dest,
                sp.find_script(dest,j['platform']),refs)
            record['scoring'] = dict(success=bool(numeric and integrity), numeric_success=numeric,
                integrity_pass=integrity, per_term=terms, estimates=estimates, repair_rule=repair,
                notes=notes+numeric_notes+integrity_notes)
        except Exception as exc:
            record['driver_error'] = repr(exc)
            record['scoring'] = dict(success=False, notes=['Driver exception; requires review.'])
            STOP.set()
        record['provider_limit'] = any(c.get('provider_limit') for c in record['calls'])
        record['infrastructure_failure'] = any(c.get('infrastructure_failure') for c in record['calls'])
        record['provider_ok'] = bool(record['calls']) and all(c['ok'] for c in record['calls'])
        record['generation_tool_use'] = j['condition']!='agent' and any(c['num_tool_calls'] for c in record['calls'])
        record['task_censored'] = any(c.get('timed_out') for c in record['calls'])
        record['protocol_violation'] = bool(record['generation_tool_use'])
        if not record['provider_ok'] or record['protocol_violation']:
            record['scoring']['success'] = False
        if record['protocol_violation']:
            STOP.set()
        record['total_wall_seconds'] = time.monotonic()-start
        record['completed_at'] = datetime.now(timezone.utc).isoformat()
        save(dest/'run_record.json',record)
        with LOCK:
            print(json.dumps(dict(id=run_id(j),success=record['scoring']['success'],seconds=round(record['total_wall_seconds'],1),provider_ok=record['provider_ok'],driver_error=record.get('driver_error'))),flush=True)
        return record

def prepare():
    specs = {}
    for name in TASK_MODULES:
        directory = OUT/'work'/name
        directory.mkdir(parents=True,exist_ok=True)
        spec = load_module(name).build(directory)
        spec['data_path'] = str(Path(spec['data_path']).resolve())
        assert set(spec['estimands']) == {r['term'] for r in spec['reference']['estimates']}
        for row in spec['reference']['estimates']:
            assert math.isfinite(row['estimate'])
            assert row.get('se') is None or math.isfinite(row['se']) and row['se'] >= 0
        specs[spec['task_id']] = spec
    save(OUT/'results/catalog.json',specs)
    jobs = [dict(task=t,platform=s,prompt=p,condition=c,rep=r)
            for r,t,s,p,c in itertools.product(range(7),specs,PLATFORMS,('zero_shot','few_shot'),CONDITIONS)]
    random.Random(914630).shuffle(jobs)
    assert len(jobs)==630 and len({run_id(j) for j in jobs})==630
    save(OUT/'results/plan.json',jobs)
    paths = [Path(__file__)] + list((OUT/'src/tasks').glob('*.py')) + [ROOT/'code/benchmark'/p for p in ('strict_posthoc.py','harness.py','artifact_repair.py','demonstrations.py','codex_backend.py')]
    save(OUT/'results/source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    save(OUT/'results/design_hashes.json', {str(p.relative_to(ROOT)):sha(p) for p in
        [OUT/'results/catalog.json', OUT/'results/plan.json'] + [Path(s['data_path']) for s in specs.values()]})
    print('Prepared 630 runs.',flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','preflight','full','summary'))
    parser.add_argument('--workers',type=int,default=24)
    args = parser.parse_args()
    RUNS.mkdir(parents=True,exist_ok=True)
    if args.phase=='prepare':
        prepare()
        return
    specs = json.loads((OUT/'results/catalog.json').read_text())
    jobs = json.loads((OUT/'results/plan.json').read_text())
    if args.phase in ('preflight','full'):
        for manifest in ('source_hashes.json', 'design_hashes.json'):
            for path, expected in json.loads((OUT/'results'/manifest).read_text()).items():
                assert sha(ROOT/path) == expected, 'Collection source changed: ' + path
        if args.phase=='full':
            assert json.loads((OUT/'results/preflight_review.json').read_text())['proceed']
        else:
            # One repetition of both prompts and all conditions on every task and language.
            jobs = [j for j in jobs if j['rep']==0]
        save(OUT/'results/limits.json',dict(workers=args.workers))
        with ThreadPoolExecutor(max_workers=24) as pool:
            for future in as_completed([pool.submit(run,j,specs) for j in jobs]):
                future.result()
    rows = [json.loads(p.read_text()) for p in sorted(RUNS.glob('*/run_record.json'))]
    save(OUT/'results/results.json',rows)
    print(json.dumps(dict(completed=len(rows),success=sum(r['scoring']['success'] for r in rows),
                         driver_errors=sum('driver_error' in r for r in rows))),flush=True)

if __name__=='__main__':
    main()
