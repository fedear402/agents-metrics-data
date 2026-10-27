"""Prepare disjoint model queues and preserve compatible candidate responses."""
from pathlib import Path
import hashlib
import json
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect as c

TASKS = ['broad_cohort_stack', 'broad_oaxaca', 'hard_conditional_logit_welfare']
BASE = ROOT / 'data/task_candidates/2026-09-20_three_expansion'
SOURCE = ROOT / 'data/task_candidates/2026-09-19_revised'
META = ROOT / 'data/collections/2026-09-20_three-expansion'
RAW = ROOT / 'data/runs/three_expansion_2026-09-20'
DECISION = ROOT / 'quality_reports/decision_records/2026-09-20_three-task-expansion.md'


def read(path):
    return json.loads(Path(path).read_text())


def suffix(platform, agency):
    language = {'python': 'Python', 'r': 'R', 'stata': 'Stata'}[platform]
    extension = {'python': 'py', 'r': 'R', 'stata': 'do'}[platform]
    if agency != 'agent':
        return f'\nWrite one self-contained {language} script. The evaluator runs it with the supplied CSV files in its working directory.\nAct only as a code generator. Do not use tools, inspect files, or execute code.\nReturn exactly one fenced {platform} code block and nothing else.\n'
    command = {'python': str(ROOT / '.venv/bin/python3') + ' run.py', 'r': 'Rscript --vanilla run.R', 'stata': '/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se -b do run.do'}[platform]
    return f'\nUse the shell to write, execute, inspect, and revise your script.\nRun the script with {command}.\nLeave the final script as run.{extension} and its requested output in this directory.\nWhen the requested output is present, finish with DONE.\nUse only the supplied CSV files, installed software, and files you create in this directory.\nDo not read benchmark references, previous results, agent skills, or project instructions.\nDo not use Git, web tools, MCP servers, or other agents.\n'


def prompts(task, spec):
    folder = BASE / task
    result = {}
    for name in ['zero_shot', 'few_shot']:
        assert (folder / (name + '.txt')).read_bytes() == (SOURCE / task / (name + '.txt')).read_bytes()
    original = (folder / 'zero_shot.txt').read_text()
    assert 'Task:\n' in original
    for platform in ['python', 'r', 'stata']:
        language = {'python': 'Python', 'r': 'R', 'stata': 'Stata'}[platform]
        if platform == 'python':
            fragments = {name: (folder / (name + '.txt')).read_text() for name in ['zero_shot', 'few_shot']}
        else:
            boundary = original.index('Use json.dump') if 'Use json.dump' in original else original.index('Reporting:\n')
            body = original[:boundary].replace('Use Python', f'Use {language}', 1)
            if platform == 'r':
                schema = original[original.index('{\n'):].replace('"platform": "python"', '"platform": "r"')
                body += 'Use jsonlite::write_json to write result.json with the following structure. Replace each <number> with the computed value. Report at least eight significant digits.\n' + schema
            else:
                body += 'Print one line per requested estimate. Use exactly this format, replacing <number> with the computed value. Report at least eight significant digits.\n'
                body += ''.join(f'ANSWER: {term} estimate=<number>\n' for term in spec['estimands'])
            example = (folder / ('example.R' if platform == 'r' else 'example.do')).read_text()
            fragments = {'zero_shot': body, 'few_shot': 'A smaller example uses separate synthetic data to illustrate the method.\n' + f'```{platform}\n' + example + '```\n\n' + body}
        assert fragments['zero_shot'].split('Task:\n', 1)[1] == fragments['few_shot'].split('Task:\n', 1)[1]
        for prompt, fragment in fragments.items():
            for agency in ['chatbot', 'self_repair', 'agent']:
                job = dict(task=task, platform=platform, prompt=prompt, condition=agency, output_format='answer' if platform == 'stata' else 'json')
                result[c.prompt_key(job)] = fragment + suffix(platform, agency)
    return result


def main():
    assert DECISION.exists()
    assert not (META / 'collections.json').exists(), 'Already prepared'
    catalog, texts = {}, {}
    for task in TASKS:
        folder = BASE / task
        validation = read(folder / 'native_validation.json')
        assert validation['passed'], task
        spec = read(folder / 'task.json')
        assert spec['reference'] == read(SOURCE / task / 'task.json')['reference']
        for filename, path in c.task_input_files(spec).items():
            assert c.sha(path) == c.sha(folder / filename)
        catalog[task] = spec
        texts.update(prompts(task, spec))
    ledger = []
    old = read(ROOT / 'data/collections/2026-09-19_candidate-revised/planned_positions.json')
    for row in old:
        if row['task'] not in TASKS:
            continue
        if row['status'] == 'retained':
            record = Path(row['source_record'])
            expected = row['effective_prompt_sha256']
        else:
            manifest_path = Path(row['manifest'])
            manifest = read(manifest_path)
            record = manifest_path.parent / 'runs' / manifest['protocol_sha256'] / c.job_id(row) / 'attempt_0001/run_record.json'
            expected = hashlib.sha256(manifest['protocol']['prompts'][c.prompt_key(row)].encode()).hexdigest()
        assert read(record)['status'] in c.TERMINAL
        job = {**row, 'output_format': 'json'}
        assert hashlib.sha256(texts[c.prompt_key(job)].encode()).hexdigest() == expected
        protocol = record.parents[2].name
        source_manifest = manifest_path if row['status'] != 'retained' else record.parents[4] / 'manifest.json'
        assert source_manifest.exists(), source_manifest
        ledger.append({**{k: row[k] for k in ['task', 'model', 'platform', 'prompt', 'condition', 'global_rep']}, 'status': 'retained', 'source_record': str(record), 'source_record_sha256': c.sha(record), 'source_manifest': str(source_manifest), 'source_protocol_sha256': protocol, 'effective_prompt_sha256': expected})
    assert len(ledger) == 60
    for block, task, selected_prompts in [('legacy_five_1', 'broad_oaxaca', ['zero_shot', 'few_shot']), ('legacy_five_2', 'broad_cohort_stack', ['zero_shot'])]:
        source_manifest = ROOT / 'data/runs/candidates_2026-09-19' / block / 'manifest.json'
        old_manifest = read(source_manifest)
        for job in old_manifest['jobs']:
            if job['task'] != task or job['condition'] != 'agent' or job['prompt'] not in selected_prompts:
                continue
            assert job['model'] == 'luna' and job['platform'] == 'python'
            old_prompt = old_manifest['protocol']['prompts'][c.prompt_key(job)]
            assert old_prompt == c.prepared_prompt(texts[c.prompt_key(job)], 'agent')
            assert c.AGENT_VERIFICATION_INSTRUCTION in old_prompt
            record = source_manifest.parent / 'runs' / old_manifest['protocol_sha256'] / c.job_id(job) / 'attempt_0001/run_record.json'
            assert read(record)['status'] in c.TERMINAL
            ledger.append(dict(**job, status='retained', source_record=str(record), source_record_sha256=c.sha(record), source_manifest=str(source_manifest), source_protocol_sha256=old_manifest['protocol_sha256'], effective_prompt_sha256=hashlib.sha256(old_prompt.encode()).hexdigest()))
    assert len(ledger) == 75
    retained_slots = {tuple(row[k] for k in ['task', 'model', 'platform', 'prompt', 'condition', 'global_rep']) for row in ledger}
    index, receipts = [], []
    for model in ['luna', 'sonnet']:
        model_index = []
        for offset in [0, 5]:
            folder = RAW / model / (str(offset) + '_' + str(offset + 4))
            paths = {}
            for key, text in texts.items():
                path = folder / 'prompts' / (key + '.txt')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                paths[key] = str(path)
            jobs = []
            for rep in range(offset, offset + 5):
                wave = [dict(block='three_' + model + '_' + str(offset), task=task, model=model, platform=platform, prompt=prompt, condition=agency, output_format='answer' if platform == 'stata' else 'json', rep=rep % 5, global_rep=rep) for task in TASKS for platform in ['python', 'r', 'stata'] for prompt in ['zero_shot', 'few_shot'] for agency in ['chatbot', 'self_repair', 'agent'] if (model, platform, agency) != ('luna', 'python', 'chatbot')]
                wave = [job for job in wave if tuple(job[k] for k in ['task', 'model', 'platform', 'prompt', 'condition', 'global_rep']) not in retained_slots]
                random.Random(20260920 + rep).shuffle(wave)
                jobs.extend(wave)
            c.save(folder / 'catalog.json', catalog)
            c.save(folder / 'config.json', dict(catalog_path=str(folder / 'catalog.json'), prompt_paths=paths, jobs=jobs, decision_record=str(DECISION)))
            for action in ['prep', 'check']:
                args = [sys.executable, str(ROOT / 'code/benchmark/prospective_collect_unlimited_additions.py'), action, '--manifest', str(folder / 'manifest.json')]
                if action == 'prep':
                    args += ['--config', str(folder / 'config.json')]
                proc = subprocess.run(args, capture_output=True, text=True)
                receipts.append(dict(model=model, action=action, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))
                c.save(META / 'preparation_receipts.json', receipts)
                assert proc.returncode == 0, proc.stdout + proc.stderr
            manifest = read(folder / 'manifest.json')
            assert all(command[1] is None for command in manifest['protocol']['commands'].values())
            c.save(folder / 'authorization.json', dict(protocol_sha256=manifest['protocol_sha256'], allow_provider_calls=True, decision_record=str(DECISION), scope='Three approved tasks, ten repetitions per cell; retain all compatible existing responses.'))
            entry = dict(batch='three_' + model + '_' + str(offset), manifest=str(folder / 'manifest.json'), planned=len(jobs), protocol_sha256=manifest['protocol_sha256'])
            index.append(entry)
            model_index.append(entry)
            for job in jobs:
                ledger.append(dict(**job, status='planned', manifest=entry['manifest'], protocol_sha256=entry['protocol_sha256']))
        c.save(META / (model + '_collections.json'), model_index)
        launch = dict(name=model, argv=['uv', 'run', '--no-sync', 'python', '-u', str(ROOT / 'code/benchmark/production_queue_unlimited_additions.py'), '--index', str(META / (model + '_collections.json')), '--solo-workers', '4', '--overlap-workers', '4', '--drain-model'], cwd=str(ROOT), env={'UV_CACHE_DIR': '/private/tmp/agents-metrics-uv-cache'}, log=str(META / (model + '_collector.log')), index=str(META / (model + '_collections.json')), batches=[item['batch'] for item in model_index])
        launch['argv'] += ['--batches'] + launch['batches']
        c.save(META / (model + '_launch_commands.json'), [launch])
    slots = [tuple(row[k] for k in ['task', 'model', 'platform', 'prompt', 'condition', 'global_rep']) for row in ledger]
    assert len(slots) == len(set(slots)) == 1080
    c.save(META / 'planned_positions.json', ledger)
    c.save(META / 'collections.json', index)
    c.save(META / 'validation.json', dict(total=1080, retained=75, new=1005, model_new=dict(luna=465, sonnet=540), workers_per_model=4, walltime_limits=None, prompt_identity_verified=True))
    print(json.dumps(read(META / 'validation.json')))


if __name__ == '__main__':
    main()
