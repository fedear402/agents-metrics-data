"""Describe how the Haiku 5.5 runs behaved, next to the paper's Sonnet 4.6 runs of the same task. No model call.

Rule file: quality_reports/decision_records/2026-10-16_haiku55-api-collection.md.
For every Haiku run of the batch and every paper Sonnet run of the same task (data/analysis/main/data/runs.csv),
the script reads the record and the saved Claude streams and reports, per run: provider calls, turns, tool calls,
the agent's shell commands, commands that install packages, use the network, or leave the workspace, the
collector's access findings, verification.txt for agents, stop reasons and errors, thinking tokens, cache writes
by duration, and models. For Stata self-repair runs it also reports whether a repair followed a script whose
ANSWER lines the current (less strict) reader would have accepted.

Outputs in data/collections/2026-10-16_haiku55-api/: <batch>_run_analysis.csv (one row per run, both models) and
<batch>_run_analysis_summary.csv (means by model and condition).

Run from the repository root:
    uv run --no-sync python code/benchmark/workflows/haiku55_api_2026_10_16/analyze.py --manifest M
"""
from pathlib import Path
import argparse
import importlib.util
import json
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
META = ROOT / 'data/collections/2026-10-16_haiku55-api'
RUNS = ROOT / 'data/analysis/main/data/runs.csv'
PATTERNS = {
    'install': r'install\.packages|pip3? install|uv (?:pip|add)|\bssc install|\bnet install|brew install|conda install',
    'network': r'https?://|\b(?:curl|wget)\b|\bgit\s',
    'outside_workspace': r'(?:^|[\s"\'])\.\./|~/|\$HOME\b|/Users/|Path\.home\(',
}
PAPER_ANSWER = re.compile(r'^ANSWER:\s*([A-Za-z0-9_]+)\s+estimate\s*=\s*(\S+)\s+se\s*=\s*(\S+)\s+n_obs\s*=\s*(\S+)\s*$')


def load_runner():
    spec = importlib.util.spec_from_file_location('current_runner', ROOT / 'code/benchmark/collection_runtime/runner.py',
                                                  submodule_search_locations=[])
    sys.path.insert(0, str(ROOT / 'code/benchmark'))
    import collection_runtime  # noqa: F401  (package for relative imports)
    from collection_runtime import runner
    return runner


def stream(path):
    facts = dict(turns=0, tool_calls=0, commands=[], stop_reasons=[], is_error=False, thinking=0, models=set(),
                 cache_5m=0, cache_1h=0)
    for line in path.read_text(errors='replace').splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'assistant':
            message = event.get('message') or {}
            facts['models'].add(message.get('model'))
            for block in message.get('content') or []:
                if isinstance(block, dict) and block.get('type') == 'tool_use':
                    facts['tool_calls'] += 1
                    command = (block.get('input') or {}).get('command')
                    if command:
                        facts['commands'].append(command)
        elif event.get('type') == 'result':
            facts['turns'] += event.get('num_turns') or 0
            facts['stop_reasons'].append(event.get('stop_reason'))
            facts['is_error'] |= bool(event.get('is_error'))
            usage = event.get('usage') or {}
            split = usage.get('cache_creation') or {}
            facts['cache_5m'] += split.get('ephemeral_5m_input_tokens', 0)
            facts['cache_1h'] += split.get('ephemeral_1h_input_tokens', 0)
            facts['thinking'] += (usage.get('output_tokens_details') or {}).get('thinking_tokens', 0)
            for name in (event.get('modelUsage') or {}):
                facts['models'].add(name)
    return facts


def strict_repairs(folder, runner):
    """Repairs of a Stata self-repair run that followed output the current reader accepts and the strict one rejects."""
    count = 0
    for execution in sorted(folder.glob('execution_attempt*.json')):
        number = int(re.search(r'attempt(\d+)', execution.name)[1])
        if not (folder / f'prompt_attempt{number + 1}.txt').exists():
            continue
        text = json.loads(execution.read_text()).get('stata_log_tail') or ''
        lines = [line.strip() for line in text.splitlines() if line.strip().startswith('ANSWER:')]
        if any(runner.ANSWER.match(line) and not PAPER_ANSWER.match(line) for line in lines):
            count += 1
    return count


def describe(row, model, runner):
    record_path = ROOT / row['record_path']
    folder = record_path.parent
    record = json.loads(record_path.read_text())
    facts = dict(turns=0, tool_calls=0, commands=[], stop_reasons=[], is_error=False, thinking=0, models=set(),
                 cache_5m=0, cache_1h=0)
    for path in sorted(folder.glob('provider_attempt*.stdout')):
        one = stream(path)
        for key in ('turns', 'tool_calls', 'thinking', 'cache_5m', 'cache_1h'):
            facts[key] += one[key]
        facts['commands'] += one['commands']
        facts['stop_reasons'] += one['stop_reasons']
        facts['is_error'] |= one['is_error']
        facts['models'] |= one['models']
    text = '\n'.join(facts['commands'])
    flags = {name: len(re.findall(pattern, text, re.I)) for name, pattern in PATTERNS.items()}
    final = folder / 'final_workspace'
    return dict(model=model, task=row['task'], platform=row['platform'], prompt=row['prompt'],
                condition=row['condition'], global_rep=row['global_rep'], primary=bool(row['primary']),
                failure_category=row['failure_category'], calls=row['calls_count'],
                provider_seconds=row['provider_seconds'], turns=facts['turns'], tool_calls=facts['tool_calls'],
                shell_commands=len(facts['commands']), install_commands=flags['install'],
                network_commands=flags['network'], outside_workspace_commands=flags['outside_workspace'],
                collector_access_findings=len(record.get('access_findings') or []),
                verification_txt=(final / 'verification.txt').is_file() if row['condition'] == 'agent' else None,
                stop_reasons='|'.join(sorted({str(s) for s in facts['stop_reasons']})), provider_error=facts['is_error'],
                thinking_tokens=facts['thinking'], cache_write_5m=facts['cache_5m'], cache_write_1h=facts['cache_1h'],
                models='|'.join(sorted(m for m in facts['models'] if m)),
                strict_reader_repairs=strict_repairs(folder, runner)
                if row['platform'] == 'stata' and row['condition'] == 'self_repair' else None,
                record_path=row['record_path'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text())
    batch = manifest['jobs'][0]['block']
    runner = load_runner()
    haiku = pd.read_csv(META / f'{batch}_paper_rows.csv')
    task = haiku.task.iloc[0]
    paper = pd.read_csv(RUNS, low_memory=False)
    paper = paper[(paper.task == task) & (paper.model == 'sonnet')]
    rows = [describe(r, 'haiku 5.5', runner) for r in haiku.to_dict('records')]
    rows += [describe(r, 'sonnet 4.6 (paper)', runner) for r in paper.to_dict('records')]
    frame = pd.DataFrame(rows)
    frame.to_csv(META / f'{batch}_run_analysis.csv', index=False)
    numeric = ['primary', 'calls', 'provider_seconds', 'turns', 'tool_calls', 'shell_commands', 'install_commands',
               'network_commands', 'outside_workspace_commands', 'collector_access_findings', 'thinking_tokens',
               'cache_write_5m', 'cache_write_1h']
    summary = frame.groupby(['model', 'condition'])[numeric].mean().round(3)
    summary['runs'] = frame.groupby(['model', 'condition']).size()
    summary.to_csv(META / f'{batch}_run_analysis_summary.csv')
    print(summary.to_string())
    agents = frame[frame.condition == 'agent']
    print('verification.txt share (agent):', agents.groupby('model').verification_txt.mean().round(3).to_dict())
    print('provider errors:', frame.groupby('model').provider_error.sum().to_dict())
    print('stop reasons:', frame.groupby('model').stop_reasons.value_counts().to_dict())
    print('models seen:', frame.groupby('model').models.value_counts().to_dict())
    print('strict-reader repairs (Stata self-repair):', frame.groupby('model').strict_reader_repairs.sum().to_dict())


if __name__ == '__main__':
    main()
