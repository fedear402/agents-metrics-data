"""Shared plan for the rerun of the original tasks, the 12 added tasks, and flagged runs.

Decision record: quality_reports/decision_records/2026-09-25_original20-fewshot-rerun.md.

The functions here read local files, render prompts, check the new example
files, and build the job list. Nothing here calls a model provider.
prepare.py, check.py, and launch.py are the command-line entry points.

Scopes (SCOPES):
- few_shot, few_shot_and_zero_shot: the 20 original tasks (3,600 or 7,200 positions).
- original20_added12_both_prompts: the 20 original tasks and the 12 added tasks without
  the full example layout, both prompts (7,200 + 4,320 = 11,520 positions).
- flagged_runs: only the selected positions listed in a run-list file, with their
  production prompts.
- original20_added12_both_prompts_and_flagged_runs: the 11,520 positions plus the listed
  positions of the other 8 added tasks, with their production prompts.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect as c  # noqa: E402

ORIGINAL_TASKS = ('anova_oneway', 'arima_forecast', 'chisq_indep', 'cluster_ols', 'cooks_distance', 'cox_hazard',
                  'iv2sls_robust', 'logit', 'lp_irf', 'ols_robust', 'poisson_offset', 'probit', 'rdd_sharp',
                  'staggered_did', 'ttest_welch', 'two_way_cluster', 'twoway_fe', 'var_granger', 'var_irf', 'vif')
TASKS = ORIGINAL_TASKS  # Name kept for the scripts and tests of the 2026-09-25 scopes.
# The 12 added tasks whose production example block lacks the full layout. They get new blocks.
ADDED12_TASKS = ('applied_multivintage_trade_iv', 'applied_nonrecipient_population_iv', 'broad_cohort_stack',
                 'broad_oaxaca', 'hard_conditional_logit_welfare', 'hard_heteroskedastic_probit', 'hard_panel_lp_iv',
                 'hard_poisson_hurdle_effects', 'hard_truncated_normal_effects', 'lp04_ppml', 'lp11_duration',
                 'pilot_probit_interaction_ame')
# The 8 added tasks whose production block already has the full layout. Their prompts stay as collected.
FULL_LAYOUT_TASKS = ('applied_cumulative_hac', 'applied_fuzzy_rd', 'applied_shift_share', 'broad_ar_confidence_set',
                     'broad_multinomial_logit', 'broad_ordered_logit', 'broad_tobit', 'hard_rolling')
ADDED20_TASKS = tuple(sorted(ADDED12_TASKS + FULL_LAYOUT_TASKS))
RERUN32_TASKS = ORIGINAL_TASKS + ADDED12_TASKS
ALL_TASKS = ORIGINAL_TASKS + ADDED20_TASKS
MODELS = ('luna', 'sonnet')
PLATFORMS = ('python', 'r', 'stata')
AGENCIES = ('chatbot', 'self_repair', 'agent')
PROMPTS = ('zero_shot', 'few_shot')
REPS = 10
BLOCK_REPS = 5  # prospective_collect.validate_jobs accepts local repetitions 0 through 4.
SEED = 20260925
WORKERS = 8  # Chosen worker limit per block, as in the 2026-09-21 collection. Not a measured capacity.
# Chosen limit for one example check. It is a check policy, not a measured runtime.
EXAMPLE_TIMEOUT_SECONDS = 600
# tasks: positions of these tasks get new few-shot blocks (and production zero-shot prompts).
# run_list: the listed positions outside these tasks are added with their production prompts.
# prefix: names the blocks and the collection folders.
SCOPES = {
    'few_shot': dict(tasks=ORIGINAL_TASKS, prompts=('few_shot',), run_list=False, prefix='original20'),
    'few_shot_and_zero_shot': dict(tasks=ORIGINAL_TASKS, prompts=PROMPTS, run_list=False, prefix='original20'),
    'original20_added12_both_prompts': dict(tasks=RERUN32_TASKS, prompts=PROMPTS, run_list=False, prefix='rerun32'),
    'flagged_runs': dict(tasks=(), prompts=(), run_list=True, prefix='flagged'),
    'original20_added12_both_prompts_and_flagged_runs': dict(tasks=RERUN32_TASKS, prompts=PROMPTS, run_list=True,
                                                             prefix='rerun32f'),
}
# Collection folder under data/collections and raw-run folder under data/runs, by block prefix.
FOLDERS = {'original20': ('2026-09-25_original20-fewshot', 'original20_fewshot_2026-09-25'),
           'rerun32': ('2026-09-26_rerun32-both-prompts', 'rerun32_both_prompts_2026-09-26'),
           'flagged': ('2026-09-26_flagged-runs', 'flagged_runs_2026-09-26'),
           'rerun32f': ('2026-09-26_rerun32-and-flagged-runs', 'rerun32_and_flagged_runs_2026-09-26')}

EXAMPLES = ROOT / 'data/task_candidates/2026-09-25_original20_examples'
ADDED12_EXAMPLES = ROOT / 'data/task_candidates/2026-09-26_added12_examples'
EXAMPLE_FILES = {'python': 'example.py', 'r': 'example.R', 'stata': 'example.do'}
PARTS_FILE = 'block_parts.json'  # title, context, explanation, and optional platform notes of one task
EXPECTED_FILES = {'python': 'expected_python.json', 'r': 'expected_r.json', 'stata': 'expected_stata.txt'}
CATALOG = ROOT / 'data/production/main/catalog.json'
ADDED_PROMPTS = ROOT / 'data/production/added20_prompts_2026-09-26/prompts'  # production_prompts.py --write
PROMPT_FOLDERS = (ROOT / 'data/production/original10_2026-09-19/prompts',
                  ROOT / 'data/collections/2026-09-17_live-check-and-checkpoint/work/old_prompts/prompts',
                  ADDED_PROMPTS)
BASELINE = ROOT / 'data/production/main/baseline_runs.csv'
ANALYSIS_RUNS = ROOT / 'data/analysis/main/data/runs.csv'
DECISION = ROOT / 'quality_reports/decision_records/2026-09-25_original20-fewshot-rerun.md'
# Default run-list file of the flagged-run scopes: the 79 flagged runs of the 2026-09-26 exposure audit
# (quality_reports/audits/2026-09-26_rerun-throughput/exposure_and_flagged_runs.md, Section 4).
# A CSV with a position_id column is the preferred form. Other text files also work (see read_run_list).
RUN_LIST = ROOT / 'quality_reports/audits/2026-09-26_rerun-throughput/work/exposure/rerun_list.csv'
SELECTED_RECORDS = ROOT / 'data/analysis/main/data/selected_records.csv'  # Record paths after the data move.
COLLECTIONS = ROOT / 'data/collections'
RUNS = ROOT / 'data/runs'
COLLECTOR = ROOT / 'code/benchmark/prospective_collect_unlimited_additions.py'
QUEUE = ROOT / 'code/benchmark/production_queue_unlimited_additions.py'
SUPERVISOR = ROOT / 'code/benchmark/collection_supervisor_unlimited_additions.py'
SONNET_GUARD = ROOT / 'data/production/paper22_2026-09-18/work/environment_recovery/shared_python_write_deny.sb'
STATA_LOCK = '/private/tmp/agents_metrics_native_stata.lock'

BLOCK_START = 'Few-shot demonstration:\n'  # First line of the original-task production block.
TASK_MARK = '\n\nTask: '
FENCE = {'python': 'python', 'r': 'r', 'stata': 'stata'}
# The new block uses the layout of the added tasks with a full example block: the
# Python builder of 2026-09-16 (data/collections/2026-09-16_clearer-examples/src/prompts/build_bank.py)
# and the R and Stata builder of 2026-09-17
# (data/collections/2026-09-17_live-check-and-checkpoint/src/new_prompts/build.py, native_block).
# Only Python has a relation line, as in the added tasks. The new examples write result.json
# with the scored task's identifier and row names, so the Python line states that relation.
LABELS = ('Related example: ', 'Separate example context: ', 'Explanation of this example: ')
RELATION = {'python': ("The example uses separate data and writes result.json. It uses the scored task's "
                       'identifier and result names. The scored task below must read its own input.csv '
                       'and write result.json.')}
# Two added tasks read several named CSV files and no input.csv, so their relation line names the supplied files.
SUPPLIED_FILE_TASKS = ('applied_multivintage_trade_iv', 'applied_nonrecipient_population_iv')
RELATION_SUPPLIED_FILES = {'python': ("The example uses separate data and writes result.json. It uses the scored "
                                      "task's identifier and result names. The scored task below must read its own "
                                      'supplied CSV files and write result.json.')}
OUTPUT_HEADER = {'python': 'Expected result.json:\n```json\n', 'r': 'Expected result.json:\n```json\n',
                 'stata': 'Expected printed output:\n```text\n'}
BLOCK_LAYOUT = dict(labels=list(LABELS), relation=RELATION, relation_supplied_files=RELATION_SUPPLIED_FILES,
                    supplied_file_tasks=list(SUPPLIED_FILE_TASKS), output_header=OUTPUT_HEADER)
# Chosen relative tolerance between a fresh example output and its stored expected output.
OUTPUT_RTOL = 1e-8
NUMBER = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
STATA_ERROR = re.compile(r'^r\(\d+\);\s*$', re.MULTILINE)
POSITION = re.compile(r'(?<![A-Za-z0-9_])([a-z][a-z0-9_]*?__(?:luna|sonnet)__(?:python|r|stata)__'
                      r'(?:zero_shot|few_shot)__(?:chatbot|self_repair|agent)__\d+)(?![A-Za-z0-9_])')
RECORD_PATH = re.compile(r'(/[^\s`\'"|,;()]+/run_record\.json)')
SLOT_FIELDS = ('task', 'model', 'platform', 'prompt', 'condition', 'global_rep')


def read(path):
    return json.loads(Path(path).read_text())


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def output_format(platform):
    return 'answer' if platform == 'stata' else 'json'


def prompt_keys(prompts=PROMPTS, tasks=ORIGINAL_TASKS):
    return [c.prompt_key(dict(task=task, platform=platform, prompt=prompt, condition=agency,
                              output_format=output_format(platform)))
            for task in tasks for platform in PLATFORMS for prompt in prompts for agency in AGENCIES]


def scope_tasks(scope):
    return SCOPES[scope]['tasks']


def scope_prompts(scope):
    return SCOPES[scope]['prompts']


def uses_run_list(scope):
    return SCOPES[scope]['run_list']


def meta(scope):
    """Collection folder of a scope under data/collections."""
    return COLLECTIONS / FOLDERS[SCOPES[scope]['prefix']][0]


def raw(scope):
    """Raw-run folder of a scope under data/runs."""
    return RUNS / FOLDERS[SCOPES[scope]['prefix']][1]


def prepared_collections():
    """Return {scope: collection folder} for each collection folder that holds a plan.json."""
    found = {}
    for folder in sorted({meta(scope) for scope in SCOPES}):
        if (folder / 'plan.json').is_file():
            found[read(folder / 'plan.json')['scope']] = folder
    return found


def find_collection(scope=None):
    """Return (scope, folder) of the prepared collection to check or launch."""
    found = prepared_collections()
    if scope is not None:
        if scope not in found or found[scope] != meta(scope):
            raise SystemExit(f'No prepared collection for scope {scope} at {meta(scope)}. Run prepare.py --write first.')
        return scope, found[scope]
    if not found:
        raise SystemExit('No prepared collection. Run prepare.py --write first.')
    if len(found) > 1:
        raise SystemExit('Several prepared collections (' + ', '.join(sorted(found)) + '). Pass --scope.')
    return next(iter(found.items()))


# Decision record ---------------------------------------------------------------------------

def decision_state(path=None):
    """Return (status, approved scope) from lines 'Status: X' and 'Approved scope: Y'."""
    path = Path(path or DECISION)
    if not path.is_file():
        return None, None
    text = path.read_text()
    status = re.search(r'^Status:\s*([A-Z]+)\b', text, re.MULTILINE)
    scope = re.search(r'^Approved scope:\s*([a-z0-9_]+)\s*$', text, re.MULTILINE)
    return (status.group(1) if status else None), (scope.group(1) if scope else None)


def approved_run_list(path=None):
    """Return the SHA-256 on the line 'Approved run list sha256: <hex>', or None."""
    path = Path(path or DECISION)
    if not path.is_file():
        return None
    found = re.search(r'^Approved run list sha256:\s*([0-9a-f]{64})\s*$', path.read_text(), re.MULTILINE)
    return found.group(1) if found else None


def decision_problems(scope, path=None, run_list_sha256=None):
    status, approved = decision_state(path)
    problems = []
    if status != 'APPROVED':
        problems.append(f'decision record status is {status!r}, not APPROVED')
    if approved != scope:
        problems.append(f'decision record approved scope is {approved!r}, not {scope!r}')
    if uses_run_list(scope):
        listed = approved_run_list(path)
        if run_list_sha256 is None:
            problems.append('run list missing')
        elif listed != run_list_sha256:
            problems.append(f'decision record approved run list sha256 is {listed!r}, not {run_list_sha256!r}')
    return problems


# Run list ----------------------------------------------------------------------------------

def parse_position(position_id):
    task, model, platform, prompt, condition, rep = position_id.split('__')
    return dict(task=task, model=model, platform=platform, prompt=prompt, condition=condition, global_rep=int(rep))


def position_id(row):
    return '__'.join(str(row[k]) for k in SLOT_FIELDS)


def read_run_list(path=None, baseline=None):
    """Return (position ids, sha256, problems) of a run-list file.

    A CSV file whose header has a position_id column lists one position id
    (task__model__language__prompt__agency__repetition) per row. Only that column is read.
    Any other text file (Markdown, JSON, plain text) names each selected position by its
    position id or by the absolute path of its selected run_record.json, and other text is
    ignored. In such a file, every position id counts, also one that the prose only mentions.
    Names that are not selected positions in baseline_runs.csv are reported as problems.
    """
    path = Path(path or RUN_LIST)
    if not path.is_file():
        return [], None, [f'run list missing: {path}']
    text = path.read_text()
    with Path(baseline or BASELINE).open() as stream:
        rows = list(csv.DictReader(stream))
    selected = {row['position_id'] for row in rows}
    by_record = {row['record_path']: row['position_id'] for row in rows}
    if SELECTED_RECORDS.is_file() and baseline is None:
        with SELECTED_RECORDS.open() as stream:
            by_record.update({row['record_path']: row['position_id'] for row in csv.DictReader(stream)})
    found, problems = set(), []
    header = next(csv.reader([text.splitlines()[0]])) if text.strip() else []
    if 'position_id' in header:
        for row in csv.DictReader(text.splitlines()):
            name = (row.get('position_id') or '').strip()
            if name in selected:
                found.add(name)
            elif name:
                problems.append(f'run list names {name}, which is not a selected position')
        if not found and not problems:
            problems.append(f'run list {path} names no selected position')
        return sorted(found), c.sha(path), sorted(set(problems))
    for name in POSITION.findall(text):
        if name in selected:
            found.add(name)
        else:
            problems.append(f'run list names {name}, which is not a selected position')
    for record in RECORD_PATH.findall(text):
        if record in by_record:
            found.add(by_record[record])
        else:
            problems.append(f'run list names {record}, which is not a selected run record')
    if not found and not problems:
        problems.append(f'run list {path} names no selected position')
    return sorted(found), c.sha(path), sorted(set(problems))


# Source prompts ----------------------------------------------------------------------------

def load_source_prompts(folders=None):
    """Return {key: text} and {key: provenance} for the 720 production prompts of the 40 tasks."""
    texts, provenance = {}, {}
    for folder in folders or PROMPT_FOLDERS:
        for path in sorted(Path(folder).glob('*.txt')):
            if path.stem in texts:
                raise ValueError(f'Duplicate production prompt {path.stem}')
            texts[path.stem] = path.read_text()
            provenance[path.stem] = dict(path=str(path), sha256=c.sha(path))
    if folders is None:
        expected = set(prompt_keys(tasks=ALL_TASKS))
        if set(texts) != expected:
            missing, extra = sorted(expected - set(texts)), sorted(set(texts) - expected)
            raise ValueError(f'Production prompts differ from the 720 expected keys: missing {missing[:3]}, '
                             f'extra {extra[:3]}. Run production_prompts.py --write for the added tasks.')
    return texts, provenance


def split_block(text, zero=None):
    """Return (head, block, tail) of a production few-shot prompt.

    Original tasks: the block runs from 'Few-shot demonstration:' to the task line.
    Added tasks: the block is the text that the few-shot prompt adds to the zero-shot prompt.
    Among the equal insertion points, the block starts at the last line start.
    """
    if BLOCK_START in text:
        if text.count(BLOCK_START) != 1:
            raise ValueError('Few-shot prompt needs exactly one demonstration block')
        start = text.index(BLOCK_START)
        end = text.find(TASK_MARK, start)
        if end < 0 or text.count(TASK_MARK) != 1:
            raise ValueError('Few-shot prompt needs exactly one task line after the block')
        return text[:start], text[start:end], text[end:]
    if zero is None:
        raise ValueError('This few-shot prompt has no demonstration header. Pass its zero-shot prompt.')
    size = len(text) - len(zero)
    if size <= 0:
        raise ValueError('Few-shot prompt is not longer than its zero-shot prompt')
    prefix = 0
    while prefix < len(zero) and text[prefix] == zero[prefix]:
        prefix += 1
    suffix = 0
    while suffix < len(zero) and text[-1 - suffix] == zero[-1 - suffix]:
        suffix += 1
    starts = [a for a in range(max(0, len(zero) - suffix), prefix + 1) if a == 0 or text[a - 1] == '\n']
    if not starts or text[:starts[-1]] + text[starts[-1] + size:] != zero:
        raise ValueError('Few-shot prompt does not equal its zero-shot prompt with one inserted block')
    start = starts[-1]
    return text[:start], text[start:start + size], text[start + size:]


def explanation_for(parts, platform):
    """The shared explanation plus the optional note for one language."""
    note = (parts.get('platform_notes') or {}).get(platform)
    return parts['explanation'] + (' ' + note if note else '')


def parts_problems(parts):
    """Problems of a block_parts.json object. Each text field is one line of prose."""
    if not isinstance(parts, dict):
        return ['block parts are not a JSON object']
    problems = []
    notes = parts.get('platform_notes') or {}
    if not isinstance(notes, dict) or set(notes) - set(PLATFORMS):
        problems.append('platform_notes must map python, r, or stata to text')
        notes = {}
    fields = [(name, parts.get(name)) for name in ('title', 'context', 'explanation')]
    fields += [(f'platform_notes.{name}', value) for name, value in notes.items()]
    for name, value in fields:
        if not isinstance(value, str) or not value.strip():
            problems.append(f'block part {name} is missing or empty')
        elif '\n' in value or '```' in value or value != value.strip():
            problems.append(f'block part {name} must be one trimmed line without a Markdown fence')
    return problems


def relation_line(platform, task=None):
    lines = RELATION_SUPPLIED_FILES if task in SUPPLIED_FILE_TASKS else RELATION
    return lines.get(platform)


def example_block(platform, code, parts, expected, task=None):
    """Build the example block in the layout of the added tasks with a full example block."""
    if '```' in code or '```' in expected:
        raise ValueError('Example code and expected output must not contain a Markdown fence')
    problems = parts_problems(parts)
    if problems:
        raise ValueError('; '.join(problems))
    lines = [LABELS[0] + parts['title'], LABELS[1] + parts['context'], LABELS[2] + explanation_for(parts, platform)]
    if relation_line(platform, task):
        lines.append(relation_line(platform, task))
    return ('\n'.join(lines) + f'\n```{FENCE[platform]}\n' + code.rstrip('\n') + '\n```\n'
            + OUTPUT_HEADER[platform] + expected.strip('\n') + '\n```\n')


def render_few_shot(production_few, production_zero, block):
    """Replace the production example block and check the zero-shot identity.

    The new block keeps the blank lines that followed the old block.
    """
    head, old, tail = split_block(production_few, production_zero)
    if head + tail != production_zero:
        raise ValueError('Production few-shot prompt without its block differs from the zero-shot prompt')
    trailing = len(old) - len(old.rstrip('\n'))
    region = block.rstrip('\n') + '\n' * max(trailing, 1)
    rendered = head + region + tail
    new_head, new_region, new_tail = split_block(rendered, production_zero) if BLOCK_START not in rendered else (
        head, region, tail)
    if new_head + new_tail != production_zero or rendered.replace(region, '', 1) != production_zero:
        raise ValueError('Rendered few-shot prompt without its block differs from the zero-shot prompt')
    return rendered


def render_prompts(sources, examples, scope, tasks=None):
    """Return {key: text} of the scope's tasks. examples maps (task, platform) to (code, parts, expected)."""
    tasks = scope_tasks(scope) if tasks is None else tasks
    prompts = {}
    for key in prompt_keys(scope_prompts(scope), tasks=tasks):
        task, platform, prompt, agency, fmt = key.split('__')
        if prompt == 'zero_shot':
            prompts[key] = sources[key]
            continue
        zero = sources['__'.join((task, platform, 'zero_shot', agency, fmt))]
        block = example_block(platform, *examples[(task, platform)], task=task)
        prompts[key] = render_few_shot(sources[key], zero, block)
    for key, text in prompts.items():
        c.prepared_prompt(text, key.split('__')[3])  # The collector's agent instruction must apply.
    return prompts


def flagged_extra(scope, flagged):
    """Listed positions that the scope adds with their production prompts (outside the scope's tasks)."""
    if not uses_run_list(scope):
        return []
    tasks = set(scope_tasks(scope))
    return [parse_position(name) for name in sorted(set(flagged)) if parse_position(name)['task'] not in tasks]


def production_prompts_for(slots, sources):
    """Return {key: production text} for listed positions."""
    prompts = {}
    for slot in slots:
        key = c.prompt_key(dict(slot, output_format=output_format(slot['platform'])))
        prompts[key] = sources[key]
        c.prepared_prompt(sources[key], slot['condition'])
    return prompts


def scope_prompt_texts(scope, sources, examples, flagged=()):
    """All prompts of a scope: rendered prompts of its tasks and production prompts of the added positions."""
    prompts = render_prompts(sources, examples, scope) if scope_tasks(scope) else {}
    prompts.update(production_prompts_for(flagged_extra(scope, flagged), sources))
    return prompts


def prompt_source(scope, job):
    if job['task'] in scope_tasks(scope):
        return 'new_example_block' if job['prompt'] == 'few_shot' else 'production_zero_shot'
    return 'production'


# Example files -----------------------------------------------------------------------------

def example_folder(task):
    return EXAMPLES if task in ORIGINAL_TASKS else ADDED12_EXAMPLES


def example_inventory(tasks=ORIGINAL_TASKS):
    """Return {(task, platform): path or None} for the required example files of the tasks."""
    inventory = {}
    for task in tasks:
        for platform, name in EXAMPLE_FILES.items():
            path = example_folder(task) / task / name
            inventory[(task, platform)] = path if path.is_file() else None
    return inventory


def parts_path(task, folder=None):
    return Path(folder or example_folder(task)) / task / PARTS_FILE


def expected_path(task, platform, folder=None):
    return Path(folder or example_folder(task)) / task / EXPECTED_FILES[platform]  # The text that the block shows.


def read_parts(path):
    """Return (parts, problems) from a block_parts.json file."""
    path = Path(path)
    if not path.is_file():
        return None, [f'{PARTS_FILE} missing']
    try:
        parts = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return None, [f'{PARTS_FILE} is not valid JSON: {error}']
    problems = parts_problems(parts)
    return (None if problems else parts), problems


def block_inputs(task_folder, platform):
    """Return (parts, expected text, problems) of one example block from its task folder."""
    parts, problems = read_parts(Path(task_folder) / PARTS_FILE)
    path = Path(task_folder) / EXPECTED_FILES[platform]
    expected = path.read_text() if path.is_file() else None
    if expected is None:
        problems.append(f'{EXPECTED_FILES[platform]} missing')
    elif not expected.strip() or '```' in expected:
        problems.append(f'{EXPECTED_FILES[platform]} is empty or contains a Markdown fence')
    return parts, expected, problems


def finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def primary_tolerance(reference):
    """Tolerance of the primary outcome: max(0.01 x |reference|, 0.005)."""
    return max(0.01 * abs(float(reference)), 0.005)


def static_example_problems(text, spec, subject='example'):
    """Failures that the example text alone shows. subject names the text in the messages."""
    failures = []
    if not text.strip():
        failures.append(f'{subject} is empty')
    if '```' in text:
        failures.append(f'{subject} contains a Markdown fence')
    if spec.get('data_file') and spec['data_file'] in text:
        failures.append(f"{subject} names the scored data file {spec['data_file']}")
    if 'data/tasks/inputs' in text:
        failures.append(f'{subject} names the task input folder')
    for row in spec['reference']['estimates']:
        value = row.get('estimate')
        if finite(value) and abs(float(value)) >= 1e-3 and not float(value).is_integer():
            if f'{float(value):.6g}' in text:
                failures.append(f"{subject} contains the reference value of {row['term']}")
    return failures


def literal_matches(text, value):
    """True when a number in text with at least four significant digits equals value at its own precision."""
    for token in NUMBER.findall(text):
        mantissa, _, exponent = token.lower().partition('e')
        if len(mantissa.lstrip('-').replace('.', '').lstrip('0')) < 4:
            continue
        decimals = len(mantissa.partition('.')[2])
        step = 10.0 ** (int(exponent or 0) - decimals)
        if abs(float(token) - value) <= 0.5 * step * (1 + 1e-9):
            return True
    return False


def block_text_problems(parts, expected, platform, spec):
    """Failures of the block text around the code: title, context, explanation, and expected output."""
    if parts is None or expected is None:
        return []
    text = '\n'.join((parts['title'], parts['context'], explanation_for(parts, platform), expected))
    failures = static_example_problems(text, spec, subject='block text')
    for row in spec['reference']['estimates']:
        value = row.get('estimate')
        if finite(value) and abs(float(value)) >= 1e-3 and not float(value).is_integer():
            failure = f"block text contains the reference value of {row['term']}"
            if failure not in failures and literal_matches(text, float(value)):
                failures.append(failure)
    return failures


def render_output(platform, cwd):
    """Return the expected-output text of one run: indented JSON, or the ANSWER lines of the Stata log."""
    if platform == 'stata':
        log = Path(cwd) / 'example.log'
        if not log.is_file():
            return None
        lines = [re.sub(r'=\s+', '=', line.rstrip()) for line in log.read_text(errors='replace').splitlines()
                 if line.startswith('ANSWER:')]
        return '\n'.join(lines) + '\n' if lines else None
    for name in ('result.json', 'example_result.json'):
        path = Path(cwd) / name
        if path.is_file():
            try:
                return json.dumps(json.loads(path.read_text()), indent=2, allow_nan=False) + '\n'
            except (json.JSONDecodeError, ValueError):
                return None
    return None


def same_output(expected, actual, rtol=OUTPUT_RTOL):
    """True when the texts agree outside numbers and every number agrees within rtol."""
    if expected is None or actual is None:
        return False
    if NUMBER.sub('#', expected.strip()) != NUMBER.sub('#', actual.strip()):
        return False
    pairs = zip(NUMBER.findall(expected), NUMBER.findall(actual))
    return all(math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=1e-12) for a, b in pairs)


def parse_json_output(cwd):
    """Return (estimates, problems) from result.json or example_result.json."""
    for name in ('result.json', 'example_result.json'):
        path = Path(cwd) / name
        if path.is_file():
            break
    else:
        return {}, ['example wrote neither result.json nor example_result.json']
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return {}, [f'{path.name} is not valid JSON: {error}']
    rows = data.get('estimates') if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        return {}, [f'{path.name} has no estimates list']
    estimates, problems = {}, []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('term'), str):
            problems.append(f'{path.name} has an estimate row without a term')
            continue
        if not finite(row.get('estimate')):
            problems.append(f"{path.name}: estimate of {row['term']} is not a finite number")
            continue
        estimates[row['term']] = float(row['estimate'])
    return estimates, problems


def parse_stata_output(cwd, printed_estimates):
    log = Path(cwd) / 'example.log'
    if not log.is_file():
        return {}, ['example.log is missing']
    text = log.read_text(errors='replace')
    problems = []
    if STATA_ERROR.search(text):
        problems.append('example.log contains a Stata error code')
    parsed, notes = printed_estimates(text)
    problems += [f'example.log: {note}' for note in notes]
    estimates = {term: row['estimate'] for term, row in parsed.items() if finite(row.get('estimate'))}
    if not estimates:
        problems.append('example.log has no parseable ANSWER line')
    return estimates, problems


def compare_with_reference(estimates, spec):
    """Return (failures, warnings, matched terms) of the example outputs against the task reference."""
    matched = []
    for row in spec['reference']['estimates']:
        value = estimates.get(row['term'])
        if value is not None and abs(value - float(row['estimate'])) <= primary_tolerance(row['estimate']):
            matched.append(row['term'])
    failures, warnings = [], []
    if matched and set(matched) == set(spec['estimands']):
        failures.append('example output reproduces the reference value of every scored term')
    else:
        for term in matched:
            reference = next(float(r['estimate']) for r in spec['reference']['estimates'] if r['term'] == term)
            if not reference.is_integer():
                warnings.append(f'example output matches the reference value of {term}')
    return failures, warnings, matched


def example_command(platform, name, tools):
    if platform == 'python':
        return [tools['python'], name]
    if platform == 'r':
        return [tools['rscript'], '--vanilla', name]
    return [tools['stata'], '-b', 'do', name]


def run_command(argv, cwd, env, timeout, platform):
    """Run one example script. Stata runs hold the collectors' native Stata lock."""
    started = time.monotonic()

    def call():
        return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)

    try:
        if platform == 'stata':
            import fcntl
            with open(STATA_LOCK, 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                process = call()
        else:
            process = call()
        return dict(returncode=process.returncode, stdout=process.stdout[-4000:], stderr=process.stderr[-4000:],
                    seconds=round(time.monotonic() - started, 2), timed_out=False)
    except subprocess.TimeoutExpired:
        return dict(returncode=None, stdout='', stderr='', seconds=round(time.monotonic() - started, 2), timed_out=True)


def run_example(platform, text, tools, runner=run_command, timeout=EXAMPLE_TIMEOUT_SECONDS):
    """Run one example alone in an empty temporary directory.

    Return (run, estimates, problems, output text). The output text is what the block shows.
    """
    with tempfile.TemporaryDirectory(prefix='original20_example_', dir='/private/tmp') as work:
        work = Path(work)
        name = EXAMPLE_FILES[platform]
        (work / name).write_text(text)
        run = runner(example_command(platform, name, tools), work, tools.get('env'), timeout, platform)
        problems = []
        if run['timed_out']:
            problems.append(f'example did not finish within {timeout} seconds')
        elif run['returncode'] != 0:
            problems.append(f"example exited with status {run['returncode']}")
        if platform == 'stata':
            estimates, parsed = parse_stata_output(work, tools['printed_estimates'])
        else:
            estimates, parsed = parse_json_output(work)
        return run, estimates, problems + parsed, render_output(platform, work)


def check_example(task, platform, path, spec, tools, runner=run_command, timeout=EXAMPLE_TIMEOUT_SECONDS):
    """Run one example and check its output, its block text, and its stored expected output.

    The block parts and the expected output are read from the folder of the example file.
    """
    result = dict(task=task, platform=platform, path=str(path) if path else None, passed=False,
                  failures=[], warnings=[], estimates={}, matched_reference_terms=[])
    if path is None:
        result['failures'].append(f'{EXAMPLE_FILES[platform]} missing')
        return result
    text = Path(path).read_text()
    result['sha256'] = c.sha(path)
    result['failures'] += static_example_problems(text, spec)
    parts, expected, block_problems = block_inputs(Path(path).parent, platform)
    result['failures'] += block_problems + block_text_problems(parts, expected, platform, spec)
    run, estimates, problems, output = run_example(platform, text, tools, runner, timeout)
    result['run'] = run
    result['failures'] += problems
    result['estimates'] = estimates
    result['output_text'] = output
    if expected is not None and not same_output(expected, output):
        result['failures'].append(f'example output differs from {EXPECTED_FILES[platform]}')
    failures, warnings, matched = compare_with_reference(estimates, spec)
    result['failures'] += failures
    result['warnings'] += warnings
    result['matched_reference_terms'] = matched
    result['prints_scored_names'] = sorted(set(estimates) & set(spec['estimands']))
    result['passed'] = not result['failures']
    return result


def runtime_tools(work):
    """Interpreters, environment, and the Stata parser of the central runtime."""
    runtime = c.load_runtime(Path(work))
    env = runtime.c.environment()
    rscript = subprocess.run(['/usr/bin/which', 'Rscript'], env=env, capture_output=True, text=True).stdout.strip()
    return dict(python=sys.executable, rscript=rscript or 'Rscript', stata=str(runtime.c.harness.STATA_BIN),
                env=env, printed_estimates=runtime.c.prior.base.printed_estimates)


def load_examples(inventory):
    """Return {(task, platform): (code, parts, expected)} for present example files.

    The block parts and the expected output come from the folder of each example file.
    """
    examples = {}
    for key, path in inventory.items():
        if path is None:
            continue
        parts, expected, problems = block_inputs(path.parent, key[1])
        if problems:
            raise ValueError(f'{key[0]}/{key[1]}: ' + '; '.join(problems))
        examples[key] = (path.read_text(), parts, expected)
    return examples


def block_file_hashes(tasks=ORIGINAL_TASKS):
    """SHA-256 of the block parts of each task and of each expected output file (None when missing)."""
    parts = {task: (c.sha(parts_path(task)) if parts_path(task).is_file() else None) for task in tasks}
    expected = {f'{task}/{platform}': (c.sha(expected_path(task, platform))
                                       if expected_path(task, platform).is_file() else None)
                for task in tasks for platform in PLATFORMS}
    return parts, expected


def block_lengths(prompts, sources, tasks=None):
    """Characters of each few-shot block: few-shot prompt length minus zero-shot prompt length (chatbot)."""
    lengths = {}
    for key, text in prompts.items():
        task, platform, prompt, agency, fmt = key.split('__')
        if prompt == 'few_shot' and agency == 'chatbot' and (tasks is None or task in tasks):
            lengths[f'{task}/{platform}'] = len(text) - len(sources['__'.join((task, platform, 'zero_shot', agency, fmt))])
    return lengths


# Jobs and provenance -----------------------------------------------------------------------

def block_name(model, offset, scope='few_shot'):
    return f"{SCOPES[scope]['prefix']}_{model}_{offset}"


def plan_jobs(scope, flagged=()):
    """Jobs of a scope. Each block holds one model and five repetitions, shuffled by repetition wave."""
    info = SCOPES[scope]
    extra = flagged_extra(scope, flagged)
    jobs = []
    for model in MODELS:
        for offset in range(0, REPS, BLOCK_REPS):
            batch = block_name(model, offset, scope)
            for rep in range(offset, offset + BLOCK_REPS):
                slots = [dict(task=task, platform=platform, prompt=prompt, condition=agency)
                         for task in info['tasks'] for platform in PLATFORMS for prompt in info['prompts']
                         for agency in AGENCIES]
                slots += [{k: s[k] for k in ('task', 'platform', 'prompt', 'condition')} for s in extra
                          if s['model'] == model and s['global_rep'] == rep]
                wave = [dict(block=batch, task=s['task'], model=model, platform=s['platform'], prompt=s['prompt'],
                             condition=s['condition'], output_format=output_format(s['platform']),
                             rep=rep % BLOCK_REPS, global_rep=rep) for s in slots]
                random.Random(SEED + rep).shuffle(wave)
                jobs.extend(wave)
    return jobs


def slot(row):
    return tuple(str(row[k]) for k in SLOT_FIELDS)


def expected_counts(scope, flagged=()):
    info = SCOPES[scope]
    jobs = plan_jobs(scope, flagged)
    task_positions = len(info['tasks']) * len(MODELS) * len(PLATFORMS) * len(info['prompts']) * len(AGENCIES) * REPS
    per_block = Counter(job['block'] for job in jobs)
    return dict(positions=len(jobs), task_positions=task_positions,
                flagged_listed=len(set(flagged)) if info['run_list'] else 0,
                flagged_added=len(jobs) - task_positions, cells=len({slot(job)[:5] for job in jobs}),
                blocks=len(per_block), per_block=dict(sorted(per_block.items())),
                prompt_keys=len({c.prompt_key(job) for job in jobs}))


def count_report(jobs):
    return {field: dict(sorted(Counter(str(job[field]) for job in jobs).items()))
            for field in ('task', 'model', 'platform', 'prompt', 'condition', 'global_rep', 'block')}


def baseline_links(jobs, baseline=None, analysis=None, verify_hashes=True):
    """Link every planned position to the selected record that it replaces."""
    baseline, analysis = baseline or BASELINE, analysis or ANALYSIS_RUNS
    tasks = {job['task'] for job in jobs}
    with Path(baseline).open() as stream:
        rows = {slot(row): row for row in csv.DictReader(stream) if row['task'] in tasks}
    outcome = {}
    if Path(analysis).is_file():
        with Path(analysis).open() as stream:
            outcome = {row['position_id']: row for row in csv.DictReader(stream) if row['task'] in tasks}
    links = []
    for job in jobs:
        old = rows.get(slot(job))
        if old is None:
            raise ValueError(f'No selected record for {slot(job)}')
        record = Path(old['record_path'])
        digest = c.sha(record) if verify_hashes else old['record_sha256']
        if digest != old['record_sha256']:
            raise ValueError(f'Selected record changed: {record}')
        current = outcome.get(old['position_id'], {})
        links.append(dict(supersedes_record=str(record), supersedes_record_sha256=old['record_sha256'],
                          previous_position_id=old['position_id'],
                          previous_y=int(float(current['y'])) if current.get('y') not in (None, '') else None,
                          previous_output_rule=current.get('output_rule')))
    return links


def production_catalog(path=None, tasks=ORIGINAL_TASKS):
    catalog = read(path or CATALOG)
    specs = {}
    for task in tasks:
        spec = dict(catalog[task])
        # Resolve compatibility links to physical inputs without duplicating data.
        spec['data_path'] = str(Path(spec['data_path']).resolve())
        if 'input_files' in spec:
            spec['input_files'] = {name: str(Path(p).resolve()) for name, p in spec['input_files'].items()}
        specs[task] = spec
    return specs


def plan_tasks(scope, flagged=()):
    """Tasks whose specifications the collection needs: the scope's tasks and the tasks of added positions."""
    return tuple(sorted(set(scope_tasks(scope)) | {s['task'] for s in flagged_extra(scope, flagged)}))


def plan_digest(plan):
    """Digest of the plan fields that fix the collection."""
    fixed = {k: plan[k] for k in ('scope', 'counts', 'prompt_sha256', 'source_prompt_sha256', 'example_sha256',
                                  'parts_sha256', 'expected_sha256', 'catalog_sha256', 'decision_record_sha256',
                                  'block_layout', 'seed', 'run_list')}
    return c.digest(fixed)


def environment_notes():
    return dict(sonnet_guard=dict(path=str(SONNET_GUARD), exists=SONNET_GUARD.is_file()),
                collector=dict(path=str(COLLECTOR), sha256=c.sha(COLLECTOR)),
                queue=dict(path=str(QUEUE), exists=QUEUE.is_file()),
                supervisor=dict(path=str(SUPERVISOR), exists=SUPERVISOR.is_file()),
                python=sys.executable, cwd=os.getcwd())
