"""Render the prompts of the final task bank from one template.

The final bank has 48 specified task packages in two folders (decision record 2026-09-29_hard-bank-final-changes):
    data/task_candidates/2026-09-28_final_v2/<task_id>/    (38 tasks)
    data/task_candidates/2026-09-28_final_hard/<task_id>/  (10 tasks: the 9 hard tasks and task 50)
and the decision versions (decision record 2026-09-29_decision-versions):
    data/task_candidates/2026-09-29_decision/<task_id>__decision/  (same data and reference as <task_id>)
Every task.json has "task_type": "specified" or "decision". A task.json may list "prompt_levels". The renderer
then writes only those levels (the decision versions whose few-shot example answers the decision have no few_shot).
A task.json may set "output_task_id", the task_id that the output format shows (the decision versions use a neutral
id, because <task>__decision names the method).

One renderer writes every prompt of every package:
    prompts/<language>__zero_shot__<condition>.txt
    prompts/<language>__few_shot__<condition>.txt
with language in python, r, stata and condition in chatbot, self_repair, agent. The bank has two prompt levels
(decision record 2026-09-30_drop-heldout-level).

Zero-shot prompt, in this order:
    role line; data files and "The data are simulated."; file descriptions; "Task: " + task_text;
    language line, version line (Python and R), and the two other standard sentences; output format of the language; condition text.
The task information (file descriptions and task text) is the same in every language and condition.

Few-shot prompt: the zero-shot prompt with one example block inserted after the role line.
Removing the block and one blank line from the few-shot prompt gives the zero-shot prompt.
The block comes from the package's examples/ folder (final_bank_examples.py writes it):
    block_parts.json      title, context, explanation, optional platform_notes {language: sentence}
    example.py, example.R, example.do
    expected_python.json, expected_r.json, expected_stata.txt  (outputs of real runs)

Held-out examples (type 2, decision record 2026-09-29_heldout-fewshot-examples): the level heldout_few_shot left the
bank on 2026-09-30 (decision record 2026-09-30_drop-heldout-level). The example set stays as an archive in
data/task_candidates/2026-09-29_heldout_examples/ (heldout_examples.py runs and checks it). heldout_block and
heldout_prompt still build the block from the archive, but no package renders or stores a held-out prompt.

Command line (repository root):
    uv run python code/benchmark/final_bank_prompts.py render [--task ID ...] [--zero-only]
    uv run python code/benchmark/final_bank_prompts.py check
    uv run python code/benchmark/final_bank_prompts.py render --paraphrase [--task ID ...]
The --paraphrase option writes the zero-shot prompts with the p1 and p2 paraphrases of the task text into a
separate folder (paraphrase_variants.py, decision record 2026-10-02_prompt-paraphrase-variants). It never writes
into a bank package.
No provider calls.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BANK_DIRS = (ROOT / 'data/task_candidates/2026-09-28_final_v2',
             ROOT / 'data/task_candidates/2026-09-28_final_hard')
# Decision versions (decision record 2026-09-29_decision-versions): <task>__decision packages with the data and the
# reference of the specified task <task>, a prompt that states facts and output conventions, and no few-shot example
# where the example answers the decision (task.json "prompt_levels").
DECISION_DIRS = (ROOT / 'data/task_candidates/2026-09-29_decision',)
LANGS = ('python', 'r', 'stata')
CONDITIONS = ('chatbot', 'self_repair', 'agent')
PROMPTS = ('zero_shot', 'few_shot')  # heldout_few_shot left the bank (decision record 2026-09-30_drop-heldout-level)
LANG_NAME = {'python': 'Python', 'r': 'R', 'stata': 'Stata'}

# The three standard sentences of every prompt.
STANDARD = ('The data are simulated.',
            'The standard packages of this language are installed. Do not install packages.',
            'Write the values at full precision.')
LANG_LINE = {
    'python': 'Use Python 3 (the python3 on PATH). Read the CSV files with pandas.',
    'r': 'Use R (Rscript). Read the CSV files with read.csv.',
    'stata': 'Use StataNow 19.5. Write a do-file. Read the CSV files with import delimited.',
}
# Installed versions (decision record 2026-09-29_version-line). The Stata line names StataNow 19.5. The Python and R
# lines name the interpreter and a short fixed list of core packages from
# code/benchmark/collection_environment/environment_manifest.json, at major.minor precision.
VERSION_LINE = {
    'python': 'The installed versions are Python 3.14, pandas 3.0, numpy 2.4, statsmodels 0.14, and scipy 1.18.',
    'r': 'The installed versions are R 4.6 and jsonlite 2.0.',
}
# The R line names no covariance or fixed-effects package (sandwich, fixest): such a name would hint at the scored
# decision of a decision version. jsonlite is named because the output format calls jsonlite::write_json.
VERSION_PACKAGES = {'python': ('pandas', 'numpy', 'statsmodels', 'scipy'), 'r': ('jsonlite',)}
RUN_LINE = {
    'python': 'Run the script with python3 run.py (the python3 on PATH).',
    'r': 'Run the script with Rscript --vanilla run.R.',
    'stata': 'Run the do-file with stata-se -b do run.do (Stata writes the log to run.log).',
}
STATA_DISPLAY = ('display "ANSWER: term_name estimate=" %24.16g estimate_value " se=" %24.16g se_value '
                 '" n_obs=" %12.0f n_value')
AGENT_VERIFICATION = (
    'Save the final solution as run.py for Python, run.R for R, or run.do for Stata.\n'
    'Before submission, verify the statistical calculation using the supplied data.\n'
    'Check the estimation sample, observation count, and variable construction against the task.\n'
    'Check the estimator and uncertainty calculation against the stated specification.\n'
    'Run at least one independent numerical check of a central calculation.\n'
    'Use a separate calculation or a mathematical identity that applies to this task.\n'
    'Inspect any discrepancy and correct errors supported by these checks.\n'
    'After edits, rerun the final script and inspect its requested output.\n'
    'Save a short plain-text account of the checks and their results in verification.txt.\n'
    'Use only the task statement, supplied data, installed software, and files you create.\n'
    'After these checks, finish with DONE.'
)
CLOSING = (
    'Use only the supplied CSV files and files you create in this directory.\n'
    'Do not read external agent skills, project instructions, prior results, or benchmark references.\n'
    'Do not create journals or use Git. Do not use MCP servers, web tools, or other agents.'
)

# Example block (option A of Decision 4): one example of the same calculation on separate data.
BLOCK_LABELS = ('Example: ', 'Separate example context: ', 'Explanation of this example: ')
RELATION = ('The example generates its own simulated data and reads no file. Its output uses the output format '
            'of the task below. The task below uses different data. Do not copy numbers from the example.')
EXAMPLE_FILES = {'python': 'example.py', 'r': 'example.R', 'stata': 'example.do'}
EXPECTED_FILES = {'python': 'expected_python.json', 'r': 'expected_r.json', 'stata': 'expected_stata.txt'}
OUTPUT_HEADER = {'python': 'Expected result.json of the example:\n```json\n',
                 'r': 'Expected result.json of the example:\n```json\n',
                 'stata': 'Expected ANSWER lines of the example in its log:\n```text\n'}
BLOCK_END = 'End of the example.'

# Held-out example block (type 2, archive only since 2026-09-30): a fixed set of examples on calculations that no bank
# task scores.
HELDOUT_DIR = ROOT / 'data/task_candidates/2026-09-29_heldout_examples'
HELDOUT_COUNT = {2: 'two', 3: 'three'}
HELDOUT_HEAD = 'Examples: {count} short scripts that show the output format on calculations unrelated to the task below.'
HELDOUT_RELATION = ('Each example generates its own simulated data and reads no file. Each example writes the output '
                    'format of this benchmark with its own task_id and row names. The task below uses different data '
                    'and a different calculation. Do not copy numbers from the examples.')
HELDOUT_LABELS = ('Example {k}: ', 'Example context: ', 'Explanation of this example: ')
HELDOUT_END = 'End of the examples.'

# Words that project rules ban from prompt text (CLAUDE.md and the v2 lint), matched as whole words.
BANNED = re.compile(r'\b(freeze|frozen|smoke test|task card|oracle|gates?|linear probability model|stat-llm|'
                    r'factorial|registry|worked example|arms?|contract|synthetic)\b', re.I)
ABSOLUTE_PATH = re.compile(r'(?<![\w.])(/Users/|/Applications/|/usr/|/opt/|/home/|/tmp/|~/|[A-Z]:\\)|\.venv')


def package_dirs() -> list[Path]:
    """Every package folder of the 48 specified tasks of the final bank, in folder order."""
    return [p for bank in BANK_DIRS for p in sorted(bank.iterdir()) if (p / 'task.json').is_file()]


def decision_dirs() -> list[Path]:
    """Every decision-version package folder, in folder order."""
    return [p for bank in DECISION_DIRS if bank.is_dir() for p in sorted(bank.iterdir()) if (p / 'task.json').is_file()]


def all_package_dirs() -> list[Path]:
    """The 48 specified packages, then the decision versions: every package that the runner plans."""
    return package_dirs() + decision_dirs()


def package_dir(task_id: str) -> Path:
    for path in all_package_dirs():
        if path.name == task_id:
            return path
    raise KeyError(f'no package {task_id}')


def prompt_levels(task: dict[str, Any]) -> tuple[str, ...]:
    """The prompt levels of a task: all three, or the subset in task.json "prompt_levels" (decision versions)."""
    levels = tuple(task.get('prompt_levels') or PROMPTS)
    if set(levels) - set(PROMPTS) or 'zero_shot' not in levels:
        raise ValueError(f'{task.get("task_id")}: prompt_levels {levels} must include zero_shot and use {PROMPTS}')
    return tuple(p for p in PROMPTS if p in levels)


def load_task(folder: Path) -> dict[str, Any]:
    """task.json with integer flags filled in from the targets (the hard packages mark them as exact)."""
    task = json.loads((Path(folder) / 'task.json').read_text())
    exact = {t['term'] for t in task.get('targets', []) if t.get('exact')}
    task['output_rows'] = [dict(row, se=bool(row.get('se')), integer=bool(row.get('integer')) or row['term'] in exact)
                           for row in task['output_rows']]
    return task


# Zero-shot prompt ---------------------------------------------------------------------------

def output_block(task: dict[str, Any], lang: str) -> str:
    """Output format text of one language."""
    rows = task['output_rows']
    if lang == 'stata':
        lines = ['Print one line per required result to the Stata log (run.log) with display, in exactly this format:']
        for row in rows:
            se = '<number>' if row['se'] else '.'
            est = '<integer>' if row['integer'] else '<number>'
            lines.append(f"ANSWER: {row['term']} estimate={est} se={se} n_obs=<integer>")
        lines += [
            'Set linesize 255 so that no ANSWER line wraps. This line shows the syntax:',
            STATA_DISPLAY,
            'Replace term_name with the row name and estimate_value, se_value, and n_value with computed scalars '
            'or expressions. Print se=. where a row has no standard error. Print each row once.',
            'Do not write a JSON file.',
        ]
        return '\n'.join(lines)
    shape = []
    for row in rows:
        se = '<float>' if row['se'] else 'null'
        est = '<integer>' if row['integer'] else '<float>'
        shape.append(f'        {{"term": "{row["term"]}", "estimate": {est}, "se": {se}, "n_obs": <int>}}')
    names = ', '.join(f'"{row["term"]}"' for row in rows)
    precision = ('json.dump of Python floats keeps full precision. Do not round.' if lang == 'python' else
                 'Call jsonlite::write_json(..., auto_unbox = TRUE, digits = NA). The default digits setting '
                 'rounds numbers.')
    return '\n'.join([
        'Required output:',
        '  - Write a file `result.json` in the working directory with this shape:',
        '    {',
        f'      "task_id": "{task.get("output_task_id") or task["task_id"]}",',
        f'      "platform": "{lang}",',
        '      "estimates": [',
        ',\n'.join(shape),
        '      ],',
        '      "method_metadata": {"estimator": "<string>", "vcov": "<string>"},',
        '      "diagnostics": {}',
        '    }',
        f'  - Row names must be exactly: {names}.',
        f'  - {precision}',
        '  - Do not include extra prose in `result.json`.',
    ])


def role_line(lang: str) -> str:
    return f'You are a careful econometrician writing a single self-contained {LANG_NAME[lang]} script.'


def task_information(task: dict[str, Any]) -> str:
    """The part of the prompt that is the same in every language and condition."""
    files = ', '.join(f'`./{name}`' for name in task['files'])
    lines = [f'Data files in the working directory: {files}. {STANDARD[0]}', 'File descriptions:']
    lines += [f'- {line}' for line in task['file_descriptions']]
    return '\n'.join(lines) + '\n\nTask: ' + task['task_text']


def zero_shot_prompt(task: dict[str, Any], lang: str, condition: str) -> str:
    language = [LANG_LINE[lang]]
    if lang in VERSION_LINE:
        language.append(VERSION_LINE[lang])
    note = (task.get('language_notes') or {}).get(lang)
    if note:
        language.append(note)
    language += [STANDARD[1], STANDARD[2]]
    parts = [role_line(lang), '', task_information(task), '', '\n'.join(language), '', output_block(task, lang), '']
    if condition == 'agent':
        finish = ('Write run.do and execute it in batch mode. Leave run.log containing all required ANSWER lines.'
                  if lang == 'stata' else 'Write and run a script. Leave result.json and the script in this directory.')
        tail = ['You may use the shell to write, execute, inspect errors, and revise your program.', RUN_LINE[lang],
                finish, AGENT_VERIFICATION, '', CLOSING]
    elif condition in ('chatbot', 'self_repair'):
        if lang == 'stata':
            run = ('The do-file will be executed in batch mode with the working directory set to a folder that '
                   'contains the data files. Read the data using relative paths. The do-file must run end-to-end '
                   'and print all required ANSWER lines to run.log.')
        else:
            run = ('The script will be executed with the working directory set to a folder that contains the data '
                   'files. Read the data using relative paths. The script must run end-to-end and produce '
                   '`result.json` in the same folder.')
        tail = [run, '', 'OUTPUT FORMAT - STRICT:',
                f'Respond with exactly ONE fenced code block in {lang} and nothing else.',
                f'No prose before or after. The fenced block must start with ```{lang} and end with ```.', '',
                'IMPORTANT: code generation only. The data files are NOT in your current directory right now, and '
                'you must NOT run any commands or read any file. Your script will be executed later in a directory '
                'that contains the data files. Return ONLY the fenced code block.']
    else:
        raise ValueError(f'unknown condition {condition}')
    return '\n'.join(parts + tail) + '\n'


# Few-shot prompt ----------------------------------------------------------------------------

def read_example(folder: Path, lang: str) -> tuple[dict[str, Any], str, str]:
    """(block parts, code, expected output) of one language from the package's examples/ folder."""
    ex = Path(folder) / 'examples'
    parts = json.loads((ex / 'block_parts.json').read_text())
    code = (ex / EXAMPLE_FILES[lang]).read_text()
    expected = (ex / EXPECTED_FILES[lang]).read_text()
    return parts, code, expected


def parts_problems(parts: Any) -> list[str]:
    if not isinstance(parts, dict):
        return ['block parts are not a JSON object']
    problems = []
    notes = parts.get('platform_notes') or {}
    if not isinstance(notes, dict) or set(notes) - set(LANGS):
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


def example_block(parts: dict[str, Any], code: str, expected: str, lang: str) -> str:
    """The example block: title, context, explanation, relation sentence, code, expected output, end line."""
    problems = parts_problems(parts)
    if problems:
        raise ValueError('; '.join(problems))
    if '```' in code or '```' in expected:
        raise ValueError('example code and expected output must not contain a Markdown fence')
    note = (parts.get('platform_notes') or {}).get(lang)
    explanation = parts['explanation'] + (' ' + note if note else '')
    lines = [BLOCK_LABELS[0] + parts['title'], BLOCK_LABELS[1] + parts['context'], BLOCK_LABELS[2] + explanation,
             RELATION]
    return ('\n'.join(lines) + f'\n```{lang}\n' + code.strip('\n') + '\n```\n' + OUTPUT_HEADER[lang]
            + expected.strip('\n') + '\n```\n' + BLOCK_END)


def insert_block(zero: str, block: str) -> str:
    """Insert the block after the role line: role line, blank line, block, blank line, rest of the prompt."""
    head, sep, rest = zero.partition('\n\n')
    if not sep:
        raise ValueError('zero-shot prompt has no blank line after the role line')
    return head + '\n\n' + block.strip('\n') + '\n\n' + rest


def remove_block(few: str, block: str) -> str:
    """Inverse of insert_block."""
    region = '\n\n' + block.strip('\n') + '\n\n'
    if few.count(region) != 1:
        raise ValueError('few-shot prompt does not contain the block exactly once')
    head, _, rest = few.partition(region)
    return head + '\n\n' + rest


def few_shot_prompt(task: dict[str, Any], lang: str, condition: str, folder: Path) -> str:
    parts, code, expected = read_example(folder, lang)
    return insert_block(zero_shot_prompt(task, lang, condition), example_block(parts, code, expected, lang))


# Held-out few-shot prompt -------------------------------------------------------------------

def heldout_ids(task_id: str, set_dir: Path = HELDOUT_DIR) -> list[str]:
    """Ordered held-out example ids of one task: the set, with the task's replacement from set.json."""
    spec = json.loads((Path(set_dir) / 'set.json').read_text())
    ids = list(spec['examples'])
    rep = (spec.get('replacements') or {}).get(task_id)
    if rep:
        ids[ids.index(rep['replace'])] = rep['by']
    return ids


def heldout_block(task_id: str, lang: str, set_dir: Path = HELDOUT_DIR) -> str:
    """The held-out block: head, relation sentence, then per example title, context, explanation, code, output."""
    ids = heldout_ids(task_id, set_dir)
    if len(ids) not in HELDOUT_COUNT or len(set(ids)) != len(ids):
        raise ValueError(f'held-out set must have 2 or 3 distinct examples, got {ids}')
    parts = [HELDOUT_HEAD.format(count=HELDOUT_COUNT[len(ids)]) + '\n' + HELDOUT_RELATION]
    for k, example_id in enumerate(ids, start=1):
        folder = Path(set_dir) / example_id
        spec = json.loads((folder / 'example.json').read_text())
        text = {key: spec.get(key) for key in ('title', 'context', 'explanation', 'platform_notes')
                if spec.get(key) is not None}
        problems = parts_problems(text)
        if problems:
            raise ValueError(f'{example_id}: ' + '; '.join(problems))
        code = (folder / EXAMPLE_FILES[lang]).read_text()
        expected = (folder / EXPECTED_FILES[lang]).read_text()
        if '```' in code or '```' in expected:
            raise ValueError(f'{example_id}: code and expected output must not contain a Markdown fence')
        note = (spec.get('platform_notes') or {}).get(lang)
        lines = [HELDOUT_LABELS[0].format(k=k) + spec['title'], HELDOUT_LABELS[1] + spec['context'],
                 HELDOUT_LABELS[2] + spec['explanation'] + (' ' + note if note else '')]
        parts.append('\n'.join(lines) + f'\n```{lang}\n' + code.strip('\n') + '\n```\n' + OUTPUT_HEADER[lang]
                     + expected.strip('\n') + '\n```')
    return '\n\n'.join(parts) + '\n\n' + HELDOUT_END


def heldout_prompt(task: dict[str, Any], lang: str, condition: str, set_dir: Path = HELDOUT_DIR) -> str:
    return insert_block(zero_shot_prompt(task, lang, condition), heldout_block(task['task_id'], lang, set_dir))


def prompt_name(lang: str, prompt: str, condition: str) -> str:
    return f'{lang}__{prompt}__{condition}.txt'


def render_package(folder: Path, zero_only: bool = False) -> dict[str, str]:
    """{file name: text} of the package's prompts."""
    task = load_task(folder)
    levels = prompt_levels(task)
    out = {}
    for lang in LANGS:
        for condition in CONDITIONS:
            out[prompt_name(lang, 'zero_shot', condition)] = zero_shot_prompt(task, lang, condition)
            if not zero_only:
                if 'few_shot' in levels:
                    out[prompt_name(lang, 'few_shot', condition)] = few_shot_prompt(task, lang, condition, folder)
    return out


def write_package(folder: Path, zero_only: bool = False) -> list[Path]:
    written = []
    task = load_task(folder)
    for lang in LANGS:
        path = Path(folder) / 'output_instructions' / f'{lang}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output_block(task, lang) + '\n')
    for name, text in render_package(folder, zero_only).items():
        path = Path(folder) / 'prompts' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written.append(path)
    return written


# Checks used by the tests --------------------------------------------------------------------

def described_columns(description: str) -> list[str]:
    """Column names that one file description names.

    Two layouts: "Columns: a (meaning), b and c (meaning), z1 to z10 (meaning)" and the codebook layout
    "Codebook. name: meaning. name: meaning.".
    """
    names: list[str] = []
    if 'Codebook.' in description:
        body = description.split('Codebook.', 1)[1]
        names += re.findall(r'(?:^|[.;]\s)\s*([a-z][a-z0-9_]*):\s', body)
        return names
    body = description.split('Columns:', 1)[1]
    depth, piece, pieces = 0, '', []
    for ch in body:
        depth += ch == '('
        depth -= ch == ')'
        if ch == ',' and depth == 0:
            pieces.append(piece)
            piece = ''
        else:
            piece += ch
    pieces.append(piece)
    for piece in pieces:
        head = re.sub(r'\(.*', '', piece.strip().rstrip('.')).strip()
        for part in re.split(r'\s+and\s+', head):
            part = part.strip()
            ranged = re.fullmatch(r'([a-z_]+)(\d+) to \1(\d+)', part)
            if ranged:
                names += [f'{ranged.group(1)}{i}' for i in range(int(ranged.group(2)), int(ranged.group(3)) + 1)]
            elif re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part):
                names.append(part)
            elif part:
                raise ValueError(f'cannot read column names from "{part}"')
    return names


def csv_header(path: Path) -> list[str]:
    with open(path) as handle:
        return [c.strip().strip('"') for c in handle.readline().rstrip('\n').split(',')]


def prompt_problems(text: str) -> list[str]:
    """Problems that the text of one prompt shows."""
    problems = []
    for sentence in STANDARD:
        if text.count(sentence) != 1:
            problems.append(f'standard sentence "{sentence}" appears {text.count(sentence)} times')
    match = BANNED.search(text)
    if match:
        problems.append(f'banned word "{match.group(0)}"')
    match = ABSOLUTE_PATH.search(text)
    if match:
        problems.append(f'absolute path "{match.group(0)}"')
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    render = sub.add_parser('render', help='write the prompts of the packages')
    render.add_argument('--task', action='append', help='task id (default: all packages, decision versions included)')
    render.add_argument('--zero-only', action='store_true', help='write only the zero-shot prompts')
    render.add_argument('--paraphrase', action='store_true',
                        help='write the zero-shot p1 and p2 paraphrase prompts into '
                             'data/task_candidates/2026-10-02_paraphrases/<task>/prompts/ (bank prompts unchanged)')
    sub.add_parser('check', help='compare the written prompts with a fresh rendering')
    args = parser.parse_args(argv)
    folders = all_package_dirs()
    if args.command == 'render' and args.paraphrase:
        import paraphrase_variants  # decision record 2026-10-02_prompt-paraphrase-variants
        return paraphrase_variants.write_variants(args.task)
    if args.command == 'render':
        if args.task:
            folders = [package_dir(t) for t in args.task]
        count = sum(len(write_package(f, args.zero_only)) for f in folders)
        print(f'wrote {count} prompts in {len(folders)} packages')
        return 0
    stale = []
    for folder in folders:
        rendered = render_package(folder)
        for name, text in rendered.items():
            path = folder / 'prompts' / name
            if not path.is_file() or path.read_text() != text:
                stale.append(str(path.relative_to(ROOT)))
        extra = sorted(p.name for p in (folder / 'prompts').glob('*.txt') if p.name not in rendered)
        stale += [str((folder / 'prompts' / name).relative_to(ROOT)) + ' (not rendered)' for name in extra]
    print(f'{len(stale)} stale prompts' + ''.join(f'\n  {p}' for p in stale))
    return 1 if stale else 0


if __name__ == '__main__':
    sys.exit(main())
