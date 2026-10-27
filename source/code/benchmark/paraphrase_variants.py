"""Paraphrase variants of the task text of the 48 specified bank tasks (prompt sensitivity).

Decision record 2026-10-02_prompt-paraphrase-variants. Each task has two paraphrases of its task_text, stored as data:
    data/task_candidates/2026-10-02_paraphrases/<task_id>/p1.json   close rewording, same order of facts
    data/task_candidates/2026-10-02_paraphrases/<task_id>/p2.json   rewording and a new order of facts
A paraphrase keeps every fact, number, identifier, convention, and the output format. Only the task_text changes:
the role line, file descriptions, language lines, output format, and condition text come from
final_bank_prompts.zero_shot_prompt unchanged.

The renderer writes zero-shot prompts into a separate folder, never into the bank package:
    data/task_candidates/2026-10-02_paraphrases/<task_id>/prompts/<lang>__zero_shot_p1__<condition>.txt
    data/task_candidates/2026-10-02_paraphrases/<task_id>/prompts/<lang>__zero_shot_p2__<condition>.txt
The names follow final_bank_prompts.prompt_name with the prompt level zero_shot_p1 or zero_shot_p2.

Command line (repository root):
    uv run python code/benchmark/paraphrase_variants.py store DRAFTS.json [DRAFTS.json ...]
    uv run python code/benchmark/paraphrase_variants.py check [--json OUT.json]
    uv run python code/benchmark/paraphrase_variants.py render [--task ID ...]
    uv run python code/benchmark/final_bank_prompts.py render --paraphrase   (same as render above)
No provider calls.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_bank_prompts as fp  # noqa: E402

ROOT = fp.ROOT
STORE = ROOT / 'data/task_candidates/2026-10-02_paraphrases'
VARIANTS = {'p1': 'close rewording, same order of facts',
            'p2': 'rewording and a new order of facts'}

NUMBER = re.compile(r'(?<![A-Za-z_\d.])\d+(?:\.\d+)?(?![A-Za-z_])')
NUMBER_WORDS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                'twelve', 'fifteen', 'twenty', 'hundred', 'thousand', 'half', 'single', 'once', 'twice', 'both')
NUMBER_WORD = re.compile(r'\b(' + '|'.join(NUMBER_WORDS) + r')\b', re.I)
IDENT = re.compile(r'(?<![\w.])(?:[A-Za-z]\w*_\w*|[A-Za-z]+\d\w*|\w+\.(?:csv|code))(?![\w])')
EXPR_CHARS = set('=<>/()[]^*|+')
# Comparisons such as "score >= 0" and "recipient = 0", bracketed formulas, and offsets such as t-1 and c+2.
COMPARISON = re.compile(r'[\w.^*+\-]+(?:\([\w.,\s]*\))?\s(?:<=|>=|==|<|>|=)\s[\w.^*+\-]+(?:\([\w.,\s]*\))?')
BRACKET = re.compile(r'\[[^\]]+\]')
OFFSET = re.compile(r'(?<![\w.])[A-Za-z]\w*[-+]\d+(?![\w])')
EM_DASH = '—'


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def variant_path(task_id: str, variant: str) -> Path:
    return STORE / task_id / f'{variant}.json'


def load_variant(task_id: str, variant: str) -> dict[str, Any]:
    return json.loads(variant_path(task_id, variant).read_text())


def paraphrased_task(task: dict[str, Any], variant: str) -> dict[str, Any]:
    """The task with its task_text replaced by the stored paraphrase."""
    record = load_variant(task['task_id'], variant)
    if record['source_task_text_sha256'] != sha256(task['task_text']):
        raise ValueError(f'{task["task_id"]} {variant}: the bank task_text changed after the paraphrase was written')
    return dict(task, task_text=record['task_text'])


# Store ---------------------------------------------------------------------------------------

def store(drafts: list[Path]) -> int:
    """Write p1.json and p2.json for every task in the draft files ({task_id: {p1: text, p2: text}})."""
    tasks = {p.name: fp.load_task(p) for p in fp.package_dirs()}
    count = 0
    for draft in drafts:
        for task_id, texts in json.loads(Path(draft).read_text()).items():
            if task_id not in tasks:
                raise KeyError(f'{draft}: {task_id} is not one of the 48 specified tasks')
            for variant in VARIANTS:
                path = variant_path(task_id, variant)
                path.parent.mkdir(parents=True, exist_ok=True)
                record = {'task_id': task_id, 'variant': variant, 'variant_rule': VARIANTS[variant],
                          'source_task_text_sha256': sha256(tasks[task_id]['task_text']),
                          'task_text': ' '.join(texts[variant].split())}
                path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
                count += 1
    print(f'stored {count} paraphrases')
    return 0


# Mechanical check ----------------------------------------------------------------------------

def numbers(text: str) -> Counter:
    return Counter(NUMBER.findall(text))


def number_words(text: str) -> Counter:
    alias = {'single': 'one'}
    return Counter(alias.get(w.lower(), w.lower()) for w in NUMBER_WORD.findall(text))


def has_word(token: str, text: str) -> bool:
    return re.search(r'(?<![\w.])' + re.escape(token) + r'(?![\w])', text) is not None


def expressions(text: str) -> set[str]:
    """Whitespace tokens with formula characters, trimmed of sentence punctuation and unbalanced parentheses."""
    out = set()
    for raw in text.split():
        tok = raw.rstrip('.,;:')
        while tok.startswith('(') and tok.count('(') > tok.count(')'):
            tok = tok[1:]
        while tok.endswith(')') and tok.count(')') > tok.count('('):
            tok = tok[:-1]
        tok = tok.rstrip('.,;:')
        if EXPR_CHARS & set(tok) and re.search(r'[A-Za-z0-9]', tok) and not re.fullmatch(r'\([\w\s-]+\)', tok):
            out.add(tok)
    out |= {m.group(0).rstrip('.,') for m in COMPARISON.finditer(text)}
    out |= set(BRACKET.findall(text)) | set(OFFSET.findall(text))
    return out


def proper_terms(text: str) -> set[str]:
    """Capitalized words that are not only sentence-initial capitals (names, acronyms, CR2, McFadden)."""
    out = set()
    for sentence in re.split(r'(?<=[.;:])\s+', text):
        words = sentence.split()
        for k, raw in enumerate(words):
            tok = raw.strip('.,;:()[]\'"')
            tok = re.sub(r"'s$", '', tok)
            if not tok or not re.search(r'[A-Z]', tok):
                continue
            if k == 0 and not re.search(r'[A-Z]', tok[1:]):
                continue
            out.add(tok)
    return out


def identifiers(task: dict[str, Any]) -> set[str]:
    """Identifiers of the original task text: row names, file and column names it uses, and code-like tokens."""
    text = task['task_text']
    names = {row['term'] for row in task['output_rows']} | set(task['files'])
    for description in task['file_descriptions']:
        try:
            names |= set(fp.described_columns(description))
        except (ValueError, IndexError):
            pass
    found = {n for n in names if has_word(n, text)}
    return found | set(IDENT.findall(text))


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.])\s+(?=[A-Za-z])', text) if s.strip()]


def check_variant(task: dict[str, Any], variant: str) -> dict[str, Any]:
    original = task['task_text']
    record = load_variant(task['task_id'], variant)
    text = record['task_text']
    errors, warnings = [], []
    if record['source_task_text_sha256'] != sha256(original):
        errors.append('stale: bank task_text changed')
    if text == original:
        errors.append('identical to the original')
    on, pn = numbers(original), numbers(text)
    missing = sorted(set(on) - set(pn))
    new = sorted(set(pn) - set(on))
    if missing:
        errors.append(f'missing numbers {missing}')
    if new:
        errors.append(f'new numbers {new}')
    for n in sorted(set(on) & set(pn)):
        if on[n] != pn[n]:
            warnings.append(f'number {n} appears {on[n]} times in the original and {pn[n]} times here')
    ow, pw = number_words(original), number_words(text)
    if set(ow) - set(pw):
        errors.append(f'missing number words {sorted(set(ow) - set(pw))}')
    if set(pw) - set(ow):
        errors.append(f'new number words {sorted(set(pw) - set(ow))}')
    for w in sorted(set(ow) & set(pw)):
        if ow[w] != pw[w]:
            warnings.append(f'number word {w} appears {ow[w]} times in the original and {pw[w]} times here')
    idents = identifiers(task)
    absent = sorted(i for i in idents if not has_word(i, text))
    if absent:
        errors.append(f'missing identifiers {absent}')
    new_idents = sorted(set(IDENT.findall(text)) - idents)
    if new_idents:
        errors.append(f'new identifiers {new_idents}')
    lost_expr = sorted(e for e in expressions(original) if e not in text)
    if lost_expr:
        errors.append(f'missing expressions {lost_expr}')
    new_expr = sorted(e for e in expressions(text) if e not in original)
    if new_expr:
        warnings.append(f'new expressions {new_expr}')
    lost_terms = sorted(t for t in proper_terms(original) if t not in text)
    if lost_terms:
        errors.append(f'missing names or acronyms {lost_terms}')
    new_terms = sorted(t for t in proper_terms(text) if t not in original)
    if new_terms:
        warnings.append(f'new capitalized terms {new_terms}')
    banned = fp.BANNED.search(text)
    if banned:
        errors.append(f'banned word "{banned.group(0)}"')
    if EM_DASH in text:
        errors.append('em dash')
    kept = sum(s in sentences(original) for s in sentences(text))
    return {'task_id': task['task_id'], 'variant': variant, 'errors': errors, 'warnings': warnings,
            'words_original': len(original.split()), 'words_variant': len(text.split()),
            'similarity': round(difflib.SequenceMatcher(None, original.split(), text.split()).ratio(), 3),
            'sentences': len(sentences(text)), 'sentences_unchanged': kept,
            'identifiers_checked': len(idents), 'numbers_checked': sum(on.values())}


def check_all(json_out: Path | None = None) -> int:
    results = []
    for folder in fp.package_dirs():
        task = fp.load_task(folder)
        for variant in VARIANTS:
            if not variant_path(task['task_id'], variant).is_file():
                results.append({'task_id': task['task_id'], 'variant': variant, 'errors': ['missing file'],
                                'warnings': []})
                continue
            results.append(check_variant(task, variant))
    bad = [r for r in results if r['errors']]
    for r in results:
        for e in r['errors']:
            print(f'ERROR {r["task_id"]} {r["variant"]}: {e}')
        for w in r['warnings']:
            print(f'warn  {r["task_id"]} {r["variant"]}: {w}')
    print(f'{len(results)} paraphrases checked, {len(bad)} with errors, '
          f'{sum(bool(r["warnings"]) for r in results)} with warnings')
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(results, indent=2) + '\n')
    return 1 if bad else 0


# Renderer ------------------------------------------------------------------------------------

def render_variants(folder: Path) -> dict[str, str]:
    """{file name: text} of the zero-shot paraphrase prompts of one package (18 per variant)."""
    task = fp.load_task(folder)
    out = {}
    for variant in VARIANTS:
        para = paraphrased_task(task, variant)
        for lang in fp.LANGS:
            for condition in fp.CONDITIONS:
                out[fp.prompt_name(lang, f'zero_shot_{variant}', condition)] = fp.zero_shot_prompt(para, lang,
                                                                                                    condition)
    return out


def write_variants(task_ids: list[str] | None = None) -> int:
    folders = [p for p in fp.package_dirs() if not task_ids or p.name in task_ids]
    if task_ids and len(folders) != len(set(task_ids)):
        raise KeyError(f'unknown task among {task_ids}')
    count = 0
    for folder in folders:
        target = STORE / folder.name / 'prompts'
        target.mkdir(parents=True, exist_ok=True)
        for name, text in render_variants(folder).items():
            (target / name).write_text(text)
            count += 1
    print(f'wrote {count} paraphrase prompts in {len(folders)} packages under {STORE.relative_to(ROOT)}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('store', help='write p1.json and p2.json from draft files')
    s.add_argument('drafts', nargs='+', type=Path)
    c = sub.add_parser('check', help='mechanical check of numbers, identifiers, and formulas')
    c.add_argument('--json', type=Path)
    r = sub.add_parser('render', help='write the paraphrase prompts into the separate folder')
    r.add_argument('--task', action='append')
    args = parser.parse_args(argv)
    if args.command == 'store':
        return store(args.drafts)
    if args.command == 'check':
        return check_all(args.json)
    return write_variants(args.task)


if __name__ == '__main__':
    sys.exit(main())
