"""Rebuild the local main paper from saved forty-task records.

The default requires the complete corrected 720-response collection.
--baseline audits the earlier sample while that collection is running.
Neither mode calls a model provider or writes to a remote repository.
"""
import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import dataset

ROOT = dataset.ROOT
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import paper_appendix_numbers
import paper_robustness
import production_paper_assets as assets


MANUSCRIPT_TARGETS = ('main', 'main_sin', 'supplemental_appendix', 'main_body', 'main_standard')
"""The five PDF targets, in build order.

supplemental_appendix reads the paper labels from main.aux (xr), so it follows main.
main_body reads the appendix labels from build/supplemental_appendix.aux, so it follows the appendix.
main_standard is the same paper in the plain article class with no counter; the website serves it.
"""


def log_problems(text):
    """Return the log lines that fail the build: errors, undefined references, overfull boxes, oversized floats."""
    return [line for line in text.splitlines()
            if (line.startswith('! ') or 'Overfull \\hbox' in line or 'Overfull \\vbox' in line
                or 'undefined references' in line or ('LaTeX Warning:' in line and 'undefined' in line)
                or 'Float too large for page' in line)]


def compile_manuscript(out):
    """Compile every configured PDF and reject unresolved source problems."""
    folder = ROOT / 'paper/manuscript'
    log_path = out / 'work/manuscript_build.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    targets = MANUSCRIPT_TARGETS
    missing = [name for name in targets if not (folder / (name + '.tex')).is_file()]
    if missing:
        raise ValueError(f'Missing manuscript target sources: {missing}')
    with log_path.open('w') as log:
        subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                        *[name + '.tex' for name in targets]], cwd=folder,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    receipts = {}
    for name in targets:
        pdf, log = folder / 'build' / (name + '.pdf'), folder / 'build' / (name + '.log')
        if not pdf.is_file() or not log.is_file():
            raise ValueError(f'Missing manuscript build output: {name}')
        errors = log_problems(log.read_text(errors='replace'))
        if errors:
            raise ValueError(f'Manuscript verification failed for {name}: {errors}')
        receipts[name] = dict(path=str(pdf.relative_to(ROOT)), sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
        shortcut = folder / (name + '.pdf')
        tracked = subprocess.check_output(['git', 'ls-files', '--', shortcut.name], cwd=folder, text=True).strip()
        if tracked:
            raise ValueError(f'Refusing to replace a tracked PDF: {shortcut}')
        if shortcut.is_symlink() or shortcut.exists():
            shortcut.unlink()
        shortcut.symlink_to(Path('build') / pdf.name)
    dataset.write_json(out / 'data/manuscript_build.json', dict(
        at=datetime.now(timezone.utc).isoformat(), targets=receipts, undefined_references=0, overfull_boxes=0,
        visual_review='Rendered baseline pages were inspected before the replacement build. New PDF hashes require a new visual review.'))


def analysis_open(path, *args, **kwargs):
    if str(path) == '/private/tmp/agents_metrics_native_stata.lock':
        path = '/private/tmp/agents_metrics_paper_stata_mp.lock'
    return builtins.open(path, *args, **kwargs)


def build(out, baseline=False, render=True, cost_valuation='list', output_rule=dataset.OUTPUT_RULE):
    receipt = dataset.build(out, use_replacements=not baseline, output_rule=output_rule)
    (out / 'data/task_eligibility.json').unlink(missing_ok=True)
    catalog = json.loads((out / 'data/catalog.json').read_text())
    # Figure 3 and the appendix task tables share the method-only labels in task_labels.py.
    unlabeled = sorted(set(catalog) - set(assets.TASK_LABELS))
    if unlabeled:
        raise ValueError(f'Tasks without a paper label in code/benchmark/task_labels.py: {unlabeled}')
    # The saved inference used Stata/MP 14 with boottest. STATA_BIN selects another Stata binary.
    assets.fa.STATA_BIN = os.environ.get('STATA_BIN', '/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp')
    assets.open = analysis_open
    old_args = sys.argv
    try:
        sys.argv = [__file__, '--out', str(out), '--bootstrap', '--cost-valuation', cost_valuation]
        assets.main()
    finally:
        sys.argv = old_args
    # Appendix D: tables/appendix_robustness.tex and data/robustness.json (numRob* macros).
    paper_robustness.build(out)
    # Appendices B and C: joint figure points, self-repair first attempts, hand-audit causes.
    paper_appendix_numbers.build(out)
    if render:
        subprocess.run([sys.executable, str(Path(__file__).with_name('manuscript.py')),
                        '--analysis', str(out)], cwd=ROOT, check=True)
        compile_manuscript(out)
    print(json.dumps(dict(complete=True, provisional=baseline, rendered=render, **receipt)))


def parse_args(argv=None):
    """Parse the command line. Only the canonical rule may write data/analysis/main."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=dataset.OUT)
    parser.add_argument('--baseline', action='store_true', help='Build the saved sample before the two complete task replacements')
    parser.add_argument('--no-render', action='store_true', help='Build data, figures, and tables only')
    parser.add_argument('--cost-valuation', choices=['list', 'sonnet_all_input'], default='list',
                        help='Token valuation for costs; sonnet_all_input reproduces the earlier Sonnet rule')
    parser.add_argument('--output-rule', choices=dataset.OUTPUT_RULES, default=dataset.OUTPUT_RULE,
                        help='channel_se (default): channel plus the V3 standard-error check in nine tasks; '
                             'channel: estimates only, also read Stata final-execution files and stdout; '
                             'log: earlier estimate-only rule. A non-default rule needs --out')
    args = parser.parse_args(argv)
    if args.output_rule != dataset.OUTPUT_RULE and args.out.resolve() == dataset.OUT.resolve():
        parser.error('a non-default --output-rule needs a separate --out folder')
    args.out = args.out.resolve()
    return args


if __name__ == '__main__':
    args = parse_args()
    build(args.out, args.baseline, not args.no_render, args.cost_valuation, args.output_rule)
