"""Preserve data companions and archive three authorized historical trees.

The archive preserves symbolic links as links. It never follows report links
into data. Source removal is a separate command after complete verification.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

ROOT = Path(__file__).resolve().parents[4]
TREES = ('archive', 'quality_reports/reports', 'quality_reports/audits')
PUBLICATION_TREES = tuple('data/runs/' + name for name in (
    'codex_few_shot', 'codex_zero_shot', 'deepseek_few_shot',
    'deepseek_zero_shot', 'sonnet_few_shot', 'sonnet_zero_shot'))
PAPER_TREES = ('2026-09-21_paper_40tasks',)
DEFAULT_OUT = Path.home() / 'ARCHIVE/AGENTS-METRICS/2026-09-21-production40'
DEFAULT_INDEX = ROOT / 'data/maintenance/production40_cleanup'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def entries(root):
    """Hash regular files and record links without following them."""
    rows = []
    for folder, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(folder) / name
            info = path.lstat()
            row = dict(path=str(path.relative_to(root)), mode=stat.S_IMODE(info.st_mode))
            if path.is_symlink():
                row.update(kind='symlink', target=os.readlink(path))
            elif path.is_file():
                row.update(kind='file', size=info.st_size, sha256=sha(path))
            elif path.is_dir():
                row.update(kind='directory')
            else:
                raise ValueError(f'Unsupported file type: {path}')
            rows.append(row)
    return sorted(rows, key=lambda row: row['path'])


def data_links(root):
    """Return data links whose resolved targets belong to archive sources."""
    sources = [root / tree for tree in TREES]
    found = []
    for folder, dirs, files in os.walk(root / 'data', followlinks=False):
        for name in sorted(dirs + files):
            path = Path(folder) / name
            if not path.is_symlink():
                continue
            target = path.resolve()
            if any(target.is_relative_to(source) for source in sources):
                if not target.exists():
                    raise ValueError(f'Missing data companion: {path}')
                found.append((path, target))
    return found


def publication_exclusions(root, index, excluded_trees=PUBLICATION_TREES):
    """Reject publication archival if current selected records or inputs use it."""
    selected = root / 'data/production/main/baseline_runs.csv'
    catalog = root / 'data/production/main/catalog.json'
    destinations = [root / tree for tree in excluded_trees]
    checked = []

    def check(value):
        path = Path(value)
        path = (path if path.is_absolute() else root / path).resolve()
        if any(path.is_relative_to(destination) for destination in destinations):
            raise ValueError(f'Current selected source uses publication corpus: {path}')
        checked.append(str(path))

    with selected.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 14400:
        raise ValueError('Expected the complete 14,400-position baseline')
    for row in rows:
        for field in ('record_path', 'output_path', 'manifest_path'):
            if row.get(field):
                check(row[field])
    inputs = []
    for task in json.loads(catalog.read_text()).values():
        if task.get('data_path'):
            inputs.append(task['data_path'])
        inputs.extend(task.get('input_files', {}).values())
    # Candidate inputs also remain available after cleanup.
    for base in (root / 'data/tasks', root / 'data/task_candidates'):
        for file in base.rglob('task.json'):
            task = json.loads(file.read_text())
            if task.get('data_path'):
                inputs.append(task['data_path'])
            inputs.extend(task.get('input_files', {}).values())
    for value in inputs:
        check(value)
    paper = excluded_trees == PAPER_TREES
    save(index / ('paper_exclusions.json' if paper else 'publication_exclusions.json'), dict(selected_positions=len(rows),
         selected_csv=str(selected), selected_sha256=sha(selected), catalog_sha256=sha(catalog),
         checked_paths=len(checked), unique_paths=len(set(checked)),
         candidate_and_task_inputs=len(inputs), archived_trees=list(excluded_trees),
         selected_dependencies=0, candidate_inputs_preserved=True,
         historical_dependencies=['dated standalone paper outputs'] if paper else ['build_full_analysis.py', 'original10_2026-09-19', 'paper22_2026-09-18'],
         historical_rebuild='Restore the corresponding archive before rebuilding these historical analyses.'))


def materialize(root, index):
    """Replace data links with byte-identical local companion files."""
    receipts_path = index / 'materialized_companions.json'
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else []
    for path, target in data_links(root):
        if not target.is_file():
            raise ValueError(f'Directory companion requires explicit review: {path}')
        expected = sha(target)
        handle, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
        os.close(handle)
        temporary = Path(name)
        try:
            shutil.copy2(target, temporary)
            if sha(temporary) != expected or sha(target) != expected:
                raise ValueError(f'Companion changed during copy: {path}')
            link_text = os.readlink(path)
            temporary.replace(path)
            receipts.append(dict(path=str(path.relative_to(root)), source=str(target.relative_to(root)),
                                 prior_link=link_text, sha256=expected, size=path.stat().st_size))
            # Persist each completed unit so an interruption can resume safely.
            save(receipts_path, receipts)
        finally:
            temporary.unlink(missing_ok=True)
    if data_links(root):
        raise ValueError('Data companions still point into archive sources')
    for receipt in receipts:
        path = root / receipt['path']
        if path.is_symlink() or sha(path) != receipt['sha256']:
            raise ValueError(f'Materialized companion changed: {path}')
    save(receipts_path, receipts)
    return receipts


def snapshot(root, out, index):
    manifest_path = index / 'archive_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_root'] != str(root) or manifest['archive_root'] != str(out):
            raise ValueError('Existing archive manifest uses different roots')
        return manifest
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), source_root=str(root),
                    archive_root=str(out), trees={}, restoration='Copy each tree back to its original relative path with symbolic links preserved.')
    for tree in TREES:
        source = root / tree
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f'Archive source must be a physical directory: {source}')
        manifest['trees'][tree] = entries(source)
    save(manifest_path, manifest)
    return manifest


def verify_tree(path, expected):
    if not path.is_dir() or path.is_symlink() or entries(path) != expected:
        raise ValueError(f'Archive content differs from inventory: {path}')


def copy_archive(root, out, index):
    if TREES in (PUBLICATION_TREES, PAPER_TREES):
        publication_exclusions(root, index, TREES)
    materialize(root, index)
    manifest = snapshot(root, out, index)
    for tree, expected in manifest['trees'].items():
        source, destination = root / tree, out / tree
        verify_tree(source, expected)
        if destination.exists():
            verify_tree(destination, expected)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + '.copying')
            if temporary.exists():
                # Reuse a complete interrupted copy. Keep partial copies for review.
                verify_tree(temporary, expected)
            else:
                shutil.copytree(source, temporary, symlinks=True)
            verify_tree(temporary, expected)
            temporary.replace(destination)
        verify_tree(source, expected)
        print(json.dumps(dict(stage='copied_and_verified', tree=tree, entries=len(expected))), flush=True)
    shutil.copy2(index / 'archive_manifest.json', out / 'archive_manifest.json')
    shutil.copy2(index / 'materialized_companions.json', out / 'materialized_companions.json')
    for name in ('publication_exclusions.json', 'paper_exclusions.json'):
        if (index / name).exists():
            shutil.copy2(index / name, out / name)
    documents = index / 'pre_migration_docs'
    if documents.exists():
        expected = entries(documents)
        destination = out / 'pre_migration_docs'
        if not destination.exists():
            shutil.copytree(documents, destination, symlinks=True)
        verify_tree(destination, expected)
        for name, digest in json.loads((documents / 'hashes.json').read_text()).items():
            if sha(destination / name) != digest:
                raise ValueError(f'Preserved document changed: {name}')
    save(index / 'copy_receipt.json', dict(archive_root=str(out),
         manifest_sha256=sha(index / 'archive_manifest.json'), verified=True, originals_removed=False))
    return manifest


def remove_originals(root, out, index):
    if TREES in (PUBLICATION_TREES, PAPER_TREES):
        publication_exclusions(root, index, TREES)
    manifest = snapshot(root, out, index)
    receipt = json.loads((index / 'copy_receipt.json').read_text())
    if not receipt['verified'] or receipt['manifest_sha256'] != sha(index / 'archive_manifest.json'):
        raise ValueError('Verified copy receipt is unavailable')
    if sha(out / 'archive_manifest.json') != receipt['manifest_sha256']:
        raise ValueError('External restoration inventory changed')
    if data_links(root):
        raise ValueError('Data still depends on archive sources')
    for companion in json.loads((index / 'materialized_companions.json').read_text()):
        path = root / companion['path']
        if path.is_symlink() or sha(path) != companion['sha256']:
            raise ValueError(f'Materialized companion changed: {path}')
    # Verify all trees before deleting the first one.
    for tree, expected in manifest['trees'].items():
        verify_tree(out / tree, expected)
        if (root / tree).exists():
            verify_tree(root / tree, expected)
    for tree in manifest['trees']:
        source = root / tree
        if source.exists():
            shutil.rmtree(source)
        print(json.dumps(dict(stage='removed_verified_original', tree=tree)), flush=True)
    receipt['originals_removed'] = True
    save(index / 'copy_receipt.json', receipt)


def main():
    global TREES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['materialize', 'inventory', 'copy', 'remove-originals'])
    parser.add_argument('--profile', choices=['historical', 'publication', 'paper'], default='historical')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--index', type=Path)
    args = parser.parse_args()
    suffix = '-' + args.profile if args.profile != 'historical' else ''
    out = (args.out or DEFAULT_OUT.with_name(DEFAULT_OUT.name + suffix)).resolve()
    index = (args.index or DEFAULT_INDEX.with_name(DEFAULT_INDEX.name + suffix)).resolve()
    if args.profile == 'publication':
        TREES = PUBLICATION_TREES
    elif args.profile == 'paper':
        TREES = PAPER_TREES
    if not out.is_relative_to(Path.home() / 'ARCHIVE/AGENTS-METRICS') or out == ROOT or out.is_relative_to(ROOT):
        raise ValueError('Archive destination must be under ~/ARCHIVE/AGENTS-METRICS outside the repository')
    if not index.is_relative_to(ROOT / 'data/maintenance'):
        raise ValueError('Local index must be under data/maintenance')
    if args.command == 'materialize':
        result = materialize(ROOT, index)
        print(json.dumps(dict(materialized_companions=len(result))))
    elif args.command == 'inventory':
        manifest = snapshot(ROOT, out, index)
        print(json.dumps({tree: len(rows) for tree, rows in manifest['trees'].items()}))
    elif args.command == 'copy':
        copy_archive(ROOT, out, index)
    else:
        remove_originals(ROOT, out, index)


if __name__ == '__main__':
    main()
