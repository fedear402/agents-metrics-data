"""Move report-hosted experiment data into data/collections without changing bytes.

Old paths remain compatibility symlinks. Source code remains at its original
path because historical modules derive repository paths from __file__.
Use plan, apply, and verify in that order. No provider calls are made.
Only use apply after checking that no collector writes into report folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DATA_SUFFIXES = {'.json', '.jsonl', '.csv', '.tsv', '.txt', '.stdout', '.stderr',
                 '.log', '.dta', '.rds', '.rdata', '.parquet', '.npy', '.npz',
                 '.xlsx', '.pkl', '.pickle', '.out', '.irf', '.gz', '.diff', '.sha256', '.lock'}
SKIP = {'.git', '.venv', '__pycache__', 'node_modules', '.pytest_cache'}


def files_under(base):
    for folder, dirs, files in os.walk(base, followlinks=False):
        for name in sorted(dirs):
            link = Path(folder) / name
            if link.is_symlink():
                yield link
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(folder) / d).is_symlink())
        for name in sorted(files):
            yield Path(folder) / name


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def make_plan(root):
    source = root / 'quality_reports/reports'
    destination = root / 'data/collections'
    all_files = list(files_under(source))
    containers = set()
    for path in all_files:
        if path.is_symlink():
            continue
        parts = path.relative_to(source).parts
        indices = [i for i, part in enumerate(parts[:-1]) if part == 'runs' or part.startswith('runs_')]
        if indices:
            containers.add(source.joinpath(*parts[:indices[0] + 1]))
        elif path.name == 'run_record.json':
            containers.add(path.parent)
    containers = sorted(p for p in containers if not any(parent in containers for parent in p.parents))
    items = []
    for path in containers:
        members = []
        for member in files_under(path):
            if member.is_symlink():
                members.append(dict(path=str(member.relative_to(path)), link=os.readlink(member),
                                    target_existed=member.exists()))
            else:
                members.append(dict(path=str(member.relative_to(path)), size=member.stat().st_size,
                                    sha256=checksum(member)))
        items.append(dict(source=str(path.relative_to(root)),
                          destination=str((destination / path.relative_to(source)).relative_to(root)),
                          kind='directory', members=members))
    for path in all_files:
        if path.is_symlink() or any(parent in containers for parent in path.parents):
            continue
        if path.suffix.lower() not in DATA_SUFFIXES:
            continue
        items.append(dict(source=str(path.relative_to(root)),
                          destination=str((destination / path.relative_to(source)).relative_to(root)),
                          kind='file', size=path.stat().st_size, sha256=checksum(path)))
    return dict(created_at=datetime.now(timezone.utc).isoformat(), items=items,
                method='rename_with_compatibility_symlink', changes_sample=False)


def move_one(root, item):
    source, destination = root / item['source'], root / item['destination']
    if source.is_symlink():
        if source.resolve() != destination.resolve() or not destination.exists():
            raise ValueError(f'Unexpected existing alias: {source}')
        return
    companion = destination.is_symlink() and destination.resolve() == source.resolve()
    if companion:
        destination.unlink()
    elif destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    try:
        source.symlink_to(os.path.relpath(destination, source.parent), target_is_directory=item['kind'] == 'directory')
    except BaseException:
        destination.rename(source)
        raise


def verify_item(root, item, moved=True):
    source, destination = root / item['source'], root / item['destination']
    if moved and (not source.is_symlink() or source.resolve() != destination.resolve()):
        raise ValueError(f'Alias differs: {source}')
    base = destination if moved else source
    members = item.get('members', [dict(path='', size=item.get('size'), sha256=item.get('sha256'))])
    for member in members:
        path = base / member['path'] if member['path'] else base
        if 'link' in member:
            if not path.is_symlink() or os.readlink(path) != member['link']:
                raise ValueError(f'Captured link changed: {path}')
            if member.get('target_existed') and not path.exists():
                raise ValueError(f'Captured link lost its target: {path}')
        elif not path.is_file() or checksum(path) != member['sha256']:
            raise ValueError(f'File bytes differ: {path}')


def companion_links(root):
    """Keep relative source/report companions available beside moved metadata."""
    source, destination = root / 'quality_reports/reports', root / 'data/collections'
    count = 0
    for path in files_under(source):
        target = destination / path.relative_to(source)
        if path.is_symlink() or target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(path, target.parent))
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'apply', 'verify'])
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    receipt = args.root / 'data/inventory/storage_migration_2026-09-18.json'
    if args.action == 'plan':
        if receipt.exists():
            raise FileExistsError('Existing receipt must be retained; do not overwrite it')
        plan = make_plan(args.root)
        save(receipt, plan)
    else:
        plan = json.loads(receipt.read_text())
        if args.action == 'apply':
            for i, item in enumerate(plan['items']):
                if not (args.root / item['source']).is_symlink():
                    verify_item(args.root, item, moved=False)
                move_one(args.root, item)
                if i % 1000 == 0:
                    print(f'Moved or verified {i + 1}/{len(plan["items"])} paths', flush=True)
            companions = companion_links(args.root)
            save(receipt.parent / 'storage_companions.json', dict(created_links=companions))
        for item in plan['items']:
            verify_item(args.root, item)
        remaining = [str(p.relative_to(args.root)) for p in files_under(args.root / 'quality_reports/reports')
                     if not p.is_symlink() and (p.name == 'run_record.json' or p.suffix.lower() in DATA_SUFFIXES)]
        result = dict(verified_at=datetime.now(timezone.utc).isoformat(),
                      moved_paths=len(plan['items']),
                      moved_files=sum(len(x.get('members', [x])) for x in plan['items']),
                      remaining_data_files=remaining, hashes_verified=True)
        save(receipt.parent / 'storage_verification.json', result)
        print(json.dumps(result, indent=2))
        if remaining:
            raise ValueError('Report data remains outside the migration receipt')
    print(json.dumps(dict(items=len(plan['items']), receipt=str(receipt))))


if __name__ == '__main__':
    main()
