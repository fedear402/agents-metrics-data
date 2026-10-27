"""Archive an explicit reviewed allowlist without following symbolic links.

Usage: archive_followup.py [inventory|copy|remove] --plan PATH
The default only inventories sources. The JSON plan contains approved_items
(repository-relative paths), protected_paths (relative or absolute paths), and
archive_root (a new directory under ~/ARCHIVE/AGENTS-METRICS). Keep the plan
under data/maintenance/production40_cleanup_followup. Review its paths before
copying. Only the explicit remove phase deletes verified source items.
Restore each item from archive_root/payload to its original relative path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat

ROOT = Path(__file__).resolve().parents[4]
INDEX = Path('data/maintenance/production40_cleanup_followup')
ARCHIVE_BASE = Path.home() / 'ARCHIVE/AGENTS-METRICS'
_spec = importlib.util.spec_from_file_location('followup_cleanup_helpers', Path(__file__).with_name('cleanup.py'))
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def present(path):
    return os.path.lexists(path)


def no_link_parents(path, include_leaf=False):
    """Reject any symbolic link in a control path or item ancestor."""
    paths = [path, *path.parents] if include_leaf else path.parents
    if any(p.is_symlink() for p in paths):
        raise ValueError(f'Symbolic link in path ancestry: {path}')


def overlap(first, second):
    return first.is_relative_to(second) or second.is_relative_to(first)


def save(path, value):
    no_link_parents(path, True)
    no_link_parents(path.with_suffix(path.suffix + '.tmp'), True)
    helpers.save(path, value)


def absolute(value, root):
    path = Path(value)
    if '..' in path.parts:
        raise ValueError(f'Path traversal is forbidden: {value}')
    return path if path.is_absolute() else root / path


def load_plan(path, root=ROOT, archive_base=ARCHIVE_BASE):
    root, archive_base = Path(root).absolute(), Path(archive_base).absolute()
    path = absolute(path, root)
    if not path.is_relative_to(root / INDEX):
        raise ValueError('The plan must be under data/maintenance/production40_cleanup_followup')
    no_link_parents(path, True)
    plan = json.loads(path.read_text())
    items, protected = plan['approved_items'], plan['protected_paths']
    if not isinstance(items, list) or not items or not all(isinstance(v, str) and v for v in items):
        raise ValueError('approved_items must be a nonempty list of paths')
    if not isinstance(protected, list) or not protected or not all(isinstance(v, str) and v for v in protected):
        raise ValueError('protected_paths must be a nonempty list of dependency paths')
    out = absolute(plan['archive_root'], root)
    if out == archive_base or not out.is_relative_to(archive_base) or overlap(out, root):
        raise ValueError('Archive destination must be a child of the external archive base')
    no_link_parents(out, True)
    sources = []
    for item in items:
        relative = Path(item)
        if relative.is_absolute() or '..' in relative.parts or relative == Path('.'):
            raise ValueError(f'Approved item must be a non-root relative path: {item}')
        source = root / relative
        no_link_parents(source)
        if any(overlap(source, other) for other in sources):
            raise ValueError(f'Approved items overlap: {item}')
        sources.append(source)
    dependencies = [path, *(path.parent / name for name in
        ('archive_manifest.json', 'copy_receipt.json', 'removal_receipt.json')),
        root / '.git', root / 'AGENTS.md', root / 'CLAUDE.md']
    for value in protected:
        dependency = absolute(value, root)
        dependencies.extend([dependency, dependency.resolve()])
    dependencies = set(dependencies)
    ancestors = {parent for dependency in dependencies for parent in dependency.parents}
    for source in sources:
        if source in dependencies or source in ancestors or any(parent in dependencies for parent in source.parents):
            raise ValueError(f'Approved source overlaps a protected dependency: {source}')
    return dict(root=root, plan=path, index=path.parent, out=out, sources=sources,
                items=[str(p.relative_to(root)) for p in sources], plan_sha256=helpers.sha(path))


def describe(path):
    """Include the root item, directory modes, file bytes, and link text."""
    no_link_parents(path)
    info = path.lstat()
    row = dict(path='.', mode=stat.S_IMODE(info.st_mode))
    if stat.S_ISLNK(info.st_mode):
        return [dict(row, kind='symlink', target=os.readlink(path))]
    if stat.S_ISREG(info.st_mode):
        return [dict(row, kind='file', size=info.st_size, sha256=helpers.sha(path))]
    if stat.S_ISDIR(info.st_mode):
        return [dict(row, kind='directory'), *helpers.entries(path)]
    raise ValueError(f'Unsupported source type: {path}')


def verify(path, expected):
    if not present(path) or describe(path) != expected:
        raise ValueError(f'Content differs from inventory: {path}')


def inventory(config):
    path = config['index'] / 'archive_manifest.json'
    if present(path):
        no_link_parents(path, True)
        manifest = json.loads(path.read_text())
        if (manifest['plan_sha256'] != config['plan_sha256'] or
                manifest['source_root'] != str(config['root']) or
                manifest['archive_root'] != str(config['out']) or
                list(manifest['items']) != config['items']):
            raise ValueError('The reviewed plan or inventory roots changed')
        return manifest
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    plan_sha256=config['plan_sha256'], source_root=str(config['root']),
                    archive_root=str(config['out']), items={})
    for item in config['items']:
        manifest['items'][item] = describe(config['root'] / item)
    save(path, manifest)
    return manifest


def copy_receipt(config):
    path = config['index'] / 'copy_receipt.json'
    no_link_parents(path, True)
    receipt = json.loads(path.read_text())
    expected = helpers.sha(config['index'] / 'archive_manifest.json')
    if receipt.get('manifest_sha256') != expected or receipt.get('verified') is not True:
        raise ValueError('Verified copy receipt is unavailable')
    for name in ('archive_manifest.json', 'plan.json', 'copy_receipt.json'):
        external = config['out'] / name
        no_link_parents(external, True)
        local = config['plan'] if name == 'plan.json' else config['index'] / name
        if helpers.sha(external) != helpers.sha(local):
            raise ValueError(f'External archive metadata changed: {external}')
    return receipt


def verify_all(config, manifest, require_destination=False):
    """Check every source and destination before any removal."""
    missing = False
    for item, expected in manifest['items'].items():
        source, destination = config['root'] / item, config['out'] / 'payload' / item
        no_link_parents(source)
        no_link_parents(destination)
        if present(source):
            verify(source, expected)
        else:
            missing = True
            verify(destination, expected)
        if require_destination or present(destination):
            verify(destination, expected)
    if missing:
        copy_receipt(config)


def copy(config, manifest):
    if present(config['index'] / 'copy_receipt.json'):
        copy_receipt(config)
    verify_all(config, manifest)
    for item, expected in manifest['items'].items():
        source, destination = config['root'] / item, config['out'] / 'payload' / item
        if present(destination):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A complete interrupted copy can resume. Partial copies remain for review.
        temporary = destination.with_name(destination.name + '.copying')
        if not present(temporary):
            if source.is_symlink():
                temporary.symlink_to(os.readlink(source))
                shutil.copystat(source, temporary, follow_symlinks=False)
            elif source.is_dir():
                shutil.copytree(source, temporary, symlinks=True)
            else:
                shutil.copy2(source, temporary, follow_symlinks=False)
        verify(temporary, expected)
        verify(source, expected)
        temporary.rename(destination)
    verify_all(config, manifest, require_destination=True)
    for name, source in [('archive_manifest.json', config['index'] / 'archive_manifest.json'), ('plan.json', config['plan'])]:
        destination = config['out'] / name
        no_link_parents(destination, True)
        if present(destination) and helpers.sha(destination) != helpers.sha(source):
            raise ValueError(f'External archive metadata changed: {destination}')
        if not present(destination):
            shutil.copy2(source, destination)
    receipt = dict(verified=True, manifest_sha256=helpers.sha(config['index'] / 'archive_manifest.json'),
                   plan_sha256=config['plan_sha256'], archive_root=str(config['out']))
    for base in (config['index'], config['out']):
        no_link_parents(base / 'copy_receipt.json', True)
        save(base / 'copy_receipt.json', receipt)
    copy_receipt(config)


def remove(config, manifest):
    copy_receipt(config)
    verify_all(config, manifest, require_destination=True)
    for item, expected in manifest['items'].items():
        source = config['root'] / item
        if present(source):
            verify(source, expected)
            if source.is_dir() and not source.is_symlink():
                shutil.rmtree(source)
            else:
                source.unlink()
    receipt = dict(removed=True, manifest_sha256=helpers.sha(config['index'] / 'archive_manifest.json'),
                   archive_root=str(config['out']), items=config['items'])
    for base in (config['index'], config['out']):
        no_link_parents(base / 'removal_receipt.json', True)
        save(base / 'removal_receipt.json', receipt)


def run(phase, plan, root=ROOT, archive_base=ARCHIVE_BASE):
    config = load_plan(plan, root, archive_base)
    manifest = inventory(config)
    if phase == 'inventory':
        verify_all(config, manifest)
    elif phase == 'copy':
        copy(config, manifest)
    elif phase == 'remove':
        remove(config, manifest)
    else:
        raise ValueError(f'Unknown phase: {phase}')
    return dict(phase=phase, items=len(config['items']), archive_root=str(config['out']),
                manifest_sha256=helpers.sha(config['index'] / 'archive_manifest.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', choices=['inventory', 'copy', 'remove'], default='inventory')
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.phase, args.plan)))


if __name__ == '__main__':
    main()
