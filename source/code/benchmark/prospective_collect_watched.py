"""Collect with the provider usage watcher on top of the additive-R collector.

The central collector, the unlimited-walltime wrapper, and the additive-R wrapper stay
unchanged. This entry point loads fresh copies of them and adds two hooks:

- Before each provider request, the quota coordinator asks ``usage_watch``. A pause waits
  until the recorded reset time plus the chosen margin, then the request starts.
- After each provider call, the watcher reads the saved stream for limit data.

Manifests prepared here record the watcher version and its chosen limits in the policy.
Commands are the same as those of ``prospective_collect.py``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import usage_watch  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Fresh copies keep other importers of the wrappers free of these hooks.
_unlimited = _load('prospective_watched_unlimited', HERE / 'prospective_collect_unlimited.py')
_previous = sys.modules.get('prospective_collect_unlimited')
sys.modules['prospective_collect_unlimited'] = _unlimited
try:
    additions = _load('prospective_watched_additions', HERE / 'prospective_collect_unlimited_additions.py')
finally:
    if _previous is None:
        sys.modules.pop('prospective_collect_unlimited', None)
    else:
        sys.modules['prospective_collect_unlimited'] = _previous
central = additions.central
watcher = usage_watch.install(central)
BASE_PREPARE = central.prepare


def install_watcher(instance: usage_watch.Watcher) -> None:
    """Use a given watcher, for example one with a test clock."""
    usage_watch.install(central, instance)


def prepare(config_path, manifest_path):
    """Prepare the additive-R protocol and record the watcher policy."""
    target = Path(manifest_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.usage_watch_prep_', dir=target.parent) as temporary:
        manifest = BASE_PREPARE(config_path, Path(temporary) / 'manifest.json')
    protocol = manifest['protocol']
    protocol['policy'].update(
        usage_watch_version=usage_watch.VERSION,
        usage_watch_config=usage_watch.load_config(),
        usage_watch='Before each provider request, wait while the usage watcher reports a pause. '
                    'Pause and slow limits are chosen settings. The central rejection handling is unchanged.')
    for path in (HERE / 'usage_watch.py', HERE / 'prospective_collect_watched.py',
                 HERE / 'tests/test_usage_watch.py'):
        protocol['source_hashes'][str(path)] = central.sha(path)
    manifest['protocol_sha256'] = central.digest(protocol)
    if target.exists():
        old = central.json.loads(target.read_text())
        if old['protocol_sha256'] != manifest['protocol_sha256']:
            raise ValueError('Preserve the existing manifest; select a new destination')
        return old
    central.save(target, manifest)
    return manifest


central.prepare = prepare

if __name__ == '__main__':
    central.main()
