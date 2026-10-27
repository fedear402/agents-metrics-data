"""Collect paper tasks with Claude Opus 5.5 at effort low, billed to an Anthropic API key.

Decision record: quality_reports/decision_records/2026-10-19_opus55-one-task.md.
This entry point is the Haiku API-key collector (prospective_collect_haiku_api.py: unlimited additive-R entry point,
Claude Code through a launch script, paper-time Stata ANSWER reader) with the model claude-opus-5-5
and Claude Code 2.1.280 through opus_api_claude.sh.
The job label stays "sonnet". The manifest pins this file and the Haiku collector.
"""
from pathlib import Path
import os
import tempfile

import prospective_collect_haiku_api as haiku

central = haiku.central
haiku.MODEL = ('claude-opus-5-5', 'low')  # read by haiku.configure and haiku.prepare
haiku.VERSION = 'prospective_opus55_api_2026_10_19_v1'
haiku.PRICE_PER_MILLION = dict(input=4.00, cache_write_5m=5.00, cache_write_1h=8.00, cache_read=0.20, output=20.00)
# Opus 5.5 needs Claude Code 2.1.280 or newer; 2.1.278 refuses the model.
haiku.LAUNCHER = haiku.HERE / 'opus_api_claude.sh'
haiku.BINARY = haiku.ROOT / 'data/collections/opus_api_bin/claude-2.1.280'
haiku.BINARY_SHA256 = '387a5c5dcdbb815085edf0baf79591f9d8894efe922bceaf3d75b1b08055229d'
os.environ['CLAUDE_BIN'] = str(haiku.LAUNCHER)
BASE = haiku.prepare


def prepare(config_path, manifest_path):
    target = Path(manifest_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.opus_prep_', dir=target.parent) as temporary:
        result = BASE(config_path, Path(temporary) / 'manifest.json')
    here = Path(__file__).resolve()
    result['protocol']['source_hashes'][str(here)] = central.sha(here)
    result['protocol']['policy']['claude_code'] = '2.1.280 from data/collections/opus_api_bin (npm darwin-arm64 package)'
    result['protocol_sha256'] = central.digest(result['protocol'])
    if target.exists():
        old = central.json.loads(target.read_text())
        if old['protocol_sha256'] != result['protocol_sha256']:
            raise ValueError('Preserve the existing manifest; select a new destination')
        return old
    central.save(target, result)
    return result


central.prepare = prepare

if __name__ == '__main__':
    central.main()
