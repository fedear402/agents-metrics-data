"""Collect paper tasks with Claude Sonnet 5.5 at effort low, billed to an Anthropic API key.

Decision record: quality_reports/decision_records/2026-10-23_sonnet55-full-design.md.
This entry point is the Opus 5.5 collector (prospective_collect_opus_api.py: the Haiku API-key collector with
Claude Code 2.1.280 through opus_api_claude.sh) with the model claude-sonnet-5-5 and its list prices.
The job label stays "sonnet". The manifest pins this file and the Haiku collector.
"""
from pathlib import Path
import tempfile

import prospective_collect_opus_api as opus

haiku = opus.haiku
central = haiku.central
haiku.MODEL = ('claude-sonnet-5-5', 'low')  # read by haiku.configure and haiku.prepare
haiku.VERSION = 'prospective_sonnet55_api_2026_10_23_v1'
# Claude Sonnet 5.5 list prices (claude-api skill model table, cached 2026-10-06; cache writes 1.25x and 2x input).
haiku.PRICE_PER_MILLION = dict(input=2.00, cache_write_5m=2.50, cache_write_1h=4.00, cache_read=0.20, output=10.00)


def prepare(config_path, manifest_path):
    target = Path(manifest_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sonnet55_prep_', dir=target.parent) as temporary:
        result = opus.prepare(config_path, Path(temporary) / 'manifest.json')
    here = Path(__file__).resolve()
    result['protocol']['source_hashes'][str(here)] = central.sha(here)
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
