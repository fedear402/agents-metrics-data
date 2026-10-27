"""Collect the paper experiment with Claude Haiku 5.5, billed to an Anthropic API key.

Rule file: quality_reports/decision_records/2026-10-16_haiku55-api-collection.md.
This entry point loads prospective_collect_unlimited_additions.py, the entry point of the last paper
batches (no wall limits), and changes three things in its own process:
- The Claude model of the runtime is claude-haiku-5-5 at effort low. The job label stays "sonnet",
  because the runtime tests that word in several places.
- CLAUDE_BIN is haiku_api_claude.sh. It starts Claude Code 2.1.278 and gives it the API key.
- The Stata ANSWER reader is the strict regular expression of the paper runs (runner.py before
  commit ab79a1b8d4). The reader decides when Stata self-repair sends a repair.
The manifest pins this file, the launch script, and the binary.
"""
from pathlib import Path
import os
import re
import tempfile
import prospective_collect_unlimited_additions as additions

central = additions.central
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERSION = 'prospective_haiku55_api_2026_10_16_v1'
MODEL = ('claude-haiku-5-5', 'low')
LAUNCHER = HERE / 'haiku_api_claude.sh'
BINARY = ROOT / 'data/collections/haiku55_api_bin/claude-2.1.278'
BINARY_SHA256 = 'bd245662fb8a0e321b3bf133e930371d6563c387527885f30b2613aef3ba14d6'
# The ANSWER reader of runner.py when the paper runs were collected (before commit ab79a1b8d4).
PAPER_ANSWER = re.compile(r'^ANSWER:\s*([A-Za-z0-9_]+)\s+estimate\s*=\s*(\S+)\s+se\s*=\s*(\S+)\s+n_obs\s*=\s*(\S+)\s*$')
PRICE_PER_MILLION = dict(input=0.10, cache_write_5m=0.125, cache_write_1h=0.20, cache_read=0.01, output=0.50)

os.environ['CLAUDE_BIN'] = str(LAUNCHER)


def configure(runtime):
    """Set the Haiku model and the paper-time Stata reader in a fresh runtime."""
    collector = runtime.c.prior
    if collector.harness.claude_bin() != str(LAUNCHER):
        raise ValueError('CLAUDE_BIN must be the Haiku API-key launch script')
    collector.MODELS['sonnet'] = MODEL
    collector.base.ANSWER = PAPER_ANSWER
    return runtime


BASE_LOAD_RUNTIME = central.load_runtime


def load_runtime(work):
    return configure(BASE_LOAD_RUNTIME(work))


BASE_PREPARE = central.prepare


def prepare(config_path, manifest_path):
    if central.sha(BINARY) != BINARY_SHA256:
        raise ValueError('Claude Code 2.1.278 binary changed')
    target = Path(manifest_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.haiku_prep_', dir=target.parent) as temporary:
        result = BASE_PREPARE(config_path, Path(temporary) / 'manifest.json')
    protocol = result['protocol']
    protocol['version'] = VERSION
    protocol['policy'].update(
        claude_model=dict(model=MODEL[0], effort=MODEL[1], job_label='sonnet'),
        provider_route='Anthropic API key, given only to the claude process by haiku_api_claude.sh',
        claude_code='2.1.278 from data/collections/haiku55_api_bin (npm darwin-arm64 package)',
        stata_answer_reader='Paper-time strict ANSWER reader (runner.py before ab79a1b8d4)',
        haiku_cost_per_million=PRICE_PER_MILLION,
        cost_fields='The collector cost fields use sonnet_cost_per_million; price Haiku from tokens.')
    for path in (Path(__file__).resolve(), LAUNCHER, BINARY, HERE / 'tests/test_haiku_api_collect.py'):
        protocol['source_hashes'][str(path)] = central.sha(path)
    result['protocol_sha256'] = central.digest(protocol)
    if target.exists():
        old = central.json.loads(target.read_text())
        if old['protocol_sha256'] != result['protocol_sha256']:
            raise ValueError('Preserve the existing manifest; select a new destination')
        return old
    central.save(target, result)
    return result


central.load_runtime = load_runtime
central.prepare = prepare

if __name__ == '__main__':
    central.main()
