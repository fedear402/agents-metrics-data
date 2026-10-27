"""Collect the paper experiment with GPT-6 Sol through codex exec, as the paper's Luna runs.

Decision record: quality_reports/decision_records/2026-10-17_haiku55-sol-full-design.md.
This entry point loads prospective_collect_unlimited_additions.py, the entry point of the last paper
batches (no wall limits), and changes three things in its own process:
- The Codex model of the runtime is gpt-6-sol at effort low. The job label stays "luna",
  because the runtime tests that word in several places.
- CODEX_BIN is the pinned Codex 0.163.0 copy in data/collections/sol_codex_bin. Codex 0.154.0, the
  version of the paper's Luna runs, cannot run gpt-6-sol with a ChatGPT account.
- The Stata ANSWER reader is the strict regular expression of the paper runs (runner.py before
  commit ab79a1b8d4). The reader decides when Stata self-repair sends a repair.
The manifest pins this file and the binary.
"""
from pathlib import Path
import os
import re
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BINARY = ROOT / 'data/collections/sol_codex_bin/codex-0.163.0/bin/codex'
BINARY_SHA256 = '8813367285524ac798dd96d1b071aea30d58708c7226fd236cd275cf1cfdee31'
os.environ['CODEX_BIN'] = str(BINARY)  # read by codex_backend when it is first imported

import prospective_collect_unlimited_additions as additions  # noqa: E402

central = additions.central
VERSION = 'prospective_sol_codex_2026_10_17_v1'
MODEL = ('gpt-6-sol', 'low')
# The ANSWER reader of runner.py when the paper runs were collected (before commit ab79a1b8d4).
PAPER_ANSWER = re.compile(r'^ANSWER:\s*([A-Za-z0-9_]+)\s+estimate\s*=\s*(\S+)\s+se\s*=\s*(\S+)\s+n_obs\s*=\s*(\S+)\s*$')


def configure(runtime):
    """Set the Sol model, the pinned Codex binary, and the paper-time Stata reader in a fresh runtime."""
    collector = runtime.c.prior
    collector.MODELS['luna'] = MODEL
    runtime.c.codex.CODEX_BIN = str(BINARY)
    collector.codex.CODEX_BIN = str(BINARY)
    collector.base.ANSWER = PAPER_ANSWER
    return runtime


BASE_LOAD_RUNTIME = central.load_runtime


def load_runtime(work):
    return configure(BASE_LOAD_RUNTIME(work))


BASE_PREPARE = central.prepare


def prepare(config_path, manifest_path):
    if central.sha(BINARY) != BINARY_SHA256:
        raise ValueError('Codex 0.163.0 binary changed')
    target = Path(manifest_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sol_prep_', dir=target.parent) as temporary:
        result = BASE_PREPARE(config_path, Path(temporary) / 'manifest.json')
    protocol = result['protocol']
    protocol['version'] = VERSION
    protocol['policy'].update(
        codex_model=dict(model=MODEL[0], effort=MODEL[1], job_label='luna'),
        provider_route='ChatGPT login of the computer, as the paper Luna runs',
        codex='0.163.0 from data/collections/sol_codex_bin (Homebrew copy); 0.154.0 cannot run gpt-6-sol',
        stata_answer_reader='Paper-time strict ANSWER reader (runner.py before ab79a1b8d4)',
        cost_fields='Price Sol from the saved token counts at the GPT-6 Sol list prices.')
    for path in (Path(__file__).resolve(), BINARY, HERE / 'tests/test_sol_collect.py'):
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
