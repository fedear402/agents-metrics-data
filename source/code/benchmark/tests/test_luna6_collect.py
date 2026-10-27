"""The GPT-6 Luna collector keeps the paper's Luna command and changes only the model and the Codex binary path.

Decision record: quality_reports/decision_records/2026-10-19_gpt6-luna-full-design.md. The collector is loaded in
a separate process, because importing it changes the shared central collector module.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parents[1]
BINARY = ROOT / 'data/collections/sol_codex_bin/codex-0.163.0/bin/codex'
PAPER_MANIFEST = ROOT / 'data/runs/production_repairs_2026-09-21/luna/0_4/manifest.json'
PROBE = r'''
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import prospective_collect_luna6 as l
runtime = l.central.load_runtime(Path(tempfile.mkdtemp()))
commands = {}
for condition in ('chatbot', 'self_repair', 'agent'):
    job = dict(task='broad_tobit', model='luna', platform='python', prompt='zero_shot', condition=condition,
               output_format='json')
    commands[condition] = runtime.c.command(job, Path('<cwd>'), Path('<last>'))
base = runtime.c.prior.base
print(json.dumps(dict(commands=commands, model=runtime.c.prior.MODELS['luna'], bin=runtime.c.codex.CODEX_BIN,
                      partial=base.ANSWER.match('ANSWER: b estimate=1.5') is not None)))
'''


@pytest.fixture(scope='module')
def probe():
    env = {k: v for k, v in os.environ.items() if k != 'CODEX_BIN'}
    out = subprocess.run([sys.executable, '-c', PROBE, str(BENCH)], capture_output=True, text=True, env=env,
                         cwd=ROOT, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_model_effort_binary_and_reader(probe):
    assert tuple(probe['model']) == ('gpt-6-luna', 'low')
    assert probe['bin'] == str(BINARY)
    assert probe['partial'] is False


@pytest.mark.skipif(not PAPER_MANIFEST.is_file(), reason='paper manifest not present')
def test_command_equals_paper_luna_command_except_model_and_binary(probe):
    paper = json.loads(PAPER_MANIFEST.read_text())['protocol']['commands']
    for condition, (argv, timeout) in probe['commands'].items():
        old_argv, old_timeout = paper['luna/' + condition]
        assert timeout is None and old_timeout is None
        expected = [str(BINARY)] + old_argv[1:]
        expected[expected.index('gpt-5.6-luna')] = 'gpt-6-luna'
        assert argv == expected, condition
