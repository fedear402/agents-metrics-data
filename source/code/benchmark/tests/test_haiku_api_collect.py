"""The Haiku API-key collector keeps the paper's Sonnet command and changes only the model, effort, and executable.

Rule file: quality_reports/decision_records/2026-10-16_haiku55-api-collection.md. The collector is loaded in a
separate process, because importing it changes the shared central collector module.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parents[1]
LAUNCHER = BENCH / 'haiku_api_claude.sh'
PAPER_MANIFEST = ROOT / 'data/runs/production_repairs_2026-09-21/sonnet/0_4/manifest.json'
PROBE = r'''
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import prospective_collect_haiku_api as h
runtime = h.central.load_runtime(Path(tempfile.mkdtemp()))
commands = {}
for condition in ('chatbot', 'self_repair', 'agent'):
    job = dict(task='broad_tobit', model='sonnet', platform='python', prompt='zero_shot', condition=condition,
               output_format='json')
    commands[condition] = runtime.c.command(job, Path('<cwd>'), Path('<last>'))
base = runtime.c.prior.base
print(json.dumps(dict(commands=commands, model=runtime.c.prior.MODELS['sonnet'],
                      full=base.ANSWER.match('ANSWER: b estimate=1.5 se=0.2 n_obs=10') is not None,
                      partial=base.ANSWER.match('ANSWER: b estimate=1.5') is not None)))
'''


@pytest.fixture(scope='module')
def probe():
    env = {k: v for k, v in os.environ.items() if k not in ('CLAUDE_BIN', 'ANTHROPIC_API_KEY')}
    out = subprocess.run([sys.executable, '-c', PROBE, str(BENCH)], capture_output=True, text=True, env=env,
                         cwd=ROOT, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_model_and_effort(probe):
    assert tuple(probe['model']) == ('claude-haiku-5-5', 'low')


@pytest.mark.skipif(not PAPER_MANIFEST.is_file(), reason='paper manifest not present')
def test_command_equals_paper_command_except_model_effort_and_executable(probe):
    paper = json.loads(PAPER_MANIFEST.read_text())['protocol']['commands']
    for condition, (argv, timeout) in probe['commands'].items():
        old_argv, old_timeout = paper['sonnet/' + condition]
        assert timeout is None and old_timeout is None
        assert argv[0] == str(LAUNCHER)
        expected = [str(LAUNCHER)] + old_argv[1:]
        expected[expected.index('claude-sonnet-4-6')] = 'claude-haiku-5-5'
        expected[expected.index('--effort') + 1] = 'low'
        assert argv == expected, condition


def test_paper_time_stata_reader(probe):
    assert probe['full'] is True
    assert probe['partial'] is False


def test_launch_script_and_collector_hold_no_key():
    launcher = LAUNCHER.read_text()
    collector = (BENCH / 'prospective_collect_haiku_api.py').read_text()
    assert 'sk-ant' not in launcher and 'sk-ant' not in collector
    assert 'HAIKU_API_KEY_FILE' in launcher
    assert 'export CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1' in launcher
