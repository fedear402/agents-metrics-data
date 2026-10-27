"""The ordinary refresh command must use the selected forty-task paper."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]


def test_refresh_dispatches_current_paper_and_preserves_arguments(tmp_path):
    capture = tmp_path / 'dispatch.json'
    fake_uv = tmp_path / 'uv'
    fake_uv.write_text(
        f'#!{sys.executable}\n'
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'Path(os.environ["DISPATCH_CAPTURE"]).write_text(json.dumps('
        '{"argv": sys.argv[1:], "cwd": os.getcwd()}))\n'
        'sys.exit(37)\n'
    )
    fake_uv.chmod(0o755)
    environment = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}',
                       DISPATCH_CAPTURE=str(capture))
    result = subprocess.run(
        ['bash', str(ROOT / 'code/benchmark/refresh_manuscript.sh'),
         '--no-render', '--out', str(tmp_path / 'analysis')],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 37
    assert json.loads(capture.read_text()) == {
        'cwd': str(ROOT),
        'argv': ['run', '--no-sync', 'python',
                 'code/paper/rebuild.py',
                 '--no-render', '--out', str(tmp_path / 'analysis')],
    }
