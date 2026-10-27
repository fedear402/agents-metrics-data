"""Keep production defaults central and prevent historical report writes."""
from pathlib import Path
import subprocess
import sys

from benchmark import production_paper_assets as assets


ROOT = Path(__file__).resolve().parents[3]


def test_assets_default_to_main_analysis():
    assert assets.DEFAULT_OUT == ROOT / 'data/analysis/main'


def test_historical_statistics_command_stops_before_writing(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / 'code/benchmark/production_statistics.py')],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert 'archived 18-task sample' in result.stderr
    assert 'code/paper/rebuild.py' in result.stderr
    assert list(tmp_path.iterdir()) == []
