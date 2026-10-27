"""The website must use the same run costs as the paper (data/run_pricing.csv)."""
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/falstuff_site"))
import main_sample


def test_site_rows_carry_paper_run_pricing():
    _catalog, rows, _summary = main_sample.load_main_sample()
    with (main_sample.DATA_DIR / "run_pricing.csv").open(newline="") as source:
        pricing = {row["record_path"]: row for row in csv.DictReader(source)}
    assert len(pricing) == len(rows)
    for row in rows:
        paper = pricing[row["record_path"]]
        assert row["cost_pricing_complete"] == paper["cost_pricing_complete"]
        if paper["cost_usd"]:
            assert math.isclose(float(row["cost_usd"]), float(paper["cost_usd"]), rel_tol=0, abs_tol=1e-12)
        else:
            assert row["cost_usd"] == ""
