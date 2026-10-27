"""Tests for the reference-validation table (code/benchmark/paper_reference_validation.py).

The installed CSV and TeX table must be the generator output. Every scored term of the
40 tasks must have an independent route in R or Stata. Only the Cox task may exceed the
1e-6 agreement level, and its R, Stata, and converged Python routes must agree with each
other. The tests read the stored route files and make no external calls.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "benchmark"))

import paper_reference_validation as prv  # noqa: E402
from task_labels import TASK_LABELS  # noqa: E402


@pytest.fixture(scope="module")
def built():
    if not prv.RECOMPUTED.exists():
        pytest.skip("recomputed routes missing; run paper_reference_validation.py recompute")
    return prv.build_rows()


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_relative_difference_definition():
    assert prv.relative(1.0 + 1e-9, 1.0) == pytest.approx(1e-9)
    assert prv.relative(-2.0, -1.0) == pytest.approx(1.0)
    assert prv.relative(1e-3, 0.0) == pytest.approx(1e-3)


def test_tex_number_format():
    assert prv.tex_number(None) == "--"
    assert prv.tex_number(0.0) == "$0$"
    assert prv.tex_number(3.594e-6) == r"$3.6\times10^{-6}$"
    assert prv.tex_number(5e-17) == r"${<}10^{-15}$"


def test_manual_cooks_mapping():
    raw = {"top5_obs_ids": [190, 214], "top5_cooks_d": [0.0285, 0.0246], "n_obs": 500}
    mapped = prv.map_manual("cooks_distance", raw)
    assert mapped["rank1_obs_id"] == 190 and mapped["rank2_cooks_d"] == 0.0246


def test_manual_se_mapping():
    mapped = prv.map_manual("twoway_fe", {"x": 0.5, "se_x_cluster_firm": 0.02})
    assert mapped == {"x": 0.5, "x:se": 0.02}
    mapped = prv.map_manual("cox_hazard", {"coef_treat": -0.5, "se_treat": 0.05})
    assert mapped == {"treat": -0.5, "treat:se": 0.05}


def test_v2_files_need_the_catalog_input():
    catalog = prv.load_catalog()
    # The v2 Cox package uses new synthetic data, so it cannot validate the stored reference.
    terms = set(prv.references(catalog["cox_hazard"]))
    assert prv.v2_routes("cox_hazard", catalog["cox_hazard"], terms) == []
    # The v2 logit package reads the catalog input, so its R and Stata routes count.
    terms = set(prv.references(catalog["logit"]))
    languages = {r["language"] for r in prv.v2_routes("logit", catalog["logit"], terms)}
    assert {"R", "Stata"} <= languages


def test_recomputed_routes_used_catalog_inputs():
    if not prv.RECOMPUTED.exists():
        pytest.skip("recomputed routes missing")
    catalog = prv.load_catalog()
    recomputed = json.loads(prv.RECOMPUTED.read_text())
    for task, digest in recomputed["input_sha256"].items():
        assert digest == prv.sha256(catalog[task]["data_path"]), task
    for task, routes in recomputed["routes"].items():
        for name, body in routes.items():
            assert body["returncode"] == 0, (task, name)
            assert body["values"], (task, name)


def test_one_row_per_task_in_table_a2_order(built):
    summaries, _ = built
    catalog = prv.load_catalog()
    assert [s["task"] for s in summaries] == sorted(catalog, key=lambda t: TASK_LABELS[t].lower())
    assert len(summaries) == 40
    assert set(prv.REFERENCE) == set(catalog)


def test_every_scored_term_has_an_r_or_stata_route(built):
    summaries, _ = built
    se_terms = prv.scored_se_terms()
    assert sum(1 for terms in se_terms.values() if terms) == 9
    for s in summaries:
        assert s["terms_without_independent_route"] == "", s["task"]
        assert s["terms_without_other_language_route"] == "", s["task"]
        assert {"R", "Stata"} <= set(s["independent_languages"].split(", ")), s["task"]
        for term in se_terms.get(s["task"], []):
            assert term + ":se" in s["scored_terms"].split(";"), s["task"]


def test_only_cox_exceeds_agreement_level(built):
    summaries, _ = built
    above = {s["task"]: s["max_rel_diff_scored"] for s in summaries if s["max_rel_diff_scored"] > prv.AGREEMENT}
    assert set(above) == {"cox_hazard"}
    assert 3e-6 < above["cox_hazard"] < 4e-6
    assert set(above) <= set(prv.EXPLANATIONS)


def test_cox_converged_routes_agree_with_each_other(built):
    _, details = built
    for term in ("treat", "x"):
        values = [r["route_value"] for r in details if r["task"] == "cox_hazard" and r["term"] == term
                  and r["route_id"] in ("manual:r", "manual:stata", "check:python")]
        assert len(values) == 3
        assert max(prv.relative(a, b) for a, b in combinations(values, 2)) < 1e-11


def test_conventions_are_exact_after_rescaling(built):
    _, details = built
    rescaled = [r for r in details if r["convention_factor"] != "" and r["compared"]]
    assert {r["task"] for r in rescaled} == {"twoway_fe", "lp_irf", "iv2sls_robust"}
    for row in rescaled:
        assert row["rel_diff"] < 1e-12, (row["task"], row["route_id"], row["term"])
    for key in prv.CONVENTIONS:
        assert any((r["task"], r["route_id"], r["term"]) == key for r in details), key


def test_installed_csv_matches_generator(built):
    summaries, _ = built
    rows = read_csv(prv.OUT_CSV)
    assert [r["task"] for r in rows] == [s["task"] for s in summaries]
    for row, s in zip(rows, summaries):
        assert float(row["max_rel_diff_scored"]) == pytest.approx(s["max_rel_diff_scored"], rel=1e-12, abs=1e-300)
        assert row["status"] == s["status"]


def test_installed_tex_matches_generator(built):
    summaries, _ = built
    assert prv.OUT_TEX.read_text() == prv.render_tex(summaries)


def test_tex_style():
    text = prv.OUT_TEX.read_text()
    spec = re.search(r"\\begin\{longtable\}\{(.*)\}\n", text).group(1)
    assert "|" not in spec
    assert r"\toprule" in text and r"\midrule" in text and r"\bottomrule" in text
    assert r"\label{tab:reference_validation}" in text
    caption = re.search(r"\\caption\{([^\\]*)\\label", text).group(1)
    assert caption == "Validation of the Reference Outputs"
    assert not caption.endswith(".")
    assert "\u2014" not in text and ";" not in text.split(r"\textit{Notes:}")[1]
    assert len(re.findall(r"\\\\\*?$", text, re.M)) >= 41
