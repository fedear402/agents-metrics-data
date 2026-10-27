"""Verify path migration without changing saved response bytes."""
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("paper_dataset", ROOT / "code/paper/dataset.py")
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)


def test_relative_report_output_resolves_after_report_archive():
    relative = "quality_reports/reports/2026-09-17_collection-readiness/work/collector/result.json"
    expected = ROOT / "data/collections/2026-09-17_collection-readiness/work/collector/result.json"
    assert dataset.canonical_path(relative) == expected.resolve()
    assert dataset.canonical_path(ROOT / relative) == expected.resolve()


def test_current_physical_path_stays_unchanged():
    path = ROOT / "data/task_candidates/example/input.csv"
    assert dataset.canonical_path(path) == path.resolve()


def test_saved_explicit_alias_takes_precedence(monkeypatch):
    old = ROOT / "quality_reports/reports/old/answer.json"
    new = ROOT / "data/collections/new/answer.json"
    monkeypatch.setitem(dataset.ALIASES, str(old), str(new))
    assert dataset.canonical_path(old) == new.resolve()


def test_repository_relative_paths_round_trip():
    path = ROOT / "data/runs/example/run_record.json"
    assert dataset.repo_relative(path) == "data/runs/example/run_record.json"
    assert dataset.repo_path(dataset.repo_relative(path)) == path
    assert dataset.repo_relative("/elsewhere/file.json") == "/elsewhere/file.json"


def test_paper_sample_is_the_default():
    import inspect
    assert inspect.signature(dataset.build).parameters["use_replacements"].default is True


def test_runs_csv_has_relative_paths_and_no_stale_columns():
    import csv
    import pytest
    path = ROOT / "data/analysis/main/data/runs.csv"
    if not path.is_file():
        pytest.skip("current analysis products are absent")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert not set(dataset.STALE_COLUMNS) & set(reader.fieldnames)
        rows = list(reader)
    for row in rows:
        assert not Path(row["record_path"]).is_absolute()
        assert not Path(row["output_path"]).is_absolute()
        assert not row.get("manifest_path") or not Path(row["manifest_path"]).is_absolute()


def test_general_rule_switches_off_the_exceptions_and_restores_them():
    import types
    sp = types.SimpleNamespace(ABS_COMPARE_ESTIMANDS={("ttest_welch", "t")},
                               MULTIPLE_CONVENTION_ESTIMANDS={("var_granger", "df_den"): (2.0,)})
    with dataset.general_rule(sp):
        assert sp.ABS_COMPARE_ESTIMANDS == set() and sp.MULTIPLE_CONVENTION_ESTIMANDS == {}
    assert sp.ABS_COMPARE_ESTIMANDS == {("ttest_welch", "t")}
    assert sp.MULTIPLE_CONVENTION_ESTIMANDS == {("var_granger", "df_den"): (2.0,)}


def test_term_checks_score_the_welch_sign_and_the_granger_reference_df():
    import pytest
    import pandas as pd
    path = ROOT / "data/analysis/main/data/term_checks.csv"
    if not path.is_file():
        pytest.skip("current analysis products are absent")
    terms = pd.read_csv(path, low_memory=False)
    terms["passed"] = (terms.coef_pass.combine_first(terms.exact_estimate_pass)
                       .combine_first(terms.estimate_pass).fillna(False).astype(bool))
    welch = terms[(terms.task == "ttest_welch") & (terms.term == "t")]
    assert not welch[welch.got_estimate * welch.ref_estimate < 0].passed.any()
    granger = terms[(terms.task == "var_granger") & (terms.term == "df_den")]
    assert not granger[(granger.got_estimate - 2 * granger.ref_estimate).abs() < 1e-9].passed.any()
