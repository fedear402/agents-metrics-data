"""The website must read every statistic from the generated data bundles.

The static pages and the client script may hold labels and prose, never a rate,
a run count, or a task-level number. Those arrive through the ``narrative``
block that ``build_site_data.py`` writes into ``site-data.js`` and
``chart-data.js``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

SITE_DIR = Path(__file__).resolve().parents[2] / "falstuff_site"
sys.path.insert(0, str(SITE_DIR))

import build_public_site  # noqa: E402
import build_site_data  # noqa: E402
from main_sample import load_main_sample, truth  # noqa: E402

# A statistic-shaped literal: a percentage, a thousands-separated count, or a
# bare three-or-four digit run count.
PERCENT_LITERAL = re.compile(r"\b\d{1,3}\.\d\s?(?:%|percent\b)")
THOUSANDS_LITERAL = re.compile(r"\b\d{1,3},\d{3}\b")
RUN_COUNT_LITERAL = re.compile(r"\b(?:630|945|1890|5670|315|189)\b")

SOURCE_PAGES = ("explore.html", "presentation.html", "index.html")
SOURCE_SCRIPTS = ("app.js", "charts.js", "site-text.js")

# Evidence-band edges name the display scale, so they are not measured rates.
BAND_PHRASES = ("50% to 74.9%", "75% or higher", "Below 50%")


def read(name: str) -> str:
    return (SITE_DIR / name).read_text(encoding="utf-8")


def read_without_bands(name: str) -> str:
    text = read(name)
    for phrase in BAND_PHRASES:
        text = text.replace(phrase, "")
    return text


def load_bundle(name: str, global_name: str) -> dict:
    text = read(name)
    payload = text.split("=", 1)[1].strip()
    assert global_name in text
    return json.loads(payload.rstrip().rstrip(";"))


@pytest.mark.parametrize("name", SOURCE_PAGES + SOURCE_SCRIPTS)
def test_source_files_carry_no_percentage_literal(name: str) -> None:
    text = read_without_bands(name)
    found = {match.group(0) for match in PERCENT_LITERAL.finditer(text)}
    assert not found, f"{name} hardcodes a rate: {sorted(found)}"


@pytest.mark.parametrize("name", SOURCE_PAGES + SOURCE_SCRIPTS)
def test_source_files_carry_no_run_count_literal(name: str) -> None:
    text = read_without_bands(name)
    counts = {match.group(0) for match in THOUSANDS_LITERAL.finditer(text)}
    counts |= {match.group(0) for match in RUN_COUNT_LITERAL.finditer(text)}
    assert not counts, f"{name} hardcodes a run count: {sorted(counts)}"


def test_pages_bind_the_generated_narrative() -> None:
    for name in ("explore.html", "presentation.html"):
        text = read(name)
        assert 'data-site-text="narrative.' in text, name
        assert 'src="site-text.js"' in text, name


def test_public_bundle_ships_the_text_binder() -> None:
    assert "site-text.js" in build_public_site.TEXT_ASSETS


def test_narrative_reaches_both_generated_bundles() -> None:
    site = load_bundle("site-data.js", "AGENTS_METRICS_SITE_DATA")
    chart = load_bundle("chart-data.js", "AGENTS_METRICS_CHART_DATA")

    for payload in (site, chart):
        narrative = payload["narrative"]
        assert narrative["headline"]["pct"].endswith("%")
        assert narrative["sample"]["runs"]
        assert narrative["outcome"]["definition"].startswith("A run succeeds when")
        for key in ("no_feedback", "self_repair", "tool_loop"):
            assert narrative["interfaces"][key]["pct"].endswith("%")
        for key in ("python", "r", "stata"):
            assert narrative["platforms"][key]["pct"].endswith("%")

    assert site["narrative"] == chart["narrative"]


def test_every_bound_key_resolves_in_the_site_bundle() -> None:
    site = load_bundle("site-data.js", "AGENTS_METRICS_SITE_DATA")
    chart = load_bundle("chart-data.js", "AGENTS_METRICS_CHART_DATA")
    pattern = re.compile(r'data-site-(?:text|bar)="([^"]+)"')

    for name, payload in (
        ("explore.html", site),
        ("presentation.html", chart),
    ):
        for path in sorted(set(pattern.findall(read(name)))):
            value = payload
            for key in path.split("."):
                assert isinstance(value, dict) and key in value, f"{name}: {path}"
                value = value[key]
            assert value not in ("", None), f"{name}: {path} is empty"


def test_narrative_matches_the_analysis_headline() -> None:
    site = load_bundle("site-data.js", "AGENTS_METRICS_SITE_DATA")
    _, rows, _ = load_main_sample()
    assert site["narrative"]["headline"]["pct"] == f"{100 * sum(truth(r["primary"]) for r in rows) / len(rows):.1f}%"
    for key, agency in (("no_feedback", "chatbot"), ("self_repair", "self_repair"), ("tool_loop", "agent")):
        subset = [r for r in rows if r["agency"] == agency]
        expected = f"{100 * sum(truth(r["primary"]) for r in subset) / len(subset):.1f}%"
        assert site["narrative"]["interfaces"][key]["pct"] == expected


def test_site_contains_only_the_paper_sample() -> None:
    site = load_bundle("site-data.js", "AGENTS_METRICS_SITE_DATA")
    catalog, rows, _ = load_main_sample()
    assert {r["id"] for r in site["tasks"]} == set(catalog)
    assert site["analysisSample"]["repetitions"] == 10
    assert "extraResults" not in site
    assert "deepseek" not in site["runTraceIndex"]["models"]
    assert "sample-switch" not in read("explore.html")
    for payload, models in ((site, ("sonnet", "luna")),):
        assert {row["id"] for row in payload["models"]} == set(models)
        expected_n = len(rows)
        expected_passed = sum(truth(row["primary"]) for row in rows)
        assert payload["analysisSample"]["runs"] == expected_n
        assert len(payload["analysisSample"]["arms"]) == len(models) * 2
        for field in ("interfaces", "platforms", "promptModes", "execPlatform", "tasks"):
            assert sum(row["records"] for row in payload[field]) == expected_n, field
            assert sum(row["taskSuccessPassed"] for row in payload[field]) == expected_passed, field
        for task in payload["tasks"]:
            for field in ("interfaces", "platforms", "prompts", "cells"):
                assert sum(row["records"] for row in task["dimensionBreakdowns"][field]) == task["records"]
        assert len(payload["costComparisons"]) == len(models)
    assert all(row["meanCost"] > 0 for row in site["interfaces"])
    assert {row["costBasis"] for group in site["costComparisons"] for row in group["interfaces"]} == {"list_price"}


def test_website_title_matches_manuscript() -> None:
    frontmatter = (SITE_DIR.parents[1] / "paper/manuscript/sections/00_frontmatter.tex").read_text()
    title = "AI Agents and Prompt Engineering in Econometric Coding"
    assert title in frontmatter.replace("\\\\", " ").replace("  ", " ")
    for name in SOURCE_PAGES:
        assert title in read(name)


def test_task_success_description_states_the_unified_rule() -> None:
    description = dict(
        (key, text) for key, _, _, text in build_site_data.SUCCESS_CRITERIA
    )["task_success"]

    assert "reference output" in description
    assert "substantive estimates" in description
    assert "integrity" not in description
    assert "all four" not in description
