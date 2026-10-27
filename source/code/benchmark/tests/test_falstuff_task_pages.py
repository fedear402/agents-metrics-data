"""Task pages and the task catalog on the project website.

Every analyzed task has a page under ``tasks/<task_id>/``; the explorer row
carries a ``More info`` link to it; the catalog lists every generated page;
no page carries a task number or a repository path.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "falstuff_site"))

import build_public_site  # noqa: E402
import build_task_pages  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SITE_DIR = PROJECT_ROOT / "code" / "falstuff_site"
TASKS_DIR = SITE_DIR / "tasks"

TASK_NUMBER_RE = re.compile(r"\bT\d{2}\b")
REPO_PATH_RE = re.compile(
    r"literature/|quality_reports/|data/tasks|textbooks/pdfs|/Users/|\.\./\.\./(?!explore\.html|tokens\.css|base\.css|explore\.css|tasks\.css|presentation\.html|tasks/)"
)
MECHANICAL_ARTICLE_PHRASES = (
    "targets its main failure mode",
    "fixes the displayed estimand",
    "maps symbols to displayed columns",
    "reads projections and distances",
    "keeps results on native scales",
    "aligns quantities before comparison",
    "tests its identifying condition",
    "rebuilds its table from retained rows",
    "records software defaults explicitly",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def prose_word_count(body: str) -> int:
    body = re.sub(r"\$\$.*?\$\$", " ", body, flags=re.S)
    kept: list[str] = []
    for line in body.splitlines():
        if line.startswith(("#", "|", "{{")):
            continue
        kept.append(line)
    return len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", "\n".join(kept)))


def analyzed_task_ids() -> list[str]:
    return list(json.loads(read(PROJECT_ROOT / "data/analysis/main/data/catalog.json")))


def test_every_analyzed_task_has_a_page() -> None:
    missing = [tid for tid in analyzed_task_ids() if not (TASKS_DIR / tid / "index.html").is_file()]
    assert not missing, f"task pages missing: {missing}"


def test_explorer_row_links_to_the_task_page() -> None:
    app = read(SITE_DIR / "app.js")
    # The link sits in the open row body, not inside <summary> (no nested controls).
    assert 'class="task-page-link"' in app
    assert 'href="tasks/${encodeURIComponent(item.id)}/"' in app
    summary = app[app.index("<summary>"):app.index("</summary>")]
    assert "<a " not in summary
    explore = read(SITE_DIR / "explore.html")
    assert 'href="tasks/"' in explore
    assert 'href="tasks.css"' in explore


def test_catalog_lists_every_generated_page() -> None:
    manifest = json.loads(read(TASKS_DIR / "_index.json"))
    catalog = read(TASKS_DIR / "index.html")
    for entry in manifest:
        assert f'href="{entry["slug"]}/"' in catalog, entry["slug"]
        assert (TASKS_DIR / entry["slug"] / "index.html").is_file(), entry["slug"]
    statuses = {entry["status"] for entry in manifest}
    assert statuses == {"production"}
    assert len(manifest) == 40


def test_pages_carry_no_task_number_and_no_repository_path() -> None:
    offenders = []
    for page in TASKS_DIR.glob("**/index.html"):
        text = read(page)
        if TASK_NUMBER_RE.search(text) or REPO_PATH_RE.search(text):
            offenders.append(str(page.relative_to(TASKS_DIR)))
    assert not offenders, offenders[:10]


def test_analyzed_page_binds_the_explorer_permalink() -> None:
    for tid in analyzed_task_ids():
        text = read(TASKS_DIR / tid / "index.html")
        assert f'href="../../explore.html#task={tid}"' in text, tid
        assert "About this benchmark task" in text, tid


def test_every_task_page_uses_one_template() -> None:
    """All 40 pages show the task, the data, the reference output, and task success the same way."""
    for tid in analyzed_task_ids():
        text = read(TASKS_DIR / tid / "index.html")
        for marker in ('id="the-task"', "Task statement", "<h3>Data</h3>", "Required output",
                       "Reference output", 'id="results"', "Agency condition", "Claude Sonnet 4.6",
                       "GPT-5.6 Luna", "About this benchmark task"):
            assert marker in text, (tid, marker)
        for stale in ("Source status", "40-task sample", "contract", "Few-shot", "few-shot", "Identifier"):
            assert stale not in text, (tid, stale)


def test_scored_standard_errors_appear_on_their_task_pages() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "paper"))
    import se_scoring

    for tid, terms in se_scoring.spec().items():
        text = read(TASKS_DIR / tid / "index.html")
        assert "Reference standard error" in text, tid
        assert "are also scored" in text, tid


def test_analyzed_pages_are_substantive_math_articles() -> None:
    for tid in analyzed_task_ids():
        if not (SITE_DIR / "task_articles" / tid / "article.md").exists():
            continue
        text = read(TASKS_DIR / tid / "index.html")
        assert 'class="task-equation"' in text, tid
        assert "mathjax@4.1.3/tex-chtml.js" in text, tid
        assert "font: 'mathjax-newcm'" in text, tid
        assert 'class="article-section' in text, tid
        assert 'class="article-references"' in text, tid
        assert 'class="task-about"' in text, tid
        assert "Sources and implementation notes" not in text, tid


def test_every_article_page_uses_new_computer_modern_math() -> None:
    manifest = json.loads(read(TASKS_DIR / "_index.json"))
    for entry in manifest:
        text = read(TASKS_DIR / entry["slug"] / "index.html")
        assert "mathjax@4.1.3/tex-chtml.js" in text, entry["slug"]
        assert "font: 'mathjax-newcm'" in text, entry["slug"]
        assert "tex-chtml-nofont" not in text, entry["slug"]


def test_article_sources_preserve_catalog_slugs_and_end_with_task_details() -> None:
    manifest = json.loads(read(TASKS_DIR / "_index.json"))
    source_slugs = {
        path.parent.name
        for path in (SITE_DIR / "task_articles").glob("*/article.md")
    }
    published_slugs = {entry["slug"] for entry in manifest}
    assert published_slugs == set(analyzed_task_ids())
    for slug in published_slugs:
        text = read(TASKS_DIR / slug / "index.html")
        main = text.split('<main class="task-page">', 1)[1].split("</main>", 1)[0]
        assert re.search(r'<section class="task-about"[^>]*>.*</section>\s*$', main, re.S), slug


def test_article_bibliography_covers_every_source_citation() -> None:
    references = build_task_pages.parse_bibliography(SITE_DIR / "references.bib")
    cited = set()
    for path in (SITE_DIR / "task_articles").glob("*/article.md"):
        cited.update(build_task_pages.CITATION_RE.findall(read(path)))
    assert cited <= set(references)


def test_every_article_source_has_educational_depth() -> None:
    paths = sorted((SITE_DIR / "task_articles").glob("*/article.md"))
    assert len(paths) == 298
    for path in paths:
        source = read(path)
        body = source.split("+++", 2)[2]
        assert prose_word_count(body) >= 300, path.parent.name
        assert len(re.findall(r"^## ", body, re.M)) >= 5, path.parent.name
        assert "[@" in body, path.parent.name
        assert "$$" in body, path.parent.name
        assert re.search(r"^\|.+\|$|^\{\{figure:", body, re.M), path.parent.name
        assert "\\(" not in source and "\\)" not in source, path.parent.name
        assert "<!--" not in source and "-->" not in source, path.parent.name
        assert not re.search(r"<(?:/?[A-Za-z]|!--|!DOCTYPE)", body), path.parent.name
        assert "\u200b" not in source and "\u2060" not in source, path.parent.name
        assert not re.search(r"(?<![\*^])\*[A-Za-z][^*\n]+\*(?!\*)", source), path.parent.name


def test_statllm_articles_use_substantive_method_prose() -> None:
    paths = sorted((SITE_DIR / "task_articles").glob("statllm-*/article.md"))
    assert len(paths) == 149
    for path in paths:
        source = read(path)
        front_matter = source.split("+++", 2)[1].lower()
        body = source.split("+++", 2)[2]
        lowered = body.lower()
        assert "explained through" not in front_matter, path.parent.name
        assert "formal notation" not in front_matter, path.parent.name
        assert "observed values" not in front_matter, path.parent.name
        assert "implementation risks" not in front_matter, path.parent.name
        assert not any(phrase in lowered for phrase in MECHANICAL_ARTICLE_PHRASES), path.parent.name
        assert "$$\\log(${" not in source, path.parent.name
        assert "\\(" not in source and "\\)" not in source, path.parent.name
        assert "<!--" not in source and "-->" not in source, path.parent.name
        assert "\u200b" not in source and "\u2060" not in source, path.parent.name
        assert len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", body)) >= 250, path.parent.name
        assert len(re.findall(r"^## ", body, re.M)) >= 5, path.parent.name
        assert "[@" in body, path.parent.name
        assert "$$" in body, path.parent.name
        assert re.search(r"^\|.+\|$", body, re.M), path.parent.name


def test_articles_do_not_share_generated_prose() -> None:
    paragraph_owners: dict[tuple[str, ...], set[str]] = {}
    sequence_owners: dict[tuple[str, ...], set[str]] = {}
    for path in sorted((SITE_DIR / "task_articles").glob("*/article.md")):
        body = read(path).split("+++", 2)[2].split("## Further reading", 1)[0]
        for paragraph in re.split(r"\n\s*\n", body):
            paragraph = paragraph.strip()
            if not paragraph or paragraph.startswith(("#", "|", "$$", "{{")):
                continue
            paragraph = re.sub(r"(?m)^(?:- |\d+\. )", "", paragraph)
            words = re.findall(r"[a-z0-9]+", paragraph.lower())
            if len(words) >= 30:
                paragraph_owners.setdefault(tuple(words), set()).add(path.parent.name)
            for index in range(len(words) - 19):
                gram = tuple(words[index : index + 20])
                sequence_owners.setdefault(gram, set()).add(path.parent.name)
    repeated_paragraphs = {
        " ".join(words): sorted(slugs)
        for words, slugs in paragraph_owners.items()
        if len(slugs) > 1
    }
    repeated_templates = {
        " ".join(gram): sorted(slugs)
        for gram, slugs in sequence_owners.items()
        if len(slugs) > 2
    }
    assert not repeated_paragraphs, list(repeated_paragraphs.items())[:10]
    assert not repeated_templates, list(repeated_templates.items())[:10]


def test_approved_article_front_matter_and_figure_rendering(tmp_path: Path) -> None:
    article_dir = tmp_path / "candidate-example"
    article_dir.mkdir()
    (article_dir / "figure.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<path d="M0 10 L10 0"/></svg>',
        encoding="utf-8",
    )
    source = article_dir / "article.md"
    source.write_text(
        """+++
title = "An Educational Article"
summary = "A compact summary for broad readers."
topic = "Linear models"
angle = "Geometry and sampling variation"
related = []
source_warnings = []

[[figures]]
id = "design-map"
src = "figure.svg"
alt = "A diagonal line in a square coordinate system."
caption = "An illustrative design map [@wooldridge2020]."
+++

Opening context can appear before the first section [@wooldridge2020].

## The model

The mean is $E[y|x]=x'\\beta$.

{{figure:design-map}}
""",
        encoding="utf-8",
    )
    expected = {"slug": "candidate-example", "status": "candidate"}
    article = build_task_pages.parse_article_source(source, expected)
    references = build_task_pages.parse_bibliography(SITE_DIR / "references.bib")
    rendered, cited = build_task_pages.render_article_markdown(article, references)
    assert 'class="article-section article-preamble"' in rendered
    assert 'class="article-figure"' in rendered
    assert 'src="assets/design-map.svg"' in rendered
    assert cited == ["wooldridge2020"]


def test_article_source_validation_fails_closed(tmp_path: Path) -> None:
    article_dir = tmp_path / "candidate-example"
    article_dir.mkdir()
    source = article_dir / "article.md"
    source.write_text(
        """+++
title = "An Educational Article"
summary = "A compact summary for broad readers."
topic = "Linear models"
angle = "Geometry and sampling variation"
related = []
source_warnings = []
+++

## The model

An unknown source appears here [@missing-key].
""",
        encoding="utf-8",
    )
    expected = {"slug": "candidate-example", "status": "candidate"}
    article = build_task_pages.parse_article_source(source, expected)
    references = build_task_pages.parse_bibliography(SITE_DIR / "references.bib")
    with pytest.raises(build_task_pages.ArticleSourceError, match="missing bibliography key"):
        build_task_pages.render_article_markdown(article, references)

    source.write_text(read(source).replace('angle = "Geometry and sampling variation"\n', ""), encoding="utf-8")
    with pytest.raises(build_task_pages.ArticleSourceError, match="missing front matter"):
        build_task_pages.parse_article_source(source, expected)


def test_task_articles_match_the_topbar_width() -> None:
    css = read(SITE_DIR / "tasks.css")
    assert ".task-page {\n  max-width: 60rem;" in css
    assert ".task-footer {\n  max-width: 60rem;" in css
    assert ".article-table-wrap" in css
    assert ".article-figure" in css
    assert ".reference-list" in css


def test_article_prose_uses_page_width_and_equations_have_no_rules() -> None:
    css = read(SITE_DIR / "tasks.css")
    assert "max-width: 42rem;" not in css
    prose_rule = css.split(".article-section > p,", 1)[1].split("}", 1)[0]
    assert "max-width: none;" in prose_rule
    assert ".task-page .article-section > p," not in css

    equation_rule = css.split(".task-equation {", 1)[1].split("}", 1)[0]
    assert "border-top" not in equation_rule
    assert "border-bottom" not in equation_rule


def test_public_bundle_versions_task_page_assets(tmp_path, monkeypatch) -> None:
    builder = read(SITE_DIR / "build_public_site.py")
    assert '"tasks"' in builder
    assert '"tasks.css"' in builder

    monkeypatch.setattr(build_public_site, "COPY_DIRS", ("tasks",))
    monkeypatch.setattr(build_public_site, "DOWNLOADS", {})
    out_dir = tmp_path / "public"
    build_public_site.build(out_dir, "20991231-1")

    task_page = read(out_dir / "tasks" / "ols_robust" / "index.html")
    assert 'href="../../tasks.css?v=20991231-1"' in task_page
    assert "mathjax@4.1.3/tex-chtml.js?v=" not in task_page


def test_build_needs_only_paper_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_nonpaper_input():
        raise AssertionError("Paper catalog must not read nonpaper inputs")

    for name in ("load_hard_block", "load_candidates", "load_statllm", "load_probe"):
        monkeypatch.setattr(build_task_pages, name, reject_nonpaper_input)
    monkeypatch.setattr(build_task_pages, "OUT_DIR", tmp_path / "tasks")
    entries = build_task_pages.build()
    assert len(entries) == 40
    assert {entry["status"] for entry in entries} == {"production"}
