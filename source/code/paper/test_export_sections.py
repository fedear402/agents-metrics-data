import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from export_sections import export, flatten, headings, markdown


def test_heading_boundaries_ignore_comments_and_prompt_text():
    text = "% \\section{Hidden}\n\\section{Parent}\nA\n\\subsection{Child {one}}\nB\n\\subsubsection{Nested}\nC\n\\subsection{Next}\nD\n\\begin{promptbox}\n\\section{Literal}\n\\end{promptbox}"
    found = headings(text)
    assert [h["title"] for h in found] == ["Parent", "Child {one}", "Nested", "Next"]
    assert found[0]["end"] == len(text)
    assert found[1]["end"] == found[3]["start"]
    assert found[2]["end"] == found[3]["start"]


def test_include_expansion_preserves_literals_and_detects_cycles(tmp_path):
    (tmp_path / "included.tex").write_text("Included words.")
    dependencies = set()
    raw = "Before. \\input{included} After. \\begin{promptbox}\\input{literal}\\end{promptbox}"
    assert "Included words." in flatten(raw, tmp_path, dependencies)
    assert "\\input{literal}" in flatten(raw, tmp_path, dependencies)
    (tmp_path / "included.tex").write_text(r"\input{included}")
    with pytest.raises(ValueError, match="cycle"):
        flatten(raw, tmp_path, dependencies)


def test_markdown_preserves_content_and_expands_values(tmp_path):
    (tmp_path / "numbers.tex").write_text(r"\newcommand{\numTest}{0.0560}")
    raw = r"""\section{Example}
Body \costcents[2]{\numTest{}} and \raminutes{0.537}{50}.
\footnote{Footnote text.}
\begin{equation}x=2\end{equation}
\begin{tabular}{lr}Name & Value \\ Alpha & 7 \\ \end{tabular}
\begin{promptbox}
Literal % sign and \numTest{}.
\end{promptbox}
"""
    result, _ = markdown(raw, tmp_path, set(), {}, {})
    for value in ("Example", "5.60", "0.6", "Footnote text.", "Alpha", "7", r"Literal % sign and \numTest{}."):
        assert value in result
    assert "x=2" in "".join(result.split())


def test_plain_export_adds_no_markdown_markup(tmp_path):
    (tmp_path / "numbers.tex").write_text("")
    text = r"\section{Heading} Plain \emph{italic} and \textbf{bold}. \footnote{A note.} \begin{itemize}\item Item text.\end{itemize}"
    result, _ = markdown(text, tmp_path, set(), {}, {})
    assert result.startswith("Heading\n")
    assert "Plain italic and bold." in result
    assert "Footnote: A note." in result
    assert "Item text." in result
    for marker in ("#", "*", "`", "[^", "{#", "- Item"):
        assert marker not in result


def test_whole_paper_order_content_and_regeneration(tmp_path):
    source = tmp_path / "manuscript"
    for directory in ("sections", "appendix", "build"):
        (source / directory).mkdir(parents=True)
    for name in ("preamble.tex", "numbers.tex", "numbers_pending.tex", "refs.bib", "AEA.cls", "aea.bst", "build/main_body.aux", "build/supplemental_appendix.aux"):
        (source / name).write_text("")
    (source / "main.tex").write_text(r"""\input{sections/00_frontmatter}
\subfile{sections/01_introduction}
% \subfile{sections/ignored}
\subfile{sections/02_results}
\bibliography{refs}
\appendix
\subfile{appendix/A_details}
""")
    (source / "sections/00_frontmatter.tex").write_text(r"\title{Paper title}\author{An Author}\begin{abstract}Abstract marker.\end{abstract}")
    (source / "sections/01_introduction.tex").write_text("Introduction marker.")
    (source / "sections/02_results.tex").write_text(r"\section{Results}Result marker.\subsection{Details}Child \emph{marker}.\footnote{Footnote marker.}")
    (source / "appendix/A_details.tex").write_text(r"\section{Supplement}Appendix marker.")
    (source / "build/main_body.bbl").write_text(r"\begin{thebibliography}{1}Reference marker.\end{thebibliography}")
    out = source / "by_section"
    export(source, out)
    combined = (out / "whole_paper.md").read_text()
    expected_order = ["Paper title", "An Author", "Abstract marker.", "Introduction marker.", "Result marker.", "Child marker.", "Footnote marker.", "Reference marker.", "Appendix marker."]
    positions = [combined.index(value) for value in expected_order]
    assert positions == sorted(positions)
    assert all(combined.count(value) == 1 for value in expected_order)
    assert "*" not in combined
    metadata = json.loads((out / "manifest.json").read_text())["whole_paper"]
    assert metadata["sections"] == ["frontmatter", "introduction", "results", "references", "appendix_a_supplement"]
    assert metadata["md_sha256"] == hashlib.sha256(combined.encode()).hexdigest()
    export(source, out)
    assert (out / "whole_paper.md").read_text() == combined
    (out / "personal.md").write_text("Keep this manual file.")
    with pytest.raises(ValueError, match="unmanaged"):
        export(source, out)
    assert (out / "personal.md").read_text() == "Keep this manual file."
    assert (out / "whole_paper.md").read_text() == combined


@pytest.mark.parametrize("uv_status,build_status,expected", [(0, 0, 0), (9, 0, 1), (0, 4, 4)])
def test_compilation_hook_exit_status(tmp_path, uv_status, build_status, expected):
    config = Path(__file__).with_name("section_exports_latexmk.pl").read_text()
    hook = config[config.index("END {"):]
    (tmp_path / "main.tex").write_text("")
    uv = tmp_path / "uv"
    uv.write_text(f"#!/bin/sh\nexit {uv_status}\n")
    uv.chmod(0o755)
    result = subprocess.run(["/usr/bin/perl", "-e", "$cleanup_mode=0;\n" + hook + f"\nexit {build_status};"], cwd=tmp_path, env={**os.environ, "PATH": str(tmp_path)}, capture_output=True)
    assert result.returncode == expected


def test_actual_exports_are_exact_source_slices():
    source = Path(__file__).resolve().parents[2] / "paper/manuscript"
    out = source / "by_section"
    if not out.exists():
        pytest.skip("Compile the local manuscript to create exports")
    manifest = json.loads((out / "manifest.json").read_text())
    names = {unit["name"] for unit in manifest["sections"]}
    assert {"abstract", "introduction", "design__tasks_and_scoring", "appendix_d_robustness", "references"} <= names
    for unit in manifest["sections"]:
        original = (source / unit["source"]).read_bytes().decode()
        actual = (out / (unit["name"] + ".tex")).read_bytes()
        assert actual == original[unit["start"]:unit["end"]].encode()
        assert hashlib.sha256(actual).hexdigest() == unit["tex_sha256"]
    for name, checksum in manifest["source_hashes"].items():
        assert hashlib.sha256((out / "_sources" / name).read_bytes()).hexdigest() == checksum
