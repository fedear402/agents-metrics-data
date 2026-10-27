"""Numerical macros must count where used, including repeated uses."""
from pathlib import Path
import runpy
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / 'paper/manuscript/citation_wordcount.pl'


def expand_citations(tmp_path, text):
    """Use bibliography display labels, including a multiword corporate author."""
    bbl = tmp_path / 'labels.bbl'
    bbl.write_text(r'''
\bibitem[\protect\citeauthoryear{Alpha and Beta}{Alpha and Beta}{2025}]{ab}
\bibitem[\protect\citeauthoryear{Yang, One, and Two}{Yang et~al.}{2023}]{yang1}
\bibitem[\protect\citeauthoryear{Yang, Three, and Four}{Yang et~al.}{2024}]{yang2}
\bibitem[\protect\citeauthoryear{{Menlo Ventures}}{{Menlo Ventures}}{2025}]{menlo}
''')
    source = tmp_path / 'source.tex'
    source.write_text(text)
    return subprocess.check_output([
        'perl', '-e',
        'require $ARGV[0]; print wc_expand_citations(wc_read($ARGV[2]), wc_labels(wc_read($ARGV[1])));',
        str(HELPER), str(bbl), str(source)], text=True)


@pytest.mark.parametrize(('source', 'words'), [
    (r'Hello \citet{ab} and \citep{yang1,yang2}.', 11),
    (r'\citep{ab} \citep{ab}', 8),
    (r'\citep{menlo}', 3),
    (r'\citep[see][p. 12]{ab}', 7),
    (r'%TC:envir figure 0 0' + '\n' + r'\citep{ab} \begin{figure}\citep{yang1}\end{figure}', 4),
])
def test_citations_count_displayed_words(tmp_path, source, words):
    if not shutil.which('texcount'):
        pytest.skip('TeXcount is required for the manuscript count')
    expanded = tmp_path / 'expanded.tex'
    expanded.write_text(expand_citations(tmp_path, source))
    result = subprocess.check_output(['texcount', '-sum', '-1', str(expanded)], text=True)
    assert int(result.strip()) == words


def test_unknown_citation_fails_instead_of_silent_undercount(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        expand_citations(tmp_path, r'\citep{unknown}')


def test_texcount_counts_each_rendered_number(tmp_path):
    if not shutil.which('texcount'):
        pytest.skip('TeXcount is required for the manuscript count')
    renderer = runpy.run_path(str(ROOT / 'code/paper/manuscript.py'))
    definitions = renderer['number_definition']('numExample', '98')
    definitions += renderer['number_definition']('numUnused', '40')
    source = tmp_path / 'count.tex'
    source.write_text(definitions + '\\numExample{} \\numExample{}\n')
    result = subprocess.check_output(['texcount', '-sum', '-1', str(source)], text=True)
    assert int(result.strip()) == 2
