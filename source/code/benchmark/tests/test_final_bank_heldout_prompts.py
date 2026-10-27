"""The held-out few-shot level (type 2 examples) is not part of the final bank.

Decision record 2026-09-30_drop-heldout-level: the level heldout_few_shot left the bank. The example set stays as an
archive in data/task_candidates/2026-09-29_heldout_examples/ (checked in test_heldout_examples.py), and the renderer
can still build its block from the archive. Rendering only: no provider call and no example run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import final_bank_prompts as fp  # noqa: E402

FOLDERS = fp.all_package_dirs()
SET = json.loads((fp.HELDOUT_DIR / 'set.json').read_text())


def test_prompt_levels():
    assert fp.PROMPTS == ('zero_shot', 'few_shot')


@pytest.mark.parametrize('folder', FOLDERS, ids=[f.name for f in FOLDERS])
def test_no_package_renders_or_stores_a_heldout_prompt(folder):
    assert not any('heldout' in name for name in fp.render_package(folder))
    assert not list((folder / 'prompts').glob('*__heldout_few_shot__*.txt'))
    with pytest.raises(ValueError):
        fp.prompt_levels({'task_id': folder.name, 'prompt_levels': ['zero_shot', 'heldout_few_shot']})


def test_the_archive_stays_and_its_block_still_builds():
    assert fp.HELDOUT_DIR.is_dir()
    ids = fp.heldout_ids('ols_robust')
    assert ids == SET['examples'] and len(ids) == 3
    block = fp.heldout_block('ols_robust', 'python')
    positions = [block.index(f'Example {k}: ') for k in (1, 2, 3)]
    assert positions == sorted(positions)
    assert block.startswith('Examples: three short scripts')
