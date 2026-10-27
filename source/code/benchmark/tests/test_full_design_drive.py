"""The full-design driver reads reset times from usage-limit messages.

Decision record: quality_reports/decision_records/2026-10-17_haiku55-sol-full-design.md.
"""
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location('full_design_drive',
                                               ROOT / 'code/benchmark/workflows/full_design_2026_10_17/drive.py')
drive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drive)
AT = datetime(2026, 10, 17, 18, 0, tzinfo=timezone.utc)


def test_relative_reset():
    text = "You've hit your usage limit. Try again in 2 hours 5 minutes."
    assert drive.reset_from_text(text, AT) == AT + timedelta(hours=2, minutes=5)


def test_clock_reset_is_in_the_future():
    text = "You've hit your usage limit. Upgrade to Pro or try again at 1:33 PM."
    reset = drive.reset_from_text(text, AT)
    assert reset > AT and reset - AT <= timedelta(days=1)
    assert (reset.astimezone().hour, reset.astimezone().minute) == (13, 33)


def test_message_without_time():
    assert drive.reset_from_text("You've hit your usage limit.", AT) is None


def test_etime_seconds():
    assert drive.etime_seconds('07:25:38') == 7 * 3600 + 25 * 60 + 38
    assert drive.etime_seconds('1-02:03:04') == 86400 + 2 * 3600 + 3 * 60 + 4
    assert drive.etime_seconds('15:01') == 901


def test_runaway_scripts_selects_only_long_task_scripts_of_this_collector():
    collector = '.venv/bin/python -u code/benchmark/prospective_collect_sol.py collect --manifest m.json'
    rows = [dict(pid=10, ppid=1, elapsed=9000, command=collector),
            dict(pid=11, ppid=10, elapsed=901,
                 command='/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se -b do /private/tmp/prospective_task_ab12/run.do'),
            dict(pid=12, ppid=10, elapsed=899, command='Rscript --vanilla /private/tmp/prospective_task_cd34/run.R'),
            dict(pid=13, ppid=10, elapsed=5000, command='codex exec --json -m gpt-6-sol'),
            dict(pid=14, ppid=99, elapsed=5000, command='python /private/tmp/prospective_task_ef56/run.py'),
            dict(pid=15, ppid=10, elapsed=1200, command='/x/.venv/bin/python /private/tmp/prospective_task_gh78/run.py')]
    found = drive.runaway_scripts(rows, 'prospective_collect_sol.py')
    assert [r['pid'] for r in found] == [11, 15]
    assert found[0]['cwd'] == '/private/tmp/prospective_task_ab12'
