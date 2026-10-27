"""Keep native Stata ranges distinct from external file paths."""
import importlib.util
from pathlib import Path
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'prospective_collect.py'
spec = importlib.util.spec_from_file_location('prospective_access_tested', SOURCE)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


@pytest.mark.parametrize('text', [
    "gen ybar_dm = ybar_n - `mean_y'/`sum_sn'",
    "scalar ratio = `numerator'/`denominator'",
    "local ratio = (`a'+`b')/`c'",
    "gen ratio = x/`denominator'",
    "gen ratio = `numerator'/x",
])
def test_stata_macro_arithmetic_is_not_a_path(tmp_path, text):
    runtime = p.load_runtime(tmp_path)
    assert runtime.c.access_findings(text, tmp_path) == []


@pytest.mark.parametrize('text', [
    'use "/Users/another/`dataset\'.dta", clear',
    'use "/`directory\'/input.dta", clear',
    'local base "/Users/another"\nuse "`base\'/input.dta", clear',
    "use `prefix'/Users/another/private.txt",
    "gen ratio = `a'/`b'\ncat /Users/another/private.txt",
])
def test_macro_normalization_keeps_external_path_findings(tmp_path, text):
    runtime = p.load_runtime(tmp_path)
    assert runtime.c.access_findings(text, tmp_path)


def test_stata_local_macro_range_is_not_a_path(tmp_path):
    runtime = p.load_runtime(tmp_path)
    text = "forvalues f = `first_forecast'/`last_forecast' {\n display `f'\n}"
    assert runtime.c.access_findings(text, tmp_path) == []


def test_range_does_not_hide_external_read(tmp_path):
    runtime = p.load_runtime(tmp_path)
    text = "forvalues f = `first_forecast'/`last_forecast' {\n}\ncat /Users/another/private.txt"
    findings = runtime.c.access_findings(text, tmp_path)
    assert any('/Users/another/private.txt' in str(value) for value in findings)


def test_macro_prefixed_real_path_remains_detected(tmp_path):
    runtime = p.load_runtime(tmp_path)
    text = 'cat "/`last_forecast\'/private.txt"'
    assert runtime.c.access_findings(text, tmp_path)


def test_installed_system_executable_in_command_position(tmp_path):
    runtime = p.load_runtime(tmp_path)
    text = '/bin/zsh -lc "ls -la && /usr/bin/file input.csv && sed -n \'1,4p\' input.csv"'
    call = {'observable_runtime': {'all_tool_items': [{'type': 'command_execution', 'command': text}]}}
    assert runtime.c.tool_violations({'model': 'luna', 'condition': 'agent'}, [call], tmp_path) == []
    assert call['observable_runtime']['all_tool_items'][0]['command'] == text


def test_system_command_keeps_external_argument(tmp_path):
    runtime = p.load_runtime(tmp_path)
    findings = runtime.c.access_findings(p.scan_shell_command('/usr/bin/file /Users/another/private.txt'), tmp_path)
    assert any('/Users/another/private.txt' in value for value in findings)


def test_system_path_as_argument_is_not_normalized(tmp_path):
    assert p.access_scan_text('cat /usr/bin/file') == 'cat /usr/bin/file'


def test_unknown_system_command_is_not_normalized(tmp_path):
    text = '/usr/bin/benchmark_nonexistent_command_728 input.csv'
    assert p.access_scan_text(text) == text


def test_source_string_and_read_path_keep_findings(tmp_path):
    runtime = p.load_runtime(tmp_path)
    for text in ['/usr/bin/file', "p = '''\n/usr/bin/file\n'''.strip(); open(p).read()"]:
        assert runtime.c.access_findings(text, tmp_path)
    call = {'observable_runtime': {'all_tool_items': [{'name': 'Read', 'input': {'file_path': '/usr/bin/file'}}]}}
    assert p.calls_for_access_review([call]) == [call]
