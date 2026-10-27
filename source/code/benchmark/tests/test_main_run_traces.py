"""Main selection trace evidence must retain the saved response and verdict."""
import importlib.util
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[2] / "falstuff_site"
sys.path.insert(0, str(SITE))
import build_run_traces as traces


def test_main_response_is_exact_and_primary_is_authoritative(tmp_path):
    import json
    answer = 'Stata 14 solution\n```stata\ndisplay 1\n```\n\n'
    (tmp_path / 'prompt_attempt1.txt').write_text('Exact prompt\n')
    (tmp_path / 'run_record.json').write_text(json.dumps({'calls': [{'text': answer}], 'executions': [], 'scoring': {'success': False}}))
    row = {'record_path': str(tmp_path / 'run_record.json'), 'output_path': '', 'platform': 'stata', 'interface': 'no_feedback', 'prompt': 'zero_shot', 'position_id': 'example', 'primary': 'True', 'failure_category': 'Success'}
    combo = traces.build_main_combination(row, 1)
    assert combo['pass'] is True
    assert combo['flow']['prompt']['text'] == 'Exact prompt\n'
    assert combo['flow']['final']['text'] == answer
    assert combo['flow']['final']['script'] == ''


def test_main_trace_sample_dimensions():
    assert set(traces.MODEL_CONFIG) == {'sonnet', 'luna'}
    assert traces.MAX_REPS == 10
    assert traces.EXPECTED_COMBINATIONS == 180


def test_saved_repair_request_and_response_order(tmp_path):
    import json
    (tmp_path / 'prompt_attempt1.txt').write_text('Initial\n')
    (tmp_path / 'prompt_attempt2.txt').write_text('Repair exact\n')
    (tmp_path / 'run_record.json').write_text(json.dumps({'calls': [{'text': 'first'}, {'text': 'last\n'}], 'executions': [{'stderr': 'error'}, {'stdout': 'ok'}]}))
    row = {'record_path': str(tmp_path / 'run_record.json'), 'platform': 'python', 'interface': 'self_repair', 'prompt': 'zero_shot', 'position_id': 'repair', 'primary': 'False'}
    combo = traces.build_main_combination(row, 1)
    events = combo['flow']['process']['events']
    assert [event['label'] for event in events] == ['Response 1', 'Execution 1', 'Request 2']
    assert events[-1]['text'] == 'Repair exact\n'
    assert combo['flow']['final']['text'] == 'last\n'
    assert combo['pass'] is False


def test_numbered_stata_license_output_is_removed():
    text = '    13\tStata license: Unlimited-user network\n    14\tSerial number: EXAMPLE\n    15\t  Licensed to: ACCOUNT\n    16\t               ORGANIZATION\n    17\t\n    18\t. display 1\n    19\t1\n'
    result = traces.redact(text)
    assert 'Serial number' not in result
    assert 'ORGANIZATION' not in result
    assert '. display 1' in result


def test_saved_stream_actions_use_source_order(tmp_path):
    import json
    path = tmp_path / 'provider_attempt1.stdout'
    path.write_text('\n'.join(json.dumps(event) for event in [
        {'type': 'item.completed', 'item': {'id': 'a', 'type': 'agent_message', 'text': 'Checking'}},
        {'type': 'item.completed', 'item': {'id': 'b', 'type': 'command_execution', 'command': 'Rscript run.R', 'aggregated_output': 'done', 'exit_code': 0}},
        {'type': 'item.completed', 'item': {'id': 'c', 'type': 'agent_message', 'text': 'DONE'}},
    ]))
    events = traces.stream_evidence(tmp_path, 'DONE', path)
    assert [e['label'] for e in events] == ['Model message', 'Command', 'Command result']
    assert events[0]['text'] == 'Checking'


def test_generated_traces_match_every_selected_primary_outcome():
    import json
    from main_sample import load_main_sample, truth
    catalog, rows, _ = load_main_sample()
    expected = {row['position_id']: truth(row['primary']) for row in rows}
    observed = {}
    for model in traces.MODEL_CONFIG:
        paths = list((traces.TRACE_DIR / model).glob('*.js'))
        assert {path.stem for path in paths} == set(catalog)
        for path in paths:
            payload = json.loads(path.read_text().split('\n', 2)[2].split(' = ', 1)[1].rstrip(';\n'))
            assert len(payload['combos']) == 180
            for combo in payload['combos'].values():
                assert combo['positionId'] not in observed
                observed[combo['positionId']] = combo['pass']
                # Constrained-agent traces carry the final script we executed; the others
                # carry the script inside the saved final response.
                if combo['agency'] == 'tool_loop':
                    assert combo['flow']['final']['script']
                else:
                    assert combo['flow']['final']['script'] == ''
    assert observed == expected
