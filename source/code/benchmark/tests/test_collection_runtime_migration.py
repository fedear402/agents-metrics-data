"""Offline checks against the report runtime captured before migration."""
import ast
import hashlib
import itertools
import json
from pathlib import Path

from benchmark import prospective_collect as p

CODE = Path(__file__).resolve().parents[1]
PACKAGE = CODE / 'collection_runtime'
BASELINE = json.loads((Path(__file__).parent / 'fixtures/collection_runtime_baseline.json').read_text())


def test_runtime_sources_are_canonical_and_complete(tmp_path, monkeypatch):
    original = Path.read_text

    def reject_report_reads(path, *args, **kwargs):
        assert 'quality_reports' not in path.parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', reject_report_reads)
    runtime = p.load_runtime(tmp_path)
    sources = p.runtime_sources(runtime)
    assert not any('quality_reports' in path.parts for path in sources)
    assert set(PACKAGE.glob('*.py')) <= set(sources)
    assert runtime.c.POLICY == BASELINE['policy']
    assert runtime.c.WORK == runtime.m.WORK == tmp_path


def test_command_arguments_and_timeouts_match_baseline(tmp_path):
    runtime = p.load_runtime(tmp_path)
    for model, condition in itertools.product(['luna', 'sonnet'], ['chatbot', 'self_repair', 'agent']):
        argv, timeout = runtime.c.command(dict(model=model, condition=condition), Path('<cwd>'), Path('<last>'))
        assert [argv[1:], timeout] == BASELINE['commands'][model+'/'+condition]


def test_prompt_bytes_match_baseline(tmp_path):
    runtime = p.load_runtime(tmp_path)
    for platform, prompt, condition in itertools.product(['python', 'r', 'stata'], ['zero_shot', 'few_shot'], ['chatbot', 'self_repair', 'agent']):
        text = runtime.c.prior.prompt(BASELINE['spec'], dict(platform=platform, prompt=prompt, condition=condition),
                                     previous='old script' if condition == 'self_repair' else None,
                                     feedback='test feedback', bundle=BASELINE['bundle'])
        assert hashlib.sha256(text.encode()).hexdigest() == BASELINE['prompts']['/'.join([platform, prompt, condition])]


def test_loaded_trees_have_independent_mutable_state(tmp_path):
    first = p.load_runtime(tmp_path / 'first')
    second = p.load_runtime(tmp_path / 'second')
    assert first.c is not second.c
    assert first.c.prior is not second.c.prior
    assert first.c.prior.base is not second.c.prior.base
    assert first.c.WORK == tmp_path / 'first'
    assert second.c.WORK == tmp_path / 'second'
    first.c.POLICY['test_mutation'] = True
    assert 'test_mutation' not in second.c.POLICY


def test_migrated_function_bodies_match_report_sources():
    for module, expected in BASELINE['functions'].items():
        path = (CODE if module in {'production_statistics', 'saved_response_analysis'} else PACKAGE) / (module+'.py')
        actual = {node.name: hashlib.sha256(ast.dump(node).encode()).hexdigest()
                  for node in ast.parse(path.read_text()).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert actual == expected, module


def test_statistics_helpers_load_from_canonical_code():
    import production_paper_assets as assets
    helper = assets.load_module('canonical_statistics_test', assets.STATISTICS)
    assert helper.ROOT == p.ROOT
    assert len(helper.weights()) >= 10
    assert 'quality_reports' not in Path(helper.__file__).parts
    provenance = json.loads(assets.RUNTIME_PROVENANCE.read_text())
    assert len(provenance['historical_figures']['sha256']) == 64
