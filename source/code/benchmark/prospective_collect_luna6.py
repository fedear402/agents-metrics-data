"""Collect the paper experiment with GPT-6 Luna through codex exec, as the paper's Luna runs.

Decision record: quality_reports/decision_records/2026-10-19_gpt6-luna-full-design.md.
This entry point is the GPT-6 Sol collector (prospective_collect_sol.py: unlimited additive-R entry point, pinned
Codex 0.163.0, paper-time Stata ANSWER reader) with the model gpt-6-luna at effort low. The job label stays "luna".
The manifest pins this file, the Sol collector, the binary, and the test.
"""
from pathlib import Path
import tempfile

import prospective_collect_sol as sol

central = sol.central
sol.MODEL = ('gpt-6-luna', 'low')  # read by sol.configure in every fresh runtime
VERSION = 'prospective_gpt6_luna_codex_2026_10_19_v1'


def prepare(config_path, manifest_path):
    if central.sha(sol.BINARY) != sol.BINARY_SHA256:
        raise ValueError('Codex 0.163.0 binary changed')
    target = Path(manifest_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.luna6_prep_', dir=target.parent) as temporary:
        result = sol.BASE_PREPARE(config_path, Path(temporary) / 'manifest.json')
    protocol = result['protocol']
    protocol['version'] = VERSION
    protocol['policy'].update(
        codex_model=dict(model=sol.MODEL[0], effort=sol.MODEL[1], job_label='luna'),
        provider_route='ChatGPT login of the computer, as the paper Luna runs',
        codex='0.163.0 from data/collections/sol_codex_bin (Homebrew copy), as the GPT-6 Sol collection',
        stata_answer_reader='Paper-time strict ANSWER reader (runner.py before ab79a1b8d4)',
        cost_fields='Price GPT-6 Luna from the saved token counts at the GPT-6 Luna list prices.')
    here = Path(__file__).resolve()
    for path in (here, Path(sol.__file__).resolve(), sol.BINARY, here.parent / 'tests/test_luna6_collect.py'):
        protocol['source_hashes'][str(path)] = central.sha(path)
    result['protocol_sha256'] = central.digest(protocol)
    if target.exists():
        old = central.json.loads(target.read_text())
        if old['protocol_sha256'] != result['protocol_sha256']:
            raise ValueError('Preserve the existing manifest; select a new destination')
        return old
    central.save(target, result)
    return result


central.prepare = prepare

if __name__ == '__main__':
    central.main()
