"""Output preservation for new prospective collections.

Raw submitted output files remain unchanged. Nonfinite diagnostic values become
JSON null only when records are saved. Numerical comparison runs before saving.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil


class AmbiguousSubmissionError(ValueError):
    """Several scripts exist without the required canonical submission."""

    def __init__(self, candidates):
        self.candidates = [path.name for path in candidates]
        super().__init__('Ambiguous submitted scripts: ' + ', '.join(self.candidates))


def select_submission_script(cwd, platform, canonical_name):
    """Select the canonical script or a unique script of the requested language."""
    cwd = Path(cwd)
    canonical = cwd / canonical_name
    if canonical.is_file():
        return canonical
    extension = {'python': '.py', 'r': '.R', 'stata': '.do'}[platform]
    candidates = sorted(path for path in cwd.glob('*' + extension) if path.is_file())
    if len(candidates) > 1:
        raise AmbiguousSubmissionError(candidates)
    return candidates[0] if candidates else None


def json_safe(value):
    """Keep record JSON valid when a model returns NaN or infinity."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_submission_output(cwd, spec, job, execution, *, loader):
    """Treat integers outside floating-point range as invalid result values."""
    estimates, rule, notes = loader(cwd, spec, job, execution)
    notes = list(notes)
    for term, row in estimates.items():
        for field in ('estimate', 'se', 'n_obs'):
            value = row.get(field)
            if isinstance(value, int):
                try:
                    float(value)
                except OverflowError:
                    row[field] = None
                    notes.append(f'{term}: {field} exceeds finite numeric range')
    return estimates, rule, notes


def execute_submission(script, platform, cwd, destination, attempt, *, executor, harness):
    """Expose the selected Stata script's fresh batch log as run.log."""
    cwd, destination = Path(cwd), Path(destination)
    batch_log = cwd / (Path(script).stem + '.log')
    if platform == 'stata':
        # The central runner clears run.log. Clear the selected batch log too.
        batch_log.unlink(missing_ok=True)
    result = executor(script, platform, cwd, destination, attempt)
    if platform != 'stata':
        return result
    canonical_log = cwd / 'run.log'
    if not canonical_log.exists() and batch_log.exists():
        shutil.copyfile(batch_log, canonical_log)
    log = canonical_log.read_text(errors='replace') if canonical_log.exists() else ''
    result['ok'] = result['returncode'] == 0 and not result.get('timed_out') and not harness.STATA_ERROR_RE.search(log)
    result['stata_log_tail'] = log[-3000:]
    result['stata_batch_log'] = batch_log.name
    # Replace the inherited receipt with the corrected execution evidence.
    import json
    receipt = destination / f'execution_attempt{attempt}.json'
    receipt.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + '\n')
    return result
