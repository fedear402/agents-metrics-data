"""Build the local data-only repository that the PNAS paper cites.

    uv run python code/paper/data_repository.py OUT_DIR [--analysis data/analysis/six_models]
    uv run python code/paper/data_repository.py OUT_DIR --analysis data/analysis/six_models --reuse-runs OLD/runs
    uv run python code/paper/data_repository.py --analysis data/analysis/six_models --check-only

The default analysis remains data/analysis/main. No remote content is changed.
OUT_DIR gets six folders, README.md, CONTENTS.txt, and .zenodo.json:
    analysis/  every CSV file of the selected analysis data/ and tables/ folders, the plotted values of the cost-group
               figure (cost_of_correctness_groups.csv), reference_validation.csv, and the column dictionary columns.csv
               (from code/paper/data_repository_columns.csv; the build stops when it misses or adds a column)
    tasks/     one folder per task: its input dataset and reference.json (from catalog.json)
    runs/      one tar.xz archive per model, language, and agency condition, holding the selected run folders
               at the paths of runs.csv (column record_path). --reuse-runs DIR copies the archives of an earlier
               export after the same membership and hash checks, so unchanged archives keep their bytes.
    prompts/   the first prompt of every run, one file per distinct text named by its SHA-256, and runs.csv
    code/      reproduce.py, score.py, and run_references.py (copies of code/paper/data_repository_reproduce.py,
               data_repository_score.py, and data_repository_references.py), scoring_spec.json
               (data/analysis/rescore_v2/scoring_spec.json), scoring_alternatives.json (from the analysis data/ folder,
               when present), and requirements.txt
    source/    the benchmark code: task definitions and data simulation, harness, scoring, analysis, and the
               reference implementations of the 40 tasks. Tracked files and untracked files that Git does not ignore
               are exported (--source-only OUT_DIR writes or replaces this folder alone).
Nothing else is written. Checks cover paths, record and output hashes, prompts, archive membership, possible
credentials, Stata license serial numbers, and home-directory paths in the generated files and the reference programs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / 'data/analysis/main'
SCORING_SPEC = ROOT / 'data/analysis/rescore_v2/scoring_spec.json'
REFERENCE_VALIDATION = ROOT / 'data/analysis/main/data/reference_validation.csv'
COLUMNS = ROOT / 'code/paper/data_repository_columns.csv'
DOCUMENTED = ('runs.csv', 'term_checks.csv', 'se_checks.csv', 'run_pricing.csv')
CODE_FILES = {'reproduce.py': 'code/paper/data_repository_reproduce.py',
              'score.py': 'code/paper/data_repository_score.py',
              'run_references.py': 'code/paper/data_repository_references.py'}
DOI = '10.5281/zenodo.23681819'
CONTENTS = """analysis/  CSV files of the analysis. columns.csv describes every column of runs.csv, term_checks.csv,
           se_checks.csv, and run_pricing.csv.
             runs.csv                  one row per run: design cell, primary outcome, failure category, paths
                                       and SHA-256 hashes of the run record and of the scored output
             run_pricing.csv           tokens and list-price cost of each run
             term_checks.csv           each scored estimate with its reference, tolerance, and verdict
             se_checks.csv             each scored standard error (nine tasks) with its accepted value and verdict
             alternative_checks.csv    each value that is closer to a wrong-method alternative than to every
                                       accepted value
             reference_validation.csv  for each task, the routes that reproduce the reference output
           Plotted and tabulated values of the paper:
             Fig. 1    benchmark_design.csv            Fig. S1   task_heterogeneity.csv
             Table 1   regression_contrasts.csv        Fig. S2   model_agency_frontier_log.csv
             Fig. 2    prompt_agency.csv               Fig. S3   appendix_pass_at_k.csv
             Fig. 3    appendix_silent_vs_loud.csv     Fig. S4   runtime_by_agency.csv
             Fig. 4    cost_of_correctness_groups.csv  SI table of success and failure types by model,
                                                       language, and agency: runs.csv
                                                       SI robustness table: robustness.csv
           The other files hold per-model versions of these values, the joint test across models (cross_model.csv),
           the cost denominators, and the scoring audit.
tasks/     One folder per task: the input dataset and the reference output (reference.json).
runs/      Run records: prompts, model responses, scripts, logs, and outputs. One archive per model, language, and
           agency condition (tar.xz). After extraction, the run folders sit at the paths in analysis/runs.csv, column
           record_path.
prompts/   The first prompt of each run: one text file per distinct prompt, named by its SHA-256 hash.
           prompts/runs.csv gives the prompt of each run (record_path, prompt_sha256).
source/    The code of the study: task definitions and the simulation of the datasets (source/code/benchmark/; the
           inputs of the 20 original tasks come from source/code/benchmark/dgp/), the harness that ran the models, the
           scoring, the analysis and the paper builds (source/code/paper/), and the reference programs of the 40
           tasks (source/reference_implementations/manual/<task>/: manual_python.py, manual_r.R, manual_stata.do,
           and, for the 20 added tasks, generate_data.py). source/pyproject.toml and source/uv.lock pin the Python
           environment.
code/      Run each script from the repository root.
             reproduce.py       recomputes every model, pooled, and stored group contrast and checks estimates and
                                CR1 standard errors against analysis/regression_contrasts.csv
             score.py           scores the extracted run records again and compares each outcome with runs.csv
                                (Python standard library only; rules in scoring_spec.json and
                                scoring_alternatives.json)
             run_references.py  runs the reference programs (Python always, R and Stata when installed) and
                                compares their output with tasks/<task>/reference.json; --generators also
                                regenerates every task input
             requirements.txt   the Python packages of reproduce.py and of the Python reference programs
"""
REQUIREMENTS = 'numpy==2.4.2\npandas==3.0.0\nscipy==1.18.0\nstatsmodels==0.14.6\nlinearmodels==6.1\n'
README = """# Data and code for: AI Agents and Prompt Engineering in Econometric Coding

Sebastian Galiani (Tulane University and NBER), Federico A. López (Universidad de San Andrés), and Raul A. Sosa (Universidad de San Andrés).

This repository holds the run records, task inputs, reference outputs, analysis data, and code of the study. The study has {runs:,} runs of six large language models on 40 econometric tasks in Stata, R, and Python.

## Citation

Galiani, S., López, F. A., and Sosa, R. A. (2026). Data and code for: AI Agents and Prompt Engineering in Econometric Coding. Zenodo. https://doi.org/{doi}

## Contents

```text
{contents}```

## Software

- Python 3.12 or later. The study used Python 3.14.6 with numpy 2.4.2, pandas 3.0.0, scipy 1.18.0, statsmodels 0.14.6, and linearmodels 6.1. `code/requirements.txt` lists these pins. `source/pyproject.toml` and `source/uv.lock` pin the full environment.
- `code/score.py` needs only the Python standard library.
- The R reference programs need R (the study used R 4.6.1) and the packages jsonlite, forecast, lmtest, MASS, nnet, plm, sandwich, survival, and vars. The Stata reference programs ran in StataNow 19.5.

## Steps

Run every command from the repository root.

1. Run `python code/reproduce.py` to recompute Table 1 and the main rates.
2. Run `tar -xJf runs/sonnet_stata_agent.tar.xz` to extract the run records of one cell, or `for f in runs/*.tar.xz; do tar -xJf "$f"; done` to extract all cells. The run folders appear under `data/runs/` at the paths of `analysis/runs.csv` (column `record_path`). The {archives} archives ({archive_mb:,} MB) hold {files:,} files and {extracted_gb:.1f} GB after extraction. Keep the file modification times. tar keeps them by default, and the Stata output-channel rule compares them with the final execution window.
3. Run `python code/score.py` to score the extracted runs again. The script checks the SHA-256 of each record and output file and compares the outcome and the failure category of each run with `analysis/runs.csv`. Its docstring states the scoring rule. The scoring modules of the study in `source/code/paper/` (`dataset.py`, `se_scoring.py`, `scoring_alternatives.py`) read files of the full project folder and do not run from this repository. `code/score.py` applies the same rule to the files of this repository.
4. Run `python code/run_references.py` to run the reference programs. Add `--generators` to regenerate the task inputs and compare them byte for byte with `tasks/`.

## Notes on the run records

- The model of a run is the `model` column of `analysis/runs.csv`, or `calls[0].model` in its run record. The model token in run folder names and in the `job.model` field of a record can be a template label. For example, `sonnet` appears in Claude Haiku 5.5 and Claude Opus 5.5 runs, and `luna` in GPT-6 Luna and GPT-6 Sol runs.
- {hash_only:,} GPT-5.6 Luna records of the first collection store the SHA-256 of the first prompt (`prompt_sha256`) and no prompt text. `prompts/<sha256>.txt` holds that text, taken from another run with the same hash.
- The field `access_review_status` is `pending_post_collection` in {pending:,} runs. The collector wrote this value when the run ended, and the record was not updated afterwards. The primary outcome does not use this field.
- The Python prompts of the constrained-agent condition name the interpreter by an absolute path in the authors' project folder on the collection machine. The run records also hold absolute paths of that machine (for example `attempt_directory`). To run this condition again, change the interpreter path in the prompt.
"""
ZENODO = {
    'title': 'Data and code for: AI Agents and Prompt Engineering in Econometric Coding',
    'upload_type': 'dataset',
    'description': ('Run records, task inputs, reference outputs, analysis data, and the code of the study (data '
                    'simulation, reference implementations, the harness that ran the models, scoring, and analysis) '
                    'for 43,200 runs of six large language models on 40 econometric tasks in Stata, R, and Python. '
                    'README.md describes the folders and the reproduction steps.'),
    'creators': [
        {'name': 'Galiani, Sebastian', 'affiliation': 'Tulane University; National Bureau of Economic Research'},
        {'name': 'López, Federico A.', 'affiliation': 'Universidad de San Andrés'},
        {'name': 'Sosa, Raul A.', 'affiliation': 'Universidad de San Andrés'},
    ],
    'keywords': ['large language models', 'AI agents', 'econometric software', 'Stata', 'R', 'Python'],
}
LICENSE_SERIAL = re.compile(br'(?im)^[ \t]*Serial[ \t]+number:[ \t]*\d{6,}')
HOME_PATH = re.compile(br'/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\')
SECRET_PATTERNS = (
    re.compile(br'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    re.compile(br'\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,})\b'),
    re.compile(br'''(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization)\b["']?\s*[:=]\s*["']?(?:Bearer\s+)?[A-Za-z0-9_./+-]{24,}'''),
)


def safe_relative(value: str) -> str:
    """Reject archive paths that can escape the export or change tar arguments."""
    if not isinstance(value, str) or not value or '\\' in value or any(ord(c) < 32 for c in value):
        raise ValueError('Unsafe export path')
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive or value.startswith('-')
            or any(p in ('', '.', '..') for p in value.split('/'))):
        raise ValueError('Unsafe export path')
    return path.as_posix()


def source_path(value: str) -> Path:
    relative = safe_relative(value)
    source = ROOT / relative
    if not source.resolve(strict=True).is_relative_to(ROOT.resolve()):
        raise ValueError('Export source escapes the project')
    if source.is_symlink():
        raise ValueError('Export source is a symbolic link')
    return source


def check_credentials(path: Path) -> None:
    """Scan bytes without printing matched content."""
    if path.name.lower() in {'.env', 'id_rsa', 'id_ed25519', 'credentials.json', 'credentials'}:
        raise ValueError('Possible credential file in export sources')
    with path.open('rb') as handle:
        previous = b''
        while block := handle.read(1024 * 1024):
            data = previous + block
            lower = data.lower()
            candidates = (b'private key' in lower,
                          any(marker in data for marker in (b'sk-', b'AKIA', b'gh')),
                          any(marker in lower for marker in (b'api', b'access', b'refresh', b'authorization')))
            if any(candidate and pattern.search(data) for candidate, pattern in zip(candidates, SECRET_PATTERNS)):
                raise ValueError('Possible credential content in export sources')
            previous = data[-1024:]


def check_privacy(path: Path, home: bool = True) -> None:
    """Reject a Stata license serial number and, when home is True, a home-directory path. Prints no content."""
    data = path.read_bytes()
    if LICENSE_SERIAL.search(data):
        raise ValueError(f'Stata license serial number in {path.name}')
    if home and HOME_PATH.search(data):
        raise ValueError(f'Home-directory path in {path.name}')


def run_folder(record_path: str) -> str:
    """The folder of one run: the parent of attempt_NNNN when the record sits in an attempt folder."""
    path = PurePosixPath(safe_relative(record_path)).parent
    if str(path) == '.':
        raise ValueError('Run record needs a run folder')
    return str(path.parent if path.name.startswith('attempt_') else path)


def prompt_data(runs: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """The first prompt of each run: its prompt_attempt1 file, the prompt field of its record, or (for records that
    store only prompt_sha256) a stored prompt of another run with the same hash."""
    texts, rows, pending = {}, [], []
    for record in runs.record_path:
        folder = source_path(run_folder(record))
        files = sorted(p for p in folder.rglob('prompt_attempt1*')
                       if p.is_file() and re.match(r'prompt_attempt1(?:\D|$)', p.name))
        data = json.loads(source_path(record).read_text())
        candidates = {p.read_text() for p in files}
        # Self-repair records can store the later repair prompt. The saved first-prompt file takes precedence.
        if not files and isinstance(data.get('prompt'), str):
            candidates.add(data['prompt'])
        if len(candidates) > 1:
            raise ValueError('Conflicting first prompt text for a selected run')
        text = next(iter(candidates)) if candidates else None
        if text is None:
            pending.append((record, data.get('prompt_sha256')))
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if not files and data.get('prompt_sha256') and data['prompt_sha256'] != digest:
            raise ValueError('Stored first prompt hash differs from its text')
        texts[digest] = text
        rows.append((record, digest))
    for record, digest in pending:
        if digest not in texts:
            raise ValueError('No stored prompt text for a selected run')
        rows.append((record, digest))
    mapping = pd.DataFrame(rows, columns=['record_path', 'prompt_sha256'])
    if mapping.record_path.duplicated().any() or set(mapping.record_path) != set(runs.record_path):
        raise ValueError('First prompt mapping coverage differs from runs.csv')
    mapping.attrs['hash_only'] = len(pending)  # records that store only the prompt hash
    return texts, mapping


def prompts(runs: pd.DataFrame, out: Path) -> int:
    texts, mapping = prompt_data(runs)
    (out / 'prompts').mkdir(exist_ok=True)
    for digest, text in texts.items():
        (out / 'prompts' / f'{digest}.txt').write_text(text)
    mapping.to_csv(out / 'prompts' / 'runs.csv', index=False)
    return len(texts)


def validate_runs(runs: pd.DataFrame) -> dict:
    if runs.record_path.isna().any() or runs.record_path.duplicated().any():
        raise ValueError('Missing or duplicate selected run records')
    folders = runs.record_path.map(run_folder)
    if folders.duplicated().any():
        raise ValueError('Duplicate selected run folders')
    folder_set = set(folders)
    if any(parent.as_posix() in folder_set for folder in folders for parent in PurePosixPath(folder).parents):
        raise ValueError('Selected run folders overlap')
    for row in runs.itertuples():
        record = source_path(row.record_path)
        if hashlib.sha256(record.read_bytes()).hexdigest() != row.record_sha256:
            raise ValueError('Selected record hash differs from runs.csv')
    groups = {}
    frame = runs.assign(folder=folders)
    for key, group in frame.groupby(['model', 'platform', 'agency'], dropna=False):
        if not all(isinstance(k, str) and re.fullmatch(r'[A-Za-z0-9_-]+', k) for k in key):
            raise ValueError('Unsafe run archive label')
        hashes = dict(zip(group.record_path, group.record_sha256))
        # The archive check also covers the scored output of every run that has one.
        scored = group[group.output_sha256.notna()] if 'output_sha256' in group else None
        for path, digest, folder in ([] if scored is None else zip(scored.output_path, scored.output_sha256,
                                                                   scored.folder)):
            if not safe_relative(path).startswith(folder + '/'):
                raise ValueError('Scored output outside its run folder')
            if hashlib.sha256(source_path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('Selected output hash differs from runs.csv')
            hashes[path] = digest
        groups[key] = (sorted(group.folder), hashes)
    names = ['_'.join(key) for key in groups]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate run archive filenames')
    return groups


def check_sources(folders: list[str]) -> int:
    count = 0
    for folder in folders:
        source = source_path(folder)
        for path in [source, *source.rglob('*')]:
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError('Links or special files in selected run folders')
            if not path.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Run archive member escapes the project')
            safe_relative(path.relative_to(ROOT).as_posix())
            if path.is_file():
                check_credentials(path)
                count += 1
    return count


def validate_archive(target: Path, folders: list[str], records: dict[str, str]) -> tuple[int, int]:
    """Check membership and the listed hashes; return the bytes and the number of files after extraction."""
    expected = set(folders)
    found, record_matches, seen = set(), set(), set()
    size = files = 0
    with tarfile.open(target, 'r|xz') as handle:
        for member in handle:
            name = safe_relative(member.name.rstrip('/'))
            if name in seen or not (member.isfile() or member.isdir()):
                raise ValueError('Duplicate or unsafe run archive member')
            seen.add(name)
            if not ({name, *(p.as_posix() for p in PurePosixPath(name).parents)} & expected):
                raise ValueError('Unexpected run archive member')
            if name in expected and member.isdir():
                found.add(name)
            if member.isfile():
                size, files = size + member.size, files + 1
            if name in records:
                content = handle.extractfile(member)
                if content is None or hashlib.sha256(content.read()).hexdigest() != records[name]:
                    raise ValueError('Archived record hash differs from runs.csv')
                record_matches.add(name)
    if found != expected or record_matches != set(records):
        raise ValueError('Run archive coverage differs from runs.csv')
    return size, files


def archive(group: tuple[str, str, str], folders: list[str], out: Path,
            records: dict[str, str]) -> tuple[str, int, int, int, int]:
    """Write one run archive; return its name, runs, compressed bytes, extracted bytes, and files."""
    name = '_'.join(group) + '.tar.xz'
    target = out / 'runs' / name
    with target.open('wb') as handle:
        process = subprocess.Popen(['xz', '-T2', '-6'], stdin=subprocess.PIPE, stdout=handle)
        try:
            with tarfile.open(fileobj=process.stdin, mode='w|', dereference=True) as tar:
                for folder in folders:
                    tar.add(source_path(folder), arcname=folder)
            process.stdin.close()
            if process.wait():
                raise ValueError(f'Compression failed for {name}')
        except BaseException:
            process.kill()
            process.wait()
            target.unlink(missing_ok=True)
            raise
    size, files = validate_archive(target, folders, records)
    return name, len(folders), target.stat().st_size, size, files


def reuse_archive(group: tuple[str, str, str], folders: list[str], out: Path, records: dict[str, str],
                  source: Path) -> tuple[str, int, int, int, int]:
    """Copy one archive of an earlier export after the checks of archive(); its bytes stay the same."""
    name = '_'.join(group) + '.tar.xz'
    if not (source / name).is_file():
        raise ValueError(f'No archive {name} to reuse')
    target = out / 'runs' / name
    shutil.copyfile(source / name, target)
    try:
        size, files = validate_archive(target, folders, records)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return name, len(folders), target.stat().st_size, size, files


SOURCE_PATHS = ('code/benchmark', 'code/paper', 'reference_implementations', 'pyproject.toml', 'uv.lock')
# The queue and site deployment code describe private infrastructure and are not part of the study code.
# dynpanel_ab and wcb_few are earlier tasks outside the 40-task sample.
SOURCE_EXCLUDED = ('code/benchmark/central/', 'code/benchmark/tests/test_falstuff_deploy_preflight.py',
                   'reference_implementations/manual/dynpanel_ab/', 'reference_implementations/manual/wcb_few/')
# Files written for the data repository: they must hold no home-directory path.
SOURCE_PRIVATE_CHECKED = ('reference_implementations/', 'code/paper/data_repository')


def export_source(out: Path) -> int:
    """Copy the study code (tracked files and untracked files that Git does not ignore) into out/source/.

    Returns the number of files. Every file is scanned for credentials and Stata license serial numbers, and the
    files under SOURCE_PRIVATE_CHECKED also for home-directory paths.
    """
    listed = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--',
                             *SOURCE_PATHS], cwd=ROOT, capture_output=True, check=True).stdout.decode().split('\0')
    files = sorted({f for f in listed if f and not f.startswith(SOURCE_EXCLUDED)})
    target = out / 'source'
    if target.exists():
        shutil.rmtree(target)
    for name in files:
        path = ROOT / safe_relative(name)
        if path.is_symlink() or not path.is_file():
            continue
        check_credentials(path)
        check_privacy(path, home=name.startswith(SOURCE_PRIVATE_CHECKED))
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return sum(1 for f in target.rglob('*') if f.is_file())


def analysis_extras(analysis: Path) -> dict[str, Path]:
    """Analysis files outside data/ and tables/: {exported name: source}, for the sources that exist."""
    extras = {'cost_of_correctness_groups.csv': analysis / 'cheap_vs_expensive/data/cost_of_correctness.csv',
              'reference_validation.csv': REFERENCE_VALIDATION}
    return {name: path for name, path in extras.items() if path.is_file()}


def check_columns(csvs: list[Path], dictionary: Path | None = None) -> None:
    """The column dictionary must list exactly the columns of each documented file."""
    with (dictionary or COLUMNS).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for name in DOCUMENTED:
        source = next((p for p in csvs if p.name == name), None)
        if source is None:
            continue
        with source.open(newline='') as handle:
            header = next(csv.reader(handle))
        listed = [r['column'] for r in rows if r['file'] == name]
        if sorted(listed) != sorted(header) or any(not r['description'].strip() for r in rows if r['file'] == name):
            missing, extra = sorted(set(header) - set(listed)), sorted(set(listed) - set(header))
            raise ValueError(f'Column dictionary differs from {name}: missing {missing}, not in the file {extra}')


def code_files(analysis: Path) -> dict[str, Path]:
    """{name in code/: source} of the scripts and the two scoring specification files."""
    files = {name: ROOT / source for name, source in CODE_FILES.items()}
    files['scoring_spec.json'] = SCORING_SPEC
    alternatives = analysis / 'data/scoring_alternatives.json'
    if alternatives.is_file():
        files['scoring_alternatives.json'] = alternatives
    return files


def write_text(path: Path, text: str) -> None:
    path.write_text(text)
    check_privacy(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', nargs='?', type=Path)
    parser.add_argument('--analysis', type=Path, default=ANALYSIS,
                        help='analysis selection with data/ and tables/; default: data/analysis/main')
    parser.add_argument('--check-only', action='store_true', help='validate export sources without writing an export')
    parser.add_argument('--source-only', action='store_true', help='write only OUT_DIR/source/ and CONTENTS.txt')
    parser.add_argument('--reuse-runs', type=Path,
                        help='copy the run archives from this runs/ folder of an earlier export (checked as when built)')
    args = parser.parse_args()
    if args.source_only:
        if args.out is None:
            parser.error('OUT_DIR is required with --source-only')
        print('source files', export_source(args.out.resolve()))
        write_text(args.out.resolve() / 'CONTENTS.txt', CONTENTS)
        return
    if not args.check_only and args.out is None:
        parser.error('OUT_DIR is required unless --check-only is used')
    analysis = args.analysis.resolve()
    catalog = json.loads((analysis / 'data/catalog.json').read_text())
    runs = pd.read_csv(analysis / 'data/runs.csv', low_memory=False)
    from data_repository_reproduce import priced_runs
    priced_runs(runs, pd.read_csv(analysis / 'data/run_pricing.csv', low_memory=False))
    groups = validate_runs(runs)
    if set(catalog) != set(runs.task):
        raise ValueError('catalog.json and runs.csv name different tasks')
    csvs = sorted([*(analysis / 'data').glob('*.csv'), *(analysis / 'tables').glob('*.csv')])
    extras = analysis_extras(analysis)
    names = [p.name for p in csvs] + list(extras) + ['columns.csv']
    if len(set(names)) != len(names):
        raise ValueError('Duplicate analysis CSV filenames')
    check_columns(csvs)
    code = code_files(analysis)
    sources = []
    for task, entry in catalog.items():
        if safe_relative(task) != task or len(PurePosixPath(task).parts) != 1:
            raise ValueError('Unsafe task folder name')
        files = entry.get('input_files') or {Path(entry['data_path']).name: entry['data_path']}
        if 'reference.json' in files:
            raise ValueError('Task input would replace reference.json')
        for name, source in files.items():
            if safe_relative(name) != name or len(PurePosixPath(name).parts) != 1:
                raise ValueError('Unsafe task input filename')
            path = Path(source)
            path = path if path.is_absolute() else ROOT / path
            if not path.resolve(strict=True).is_relative_to(ROOT.resolve()) or path.is_symlink():
                raise ValueError('Task input escapes the project')
            sources.append((task, name, path))
    for source in [analysis / 'data/catalog.json', *csvs, *extras.values(), *(p for _, _, p in sources),
                   *code.values(), COLUMNS]:
        check_credentials(source)
    for source in [*code.values(), COLUMNS]:
        check_privacy(source)
    checked = 0
    if args.reuse_runs is None:
        for key, (folders, _) in groups.items():
            checked += check_sources(folders)
            print(f'Checked {" / ".join(key)} source files.', flush=True)
    texts, mapping = prompt_data(runs)
    print(f'Checked {len(runs)} records, {len(catalog)} tasks, {len(texts)} first prompts, {checked} run files.')
    if args.check_only:
        return
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty')
    if any(out.is_relative_to(source_path(folder)) for folders, _ in groups.values() for folder in folders):
        raise ValueError('Export destination is inside a selected run folder')
    for folder in ('analysis', 'tasks', 'runs', 'prompts', 'code'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for csv_path in csvs:
        shutil.copyfile(csv_path, out / 'analysis' / csv_path.name)
    for name, source in extras.items():
        shutil.copyfile(source, out / 'analysis' / name)
    shutil.copyfile(COLUMNS, out / 'analysis/columns.csv')
    for task, entry in catalog.items():
        folder = out / 'tasks' / task
        folder.mkdir()
        for _, name, source in [s for s in sources if s[0] == task]:
            shutil.copyfile(source, folder / name)
        (folder / 'reference.json').write_text(json.dumps(entry['reference'], indent=1) + '\n')
    for digest, text in texts.items():
        (out / 'prompts' / f'{digest}.txt').write_text(text)
    mapping.to_csv(out / 'prompts' / 'runs.csv', index=False)
    reuse = args.reuse_runs.resolve() if args.reuse_runs else None
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(reuse_archive, key, folders, out, records, reuse) if reuse else
                   pool.submit(archive, key, folders, out, records) for key, (folders, records) in groups.items()]
        done = []
        for future in as_completed(pending):
            name, count, size, extracted, files = future.result()
            done.append((name, count, size, extracted, files))
            print(f'{name:40s} {count:5d} runs {size / 1e6:7.1f} MB', flush=True)
    for name, source in code.items():
        shutil.copyfile(source, out / 'code' / name)
    (out / 'code' / 'requirements.txt').write_text(REQUIREMENTS)
    export_source(out)
    write_text(out / 'CONTENTS.txt', CONTENTS)
    write_text(out / 'README.md', README.format(
        runs=len(runs), doi=DOI, contents=CONTENTS, archives=len(done),
        archive_mb=round(sum(d[2] for d in done) / 1e6), files=sum(d[4] for d in done),
        extracted_gb=sum(d[3] for d in done) / 1e9, hash_only=mapping.attrs['hash_only'],
        pending=int(runs.get('access_review_status', pd.Series(dtype=str)).eq('pending_post_collection').sum())))
    write_text(out / '.zenodo.json', json.dumps(ZENODO, indent=2, ensure_ascii=False) + '\n')
    print('runs', sum(d[1] for d in done), 'archives MB', round(sum(d[2] for d in done) / 1e6, 1),
          'extracted GB', round(sum(d[3] for d in done) / 1e9, 2))


if __name__ == '__main__':
    main()
