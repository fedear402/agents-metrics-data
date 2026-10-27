"""Build the local data-only repository that the PNAS paper cites.

    uv run python code/paper/data_repository.py OUT_DIR [--analysis data/analysis/six_models]
    uv run python code/paper/data_repository.py --analysis data/analysis/six_models --check-only

The default analysis remains data/analysis/main. No remote content is changed.
OUT_DIR gets five folders and CONTENTS.txt:
    analysis/  every CSV file of the selected analysis data/ and tables/ folders
    tasks/     one folder per task: its input dataset and reference.json (from catalog.json)
    runs/      one tar.xz archive per model, language, and agency condition, holding the selected run folders
               at the paths of runs.csv (column record_path)
    prompts/   the first prompt of every run, one file per distinct text named by its SHA-256, and runs.csv
    code/      reproduce.py (a copy of code/paper/data_repository_reproduce.py) and requirements.txt
    source/    the tracked benchmark code: task definitions and data simulation, harness, scoring, analysis, and the
               reference implementations (--source-only OUT_DIR writes or replaces this folder alone)
Nothing else is written. Checks cover paths, record hashes, prompts, archive membership, and possible credentials.
"""
from __future__ import annotations

import argparse
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
CONTENTS = """analysis/  CSV files of the analysis. runs.csv has one row per run; run_pricing.csv has the cost of each run;
           term_checks.csv and se_checks.csv have each scored value and its reference; the other files are the
           tables and figure data of the paper.
tasks/     One folder per task: the input dataset and the reference output (reference.json).
runs/      Run records: prompts, model responses, scripts, logs, and outputs. One archive per model, language, and
           agency condition (tar.xz). After extraction, the run folders sit at the paths in runs.csv, column
           record_path.
prompts/   The first prompt of each run: one text file per distinct prompt, named by its SHA-256 hash. runs.csv gives
           the prompt of each run (record_path, prompt_sha256).
source/    The code of the study: task definitions and the simulation of the datasets (source/code/benchmark/),
           the harness that ran the models, the scoring, the analysis and the paper builds (source/code/paper/), and
           the reference implementations in Stata, R, and Python (source/reference_implementations/).
           source/pyproject.toml and source/uv.lock pin the Python environment.
code/      Run python code/reproduce.py from the repository root. It recomputes every model, pooled, and stored
           group contrast and checks estimates and CR1 standard errors against analysis/regression_contrasts.csv.
           Accuracy uses every run. Mean cost uses priced runs. requirements.txt lists the Python packages.
"""
REQUIREMENTS = 'numpy==2.4.2\npandas==3.0.0\nscipy==1.18.0\nstatsmodels==0.14.6\n'
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
        groups[key] = (sorted(group.folder), dict(zip(group.record_path, group.record_sha256)))
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


def validate_archive(target: Path, folders: list[str], records: dict[str, str]) -> None:
    expected = set(folders)
    found, record_matches, seen = set(), set(), set()
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
            if name in records:
                content = handle.extractfile(member)
                if content is None or hashlib.sha256(content.read()).hexdigest() != records[name]:
                    raise ValueError('Archived record hash differs from runs.csv')
                record_matches.add(name)
    if found != expected or record_matches != set(records):
        raise ValueError('Run archive coverage differs from runs.csv')


def archive(group: tuple[str, str, str], folders: list[str], out: Path,
            records: dict[str, str]) -> tuple[str, int, int]:
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
    validate_archive(target, folders, records)
    return name, len(folders), target.stat().st_size


SOURCE_PATHS = ('code/benchmark', 'code/paper', 'reference_implementations', 'pyproject.toml', 'uv.lock')
# The queue and site deployment code describe private infrastructure and are not part of the study code.
SOURCE_EXCLUDED = ('code/benchmark/central/', 'code/benchmark/tests/test_falstuff_deploy_preflight.py')


def export_source(out: Path) -> int:
    """Copy the tracked study code into out/source/ and return the number of files."""
    listed = subprocess.run(['git', 'ls-files', '-z', '--', *SOURCE_PATHS], cwd=ROOT, capture_output=True,
                            check=True).stdout.decode().split('\0')
    files = [f for f in listed if f and not f.startswith(SOURCE_EXCLUDED)]
    target = out / 'source'
    if target.exists():
        shutil.rmtree(target)
    for name in files:
        path = ROOT / safe_relative(name)
        if path.is_symlink() or not path.is_file():
            continue
        check_credentials(path)
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return sum(1 for f in target.rglob('*') if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', nargs='?', type=Path)
    parser.add_argument('--analysis', type=Path, default=ANALYSIS,
                        help='analysis selection with data/ and tables/; default: data/analysis/main')
    parser.add_argument('--check-only', action='store_true', help='validate export sources without writing an export')
    parser.add_argument('--source-only', action='store_true', help='write only OUT_DIR/source/ and CONTENTS.txt')
    args = parser.parse_args()
    if args.source_only:
        if args.out is None:
            parser.error('OUT_DIR is required with --source-only')
        print('source files', export_source(args.out.resolve()))
        (args.out.resolve() / 'CONTENTS.txt').write_text(CONTENTS)
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
    if len({p.name for p in csvs}) != len(csvs):
        raise ValueError('Duplicate analysis CSV filenames')
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
    for source in [analysis / 'data/catalog.json', *csvs, *(p for _, _, p in sources),
                   ROOT / 'code/paper/data_repository_reproduce.py']:
        check_credentials(source)
    checked = 0
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
    for csv in csvs:
        shutil.copyfile(csv, out / 'analysis' / csv.name)
    for task, entry in catalog.items():
        folder = out / 'tasks' / task
        folder.mkdir()
        for _, name, source in [s for s in sources if s[0] == task]:
            shutil.copyfile(source, folder / name)
        (folder / 'reference.json').write_text(json.dumps(entry['reference'], indent=1) + '\n')
    for digest, text in texts.items():
        (out / 'prompts' / f'{digest}.txt').write_text(text)
    mapping.to_csv(out / 'prompts' / 'runs.csv', index=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(archive, key, folders, out, records) for key, (folders, records) in groups.items()]
        done = []
        for future in as_completed(pending):
            name, count, size = future.result()
            done.append((name, count, size))
            print(f'{name:40s} {count:5d} runs {size / 1e6:7.1f} MB', flush=True)
    shutil.copyfile(ROOT / 'code/paper/data_repository_reproduce.py', out / 'code' / 'reproduce.py')
    (out / 'code' / 'requirements.txt').write_text(REQUIREMENTS)
    export_source(out)
    (out / 'CONTENTS.txt').write_text(CONTENTS)
    print('runs', sum(c for _, c, _ in done), 'archives MB', round(sum(s for *_, s in done) / 1e6, 1))


if __name__ == '__main__':
    main()
