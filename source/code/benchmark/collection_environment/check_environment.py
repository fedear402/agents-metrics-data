"""Check the collection environment and write its manifest.

Usage (from the repository root):

    uv run python code/benchmark/collection_environment/check_environment.py
    uv run python code/benchmark/collection_environment/check_environment.py --output PATH

The script loads every package in packages.json (and every package in
r_lock.json) in Python, R, and Stata. It records the installed versions and a
SHA-256 digest of each library tree. It writes environment_manifest.json and
prints the manifest hash. The exit status is 1 if a package does not load.

The manifest hash covers versions, file digests, and check results. It excludes
the timestamp, so two checks of an unchanged environment give the same hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = json.loads((HERE / "packages.json").read_text())
SKIP_PARTS = {"__pycache__"}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_files(root: Path) -> dict[str, str]:
    """Map each regular file under root to its SHA-256. Skip bytecode caches."""
    files = {}
    if not root.is_dir():
        return files
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if SKIP_PARTS & set(path.parts) or path.suffix == ".pyc":
            continue
        files[path.relative_to(root).as_posix()] = file_sha(path)
    return files


def digest(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def grouped_digests(files: dict[str, str]) -> dict[str, str]:
    """Digest the files of each top-level folder (one package or module)."""
    groups: dict[str, dict[str, str]] = {}
    for name, sha in files.items():
        groups.setdefault(name.split("/", 1)[0], {})[name] = sha
    return {group: digest(items) for group, items in sorted(groups.items())}


PYTHON_PROBE = r"""
import importlib, importlib.metadata as md, json, sys, sysconfig, warnings
warnings.simplefilter("ignore")
imports = json.loads(sys.argv[1])
submodules = json.loads(sys.argv[2])
checks = {}
for dist, module in imports.items():
    try:
        importlib.import_module(module)
        for name in submodules.get(dist, []):
            importlib.import_module(name)
        checks[dist] = {"ok": True, "version": md.version(dist)}
    except Exception as error:
        checks[dist] = {"ok": False, "error": f"{type(error).__name__}: {error}"[:300]}
versions = {d.metadata["Name"].lower(): d.version for d in md.distributions()}
print(json.dumps({"python": sys.version.split()[0], "executable": sys.executable,
                  "purelib": sysconfig.get_paths()["purelib"], "checks": checks,
                  "distributions": dict(sorted(versions.items()))}))
"""


def check_python() -> dict:
    spec = SPEC["python"]
    interpreter = ROOT / spec["interpreter"]
    # A package passes only if its top module and every listed submodule load. py4etrics loads its
    # top module under statsmodels 0.15 but fails in py4etrics.heckit and py4etrics.tobit.
    process = subprocess.run([str(interpreter), "-c", PYTHON_PROBE, json.dumps(spec["imports"]),
                              json.dumps(spec.get("submodules", {}))],
                             capture_output=True, text=True, timeout=600, cwd=tempfile.gettempdir())
    if process.returncode:
        raise SystemExit("Python probe failed: " + process.stderr[-2000:])
    result = json.loads(process.stdout)
    files = tree_files(Path(result["purelib"]))
    pins = SPEC["must_not_change"]["python"]
    result["pins_held"] = {name: result["distributions"].get(name) == version for name, version in pins.items()}
    result["site_packages_sha256"] = digest(files)
    result["site_packages_file_count"] = len(files)
    result["module_sha256"] = grouped_digests(files)
    result["interpreter_sha256"] = file_sha(Path(result["executable"]).resolve())
    return result


R_PROBE = r"""
args <- commandArgs(trailingOnly = TRUE)
pkgs <- jsonlite::fromJSON(args[[1]])
checks <- list()
for (p in pkgs) {
  res <- tryCatch({
    suppressPackageStartupMessages(library(p, character.only = TRUE))
    list(ok = TRUE, version = as.character(packageVersion(p)))
  }, error = function(e) list(ok = FALSE, error = substr(conditionMessage(e), 1, 300)))
  checks[[p]] <- res
}
ip <- installed.packages(noCache = TRUE)
rows <- lapply(seq_len(nrow(ip)), function(i) list(package = ip[i, "Package"], version = ip[i, "Version"],
                                                     libpath = ip[i, "LibPath"], built = ip[i, "Built"]))
cat(jsonlite::toJSON(list(r = R.version.string, rscript = file.path(R.home("bin"), "Rscript"),
                          libpaths = .libPaths(), checks = checks, installed = rows),
                     auto_unbox = TRUE, null = "null"))
"""


def r_packages() -> list[str]:
    spec = SPEC["r"]
    names = list(spec["already_installed_checked"])
    names += [item["package"] for item in spec["cran"]] + [item["package"] for item in spec["github"]]
    lock = HERE / "r_lock.json"
    if lock.exists():
        names += list(json.loads(lock.read_text())["packages"])
    return sorted(set(names), key=str.lower)


def check_r() -> dict:
    with tempfile.TemporaryDirectory(prefix="am_env_r_") as folder:
        script = Path(folder) / "probe.R"
        script.write_text(R_PROBE)
        process = subprocess.run(["Rscript", "--vanilla", str(script), json.dumps(r_packages())],
                                 capture_output=True, text=True, timeout=1800, cwd=folder)
    if process.returncode:
        raise SystemExit("R probe failed: " + process.stderr[-2000:])
    result = json.loads(process.stdout)
    libraries = {}
    for path in result["libpaths"]:
        files = tree_files(Path(path))
        libraries[path] = dict(sha256=digest(files), file_count=len(files), package_sha256=grouped_digests(files))
    result["libraries"] = libraries
    result["rscript_sha256"] = file_sha(Path(result["rscript"]).resolve())
    return result


def stata_script(commands: list[str], libraries: list[str]) -> str:
    lines = ["version 19", "clear all", "set more off", 'file open out using "checks.txt", write replace']
    for name in ("base", "site", "plus", "personal", "oldplace"):
        lines.append(f'file write out "sysdir_{name}=" (c(sysdir_{name})) _n')
    lines.append('file write out "version=" (c(stata_version)) _n')
    lines.append('file write out "born=" (c(born_date)) _n')
    lines.append('file write out "flavor=" (c(flavor)) _n')
    for command in commands:
        lines.append(f"capture which {command}")
        lines.append(f'file write out "command_{command}=" (_rc) _n')
    for library in libraries:
        lines.append(f"capture findfile {library}.mlib")
        lines.append(f'file write out "mlib_{library}=" (_rc) _n')
    # csdid runs only with a current drdid. A small staggered panel checks it.
    lines += ["clear", "set seed 20260928", "set obs 200", "generate id = _n",
              "generate g = cond(id <= 50, 0, cond(id <= 100, 2004, cond(id <= 150, 2006, 2007)))",
              "expand 5", "bysort id: generate year = 2003 + _n",
              "generate y = rnormal() + 0.5 * (g > 0 & year >= g)",
              "capture noisily csdid y, ivar(id) time(year) gvar(g) method(dripw)",
              'file write out "run_csdid=" (_rc) _n',
              "capture noisily drdid y if inlist(year, 2004, 2005) & inlist(g, 0, 2006), ivar(id) time(year) tr(g)",
              'file write out "run_drdid=" (_rc) _n',
              'file write out "complete=yes" _n', "file close out", "exit, clear"]
    return "\n".join(lines) + "\n"


def check_stata() -> dict:
    spec = SPEC["stata"]
    binary = Path(spec["binary"])
    with tempfile.TemporaryDirectory(prefix="am_env_stata_") as folder:
        directory = Path(folder)
        (directory / "check.do").write_text(stata_script(spec["check_commands"], spec["check_mata_libraries"]))
        process = subprocess.run([str(binary), "-b", "do", "check.do"], cwd=directory,
                                 capture_output=True, text=True, timeout=600)
        output = directory / "checks.txt"
        text = output.read_text() if output.exists() else ""
    if process.returncode or not text.endswith("complete=yes\n"):
        raise SystemExit("Stata probe failed")
    values = dict(line.split("=", 1) for line in text.splitlines())
    checks = {key: {"ok": value == "0", "rc": int(value)}
              for key, value in values.items() if key.startswith(("command_", "mlib_", "run_"))}
    folders = {}
    for name in ("plus", "personal", "site", "oldplace"):
        path = Path(values[f"sysdir_{name}"]).expanduser()
        files = tree_files(path)
        folders[name] = dict(path=str(path), sha256=digest(files), file_count=len(files), files=files)
    return dict(binary=str(binary), binary_sha256=file_sha(binary.resolve()), version=values["version"],
                born=values["born"], flavor=values["flavor"], checks=checks, ado_folders=folders)


def failures(manifest: dict) -> list[str]:
    bad = []
    for language in ("python", "r", "stata"):
        for name, result in manifest[language]["checks"].items():
            if not result["ok"]:
                bad.append(f"{language}:{name}")
    bad += [f"python pin changed:{name}" for name, held in manifest["python"]["pins_held"].items() if not held]
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=HERE / "environment_manifest.json")
    args = parser.parse_args()
    body = dict(spec_sha256=file_sha(HERE / "packages.json"),
                r_lock_sha256=file_sha(HERE / "r_lock.json") if (HERE / "r_lock.json").exists() else None,
                uv_lock_sha256=file_sha(ROOT / "uv.lock"),
                python=check_python(), r=check_r(), stata=check_stata())
    body["failures"] = failures(body)
    manifest = dict(manifest_sha256=digest(body), checked_at=datetime.now(timezone.utc).isoformat(), **body)
    args.output.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"manifest_sha256={manifest['manifest_sha256']}")
    print(f"python checks: {sum(v['ok'] for v in body['python']['checks'].values())}/{len(body['python']['checks'])}")
    print(f"r checks: {sum(v['ok'] for v in body['r']['checks'].values())}/{len(body['r']['checks'])}")
    print(f"stata checks: {sum(v['ok'] for v in body['stata']['checks'].values())}/{len(body['stata']['checks'])}")
    for item in body["failures"]:
        print("FAIL", item)
    return 1 if body["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
