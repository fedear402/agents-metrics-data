# Collection environment

This folder holds the one standard package set for Python, R, and Stata that collection runs use (decision 7). The packages are installed before launch. The sandbox makes the libraries read-only during the collection, so no run installs a package.

## Files

| File | Purpose |
|---|---|
| `packages.json` | The package set, the versions that must not change, and the Stata commands to check. |
| `pyproject.toml` group `collection` (repository root) | Python packages, pinned. `uv.lock` holds the full resolution. `default-groups` includes `collection`, so `uv sync` keeps it in `.venv`. |
| `r_packages.R` | `plan` writes `r_lock.json` from the CRAN binary index. `install` installs the locked versions and stops if an installed package changes. |
| `r_lock.json` | Every R package added (targets and new dependencies) with its exact version and source. |
| `stata_packages.do` | SSC installs without `replace`, and the `drdid` update. |
| `stata_install_results.txt` | Return code of each Stata install. |
| `check_environment.py` | Loads every package in each language and writes `environment_manifest.json`. |
| `environment_manifest.json` | Versions, file digests, check results, and `manifest_sha256`. |

## Commands

Run from the repository root.

```bash
uv sync                                                               # Python: dev and collection groups
Rscript --vanilla code/benchmark/collection_environment/r_packages.R plan
Rscript --vanilla code/benchmark/collection_environment/r_packages.R install
/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se -b do code/benchmark/collection_environment/stata_packages.do
uv run python code/benchmark/collection_environment/check_environment.py
```

`check_environment.py` exits with status 1 if a package does not load or a Python analysis pin changed. A Python package passes only if its top module and every submodule listed under `python.submodules` in `packages.json` load. The manifest hash excludes the time of the check. Two checks of an unchanged environment give the same hash.

## Install rules

- Python: the `collection` group resolves with every `dev` pin held (numpy 2.4.2, pandas 3.0.0, scipy 1.18.0, statsmodels 0.14.6, linearmodels 6.1, matplotlib 3.11.0).
- Python: py4etrics 0.1.9 needs statsmodels 0.14.x. Its `heckit` and `tobit` modules import `statsmodels.tools.decorators`, which statsmodels 0.15 removed. Keep statsmodels at 0.14.6 while py4etrics is in the group.

## Packages installed but never named

py4etrics 0.1.9 and ivmodels 0.10.0 are in the `collection` group since 2026-09-29 (decision record `quality_reports/decision_records/2026-09-29_routine-classes.md`). An agent can use them if it thinks of them. No prompt, example, or instruction names them. `final_bank_examples.py` reports an example that names them, and `test_final_bank_routines.py` checks every prompt, example, and `task.json`.

- py4etrics (Tetsu Haruyama) is a small personal package. It gives the Heckman two-step model (`heckit.Heckit`), the Tobit model (`tobit.Tobit`), and truncated regression (`truncreg.Truncreg`). `routines.json` classes it as `small_package`: installed, not a standard routine.
- ivmodels (Malte Londschien) is the package of the method authors (Londschien and Buhlmann, Journal of Econometrics, 106239). It gives weak-instrument tests and their confidence sets and k-class estimators (`KClass`). `routines.json` classes it as `method_author_package`.

The lock added eight packages and changed no version: glum 3.4.1, ivmodels 0.10.0, llvmlite 0.49.0, numba 0.67.0, numexpr 2.14.2, py4etrics 0.1.9, pyarrow 25.0.1, and tabmat 4.2.1.
- R: `plan` rejects a target if a new package needs a newer version of an installed package. `install` never replaces an installed package. A version that CRAN no longer serves as a binary is built from the CRAN source archive. `synthdid` is built from the tarball of its locked GitHub commit.
- Stata: `net install` without `replace` stops with r(602) and copies no file if a file exists. The only `replace` is `drdid`. csdid 1.81 stops with "Program DRDID is outdated" under drdid 1.71. The analysis does not use drdid. SSC serves only current files, so the pin is the SSC Distribution-Date and the file digests in the manifest.

## Read-only libraries during the collection

The collector runs every script execution and every agent tool command under the Seatbelt allowlist tested in `quality_reports/audits/2026-09-26_rerun-design/work/isolation/` (`pilot/offline/execution.sb`, `claude.sb`, and the Codex permission profile).

- The profile denies every write (`(deny file-write* (subpath "/"))`). It then allows writes only in the run folder and its `TMPDIR`.
- It allows reads of the project `.venv`, the uv Python, the R user library, Stata PLUS, and Stata PERSONAL. The R framework library is outside the denied roots, so it is readable and not writable.
- Network is denied for script executions and Codex tools.

The probe of 2026-09-28 used this profile after the install (`quality_reports/audits/2026-09-26_rerun-design/work/final-bank-env/readonly_probe/`). All new packages loaded. A write to Python site-packages gave `PermissionError`. `install.packages` reported that the R user library is not writable. `ssc install` gave r(677), and a file write to PLUS gave r(603).

Set these variables in the clean run environment. Without them, a package writes a warning to stderr when it cannot create a folder in the home folder.

- `MPLCONFIGDIR=$TMPDIR/mpl` (matplotlib, also loaded by pyfixest and lifelines).
- `R_USER_CONFIG_DIR`, `R_USER_DATA_DIR`, and `R_USER_CACHE_DIR` under `$TMPDIR` (marginaleffects creates `~/Library/Preferences/org.R-project.R` on load).
- Keep `HOME` at the real home folder, or set `R_LIBS_USER` to the R user library. Another `HOME` hides the R user library from `Rscript --vanilla`.
- Copy a prepared matplotlib font cache into `$TMPDIR/mpl`. An empty `MPLCONFIGDIR` prints "Matplotlib is building the font cache" to stderr in every run.

`code/benchmark/final_bank_runner.py` sets all of these (`clean_env`, `seed_matplotlib`, `prepare_run`). Since 2026-09-29 the runner also sets `TZ=UTC` and gives each cell the PATH of its language only: Python runs see `python` and `python3` (the project venv), R runs see `R` and `Rscript`, and Stata runs see `stata` and `stata-se`. The other interpreter names answer "command not found", and the R and Stata profiles deny reads of the Python venv (decision record `quality_reports/decision_records/2026-09-29_harness-harmonization.md`). The dry run of 2026-09-28 found no stderr line from matplotlib, lifelines, pyfixest, or marginaleffects under the execution profile with this environment (`quality_reports/audits/2026-09-26_rerun-design/work/final-bank-dryrun/step4_env_probe.json`).

## Collection manifest

The current manifest (2026-09-29, after the py4etrics and ivmodels install) has `manifest_sha256` `2e820d09c1bed5f19f640d1f9b5b6dcd5f9db47964dabdc4d1d8470101754e8b`. Two checks gave the same hash. The manifest of 2026-09-28 had `0cffb1eb8a359d4da8a10b99679749cd6eca2a056b91c446247511e9f5e23b79`.

The collection manifest records the environment by these fields:

- `environment.manifest_sha256`: the `manifest_sha256` of `environment_manifest.json`.
- `environment.manifest_file_sha256`: the SHA-256 of the file.
- `environment.packages_json_sha256`, `environment.r_lock_sha256`, and `environment.uv_lock_sha256`.

The runner runs `check_environment.py --output <collection>/environment_check_<block>.json` at each block start and end and before scoring. A `manifest_sha256` different from the manifest value stops the block. `final_bank_runner.run_block` implements the start and end checks and writes `environment.manifest_sha256` into the block record and every run record. One check takes about 21 seconds.
