# Data and code for: AI Agents and Prompt Engineering in Econometric Coding

Sebastian Galiani (Tulane University and NBER), Federico A. López (Universidad de San Andrés), and Raul A. Sosa (Universidad de San Andrés).

This repository holds the run records, task inputs, reference outputs, analysis data, and code of the study. The study has 43,200 runs of six large language models on 40 econometric tasks in Stata, R, and Python.

## Citation

Galiani, S., López, F. A., and Sosa, R. A. (2026). Data and code for: AI Agents and Prompt Engineering in Econometric Coding. Zenodo. https://doi.org/10.5281/zenodo.23681819

## Contents

```text
analysis/  CSV files of the analysis. columns.csv describes every column of runs.csv, term_checks.csv,
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
```

## Software

- Python 3.12 or later. The study used Python 3.14.6 with numpy 2.4.2, pandas 3.0.0, scipy 1.18.0, statsmodels 0.14.6, and linearmodels 6.1. `code/requirements.txt` lists these pins. `source/pyproject.toml` and `source/uv.lock` pin the full environment.
- `code/score.py` needs only the Python standard library.
- The R reference programs need R (the study used R 4.6.1) and the packages jsonlite, forecast, lmtest, MASS, nnet, plm, sandwich, survival, and vars. The Stata reference programs ran in StataNow 19.5.

## Steps

Run every command from the repository root.

1. Run `python code/reproduce.py` to recompute Table 1 and the main rates.
2. Run `tar -xJf runs/sonnet_stata_agent.tar.xz` to extract the run records of one cell, or `for f in runs/*.tar.xz; do tar -xJf "$f"; done` to extract all cells. The run folders appear under `data/runs/` at the paths of `analysis/runs.csv` (column `record_path`). The 54 archives (408 MB) hold 1,287,815 files and 21.2 GB after extraction. Keep the file modification times. tar keeps them by default, and the Stata output-channel rule compares them with the final execution window.
3. Run `python code/score.py` to score the extracted runs again. The script checks the SHA-256 of each record and output file and compares the outcome and the failure category of each run with `analysis/runs.csv`. Its docstring states the scoring rule. The scoring modules of the study in `source/code/paper/` (`dataset.py`, `se_scoring.py`, `scoring_alternatives.py`) read files of the full project folder and do not run from this repository. `code/score.py` applies the same rule to the files of this repository.
4. Run `python code/run_references.py` to run the reference programs. Add `--generators` to regenerate the task inputs and compare them byte for byte with `tasks/`.

## Notes on the run records

- The model of a run is the `model` column of `analysis/runs.csv`, or `calls[0].model` in its run record. The model token in run folder names and in the `job.model` field of a record can be a template label. For example, `sonnet` appears in Claude Haiku 5.5 and Claude Opus 5.5 runs, and `luna` in GPT-6 Luna and GPT-6 Sol runs.
- 1,440 GPT-5.6 Luna records of the first collection store the SHA-256 of the first prompt (`prompt_sha256`) and no prompt text. `prompts/<sha256>.txt` holds that text, taken from another run with the same hash.
- The field `access_review_status` is `pending_post_collection` in 40,048 runs. The collector wrote this value when the run ended, and the record was not updated afterwards. The primary outcome does not use this field.
- The Python prompts of the constrained-agent condition name the interpreter by an absolute path in the authors' project folder on the collection machine. The run records also hold absolute paths of that machine (for example `attempt_directory`). To run this condition again, change the interpreter path in the prompt.
