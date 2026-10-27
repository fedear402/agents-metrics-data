# Data-generating scripts

These scripts generated the seeded task-input CSV files used by the benchmark.
They are the only copy. Each script prints the seed it uses.
The five `generate_data*.py` scripts cover the shared input files. The two
task-specific scripts, `generate_bp_koenker.py` and `generate_synth_control.py`,
generated the inputs of the two exploratory tasks outside the analysis sample.
`task_registry.json` cites this directory as the provenance of each simulated dataset.
They moved here from the repository archive on 2026-08-10. Run them only to
regenerate the canonical inputs from scratch; the shipped CSV files are the
reference the analysis uses.
