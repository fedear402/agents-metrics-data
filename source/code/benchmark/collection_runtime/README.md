# Collection runtime

`prospective_collect.load_runtime()` loads the canonical runtime from this package. Each load creates a separate module tree. This preserves the mutable settings and adapters used by the collector.

The load order is `clearer_examples`, `expanded_recovery`, `expanded`, `prompt_variation`, `corrected`, `collector`, and `runner`. `_loader.py` resolves each module from this package. The repository root resolves from the canonical source location.

The migration preserves provider commands, prompts, timeouts, scoring functions, and execution behavior. `provenance.json` records the source paths and source hashes before migration. Historical preparation functions remain for source comparison. Use `prospective_collect.py` for current preparation and collection. The historical preparation functions reference the original report inputs and are not current entry points.

`production_statistics.py` contains the saved regression weights and bootstrap helper. `production_paper_assets.py` imports that canonical file. Build metadata retains the historical statistics and figure source hashes without requiring the report files.

`saved_response_analysis.py` contains the retained-response analysis helper. Run `uv run python code/benchmark/saved_response_analysis.py --help` for its options.

Offline migration checks compare command arguments, prompt bytes, policy settings, and every copied function body against the captured baseline. They also check independent mutable state and canonical source coverage. Run `uv run pytest -q code/benchmark/tests/test_collection_runtime_migration.py`.
