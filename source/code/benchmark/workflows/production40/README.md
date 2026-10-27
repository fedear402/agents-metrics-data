# Historical cleanup

`cleanup.py` archives the historical report, audit, and archive trees outside the repository. It materializes data companions first. Every copied file is checked by SHA-256. Symbolic links remain links. Source removal requires a matching external inventory and a verified copy receipt.

The default archive is `~/ARCHIVE/AGENTS-METRICS/2026-09-21-production40`. The local index is `data/maintenance/production40_cleanup`. The `publication` profile uses separate directories with the suffix `-publication`.

The `paper` profile archives the obsolete dated standalone paper folder under the suffix `-paper`. Its selected-path check confirms that the current paper40 sources remain local. `obsolete_helper.json` separately records the archived, unused prose-update helper. The main manuscript is in `paper/manuscript/`.

Commands:

```sh
uv run python code/benchmark/workflows/production40/cleanup.py materialize
uv run python code/benchmark/workflows/production40/cleanup.py inventory
uv run python code/benchmark/workflows/production40/cleanup.py copy
uv run python code/benchmark/workflows/production40/cleanup.py remove-originals
```

For the six historical model collections, add `--profile publication`. That profile rejects current selected records or task inputs within those collections. It preserves candidate run collections and the usage-retry collection. The selected-path check and historical restoration requirements are recorded in `publication_exclusions.json`.

Run `uv run pytest -q code/benchmark/tests/test_production40_cleanup.py` for the temporary-directory verification checks. These tests cover raw bytes, symbolic links, resumed copies, changed sources, and changed archive files.

## Follow-up archive

`prepare_followup.py` records the reviewed obsolete paths and protects the baseline, final selected records, manifest sources, candidate data, and task inputs. Its existing plan is immutable after inventory. `archive_followup.py` supports individual files and directory trees. It defaults to inventory. Copy and removal are separate commands:

```sh
uv run --no-sync python code/benchmark/workflows/production40/archive_followup.py inventory --plan data/maintenance/production40_cleanup_followup/plan.json
uv run --no-sync python code/benchmark/workflows/production40/archive_followup.py copy --plan data/maintenance/production40_cleanup_followup/plan.json
uv run --no-sync python code/benchmark/workflows/production40/archive_followup.py remove --plan data/maintenance/production40_cleanup_followup/plan.json
```

The copy phase verifies every source and destination. Removal requires the verified local and external receipts. Repeated phases verify completed work. Changed files or protected dependencies stop the operation. Restore files from `<archive_root>/payload/<original-relative-path>`. `data/maintenance/production40_cleanup_followup/README.md` records the completed move and rebuild checks.
