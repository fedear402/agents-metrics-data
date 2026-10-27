"""Contract tests for the unified collection runner (``collect.py``).

Offline: no provider calls, all disk fixtures under ``tmp_path``. Covers arm
resolution from the ``ARMS`` table, the arm-root guard, next-free-index
selection, ``--reps-to`` eligibility, prompt-identity refusal, the single
wait-and-reprobe behavior on a provider rejection across all three backends,
deadline logic with an injected clock, balanced-first ordering, shard
disjointness, the corpus scoring post-pass, legacy record tolerance, and the
status output.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis_dataset as ad  # noqa: E402
import collect  # noqa: E402
import strict_posthoc as sp  # noqa: E402

REJECTION_STDOUT = (
    '{"api_error_status":429} You\'ve hit your session limit'
)


def _sonnet_record(cell_id: str, rep: int, **extra):
    record = {
        "cell_id": cell_id,
        "rep": rep,
        "task_id": cell_id.split("__")[0],
        "platform": cell_id.split("__")[1],
        "prompt_mode": cell_id.split("__")[2],
        "exec_mode": cell_id.split("__")[3],
        "model": "claude-sonnet-4-6",
        "claude": {"ok": True},
    }
    record.update(extra)
    return record


def _write_run(root: Path, base: str, rep: int, record=None) -> Path:
    run_dir = root / f"{base}__rep{rep:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if record is not None:
        (run_dir / "run_record.json").write_text(json.dumps(record))
    return run_dir


def _rejection_record(cell_id: str, rep: int):
    return _sonnet_record(
        cell_id,
        rep,
        claude=None,
        claude_attempt_1={
            "ok": False,
            "error": "claude exited 1",
            "stdout_tail": REJECTION_STDOUT,
        },
    )


@pytest.fixture
def few_shot_arm(monkeypatch, tmp_path):
    """The Sonnet few-shot arm re-rooted into tmp_path."""
    spec = dataclasses.replace(
        ad.ARMS["sonnet_few_shot"], path=tmp_path / "sonnet_few_shot"
    )
    monkeypatch.setitem(ad.ARMS, "sonnet_few_shot", spec)
    spec.path.mkdir(parents=True)
    return spec


# --------------------------------------------------------------------------- #
# Arm resolution
# --------------------------------------------------------------------------- #
def test_each_model_prompt_pair_resolves_to_the_arm_table():
    parser = collect.build_parser()
    for model in collect.MODEL_CHOICES:
        for prompt in collect.PROMPT_CHOICES:
            (spec,) = collect.resolve_arms(parser, model, [prompt])
            expected = ad.ARMS[f"{model}_{prompt}"]
            assert spec.path == expected.path
            assert spec.model_id == expected.model_id


def test_unknown_pair_is_a_usage_error(monkeypatch):
    parser = collect.build_parser()
    monkeypatch.delitem(ad.ARMS, "sonnet_few_shot")
    with pytest.raises(SystemExit):
        collect.resolve_arms(parser, "sonnet", ["few_shot"])


def test_both_prompt_modes_build_a_two_root_universe():
    parser = collect.build_parser()
    specs = collect.resolve_arms(parser, "sonnet", ["zero_shot", "few_shot"])
    universe = collect.build_universe(
        specs, ["ols_robust"], ["python"], ["no_feedback"]
    )
    assert {cell.arm_id for cell in universe} == {
        "sonnet_zero_shot",
        "sonnet_few_shot",
    }


def test_agency_names_map_to_the_stored_condition_per_arm():
    mapping = {
        "sonnet_few_shot": {
            "no_feedback": "no_feedback",
            "self_repair": "self_repair",
            "tool_loop": "tool_loop",
        },
        "codex_few_shot": {
            "no_feedback": "codex_generate",
            "self_repair": "codex_repair",
            "tool_loop": "codex_agent",
        },
        "deepseek_few_shot": {
            "no_feedback": "or_generate",
            "self_repair": "or_repair",
            "tool_loop": "or_agent_swe",
        },
    }
    for arm_id, expected in mapping.items():
        for interface, condition in expected.items():
            cell = collect.Cell(arm_id, "ols_robust", "python", interface)
            assert cell.condition == condition


def test_deepseek_tool_loop_prompt_identity_uses_the_miniswe_prompt():
    """The promoted agent condition stores the mini-swe prompt hash, so the
    identity check must rebuild that prompt, never the or_agent one."""
    import hashlib

    import cells as cells_mod
    import harness
    import miniswe_backend

    cell = collect.Cell("deepseek_zero_shot", "ols_robust", "python", "tool_loop")
    assert cell.condition == "or_agent_swe"
    task = cells_mod.TASKS_BY_ID["ols_robust"]
    swe_hash = hashlib.sha256(
        miniswe_backend.swe_tool_loop_prompt(task, "python", "zero_shot")
        .encode("utf-8")
    ).hexdigest()
    or_agent_hash = hashlib.sha256(
        harness.tool_loop_prompt(task, "python", "zero_shot").encode("utf-8")
    ).hexdigest()
    assert collect.expected_prompt_sha256(cell) == swe_hash
    assert swe_hash != or_agent_hash


def test_deepseek_constrained_agent_routes_to_the_swe_backend(monkeypatch):
    """collect resolves everything from the ArmSpec: the agent condition goes
    to the mini-swe path under the whole-run gate; the other two conditions
    stay on the unchanged or paths under the exec gates."""
    seen: dict[str, object] = {}

    def fake_or_cell(task, platform, prompt_mode, condition, rep, model,
                     overwrite, exec_gate=None, runs_dir=None, **kwargs):
        seen.update(
            condition=condition, model=model, prompt_mode=prompt_mode,
            exec_gate=exec_gate, runs_dir=runs_dir,
        )
        return {"cell_id": f"{condition}__rep{rep:02d}"}

    monkeypatch.setattr(collect.cells, "run_openrouter_cell", fake_or_cell)
    collector = collect.Collector([], reps_to=1)

    agent = collect.Cell("deepseek_zero_shot", "ols_robust", "python", "tool_loop")
    collector.run_cell_backend(agent, 0)
    assert seen["condition"] == "or_agent_swe"
    assert seen["model"] == "deepseek/deepseek-v4-flash"
    assert seen["exec_gate"] is collector.swe_gates["python"]
    assert seen["runs_dir"] == ad.ARMS["deepseek_zero_shot"].path

    for interface, condition in (
        ("no_feedback", "or_generate"),
        ("self_repair", "or_repair"),
    ):
        cell = collect.Cell("deepseek_few_shot", "ols_robust", "r", interface)
        collector.run_cell_backend(cell, 0)
        assert seen["condition"] == condition
        assert seen["exec_gate"] is collector.exec_gates["r"]
        assert seen["runs_dir"] == ad.ARMS["deepseek_few_shot"].path


def test_default_task_set_is_the_21_analyzed_tasks():
    assert len(ad.ANALYSIS_TASK_IDS) == 21
    assert set(sp.QUARANTINED_TASKS).isdisjoint(ad.ANALYSIS_TASK_IDS)


def test_quarantined_task_is_refused_without_override():
    with pytest.raises(SystemExit):
        collect.main(["--model", "sonnet", "--tasks", "wcb_few", "--status"])


def test_parser_has_no_runs_dir_and_no_free_form_model():
    parser = collect.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "sonnet", "--runs-dir", "somewhere"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "claude-sonnet-4-6"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "sonnet", "--overwrite"])


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_guard_refuses_a_foreign_arm_marker(few_shot_arm):
    working = few_shot_arm.path / "_working"
    working.mkdir()
    (working / ".pilot_arm").write_text("codex_pilot")
    with pytest.raises(SystemExit, match="codex_pilot"):
        collect.assert_arm_root(few_shot_arm)


def test_guard_refuses_a_record_with_a_conflicting_arm(few_shot_arm):
    base = "ols_robust__python__few_shot__no_feedback"
    _write_run(
        few_shot_arm.path,
        base,
        0,
        _sonnet_record(f"{base}__rep00", 0, arm="codex_pilot"),
    )
    with pytest.raises(SystemExit, match="codex_pilot"):
        collect.assert_arm_root(few_shot_arm)


def test_guard_tolerance_for_missing_arm_stamp_is_a_flag(few_shot_arm):
    base = "ols_robust__python__few_shot__no_feedback"
    _write_run(
        few_shot_arm.path, base, 0, _sonnet_record(f"{base}__rep00", 0)
    )
    collect.assert_arm_root(few_shot_arm, tolerate_missing_arm=True)
    with pytest.raises(SystemExit, match="no arm stamp"):
        collect.assert_arm_root(few_shot_arm, tolerate_missing_arm=False)


def test_guard_refuses_a_push_capable_environment(few_shot_arm, monkeypatch):
    monkeypatch.setenv("AGENTS_METRICS_WORKER_KEY", "wk")
    with pytest.raises(SystemExit, match="AGENTS_METRICS_WORKER_KEY"):
        collect.assert_arm_root(few_shot_arm)


# --------------------------------------------------------------------------- #
# Index selection and eligibility
# --------------------------------------------------------------------------- #
def test_next_free_index_is_one_past_the_highest_stored(few_shot_arm):
    base = "ols_robust__python__few_shot__no_feedback"
    cell = collect.Cell("sonnet_few_shot", "ols_robust", "python", "no_feedback")
    for rep in (0, 1, 5):
        _write_run(
            few_shot_arm.path, base, rep,
            _sonnet_record(f"{base}__rep{rep:02d}", rep),
        )
    status = collect.cell_status(cell)
    assert status.next_rep == 6
    assert status.eligible == 3


def test_a_directory_with_no_record_still_reserves_its_index(few_shot_arm):
    base = "ols_robust__python__few_shot__no_feedback"
    cell = collect.Cell("sonnet_few_shot", "ols_robust", "python", "no_feedback")
    _write_run(few_shot_arm.path, base, 0, _sonnet_record(f"{base}__rep00", 0))
    _write_run(few_shot_arm.path, base, 1, record=None)  # interrupted run
    status = collect.cell_status(cell)
    assert status.next_rep == 2
    assert status.eligible == 1


def test_a_rejection_occupies_its_index_but_does_not_count(few_shot_arm):
    base = "ols_robust__python__few_shot__no_feedback"
    cell = collect.Cell("sonnet_few_shot", "ols_robust", "python", "no_feedback")
    _write_run(few_shot_arm.path, base, 0, _sonnet_record(f"{base}__rep00", 0))
    _write_run(
        few_shot_arm.path, base, 1, _rejection_record(f"{base}__rep01", 1)
    )
    status = collect.cell_status(cell)
    assert status.eligible == 1
    assert status.rejections == 1
    # The next round collects the still-short cell at the following index.
    assert status.next_rep == 2


def test_reps_to_skips_full_cells_but_not_cells_holding_a_rejection(few_shot_arm):
    full = "ols_robust__python__few_shot__no_feedback"
    short = "ols_robust__r__few_shot__no_feedback"
    for rep in (0, 1):
        _write_run(
            few_shot_arm.path, full, rep,
            _sonnet_record(f"{full}__rep{rep:02d}", rep),
        )
    _write_run(few_shot_arm.path, short, 0, _sonnet_record(f"{short}__rep00", 0))
    _write_run(few_shot_arm.path, short, 1, _rejection_record(f"{short}__rep01", 1))

    full_cell = collect.Cell(
        "sonnet_few_shot", "ols_robust", "python", "no_feedback"
    )
    short_cell = collect.Cell(
        "sonnet_few_shot", "ols_robust", "r", "no_feedback"
    )
    target = 2
    assert collect.cell_status(full_cell).eligible >= target
    assert collect.cell_status(short_cell).eligible < target


# --------------------------------------------------------------------------- #
# Prompt identity
# --------------------------------------------------------------------------- #
def test_a_mismatched_prompt_hash_refuses_the_cell(few_shot_arm, capsys):
    base = "ols_robust__python__few_shot__no_feedback"
    cell = collect.Cell("sonnet_few_shot", "ols_robust", "python", "no_feedback")
    _write_run(
        few_shot_arm.path, base, 0,
        _sonnet_record(f"{base}__rep00", 0, prompt_sha256="deadbeef"),
    )
    collector = collect.Collector([cell], reps_to=2)
    status = collect.cell_status(cell)

    assert collector.prompt_identity_ok(cell, status) is False
    assert cell in collector.refused
    out = capsys.readouterr().out
    assert "deadbeef" in out
    assert collect.expected_prompt_sha256(cell) in out


def test_legacy_records_with_no_hash_do_not_block(few_shot_arm):
    base = "ols_robust__python__few_shot__no_feedback"
    cell = collect.Cell("sonnet_few_shot", "ols_robust", "python", "no_feedback")
    _write_run(few_shot_arm.path, base, 0, _sonnet_record(f"{base}__rep00", 0))
    collector = collect.Collector([cell], reps_to=2)
    assert collector.prompt_identity_ok(cell, collect.cell_status(cell)) is True


def test_a_matching_prompt_hash_is_accepted(few_shot_arm):
    base = "ols_robust__python__few_shot__no_feedback"
    cell = collect.Cell("sonnet_few_shot", "ols_robust", "python", "no_feedback")
    _write_run(
        few_shot_arm.path, base, 0,
        _sonnet_record(
            f"{base}__rep00", 0,
            prompt_sha256=collect.expected_prompt_sha256(cell),
        ),
    )
    collector = collect.Collector([cell], reps_to=2)
    assert collector.prompt_identity_ok(cell, collect.cell_status(cell)) is True


# --------------------------------------------------------------------------- #
# One wait-and-reprobe behavior on a provider rejection
# --------------------------------------------------------------------------- #
def _run_rejection_sequence(monkeypatch, arm_id, rejections=2):
    """Drive ``fill_one`` with a backend that rejects N times then answers."""
    cell = collect.Cell(arm_id, "ols_robust", "python", "no_feedback")
    calls: list[tuple[str, int]] = []
    sleeps: list[float] = []
    outcomes = [True] * rejections + [False]

    def fake_backend(self, target_cell, rep):
        calls.append((target_cell.label(), rep))
        rejected = outcomes.pop(0)
        record = {"cell_id": f"rep{rep:02d}"}
        if rejected:
            record["provider_limit"] = True
            record["failure_reason"] = "provider limit"
        else:
            record["execution_success"] = True
        return record

    monkeypatch.setattr(collect.Collector, "run_cell_backend", fake_backend)
    monkeypatch.setattr(collect, "meter_percentages", lambda spec, probes: (0.0, 0.0))
    collector = collect.Collector(
        [cell], reps_to=5, meter_wait_seconds=7.0, sleep_func=sleeps.append
    )
    status = collect.CellStatus(stored_reps={0, 1, 2})  # next index 3
    collector.fill_one(cell, status)
    return calls, sleeps, collector


@pytest.mark.parametrize(
    "arm_id", ["sonnet_few_shot", "codex_few_shot", "deepseek_few_shot"]
)
def test_rejection_waits_reprobes_and_retries_the_same_cell(monkeypatch, arm_id):
    calls, sleeps, collector = _run_rejection_sequence(monkeypatch, arm_id)
    # Same cell throughout, at consecutive following indexes.
    assert [rep for _, rep in calls] == [3, 4, 5]
    assert len({label for label, _ in calls}) == 1
    # One wait per rejection before the retry; never an exit.
    assert sleeps == [7.0, 7.0]
    # The stored rejections and the final answer are all recorded.
    assert collector.collected == 3


def test_rejection_records_are_kept_not_deleted(monkeypatch):
    calls, _, collector = _run_rejection_sequence(
        monkeypatch, "sonnet_few_shot", rejections=1
    )
    assert [rep for _, rep in calls] == [3, 4]
    with collector.lock:
        rows = list(collector.summary)
    assert len(rows) == 2  # the rejection row stays in the summary


# --------------------------------------------------------------------------- #
# Deadline and ordering
# --------------------------------------------------------------------------- #
def test_deadline_with_injected_clock():
    import datetime as dt

    noon = dt.datetime(2026, 8, 18, 12, 0, 0).timestamp()
    past = collect.Deadline("11:30", clock=lambda: noon)
    future = collect.Deadline("12:30", clock=lambda: noon)
    assert past.reached() is True
    assert future.reached() is False
    assert collect.Deadline(None, clock=lambda: noon).reached() is False
    with pytest.raises(SystemExit):
        collect.Deadline("25:99", clock=lambda: noon)


def test_a_reached_deadline_collects_nothing_and_exits_zero(few_shot_arm):
    cell = collect.Cell("sonnet_few_shot", "ols_robust", "python", "no_feedback")
    deadline = collect.Deadline("10:00", clock=lambda: 1.0e10)
    deadline.stamp = 0.0  # already past under the injected clock
    collector = collect.Collector([cell], reps_to=2, deadline=deadline)
    assert collector.run() == 0
    assert collector.collected == 0
    assert list(few_shot_arm.path.iterdir()) == []


def test_deadline_stops_the_shard_between_runs(monkeypatch, few_shot_arm):
    """The clock is checked before each run: the second run never starts."""
    cell_a = collect.Cell("sonnet_few_shot", "ols_robust", "python", "no_feedback")
    cell_b = collect.Cell("sonnet_few_shot", "ols_robust", "r", "no_feedback")
    now = {"t": 0.0}
    deadline = collect.Deadline(None, clock=lambda: now["t"])
    deadline.stamp = 10.0
    calls = []

    def fake_backend(self, cell, rep):
        calls.append(cell.label())
        now["t"] = 11.0  # the deadline passes during the first run
        base = f"{cell.task_id}__{cell.platform}__few_shot__{cell.condition}"
        record = _sonnet_record(f"{base}__rep{rep:02d}", rep)
        _write_run(few_shot_arm.path, base, rep, record)
        return record

    monkeypatch.setattr(collect.Collector, "run_cell_backend", fake_backend)
    collector = collect.Collector(
        [cell_a, cell_b], reps_to=1, deadline=deadline, sleep_func=lambda s: None
    )
    collector.run()
    assert len(calls) == 1


def test_balanced_first_fills_only_the_lowest_level_each_round(
    monkeypatch, few_shot_arm
):
    base_b = "ols_robust__r__few_shot__no_feedback"
    _write_run(
        few_shot_arm.path, base_b, 0, _sonnet_record(f"{base_b}__rep00", 0)
    )  # cell B starts at 1 eligible; cell A at 0
    cell_a = collect.Cell("sonnet_few_shot", "ols_robust", "python", "no_feedback")
    cell_b = collect.Cell("sonnet_few_shot", "ols_robust", "r", "no_feedback")
    order = []

    def fake_backend(self, cell, rep):
        order.append((cell.platform, rep))
        base = f"{cell.task_id}__{cell.platform}__few_shot__{cell.condition}"
        record = _sonnet_record(f"{base}__rep{rep:02d}", rep)
        _write_run(few_shot_arm.path, base, rep, record)
        return record

    monkeypatch.setattr(collect.Collector, "run_cell_backend", fake_backend)
    collector = collect.Collector(
        [cell_a, cell_b], reps_to=2, sleep_func=lambda s: None
    )
    collector.shard_loop(0, [cell_a, cell_b])

    # Round one touches only the emptier cell A; then both rise to the target.
    assert order[0] == ("python", 0)
    assert sorted(order) == [("python", 0), ("python", 1), ("r", 1)]
    assert collect.cell_status(cell_a).eligible == 2
    assert collect.cell_status(cell_b).eligible == 2


def test_shards_are_disjoint_and_cover_the_universe(few_shot_arm):
    universe = collect.build_universe(
        [few_shot_arm],
        ["ols_robust", "probit"],
        ["python", "r", "stata"],
        ["no_feedback"],
    )
    collector = collect.Collector(universe, reps_to=1, shards=4)
    shards = collector.split_shards()
    seen: list[collect.Cell] = []
    for shard in shards:
        seen.extend(shard)
    assert len(seen) == len(universe)
    assert set(seen) == set(universe)
    assert len(shards) == 4


# --------------------------------------------------------------------------- #
# Scoring post-pass
# --------------------------------------------------------------------------- #
def test_new_corpus_runs_receive_a_score_sidecar(monkeypatch, tmp_path):
    spec = dataclasses.replace(
        ad.ARMS["sonnet_zero_shot"], path=tmp_path / "sonnet_zero_shot"
    )
    monkeypatch.setitem(ad.ARMS, "sonnet_zero_shot", spec)
    spec.path.mkdir(parents=True)
    cell = collect.Cell("sonnet_zero_shot", "ols_robust", "python", "no_feedback")
    base = "ols_robust__python__zero_shot__no_feedback"
    record = _sonnet_record(f"{base}__rep00", 0, prompt_mode="zero_shot")
    run_dir = _write_run(spec.path, base, 0, record)

    collector = collect.Collector([cell], reps_to=1)
    collector.new_run_dirs.append(("sonnet_zero_shot", run_dir))
    written = collector.score_new_corpus_runs()

    assert written == 1
    score = json.loads((run_dir / "score.json").read_text())
    assert score["task_id"] == "ols_robust"

    # A non-corpus arm gets no sidecar from the post-pass.
    collector.new_run_dirs[:] = [("sonnet_few_shot", run_dir)]
    (run_dir / "score.json").unlink()
    assert collector.score_new_corpus_runs() == 0


# --------------------------------------------------------------------------- #
# Legacy record tolerance
# --------------------------------------------------------------------------- #
def test_truncated_stub_record_is_occupied_not_missing(monkeypatch, tmp_path):
    """The nine-key stub of design 4.5: no model, no provider block, a
    surviving result.json. Its index must never be re-collected."""
    spec = dataclasses.replace(
        ad.ARMS["sonnet_zero_shot"], path=tmp_path / "sonnet_zero_shot"
    )
    monkeypatch.setitem(ad.ARMS, "sonnet_zero_shot", spec)
    spec.path.mkdir(parents=True)
    base = "ols_robust__r__zero_shot__tool_loop"
    stub = {
        "cell_id": f"{base}__rep01",
        "cell_id_base": "ols_robust__r__tool_loop",
        "rep": 1,
        "task_id": "ols_robust",
        "platform": "r",
        "prompt_mode": "zero_shot",
        "exec_mode": "tool_loop",
        "error": "harness exception: TypeError(...)",
        "execution_success": False,
    }
    run_dir = _write_run(spec.path, base, 1, stub)
    (run_dir / "result.json").write_text(json.dumps({
        "task_id": "ols_robust", "platform": "r", "estimates": [],
        "method_metadata": {}, "diagnostics": {},
    }))

    cell = collect.Cell("sonnet_zero_shot", "ols_robust", "r", "tool_loop")
    status = collect.cell_status(cell)
    assert 1 in status.stored_reps
    assert 1 in status.eligible_reps  # result_json_rescue
    assert status.next_rep == 2


def test_rejection_stored_in_an_attempt_block_is_found(few_shot_arm):
    """Eligibility reads ``harness.claude_blocks``: a rejection hidden in
    ``claude_attempt_1`` (how the 2026-08-17 records hid) is detected."""
    base = "ols_robust__stata__few_shot__self_repair"
    _write_run(
        few_shot_arm.path, base, 0, _rejection_record(f"{base}__rep00", 0)
    )
    cell = collect.Cell("sonnet_few_shot", "ols_robust", "stata", "self_repair")
    status = collect.cell_status(cell)
    assert status.rejections == 1
    assert status.eligible == 0


def test_collect_never_reads_cell_id_base_or_infers_prompt_from_paths():
    source = (
        Path(__file__).resolve().parents[1] / "collect.py"
    ).read_text()
    assert "cell_id_base" not in source
    assert 'get("prompt_mode")' not in source
    assert "get('prompt_mode')" not in source


# --------------------------------------------------------------------------- #
# Status and dry run
# --------------------------------------------------------------------------- #
def test_status_output_lists_cells_emptiest_first(few_shot_arm, capsys):
    base_full = "ols_robust__python__few_shot__no_feedback"
    for rep in (0, 1):
        _write_run(
            few_shot_arm.path, base_full, rep,
            _sonnet_record(f"{base_full}__rep{rep:02d}", rep),
        )
    base_short = "ols_robust__r__few_shot__no_feedback"
    _write_run(
        few_shot_arm.path, base_short, 0,
        _rejection_record(f"{base_short}__rep00", 0),
    )
    universe = [
        collect.Cell("sonnet_few_shot", "ols_robust", "python", "no_feedback"),
        collect.Cell("sonnet_few_shot", "ols_robust", "r", "no_feedback"),
    ]
    assert collect.print_status(universe) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "sonnet_few_shot ols_robust r chatbot eligible=0 next=1 "
        "rejections=1 missing_score=0"
    )
    assert lines[1] == (
        "sonnet_few_shot ols_robust python chatbot eligible=2 next=2 "
        "rejections=0 missing_score=0"
    )
    assert lines[2].startswith("# sonnet_few_shot: 2 cells, 2 eligible runs")


def test_main_dry_run_prints_the_plan_and_makes_no_call(
    monkeypatch, tmp_path, capsys
):
    for arm_id in ("sonnet_zero_shot", "sonnet_few_shot"):
        spec = dataclasses.replace(ad.ARMS[arm_id], path=tmp_path / arm_id)
        monkeypatch.setitem(ad.ARMS, arm_id, spec)
        spec.path.mkdir(parents=True)
    rc = collect.main([
        "--model", "sonnet", "--tasks", "ols_robust", "--language", "python",
        "--agency", "chatbot", "--reps-to", "1", "--dry-run",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "arm sonnet_zero_shot" in out
    assert "arm sonnet_few_shot" in out
    assert "claude-sonnet-4-6" in out
    assert "DRY" in out
    # No run directory was created by the dry run.
    assert list((tmp_path / "sonnet_zero_shot").iterdir()) == []
    assert list((tmp_path / "sonnet_few_shot").iterdir()) == []
