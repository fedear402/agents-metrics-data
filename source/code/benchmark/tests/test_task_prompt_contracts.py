from __future__ import annotations

import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

from tasks import TASKS  # noqa: E402


def task_by_id(task_id: str):
    return next(task for task in TASKS if task.task_id == task_id)


def test_dynpanel_prompt_pins_plm_pgmm_options() -> None:
    prompt = task_by_id("dynpanel_ab").prompt

    assert 'effect = "individual"' in prompt
    assert 'model = "onestep"' in prompt
    assert 'transformation = "d"' in prompt
    assert "do not collapse" in prompt.lower()
    assert "Do not use two-step GMM" in prompt


def test_poisson_offset_prompt_matches_scored_reference_contract() -> None:
    prompt = task_by_id("poisson_offset").prompt.lower()

    assert "natural log of exposure" in prompt
    assert "asymptotic model-based standard errors" in prompt


def test_ttest_welch_prompt_pins_df_convention() -> None:
    prompt = task_by_id("ttest_welch").prompt

    assert "Welch-Satterthwaite approximation" in prompt


def test_arima_forecast_prompt_pins_reference_scale_and_drift() -> None:
    prompt = task_by_id("arima_forecast").prompt.lower()

    assert "level series y" in prompt
    assert "no constant or drift term" in prompt
    assert "forecast of y in levels" in prompt


def test_cox_hazard_prompt_does_not_claim_inert_breslow_ties() -> None:
    prompt = task_by_id("cox_hazard").prompt.lower()

    assert "cox proportional-hazards model" in prompt
    assert "breslow" not in prompt
    assert "ties" not in prompt
