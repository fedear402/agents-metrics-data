from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import strict_posthoc  # noqa: E402
from tasks import TASKS  # noqa: E402


def task_by_id(task_id: str):
    return next(task for task in TASKS if task.task_id == task_id)


def test_record_cost_is_missing_when_no_claude_block_reports_cost() -> None:
    record = {
        "claude": {
            "ok": False,
            "timed_out": True,
            "total_cost_usd": None,
        }
    }

    assert strict_posthoc.record_cost(record) is None


def test_record_cost_sums_only_blocks_with_recorded_cost() -> None:
    record = {
        "claude_attempt_1": {"total_cost_usd": 0.12},
        "claude_attempt_2": {"total_cost_usd": None},
        "claude_attempt_3": {"total_cost_usd": 0.08},
    }

    assert strict_posthoc.record_cost(record) == 0.20


def test_record_cost_recovers_structured_agent_result(tmp_path: Path) -> None:
    (tmp_path / "agent_stream.jsonl").write_text(
        "not json\n"
        + json.dumps({"type": "assistant", "total_cost_usd": 9.0})
        + "\n"
        + json.dumps({"type": "result", "total_cost_usd": 0.48})
        + "\n"
    )

    assert strict_posthoc.record_cost({}, tmp_path) == 0.48


def test_record_cost_preserves_explicit_zero(tmp_path: Path) -> None:
    (tmp_path / "agent_stream.jsonl").write_text(
        json.dumps({"type": "result", "total_cost_usd": 0.48}) + "\n"
    )

    assert strict_posthoc.record_cost({"claude": {"total_cost_usd": 0.0}}, tmp_path) == 0.0


def test_numeric_gate_requires_se_when_reference_has_se() -> None:
    task = task_by_id("ols_robust")
    references = strict_posthoc.load_references()
    estimates = {
        "const": {"term": "const", "estimate": 1.0115765173916533, "se": None, "n_obs": 2000},
        "x1": {"term": "x1", "estimate": 0.4926639840880423, "se": None, "n_obs": 2000},
        "x2": {"term": "x2", "estimate": -0.32191134940720445, "se": None, "n_obs": 2000},
    }

    passed, per_term, notes = strict_posthoc.compare_numeric(estimates, task, references)

    assert passed is False
    assert per_term["x1"]["se_pass"] is False
    assert any("missing finite standard error" in note for note in notes)


def test_cooks_distance_obs_ids_require_exact_integer_match() -> None:
    task = task_by_id("cooks_distance")
    references = strict_posthoc.load_references()
    ref_rows = references[task.reference_key]["estimates"]
    estimates = {
        row["term"]: {
            "term": row["term"],
            "estimate": row["estimate"],
            "se": row["se"],
            "n_obs": row["n_obs"],
        }
        for row in ref_rows
    }
    for term in list(estimates):
        if term.endswith("_obs_id"):
            estimates[term]["estimate"] += 1

    passed, per_term, notes = strict_posthoc.compare_numeric(
        estimates, task, references
    )

    assert passed is False
    assert per_term["rank1_obs_id"]["coef_pass"] is False
    assert any("rank1_obs_id: estimate outside tolerance" in note for note in notes)


def test_wcb_few_p_value_scoring_follows_quarantine() -> None:
    task = task_by_id("wcb_few")
    references = strict_posthoc.load_references()
    ref_rows = references[task.reference_key]["estimates"]
    estimates = {
        row["term"]: {
            "term": row["term"],
            "estimate": row["estimate"],
            "se": row["se"],
            "n_obs": row["n_obs"],
        }
        for row in ref_rows
    }
    estimates["p_wcb"]["estimate"] = 0.0

    # Legacy-wave wcb_few runs (records without prompt_sha256) answered the old
    # B=999 contract: p_wcb is skipped, so a value from the old contract cannot
    # fail the run against the new exact-enumeration reference.
    assert "wcb_few" in strict_posthoc.QUARANTINED_TASKS
    passed, per_term, notes = strict_posthoc.compare_numeric(
        estimates, task, references, legacy_quarantined=True
    )
    assert passed is True
    assert per_term["p_wcb"] == {"skipped": True}

    # Respecified-wave runs (records carrying prompt_sha256) are scored against
    # the exact-enumeration reference even while the task stays in
    # QUARANTINED_TASKS for its legacy records.
    passed, per_term, notes = strict_posthoc.compare_numeric(
        estimates, task, references, legacy_quarantined=False
    )
    assert passed is False
    assert per_term["p_wcb"]["coef_pass"] is False
    assert any("p_wcb: estimate outside tolerance" in note for note in notes)


def test_legacy_wave_detection_uses_prompt_sha256() -> None:
    assert strict_posthoc.record_is_legacy_wave({}) is True
    assert strict_posthoc.record_is_legacy_wave({"prompt_sha256": None}) is True
    assert strict_posthoc.record_is_legacy_wave({"prompt_sha256": "ab" * 32}) is False


def test_var_granger_accepts_documented_system_df_den_convention() -> None:
    task = task_by_id("var_granger")
    references = strict_posthoc.load_references()
    ref_rows = references[task.reference_key]["estimates"]
    estimates = {
        row["term"]: {
            "term": row["term"],
            "estimate": row["estimate"],
            "se": row["se"],
            "n_obs": row["n_obs"],
        }
        for row in ref_rows
    }
    estimates["df_den"]["estimate"] *= 2

    passed, per_term, notes = strict_posthoc.compare_numeric(
        estimates, task, references
    )

    assert passed is True, notes
    assert per_term["df_den"]["coef_pass"] is True


def test_var_granger_rejects_undocumented_half_df_den_convention() -> None:
    task = task_by_id("var_granger")
    references = strict_posthoc.load_references()
    ref_rows = references[task.reference_key]["estimates"]
    estimates = {
        row["term"]: {
            "term": row["term"],
            "estimate": row["estimate"],
            "se": row["se"],
            "n_obs": row["n_obs"],
        }
        for row in ref_rows
    }
    estimates["df_den"]["estimate"] *= 0.5

    passed, per_term, notes = strict_posthoc.compare_numeric(
        estimates, task, references
    )

    assert passed is False
    assert per_term["df_den"]["coef_pass"] is False
    assert any("df_den: estimate outside tolerance" in note for note in notes)


def test_artifact_gate_rejects_extra_terms() -> None:
    task = task_by_id("ttest_welch")
    result = {
        "task_id": "ttest_welch",
        "platform": "python",
        "estimates": [
            {"term": "t", "estimate": 1.0, "se": None, "n_obs": 200},
            {"term": "df", "estimate": 10.0, "se": None, "n_obs": 200},
            {"term": "p_value", "estimate": 0.1, "se": None, "n_obs": 200},
            {"term": "extra", "estimate": 0.0, "se": None, "n_obs": 200},
        ],
        "method_metadata": {"estimator": "Welch t-test"},
        "diagnostics": {},
    }

    passed, _estimates, notes = strict_posthoc.score_artifact(result, task, "python")

    assert passed is False
    assert any("extra terms" in note for note in notes)


def test_integrity_gate_flags_hardcoded_reference_number(tmp_path: Path) -> None:
    task = task_by_id("ols_robust")
    references = strict_posthoc.load_references()
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir()
    run_dir.mkdir()
    (data_dir / task.data_file).write_text("x,y\n1,2\n")
    (run_dir / "input.csv").write_text("x,y\n1,2\n")
    script = run_dir / "run.py"
    script.write_text("result = 0.4926639841\n")

    passed, notes = strict_posthoc.score_integrity(
        task, run_dir, script, references, data_dir
    )

    assert passed is False
    assert any("reference numeric literal" in note for note in notes)


def test_method_gate_accepts_ols_hc1_evidence(tmp_path: Path) -> None:
    script = tmp_path / "run.py"
    script.write_text("fit = sm.OLS(y, X).fit(cov_type='HC1')\n")

    passed, notes = strict_posthoc.score_method("ols_robust", script.read_text(), script)

    assert passed is True
    assert notes == []


def test_method_gate_accepts_cox_without_tie_keyword_for_tie_free_task(
    tmp_path: Path,
) -> None:
    script = tmp_path / "run.py"
    script.write_text(
        "from statsmodels.duration.hazard_regression import PHReg\n"
        "fit = PHReg(df['t_obs'], df[['treat', 'x']], status=df['event']).fit()\n"
    )

    passed, notes = strict_posthoc.score_method("cox_hazard", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_accepts_panelols_twoway_fe(tmp_path: Path) -> None:
    # Regression for the 2026-06-15 fix: PanelOLS with entity/time effects is a
    # valid two-way FE implementation and must satisfy specification correctness.
    script = tmp_path / "run.py"
    script.write_text(
        "from linearmodels import PanelOLS  # panel by firm, year\n"
        "m = PanelOLS(y, X, entity_effects=True, time_effects=True)\n"
        "res = m.fit(cov_type='clustered', clusters=df['firm'])\n"
    )

    passed, notes = strict_posthoc.score_method("twoway_fe", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_accepts_areg_twoway_fe(tmp_path: Path) -> None:
    script = tmp_path / "run.do"
    script.write_text("areg y x i.year, absorb(firm) vce(cluster firm)\n")

    passed, notes = strict_posthoc.score_method("twoway_fe", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_rejects_pooled_ols_for_twoway_fe(tmp_path: Path) -> None:
    # Guard against over-broadening: pooled OLS (no FE) must still fail even when
    # firm, year, and clustering are present.
    script = tmp_path / "run.py"
    script.write_text(
        "# variables: firm, year\n"
        "fit = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['firm']})\n"
    )

    passed, notes = strict_posthoc.score_method("twoway_fe", script.read_text(), script)

    assert passed is False
    assert any("group 1" in note for note in notes)


def test_method_gate_accepts_local_projection_via_shift(tmp_path: Path) -> None:
    # Regression for the 2026-06-15 fix: LP as per-horizon OLS of a led outcome
    # (shift(-h)) with Newey-West HAC lag 4 must satisfy specification correctness.
    script = tmp_path / "run.py"
    script.write_text(
        "for h in range(0, 5):\n"
        "    df['y_lead'] = df['y'].shift(-h)\n"
        "    fit = sm.OLS(yv, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})  # lag 4\n"
    )

    passed, notes = strict_posthoc.score_method("lp_irf", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_rejects_task_id_only_for_classical_tasks() -> None:
    cases = {
        "vif": '{"task_id": "vif", "term": "vif_x1"}',
        "cooks_distance": '{"task_id": "cooks_distance", "term": "rank1_cooks_d"}',
        "anova_oneway": '{"task_id": "anova_oneway", "term": "F"}',
        "chisq_indep": '{"task_id": "chisq_indep", "term": "chi2"}',
    }

    for task_id, evidence in cases.items():
        passed, notes = strict_posthoc.score_method(task_id, evidence, None)
        assert passed is False, (task_id, notes)


def test_method_gate_accepts_valid_classical_task_methods(tmp_path: Path) -> None:
    valid_scripts = {
        "vif": "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
        "cooks_distance": "infl = sm.OLS(y, X).fit().get_influence().cooks_distance\n",
        "anova_oneway": "fit <- lm(y ~ factor(group), data = df)\nav <- anova(fit)\n",
        "chisq_indep": "test <- chisq.test(table(df$row, df$col), correct = FALSE)\n",
    }

    for task_id, script_text in valid_scripts.items():
        script = tmp_path / f"{task_id}.txt"
        script.write_text(script_text)
        passed, notes = strict_posthoc.score_method(task_id, script_text, script)
        assert passed is True, (task_id, notes)


def test_method_gate_accepts_manual_classical_task_methods(tmp_path: Path) -> None:
    valid_scripts = {
        "vif": "vif_vals <- 1 / (1 - summary(fit_j)$r.squared)\n",
        "cooks_distance": "generate double _cooks_d = (resid^2 / (p * mse)) * (lev / (1 - lev)^2)\n",
        "anova_oneway": "local gvar group\nanova y `gvar'\n",
        "chisq_indep": "tab row col, chi2\n",
    }

    for task_id, script_text in valid_scripts.items():
        script = tmp_path / f"{task_id}_manual.txt"
        script.write_text(script_text)
        passed, notes = strict_posthoc.score_method(task_id, script_text, script)
        assert passed is True, (task_id, notes)


def test_method_gate_rejects_numeric_trend_for_anova_oneway(tmp_path: Path) -> None:
    script = tmp_path / "run.R"
    script.write_text(
        'out <- list(task_id = "anova_oneway")\n'
        "fit <- lm(y ~ group, data = df)\n"
        "av <- anova(fit)\n"
    )

    passed, notes = strict_posthoc.score_method("anova_oneway", script.read_text(), script)

    assert passed is False
    assert any("missing method evidence group 1" in note for note in notes)


def test_method_gate_accepts_rdd_stata_macro_cutoff(tmp_path: Path) -> None:
    script = tmp_path / "run.do"
    script.write_text(
        "local c = 0\n"
        "local h = 0.5\n"
        "gen inband = abs(z - `c') <= `h'\n"
        "gen w = 1 - abs(z - `c') / `h'\n"
        "reg y z [aw=w] if z < `c' & inband\n"
        "reg y z [aw=w] if z >= `c' & inband\n"
    )

    passed, notes = strict_posthoc.score_method("rdd_sharp", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_accepts_rdd_stata_centered_running_variable(
    tmp_path: Path,
) -> None:
    script = tmp_path / "run.do"
    script.write_text(
        "local h = 0.5\n"
        "gen z_c = z - 0\n"
        "gen kw = 1 - abs(z_c) / `h'\n"
        "reg y z_c [aw=kw] if z_c < 0\n"
        "reg y z_c [aw=kw] if z_c >= 0\n"
    )

    passed, notes = strict_posthoc.score_method("rdd_sharp", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_accepts_rdd_stata_renamed_running_variable(
    tmp_path: Path,
) -> None:
    script = tmp_path / "run.do"
    script.write_text(
        "local h = 0.5\n"
        "generate double running = z\n"
        "generate double w = 1 - abs(running) / `h'\n"
        "regress y c.running [aw=w] if running < 0\n"
        "regress y c.running [aw=w] if running >= 0\n"
    )

    passed, notes = strict_posthoc.score_method("rdd_sharp", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_accepts_rdd_stata_interacted_running_variable(
    tmp_path: Path,
) -> None:
    script = tmp_path / "run.do"
    script.write_text(
        "local h = 0.5\n"
        "generate double running = z\n"
        "generate byte right = running >= 0\n"
        "generate double w = 1 - abs(running) / `h'\n"
        "regress y c.running##i.right [aw=w]\n"
    )

    passed, notes = strict_posthoc.score_method("rdd_sharp", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_accepts_manual_stata_welch_calculation(tmp_path: Path) -> None:
    script = tmp_path / "run.do"
    script.write_text(
        "* Welch two-sample t-test with unequal variances\n"
        "summarize y if group == 1\n"
        "local v1 = r(Var)\n"
        "local n1 = r(N)\n"
        "summarize y if group == 0\n"
        "local v0 = r(Var)\n"
        "local n0 = r(N)\n"
        "local se = sqrt(`v1'/`n1' + `v0'/`n0')\n"
        "local t = (`m1' - `m0') / `se'\n"
        "local den = (`v1'/`n1')^2/(`n1'-1) + (`v0'/`n0')^2/(`n0'-1)\n"
        "local df = (`v1'/`n1' + `v0'/`n0')^2 / `den'\n"
        "local p = 2 * ttail(`df', abs(`t'))\n"
    )

    passed, notes = strict_posthoc.score_method(
        "ttest_welch", script.read_text(), script
    )

    assert passed is True, notes


def test_method_gate_accepts_manual_python_difference_gmm(tmp_path: Path) -> None:
    script = tmp_path / "run.py"
    script.write_text(
        "# Arellano-Bond one-step difference GMM\n"
        "dy = np.diff(y)\n"
        "yl_diff = dy[:-1]\n"
        "for l in range(t - 2):\n"
        "    Zi[row, col_off + l] = y[l]\n"
        "AN_inv = np.linalg.pinv(sum(Zi.T @ Hi @ Zi for Zi, Hi in blocks))\n"
        "theta = np.linalg.solve(X.T @ Z @ AN_inv @ Z.T @ X, rhs)\n"
    )

    passed, notes = strict_posthoc.score_method(
        "dynpanel_ab", script.read_text(), script
    )

    assert passed is True, notes


def test_method_gate_rejects_incomplete_staggered_did_script(tmp_path: Path) -> None:
    script = tmp_path / "run.py"
    script.write_text(
        "df = pd.read_csv('input.csv')\n"
        "print(df['cohort'].value_counts())\n"
    )

    passed, notes = strict_posthoc.score_method(
        "staggered_did", script.read_text(), script
    )

    assert passed is False
    assert notes


def test_method_gate_accepts_stata_oirf_for_var_irf(tmp_path: Path) -> None:
    script = tmp_path / "run.do"
    script.write_text(
        "tsset t\n"
        "var y1 y2, lags(1/2) dfk\n"
        "irf create bench, replace step(4)\n"
        "irf table oirf, impulse(y2) response(y1) step(4)\n"
    )

    passed, notes = strict_posthoc.score_method("var_irf", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_ignores_staggered_did_reject_words_in_comments(tmp_path: Path) -> None:
    script = tmp_path / "run.py"
    script.write_text(
        "# Do not estimate TWFE here, and do not bootstrap this ATT.\n"
        "never = df['cohort'].eq(-1)\n"
        "comparison = df.loc[never]\n"
        "att_cells = df.groupby(['cohort', 'time'])['y'].mean()\n"
    )

    passed, notes = strict_posthoc.score_method("staggered_did", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_rejects_bare_f_lag_evidence_for_var_granger(tmp_path: Path) -> None:
    script = tmp_path / "run.py"
    script.write_text(
        "lag = 2\n"
        "y1 = df['y1']\n"
        "y2 = df['y2']\n"
        "F = 0.0\n"
    )

    passed, notes = strict_posthoc.score_method("var_granger", script.read_text(), script)

    assert passed is False
    assert any("missing method evidence group 1" in note for note in notes)


def test_method_gate_accepts_stata_restriction_test_for_var_granger(
    tmp_path: Path,
) -> None:
    script = tmp_path / "run.do"
    script.write_text(
        "tsset t\n"
        "regress y1 L.y1 L2.y1 L.y2 L2.y2\n"
        "test L.y2 L2.y2\n"
        "local F_val = r(F)\n"
        "local df_den = r(df_r)\n"
    )

    passed, notes = strict_posthoc.score_method("var_granger", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_accepts_manual_lag_restriction_for_var_granger(
    tmp_path: Path,
) -> None:
    script = tmp_path / "run.do"
    script.write_text(
        "gen y2_lag1 = y2[_n-1]\n"
        "gen y2_lag2 = y2[_n-2]\n"
        "regress y1 y1_lag1 y1_lag2 y2_lag1 y2_lag2\n"
        "test y2_lag1 y2_lag2\n"
        "local F_stat = r(F)\n"
    )

    passed, notes = strict_posthoc.score_method("var_granger", script.read_text(), script)

    assert passed is True, notes


def test_method_gate_accepts_short_lag_names_for_var_granger(
    tmp_path: Path,
) -> None:
    script = tmp_path / "run.do"
    script.write_text(
        "gen y2_l1 = y2[_n-1]\n"
        "gen y2_l2 = y2[_n-2]\n"
        "regress y1 y1_l1 y1_l2 y2_l1 y2_l2\n"
        "test y2_l1 y2_l2\n"
        "local F_stat = r(F)\n"
    )

    passed, notes = strict_posthoc.score_method("var_granger", script.read_text(), script)

    assert passed is True, notes


def test_build_report_scores_minimal_successful_run(tmp_path: Path) -> None:
    task = task_by_id("ols_robust")
    references = strict_posthoc.load_references()
    ref_rows = references[task.reference_key]["estimates"]
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "runs" / "cell"
    data_dir.mkdir()
    run_dir.mkdir(parents=True)
    (data_dir / task.data_file).write_text("x,y\n1,2\n")
    (run_dir / "input.csv").write_text("x,y\n1,2\n")
    (run_dir / "run.py").write_text("fit = sm.OLS(y, X).fit(cov_type='HC1')\n")
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "platform": "python",
                "estimates": ref_rows,
                "method_metadata": {"estimator": "OLS", "vcov": "HC1"},
                "diagnostics": {},
            }
        )
    )
    record = {
        "cell_id": "cell",
        "task_id": task.task_id,
        "platform": "python",
        "exec_mode": "tool_loop",
        "prompt_mode": "zero_shot",
        "rep": 0,
        "claude": {"ok": True, "total_cost_usd": 0.1, "usage": {"input_tokens": 1}},
        "execution_success": True,
    }

    report = strict_posthoc.build_report([(run_dir, record)], references=references, data_dir=data_dir)

    assert report["analysis_summary"]["strict_success"] == 1
    assert report["by_exec_mode"]["tool_loop"]["strict_success"] == 1


def test_build_report_quarantines_wcb_few_from_headline_aggregates(tmp_path: Path) -> None:
    task = task_by_id("wcb_few")
    references = strict_posthoc.load_references()
    ref_rows = json.loads(json.dumps(references[task.reference_key]["estimates"]))
    for row in ref_rows:
        if row["term"] == "p_wcb":
            row["estimate"] = 0.033  # legacy B=999 contract answer
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "runs" / "cell"
    data_dir.mkdir()
    run_dir.mkdir(parents=True)
    (data_dir / task.data_file).write_text("x,y,cluster\n1,2,1\n")
    (run_dir / "input.csv").write_text("x,y,cluster\n1,2,1\n")
    (run_dir / "run.R").write_text(
        "# wild cluster bootstrap by cluster with B = 999 Rademacher draws\n"
        "for (b in 1:999) { t_boot[b] <- wild_draw(res, cluster) }\n"
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "platform": "r",
                "estimates": ref_rows,
                "method_metadata": {"estimator": "OLS", "bootstrap": "wild cluster"},
                "diagnostics": {},
            }
        )
    )
    record = {
        "cell_id": "cell",
        "task_id": task.task_id,
        "platform": "r",
        "exec_mode": "no_feedback",
        "prompt_mode": "zero_shot",
        "rep": 0,
        "claude": {"ok": True, "total_cost_usd": 0.1, "usage": {"input_tokens": 1}},
        "execution_success": True,
    }

    report = strict_posthoc.build_report([(run_dir, record)], references=references, data_dir=data_dir)

    # The run is scored (legacy contract) but excluded from every headline aggregate.
    assert report["n_model_response_records"] == 1
    assert report["n_analysis_records"] == 0
    assert report["n_quarantined_records"] == 1
    assert report["analysis_summary"]["n"] == 0
    assert report["by_task"] == {}
    assert report["quarantined"]["tasks"] == ["wcb_few"]
    assert report["quarantined"]["summary"]["n"] == 1
    assert report["quarantined"]["summary"]["task_success"] == 1
    assert report["scores"][0]["quarantined"] is True
    # p_wcb was skipped, not scored against the new enumeration reference.
    assert report["scores"][0]["gates"]["numeric"]["per_term"]["p_wcb"] == {"skipped": True}


def test_wcb_method_gate_accepts_expand_grid_enumeration(tmp_path: Path) -> None:
    # R idiom measured on 5 numerically-exact pinned-wave runs (audit 2026-07-06):
    # builds all sign patterns via expand.grid without the literal "32"/"2^5".
    script = tmp_path / "run.R"
    script.write_text(
        "library(sandwich)\n"
        "# cluster-robust CR1\n"
        "signs_mat <- as.matrix(expand.grid(replicate(G, c(-1, 1), simplify = FALSE)))\n"
        "for (b in seq_len(nrow(signs_mat))) t_boot[b] <- wild_t(signs_mat[b, ])\n"
    )

    passed, notes = strict_posthoc.score_method("wcb_few", script.read_text(), script)

    assert passed is True
    assert notes == []


def test_wcb_method_gate_accepts_itertools_product_enumeration(tmp_path: Path) -> None:
    # Python idiom (pinned-wave rep35): product([-1, 1], repeat=G) with a
    # signs_dict, no "boot"/"wild" token anywhere.
    script = tmp_path / "run.py"
    script.write_text(
        "from itertools import product\n"
        "# cluster CR1 se\n"
        "for signs in product([-1, 1], repeat=G):\n"
        "    signs_dict = {clusters[i]: signs[i] for i in range(G)}\n"
        "    t_star = wcb_t(signs_dict)\n"
    )

    passed, notes = strict_posthoc.score_method("wcb_few", script.read_text(), script)

    assert passed is True
    assert notes == []


def test_wcb_method_gate_accepts_stata_mod_floor_enumeration(tmp_path: Path) -> None:
    # Stata idiom (3 pinned-wave runs): decode the b-th of 32 sign patterns via
    # mod(floor(b / 2^g), 2) inside a forvalues loop.
    script = tmp_path / "run.do"
    script.write_text(
        "local npat = 32\n"
        "egen cluster_id = group(cluster)\n"
        "forvalues b = 0/31 {\n"
        "    local bval = mod(floor(`b' / `p2'), 2)\n"
        "}\n"
    )

    passed, notes = strict_posthoc.score_method("wcb_few", script.read_text(), script)

    assert passed is True
    assert notes == []


def test_wcb_method_gate_still_rejects_plain_ols_script(tmp_path: Path) -> None:
    script = tmp_path / "run.py"
    script.write_text("fit = sm.OLS(y, X).fit()\nprint(fit.params)\n")

    passed, notes = strict_posthoc.score_method("wcb_few", script.read_text(), script)

    assert passed is False
    assert notes
