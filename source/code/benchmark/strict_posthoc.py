"""Post-hoc task-success scoring for saved benchmark runs.

This module does not make model calls and does not rerun generated code. It
reads each saved run directory and scores the saved files that already exist:
``run_record.json``, ``result.json``, generated scripts, logs, and ``input.csv``.
The goal is to provide a task-success audit beyond the live harness' first-pass
numeric scorer while keeping the live runner stable.

With ``--write-run-scores`` the scorer also materializes each run's score row as
``score.json`` inside its own run directory. That per-run file is the storage
format the analysis loader reads (``analysis_dataset`` kind ``corpus_records``),
so every arm consumes per-run records. ``run_record.json`` is never read back,
rewritten, or touched; ``score.json`` is a new sibling file. The aggregate
``task_success_report.json`` stays a derived human-readable summary of the same
scores. Decision record 2026-08-10_corpus-records-unification.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics as stats
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tasks import TASKS, Task

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent.parent
# Every arm directory under data/runs holds one directory per run. Sweep logs,
# aggregates, and derived reports live in the arm's ``_working`` directory.
DEFAULT_RUNS_DIR = PROJECT_ROOT / "data" / "runs" / "sonnet_zero_shot"
DEFAULT_OUT_DIR = DEFAULT_RUNS_DIR / "_working"
DATA_DIR = PROJECT_ROOT / "data" / "tasks" / "inputs"
REFERENCE_DIR = PROJECT_ROOT / "data" / "tasks" / "references"

SCORING_VERSION = "strict_posthoc_2026-08-03"

# Per-run score file written inside each run directory by --write-run-scores.
# It is the storage format the analysis loader reads; the aggregate report is
# derived from the same score rows.
RUN_SCORE_NAME = "score.json"

COEF_REL_TOL = 1e-2
COEF_ABS_TOL = 5e-3
SE_REL_TOL = 5e-2
SE_ABS_TOL = 5e-3

# Tasks where n_obs is not a well-defined success criterion — score on the estimate(s)
# only. These are panel/time-series tasks where the "estimation sample" is genuinely
# ambiguous (raw rows vs effective sample after lags/differencing vs treated cells), and
# models legitimately disagree: staggered_did ATT (616..2336 reported), var_irf (198 vs
# 200 after VAR(2) lags), arima_forecast (249 vs 250 after differencing). Setting a single
# reference value just flips which convention fails; the point estimate is the real target.
# Decision record 2026-06-26.
SKIP_NOBS_TASKS = {
    "staggered_did",
    "var_irf",
    "arima_forecast",
    # cox_hazard: Cox estimation sample is genuinely ambiguous (n_subjects=2000 vs
    #   n_events=1470); the fit is verified correct either way. Decision record 2026-06-30.
    "cox_hazard",
    # dynpanel_ab: after re-referencing to canonical Arellano-Bond, n reported differs by
    #   convention (R plm::pgmm reports 100 firms, Stata xtabond reports 800 obs).
    "dynpanel_ab",
    # synth_weights_fixedv: n is ambiguous (153 panel rows vs 12 pre-fit periods vs
    #   8 donors); the weights/ATT reference is exact regardless. Exploratory task.
    "synth_weights_fixedv",
}
# Tasks where a standard error is not a required output (e.g. a point forecast plus a
# prediction interval). Decision record 2026-06-26.
SKIP_SE_TASKS = {"arima_forecast"}

# (task_id, term) pairs compared on MAGNITUDE only: the sign is a benign group-ordering
# convention and the two-sided inference is fully carried by |stat|, df, and p.
# ttest_welch t: Stata's default mean(g0)-mean(g1) flips the sign; df and p match exactly.
# Decision record 2026-06-30.
ABS_COMPARE_ESTIMANDS = {("ttest_welch", "t")}

# (task_id, term) pairs excluded from numeric scoring.
SKIP_ESTIMANDS: set[tuple[str, str]] = set()

# Tasks quarantined from the headline aggregates. Their saved runs were collected under
# a task contract that has since been respecified, so they are scored under the LEGACY
# contract (see the LEGACY_* constants below), kept in `scores` and reported under the
# report's `quarantined` section, and excluded from `analysis_summary` and every `by_*`
# grouping until the task is repopulated under the new contract. This is a quarantine,
# not a drop: the per-run records remain in the report, individually flagged.
# User-ratified 2026-07-01; decision records 2026-07-01_wcb_exact_enumeration.md and
# 2026-07-01_official-rescore-and-wcb-quarantine.md.
QUARANTINED_TASKS = {"wcb_few"}

# Wave discrimination (2026-07-04): run records written by the post-pin harness carry a
# `prompt_sha256` field; legacy records do not. Quarantine (and the LEGACY_* leniencies
# below) apply ONLY to legacy-wave records of a quarantined task. Respecified-wave runs
# of the same task are scored under the active contract and enter the analysis sample.
# Decision record 2026-07-04_wave-provenance-and-wcb-repopulation.md.


def record_is_legacy_wave(record: dict[str, Any]) -> bool:
    return not record.get("prompt_sha256")

# (task_id, term) pairs skipped from numeric scoring ONLY while the task is quarantined.
# Legacy wcb_few runs answered the seeded B=999 prompt, whose p_wcb is a single RNG
# realization; scoring them against the exact-enumeration reference (0.0625) would judge
# them on a contract they never saw. Clears automatically when the task leaves
# QUARANTINED_TASKS for the respecified wave.
LEGACY_SKIP_ESTIMANDS = {("wcb_few", "p_wcb")}

# Per-task method-evidence alternates accepted ONLY while the task is quarantined,
# keyed by require_any group index. The legacy wcb_few prompt asked for B = 999 random
# Rademacher draws, not the 32-pattern exact enumeration the respecified contract names.
LEGACY_METHOD_GROUP_ALTERNATES: dict[str, dict[int, tuple[str, ...]]] = {
    "wcb_few": {3: (r"999", r"\bB\s*=\s*999\b")},
}

# (task_id, term) pairs that are labels/row identifiers, not continuous estimates.
# Cook's-distance row IDs are zero-based integer observation indices; a one-based index is
# off by one even when it falls inside the usual 1% coefficient tolerance.
EXACT_INTEGER_ESTIMANDS = {
    ("cooks_distance", f"rank{rank}_obs_id") for rank in range(1, 6)
}

# (task_id, term) -> multipliers accepted for a coefficient because a defensible alternative
# convention rescales it. var_granger df_den: vars::causality reports the system Wald
# df_den = 2 * aligned-OLS df_den; F and p of the same test are internally consistent.
# The previous 0.5x allowance had no documented convention; 1.0x is the reference.
# Decision records 2026-06-30 and 2026-07-01.
MULTIPLE_CONVENTION_ESTIMANDS = {("var_granger", "df_den"): (2.0,)}

SCRIPT_EXT = {"python": ".py", "r": ".R", "stata": ".do"}
SCRIPT_NAME = {"python": "run.py", "r": "run.R", "stata": "run.do"}
REQUIRED_OUTPUT_KEYS = {"task_id", "platform", "estimates", "method_metadata", "diagnostics"}
USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


@dataclass(frozen=True)
class MethodRule:
    """Static evidence rule for task-level method plausibility."""

    require_any: tuple[tuple[str, ...], ...]
    reject_any: tuple[str, ...] = ()


METHOD_RULES: dict[str, MethodRule] = {
    "ols_robust": MethodRule(
        require_any=(
            (r"\bOLS\b", r"\blm\s*\(", r"\bregress\b"),
            (r"HC1", r"heteroskedastic", r"robust", r"vce\s*\(\s*robust\s*\)"),
        )
    ),
    "arima_forecast": MethodRule(
        require_any=((r"ARIMA", r"arima"), (r"\(1\s*,\s*1\s*,\s*1\)",), (r"forecast", r"predict"))
    ),
    "cluster_ols": MethodRule(
        require_any=(
            (r"\bOLS\b", r"\blm\s*\(", r"\bregress\b"),
            (r"cluster", r"vcovCL", r"vce\s*\(\s*cluster"),
        )
    ),
    "two_way_cluster": MethodRule(
        require_any=(
            (r"\bOLS\b", r"\blm\s*\(", r"\bregress\b"),
            (r"cluster", r"vcovCL", r"cov_cluster_2groups", r"Cameron", r"CGM"),
            (r"\bfirm\b",),
            (r"\byear\b",),
        )
    ),
    "vif": MethodRule(
        require_any=(
            (
                r"variance_inflation_factor",
                r"estat\s+vif",
                r"\bcar::vif\s*\(",
                r"(?<![A-Za-z0-9_:.])vif\s*\(",
                r"1\s*/\s*\(\s*1\s*-\s*(?:`?r2'?|r_?squared|r\.squared|rsquared|e\s*\(\s*r2\s*\))",
                r"1\s*/\s*\(\s*1\s*-\s*summary\s*\([^)]*\)\$r\.squared",
            ),
        )
    ),
    "cooks_distance": MethodRule(
        require_any=(
            (
                r"cooks\.distance",
                r"get_influence",
                r"\bcooksd\b",
                r"OLSInfluence",
                r"predict\b.*\bcooksd\b",
                r"gen(?:erate)?\s+(?:double\s+)?_?cooks_?d\b",
            ),
        )
    ),
    "ttest_welch": MethodRule(
        require_any=(
            (r"Welch", r"unequal", r"equal_var\s*=\s*False", r"var\.equal\s*=\s*FALSE"),
            # Manual Welch implementations compute t directly (t_stat = diff / se) and
            # use stats.t.sf for the p-value rather than calling a t-test function.
            (
                r"t\.test",
                r"ttest_ind",
                r"\bttest\b",
                r"t_stat",
                r"stats\.t\.",
                r"t_statistic",
                # Measured Stata manual implementation: the first evidence
                # group separately requires Welch or unequal-variance text.
                r"\bttail\s*\(",
            ),
        )
    ),
    "anova_oneway": MethodRule(
        require_any=(
            (
                r"f_oneway",
                r"\boneway\s+",
                r"\banova\s+\w+\s+group\b",
                r"\banova\s+\w+\s+",
                r"\baov\s*\([^)]*factor\s*\(\s*group",
                r"\banova\s*\([^)]*factor\s*\(\s*group",
                r"factor\s*\(\s*group",
                r"as\.factor\s*\([^)]*group",
                r"C\s*\(\s*group\s*\)",
                r"i\.group",
            ),
        )
    ),
    "chisq_indep": MethodRule(
        require_any=(
            (
                r"chisq\.test",
                r"chi2_contingency",
                r"\btab(?:ulate)?\b.*,\s*chi2",
                r"chi2\.sf",
                r"stats\.chi2",
            ),
        )
    ),
    "logit": MethodRule(
        require_any=((r"logit", r"Logit", r"family\s*=\s*binomial"),)
    ),
    "probit": MethodRule(
        require_any=((r"probit", r"Probit", r"link\s*=\s*[\"']probit", r"binomial\(link\s*=\s*['\"]probit"),)
    ),
    "poisson_offset": MethodRule(
        require_any=((r"Poisson", r"poisson"), (r"offset", r"exposure"))
    ),
    "cox_hazard": MethodRule(
        require_any=((r"Cox", r"coxph", r"PHReg", r"\bstcox\b"),)
    ),
    "twoway_fe": MethodRule(
        require_any=(
            # Two-way FE can be expressed many valid ways across platforms; the
            # original group only matched feols/reghdfe/C(firm)/i.firm and so
            # false-negatived PanelOLS, xtreg/areg, factor(), and dummy-based
            # implementations (see decision record 2026-06-15).
            (
                r"fixed effect", r"feols", r"reghdfe", r"C\s*\(\s*firm\s*\)", r"i\.firm",
                r"PanelOLS", r"entity_effects", r"time_effects", r"xtreg", r"\bareg\b",
                r"\bwithin\b", r"factor\s*\(", r"get_dummies", r"absorb", r"two-?way\s*fe",
                r"\bfelm\b", r"\blfe\b",
            ),
            (r"\bfirm\b",),
            (r"\byear\b",),
            (r"cluster", r"vce\s*\(\s*cluster", r"cov_type\s*=\s*[\"']cluster"),
        )
    ),
    "iv2sls_robust": MethodRule(
        require_any=(
            # Manual matrix-algebra 2SLS builds the instrument cross-product
            # (ZtZ_inv/ZTZ_inv) and names the endogenous regressor (x_endog)
            # without writing "2SLS" or "instrument" anywhere. Measured on the
            # 3 runs whose only group-1 match had been the task-id literal
            # before masking (probe 2026-07-30).
            (r"2SLS", r"IV2SLS", r"ivreg", r"instrument", r"ivregress",
             r"\bZtZ", r"\bendog"),
            (r"robust", r"HC1", r"vce\s*\(\s*robust"),
            (r"\bz\b",),
        )
    ),
    "dynpanel_ab": MethodRule(
        require_any=(
            (r"Arellano", r"Bond", r"first.?diff", r"\bdiff\s*\(", r"\bxtabond\b", r"\bpgmm\b"),
            (
                r"lag",
                r"shift",
                r"\bL\.",
                # Manual difference-GMM code uses dy[:-1] as the lagged
                # differenced outcome without naming it as a lag.
                r"\bdy\s*\[\s*:-1\s*\]",
            ),
            (r"GMM", r"\bIV\b", r"instrument", r"2SLS", r"2sls", r"ivreg", r"projection"),
        )
    ),
    "staggered_did": MethodRule(
        require_any=(
            (r"groupby", r"collapse", r"mean"),
            (r"cohort",),
            (r"never", r"untreated", r"not[_\s-]?treated", r"comparison",
             r"cohort\s*==\s*-?1", r"\bnt\b", r"is_nt"),
        ),
        # Reject the package aggregators as CALLS (att_gt(...), aggte(...), feols(...)) so
        # an ordinary variable binding like `att_gt = ...` for ATT(g,t) is not penalized.
        # (See 2026-06-26 decision record: substring gate had no discriminant validity.)
        reject_any=(r"\bTWFE\b", r"att_gt\s*\(", r"aggte\s*\(", r"\bbootstrap\b", r"feols\s*\("),
    ),
    "rdd_sharp": MethodRule(
        require_any=(
            (r"local", r"bandwidth", r"\bh\s*=\s*0\.5"),
            # Triangular weighting is often implemented inline (weight = 1 - |dist|/h)
            # and fit with WLS, without writing the words "triangular"/"kernel".
            (r"triangular", r"kernel", r"\bweight", r"\bWLS\b", r"\bwls\b", r"aw\s*=", r"pweight"),
            (
                r"cutoff",
                r"z\s*>=\s*0",
                r"z\s*<\s*0",
                r"z\s*>=\s*`?[A-Za-z_][A-Za-z0-9_]*'?",
                r"z\s*<\s*`?[A-Za-z_][A-Za-z0-9_]*'?",
                # Centered or bandwidth-restricted running variables retain
                # the z prefix, for example z_c and z_bw.
                r"\bz_[A-Za-z0-9_]+\s*>=\s*0",
                r"\bz_[A-Za-z0-9_]+\s*<\s*0",
                # Measured Stata implementations rename z to `running` and
                # either split the local regressions or define the right-side
                # indicator for an interacted local-linear regression.
                r"\brunning\s*>=\s*0",
                r"\brunning\s*<\s*0",
            ),
        )
    ),
    "lp_irf": MethodRule(
        require_any=(
            # Local projection is OLS of a *led* outcome on the shock per horizon;
            # the original group only matched the literal "local projection"/
            # "horizon"/"shift(-4)" and missed shift(-h) loops, lead(), y_lead
            # variables, and Stata forward operators (see decision record 2026-06-15).
            # The HAC and lag-4 groups below still pin the LP convention.
            (
                r"local projection", r"horizon", r"shift\s*\(\s*-", r"lead\s*\(",
                r"y_lead", r"\bprojection\b", r"impulse", r"\birf\b", r"\bF\d*\.",
            ),
            (r"Newey", r"HAC", r"cov_hac"),
            (r"\blag\s*4\b", r"maxlags\s*=\s*4", r"lag\s*=\s*4", r"lag\s*\(\s*4"),
        )
    ),
    "wcb_few": MethodRule(
        require_any=(
            # Manual wild-cluster bootstraps loop over draws with names like t_boot and
            # resample residuals rather than writing the literal word "bootstrap".
            # Exact enumerations also build sign matrices (signs_mat / t_star) or decode
            # the b-th pattern via mod(floor(b/2^g), 2) without any "boot" token —
            # measured on 10 numerically-exact pinned-wave runs (audit 2026-07-06).
            (r"wild", r"bootstrap", r"boot", r"Rademacher", r"boottest", r"resample",
             r"runiform", r"cond\s*\(", r"sign pattern", r"enumerat",
             r"signs?_(?:mat|dict|vec)", r"\bt_star\b", r"mod\s*\(\s*(?:floor|int)\s*\("),
            (r"cluster",),
            # Exact-enumeration evidence: the literal count, or the constructs that
            # generate all 2^G Rademacher sign patterns.
            (r"32", r"2\^5", r"enumerat", r"expand\.grid", r"itertools",
             r"product\s*\(", r"repeat\s*=\s*G", r"replicate\s*\(\s*G",
             r"2\s*\^\s*G", r"2\s*\*\*\s*G"),
        )
    ),
    "var_irf": MethodRule(
        require_any=((r"\bVAR\b", r"VAR\s*\("), (r"Cholesky", r"orthogonal", r"ortho", r"\boirf\b"), (r"horizon", r"\b4\b"))
    ),
    "var_granger": MethodRule(
        require_any=(
            (
                r"Granger",
                r"grangercausalitytests",
                r"grangertest",
                r"test_causality",
                r"causality\s*\(",
                r"\bf_test\s*\(",
                r"\bwald_test\s*\(",
                r"\btest\s+L\.y2\s+L2\.y2",
                r"\btest\s+y2_?lag1\s+y2_?lag2",
                r"\btest\s+y2_?l1\s+y2_?l2",
                r"linearHypothesis",
                r"anova\s*\(",
                r"restricted",
                r"unrestricted",
                # Manual restricted-vs-unrestricted F tests abbreviate the sums
                # of squares (rss_res/RSS_u), read statsmodels .ssr, call
                # lmtest::waldtest(fit_res, fit_unres), or run the Stata
                # numbered-lag restriction test in the forms test l1.y2 l2.y2,
                # test L1y2 L2y2, and test (L1.y2 = 0) (L2.y2 = 0). Measured on
                # the 13 runs whose only group-1 match had been the task-id
                # literal before masking (probes 2026-07-30).
                r"\brss_",
                r"\.ssr\b",
                r"\bwald_?test\s*\(",
                r"\btest\s+\(?\s*l1[._]?y2",
            ),
            (
                r"\blag",
                r"lags?\s*=\s*2",
                r"maxlag\s*=\s*2",
                r"\bp\s*=\s*2",
                r"fit\s*\(\s*2",
                r"\bL\.y2\b",
                r"\bL2\.y2\b",
                r"y2_?lag",
                r"y2_?l[12]",
                r"y2.*shift",
                r"shift.*y2",
            ),
            (r"\by1\b",),
            (r"\by2\b",),
        )
    ),
}

FORBIDDEN_INTEGRITY_PATTERNS = (
    # Both the retired and the current references path stay listed. Historical
    # scripts saved in 2026 contain the literal "data/references"; removing it
    # would silently disable the integrity gate for every stored run.
    r"data/references",
    r"data/tasks/references",
    r"reference_implementations",
    r"manual_references",
    r"strict_scoring_inputs",
    r"reference(?:_v\d*)?\.json",
)

DISPLAY_LABELS = {
    "specification_correctness": "Specification correctness",
    "executability": "Executability",
    "output_correctness": "Output correctness",
    "submission_validity": "Submission validity",
    "no_model_response": "No model response",
}


def finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def number_or_none(value: Any) -> float | None:
    return float(value) if finite_number(value) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def load_references(reference_dir: Path = REFERENCE_DIR) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for path in sorted(reference_dir.glob("reference*.json")):
        refs.update(json.loads(path.read_text()))
    return refs


def load_records(runs_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Read every run record one level below ``runs_dir``.

    Underscore-prefixed directories hold working files, quarantined runs, and
    retired records. They are never part of an arm's scored sample, so they are
    skipped here as they are in ``analysis_dataset``.
    """
    records = []
    for path in sorted(runs_dir.glob("*/run_record.json")):
        if path.parent.name.startswith("_"):
            continue
        records.append((path.parent, json.loads(path.read_text())))
    return records


def claude_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for key in sorted(record):
        value = record.get(key)
        if key == "claude" or key.startswith("claude_attempt_"):
            if isinstance(value, dict):
                blocks.append(value)
    return blocks


def has_successful_model_response(record: dict[str, Any]) -> bool:
    return any(block.get("ok") is True for block in claude_blocks(record))


def stream_result_cost(run_dir: Path) -> float | None:
    stream_path = run_dir / "agent_stream.jsonl"
    if not stream_path.is_file():
        return None
    result_cost = None
    for line in stream_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result" and event.get("total_cost_usd") is not None:
            result_cost = float(event["total_cost_usd"])
    return result_cost


def record_cost(record: dict[str, Any], run_dir: Path | None = None) -> float | None:
    costs = [
        float(block["total_cost_usd"])
        for block in claude_blocks(record)
        if block.get("total_cost_usd") is not None
    ]
    if costs:
        return sum(costs)
    return stream_result_cost(run_dir) if run_dir is not None else None


def record_wall_seconds(record: dict[str, Any]) -> float:
    return sum(float(block.get("wall_seconds") or 0.0) for block in claude_blocks(record))


def record_tokens(record: dict[str, Any]) -> dict[str, int]:
    totals = {key: 0 for key in USAGE_KEYS}
    for block in claude_blocks(record):
        usage = block.get("usage") or {}
        for key in USAGE_KEYS:
            totals[key] += int(float(usage.get(key) or 0))
    return totals


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def attempt_number(path: Path) -> int:
    match = re.search(r"attempt(\d+)_run", path.name)
    return int(match.group(1)) if match else -1


def find_script(run_dir: Path, platform: str) -> Path | None:
    ext = SCRIPT_EXT.get(platform)
    if ext is None:
        return None
    attempts = sorted(run_dir.glob(f"attempt*_run{ext}"), key=attempt_number)
    if attempts:
        return attempts[-1]
    canonical = run_dir / SCRIPT_NAME[platform]
    if canonical.exists():
        return canonical
    candidates = sorted(run_dir.glob(f"*{ext}"))
    return candidates[-1] if candidates else None


def combined_evidence_text(
    run_dir: Path,
    record: dict[str, Any],
    result_data: dict[str, Any] | None,
    script_path: Path | None,
) -> str:
    parts = [json.dumps(record.get("scoring") or {}, default=str)]
    if result_data:
        parts.append(json.dumps(result_data.get("method_metadata") or {}, default=str))
        parts.append(json.dumps(result_data.get("diagnostics") or {}, default=str))
    if script_path:
        parts.append(read_text(script_path))
    for log_path in sorted(run_dir.glob("*.log")) + sorted(run_dir.glob("attempt*_run.log")):
        parts.append(read_text(log_path)[-5000:])
    return "\n".join(parts)


def repair_stata_json(text: str) -> str:
    """Repair Stata's non-standard JSON so it parses.

    Stata's fixed/%g writers emit leading-dot decimals (``"se": .0219``) and bare
    missing-value dots (``"estimate": .``), both invalid JSON. Insert leading zeros
    and encode lone dots as ``null``. Only applied as a fallback after strict parsing
    fails, so valid JSON is never disturbed.
    """

    # A lone '.' value (dot NOT followed by a digit) -> null.
    text = re.sub(r"(?<=[:,\[\s])-?\.(?=\s*[,}\]])", "null", text)
    # A leading-dot decimal (dot followed by a digit) -> 0.<digits>.
    text = re.sub(r"(?<=[:,\[\s])(-?)\.(?=\d)", r"\g<1>0.", text)
    return text


def load_result(result_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not result_path.exists():
        return None, ["result.json missing"]
    raw = result_path.read_text()
    try:
        loaded = json.loads(raw)
    except Exception:
        try:
            loaded = json.loads(repair_stata_json(raw))
        except Exception as exc:
            return None, [f"result.json invalid JSON: {exc}"]
    if not isinstance(loaded, dict):
        return None, ["result.json root is not an object"]
    # jsonlite/write_json(auto_unbox=TRUE) serializes an empty object as [] rather than
    # {}; normalize so R runs with an empty diagnostics/method_metadata are not failed
    # purely on serialization. Decision record 2026-06-30.
    for key in ("diagnostics", "method_metadata"):
        if loaded.get(key) == []:
            loaded[key] = {}
    return loaded, []


def score_artifact(
    result_data: dict[str, Any] | None,
    task: Task,
    platform: str,
) -> tuple[bool, dict[str, dict[str, Any]], list[str]]:
    notes: list[str] = []
    if result_data is None:
        return False, {}, notes

    missing_keys = REQUIRED_OUTPUT_KEYS - set(result_data)
    if missing_keys:
        notes.append(f"missing output keys: {sorted(missing_keys)}")
    if result_data.get("task_id") != task.task_id:
        notes.append(f"task_id mismatch: {result_data.get('task_id')!r}")
    if result_data.get("platform") != platform:
        notes.append(f"platform mismatch: {result_data.get('platform')!r}")

    estimates_raw = result_data.get("estimates")
    if not isinstance(estimates_raw, list):
        notes.append("estimates is not a list")
        return False, {}, notes

    estimates: dict[str, dict[str, Any]] = {}
    duplicate_terms = []
    for row in estimates_raw:
        if not isinstance(row, dict):
            notes.append("estimate row is not an object")
            continue
        term = row.get("term")
        if not isinstance(term, str):
            notes.append("estimate row has missing/non-string term")
            continue
        if term in estimates:
            duplicate_terms.append(term)
        estimates[term] = row
        missing_row_keys = {"term", "estimate", "se", "n_obs"} - set(row)
        if missing_row_keys:
            notes.append(f"{term}: missing estimate row keys {sorted(missing_row_keys)}")

    if duplicate_terms:
        notes.append(f"duplicate terms: {sorted(set(duplicate_terms))}")

    expected = set(task.estimands)
    observed = set(estimates)
    missing_terms = expected - observed
    extra_terms = observed - expected
    if missing_terms:
        notes.append(f"missing terms: {sorted(missing_terms)}")
    if extra_terms:
        notes.append(f"extra terms: {sorted(extra_terms)}")

    if not isinstance(result_data.get("method_metadata"), dict):
        notes.append("method_metadata is not an object")
    if not isinstance(result_data.get("diagnostics"), dict):
        notes.append("diagnostics is not an object")

    return len(notes) == 0, estimates, notes


def compare_numeric(
    estimates: dict[str, dict[str, Any]],
    task: Task,
    references: dict[str, Any],
    legacy_quarantined: bool = False,
) -> tuple[bool, dict[str, Any], list[str]]:
    notes: list[str] = []
    per_term: dict[str, Any] = {}
    if task.reference_key not in references:
        return False, per_term, [f"reference key missing: {task.reference_key}"]

    ref_rows = references[task.reference_key].get("estimates") or []
    ref_by_term = {row["term"]: row for row in ref_rows if isinstance(row, dict) and "term" in row}
    all_pass = True

    for term in task.estimands:
        if (task.task_id, term) in SKIP_ESTIMANDS or (
            legacy_quarantined and (task.task_id, term) in LEGACY_SKIP_ESTIMANDS
        ):
            per_term[term] = {"skipped": True}
            continue

        row = estimates.get(term)
        ref_row = ref_by_term.get(term)
        if row is None or ref_row is None:
            notes.append(f"{term}: missing result or reference row")
            all_pass = False
            continue

        estimate = number_or_none(row.get("estimate"))
        ref_estimate = number_or_none(ref_row.get("estimate"))
        if estimate is None or ref_estimate is None:
            notes.append(f"{term}: non-finite estimate")
            all_pass = False
            per_term[term] = {"estimate_pass": False}
            continue

        # Some terms are compared on magnitude only (benign sign convention).
        if (task.task_id, term) in ABS_COMPARE_ESTIMANDS:
            cmp_estimate, cmp_ref = abs(estimate), abs(ref_estimate)
        else:
            cmp_estimate, cmp_ref = estimate, ref_estimate
        coef_err = abs(cmp_estimate - cmp_ref)
        if (task.task_id, term) in EXACT_INTEGER_ESTIMANDS:
            coef_tol = 0.0
            coef_pass = (
                estimate.is_integer()
                and ref_estimate.is_integer()
                and int(estimate) == int(ref_estimate)
            )
        else:
            coef_tol = max(COEF_REL_TOL * abs(cmp_ref), COEF_ABS_TOL)
            coef_pass = coef_err <= coef_tol
        # Accept a defensible alternative convention that rescales the term.
        multipliers = MULTIPLE_CONVENTION_ESTIMANDS.get((task.task_id, term))
        if not coef_pass and multipliers:
            for factor in multipliers:
                scaled = factor * ref_estimate
                if abs(estimate - scaled) <= max(COEF_REL_TOL * abs(scaled), COEF_ABS_TOL):
                    coef_pass = True
                    coef_err = abs(estimate - scaled)
                    break

        got_se = number_or_none(row.get("se"))
        ref_se = number_or_none(ref_row.get("se"))
        se_err = None
        se_tol = None
        se_pass = None
        if ref_se is not None and task.task_id not in SKIP_SE_TASKS:
            if got_se is None:
                notes.append(f"{term}: missing finite standard error")
                se_pass = False
            else:
                se_err = abs(got_se - ref_se)
                se_tol = max(SE_REL_TOL * abs(ref_se), SE_ABS_TOL)
                se_pass = se_err <= se_tol

        got_n = number_or_none(row.get("n_obs"))
        ref_n = number_or_none(ref_row.get("n_obs"))
        n_obs_pass = True
        if ref_n is not None and task.task_id not in SKIP_NOBS_TASKS:
            if got_n is None or int(round(got_n)) != int(round(ref_n)):
                notes.append(f"{term}: n_obs mismatch")
                n_obs_pass = False

        per_term[term] = {
            "ref_estimate": ref_estimate,
            "got_estimate": estimate,
            "coef_err": coef_err,
            "coef_tol": coef_tol,
            "coef_pass": coef_pass,
            "ref_se": ref_se,
            "got_se": got_se,
            "se_err": se_err,
            "se_tol": se_tol,
            "se_pass": se_pass,
            "ref_n_obs": ref_n,
            "got_n_obs": got_n,
            "n_obs_pass": n_obs_pass,
        }
        if not coef_pass:
            notes.append(f"{term}: estimate outside tolerance")
        if se_pass is False:
            notes.append(f"{term}: standard error outside tolerance or missing")
        if not n_obs_pass:
            notes.append(f"{term}: sample size outside exact match")
        all_pass = all_pass and coef_pass and (se_pass is not False) and n_obs_pass

    return all_pass, per_term, notes


def mask_task_id_evidence(evidence_text: str, task_id: str) -> str:
    """Mask the mandatory task-id literal so it cannot satisfy method evidence.

    Every valid submission must write the exact task id into ``result.json``, so
    the id itself (e.g. ``ols_robust`` contains ``robust``) satisfied evidence
    regexes vacuously (review finding F4, fix 2026-07-30). Two boilerplate
    forms are masked: the id after a ``task_id`` key (any quoting or assignment
    style within a few non-word characters) and the id as a cell-id prefix
    (``<task_id>__``). Code tokens and other metadata values are never masked,
    so genuine method usage whose token equals the task id (Stata ``logit`` and
    ``estat vif``, R ``car::vif(``) and honest estimator labels for tasks named
    after their estimator (``"estimator": "probit"``) keep their evidence.
    Measured on every saved run collection (masking probes 2026-07-30): zero
    genuine-method false fails.
    """
    if not task_id:
        return evidence_text
    mask = "~" * len(task_id)
    escaped = re.escape(task_id)
    evidence_text = re.sub(
        rf"(task_id\W{{0,8}}?){escaped}",
        lambda m: m.group(1) + mask,
        evidence_text,
        flags=re.IGNORECASE,
    )
    return re.sub(rf"{escaped}(?=__)", mask, evidence_text, flags=re.IGNORECASE)


def score_method(
    task_id: str,
    evidence_text: str,
    script_path: Path | None,
    legacy_quarantined: bool = False,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    rule = METHOD_RULES.get(task_id)
    if rule is None:
        return False, [f"method rule missing for {task_id}"]
    # A tool_loop/workspace run can produce a valid result.json without the final script
    # being persisted; fall back to the other evidence (method_metadata, diagnostics,
    # logs) rather than auto-failing. Only fail outright if there is no evidence at all.
    if script_path is None and not (evidence_text and evidence_text.strip()):
        return False, ["generated script missing"]

    # The mandatory task-id string is not method evidence in either direction.
    evidence_text = mask_task_id_evidence(evidence_text, task_id)
    reject_evidence_text = strip_method_comments(evidence_text)
    for pattern in rule.reject_any:
        if re.search(pattern, reject_evidence_text, flags=re.IGNORECASE):
            notes.append(f"rejected method evidence: {pattern}")

    legacy_groups = (
        LEGACY_METHOD_GROUP_ALTERNATES.get(task_id, {})
        if legacy_quarantined
        else {}
    )
    for group_idx, group in enumerate(rule.require_any, start=1):
        patterns = group + legacy_groups.get(group_idx, ())
        if not any(re.search(pattern, evidence_text, flags=re.IGNORECASE) for pattern in patterns):
            notes.append(f"missing method evidence group {group_idx}")

    return len(notes) == 0, notes


def strip_method_comments(evidence_text: str) -> str:
    """Remove common script comments before applying method reject patterns."""
    cleaned: list[str] = []
    for line in evidence_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("#", "//", "*")):
            continue
        line = re.split(r"\s+#|\s+//", line, maxsplit=1)[0]
        cleaned.append(line)
    return "\n".join(cleaned)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_number_literals(reference: dict[str, Any]) -> list[str]:
    literals: set[str] = set()
    for row in reference.get("estimates") or []:
        if not isinstance(row, dict):
            continue
        for key in ("estimate", "se"):
            value = number_or_none(row.get(key))
            if value is None or abs(value) < 1e-8:
                continue
            if abs(value - round(value)) < 1e-12 and abs(value) <= 10:
                continue
            for precision in (6, 8, 10, 12):
                literals.add(format(value, f".{precision}g"))
            if abs(value) < 1_000_000:
                fixed = f"{value:.6f}".rstrip("0").rstrip(".")
                if fixed and fixed not in {"-0", "0"}:
                    literals.add(fixed)
    return sorted(literals, key=len, reverse=True)


# Minimum significant digits before a stored literal also fires as a PREFIX of a
# longer digit string in the script (review finding F4, fix 2026-07-30).
PREFIX_MATCH_MIN_DIGITS = 8


def numeric_literal_pattern(literal: str) -> re.Pattern[str]:
    # A high-precision stored literal also fires when it is a PREFIX of a longer
    # digit string, so a script that hard-codes the reference value at FULL
    # precision cannot evade the check that catches its rounded copy. This is
    # safe because the check scans generated script text: a genuinely computed
    # value never appears as a source literal, and an 8+ significant-digit
    # prefix collision is not a coincidence. Short literals (integer-like
    # references such as a df of 193) keep the strict two-sided boundary so they
    # do not fire inside unrelated longer numbers like a seed of 1934.
    digits = sum(ch.isdigit() for ch in literal)
    if digits >= PREFIX_MATCH_MIN_DIGITS:
        lookahead = r"(?![A-Za-z_.-])"
    else:
        lookahead = r"(?![A-Za-z0-9_.-])"
    return re.compile(rf"(?<![A-Za-z0-9_.-]){re.escape(literal)}{lookahead}")


def score_integrity(
    task: Task,
    run_dir: Path,
    script_path: Path | None,
    references: dict[str, Any],
    data_dir: Path = DATA_DIR,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    # When the final script was not persisted (some tool_loop/workspace runs), the
    # script-content anti-cheat checks are not evaluable; still enforce the input-hash
    # check below. No false passes were found in the corpus. Decision record 2026-06-30.
    script_text = read_text(script_path) if script_path is not None else ""

    if script_text:
        for pattern in FORBIDDEN_INTEGRITY_PATTERNS:
            if re.search(pattern, script_text, flags=re.IGNORECASE):
                notes.append(f"forbidden reference-output access: {pattern}")

        reference = references.get(task.reference_key) or {}
        for literal in reference_number_literals(reference):
            if numeric_literal_pattern(literal).search(script_text):
                notes.append(f"reference numeric literal appears in script: {literal}")
                break

    run_input = run_dir / "input.csv"
    canonical_input = data_dir / task.data_file
    if not run_input.exists():
        notes.append("run-local input.csv missing")
    elif not canonical_input.exists():
        notes.append(f"canonical input missing: {canonical_input}")
    elif sha256_file(run_input) != sha256_file(canonical_input):
        notes.append("run-local input.csv hash does not match canonical data file")

    return len(notes) == 0, notes


def score_one_run(
    run_dir: Path,
    record: dict[str, Any],
    tasks_by_id: dict[str, Task],
    references: dict[str, Any],
    data_dir: Path = DATA_DIR,
) -> dict[str, Any]:
    task_id = str(record.get("task_id") or "")
    platform = str(record.get("platform") or "")
    task = tasks_by_id.get(task_id)
    legacy_wave = record_is_legacy_wave(record)
    legacy_quarantined = task_id in QUARANTINED_TASKS and legacy_wave
    base = {
        "run_dir": str(run_dir),
        "cell_id": record.get("cell_id", run_dir.name),
        "task_id": task_id,
        "platform": platform,
        "prompt_mode": record.get("prompt_mode"),
        "exec_mode": record.get("exec_mode"),
        "rep": record.get("rep"),
        "model_response": has_successful_model_response(record),
        "quarantined": legacy_quarantined,
        "prompt_wave": "legacy" if legacy_wave else "pinned-2026-07",
        "cost_usd": record_cost(record, run_dir),
        "wall_seconds": record_wall_seconds(record),
        "tokens": record_tokens(record),
        "gates": {},
        "strict_success": False,
        "failure_labels": [],
        "notes": [],
    }
    if task is None:
        base["notes"].append(f"unknown task_id: {task_id}")
        return base

    result_data, result_notes = load_result(run_dir / "result.json")
    script_path = find_script(run_dir, platform)
    # A run that produced a parseable result.json is a model response even if the CLI
    # wrapper exited non-zero (e.g. tool_loop wrote a correct result then exited 1).
    # Decision record 2026-06-30.
    base["model_response"] = base["model_response"] or (result_data is not None)
    artifact_pass, estimates, artifact_notes = score_artifact(result_data, task, platform)
    numeric_pass, per_term, numeric_notes = compare_numeric(
        estimates, task, references, legacy_quarantined=legacy_quarantined
    )
    evidence_text = combined_evidence_text(run_dir, record, result_data, script_path)
    method_pass, method_notes = score_method(
        task_id, evidence_text, script_path, legacy_quarantined=legacy_quarantined
    )
    integrity_pass, integrity_notes = score_integrity(
        task, run_dir, script_path, references, data_dir
    )
    # Executability = the run produced a valid result artifact OR the harness recorded a
    # successful execution; not tied to the CLI wrapper exit status. Decision record 2026-06-30.
    execution_pass = base["model_response"] and (
        bool(record.get("execution_success")) or artifact_pass
    )

    base["gates"] = {
        "execution": {"pass": execution_pass, "notes": [] if execution_pass else ["execution did not succeed"]},
        "artifact": {"pass": artifact_pass, "notes": result_notes + artifact_notes},
        "numeric": {"pass": numeric_pass, "notes": numeric_notes, "per_term": per_term},
        "method": {"pass": method_pass, "notes": method_notes},
        "integrity": {"pass": integrity_pass, "notes": integrity_notes},
    }
    base["criteria"] = {
        "specification_correctness": {
            "pass": method_pass,
            "notes": method_notes,
        },
        "executability": {
            "pass": execution_pass,
            "notes": [] if execution_pass else ["execution did not succeed"],
        },
        "output_correctness": {
            "pass": artifact_pass and numeric_pass,
            "notes": result_notes + artifact_notes + numeric_notes,
        },
        "submission_validity": {
            "pass": integrity_pass,
            "notes": integrity_notes,
        },
    }
    base["task_success"] = all(
        criterion["pass"] for criterion in base["criteria"].values()
    )
    base["strict_success"] = base["task_success"]
    labels = [
        label
        for label, gate in base["gates"].items()
        if not gate["pass"]
    ]
    criterion_labels = [
        label
        for label, criterion in base["criteria"].items()
        if not criterion["pass"]
    ]
    if not base["model_response"]:
        labels.append("no_model_response")
        criterion_labels.append("no_model_response")
    base["failure_labels"] = labels if labels else ["passed"]
    base["failure_criteria"] = criterion_labels if criterion_labels else ["passed"]
    return base


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(scores)
    if n == 0:
        return {
            "n": 0,
            "specification_correctness_pass": 0,
            "executability_pass": 0,
            "output_correctness_pass": 0,
            "submission_validity_pass": 0,
            "task_success": 0,
            "execution_pass": 0,
            "artifact_pass": 0,
            "numeric_pass": 0,
            "method_pass": 0,
            "integrity_pass": 0,
            "strict_success": 0,
            "cost_usd": 0.0,
            "median_cost_usd": None,
            "median_wall_seconds": None,
        }
    costs = [
        float(score["cost_usd"])
        for score in scores
        if score.get("cost_usd") is not None
    ]
    walls = [score["wall_seconds"] for score in scores if score["wall_seconds"] > 0]
    return {
        "n": n,
        "specification_correctness_pass": sum(
            score["criteria"]["specification_correctness"]["pass"] for score in scores
        ),
        "executability_pass": sum(
            score["criteria"]["executability"]["pass"] for score in scores
        ),
        "output_correctness_pass": sum(
            score["criteria"]["output_correctness"]["pass"] for score in scores
        ),
        "submission_validity_pass": sum(
            score["criteria"]["submission_validity"]["pass"] for score in scores
        ),
        "task_success": sum(score["task_success"] for score in scores),
        "execution_pass": sum(score["gates"]["execution"]["pass"] for score in scores),
        "artifact_pass": sum(score["gates"]["artifact"]["pass"] for score in scores),
        "numeric_pass": sum(score["gates"]["numeric"]["pass"] for score in scores),
        "method_pass": sum(score["gates"]["method"]["pass"] for score in scores),
        "integrity_pass": sum(score["gates"]["integrity"]["pass"] for score in scores),
        "strict_success": sum(score["strict_success"] for score in scores),
        "cost_usd": round(sum(costs), 6),
        "median_cost_usd": round(stats.median(costs), 6) if costs else None,
        "median_wall_seconds": round(stats.median(walls), 3) if walls else None,
    }


def grouped(scores: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        label = "__".join(str(score.get(key, "<unknown>")) for key in keys)
        groups[label].append(score)
    return {label: summarize(rows) for label, rows in sorted(groups.items())}


def failure_counts(scores: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for score in scores:
        labels = score["failure_labels"]
        if labels == ["passed"]:
            continue
        counter.update(labels)
    return counter.most_common()


def failure_criteria_counts(scores: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for score in scores:
        labels = score["failure_criteria"]
        if labels == ["passed"]:
            continue
        counter.update(labels)
    return counter.most_common()


def top_gate_notes(scores: list[dict[str, Any]], limit: int = 12) -> dict[str, list[tuple[str, int]]]:
    notes_by_gate: dict[str, list[tuple[str, int]]] = {}
    for gate in ("artifact", "numeric", "method", "integrity"):
        counter: Counter[str] = Counter()
        for score in scores:
            gate_data = score["gates"][gate]
            if gate_data["pass"]:
                continue
            counter.update(str(note) for note in gate_data.get("notes", []))
        notes_by_gate[gate] = counter.most_common(limit)
    return notes_by_gate


def top_criterion_notes(scores: list[dict[str, Any]], limit: int = 12) -> dict[str, list[tuple[str, int]]]:
    notes_by_criterion: dict[str, list[tuple[str, int]]] = {}
    for criterion in (
        "specification_correctness",
        "executability",
        "output_correctness",
        "submission_validity",
    ):
        counter: Counter[str] = Counter()
        for score in scores:
            criterion_data = score["criteria"][criterion]
            if criterion_data["pass"]:
                continue
            counter.update(str(note) for note in criterion_data.get("notes", []))
        notes_by_criterion[criterion] = counter.most_common(limit)
    return notes_by_criterion


def write_run_score(run_dir: Path, score: dict[str, Any]) -> Path:
    """Write one run's score row as ``score.json`` inside its run directory.

    A new sibling file; ``run_record.json`` is never read back or rewritten.
    """
    path = run_dir / RUN_SCORE_NAME
    path.write_text(json.dumps(json_safe(score), indent=2, allow_nan=False) + "\n")
    return path


def build_report(
    records: list[tuple[Path, dict[str, Any]]],
    tasks: list[Task] = TASKS,
    references: dict[str, Any] | None = None,
    data_dir: Path = DATA_DIR,
    write_run_scores: bool = False,
) -> dict[str, Any]:
    refs = references if references is not None else load_references()
    tasks_by_id = {task.task_id: task for task in tasks}
    scores = []
    for run_dir, record in records:
        score = score_one_run(run_dir, record, tasks_by_id, refs, data_dir)
        if write_run_scores:
            write_run_score(run_dir, score)
        scores.append(score)
    response_scores = [score for score in scores if score["model_response"]]
    analysis_scores = [score for score in response_scores if not score["quarantined"]]
    quarantined_scores = [score for score in response_scores if score["quarantined"]]
    return {
        "scoring_version": SCORING_VERSION,
        "n_saved_records": len(scores),
        "n_model_response_records": len(response_scores),
        "n_analysis_records": len(analysis_scores),
        "n_quarantined_records": len(quarantined_scores),
        "all_records_summary": summarize(scores),
        "analysis_summary": summarize(analysis_scores),
        "by_exec_mode": grouped(analysis_scores, ("exec_mode",)),
        "by_platform": grouped(analysis_scores, ("platform",)),
        "by_prompt_mode": grouped(analysis_scores, ("prompt_mode",)),
        "by_exec_platform": grouped(analysis_scores, ("exec_mode", "platform")),
        "by_task": grouped(analysis_scores, ("task_id",)),
        # Wave visibility: legacy vs pinned-2026-07 aggregates, overall and per task.
        # Several tasks' prompt contracts were respecified post-sweep (arima, ttest,
        # poisson, cox, dynpanel — remediation backlog P3); those legacy and pinned
        # records are currently POOLED in the analysis sample. These groupings expose
        # the split so any pooling/quarantine decision is made on visible numbers.
        # Audit 2026-07-06.
        "by_prompt_wave": grouped(analysis_scores, ("prompt_wave",)),
        "by_task_wave": grouped(analysis_scores, ("task_id", "prompt_wave")),
        "top_failure_labels": failure_counts(analysis_scores)[:20],
        "top_failure_criteria": failure_criteria_counts(analysis_scores)[:20],
        "top_gate_notes": top_gate_notes(analysis_scores),
        "top_criterion_notes": top_criterion_notes(analysis_scores),
        "quarantined": {
            "tasks": sorted(QUARANTINED_TASKS),
            "n_records": len(quarantined_scores),
            "summary": summarize(quarantined_scores),
            "by_exec_mode": grouped(quarantined_scores, ("exec_mode",)),
            "by_platform": grouped(quarantined_scores, ("platform",)),
            "by_task": grouped(quarantined_scores, ("task_id",)),
        },
        "scores": scores,
        "notes": [
            "This report is computed only from saved run files; it does not rerun generated scripts or make model calls.",
            "Output correctness requires the required result-file structure and a numeric reference match.",
            "Specification correctness uses deterministic task-specific evidence patterns in scripts, logs, and method metadata, with the mandatory task-id string masked before matching.",
            "Submission validity checks canonical input hashes, reference-output paths, and hard-coded reference numbers, including full-precision copies caught by prefix matching.",
            "Quarantined tasks were collected under a since-respecified contract; their runs are scored under the legacy contract, excluded from analysis_summary and every by_* grouping, and reported under `quarantined` pending repopulation.",
        ],
    }


def pct(count: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100 * count / total:.1f}%"


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def count_pct(row: dict[str, Any], key: str) -> str:
    return f"{row[key]}/{row['n']} ({pct(row[key], row['n'])})"


def markdown_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        lines.append("| " + " | ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)) + " |")
        if idx == 0:
            lines.append("| " + " | ".join("-" * widths[col] for col in range(len(row))) + " |")
    return lines


def add_group_table(lines: list[str], title: str, grouped_rows: dict[str, dict[str, Any]]) -> None:
    lines.extend(["", f"## {title}", ""])
    rows = [
        [
            "Group",
            "N",
            "Specification",
            "Executability",
            "Output",
            "Submission",
            "Task success",
        ]
    ]
    for label, row in grouped_rows.items():
        rows.append(
            [
                label,
                str(row["n"]),
                count_pct(row, "specification_correctness_pass"),
                count_pct(row, "executability_pass"),
                count_pct(row, "output_correctness_pass"),
                count_pct(row, "submission_validity_pass"),
                count_pct(row, "task_success"),
            ]
        )
    lines.extend(markdown_table(rows))


def markdown_summary(report: dict[str, Any]) -> str:
    summary = report["analysis_summary"]
    lines = [
        "# Task Success Scoring Report",
        "",
        f"- Saved run records: **{report['n_saved_records']}**.",
        f"- Records with a successful model response: **{report['n_model_response_records']}**.",
        f"- Task success: **{summary['task_success']}/{summary['n']} ({pct(summary['task_success'], summary['n'])})**.",
        f"- Specification correctness: **{summary['specification_correctness_pass']}/{summary['n']} ({pct(summary['specification_correctness_pass'], summary['n'])})**.",
        f"- Executability: **{summary['executability_pass']}/{summary['n']} ({pct(summary['executability_pass'], summary['n'])})**.",
        f"- Output correctness: **{summary['output_correctness_pass']}/{summary['n']} ({pct(summary['output_correctness_pass'], summary['n'])})**.",
        f"- Submission validity: **{summary['submission_validity_pass']}/{summary['n']} ({pct(summary['submission_validity_pass'], summary['n'])})**.",
        f"- Recorded cost in analyzed records: **${summary['cost_usd']:.4f}**.",
        "",
        (
            "A run is task successful when four criteria pass for the same "
            "run: specification correctness, executability, output "
            "correctness, and submission validity."
        ),
    ]
    if report.get("quarantined", {}).get("n_records"):
        quarantined = report["quarantined"]
        lines.extend(
            [
                "",
                (
                    f"- Quarantined records (excluded from all numbers above): "
                    f"**{quarantined['n_records']}** across "
                    f"**{', '.join(quarantined['tasks'])}** — collected under a "
                    "since-respecified task contract, scored under the legacy "
                    "contract, pending repopulation."
                ),
            ]
        )
    add_group_table(lines, "By Harness", report["by_exec_mode"])
    add_group_table(lines, "By Platform", report["by_platform"])
    add_group_table(lines, "By Task", report["by_task"])
    if report.get("quarantined", {}).get("n_records"):
        add_group_table(
            lines,
            "Quarantined Tasks (legacy contract, excluded from headline)",
            report["quarantined"]["by_task"],
        )

    lines.extend(["", "## Failure Criteria", ""])
    rows = [["Label", "Count"]]
    for label, count in report["top_failure_criteria"]:
        rows.append([display_label(label), str(count)])
    lines.extend(markdown_table(rows))

    lines.extend(["", "## Frequent Success-Criteria Notes", ""])
    rows = [["Criterion", "Note", "Count"]]
    for criterion, notes in report["top_criterion_notes"].items():
        for note, count in notes[:8]:
            rows.append([display_label(criterion), note, str(count)])
    lines.extend(markdown_table(rows))

    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument(
        "--write-run-scores",
        action="store_true",
        help=(
            f"also write each run's score row to <run_dir>/{RUN_SCORE_NAME} "
            "(new file; run_record.json is never modified)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    references = load_references(args.reference_dir.expanduser().resolve())
    report = build_report(
        load_records(runs_dir),
        references=references,
        data_dir=args.data_dir.expanduser().resolve(),
        write_run_scores=args.write_run_scores,
    )
    json_text = json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n"
    markdown_text = markdown_summary(report)
    for name in ("task_success_report.json", "strict_posthoc_report.json"):
        (out_dir / name).write_text(json_text)
    for name in ("task_success_report.md", "strict_posthoc_report.md"):
        (out_dir / name).write_text(markdown_text)
    print(out_dir / "task_success_report.md")


if __name__ == "__main__":
    main()
