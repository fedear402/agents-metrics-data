"""Generate ``paper/manuscript/numbers.tex`` from the analysis JSON files.

The manuscript prose references analysis numbers through named LaTeX macros
instead of literal digits. This builder computes every macro value from the
canonical analysis outputs and writes the macro file the manuscript inputs
from ``preamble.tex``.

Inputs (read only):

* ``data/analysis/manuscript_numbers.json``
* ``data/analysis/full_analysis.json``
* ``data/analysis/factorial_by_model.json``

Output:

* ``paper/manuscript/numbers.tex`` (or ``--out PATH``)

Conventions, also stated in the generated file header:

* Each macro renders the bare value string exactly as the prose prints it.
  Signs, dollar signs, percent words, and unit words stay in the prose.
* Rounding is half-up on the exact ratio unless a macro notes otherwise.
* Every macro carries a ``%TC:macroword`` weight equal to the TeXcount word
  count of its rendered value, so the automated submission word count is
  unchanged relative to literal numbers.

Run from the repository root:

    uv run python code/benchmark/build_numbers_tex.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "analysis"
DEFAULT_OUT = REPO_ROOT / "paper" / "manuscript" / "numbers.tex"

# The paper's main specification: two model arms (decision record
# 2026-08-21_two-model-main-specification). "Pooled" means these two arms
# pooled. The corpus keeps a third arm; no macro here reads it.
MODELS = ("sonnet", "codex")
#: Estimation keys: the two arms plus the two-model pooled fit.
ESTIMATION_KEYS = (*MODELS, "pooled")
PROMPTS = ("zero_shot", "few_shot")
INTERFACES = ("chatbot", "one-repair", "constrained agent")
PLATFORMS = ("python", "r", "stata")

# LaTeX-facing arm names used in macro names.
ARM = {"sonnet": "Sonnet", "codex": "Codex", "pooled": "Pooled"}
IFACE = {"chatbot": "Chatbot", "one-repair": "OneRepair", "constrained agent": "Agent"}


# ---------------------------------------------------------------------------
# Formatting helpers. All rounding is Decimal half-up, never float round().
# ---------------------------------------------------------------------------


def _dec(x: object) -> Decimal:
    if isinstance(x, Decimal):
        return x
    if isinstance(x, float):
        return Decimal(repr(x))
    return Decimal(str(x))


def pct_ratio(successes: object, n: object, places: str = "0.1") -> str:
    """Percent from an exact count ratio, half-up."""
    value = _dec(successes) * Decimal(100) / _dec(n)
    return str(value.quantize(Decimal(places), rounding=ROUND_HALF_UP))


def fmt1(x: object) -> str:
    """One-decimal half-up rounding of a stored value."""
    return str(_dec(x).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def fmt_int(x: object) -> str:
    return str(_dec(x).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def fmt_int_comma(x: object) -> str:
    return f"{int(_dec(x).quantize(Decimal('1'), rounding=ROUND_HALF_UP)):,d}"


def fmt_pp_abs(fraction: object) -> str:
    """Magnitude of a contrast estimate, in percentage points, one decimal."""
    value = abs(_dec(fraction)) * Decimal(100)
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def fmt_p(p: object) -> str:
    """Wild-bootstrap p-value at three decimals, trailing zeros kept."""
    return str(_dec(p).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def fmt_money2(x: object) -> str:
    return str(_dec(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def latex_version(version: str) -> str:
    """Render a scorer version string for use inside ``\\texttt{}``."""
    return version.replace("_", "\\_\\allowbreak ")


def texcount_weight(value: str) -> int:
    """TeXcount word weight of the rendered value in text mode."""
    return 1 + value.count(",")


# ---------------------------------------------------------------------------
# Macro registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Macro:
    name: str
    value: str
    comment: str = ""


def _headline_counts(fa: dict, model: str, prompt: str, axis: str, key: str) -> tuple[int, int]:
    cell = fa["headline"][model][prompt][axis][key]
    return int(cell["successes"]), int(cell["n"])


def _pooled(fa: dict, model: str, axis: str, key: str) -> tuple[int, int]:
    s = n = 0
    for prompt in PROMPTS:
        cs, cn = _headline_counts(fa, model, prompt, axis, key)
        s += cs
        n += cn
    return s, n


def _overall(fa: dict, model: str) -> tuple[int, int]:
    s = n = 0
    for prompt in PROMPTS:
        cell = fa["headline"][model][prompt]["overall"]
        s += int(cell["successes"])
        n += int(cell["n"])
    return s, n


def _nine_cell(fa: dict, model: str, prompt: str, platform: str, iface: str) -> dict:
    for cell in fa["nine_cell"][model][prompt]:
        if cell["platform"] == platform and cell["interface_label"] == iface:
            return cell
    raise KeyError(f"nine_cell missing {model}/{prompt}/{platform}/{iface}")


def _task_cells(mn: dict, model: str, task: str):
    per = mn["per_task_model_platform"][model]
    for prompt in PROMPTS:
        for platform in PLATFORMS:
            for iface in INTERFACES:
                yield prompt, platform, iface, per[prompt][task][platform][iface]


def _task_rate(mn: dict, model: str, task: str, *, platform: str | None = None,
               iface: str | None = None) -> str:
    s = n = 0
    for _prompt, cell_platform, cell_iface, cell in _task_cells(mn, model, task):
        if platform is not None and cell_platform != platform:
            continue
        if iface is not None and cell_iface != iface:
            continue
        s += int(cell["successes"])
        n += int(cell["n"])
    return pct_ratio(s, n)


LOUD_GATES = ("execution", "artifact")
SILENT_GATES = ("numeric", "method", "integrity")


def _cascade_sum(fa: dict, model: str, iface: str, gates: tuple[str, ...],
                 prompts: tuple[str, ...] = PROMPTS) -> tuple[int, int]:
    total = 0
    n = 0
    for prompt in prompts:
        cell = fa["extended"]["criterion_cascade"]["models"][model][prompt][iface]
        counts = cell["counts"]
        total += sum(int(counts[g]) for g in gates)
        n += int(cell["n"])
    return total, n


def build_macros(mn: dict, fa: dict, fbm: dict) -> list[tuple[str, list[Macro]]]:
    groups: list[tuple[str, list[Macro]]] = []

    # -- Sample sizes -------------------------------------------------------
    total_s, total_n = 0, 0
    for model in MODELS:
        s, n = _overall(fa, model)
        total_s += s
        total_n += n
    per_arm_n = {model: _overall(fa, model)[1] for model in MODELS}
    if len(set(per_arm_n.values())) != 1:
        raise ValueError(f"unbalanced arms: {per_arm_n}")
    arm_n = next(iter(per_arm_n.values()))
    prompt_n = {fa["headline"][m][p]["overall"]["n"] for m in MODELS for p in PROMPTS}
    if len(prompt_n) != 1:
        raise ValueError(f"unbalanced prompt halves: {prompt_n}")
    reps = {fbm[key]["wcb"]["reps"] for key in ESTIMATION_KEYS}
    if len(reps) != 1:
        raise ValueError(f"wild-bootstrap replication counts differ: {reps}")
    groups.append((
        "Sample sizes (Sections 3 and 4)",
        [
            Macro("numRunsTotal", fmt_int_comma(total_n)),
            Macro("numRunsPerArm", fmt_int_comma(arm_n)),
            Macro("numRunsPerPrompt", fmt_int_comma(next(iter(prompt_n)))),
            Macro("numWcbReps", f"{next(iter(reps)):,d}".replace(",", "{,}"),
                  "math mode only"),
        ],
    ))

    # -- Headline task success ---------------------------------------------
    headline = [Macro("numPooledSuccess", pct_ratio(total_s, total_n))]
    for model in MODELS:
        s, n = _overall(fa, model)
        headline.append(Macro(f"num{ARM[model]}Overall", pct_ratio(s, n)))
    for model in MODELS:
        for iface in INTERFACES:
            s, n = _pooled(fa, model, "by_interface", iface)
            headline.append(Macro(f"num{ARM[model]}{IFACE[iface]}", pct_ratio(s, n)))
    groups.append(("Headline task success (Sections 1 and 4.1)", headline))

    # -- Software rates -----------------------------------------------------
    software = []
    for platform, tex in (("python", "Python"), ("r", "R")):
        s, n = _pooled(fa, "sonnet", "by_platform", platform)
        software.append(Macro(f"numSonnet{tex}", pct_ratio(s, n)))
    software.append(Macro(
        "numSonnetStata",
        fmt1(mn["headline_sonnet"]["pooled_prompts"]["by_platform"]["stata"]["success_pct"]),
        "rounds the stored two-decimal rate, as printed",
    ))
    for iface, tex in (("chatbot", "ChatbotZS"), ("constrained agent", "AgentZS")):
        cell = _nine_cell(fa, "sonnet", "zero_shot", "stata", iface)
        software.append(Macro(
            f"numSonnetStata{tex}", pct_ratio(cell["successes"], cell["n"])))
    groups.append(("Software rates (Sections 1 and 4.2)", software))

    # -- Estimated contrasts ------------------------------------------------
    contrasts = []
    spec = [
        ("H1_r", "HOneR", True),
        ("H1_stata", "HOneStata", True),
        ("H2_fewshot", "HTwo", True),
        ("H3_self_repair", "HThreeARepair", True),
        ("H3_tool_loop", "HThreeBAgent", True),
        # The paper reports one prompt-by-agency interaction, few-shot by
        # constrained agent, named H4 (decision record
        # 2026-08-24_h4-single-interaction). The self-repair interaction and
        # the joint test stay in the analysis JSON as diagnostics and get no
        # macro.
        ("H4_tool_loop", "HFour", True),
    ]
    for key, stem, _with_p in spec:
        for estimation_key in ESTIMATION_KEYS:
            c = fbm[estimation_key]["contrasts"][key]
            arm = ARM[estimation_key]
            contrasts.append(Macro(f"num{stem}{arm}", fmt_pp_abs(c["estimate"]),
                                   "magnitude; sign stays in the prose"))
            contrasts.append(Macro(f"num{stem}{arm}SE", fmt_pp_abs(c["se_cr1"]),
                                   "CR1 cluster-by-task standard error"))
            p_value = fmt_p(c["p_wcb"])
            # A p-value below 0.001 gets no macro; the prose prints $p<0.001$.
            if p_value != "0.000":
                contrasts.append(Macro(f"num{stem}{arm}P", p_value))
    for estimation_key in ESTIMATION_KEYS:
        p_value = fmt_p(fbm[estimation_key]["h1_joint_p_wcb"])
        if p_value != "0.000":
            contrasts.append(
                Macro(f"numHOneJoint{ARM[estimation_key]}P", p_value))
    groups.append(("Estimated contrasts and wild-bootstrap p-values "
                   "(Sections 1, 4.1 to 4.2, and 5)", contrasts))

    # -- Cross-model H4 difference and software-gap diagnostics -------------
    extra = []
    pair_stems = (("sonnet_minus_codex", "SonnetCodex"),)
    diffs = fbm["cross_model"]["differences"]
    for key, stem in (("H4_tool_loop", "HFourDiff"),):
        for pair, tail in pair_stems:
            d = diffs[key][pair]
            extra.append(Macro(f"num{stem}{tail}", fmt_pp_abs(d["estimate"]),
                               "magnitude; sign stays in the prose"))
            p_value = fmt_p(d["p_t20"])
            if p_value != "0.000":
                extra.append(Macro(f"num{stem}{tail}P", p_value))
    for key, stem in (("GAP_stata_repair_zs", "numStataGapOneRepair"),
                      ("GAP_stata_agent_zs", "numStataGapAgent"),
                      ("GAP_stata_avg", "numStataGapAvg")):
        for estimation_key in ESTIMATION_KEYS:
            c = fbm[estimation_key]["contrasts"][key]
            arm = ARM[estimation_key]
            extra.append(Macro(f"{stem}{arm}", fmt_pp_abs(c["estimate"]),
                               "magnitude; sign stays in the prose"))
            p_value = fmt_p(c["p_wcb"])
            if p_value != "0.000":
                extra.append(Macro(f"{stem}{arm}P", p_value))
    groups.append(("Cross-model H4 difference and software-gap diagnostics "
                   "(Section 4.2)", extra))

    # -- Raw prompt gains ---------------------------------------------------
    gains = []
    for model in MODELS:
        for iface in INTERFACES:
            fs, fn = _headline_counts(fa, model, "few_shot", "by_interface", iface)
            zs, zn = _headline_counts(fa, model, "zero_shot", "by_interface", iface)
            if fn != zn:
                raise ValueError(f"prompt halves differ in n for {model}/{iface}")
            gains.append(Macro(
                f"numPromptGain{IFACE[iface]}{ARM[model]}", pct_ratio(fs - zs, fn)))
    groups.append(("Raw prompt gains by agency condition, few-shot minus "
                   "zero-shot (Section 4.2)", gains))

    # -- pass@k -------------------------------------------------------------
    # Unbiased pass@k on the twelve-repetition paper sample, prompt conditions
    # pooled (code/benchmark/pass_at_k.py). The prose quotes the two ends of
    # each curve; the appendix figure draws every k.
    passk = mn["pass_at_k_paper"]
    if passk["k_values"][0] != 1 or passk["k_values"][-1] != 12:
        raise ValueError(f"pass@k curve does not run 1 to 12: {passk['k_values']}")
    pak = []
    for model in MODELS:
        block = passk["by_model"][model]
        for k, tex in ((1, "One"), (12, "Twelve")):
            pak.append(Macro(
                f"numPass{tex}{ARM[model]}",
                fmt1(block["overall"]["pass_at_k_pct"][str(k)])))
        for iface in INTERFACES:
            for k, tex in ((1, "One"), (12, "Twelve")):
                pak.append(Macro(
                    f"numPass{tex}{IFACE[iface]}{ARM[model]}",
                    fmt1(block["by_agency"][iface]["pass_at_k_pct"][str(k)])))
    groups.append(("pass@k on the twelve-repetition sample, prompts pooled "
                   "(Appendix C)", pak))

    # -- Task-level rates ---------------------------------------------------
    tasks = [
        Macro("numTaskStaggeredSonnet", _task_rate(mn, "sonnet", "staggered_did")),
        Macro("numTaskDynpanelSonnet", _task_rate(mn, "sonnet", "dynpanel_ab")),
        Macro("numTaskAnovaSonnet", _task_rate(mn, "sonnet", "anova_oneway")),
        Macro("numDynpanelSonnetStata",
              _task_rate(mn, "sonnet", "dynpanel_ab", platform="stata")),
        Macro("numDynpanelSonnetR",
              _task_rate(mn, "sonnet", "dynpanel_ab", platform="r")),
        Macro("numDynpanelSonnetPython",
              _task_rate(mn, "sonnet", "dynpanel_ab", platform="python")),
    ]
    for prompt, tex in (("zero_shot", "ZeroShot"), ("few_shot", "FewShot")):
        cell = mn["per_task_sonnet"]["dynpanel_ab"]["by_prompt"][prompt]
        tasks.append(Macro(f"numDynpanelSonnet{tex}",
                           pct_ratio(cell["successes"], cell["n"])))
    for iface, tex in (("chatbot", "Chatbot"), ("constrained agent", "Agent")):
        tasks.append(Macro(
            f"numTaskStaggered{tex}Sonnet",
            _task_rate(mn, "sonnet", "staggered_did", iface=iface)))
    for model in MODELS:
        tasks.append(Macro(
            f"numDynpanelAgent{ARM[model]}",
            _task_rate(mn, model, "dynpanel_ab", iface="constrained agent")))
    groups.append(("Task-level rates (Section 4.3 and Appendix C)", tasks))

    # -- Silent failure and scoring checks ---------------------------------
    silent = []
    for iface in INTERFACES:
        f = n = 0
        for prompt in PROMPTS:
            cell = mn["silent_failure"]["models"]["sonnet"][prompt][iface]
            f += int(cell["silent_failures"])
            n += int(cell["n_clean"])
        silent.append(Macro(f"numSilent{IFACE[iface]}Sonnet", pct_ratio(f, n)))
    # The snapshot's own overall block covers the three-arm corpus, so the
    # paper's pooled gap is summed over the paper arms instead.
    svu = fa["strict_vs_unified"]
    paper_arms = [svu["by_model"][model]["overall"] for model in MODELS]
    silent.append(Macro(
        "numStrictGapOverall",
        pct_ratio(
            sum(int(b["task_success_n"]) - int(b["task_success_strict_n"])
                for b in paper_arms),
            sum(int(b["n"]) for b in paper_arms))))
    sm = svu["by_model"]["sonnet"]["overall"]
    silent.append(Macro(
        "numStrictGapSonnet",
        pct_ratio(sm["task_success_n"] - sm["task_success_strict_n"], sm["n"])))
    tol = mn["tolerance_sensitivity_sonnet"]
    silent += [
        Macro("numTolBaseZS", fmt1(tol["zero_shot"]["task_success_pct_by_scale"]["1.0"])),
        Macro("numTolHalfZS", fmt1(tol["zero_shot"]["task_success_pct_by_scale"]["0.5"])),
        Macro("numTolBaseFS", fmt1(tol["few_shot"]["task_success_pct_by_scale"]["1.0"])),
        Macro("numTolHalfFS", fmt1(tol["few_shot"]["task_success_pct_by_scale"]["0.5"])),
        Macro("numTolDoubleFS", fmt1(tol["few_shot"]["task_success_pct_by_scale"]["2.0"])),
    ]
    groups.append(("Silent failure and scoring checks (Sections 4.3 to 4.4, "
                   "Appendix C)", silent))

    # -- Failure anatomy ----------------------------------------------------
    anatomy = []
    for iface, tex in (("chatbot", "Chatbot"), ("constrained agent", "Agent")):
        loud, n = _cascade_sum(fa, "sonnet", iface, LOUD_GATES, prompts=("zero_shot",))
        anatomy.append(Macro(f"numExecArtifact{tex}ZSSonnet", pct_ratio(loud, n)))
    for iface, tex in (("chatbot", "Chatbot"), ("constrained agent", "Agent")):
        numeric, n = _cascade_sum(fa, "sonnet", iface, ("numeric",), prompts=("zero_shot",))
        anatomy.append(Macro(f"numNumeric{tex}ZSSonnet", pct_ratio(numeric, n)))
    for iface, tex in (("chatbot", "Chatbot"), ("constrained agent", "Agent")):
        loud, n = _cascade_sum(fa, "sonnet", iface, LOUD_GATES)
        anatomy.append(Macro(f"numLoud{tex}Sonnet", pct_ratio(loud, n)))
    for iface, tex in (("chatbot", "Chatbot"), ("constrained agent", "Agent")):
        s, n = _cascade_sum(fa, "sonnet", iface, SILENT_GATES)
        anatomy.append(Macro(f"numSilentPerHundred{tex}Sonnet", pct_ratio(s, n)))
    codex_rates = []
    for iface in INTERFACES:
        s, n = _cascade_sum(fa, "codex", iface, SILENT_GATES)
        codex_rates.append(Decimal(pct_ratio(s, n)))
    anatomy.append(Macro("numSilentPerHundredMinCodex", str(min(codex_rates))))
    anatomy.append(Macro("numSilentPerHundredMaxCodex", str(max(codex_rates))))
    for model in MODELS:
        loud, _ = _cascade_sum(fa, model, "constrained agent", LOUD_GATES)
        s, _ = _cascade_sum(fa, model, "constrained agent", SILENT_GATES)
        anatomy.append(Macro(
            f"numSilentShareAgent{ARM[model]}", fmt_int(_dec(s) * 100 / _dec(s + loud)),
            "share of remaining failures that are silent"))
    groups.append(("Failure anatomy, first-failing-check diagnostics "
                   "(Section 4.3 and Appendix C)", anatomy))

    # -- Cost ---------------------------------------------------------------
    sn = mn["cost"]["sonnet_native"]
    pooled = sn["pooled_prompts"]
    priced = sum(int(pooled["by_interface"][i]["n_priced"]) for i in INTERFACES)
    cost = [Macro("numSonnetPricedCoveragePct", pct_ratio(priced, arm_n))]

    def money(node: dict, field: str) -> str:
        stored = node.get(f"{field}_2dp")
        computed = fmt_money2(node[field])
        if stored is not None and stored != computed:
            raise ValueError(f"stored {field}_2dp {stored!r} != computed {computed!r}")
        return computed

    cost += [
        Macro("numSonnetMeanCostChatbot", money(pooled["by_interface"]["chatbot"], "mean_cost_usd")),
        Macro("numSonnetMeanCostAgent", money(pooled["by_interface"]["constrained agent"], "mean_cost_usd")),
        Macro("numSonnetCpsChatbot", money(pooled["by_interface"]["chatbot"], "cost_per_success_usd")),
        Macro("numSonnetCpsOneRepair", money(pooled["by_interface"]["one-repair"], "cost_per_success_usd")),
        Macro("numSonnetCpsAgent", money(pooled["by_interface"]["constrained agent"], "cost_per_success_usd")),
        Macro("numSonnetIncrementalCps",
              money(pooled["conversion_price_chatbot_to_constrained_agent"],
                    "usd_per_additional_success")),
        Macro("numSonnetMedianCostChatbot", money(pooled["by_interface"]["chatbot"], "median_cost_usd")),
        Macro("numSonnetMedianCostOneRepair", money(pooled["by_interface"]["one-repair"], "median_cost_usd")),
        Macro("numSonnetMedianCostAgent", money(pooled["by_interface"]["constrained agent"], "median_cost_usd")),
        Macro("numSonnetMedianCostChatbotZS", money(sn["zero_shot"]["by_interface"]["chatbot"], "median_cost_usd")),
        Macro("numSonnetMedianCostOneRepairZS", money(sn["zero_shot"]["by_interface"]["one-repair"], "median_cost_usd")),
        Macro("numSonnetMedianCostAgentZS", money(sn["zero_shot"]["by_interface"]["constrained agent"], "median_cost_usd")),
        Macro("numSonnetMedianCostChatbotFS", money(sn["few_shot"]["by_interface"]["chatbot"], "median_cost_usd")),
        Macro("numSonnetMedianCostOneRepairFS", money(sn["few_shot"]["by_interface"]["one-repair"], "median_cost_usd")),
        Macro("numSonnetMedianCostAgentFS", money(sn["few_shot"]["by_interface"]["constrained agent"], "median_cost_usd")),
    ]
    cost += [
        Macro("numSonnetMedianCostPython",
              money(pooled["by_platform"]["python"], "median_cost_usd")),
        Macro("numSonnetMedianCostR",
              money(pooled["by_platform"]["r"], "median_cost_usd")),
        Macro("numSonnetMedianCostStata", money(pooled["by_platform"]["stata"], "median_cost_usd")),
        Macro("numSonnetCpsPython", money(pooled["by_platform"]["python"], "cost_per_success_usd")),
        Macro("numSonnetCpsR", money(pooled["by_platform"]["r"], "cost_per_success_usd")),
        Macro("numSonnetCpsStata", money(pooled["by_platform"]["stata"], "cost_per_success_usd")),
        Macro("numCostPerSuccessAgentZS",
              money(sn["zero_shot"]["by_interface"]["constrained agent"], "cost_per_success_usd"),
              "Claude Sonnet 4.6 arm"),
        Macro("numSonnetCpsAgentFS",
              money(sn["few_shot"]["by_interface"]["constrained agent"], "cost_per_success_usd")),
        Macro("numCodexCpsAgentZS",
              money(mn["cost"]["gpt54_imputed"]["zero_shot"]["by_interface"]["constrained agent"],
                    "imputed_cost_per_success_usd")),
        Macro("numCodexCpsAgentFS",
              money(mn["cost"]["gpt54_imputed"]["few_shot"]["by_interface"]["constrained agent"],
                    "imputed_cost_per_success_usd")),
        Macro("numConvDeltaZS",
              fmt1(sn["zero_shot"]["conversion_price_chatbot_to_constrained_agent"]["delta_success_pp"])),
        Macro("numConvPriceZS",
              money(sn["zero_shot"]["conversion_price_chatbot_to_constrained_agent"],
                    "usd_per_additional_success")),
        Macro("numConvDeltaFS",
              fmt1(sn["few_shot"]["conversion_price_chatbot_to_constrained_agent"]["delta_success_pp"])),
        Macro("numConvPriceFS",
              money(sn["few_shot"]["conversion_price_chatbot_to_constrained_agent"],
                    "usd_per_additional_success")),
    ]
    gpt = mn["cost"]["gpt54_imputed"]
    cost += [
        Macro("numCodexMeanCostChatbot",
              fmt_money2(gpt["pooled_prompts"]["by_interface"]["chatbot"]
                         ["imputed_mean_cost_usd"])),
        Macro("numCodexMeanCostAgent",
              fmt_money2(gpt["pooled_prompts"]["by_interface"]["constrained agent"]
                         ["imputed_mean_cost_usd"])),
        Macro("numCodexCpsChatbot",
              money(gpt["pooled_prompts"]["by_interface"]["chatbot"],
                    "imputed_cost_per_success_usd")),
        Macro("numCodexCpsOneRepair",
              money(gpt["pooled_prompts"]["by_interface"]["one-repair"],
                    "imputed_cost_per_success_usd")),
        Macro("numCodexCpsAgent",
              money(gpt["pooled_prompts"]["by_interface"]["constrained agent"],
                    "imputed_cost_per_success_usd")),
    ]
    cost += [
        Macro("numSonnetFewShotCents",
              fmt1(sn["few_shot_increment"]["overall"]["mean_increment_cents"]),
              "few-shot minus zero-shot cost per run, in cents"),
        Macro("numCodexFewShotCents",
              fmt1(gpt["few_shot_increment"]["overall"]["mean_increment_cents"]),
              "few-shot minus zero-shot imputed cost per run, in cents"),
    ]
    prices = mn["cost"]["gpt54_imputed"]["pricing_usd_per_mtok"]
    cost += [
        Macro("numGptPriceInput", fmt_money2(prices["input"])),
        Macro("numGptPriceCachedInput", fmt_money2(prices["cached_input"])),
        Macro("numGptPriceOutput", fmt_money2(prices["output"])),
    ]
    codex_agent_zs = fa["extended"]["cost"]["models"]["codex"]["zero_shot"]["constrained agent"]
    means = codex_agent_zs["tokens"]["mean_per_run"]
    cost += [
        Macro("numCodexAgentZSInputTokens", fmt_int_comma(means["input_tokens"])),
        Macro("numCodexAgentZSCachedTokens", fmt_int_comma(means["cached_input_tokens"])),
        Macro("numCodexAgentZSOutputTokens", fmt_int_comma(means["output_tokens"])),
        Macro("numCodexAgentZSMeanCost", fmt_money2(codex_agent_zs["imputed"]["mean_cost_usd"])),
    ]
    groups.append(("Cost, Claude Sonnet 4.6 dollars unless the name says "
                   "Codex (Sections 1 and 5, Appendix C)", cost))

    # -- Self-repair revision disclosure ------------------------------------
    disclosure = []
    for model in MODELS:
        block = mn["revision_disclosure"]["models"][model]["overall"]
        if int(block["n_unknown"]) != 0:
            raise ValueError(
                f"{model}: {block['n_unknown']} self-repair rows carry no "
                "attempt count; the disclosure macros assume full coverage")
        disclosure.append(Macro(
            f"numRevShareGtOne{ARM[model]}",
            pct_ratio(block["n_gt1_revision"], block["n_known"]),
            "share of self-repair runs that used more than one revision"))
        disclosure.append(Macro(
            f"numRevMax{ARM[model]}", fmt_int(block["max_revisions"]),
            "largest revision count observed in the arm"))
    groups.append(("Self-repair revision disclosure (Appendix B)",
                   disclosure))

    # -- Runtime ------------------------------------------------------------
    runtime = []
    for model in MODELS:
        by_model = mn["runtime"]["by_model"][model]
        for iface in INTERFACES:
            runtime.append(Macro(
                f"numRuntime{IFACE[iface]}Median{ARM[model]}",
                fmt1(by_model["by_interface"][iface]["median_provider_wall_seconds"])))
    for model in MODELS:
        runtime.append(Macro(
            f"numRuntimeRatio{ARM[model]}",
            fmt1(mn["runtime"]["by_model"][model]["constrained_agent_to_chatbot_ratio"])))
    runtime.append(Macro("numProviderHoursTotal", fmt1(mn["runtime"]["total_provider_hours"])))
    for model in MODELS:
        runtime.append(Macro(
            f"numProviderHours{ARM[model]}",
            fmt1(mn["runtime"]["total_provider_hours_by_model"][model])))
    groups.append(("Runtime, provider-call seconds and hours (Sections 1 and 5)", runtime))

    # -- Scoring rule versions ---------------------------------------------
    groups.append((
        "Scoring rule versions (Appendix B)",
        [
            Macro("numScoringVersion", latex_version(mn["scoring_version"]),
                  "for use inside \\texttt{}"),
            Macro("numPrimaryOutcomeVersion", latex_version(mn["primary_outcome_version"]),
                  "for use inside \\texttt{}"),
        ],
    ))

    names = [m.name for _title, ms in groups for m in ms]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate macro names: {dupes}")
    return groups


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_numbers_tex(data_dir: Path = DATA_DIR) -> str:
    mn = json.loads((data_dir / "manuscript_numbers.json").read_text())
    fa = json.loads((data_dir / "full_analysis.json").read_text())
    fbm = json.loads((data_dir / "factorial_by_model.json").read_text())
    groups = build_macros(mn, fa, fbm)
    count = sum(len(ms) for _t, ms in groups)

    lines = [
        "% ==========================================================================",
        "% numbers.tex -- generated by code/benchmark/build_numbers_tex.py.",
        "% Do not edit by hand; regenerate instead.",
        "%",
        "% Sources:",
        "%   data/analysis/manuscript_numbers.json",
        "%   data/analysis/full_analysis.json",
        "%   data/analysis/factorial_by_model.json",
        f"% Primary outcome version: {mn['primary_outcome_version']}",
        f"% Scoring version: {mn['scoring_version']}",
        f"% Macro count: {count}",
        "%",
        "% Convention: each macro renders the bare value string exactly as the prose",
        "% prints it. Signs, dollar signs, percent words, and unit words stay in the",
        "% prose. Invoke as \\numName{} in text and \\numName in math. Rounding is",
        "% half-up on the exact ratio unless a macro comment says otherwise.",
        "%",
        "% The %TC:macroword line above each macro gives TeXcount the word weight of",
        "% the rendered value, so the automated submission word count matches the",
        "% count with literal numbers. submission_wordcount.tex inputs this file.",
        "%",
        "% Refresh: run code/benchmark/refresh_manuscript.sh from the repository",
        "% root. It rebuilds the analysis JSONs, the installed exhibits, this file,",
        "% and the manuscript PDFs, in that order.",
        "% ==========================================================================",
    ]
    for title, macros in groups:
        lines.append("")
        lines.append(f"% --- {title}")
        for macro in macros:
            lines.append(f"%TC:macroword \\{macro.name} {texcount_weight(macro.value)}")
            suffix = f"  % {macro.comment}" if macro.comment else ""
            lines.append(f"\\newcommand{{\\{macro.name}}}{{{macro.value}}}{suffix}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output path (default: paper/manuscript/numbers.tex)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="directory holding the analysis JSON files")
    args = parser.parse_args()
    content = render_numbers_tex(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content)
    macro_count = content.count("\\newcommand{")
    print(f"wrote {args.out} ({macro_count} macros)")


if __name__ == "__main__":
    main()
