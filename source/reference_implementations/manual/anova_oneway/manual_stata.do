* Manual reference for T08 (one-way ANOVA) — Stata.
clear all
set more off
local data_path "data/benchmark/pilot_v5_anova.csv"
capture confirm file "`data_path'"
if _rc != 0 local data_path "../../../data/benchmark/pilot_v5_anova.csv"
import delimited "`data_path'", clear

oneway y group
local F = r(F)
local df_num = r(df_m)
local df_den = r(df_r)
local p = Ftail(`df_num', `df_den', `F')
local n_obs = _N
display "JSON_RESULT: " `"{"platform":"stata","F":"' %12.6f `F' `","df_num":"' `df_num' `","df_den":"' `df_den' `","p_value":"' %12.6f `p' `","n_obs":"' `n_obs' `"}"'
