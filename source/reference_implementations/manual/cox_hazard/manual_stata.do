* Manual reference for T13 (Cox PH) — Stata.
clear all
set more off

local data_path "data/benchmark/pilot_v4_cox.csv"
capture confirm file "`data_path'"
if _rc != 0 local data_path "../../../data/benchmark/pilot_v4_cox.csv"
import delimited "`data_path'", clear

stset t_obs, failure(event)
stcox treat x, nohr breslow
local b_treat = _b[treat]
local se_treat = _se[treat]
local b_x = _b[x]
local se_x = _se[x]
quietly count if event == 1
local events = r(N)
quietly count
local n_obs = r(N)
display ""
display "Stata reference T13 (Cox PH):"
display "  coef_treat   = " %12.6f `b_treat'
display "  se_treat     = " %12.6f `se_treat'
display "  coef_x       = " %12.6f `b_x'
display "  se_x         = " %12.6f `se_x'
display "  events       = " `events'
display "  n_obs        = " `n_obs'
display "JSON_RESULT: " `"{"platform":"stata","coef_treat":"' %12.6f `b_treat' `","se_treat":"' %12.6f `se_treat' `","coef_x":"' %12.6f `b_x' `","se_x":"' %12.6f `se_x' `","events":"' `events' `","n_obs":"' `n_obs' `"}"'
