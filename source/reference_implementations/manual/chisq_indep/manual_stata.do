* Manual reference for T09 (chi-square test of independence) — Stata.
clear all
set more off
args input_dir
if `"`input_dir'"' == "" local input_dir "tasks/chisq_indep"
local data_path "`input_dir'/pilot_v5_chisq.csv"
import delimited "`data_path'", clear

tabulate row col, chi2
local chi2 = r(chi2)
local p = r(p)
local df = (r(r) - 1) * (r(c) - 1)
local n_obs = _N
display "JSON_RESULT: " `"{"platform":"stata","chi2":"' %12.6f `chi2' `","df":"' `df' `","p_value":"' %12.6f `p' `","n_obs":"' `n_obs' `"}"'
