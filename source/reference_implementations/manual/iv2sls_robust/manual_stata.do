* Manual reference for T15 (just-identified 2SLS with robust SE) — Stata.
clear all
set more off

args input_dir
if `"`input_dir'"' == "" local input_dir "tasks/iv2sls_robust"
local data_path "`input_dir'/pilot_v2.csv"
import delimited "`data_path'", clear

ivregress 2sls y_iv (x1_iv = z), vce(robust)
local b_const = _b[_cons]
local se_const = _se[_cons]
local b_x1iv = _b[x1_iv]
local se_x1iv = _se[x1_iv]
local n_obs = e(N)

display ""
display "Stata reference T15 (2SLS robust):"
display "  const   = " %12.6f `b_const' "  SE = " %12.6f `se_const'
display "  x1_iv   = " %12.6f `b_x1iv' "  SE = " %12.6f `se_x1iv'
display "  n_obs   = " `n_obs'
display "JSON_RESULT: " `"{"platform":"stata","const":"' %12.6f `b_const' `","se_const":"' %12.6f `se_const' `","x1_iv":"' %12.6f `b_x1iv' `","se_x1_iv":"' %12.6f `se_x1iv' `","n_obs":"' `n_obs' `"}"'
