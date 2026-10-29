* Manual reference for T11-precursor (binary logit on the pilot v1 dataset) — Stata.
clear all
set more off

args input_dir
if `"`input_dir'"' == "" local input_dir "tasks/logit"
local data_path "`input_dir'/pilot.csv"
import delimited "`data_path'", clear

logit y_bin_logit x1 x2
local b_const = _b[_cons]
local se_const = _se[_cons]
local b_x1 = _b[x1]
local se_x1 = _se[x1]
local b_x2 = _b[x2]
local se_x2 = _se[x2]
local n_obs = e(N)

display ""
display "Stata reference logit:"
display "  const = " %12.6f `b_const' "  SE = " %12.6f `se_const'
display "  x1    = " %12.6f `b_x1' "  SE = " %12.6f `se_x1'
display "  x2    = " %12.6f `b_x2' "  SE = " %12.6f `se_x2'
display "  n_obs = " `n_obs'
display "JSON_RESULT: " `"{"platform":"stata","const":"' %12.6f `b_const' `","se_const":"' %12.6f `se_const' `","x1":"' %12.6f `b_x1' `","se_x1":"' %12.6f `se_x1' `","x2":"' %12.6f `b_x2' `","se_x2":"' %12.6f `se_x2' `","n_obs":"' `n_obs' `"}"'
