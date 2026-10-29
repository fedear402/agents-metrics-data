* Manual reference for T07 (Welch two-sample t-test) — Stata.
clear all
set more off
args input_dir
if `"`input_dir'"' == "" local input_dir "tasks/ttest_welch"
local data_path "`input_dir'/pilot_v5_ttest.csv"
import delimited "`data_path'", clear

ttest y, by(group) unequal
local t = r(t)
local df = r(df_t)
local p = r(p)
quietly count if group == 0
local n_a = r(N)
quietly count if group == 1
local n_b = r(N)
* Stata signs ttest as (group 0) - (group 1); flip sign to match Python (b - a) convention.
local t_signed = -1 * `t'
display "JSON_RESULT: " `"{"platform":"stata","t":"' %12.6f `t_signed' `","df":"' %12.6f `df' `","p_value":"' %12.6f `p' `","n_a":"' `n_a' `","n_b":"' `n_b' `"}"'
