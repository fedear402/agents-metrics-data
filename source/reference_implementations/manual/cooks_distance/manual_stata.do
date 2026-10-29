* Manual reference for T06 (Cook's distance top-5) — Stata.
clear all
set more off

args input_dir
if `"`input_dir'"' == "" local input_dir "tasks/cooks_distance"
local data_path "`input_dir'/pilot_v5_collinear.csv"
import delimited "`data_path'", clear

gen long row_id = _n - 1   // zero-based to match Python and R; renamed to avoid Stata's [..] equation parsing
regress y x1 x2 x3 x4 x5
predict cd, cooksd
local n_obs = _N
gsort -cd
forvalues k = 1/5 {
    quietly summarize row_id in `k'/`k', meanonly
    local id`k' = r(mean)
    quietly summarize cd in `k'/`k', meanonly
    local cd`k' = r(mean)
}
display "JSON_RESULT: " `"{"platform":"stata","top5_obs_ids":["' `id1' `","' `id2' `","' `id3' `","' `id4' `","' `id5' `"],"top5_cooks_d":["' %12.6f `cd1' `","' %12.6f `cd2' `","' %12.6f `cd3' `","' %12.6f `cd4' `","' %12.6f `cd5' `"],"n_obs":"' `n_obs' `"}"'
