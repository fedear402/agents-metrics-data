* Manual equal-weight ATT(g,t) estimator for T17 (staggered DiD) — Stata reference.
*
* Computes the same long-difference Callaway-Sant'Anna-style ATT(g, t) used by
* manual_python.py and manual_r.R: for each treated cohort g in {4, 6, 8} and
* each post-period year t >= g,
*
*     ATT(g, t)
*         = (mean(y_did | cohort=g, year=t)  - mean(y_did | cohort=g, year=g-1))
*         - (mean(y_did | cohort=-1, year=t) - mean(y_did | cohort=-1, year=g-1))
*
* The reference estimand is the uniform average over the |S| = 12 supported
* (g, t) cells.

clear all
set more off

* Resolve data path relative to this do-file's directory, with a fallback.
local script_dir : pwd
local data_path "data/benchmark/pilot_v3.csv"
capture confirm file "`data_path'"
if _rc != 0 {
    local data_path "../../../data/benchmark/pilot_v3.csv"
}

import delimited "`data_path'", clear

* Build the (g, t) cell list and compute ATT(g, t) for each cell.
local sum_att = 0
local n_cells = 0
local cells_csv ""

foreach g in 4 6 8 {
    local base = `g' - 1
    forvalues t = `g'/9 {
        quietly summarize y_did if cohort == `g' & year == `t', meanonly
        local tT = r(mean)
        quietly summarize y_did if cohort == `g' & year == `base', meanonly
        local tB = r(mean)
        quietly summarize y_did if cohort == -1 & year == `t', meanonly
        local cT = r(mean)
        quietly summarize y_did if cohort == -1 & year == `base', meanonly
        local cB = r(mean)
        local att_gt = (`tT' - `tB') - (`cT' - `cB')
        local sum_att = `sum_att' + `att_gt'
        local n_cells = `n_cells' + 1
        local cells_csv "`cells_csv'`g',`t',`att_gt';"
        display "  ATT(g=`g', t=`t') = " %9.4f `att_gt'
    }
}

local att_uniform = `sum_att' / `n_cells'

display ""
display "Stata reference (manual equal-weight ATT(g,t)):"
display "  n_cells       = " `n_cells'
display "  att_uniform   = " %9.4f `att_uniform'
display "  population    = " %9.4f 20/12
display ""

* Emit a parseable line for downstream verification.
display "JSON_RESULT: " `"{"platform":"stata","att_uniform":"' %9.6f `att_uniform' `","n_cells":"' `n_cells' `",pop":"' %9.6f 20/12 `"}"'
