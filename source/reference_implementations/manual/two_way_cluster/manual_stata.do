* Manual reference for T04 (OLS with two-way cluster-robust SE, CGM) — Stata.
clear all
set more off

local data_path "data/benchmark/pilot_v3.csv"
capture confirm file "`data_path'"
if _rc != 0 local data_path "../../../data/benchmark/pilot_v3.csv"
import delimited "`data_path'", clear

* Two-way clustering via the user-written reghdfe/ivreg2 or via Petersen's
* approach. Stata's `regress, vce(cluster ...)` is one-way only, so we
* implement CGM manually: V_2way = V_firm + V_year - V_intersection.
gen long intersection_id = firm * 100 + year   // unique per (firm, year) cell
quietly regress y_twocluster x, vce(cluster firm)
matrix V_firm = e(V)
local n = e(N)
local k = e(rank)
quietly regress y_twocluster x, vce(cluster year)
matrix V_year = e(V)
quietly regress y_twocluster x, vce(cluster intersection_id)
matrix V_inter = e(V)
matrix V_2way = V_firm + V_year - V_inter

* Re-fit baseline regression to expose coefficients without per-call vce overhead.
quietly regress y_twocluster x
local b_const = _b[_cons]
local b_x    = _b[x]
local se_const = sqrt(V_2way["_cons", "_cons"])
local se_x    = sqrt(V_2way["x", "x"])
local n_obs = e(N)

display ""
display "Stata reference T04 (OLS 2-way cluster, CGM):"
display "  const = " %12.6f `b_const' "  SE = " %12.6f `se_const'
display "  x     = " %12.6f `b_x' "  SE = " %12.6f `se_x'
display "  n_obs = " `n_obs'
display "JSON_RESULT: " `"{"platform":"stata","const":"' %12.6f `b_const' `","se_const":"' %12.6f `se_const' `","x":"' %12.6f `b_x' `","se_x":"' %12.6f `se_x' `","n_obs":"' `n_obs' `"}"'
