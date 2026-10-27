* Manual reference for T19 (local-projection IRF at h=4, NW SE) — Stata.
clear all
set more off
local data_path "data/benchmark/pilot_v5_lp.csv"
capture confirm file "`data_path'"
if _rc != 0 local data_path "../../../data/benchmark/pilot_v5_lp.csv"
import delimited "`data_path'", clear

tsset t
local h = 4
gen double y_lead = F`h'.y
gen double y_lag = L.y
newey y_lead shock y_lag, lag(`h')
local b = _b[shock]
local se = _se[shock]
local n_used = e(N)
display "JSON_RESULT: " `"{"platform":"stata","irf_h4":"' %12.6f `b' `","se_irf_h4":"' %12.6f `se' `","horizon":"' `h' `","n_used":"' `n_used' `"}"'
