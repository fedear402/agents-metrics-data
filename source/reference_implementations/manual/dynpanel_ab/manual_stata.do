* Manual reference for T16 (Arellano-Bond) — Stata via xtabond.
clear all
set more off
local data_path "data/benchmark/pilot_v5_dynpanel.csv"
capture confirm file "`data_path'"
if _rc != 0 local data_path "../../../data/benchmark/pilot_v5_dynpanel.csv"
import delimited "`data_path'", clear

xtset firm year
xtabond y x, lags(1) twostep noconstant
* xtabond reports coefficients on lagged y and x.
local rho = _b[L.y]
local se_rho = _se[L.y]
local beta_x = _b[x]
local se_beta_x = _se[x]
local n_obs = e(N)
quietly tab firm
local n_firms = r(r)
display "JSON_RESULT: " `"{"platform":"stata","rho":"' %12.6f `rho' `","se_rho":"' %12.6f `se_rho' `","beta_x":"' %12.6f `beta_x' `","se_beta_x":"' %12.6f `se_beta_x' `","n_obs":"' `n_obs' `","n_firms":"' `n_firms' `"}"'
