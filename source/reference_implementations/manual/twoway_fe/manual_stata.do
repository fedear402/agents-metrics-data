* Manual reference for T14 (two-way FE OLS with cluster-robust SE on firm) — Stata.
clear all
set more off
args input_dir
if `"`input_dir'"' == "" local input_dir "tasks/twoway_fe"
local data_path "`input_dir'/pilot_v3.csv"
import delimited "`data_path'", clear

xtset firm year
* areg absorbs the firm FE and reports cluster SE on firm. To get two-way FE
* (firm + year), include i.year explicitly and absorb firm.
areg y_twocluster x i.year, absorb(firm) vce(cluster firm)
local b = _b[x]
local se = _se[x]
local n_obs = e(N)
quietly tab firm
local n_firms = r(r)
local K_absorbed = e(df_a) + e(df_m) + 1
display "JSON_RESULT: " `"{"platform":"stata","x":"' %12.6f `b' `","se_x_cluster_firm":"' %12.6f `se' `","n_obs":"' `n_obs' `","n_firms":"' `n_firms' `","K_absorbed":"' `K_absorbed' `"}"'
