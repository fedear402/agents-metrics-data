* Manual reference for T20 (wild cluster bootstrap, 5 clusters, B=999) — Stata.
clear all
set more off
local data_path "data/benchmark/pilot_v5_wcb.csv"
capture confirm file "`data_path'"
if _rc != 0 local data_path "../../../data/benchmark/pilot_v5_wcb.csv"
import delimited "`data_path'", clear

* Manual wild cluster bootstrap (Rademacher weights, restricted residuals).
* This algorithm mirrors the Python and R manual references.
set seed 20260520
quietly regress y x, vce(cluster cluster)
local b = _b[x]
local se = _se[x]
local t_obs = `b' / `se'

quietly regress y, vce(cluster cluster)
predict double resid_r, residuals

* Discover the cluster ids actually present in the data.
quietly levelsof cluster, local(cluster_ids)
local extreme = 0
local B = 999
forvalues iter = 1/`B' {
    capture drop wstar ystar
    gen double wstar = .
    foreach c of local cluster_ids {
        local r = cond(runiformint(0, 1) == 1, 1, -1)
        quietly replace wstar = `r' if cluster == `c'
    }
    gen double ystar = resid_r * wstar
    quietly regress ystar x, vce(cluster cluster)
    local t_star = _b[x] / _se[x]
    if abs(`t_star') >= abs(`t_obs') local extreme = `extreme' + 1
}
local p = `extreme' / `B'
quietly tab cluster
local G = r(r)
local n_obs = _N
display "JSON_RESULT: " `"{"platform":"stata","x":"' %12.6f `b' `","se_x_cr1":"' %12.6f `se' `","t_obs":"' %12.6f `t_obs' `","p_wcb":"' %12.6f `p' `","B":999,"n_clusters":"' `G' `","n_obs":"' `n_obs' `"}"'
