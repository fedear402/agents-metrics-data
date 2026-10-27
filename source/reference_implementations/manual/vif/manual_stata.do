* Manual reference for T05 (VIF for x1..x5) — Stata.
clear all
set more off

local data_path "data/benchmark/pilot_v5_collinear.csv"
capture confirm file "`data_path'"
if _rc != 0 local data_path "../../../data/benchmark/pilot_v5_collinear.csv"
import delimited "`data_path'", clear

* Compute VIF_j = 1/(1-R^2_j) for each x_j regressed on the other four xs.
local cols x1 x2 x3 x4 x5
local n_obs = _N
foreach c of local cols {
    local others : list cols - c
    quietly regress `c' `others'
    local r2 = e(r2)
    local vif_`c' = 1 / (1 - `r2')
}
display "JSON_RESULT: " `"{"platform":"stata","vif_x1":"' %12.6f `vif_x1' `","vif_x2":"' %12.6f `vif_x2' `","vif_x3":"' %12.6f `vif_x3' `","vif_x4":"' %12.6f `vif_x4' `","vif_x5":"' %12.6f `vif_x5' `","n_obs":"' `n_obs' `"}"'
