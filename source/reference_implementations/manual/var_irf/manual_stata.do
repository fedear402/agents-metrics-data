* Manual reference for T21 (VAR(2) orthogonal IRF, y1 response to y2 shock, h=4) — Stata.
clear all
set more off

args input_dir
if `"`input_dir'"' == "" local input_dir "tasks/var_irf"
local data_path "`input_dir'/pilot_v4_var.csv"
import delimited "`data_path'", clear

tsset t
* Fit VAR(2) with constant. Equation order y1 then y2 (so a y2 shock has no
* contemporaneous effect on y1 under Cholesky).
var y1 y2, lags(1/2)
local n_obs = e(N)

* Build the IRF set; the equation order matches Cholesky order y1 -> y2.
tempfile irf_set
quietly irf create var_irf, set(`irf_set', replace) step(10) order(y1 y2)

* Open the IRF result dataset and pull out the OIRF value at step 4
* (impulse = y2, response = y1).
preserve
use `irf_set'.irf, clear
quietly summarize oirf if impulse == "y2" & response == "y1" & step == 4
local val = r(mean)
restore

display ""
display "Stata reference T21 (VAR(2) orth. IRF y1 ← y2, h=4):"
display "  irf_y1_to_y2_h4 = " %12.6f `val'
display "  lag_order      = 2"
display "  identification = Cholesky, y1 ordered first"
display "  n_obs          = " `n_obs'
display "JSON_RESULT: " `"{"platform":"stata","irf_y1_to_y2_h4":"' %12.6f `val' `","lag_order":2,"identification":"Cholesky, y1 ordered first","n_obs":"' `n_obs' `"}"'
