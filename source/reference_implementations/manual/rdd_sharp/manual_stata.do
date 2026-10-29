* Manual reference for T18 (sharp RD, local-linear, triangular kernel, h=0.5) — Stata.
clear all
set more off
args input_dir
if `"`input_dir'"' == "" local input_dir "tasks/rdd_sharp"
local data_path "`input_dir'/pilot_v5_rdd.csv"
import delimited "`data_path'", clear

local h = 0.5
gen byte inband = abs(z) < `h'
gen double w = 1 - abs(z) / `h' if inband
quietly count if inband
local n_band = r(N)
local n_file_rows = _N
* n_obs is the effective estimation sample (observations inside the bandwidth),
* matching the benchmark result schema and what models report.
local n_obs = `n_band'

quietly regress y z [aweight=w] if inband & z >= 0
local mu_r = _b[_cons]
quietly regress y z [aweight=w] if inband & z < 0
local mu_l = _b[_cons]
local late = `mu_r' - `mu_l'
display "JSON_RESULT: " `"{"platform":"stata","late":"' %12.6f `late' `","bandwidth":"' %6.4f `h' `","n_inside_band":"' `n_band' `","n_obs":"' `n_obs' `"}"'
