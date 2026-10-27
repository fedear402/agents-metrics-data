* Manual reference for T22 (Granger F-test, y2 does not Granger-cause y1, 2 lags) — Stata.
clear all
set more off

local data_path "data/benchmark/pilot_v4_var.csv"
capture confirm file "`data_path'"
if _rc != 0 local data_path "../../../data/benchmark/pilot_v4_var.csv"
import delimited "`data_path'", clear

tsset t
* Build the unrestricted and restricted OLS regressions explicitly to mirror
* the manual Python and R scripts (same data window: t = 3..N).
quietly regress y1 L.y1 L2.y1 L.y2 L2.y2
local rss_unr = e(rss)
local df_unr = e(df_r)
local k_unr = e(rank)
local n_used = e(N)
quietly regress y1 L.y1 L2.y1
local rss_res = e(rss)

local q = 2
local F = ((`rss_res' - `rss_unr') / `q') / (`rss_unr' / `df_unr')
local pv = Ftail(`q', `df_unr', `F')
display ""
display "Stata reference T22 (Granger 2-lag F):"
display "  F            = " %12.6f `F'
display "  df_num       = " `q'
display "  df_den       = " `df_unr'
display "  p_value      = " %12.6f `pv'
display "  n_obs        = " `n_used'
display "JSON_RESULT: " `"{"platform":"stata","F":"' %12.6f `F' `","df_num":"' `q' `","df_den":"' `df_unr' `","p_value":"' %12.6f `pv' `","n_obs":"' `n_used' `"}"'
