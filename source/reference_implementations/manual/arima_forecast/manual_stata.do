* Manual reference for T02 (ARIMA(1,1,1) one-step forecast) — Stata.
clear all
set more off

local data_path "data/tasks/inputs/pilot_v4_arima.csv"
capture confirm file "`data_path'"
if _rc != 0 local data_path "../../../data/tasks/inputs/pilot_v4_arima.csv"
import delimited "`data_path'", clear

tsset t
* The task excludes drift in the differenced series.
arima y, arima(1,1,1) noconstant
local n_obs = e(N)
* One-step-ahead forecast at t = N+1 (out-of-sample).
predict yhat, y
* Append a synthetic next period and re-predict to get the out-of-sample mean.
quietly summarize t, meanonly
local last_t = r(max)
quietly count
local n = r(N)
* Use predict, dynamic at the post-sample horizon.
predict yfc, dynamic(`=`last_t' + 1') y
predict se_pred, mse
* Extract the t = N+1 forecast (created by `tsappend` if needed).
* For a fitted ARIMA(p,d,q), `predict y` returns in-sample fitted values; the
* one-step-ahead out-of-sample forecast equals the conditional mean using the
* last observations. We use `arima` postestimation `predict, y dynamic()`:
display ""
display "Stata reference T02 (ARIMA(1,1,1) one-step forecast):"
quietly tsappend, add(1)
predict yfc2, y dynamic(`=`last_t' + 1')
predict mse2, mse dynamic(`=`last_t' + 1')
quietly summarize yfc2 if t == `last_t' + 1, meanonly
local fc = r(mean)
quietly summarize mse2 if t == `last_t' + 1, meanonly
local mse = r(mean)
local se = sqrt(`mse')
local lo = `fc' - 1.959964 * `se'
local hi = `fc' + 1.959964 * `se'
display "  forecast     = " %12.6f `fc'
display "  se           = " %12.6f `se'
display "  lower_95     = " %12.6f `lo'
display "  upper_95     = " %12.6f `hi'
display "  n_obs        = " `n_obs'
display "JSON_RESULT: " `"{"platform":"stata","forecast":"' %12.6f `fc' `","se":"' %12.6f `se' `","lower_95":"' %12.6f `lo' `","upper_95":"' %12.6f `hi' `","n_obs":"' `n_obs' `"}"'
