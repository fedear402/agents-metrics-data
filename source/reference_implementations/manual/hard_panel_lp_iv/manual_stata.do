* Native Stata check for hard_panel_lp_iv.
* Source: the native Stata target check of this task (native_stata_target.do, six-task expansion package of 2026-09-19).
* Route: xtset calendar lags and leads, common complete sample, and
* ivregress 2sls with unit and period factor variables at each horizon.
*
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/hard_panel_lp_iv. The log is the output.
args input_dir
if "`input_dir'" == "" local input_dir "tasks/hard_panel_lp_iv"
clear all
set more off
set type double
set linesize 255
import delimited using "`input_dir'/input.csv", clear varnames(1)
xtset unit period
gen double lag_y=L.y
gen double lead1=F.y
gen double lead2=F2.y
keep if !missing(lag_y,y,lead1,lead2)
ivregress 2sls y lag_y i.unit i.period (x=z)
display "ANSWER: lp_h0 estimate=" %21.15g (_b[x]) " se=."
ivregress 2sls lead1 lag_y i.unit i.period (x=z)
display "ANSWER: lp_h1 estimate=" %21.15g (_b[x]) " se=."
ivregress 2sls lead2 lag_y i.unit i.period (x=z)
display "ANSWER: lp_h2 estimate=" %21.15g (_b[x]) " se=."
exit, clear
