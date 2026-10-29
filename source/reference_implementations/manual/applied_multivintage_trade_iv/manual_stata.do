* Native Stata check for applied_multivintage_trade_iv.
* Source: the native Stata target check of this task (native_stata_target.do, six-task expansion package of 2026-09-19).
* Route: merges, collapsed exposure sums, and ivregress 2sls with analytic weights.
*
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/applied_multivintage_trade_iv. The log is the output.
args input_dir
if "`input_dir'" == "" local input_dir "tasks/applied_multivintage_trade_iv"
clear all
set more off
set type double
set linesize 255
tempfile cross trades area0 area1 exp0 exp1
import delimited using "`input_dir'/crosswalk.csv", clear varnames(1)
save `cross'
import delimited using "`input_dir'/trade.csv", clear varnames(1)
merge m:1 code using `cross', assert(match) nogen
save `trades'
import delimited using "`input_dir'/areas.csv", clear varnames(1)
keep if vintage=="base"
keep area period total_employment population
save `area0'
import delimited using "`input_dir'/employment.csv", clear varnames(1)
keep if vintage=="base"
bysort period industry: egen double national=total(employment)
merge m:1 area period using `area0', assert(match) nogen
merge m:1 period industry using `trades', assert(match) nogen
gen double x=employment/total_employment*us_change/national/(price_index/100)/1000*(10/years)
collapse (sum) x, by(area period)
save `exp0'
import delimited using "`input_dir'/areas.csv", clear varnames(1)
keep if vintage=="early"
keep area period total_employment population
save `area1'
import delimited using "`input_dir'/employment.csv", clear varnames(1)
keep if vintage=="early"
bysort period industry: egen double national=total(employment)
merge m:1 area period using `area1', assert(match) nogen
merge m:1 period industry using `trades', assert(match) nogen
gen double z=employment/total_employment*other_change/national/(price_index/100)/1000*(10/years)
collapse (sum) z, by(area period)
save `exp1'
import delimited using "`input_dir'/outcomes.csv", clear varnames(1)
merge 1:1 area period using `exp0', assert(match) nogen
merge 1:1 area period using `exp1', assert(match) nogen
merge 1:1 area period using `area0', assert(match) nogen
ivregress 2sls change_pp period control (x=z) [aw=population]
display "ANSWER: iv_slope estimate=" %21.15g (_b[x]) " se=."
summarize x if area==0 & period==0, meanonly
display "ANSWER: exposure_area0_period0 estimate=" %21.15g (r(mean)) " se=."
summarize x if area==0 & period==1, meanonly
display "ANSWER: exposure_area0_period1 estimate=" %21.15g (r(mean)) " se=."
summarize z if area==0 & period==0, meanonly
display "ANSWER: instrument_area0_period0 estimate=" %21.15g (r(mean)) " se=."
exit, clear
