* Native Stata check for applied_nonrecipient_population_iv.
* Source: the native Stata target check of this task (native_stata_target.do, six-task expansion package of 2026-09-19).
* Route: ivregress 2sls with analytic weights within each eligibility group,
* weighted mean dose, and the census-weighted mean of the two group effects.
*
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/applied_nonrecipient_population_iv. The log is the output.
args input_dir
if "`input_dir'" == "" local input_dir "tasks/applied_nonrecipient_population_iv"
clear all
set more off
set type double
set linesize 255
import delimited using "`input_dir'/households.csv", clear varnames(1)
keep if recipient==0
ivregress 2sls outcome baseline (dose=instrument) [aw=weight] if eligible==0
scalar b0=_b[dose]
summarize dose [aw=weight] if eligible==0, meanonly
scalar e0=b0*r(mean)
ivregress 2sls outcome baseline (dose=instrument) [aw=weight] if eligible==1
scalar b1=_b[dose]
summarize dose [aw=weight] if eligible==1, meanonly
scalar e1=b1*r(mean)
display "ANSWER: ineligible_effect estimate=" %21.15g (e0) " se=."
display "ANSWER: eligible_nonrecipient_effect estimate=" %21.15g (e1) " se=."
import delimited using "`input_dir'/census.csv", clear varnames(1)
summarize nonrecipient_population if eligible==0, meanonly
scalar n0=r(mean)
summarize nonrecipient_population if eligible==1, meanonly
scalar n1=r(mean)
display "ANSWER: population_effect estimate=" %21.15g ((e0*n0+e1*n1)/(n0+n1)) " se=."
exit, clear
