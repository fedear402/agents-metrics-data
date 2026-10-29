* Native Stata check for lp11_duration.
* Source: the native Stata target check of this task (native_stata_target.do, six-task expansion package of 2026-09-19).
* Route: expanded worker-month risk sets, clogit with one group per cohort-month,
* and the first event-set probability through binomial-count normalizers.
*
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/lp11_duration. The log is the output.
args input_dir
if "`input_dir'" == "" local input_dir "tasks/lp11_duration"
clear all
set more off
set type double
set linesize 255
import delimited using "`input_dir'/input.csv", clear varnames(1)
gen long row=_n
expand end_month
bysort row: gen month=_n
gen byte failure=event==1 & month==end_month
egen long riskset=group(cohort month)
bysort riskset: egen exits=total(failure)
keep if exits>0
clogit failure training, group(riskset) nolog iterate(200) tolerance(1e-12)
scalar b=_b[training]
display "ANSWER: training_coefficient estimate=" %21.15g (b) " se=."
display "ANSWER: conditional_odds_ratio estimate=" %21.15g (exp(b)) " se=."
display "ANSWER: log_likelihood estimate=" %21.15g (e(ll)) " se=."
keep if cohort==0 & month==1
summarize training, meanonly
scalar n1=r(sum)
scalar n0=r(N)-n1
summarize failure, meanonly
scalar d=r(sum)
gen double observed=training*failure
summarize observed, meanonly
scalar observed_sum=r(sum)
scalar denominator=0
forvalues k=0/`=scalar(d)' {
 if `k'<=n1 & d-`k'<=n0 {
 scalar denominator=denominator+exp(lnfactorial(n1)-lnfactorial(`k')-lnfactorial(n1-`k')+lnfactorial(n0)-lnfactorial(d-`k')-lnfactorial(n0-d+`k')+b*`k')
 }
}
scalar logp=b*observed_sum-ln(denominator)
display "ANSWER: first_event_set_probability estimate=" %21.15g (exp(logp)) " se=."
display "ANSWER: first_event_set_log_probability estimate=" %21.15g (logp) " se=."
exit, clear
