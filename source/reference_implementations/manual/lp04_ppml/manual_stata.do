* Native Stata check for lp04_ppml.
* Source: the native Stata target check of this task (native_stata_target.do, six-task expansion package of 2026-09-19).
* Route: iterative removal of zero-only fixed-effect groups (the separated zeros
* of this design), then poisson with pair, exporter-year, and importer-year
* factor variables.
*
* The scored terms are agreement_coefficient and proportional_mean_change.
* The other quantities of the original task print as DIAGNOSTIC lines.
*
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/lp04_ppml. The log is the output.
args input_dir
if "`input_dir'" == "" local input_dir "tasks/lp04_ppml"
clear all
set more off
set type double
set linesize 255
import delimited using "`input_dir'/input.csv", clear varnames(1)
foreach v in pair exporter_year importer_year {
 egen long num_`v'=group(`v')
 drop `v'
 rename num_`v' `v'
}
scalar original_N=_N
local changed=1
while `changed' {
 local before=_N
 foreach v in pair exporter_year importer_year {
  bysort `v': egen double total_y=total(trade)
  drop if total_y==0
  drop total_y
 }
 local changed=(_N<`before')
}
poisson trade agreement i.pair i.exporter_year i.importer_year, nolog iterate(200) tolerance(1e-12)
scalar K=e(rank)
poisson trade agreement i.pair i.exporter_year i.importer_year, vce(cluster pair) nolog iterate(200) tolerance(1e-12)
scalar N=e(N)
scalar b=_b[agreement]
scalar se=_se[agreement]*sqrt((N-1)/(N-K))
display "ANSWER: agreement_coefficient estimate=" %21.15g (b) " se=."
display "ANSWER: proportional_mean_change estimate=" %21.15g (exp(b)-1) " se=."
display "DIAGNOSTIC: pair_cluster_se " %21.15g (se)
display "DIAGNOSTIC: retained_observations " %21.15g (N)
display "DIAGNOSTIC: separated_observations " %21.15g (original_N-N)
count if trade==0
display "DIAGNOSTIC: retained_zero_observations " %21.15g (r(N))
exit, clear
