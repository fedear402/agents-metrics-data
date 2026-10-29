* Reference program for hard_conditional_logit_welfare (Stata).
* This is the Stata native reference of the task package (welfare inputs).
* It fits clogit with one chosen alternative per person and computes the
* logsum welfare change and the alternative-1 share change with fixed
* coefficients. Person weights enter only the two policy averages.
* Run from the data repository root: do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/hard_conditional_logit_welfare. The log is the output.
version 14
clear all
set more off
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/hard_conditional_logit_welfare"
import delimited "`input_dir'/input.csv", clear
sort person alternative
bys person: assert _N == 3
generate byte a1 = alternative == 1
generate byte a2 = alternative == 2
quietly clogit chosen cost quality a1 a2, group(person) tolerance(1e-10)
scalar bc = _b[cost]
predict double u0, xb
generate double u1 = u0 - .3*bc*cost*a1
bys person: egen double max0 = max(u0)
bys person: egen double max1 = max(u1)
generate double ex0 = exp(u0-max0)
generate double ex1 = exp(u1-max1)
bys person: egen double sum0 = total(ex0)
bys person: egen double sum1 = total(ex1)
generate double gain = (max1+ln(sum1)-max0-ln(sum0))/(-bc)
generate double change = ex1/sum1-ex0/sum0
quietly summarize gain [aw=weight] if alternative==1, meanonly
scalar wg = r(mean)
quietly summarize change [aw=weight] if alternative==1, meanonly
scalar sc = r(mean)
display "ANSWER: welfare_gain estimate=" %21.15g wg " se=."
display "ANSWER: share_change_alt1 estimate=" %21.15g sc " se=."
display "ANSWER: cost_coefficient estimate=" %21.15g bc " se=."
