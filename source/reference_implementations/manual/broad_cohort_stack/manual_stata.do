* Reference program for broad_cohort_stack (Stata).
* This is the native Stata check of the task build. For each cohort, regress
* fits y on did and unit and time indicators in the balanced comparison sample.
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/broad_cohort_stack.
* The log holds one ANSWER line for each term.
version 16.0
clear all
set more off
set type double
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/broad_cohort_stack"
import delimited "`input_dir'/input.csv", clear asdouble
scalar numerator=0
scalar denominator=0
scalar total=0
foreach cc in 3 5 7 {
    preserve
    keep if inrange(time,`cc'-2,`cc'+2) & (g==`cc' | g==0 | g>`cc'+2)
    bysort id: keep if _N==5
    generate byte did=g==`cc' & time>=`cc'
    quietly regress y did i.id i.time
    scalar effect=_b[did]
    quietly count if g==`cc'
    scalar wt=r(N)/5
    scalar numerator=numerator+wt*effect
    scalar denominator=denominator+wt
    scalar total=total+_N
    display "ANSWER: att_g`cc' estimate=" %21.15g (effect) " se=."
    restore
}
display "ANSWER: att_weighted estimate=" %21.15g (numerator/denominator) " se=."
exit 0
