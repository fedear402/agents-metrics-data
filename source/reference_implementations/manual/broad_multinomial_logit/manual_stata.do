* Reference program for broad_multinomial_logit (Stata).
* This is the native Stata check of the task build: mlogit with base outcome 0.
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/broad_multinomial_logit.
* The log holds one ANSWER line for each term.
version 16.0
clear all
set more off
set type double
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/broad_multinomial_logit"
import delimited "`input_dir'/input.csv", clear asdouble
quietly mlogit y x z, baseoutcome(0)
display "ANSWER: x_cat1 estimate=" %21.15g (_b[1:x]) " se=" %21.15g (_se[1:x])
display "ANSWER: z_cat1 estimate=" %21.15g (_b[1:z]) " se=" %21.15g (_se[1:z])
display "ANSWER: x_cat2 estimate=" %21.15g (_b[2:x]) " se=" %21.15g (_se[2:x])
display "ANSWER: z_cat2 estimate=" %21.15g (_b[2:z]) " se=" %21.15g (_se[2:z])
display "ANSWER: prob_cat2_zero estimate=" %21.15g (exp(_b[2:_cons])/(1+exp(_b[1:_cons])+exp(_b[2:_cons]))) " se=."
exit 0
