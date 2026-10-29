* Reference program for broad_ordered_logit (Stata).
* This is the native Stata check of the task build: ologit.
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/broad_ordered_logit.
* The log holds one ANSWER line for each term.
version 16.0
clear all
set more off
set type double
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/broad_ordered_logit"
import delimited "`input_dir'/input.csv", clear asdouble
quietly ologit y x z
display "ANSWER: x estimate=" %21.15g (_b[x]) " se=" %21.15g (_se[x])
display "ANSWER: z estimate=" %21.15g (_b[z]) " se=" %21.15g (_se[z])
display "ANSWER: prob_high estimate=" %21.15g (1-invlogit(_b[/cut2]-_b[x])) " se=."
exit 0
