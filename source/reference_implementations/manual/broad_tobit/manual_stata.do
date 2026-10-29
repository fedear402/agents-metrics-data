* Reference program for broad_tobit (Stata).
* This is the native Stata check of the task build: tobit with ll(0).
* Stata 15 and later store the scale as the variance /var(e.y), so sigma is its square root.
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/broad_tobit.
* The log holds one ANSWER line for each term.
version 16.0
clear all
set more off
set type double
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/broad_tobit"
import delimited "`input_dir'/input.csv", clear asdouble
quietly tobit y x z, ll(0)
display "ANSWER: const estimate=" %21.15g (_b[_cons]) " se=" %21.15g (_se[_cons])
display "ANSWER: x estimate=" %21.15g (_b[x]) " se=" %21.15g (_se[x])
display "ANSWER: z estimate=" %21.15g (_b[z]) " se=" %21.15g (_se[z])
display "ANSWER: sigma estimate=" %21.15g (sqrt(_b[/var(e.y)])) " se=."
exit 0
