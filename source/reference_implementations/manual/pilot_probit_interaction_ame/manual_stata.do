* Native Stata check for pilot_probit_interaction_ame.
* Source: the native Stata target check of this task (native_stata_target.do, six-task expansion package of 2026-09-19).
* Route: probit with factor-variable interaction and square, then margins
* over target==1 rows for dydx(x) and the discrete change in d.
*
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/pilot_probit_interaction_ame. The log is the output.
args input_dir
if "`input_dir'" == "" local input_dir "tasks/pilot_probit_interaction_ame"
clear all
set more off
set type double
set linesize 255
import delimited using "`input_dir'/input.csv", clear varnames(1)
probit y c.x##i.d z c.x#c.x, nolog difficult iterate(200)
margins if target==1, dydx(x)
matrix A=r(b)
display "ANSWER: ame_x estimate=" %21.15g (A[1,1]) " se=."
margins if target==1, dydx(d)
matrix A=r(b)
display "ANSWER: effect_d estimate=" %21.15g (A[1,2]) " se=."
exit, clear
