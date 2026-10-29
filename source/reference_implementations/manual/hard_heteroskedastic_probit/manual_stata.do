* Reference program for hard_heteroskedastic_probit (Stata).
* This is the native Stata check of the task package: hetprobit, then margins.
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/hard_heteroskedastic_probit.
* The log holds one ANSWER line for each term.
version 19
clear all
set more off
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/hard_heteroskedastic_probit"
import delimited "`input_dir'/input.csv", clear
hetprobit y c.x i.d, het(z) nolog
scalar gamma_z = _b[lnsigma:z]
margins, dydx(x d) predict(pr)
matrix M = r(b)
scalar ame_x = M[1,1]
scalar effect_d = M[1,colnumb(M,"1.d")]
display "ANSWER: gamma_z estimate=" %21.15g gamma_z " se=."
display "ANSWER: ame_x estimate=" %21.15g ame_x " se=."
display "ANSWER: effect_d estimate=" %21.15g effect_d " se=."
