* Reference program for hard_truncated_normal_effects (Stata).
* This is the native Stata check of the task package: truncreg, then predict and margins.
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/hard_truncated_normal_effects.
* The log holds one ANSWER line for each term.
version 19
clear all
set more off
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/hard_truncated_normal_effects"
import delimited "`input_dir'/input.csv", clear
truncreg y x z, ll(0) nolog
scalar latent_x = _b[x]
predict double cm, e(0,.)
quietly summarize cm
scalar mean_observed = r(mean)
margins, dydx(x) predict(e(0,.))
matrix M = r(b)
scalar ame_x = M[1,1]
display "ANSWER: mean_observed estimate=" %21.15g mean_observed " se=."
display "ANSWER: ame_x estimate=" %21.15g ame_x " se=."
display "ANSWER: latent_x estimate=" %21.15g latent_x " se=."
