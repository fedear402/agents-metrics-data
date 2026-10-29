* Reference program for broad_ar_confidence_set (Stata).
* This is the native Stata check of the task build. regress and e(rss) at
* beta = -1, 0, 1 fix the quadratic acceptance region of the Anderson-Rubin F test.
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/broad_ar_confidence_set.
* The log holds one ANSWER line for each term.
version 16.0
clear all
set more off
set type double
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/broad_ar_confidence_set"
import delimited "`input_dir'/input.csv", clear asdouble
scalar NN=_N
scalar critical=invFtail(2,NN-4,.05)
forvalues j=0/2 {
    local beta=`j'-1
    generate double transformed=y-`beta'*endog
    quietly regress transformed x
    scalar restricted=e(rss)
    quietly regress transformed x z1 z2
    scalar unrestricted=e(rss)
    scalar q`j'=restricted-unrestricted-2*critical*unrestricted/(NN-4)
    if `j'==2 scalar fone=((restricted-unrestricted)/2)/(unrestricted/(NN-4))
    drop transformed
}
scalar aa=(q2+q0)/2-q1
scalar bb=(q2-q0)/2
scalar root1=(-bb-sqrt(bb^2-4*aa*q1))/(2*aa)
scalar root2=(-bb+sqrt(bb^2-4*aa*q1))/(2*aa)
display "ANSWER: boundary_low estimate=" %21.15g (min(root1,root2)) " se=."
display "ANSWER: boundary_high estimate=" %21.15g (max(root1,root2)) " se=."
display "ANSWER: accepts_zero estimate=" %21.15g (q1<=0) " se=."
display "ANSWER: unbounded estimate=" %21.15g (aa<0) " se=."
display "ANSWER: ar_f_at_one estimate=" %21.15g (fone) " se=."
exit 0
