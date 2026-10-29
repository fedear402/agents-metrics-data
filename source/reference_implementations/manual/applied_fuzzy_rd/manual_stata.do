* Reference program for applied_fuzzy_rd (Stata).
* This is the Stata native check of the task build program (applied-five
* pilot, fuzzy_rd). It fits two regress [aw=weight] models, builds HC0
* influence functions of the two jumps in Mata, and uses the delta method
* with their joint covariance for the ratio.
* Run from the data repository root: do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/applied_fuzzy_rd. The log is the output.
version 16.0
clear all
set more off
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/applied_fuzzy_rd"
import delimited "`input_dir'/input.csv", clear varnames(1) asdouble
keep if abs(score)<1
gen byte right=score>=0
gen double interaction=right*score
gen double weight=1-abs(score)
quietly regress earnings right score interaction [aw=weight]
scalar reduced=_b[right]
predict double uy, residuals
quietly regress participated right score interaction [aw=weight]
scalar first=_b[right]
predict double ud, residuals
mata:
x=st_data(.,"score"); d=st_data(.,"right"); w=st_data(.,"weight")
X=(J(rows(x),1,1),d,x,d:*x)
A=invsym(quadcross(X,w,X))
h=(X*A[,2]):*w
U=st_data(.,("uy","ud"))
I=U:*(h*J(1,2,1)); C=quadcross(I,I)
rf=st_numscalar("reduced"); fs=st_numscalar("first")
g=(1/fs,-rf/fs^2)'
v=(rf/fs,rf,fs)'; se=(sqrt(g'*C*g)\sqrt(C[1,1])\sqrt(C[2,2]))
for(i=1;i<=3;i++) {
 st_numscalar("est"+strofreal(i),v[i])
 st_numscalar("se"+strofreal(i),se[i])
}
end
display "ANSWER: treatment_effect estimate=" %21.15g est1 " se=" %21.15g se1
display "ANSWER: reduced_form estimate=" %21.15g est2 " se=" %21.15g se2
display "ANSWER: first_stage estimate=" %21.15g est3 " se=" %21.15g se3
exit 0
