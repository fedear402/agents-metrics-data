* Reference program for applied_cumulative_hac (Stata).
* This is the Stata native check of the task build program (applied-five
* pilot, cumulative_hac). It uses newey with lag(15), which applies the
* N/(N-K) factor, and the full covariance matrix for the cumulative sum.
* Run from the data repository root: do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/applied_cumulative_hac. The log is the output.
version 16.0
clear all
set more off
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/applied_cumulative_hac"
import delimited "`input_dir'/input.csv", clear varnames(1) asdouble
sort date
gen long trading_day=_n
tsset trading_day
gen double lv=log(vix)
gen double ls=log(sales)
gen double change=D.lv
forvalues j=1/15 {
 gen double lag_`j'=L`j'.change
}
quietly newey ls lag_1-lag_15, lag(15)
scalar Nsample=e(N)
matrix beta=e(b)
matrix covariance=e(V)
mata:
b=st_matrix("beta")'; V=st_matrix("covariance")
C=J(3,16,0); C[1,1..15]=J(1,15,1); C[2,1]=1; C[3,15]=1
values=C*b; se=sqrt(diagonal(C*V*C'))
for(i=1;i<=3;i++) {
 st_numscalar("est"+strofreal(i),values[i])
 st_numscalar("se"+strofreal(i),se[i])
}
end
display "ANSWER: cumulative_response estimate=" %21.15g est1 " se=" %21.15g se1
display "ANSWER: lag_1 estimate=" %21.15g est2 " se=" %21.15g se2
display "ANSWER: lag_15 estimate=" %21.15g est3 " se=" %21.15g se3
exit 0
