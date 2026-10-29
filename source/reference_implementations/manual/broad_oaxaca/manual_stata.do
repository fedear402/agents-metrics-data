* Reference program for broad_oaxaca (Stata).
* This is the native Stata check of the task build. regress fits each group,
* and the decomposition uses group 0 coefficients as the reference.
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/broad_oaxaca.
* The log holds one ANSWER line for each term.
version 16.0
clear all
set more off
set type double
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/broad_oaxaca"
import delimited "`input_dir'/input.csv", clear asdouble
quietly regress y x z if group==0
matrix b0=(_b[_cons],_b[x],_b[z])
quietly regress y x z if group==1
matrix b1=(_b[_cons],_b[x],_b[z])
foreach g in 0 1 {
 foreach v in y x z {
  quietly summarize `v' if group==`g', meanonly
  scalar m`v'`g'=r(mean)
 }
}
scalar explained=(mx1-mx0)*b0[1,2]+(mz1-mz0)*b0[1,3]
scalar unexplained=(b1[1,1]-b0[1,1])+mx1*(b1[1,2]-b0[1,2])+mz1*(b1[1,3]-b0[1,3])
display "ANSWER: gap estimate=" %21.15g (my1-my0) " se=."
display "ANSWER: explained estimate=" %21.15g (explained) " se=."
display "ANSWER: unexplained estimate=" %21.15g (unexplained) " se=."
display "ANSWER: explained_x estimate=" %21.15g ((mx1-mx0)*b0[1,2]) " se=."
exit 0
