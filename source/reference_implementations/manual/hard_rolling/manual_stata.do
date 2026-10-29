* Reference program for hard_rolling (Stata).
* This is the Stata native check of the task build program
* (luna-hard-five-seven, rolling). It sorts by time and fits each rolling
* AR(1) with Mata qrsolve on the 99 pairs inside the 100-observation window.
* Run from the data repository root: do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/hard_rolling. The log is the output.
version 16.0
clear all
set more off
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/hard_rolling"
import delimited "`input_dir'/input.csv", clear varnames(1) asdouble
sort time
mata:
y=st_data(.,"y")
pred=J(150,1,.)
for(f=101;f<=250;f++) {
    w=y[(f-100)..(f-1)]
    X=J(99,1,1),w[1..99]
    b=qrsolve(X,w[2..100])
    pred[f-100]=b[1]+b[2]*w[100]
}
rmse=sqrt(mean((y[101..250]-pred):^2))
st_numscalar("rmse",rmse)
dates=(101,150,200,250)
for(k=1;k<=4;k++) st_numscalar("forecast_"+strofreal(dates[k]),pred[dates[k]-100])
end
display "ANSWER: rmse estimate=" %21.15g rmse " se=."
foreach f in 101 150 200 250 {
    display "ANSWER: forecast_`f' estimate=" %21.15g forecast_`f' " se=."
}
exit 0
