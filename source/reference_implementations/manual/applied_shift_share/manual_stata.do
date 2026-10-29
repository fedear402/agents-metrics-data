* Reference program for applied_shift_share (Stata).
* This is the Stata native check of the task build program (applied-five
* pilot, shift_share). It collapses the regional residuals to the industry
* level and fits ivregress 2sls [aw=sn], vce(robust) small, which applies
* the N/(N-2) factor.
* Run from the data repository root: do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/applied_shift_share. The log is the output.
version 16.0
clear all
set more off
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/applied_shift_share"
import delimited "`input_dir'/input.csv", clear asdouble
generate double part = share*shock
bysort region_id: egen double ziv = total(part)
preserve
bysort region_id: keep if _n==1
summarize region_weight, meanonly
scalar total_weight = r(sum)
foreach v in y x ziv {
    summarize `v' [aw=region_weight], meanonly
    scalar mean_`v' = r(mean)
}
quietly ivregress 2sls y (x=ziv) [aw=region_weight], vce(robust) small
scalar region_effect = _b[x]
scalar region_effect_se = _se[x]
quietly regress x ziv [aw=region_weight], vce(robust)
scalar region_first = _b[ziv]
restore
generate double sn = region_weight/total_weight*share
generate double ysum = sn*(y-mean_y)
generate double xsum = sn*(x-mean_x)
generate double zsum = sn*(ziv-mean_ziv)
collapse (sum) sn ysum xsum zsum (firstnm) shock, by(industry_id)
generate double ybar = ysum/sn
generate double xbar = xsum/sn
generate double zbar = zsum/sn
quietly ivregress 2sls ybar (xbar=shock) [aw=sn], vce(robust) small
scalar b_effect = _b[xbar]
scalar se_effect = _se[xbar]
scalar n_shocks = e(N)
assert abs(b_effect-region_effect)<1e-10
quietly ivregress 2sls xbar (zbar=shock) [aw=sn], vce(robust) small
scalar b_first = _b[zbar]
scalar se_first = _se[zbar]
assert abs(b_first-region_first)<1e-10
generate double sn2 = sn^2
summarize sn2, meanonly
scalar effective_shocks = 1/r(sum)
display "ANSWER: effect estimate=" %21.15g b_effect " se=" %21.15g se_effect
display "ANSWER: first_stage estimate=" %21.15g b_first " se=" %21.15g se_first
display "ANSWER: effective_shocks estimate=" %21.15g effective_shocks " se=."
exit 0
