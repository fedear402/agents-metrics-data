clear all
set more off
capture log close
log using native_check.log, text replace
which ppmlhdfe
which reghdfe
which ftools
ppmlhdfe, version
ftools, compile
sysuse auto, clear
ppmlhdfe price weight, absorb(foreign) vce(robust)
assert e(converged)==1
display "NATIVE_PACKAGES_COMPLETE"
log close
exit, clear
