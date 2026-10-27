* Install the Stata packages of the collection environment into PLUS.
*
* Run from the repository root with StataNow 19.5 SE in batch mode:
*   /Applications/StataNow/StataSE.app/Contents/MacOS/stata-se -b do code/benchmark/collection_environment/stata_packages.do
*
* New packages install without the replace option. If a package file already
* exists in PLUS, net install stops with r(602) and copies no file, so no
* installed package changes. The only replace is the drdid update: csdid 1.81
* stops with "Program DRDID is outdated" under drdid 1.71. drdid is not used by
* the analysis, and its package holds only drdid*.ado files.
*
* SSC serves only the current files. The pin is the Distribution-Date of each
* package (packages.json) and the file hashes in environment_manifest.json.
* The result of each install goes to stata_install_results.txt.

version 19
clear all
set more off

tempname out
file open `out' using "code/benchmark/collection_environment/stata_install_results.txt", write replace

foreach p in weakiv avar twostepweakiv ivreghdfe xtivreg2 oaxaca distinct moremata jwdid eventstudyinteract sdid ppml ssaggregate {
    capture noisily ssc install `p'
    file write `out' "new `p' rc=" (_rc) _n
}

foreach p in drdid {
    capture noisily ssc install `p', replace
    file write `out' "update `p' rc=" (_rc) _n
}

file close `out'
