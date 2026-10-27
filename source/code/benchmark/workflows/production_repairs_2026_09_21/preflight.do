clear all
set more off
capture log close
log using preflight_stata.log, text replace
mata:
void ppml_covariance() {
    X=st_data(.,st_varname(4..(st_nvar()-1)))
    y=st_data(.,"trade"); mu=st_data(.,"mu"); pair=st_data(.,"pair")
    B=invsym(quadcross(X,mu,X))
    ids=uniqrows(sort(pair,1)); S=J(rows(ids),cols(X),0)
    for (i=1;i<=rows(ids);i++) S[i,.]=colsum(select(X:*(y-mu),pair:==ids[i]))
    N=rows(X); K=rank(X); G=rows(ids)
    V=B*quadcross(S,S)*B*G/(G-1)*(N-1)/(N-K)
    printf("PPML covariance: N=%g K=%g G=%g V11=%g missingX=%g missingmu=%g\n",N,K,G,V[1,1],hasmissing(X),hasmissing(mu))
    st_numscalar("full_rank_se",sqrt(V[1,1]))
}
end
foreach kind in target example {
    import delimited using lp_`kind'.csv, clear
    xtset unit period
    generate lag_y=L.y
    generate lead1=F.y
    if "`kind'"=="target" {
        generate lead2=F2.y
        keep if !missing(lag_y,y,lead1,lead2)
        local maxh=2
    }
    else {
        keep if !missing(lag_y,y,lead1)
        local maxh=1
    }
    file open result using stata_lp_`kind'.csv, write replace
    file write result "term,estimate,n_obs" _n
    forvalues h=0/`maxh' {
        local outcome=cond(`h'==0,"y","lead`h'")
        quietly ivregress 2sls `outcome' lag_y i.unit i.period (x=z)
        file write result "lp_h`h'," %21.14g (_b[x]) "," (e(N)) _n
    }
    file close result
    import delimited using ppml_`kind'.csv, clear
    local original=_N
    local effects "pair time"
    if "`kind'"=="target" local effects "pair exporter_year importer_year"
    ppmlhdfe trade agreement, absorb(`effects') vce(cluster pair) tolerance(1e-12)
    assert e(converged)==1
    local coef=_b[agreement]
    local N=e(N)
    quietly count if e(sample) & trade==0
    local zeros=r(N)
    * Compute the requested full-design-rank covariance from native fitted values.
    * The preflight design contains the same retained observations and a full-rank FE basis.
    import delimited using ppml_design_`kind'.csv, clear
    capture confirm string variable pair
    if _rc==0 {
        encode pair, generate(pair_id)
        drop pair
        rename pair_id pair
        order trade pair row_id
    }
    quietly glm trade v*, family(poisson) link(log) noconstant iterate(100) tolerance(1e-12)
    predict double mu, mu
    mata: ppml_covariance()
    file open result using stata_ppml_`kind'.csv, write replace
    file write result "term,estimate,n_obs" _n
    file write result "agreement_coefficient," %21.14g (`coef') ",`N'" _n
    file write result "proportional_mean_change," %21.14g (exp(`coef')-1) ",`N'" _n
    file write result "pair_cluster_se," %21.14g (full_rank_se) ",`N'" _n
    file write result "retained_observations,`N',`N'" _n
    file write result "separated_observations," (`original'-`N') ",`N'" _n
    file write result "retained_zero_observations,`zeros',`N'" _n
    file close result
}
display "STATA_PREFLIGHT_COMPLETE"
log close
exit, clear
