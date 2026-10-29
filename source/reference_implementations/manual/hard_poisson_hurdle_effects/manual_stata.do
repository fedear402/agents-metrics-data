* Reference program for hard_poisson_hurdle_effects (Stata).
* This is the native Stata check of the task package: logit for the hurdle,
* tpoisson with ll(0) for the positive counts, then the unconditional-mean terms.
* Run from the data repository root:
*   stata-se -b do manual_stata.do [INPUT_DIR]
* INPUT_DIR defaults to tasks/hard_poisson_hurdle_effects.
* The log holds one ANSWER line for each term.
version 19
clear all
set more off
set linesize 255
args input_dir
if "`input_dir'" == "" local input_dir "tasks/hard_poisson_hurdle_effects"
import delimited "`input_dir'/input.csv", clear
generate byte positive = y>0
logit positive x d, nolog
predict double eta, xb
scalar a_x = _b[x]
scalar a_d = _b[d]
tpoisson y x d if positive, ll(0) nolog
predict double loglambda, xb
scalar b_x = _b[x]
scalar b_d = _b[d]
generate double p = invlogit(eta)
generate double lambda = exp(loglambda)
generate double q = 1-exp(-lambda)
generate double mu = p*lambda/q
generate double deriv = p*(1-p)*a_x*lambda/q + p*b_x*lambda*(q-lambda*exp(-lambda))/(q*q)
generate double lambda1 = exp(loglambda+b_d*(1-d))
generate double lambda0 = exp(loglambda-b_d*d)
generate double delta = invlogit(eta+a_d*(1-d))*lambda1/(1-exp(-lambda1))-invlogit(eta-a_d*d)*lambda0/(1-exp(-lambda0))
generate double tail = p*poissontail(lambda,3)/q
quietly summarize mu
scalar mean_y = r(mean)
quietly summarize deriv
scalar ame_x = r(mean)
quietly summarize delta
scalar effect_d = r(mean)
quietly summarize tail
scalar prob_at_least_three = r(mean)
display "ANSWER: mean_y estimate=" %21.15g mean_y " se=."
display "ANSWER: ame_x estimate=" %21.15g ame_x " se=."
display "ANSWER: effect_d estimate=" %21.15g effect_d " se=."
display "ANSWER: prob_at_least_three estimate=" %21.15g prob_at_least_three " se=."
