"""Reference program for hard_poisson_hurdle_effects (Python).

This is the reference route of the task build program (function hurdle).
It fits the hurdle by statsmodels Logit and the zero-truncated Poisson part by
BFGS on the direct likelihood with the analytic score. The independent route
of the same build program (score roots, finite-difference derivative, explicit
Poisson tail sum) is printed as an extra key.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/hard_poisson_hurdle_effects.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from scipy.special import expit, gammaln
from scipy.stats import poisson
import statsmodels.api as sm

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/hard_poisson_hurdle_effects")


def minimize_checked(fun, x, jac=None):
    fit=minimize(fun,x,jac=jac,method='BFGS',options={'gtol':1e-8,'maxiter':3000})
    assert np.isfinite(fit.fun)
    return fit.x


def root_checked(score, x):
    fit=least_squares(score,x,xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=4000)
    assert np.max(np.abs(score(fit.x)))<1e-5,fit.message
    return fit.x


def check_gradient(fun,score,point):
    point=np.asarray(point,dtype=float);numeric=[]
    for j in range(len(point)):
        delta=np.zeros(len(point));delta[j]=1e-5
        numeric.append((fun(point+delta)-fun(point-delta))/(2e-5))
    np.testing.assert_allclose(numeric,-score(point),rtol=2e-6,atol=2e-5)


# The build program used the simulated arrays. The CSV was written with
# %.17g, so a round-trip read restores the same values.
frame=pd.read_csv(INPUT / "input.csv", float_precision="round_trip")
y=frame.y.to_numpy();x=frame.x.to_numpy();d=frame.d.to_numpy();n=len(frame)
X=np.column_stack([np.ones(n),x,d]);positive=y>0
Xp=X[positive];yp=y[positive]
a=sm.Logit(positive.astype(int),X).fit(disp=False,tol=1e-12).params
def countll(b):
    lin=Xp@b;l=np.exp(lin)
    return -float(np.sum(yp*lin-l-gammaln(yp+1)-np.log(-np.expm1(-l))))
def countscore(b):
    l=np.exp(Xp@b);return Xp.T@(yp-l/(-np.expm1(-l)))
check_gradient(countll,countscore,[0,.2,.2])
b=minimize_checked(countll,[0,.2,.2],lambda v:-countscore(v));bb=root_checked(countscore,[.3,.5,.1]);aa=root_checked(lambda v:X.T@(positive-expit(X@v)),[0,0,0])
def calc(aa,bb,finite=False):
    def mean(xx):
        ll=np.exp(xx@bb);return expit(xx@aa)*ll/(-np.expm1(-ll))
    p=expit(X@aa);l=np.exp(X@bb);q=-np.expm1(-l);mu=l/q
    ame=p*(1-p)*aa[1]*mu+p*bb[1]*l*(q-l*np.exp(-l))/(q*q)
    if finite:
        plus=X.copy();minus=X.copy();plus[:,1]+=1e-5;minus[:,1]-=1e-5;ame=(mean(plus)-mean(minus))/(2e-5)
    one=X.copy();zero=X.copy();one[:,2]=1;zero[:,2]=0
    tail=(1-np.exp(-l)*(1+l+l*l/2)) if finite else poisson.sf(2,l)
    return dict(mean_y=float(mean(X).mean()),ame_x=float(ame.mean()),effect_d=float((mean(one)-mean(zero)).mean()),prob_at_least_three=float(np.mean(p*tail/q)))
main=calc(a,b);independent=calc(aa,bb,True)

result={
    "task_id":"hard_poisson_hurdle_effects",
    "platform":"python",
    "estimates":[dict(term=k,estimate=v,se=None,n_obs=int(n)) for k,v in main.items()],
    "parameters":dict(logit=[float(v) for v in a],count=[float(v) for v in b]),
    "independent_route":independent,
}
print(json.dumps(result,indent=2))
