"""Reference program for hard_heteroskedastic_probit (Python).

This is the reference route of the task build program (function
heteroprobit). It fits the heteroskedastic probit by BFGS on the direct
normal-CDF likelihood with the analytic score. The independent route of the
same build program (score root, finite-difference derivative) is printed as
an extra key.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/hard_heteroskedastic_probit.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/hard_heteroskedastic_probit")


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
y=frame.y.to_numpy();x=frame.x.to_numpy();d=frame.d.to_numpy();z=frame.z.to_numpy();n=len(frame)
X=np.column_stack([np.ones(n),x,d]);sign=2*y-1;Z=z
def ll(v):return -float(log_ndtr(sign*(X@v[:3])*np.exp(-v[3]*Z)).sum())
def score(v):
    inv=np.exp(-v[3]*Z);index=X@v[:3];a=index*inv;t=sign*a;r=sign*np.exp(norm.logpdf(t)-log_ndtr(t))
    return np.r_[X.T@(r*inv),np.sum(r*(-a*Z))]
check_gradient(ll,score,[0,.4,-.3,.2])
b=minimize_checked(ll,[0,.4,-.3,.2],lambda v:-score(v));bb=root_checked(score,[.2,.8,-.2,.4])
def calc(v,finite=False):
    s=np.exp(v[3]*z);mu=X@v[:3];ame=norm.pdf(mu/s)*v[1]/s
    if finite:ame=(ndtr((mu+1e-5*v[1])/s)-ndtr((mu-1e-5*v[1])/s))/(2e-5)
    mu0=v[0]+v[1]*x;diff=ndtr((mu0+v[2])/s)-ndtr(mu0/s)
    return dict(gamma_z=float(v[3]),ame_x=float(ame.mean()),effect_d=float(diff.mean()))
main=calc(b);independent=calc(bb,True)

result={
    "task_id":"hard_heteroskedastic_probit",
    "platform":"python",
    "estimates":[dict(term=k,estimate=v,se=None,n_obs=int(n)) for k,v in main.items()],
    "parameters":dict(beta0=float(b[0]),beta_x=float(b[1]),beta_d=float(b[2]),gamma_z=float(b[3])),
    "independent_route":independent,
}
print(json.dumps(result,indent=2))
