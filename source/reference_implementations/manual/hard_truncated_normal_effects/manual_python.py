"""Reference program for hard_truncated_normal_effects (Python).

This is the reference route of the task build program (function truncated).
It fits the left-truncated normal model by BFGS on the direct likelihood with
the analytic score. The independent route of the same build program (score
root, finite-difference derivative) is printed as an extra key.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/hard_truncated_normal_effects.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from scipy.special import log_ndtr
from scipy.stats import norm

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/hard_truncated_normal_effects")


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
d=pd.read_csv(INPUT / "input.csv", float_precision="round_trip")
X=np.column_stack([np.ones(len(d)),d.x,d.z]);Y=d.y.to_numpy()
def ll(v):
    sigma=np.exp(v[3]);mu=X@v[:3];a=-mu/sigma;t=(Y-mu)/sigma
    return -float(np.sum(norm.logpdf(t)-v[3]-log_ndtr(-a)))
def score(v):
    sigma=np.exp(v[3]);mu=X@v[:3];a=-mu/sigma;t=(Y-mu)/sigma;lam=np.exp(norm.logpdf(a)-log_ndtr(-a))
    return np.r_[X.T@((t-lam)/sigma),np.sum(t*t-1-a*lam)]
check_gradient(ll,score,[0,.5,0,0])
b=minimize_checked(ll,[0,.5,0,0],lambda v:-score(v));bb=root_checked(score,[-.2,.3,-.1,.1])
def calc(v,finite=False):
    mu=X@v[:3];s=np.exp(v[3]);a=-mu/s;lam=np.exp(norm.logpdf(a)-log_ndtr(-a));mean=mu+s*lam
    ame=v[1]*(1+a*lam-lam*lam)
    if finite:
        h=1e-5;mp=mu+h*v[1];mm=mu-h*v[1]
        def cond(m):return m+s*np.exp(norm.logpdf(-m/s)-log_ndtr(m/s))
        ame=(cond(mp)-cond(mm))/(2*h)
    return dict(mean_observed=float(mean.mean()),ame_x=float(ame.mean()),latent_x=float(v[1]))
main=calc(b);independent=calc(bb,True)

result={
    "task_id":"hard_truncated_normal_effects",
    "platform":"python",
    "estimates":[dict(term=k,estimate=v,se=None,n_obs=int(len(d))) for k,v in main.items()],
    "parameters":dict(beta0=float(b[0]),beta_x=float(b[1]),beta_z=float(b[2]),log_sigma=float(b[3])),
    "independent_route":independent,
}
print(json.dumps(result,indent=2))
