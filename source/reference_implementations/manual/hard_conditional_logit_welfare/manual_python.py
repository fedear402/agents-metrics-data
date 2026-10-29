"""Reference program for hard_conditional_logit_welfare (Python).

This is the reference route of the task build program (twenty-harder-tasks,
nonlinear group, function choice_welfare). It fits the grouped
conditional-logit likelihood by BFGS with the analytic score and computes the
logsum welfare change and the alternative-1 share change with fixed
coefficients. The build program used its simulated arrays. This program
rebuilds the same arrays from input.csv. The independent route of the same
build program (score root, scalar logsum loop) is printed as an extra key.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/hard_conditional_logit_welfare.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from scipy.special import logsumexp

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/hard_conditional_logit_welfare")


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


# Input step: rebuild the person-by-alternative arrays of the build program.
d=pd.read_csv(INPUT / "input.csv",float_precision="round_trip").sort_values(["person","alternative"])
n=d.person.nunique();J=3
assert len(d)==n*J and (d.groupby("person").chosen.sum()==1).all()
cost=d.cost.to_numpy().reshape(n,J);quality=d.quality.to_numpy().reshape(n,J);asc=np.tile(np.eye(J)[None,:,:],(n,1,1))[:,:,1:]
X=np.concatenate([cost[:,:,None],quality[:,:,None],asc],axis=2)
choice=d.chosen.to_numpy().reshape(n,J).argmax(axis=1);w=d.weight.to_numpy().reshape(n,J)[:,0]

# Estimation and policy calculation, copied from the build program.
def ll(b):
    v=X@b;return -float(np.sum(v[np.arange(n),choice]-logsumexp(v,axis=1)))
def score(b):
    v=X@b;p=np.exp(v-logsumexp(v,axis=1)[:,None]);return np.sum(X[np.arange(n),choice]-np.sum(p[:,:,None]*X,axis=1),axis=0)
check_gradient(ll,score,[-.5,.2,0,0])
b=minimize_checked(ll,[-.5,.2,0,0],lambda v:-score(v));bb=root_checked(score,[-1,.4,.1,-.1])
policy=X.copy();policy[:,1,0]*=.7
def calc(v,scalar=False):
    old=X@v;new=policy@v;gain=(logsumexp(new,axis=1)-logsumexp(old,axis=1))/(-v[0]);pn=np.exp(new-logsumexp(new,axis=1)[:,None]);po=np.exp(old-logsumexp(old,axis=1)[:,None]);delta=pn[:,1]-po[:,1]
    if scalar:
        gain=np.array([(np.log(sum(np.exp(row@v) for row in policy[i]))-np.log(sum(np.exp(row@v) for row in X[i])))/(-v[0]) for i in range(n)])
    return dict(welfare_gain=float(np.average(gain,weights=w)),share_change_alt1=float(np.average(delta,weights=w)),cost_coefficient=float(v[0]))
main=calc(b);independent=calc(bb,True)

estimates=[dict(term=k,estimate=v,se=None,n_obs=int(n)) for k,v in main.items()]
print(json.dumps(dict(estimates=estimates,independent=independent,parameters=dict(main=b.tolist(),independent=bb.tolist()))))
