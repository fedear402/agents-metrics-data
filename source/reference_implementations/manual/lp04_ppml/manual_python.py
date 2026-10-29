"""Reference program for lp04_ppml (Python).

This is the reference route of the task build program (function ppml):
one bounded linear program per zero outcome detects separated zeros, a
pivoted QR keeps a full-rank fixed-effect basis, and a damped Newton
iteration fits the Poisson pseudo maximum likelihood on the full dummy design.

The scored terms are agreement_coefficient and proportional_mean_change.
The pair-clustered standard error and the three observation counts of the
original task are printed under the extra key "diagnostics".

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/lp04_ppml. It must hold input.csv.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy import linalg, optimize

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/lp04_ppml")


def full_rank(A):
    _,r,p=linalg.qr(A,mode='economic',pivoting=True)
    rank=int(np.sum(abs(np.diag(r))>1e-9))
    return A[:,np.sort(p[:rank])]


def dummies(df,cols):
    return np.column_stack([pd.get_dummies(df[c],dtype=float).to_numpy() for c in cols])


def poisson_fit(x,y, method='newton'):
    b=np.zeros(x.shape[1]); b[0]=0
    def fun(b):
        z=x@b;mu=np.exp(z)
        return float(np.sum(mu-y*z))
    if method=='bfgs':
        fit=optimize.minimize(fun,b,jac=lambda b:x.T@(np.exp(x@b)-y),method='BFGS',options={'gtol':1e-8,'maxiter':1000})
        return fit.x
    for _ in range(200):
        mu=np.exp(x@b);score=x.T@(y-mu)
        delta=linalg.solve(x.T@(mu[:,None]*x),score,assume_a='pos')
        step=1
        while fun(b+step*delta)>fun(b)+1e-10:step/=2
        b+=step*delta
        if np.max(abs(score))<1e-9:break
    return b


df=pd.read_csv(INPUT / 'input.csv')
z=dummies(df,['pair','exporter_year','importer_year']);x=np.column_stack([df.agreement,z]);y=df.trade.to_numpy(float)
# Per-zero LP finds all observations with a negative separating index.
positive=y>0;zero=~positive; separated=np.zeros(len(df),bool)
for j in np.flatnonzero(zero):
    lp=optimize.linprog(x[j],A_ub=x[zero],b_ub=np.zeros(zero.sum()),A_eq=x[positive],b_eq=np.zeros(positive.sum()),bounds=[(-1,1)]*x.shape[1],method='highs')
    assert lp.success
    separated[j]=lp.fun < -1e-7
kept=df[~separated];yy=y[~separated];zz=full_rank(z[~separated]);xx=np.column_stack([kept.agreement,zz]);G=kept.pair.nunique();N,K=xx.shape
b=poisson_fit(xx,yy);mu=np.exp(xx@b)
bread=linalg.inv(xx.T@(mu[:,None]*xx));scores=xx*(yy-mu)[:,None]
agg=pd.DataFrame(scores).groupby(kept.pair.to_numpy()).sum().to_numpy()
cov=bread@agg.T@agg@bread*(G/(G-1))*((N-1)/(N-K))
ref={'agreement_coefficient':float(b[0]),'proportional_mean_change':float(np.expm1(b[0])),'pair_cluster_se':float(np.sqrt(cov[0,0])),'retained_observations':int(N),'separated_observations':int(separated.sum()),'retained_zero_observations':int(np.sum(yy==0))}

scored=['agreement_coefficient','proportional_mean_change']
print(json.dumps({
    "task_id": "lp04_ppml",
    "estimates": [{"term": k, "estimate": ref[k], "se": None, "n_obs": int(N)} for k in scored],
    "diagnostics": {k: v for k, v in ref.items() if k not in scored},
}))
