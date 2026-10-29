"""Reference program for broad_tobit (Python).

This is the Python reference program of the task build. It fits a Gaussian
Tobit model with left censoring at zero by maximum likelihood
(statsmodels GenericLikelihoodModel, BFGS, log sigma parameterization).
Standard errors come from the numerical Hessian of the joint likelihood.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/broad_tobit.
The program prints one JSON object and writes no files.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/broad_tobit")
d=pd.read_csv(INPUT / "input.csv")
rows=[]
def add(term,estimate,se=None,n=None):
    rows.append(dict(term=term,estimate=float(estimate),se=None if se is None else float(se),n_obs=int(len(d) if n is None else n)))
from scipy.special import log_ndtr
from statsmodels.base.model import GenericLikelihoodModel
class Tobit(GenericLikelihoodModel):
    def nloglikeobs(self,params):
        sigma=np.exp(params[-1]); mu=self.exog@params[:-1]
        ll=np.where(self.endog>0,-np.log(sigma)-.5*np.log(2*np.pi)-.5*((self.endog-mu)/sigma)**2,log_ndtr(-mu/sigma))
        return -ll
X=sm.add_constant(d[["x","z"]]).to_numpy()
f=Tobit(d.y.to_numpy(),X,extra_params_names=["lnsigma"]).fit(start_params=[.2,.5,-.5,0.],method="bfgs",disp=False,gtol=1e-9,maxiter=1000)
for j,t in enumerate(["const","x","z"]):add(t,f.params[j],f.bse[j])
add("sigma",np.exp(f.params[-1]))
print(json.dumps({"estimates":rows},indent=2))
