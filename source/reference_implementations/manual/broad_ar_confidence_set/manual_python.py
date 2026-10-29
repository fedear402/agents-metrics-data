"""Reference program for broad_ar_confidence_set (Python).

This is the Python reference program of the task build. It inverts the
homoskedastic Anderson-Rubin F test in closed form. The acceptance region is
{b : A b^2 + B b + C <= 0}. The coefficients come from residual cross-products
after the program partials out the intercept and x (restricted model) or the
intercept, x, z1, and z2 (unrestricted model).

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/broad_ar_confidence_set.
The program prints one JSON object and writes no files.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/broad_ar_confidence_set")
d=pd.read_csv(INPUT / "input.csv")
rows=[]
def add(term,estimate,se=None,n=None):
    rows.append(dict(term=term,estimate=float(estimate),se=None if se is None else float(se),n_obs=int(len(d) if n is None else n)))
from scipy.stats import f
X=sm.add_constant(d[["x","z1","z2"]]).to_numpy();R=X[:,:2]
res=lambda x,a:a-x@np.linalg.lstsq(x,a,rcond=None)[0]
a=res(R,d.y.to_numpy());c=res(R,d.endog.to_numpy())
u=res(X,d.y.to_numpy());v=res(X,d.endog.to_numpy())
k=2*f.ppf(.95,2,len(d)-4)/(len(d)-4)
q=np.array([c@c-(1+k)*(v@v),-2*(a@c-(1+k)*(u@v)),a@a-(1+k)*(u@u)])
roots=np.sort(np.roots(q));assert np.isreal(roots).all()
add("boundary_low",roots[0]);add("boundary_high",roots[1])
add("accepts_zero",q[2]<=0);add("unbounded",q[0]<0)
ssr=(a-c)@(a-c);ssu=(u-v)@(u-v)
add("ar_f_at_one",((ssr-ssu)/2)/(ssu/(len(d)-4)))
print(json.dumps({"estimates":rows},indent=2))
