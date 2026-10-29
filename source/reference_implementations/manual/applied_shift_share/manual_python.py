"""Reference program for applied_shift_share (Python).

This is the Python reference of the task build program (applied-five pilot,
shift_share). It aggregates the regional residuals to the industry level and
fits the Borusyak-Hull-Jaravel shock-level IV regressions with linearmodels
IV2SLS, analytic weights s_n, and robust covariance with the N/(N-2) factor
(debiased=True). The regional values of the build program are printed as
extra keys.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/applied_shift_share.
The program prints one JSON object and writes no files.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/applied_shift_share")

d = pd.read_csv(INPUT / "input.csv")
r = d.drop_duplicates("region_id").set_index("region_id").sort_index()
r["ziv"] = d.assign(part=d.share*d.shock).groupby("region_id").part.sum()
r["e"] = r.region_weight/r.region_weight.sum()
regional = IV2SLS(r.y, np.ones((len(r),1)), r[["x"]], r[["ziv"]], weights=r.e).fit(cov_type="robust",debiased=True)
regional_first = IV2SLS(r.x, pd.DataFrame({"const":1.,"ziv":r.ziv},index=r.index), None, None, weights=r.e).fit(cov_type="robust",debiased=True)
for v in ["y","x","ziv"]:
    r[v+"0"] = r[v]-np.average(r[v],weights=r.e)
a = d[["region_id","industry_id","share","shock"]].merge(r[["e","y0","x0","ziv0"]],left_on="region_id",right_index=True,validate="many_to_one")
a["sn"] = a.e*a.share
for v in ["y0","x0","ziv0"]:
    a[v] *= a.sn
s = a.groupby("industry_id").agg(sn=("sn","sum"),ybar=("y0","sum"),xbar=("x0","sum"),zbar=("ziv0","sum"),shock=("shock","first"))
for v in ["ybar","xbar","zbar"]:
    s[v] /= s.sn
effect = IV2SLS(s.ybar,np.ones((len(s),1)),s[["xbar"]],s[["shock"]],weights=s.sn).fit(cov_type="robust",debiased=True)
first = IV2SLS(s.xbar,np.ones((len(s),1)),s[["zbar"]],s[["shock"]],weights=s.sn).fit(cov_type="robust",debiased=True)
n = len(s)
rows = [dict(term="effect",estimate=float(effect.params["xbar"]),se=float(effect.std_errors["xbar"]),n_obs=n),
        dict(term="first_stage",estimate=float(first.params["zbar"]),se=float(first.std_errors["zbar"]),n_obs=n),
        dict(term="effective_shocks",estimate=float(1/np.sum(s.sn**2)),se=None,n_obs=n)]
result = dict(estimates=rows,regional_effect=float(regional.params["x"]),regional_first_stage=float(regional_first.params["ziv"]),
              regional_effect_hc1_se=float(regional.std_errors["x"]),regional_first_hc1_se=float(regional_first.std_errors["ziv"]),
              shock_weight_sum=float(s.sn.sum()),first_stage_exposure_F=float((first.params["zbar"]/first.std_errors["zbar"])**2))
print(json.dumps(result))
