"""Reference program for hard_rolling (Python).

This is the Python reference of the task build program (luna-hard-five-seven,
rolling). The build program computes the reference with the same loop. It
fits each rolling AR(1) by NumPy least squares on the 99 pairs inside the
100-observation window and forecasts one step ahead. The program reads the
rows in file order. The shipped input is already sorted by time.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/hard_rolling.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/hard_rolling")

d = np.genfromtxt(INPUT / "input.csv",delimiter=",",names=True)
y = d["y"]
forecasts=[]
for f in range(101,251):
    w=y[f-101:f-1]
    b=np.linalg.lstsq(np.column_stack([np.ones(99),w[:-1]]),w[1:],rcond=None)[0]
    forecasts.append(float(b[0]+b[1]*w[-1]))
values={"rmse":float(np.sqrt(np.mean((y[100:]-forecasts)**2)))}
values.update({"forecast_"+str(f):forecasts[f-101] for f in (101,150,200,250)})
estimates=[dict(term=k,estimate=v,se=None,n_obs=150 if k=="rmse" else 99) for k,v in values.items()]
print(json.dumps(dict(estimates=estimates)))
