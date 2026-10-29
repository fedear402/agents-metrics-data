"""Generate the input of broad_ar_confidence_set.

This is the data step of the task build program (seed 2026091561).
Usage:
    python generate_data.py OUT_DIR
The program writes OUT_DIR/input.csv with the shipped file name.
It prints the SHA-256 of the file and compares it with the shipped file.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SHA256 = "d5ce48ec4edf86df37802a2b107c77a4ee6f8c4f5003e0fae9450fa1093d7076"

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(2026091561);n=240
x=rng.normal(size=n);z=rng.normal(size=(n,2));v=rng.normal(size=n)
e=.75*v+np.sqrt(1-.75**2)*rng.normal(size=n)
end=.08*z[:,0]+.06*z[:,1]+.5*x+v;y=end+.5*x+e
data=pd.DataFrame(dict(y=y,endog=end,x=x,z1=z[:,0],z2=z[:,1]))
data.to_csv(OUT/'input.csv',index=False,float_format='%.17g')

digest = hashlib.sha256((OUT / "input.csv").read_bytes()).hexdigest()
print("input.csv", digest, "matches the shipped file" if digest == EXPECTED_SHA256 else "DIFFERS from the shipped file")
