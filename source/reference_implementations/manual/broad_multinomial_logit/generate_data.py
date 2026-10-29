"""Generate the input of broad_multinomial_logit.

This is the data step of the task build program (seed 2026091552).
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

EXPECTED_SHA256 = "b4f9e5e2d64a6e96e5704aa4f25715c64b4c8897b42f5772eccc3b0ff7a71a02"

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(2026091552);n=480;x=rng.normal(size=n);z=rng.normal(size=n)
eta=np.column_stack([np.zeros(n),.2+.65*x-.45*z,-.35-.5*x+.75*z]);p=np.exp(eta);p/=p.sum(1)[:,None]
y=(rng.random(n)[:,None]>np.cumsum(p,1)).sum(1)
d=pd.DataFrame(dict(id=np.arange(1,n+1),y=y,x=x,z=z))
d.to_csv(OUT/'input.csv',index=False,float_format='%.17g')

digest = hashlib.sha256((OUT / "input.csv").read_bytes()).hexdigest()
print("input.csv", digest, "matches the shipped file" if digest == EXPECTED_SHA256 else "DIFFERS from the shipped file")
