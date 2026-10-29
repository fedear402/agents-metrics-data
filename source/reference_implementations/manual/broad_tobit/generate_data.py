"""Generate the input of broad_tobit.

This is the data step of the task build program (seed 2026091553).
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

EXPECTED_SHA256 = "efd9fa1b3d0062c3a4da223ad9c1673535368fa1347ec280572f5c6f65e0c01c"

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(2026091553);n=400;x=rng.normal(size=n);z=rng.normal(size=n);latent=.25+.8*x-.6*z+1.1*rng.normal(size=n)
d=pd.DataFrame(dict(id=np.arange(1,n+1),y=np.maximum(latent,0),x=x,z=z))
d.to_csv(OUT/'input.csv',index=False,float_format='%.17g')

digest = hashlib.sha256((OUT / "input.csv").read_bytes()).hexdigest()
print("input.csv", digest, "matches the shipped file" if digest == EXPECTED_SHA256 else "DIFFERS from the shipped file")
