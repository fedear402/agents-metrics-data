"""Data generator for hard_rolling.

This is the data step of the task build program (luna-hard-five-seven,
rolling, function build). It simulates 250 observations of an AR(1) with a
break at time 141, with seed 2026091412.

Usage: python generate_data.py OUT_DIR
The program writes OUT_DIR/input.csv.
"""
from pathlib import Path
import sys

import numpy as np

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
out_dir = Path(sys.argv[1]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2026091412)
y = np.zeros(250)
y[0] = .3
innovations = rng.normal(size=250)
for j in range(1, 250):
    intercept = .4 if j < 140 else -.25
    persistence = .55 if j < 140 else .8
    y[j] = intercept + persistence * y[j - 1] + innovations[j]
path = out_dir / 'input.csv'
np.savetxt(path, np.column_stack([np.arange(1, 251), y]), delimiter=',',
           header='time,y', comments='', fmt='%.17g')
