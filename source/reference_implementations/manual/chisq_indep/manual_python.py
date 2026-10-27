"""Manual reference for T09 (chi-square test of independence on a 2x3 table) — Python."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "pilot_v5_chisq.csv"

df = pd.read_csv(DATA)
ct = pd.crosstab(df["row"], df["col"]).values
chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
out = {"platform": "python", "chi2": float(chi2), "df": int(dof), "p_value": float(p), "n_obs": int(len(df))}
print(json.dumps(out, indent=2))
