"""Manual reference for T07 (Welch two-sample t-test) — Python."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "pilot_v5_ttest.csv"

df = pd.read_csv(DATA)
a = df.loc[df["group"] == 0, "y"].values
b = df.loc[df["group"] == 1, "y"].values
# Manual Welch t with Welch-Satterthwaite df
ma, mb = a.mean(), b.mean()
va, vb = a.var(ddof=1), b.var(ddof=1)
na, nb = len(a), len(b)
se = float(np.sqrt(va / na + vb / nb))
t = float((mb - ma) / se)
df_welch = float((va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)))
p = float(2.0 * (1.0 - stats.t.cdf(abs(t), df_welch)))
out = {"platform": "python", "t": t, "df": df_welch, "p_value": p, "n_a": int(na), "n_b": int(nb)}
print(json.dumps(out, indent=2))
