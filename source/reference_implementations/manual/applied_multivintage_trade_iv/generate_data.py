"""Generate the input files of applied_multivintage_trade_iv.

Copied from the data-generation part of function a1 in the task build program
(seed 2026091708, NumPy default_rng). The build program generated four tasks
from one random stream, and a1 ran first, so this program reproduces the stream.

Usage:
    python generate_data.py OUT_DIR
Writes areas.csv, employment.csv, trade.csv, crosswalk.csv, and outcomes.csv
into OUT_DIR.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SEED = 2026091708
rng = np.random.default_rng(SEED)


def save_csv(folder, name, rows):
    p = folder / (name + '.csv')
    pd.DataFrame(rows).to_csv(p, index=False, float_format='%.14g')
    return pd.read_csv(p)


p = OUT
area=[]; emp=[]; trade=[]
# Employment includes a nonmanufacturing residual through total_employment.
for period in (0,1):
    base_counts = {}
    for vintage in ('base','early'):
        for i in range(36):
            counts=rng.integers(20,300,4).astype(float)
            if vintage == 'early':
                counts = np.round(.75*base_counts[i] + .25*counts)
            counts[(i+period)%4] = 0 if i%5==0 else counts[(i+period)%4]
            if vintage == 'base':
                base_counts[i] = counts.copy()
            total=float(counts.sum()+rng.integers(400,1600))
            area.append(dict(area=i,period=period,vintage=vintage,total_employment=total,population=int(2000+80*i+900*period)))
            for j,v in enumerate(counts):
                if v: emp.append(dict(area=i,period=period,vintage=vintage,industry=j,employment=v))
    for j in range(4):
        shock=[110,420,250,690][j]*(1+.6*period)
        trade.append(dict(period=period,code=f'v{period}_{j}',us_change=shock*1e6,other_change=(shock*[1.2,.7,1.4,.9][j])*1e6,price_index=100+25*period,years=10 if period==0 else 7))
a=save_csv(p,'areas',area); e=save_csv(p,'employment',emp); t=save_csv(p,'trade',trade)
cross=save_csv(p,'crosswalk',[dict(code=f'v{q}_{j}',industry=j) for q in (0,1) for j in range(4)])


def exposure(kind, vintage, matrix=False, denominator='total'):
    ans=[]
    for q in (0,1):
        ee=e[(e.period==q)&(e.vintage==vintage)]
        aa=a[(a.period==q)&(a.vintage==vintage)].sort_values('area')
        tt=t[t.period==q].merge(cross,on='code').sort_values('industry')
        national=ee.groupby('industry').employment.sum().reindex(range(4)).to_numpy()
        shocks=tt[kind+'_change'].to_numpy()/(tt.price_index.to_numpy()/100)/1000*10/tt.years.to_numpy()/national
        mat=ee.pivot(index='area',columns='industry',values='employment').reindex(index=range(36),columns=range(4)).fillna(0).to_numpy()
        denom=aa.total_employment.to_numpy() if denominator=='total' else mat.sum(axis=1)
        if matrix: ans.extend((mat@shocks/denom).tolist())
        else:
            lookup=dict(zip(tt.industry,shocks))
            ans.extend([sum(row.employment*lookup[row.industry] for row in ee[ee.area==i].itertuples())/denom[i] for i in range(36)])
    return np.array(ans)


x=exposure('us','base'); z=exposure('other','early')
control=rng.normal(size=72); y=-1.4*x+.6*control+.4*np.repeat([0,1],36)+rng.normal(0,.1,72)
o=save_csv(p,'outcomes',{'area':np.tile(np.arange(36),2),'period':np.repeat([0,1],36),'control':control,'change_pp':y})
