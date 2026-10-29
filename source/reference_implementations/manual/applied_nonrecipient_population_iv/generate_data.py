"""Generate the input files of applied_nonrecipient_population_iv.

Copied from the task build program of the four spatial tasks (seed
2026091708, NumPy default_rng). That program draws all four tasks from one
random stream in the order a1, b1, c2, c4. This task is c2. The program first
runs the data-generation code of a1 and b1 in a temporary folder to advance the
stream, then runs c2 and writes its files.

Usage:
    python generate_data.py OUT_DIR
Writes households.csv and census.csv into OUT_DIR.
"""
from pathlib import Path
import sys
import tempfile

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


def a1(p):
    """Data part of a1 (applied_multivintage_trade_iv); it only advances the stream here."""
    area=[]; emp=[]; trade=[]
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
    def exposure(kind, vintage):
        ans=[]
        for q in (0,1):
            ee=e[(e.period==q)&(e.vintage==vintage)]
            aa=a[(a.period==q)&(a.vintage==vintage)].sort_values('area')
            tt=t[t.period==q].merge(cross,on='code').sort_values('industry')
            national=ee.groupby('industry').employment.sum().reindex(range(4)).to_numpy()
            shocks=tt[kind+'_change'].to_numpy()/(tt.price_index.to_numpy()/100)/1000*10/tt.years.to_numpy()/national
            denom=aa.total_employment.to_numpy()
            lookup=dict(zip(tt.industry,shocks))
            ans.extend([sum(row.employment*lookup[row.industry] for row in ee[ee.area==i].itertuples())/denom[i] for i in range(36)])
        return np.array(ans)
    x=exposure('us','base')
    control=rng.normal(size=72); y=-1.4*x+.6*control+.4*np.repeat([0,1],36)+rng.normal(0,.1,72)
    save_csv(p,'outcomes',{'area':np.tile(np.arange(36),2),'period':np.repeat([0,1],36),'control':control,'change_pp':y})


def b1(p):
    """Data part of b1 (applied_spatial_exclusion_effects); it only advances the stream here."""
    villages=[]
    for i in range(65):
        villages.append(dict(village=i,mandal=i//3,x_km=float((i%9)*1.3),y_km=float((i//9)*1.2),study=int(i<60),buffer=int(i in (8,19,37)),treated=int((i//3)%3!=1)))
    villages[-1]['x_km']=100.; villages[-1]['y_km']=100.; villages[-1]['study']=1
    v=save_csv(p,'villages',villages)
    def exposure():
        eligible=((v.study==1)&(v.buffer==0)).to_numpy()
        ans=[]
        for i,r in v.iterrows():
            near=[j for j,s in v.iterrows() if j!=i and eligible[j] and s.mandal!=r.mandal and np.hypot(r.x_km-s.x_km,r.y_km-s.y_km)<=2.7]
            ans.append(float(v.iloc[near].treated.mean()) if near else 0.)
        return np.array(ans)
    s=exposure(); rows=[]
    for i,r in v.iterrows():
        if r.study==1 and r.buffer==0:
            for h in range(5):
                baseline=float(rng.normal()); weight=float(1+(i+2*h)%7)
                rows.append(dict(village=i,household=f'{i}_{h}',baseline=baseline,weight=weight,outcome=3+1.7*r.treated+4.3*s[i]+.6*baseline+rng.normal(0,.25)))
    save_csv(p,'households',rows)


def c2(p):
    rows=[]
    for i in range(480):
        eligible=int(i%4!=0); treated=int((i//4)%2==0); recipient=eligible*treated
        z=float(rng.uniform(.1,2)); u=float(rng.normal()); dose=.4+1.3*z+.2*u
        base=float(rng.normal()); w=float(1+i%5)
        y=2+.5*eligible+.7*base+(1.1+2.2*eligible)*dose+.8*u+rng.normal(0,.2)
        rows.append(dict(household=i,eligible=eligible,treated=treated,recipient=recipient,instrument=z,dose=dose,baseline=base,weight=w,outcome=y))
    save_csv(p,'households',rows)
    save_csv(p,'census',[dict(eligible=0,nonrecipient_population=8400),dict(eligible=1,nonrecipient_population=1600)])


with tempfile.TemporaryDirectory() as scratch:
    a1(Path(scratch))
    b1(Path(scratch))
c2(OUT)
