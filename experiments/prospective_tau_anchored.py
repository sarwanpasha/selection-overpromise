"""
Same leave-one-shape-out test, but the tau rule is anchored at the theory's null:
Corollary 3.2 says the gap is exactly zero when fitness is Gaussian (tau = 1, so
log tau = 0). Dropping the intercept imposes that, rather than fitting it.
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
df = pd.read_csv('tailcv.csv'); df['s']=(df.fc_raw-df.realized)/df.ceiling
shapes = sorted(df['shape'].unique())

def free(d):     return np.column_stack([np.ones(len(d)), np.log(d.tau.values), d.rho.values])
def anchored(d): # no intercept: gap -> 0 as tau -> 1, scaled by accuracy
    lt = np.log(d.tau.values)
    return np.column_stack([lt, lt*d.rho.values])

rows=[]
for held in shapes:
    tr, te = df[df['shape']!=held], df[df['shape']==held]
    cf,*_ = np.linalg.lstsq(free(tr), tr.s.values, rcond=None)
    ca,*_ = np.linalg.lstsq(anchored(tr), tr.s.values, rcond=None)
    preds = {'tau_free': free(te)@cf, 'tau_anchored': anchored(te)@ca,
             'assume_zero': np.zeros(len(te)), 'pooled_mean': np.full(len(te), tr.s.mean())}
    for nm,p in preds.items():
        rows.append(dict(shape=held, rule=nm, mae=float(np.mean(np.abs(te.s.values-p)))))
r = pd.DataFrame(rows).pivot(index='shape', columns='rule', values='mae')
print("leave-one-shape-out MAE (lower is better)\n")
print(f"{'held-out shape':>14} {'anchored':>9} {'free':>9} {'assume 0':>9} {'pooled':>9}")
for s in r.index:
    print(f"{s:>14} {r.tau_anchored[s]:9.4f} {r.tau_free[s]:9.4f} {r.assume_zero[s]:9.4f} {r.pooled_mean[s]:9.4f}")
print(f"\n{'MEAN':>14} {r.tau_anchored.mean():9.4f} {r.tau_free.mean():9.4f} {r.assume_zero.mean():9.4f} {r.pooled_mean.mean():9.4f}")
for b in ('assume_zero','pooled_mean','tau_free'):
    w=int((r.tau_anchored<r[b]).sum())
    print(f"anchored beats {b:<12}: {w}/{len(r)}  p={wilcoxon(r.tau_anchored, r[b]).pvalue:.4f}  "
          f"MAE ratio {r[b].mean()/r.tau_anchored.mean():.2f}x")
