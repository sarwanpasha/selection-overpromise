"""
Prospective validation of tau: leave-one-shape-out.

Correlation between tau and the gap is not enough. The practical claim is that a
team can compute tau on a NEW assay, before fitting any model, and forecast the
gap better than they could without it. Fit the tau rule on 11 distributions,
predict the held-out 12th, and compare against two baselines that ignore tau.
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv('tailcv.csv')
df['s'] = (df.fc_raw - df.realized) / df.ceiling   # gap the team would suffer
shapes = sorted(df['shape'].unique())

def design(d):
    # log tau is the natural scale: tau is a ratio of quantile spacings
    return np.column_stack([np.ones(len(d)), np.log(d.tau.values), d.rho.values])

rows = []
for held in shapes:
    tr = df[df['shape'] != held]
    te = df[df['shape'] == held]
    # tau rule: least squares on log tau and accuracy
    coef, *_ = np.linalg.lstsq(design(tr), tr.s.values, rcond=None)
    pred_tau  = design(te) @ coef
    pred_mean = np.full(len(te), tr.s.mean())            # baseline 1: pooled mean gap
    pred_zero = np.zeros(len(te))                        # baseline 2: assume no gap
    for nm, p in (('tau', pred_tau), ('pooled_mean', pred_mean), ('assume_zero', pred_zero)):
        rows.append(dict(shape=held, rule=nm,
                         mae=float(np.mean(np.abs(te.s.values - p))),
                         rmse=float(np.sqrt(np.mean((te.s.values - p)**2)))))

r = pd.DataFrame(rows).pivot(index='shape', columns='rule', values='mae')
r['tau_vs_zero'] = r.assume_zero / r.tau
r['tau_vs_mean'] = r.pooled_mean / r.tau
print("leave-one-shape-out MAE of the forecast gap (lower is better)\n")
print(f"{'held-out shape':>14} {'tau rule':>9} {'pooled':>9} {'assume 0':>9} {'x vs 0':>7} {'x vs mean':>10}")
for s in r.index:
    print(f"{s:>14} {r.tau[s]:9.4f} {r.pooled_mean[s]:9.4f} {r.assume_zero[s]:9.4f} "
          f"{r.tau_vs_zero[s]:7.2f} {r.tau_vs_mean[s]:10.2f}")
print(f"\n{'MEAN':>14} {r.tau.mean():9.4f} {r.pooled_mean.mean():9.4f} {r.assume_zero.mean():9.4f} "
      f"{r.assume_zero.mean()/r.tau.mean():7.2f} {r.pooled_mean.mean()/r.tau.mean():10.2f}")
print(f"\ntau beats 'assume zero' on {int((r.tau<r.assume_zero).sum())}/{len(r)} held-out shapes, "
      f"p={wilcoxon(r.tau, r.assume_zero).pvalue:.4f}")
print(f"tau beats pooled mean on   {int((r.tau<r.pooled_mean).sum())}/{len(r)} held-out shapes, "
      f"p={wilcoxon(r.tau, r.pooled_mean).pvalue:.4f}")
