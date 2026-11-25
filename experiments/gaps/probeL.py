"""Probe L -- prospective tau on real assays, leave-one-assay-out.

The practical claim is that tau, computed on a NEW assay before any model is fit,
forecasts the gap better than trusting the surrogate. Simulation showed this on
held-out distribution families; this tests it on the 205 ProteinGym assays.

Rules compared, each fitted on all assays but one and evaluated on the held-out one:
  anchored : gap ~ b1*log(tau) + b2*log(tau)*rho, no intercept, so the prediction
             goes to zero at tau = 1 where Corollary 3.2 says the gap vanishes
  free     : gap ~ a + b1*log(tau) + b2*rho
  pooled   : the mean gap over the training assays
  zero     : assume no gap, which is what trusting the surrogate amounts to
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

src = sys.argv[1] if len(sys.argv) > 1 else 'probeH.csv'
d = pd.read_csv(src)
d = d[(d.ceiling > 1e-9) & np.isfinite(d.tau) & (d.tau > 0)]
g = d.groupby(['assay', 'frac', 'K'], as_index=False).median(numeric_only=True)
g['s'] = (g.fc_raw - g.realized) / g.ceiling
g = g[np.isfinite(g.s) & np.isfinite(g.spearman)]

def free(x):     return np.column_stack([np.ones(len(x)), np.log(x.tau.values), x.spearman.values])
def anchored(x):
    lt = np.log(x.tau.values)
    return np.column_stack([lt, lt * x.spearman.values])

assays = sorted(g.assay.unique())
rows = []
for a in assays:
    tr, te = g[g.assay != a], g[g.assay == a]
    if len(te) == 0 or len(tr) < 50:
        continue
    cf, *_ = np.linalg.lstsq(free(tr), tr.s.values, rcond=None)
    ca, *_ = np.linalg.lstsq(anchored(tr), tr.s.values, rcond=None)
    preds = {'anchored': anchored(te) @ ca, 'free': free(te) @ cf,
             'pooled': np.full(len(te), tr.s.mean()), 'zero': np.zeros(len(te))}
    for nm, p in preds.items():
        rows.append(dict(assay=a, rule=nm,
                         mae=float(np.mean(np.abs(te.s.values - p))),
                         n=len(te), tau=float(te.tau.median())))

r = pd.DataFrame(rows).pivot(index='assay', columns='rule', values='mae')
lines = ['ASSAYS = %d' % len(r), 'CELLS = %d' % len(g)]
for nm in ('anchored', 'free', 'pooled', 'zero'):
    lines.append('%s mean_mae = %.4f  median_mae = %.4f' % (nm, r[nm].mean(), r[nm].median()))
for base in ('zero', 'pooled', 'free'):
    w = int((r.anchored < r[base]).sum())
    p = wilcoxon(r.anchored, r[base]).pvalue
    lines.append('anchored beats %-6s on %d/%d assays, log10p = %.1f, mae ratio %.2f'
                 % (base, w, len(r), np.log10(max(p, 1e-300)), r[base].mean() / r.anchored.mean()))
# does the advantage concentrate where tau is far from 1?
tau = pd.DataFrame(rows).groupby('assay').tau.first()
far = (tau - 1.0).abs() > 0.25
for nm, m in (('far', far), ('near', ~far)):
    sub = r[m.values]
    if len(sub):
        lines.append('%s-from-gaussian assays n=%d anchored=%.4f zero=%.4f'
                     % (nm, len(sub), sub.anchored.mean(), sub.zero.mean()))
lines.append('PROBE_L_DONE')
open('probeL_report.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
