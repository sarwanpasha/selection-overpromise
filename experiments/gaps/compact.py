"""Compact, integer-valued summaries of probeQ, probeR and probeS.

Values are medians over assays, scaled by 1000 and rounded, so that every figure is a
short integer. Win rates are percentages rounded to whole numbers. Reading long decimal
strings off a remote screen is error prone; this keeps every number to a few digits.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def assay_level(d, keys):
    cell = d.groupby(keys, as_index=False).median(numeric_only=True)
    return cell.groupby('assay', as_index=False).median(numeric_only=True)

def m1000(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return int(round(float(np.median(x)) * 1000)) if len(x) else -9999

def line(tag, vals):
    print(tag + ' ' + ' '.join(str(v) for v in vals), flush=True)

def q():
    if not os.path.exists('probeQ.csv'): return
    d = pd.read_csv('probeQ.csv'); d = d[d.ceiling > 1e-9]
    a = assay_level(d, ['assay', 'frac', 'K'])
    print('=== Q  assays %d ===' % len(a), flush=True)
    rules = ('raw', 'lin', 'iso', 'cviso', 'tailiso')
    s = {r: (a['fc_' + r] - a['realized']) / a['ceiling'] for r in rules}
    line('Qmed', [m1000(s[r]) for r in rules])
    line('Qabs', [m1000(s[r].abs()) for r in rules])
    for x, y in (('iso','raw'), ('lin','raw'), ('iso','lin'), ('cviso','iso'), ('tailiso','iso')):
        da, db = s[x].abs(), s[y].abs()
        try: p = wilcoxon(da, db).pvalue
        except Exception: p = float('nan')
        lp = int(round(np.log10(p))) if np.isfinite(p) and p > 0 else -999
        line('Qcmp_%s_%s' % (x, y), [int(round(100 * float((da < db).mean()))), lp])
    line('Qsigma_byfrac', [m1000(d[d.frac == f].sigma_n) for f in (0.1, 0.3, 0.6)])
    line('Qgap_byfrac', [m1000((d[d.frac == f].fc_raw - d[d.frac == f].realized) /
                               d[d.frac == f].ceiling) for f in (0.1, 0.3, 0.6)])
    line('Qbound_byfrac', [m1000(d[d.frac == f].fs_bound / d[d.frac == f].ceiling)
                           for f in (0.1, 0.3, 0.6)])
    cell = d.groupby(['assay', 'frac', 'K'], as_index=False).median(numeric_only=True)
    sr = (cell.fc_raw - cell.realized) / cell.ceiling
    from scipy.stats import spearmanr
    for col in ('tau', 'tau_train'):
        ok = np.isfinite(cell[col]) & np.isfinite(sr)
        r = spearmanr(cell[col][ok], sr[ok])
        line('Qtau_%s' % col, [int(round(1000 * r.statistic)),
                               int(round(np.log10(r.pvalue))) if r.pvalue > 0 else -999])
    line('Qlin_slope_pos_pct', [int(round(100 * float((cell.lin_slope > 0).mean())))])
    # mean of |s| over assays, to sit alongside the medians
    line('Qabsmean', [int(round(1000 * float(s[r].abs().mean()))) for r in rules])
    line('Qmedmean', [int(round(1000 * float(s[r].mean()))) for r in rules])
    # by training fraction, median and mean of the raw gap over cells
    for f in (0.1, 0.3, 0.6):
        c = cell[cell.frac == f]
        sr = (c.fc_raw - c.realized) / c.ceiling
        line('Qfrac_%s' % f, [int(round(1000 * float(sr.median()))),
                              int(round(1000 * float(sr.mean()))),
                              int(round(1000 * float(sr.abs().mean()))),
                              int(round(1000 * float(c.sigma_n.median())))])
    line('Qcells', [len(cell), int(cell.assay.nunique())])

def r():
    if not os.path.exists('probeR.csv'): return
    d = pd.read_csv('probeR.csv'); d = d[d.ceiling > 1e-9]
    rules = ('raw', 'lin', 'iso', 'cviso', 'tailiso')
    for mode, m in d.groupby('mode'):
        a = assay_level(m, ['assay', 'K'])
        s = {k: (a['fc_' + k] - a['realized']) / a['ceiling'] for k in rules}
        print('=== R %s assays %d rho %d ===' % (mode, len(a), m1000(a.spearman)), flush=True)
        line('Rmed_' + mode, [m1000(s[k]) for k in rules])
        line('Rabs_' + mode, [m1000(s[k].abs()) for k in rules])
        for x, y in (('iso','raw'), ('lin','raw'), ('iso','lin')):
            da, db = s[x].abs(), s[y].abs()
            try: p = wilcoxon(da, db).pvalue
            except Exception: p = float('nan')
            lp = int(round(np.log10(p))) if np.isfinite(p) and p > 0 else -999
            line('Rcmp_%s_%s_%s' % (mode, x, y),
                 [int(round(100 * float((da < db).mean()))), lp])

def s_():
    if not os.path.exists('probeS.csv'): return
    d = pd.read_csv('probeS.csv'); d = d[d.ceiling > 1e-9]
    for acq, m in d.groupby('acq'):
        a = assay_level(m, ['assay', 'frac', 'K'])
        sr = (a.fc_raw - a.realized) / a.ceiling
        si = (a.fc_iso - a.realized) / a.ceiling
        sl = (a.fc_lin - a.realized) / a.ceiling
        try: p = wilcoxon(si.abs(), sr.abs()).pvalue
        except Exception: p = float('nan')
        lp = int(round(np.log10(p))) if np.isfinite(p) and p > 0 else -999
        line('S_' + acq, [m1000(sr), m1000(si), m1000(sl),
                          m1000(a.realized / a.ceiling),
                          int(round(100 * float((si.abs() < sr.abs()).mean()))), lp])


def t():
    if not os.path.exists('probeT.csv'): return
    d = pd.read_csv('probeT.csv'); d = d[d.ceiling > 1e-9]
    rules = ('raw', 'lin', 'iso', 'auto', 'oracle')
    print('ORDER T rules: raw lin iso auto oracle', flush=True)
    for mode, m in d.groupby('mode'):
        cell = m.groupby(['assay', 'K'], as_index=False).median(numeric_only=True)
        a = cell.groupby('assay', as_index=False).median(numeric_only=True)
        s = {k: (a['fc_' + k] - a['realized']) / a['ceiling'] for k in rules}
        print('=== T %s assays %d rho %d drift %d ===' %
              (mode, len(a), m1000(a.spearman), m1000(a.drift)), flush=True)
        line('Tmed_' + mode, [m1000(s[k]) for k in rules])
        line('Tabs_' + mode, [m1000(s[k].abs()) for k in rules])
        for x, y in (('iso', 'raw'), ('auto', 'raw'), ('auto', 'iso'), ('auto', 'lin')):
            da, db = s[x].abs(), s[y].abs()
            try: p_ = wilcoxon(da, db).pvalue
            except Exception: p_ = float('nan')
            line('Tcmp_%s_%s_%s' % (mode, x, y),
                 [int(round(100 * float((da < db).mean()))),
                  int(round(np.log10(p_))) if np.isfinite(p_) and p_ > 0 else -999])
        pk = m.drop_duplicates(['assay', 'K', 'seed']).pick.value_counts(normalize=True)
        line('Tpick_' + mode, [int(round(100 * pk.get(k, 0.0))) for k in ('raw', 'lin', 'iso')])
    cell = d.groupby(['assay', 'mode', 'K'], as_index=False).median(numeric_only=True)
    gain = ((cell.fc_raw - cell.realized).abs() -
            (cell.fc_iso - cell.realized).abs()) / cell.ceiling
    ok = np.isfinite(gain) & np.isfinite(cell.drift)
    from scipy.stats import spearmanr as sp
    r = sp(cell.drift.abs()[ok], gain[ok])
    line('TDRIFT_vs_gain', [int(round(1000 * r.statistic)),
         int(round(np.log10(r.pvalue))) if r.pvalue > 0 else -999])

if __name__ == '__main__':
    print('ORDER rules: raw lin iso cviso tailiso', flush=True)
    print('S_<acq>: raw_med iso_med lin_med captured win% log10p', flush=True)
    for f in (q, r, s_, t):
        try: f()
        except Exception as e: print('FAIL', f.__name__, type(e).__name__, e, flush=True)
    print('COMPACT_DONE', flush=True)
