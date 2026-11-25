"""Probe O -- does the closed-form model predict the gap on real assays?

For each assay the fitness transform G is taken as the empirical quantile function of
the measured values. For each cell of probeH.csv (assay, training fraction, budget) the
observed rank accuracy rho is mapped to the latent accuracy by Lemma 3.1,
rho_star = 2 sin(pi rho / 6), and the Gaussian-copula model is evaluated by quadrature:

  forecast = mu + rho_star * c1 * E[u | u > z_q]          (calibrated linear score)
  realized = E[ G(rho_star u + sqrt(1-rho_star^2) eps) | u > z_q ]
  ceiling  = E[ G(z) | z > z_q ]

all relative to G(0). Nothing is fitted to the observed gaps.
"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr, pearsonr, wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse

ROWCAP = 20000
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(60)
GH_W = GH_W / GH_W.sum()


def load_y(f):
    df = pd.read_csv(f)
    sc = 'DMS_score' if 'DMS_score' in df.columns else df.columns[-1]
    mc = 'mutant' if 'mutant' in df.columns else df.columns[0]
    if len(df) > ROWCAP:
        df = df.sample(ROWCAP, random_state=0)
    keep = []
    for i, mm in enumerate(df[mc].astype(str).values):
        if '/' in mm or ':' in mm or ',' in mm:
            continue
        if parse(mm) is not None:
            keep.append(i)
    y = pd.to_numeric(df[sc].values[keep], errors='coerce').astype(float)
    return np.sort(y[np.isfinite(y)])


def model_gap(ys, rho_star, q):
    n = len(ys)
    pp = (np.arange(n) + 0.5) / n
    G = lambda z: np.interp(norm.cdf(z), pp, ys)
    base = float(G(0.0))
    mu = float(ys.mean())
    zz = norm.ppf(pp)
    c1 = float(np.mean(ys * zz))
    zq = norm.ppf(1 - q)
    m = 400
    pu = (1 - q) + q * (np.arange(m) + 0.5) / m
    u = norm.ppf(pu)
    s = np.sqrt(max(1 - rho_star ** 2, 0.0))
    z = rho_star * u[:, None] + s * GH_X[None, :]
    realized = float((G(z) * GH_W[None, :]).sum(1).mean()) - base
    forecast = mu + rho_star * c1 * float(u.mean()) - base
    ceiling = float(G(u).mean()) - base
    return forecast, realized, ceiling


if __name__ == '__main__':
    d = pd.read_csv('probeH.csv')
    d = d[d.ceiling > 1e-9]
    g = d.groupby(['assay', 'frac', 'K'], as_index=False).median(numeric_only=True)
    g['s_obs'] = (g.fc_raw - g.realized) / g.ceiling
    g['cap_obs'] = g.realized / g.ceiling
    rows = []
    for a, x in g.groupby('assay'):
        f = os.path.join(sys.argv[1], a)
        if not os.path.exists(f):
            continue
        ys = load_y(f)
        if len(ys) < 200:
            continue
        for _, r in x.iterrows():
            if not np.isfinite(r.spearman):
                continue
            rs = float(np.clip(2 * np.sin(np.pi * r.spearman / 6), 0.01, 0.999))
            ntest = r.n - round(r.frac * r.n)
            q = float(np.clip(r.K / max(ntest, 1), 1e-4, 0.5))
            fc, re, ce = model_gap(ys, rs, q)
            if ce <= 1e-9:
                continue
            rows.append(dict(assay=a, frac=r.frac, K=r.K, tau=r.tau, rho=r.spearman,
                             s_obs=r.s_obs, s_th=(fc - re) / ce,
                             cap_obs=r.cap_obs, cap_th=re / ce))
    o = pd.DataFrame(rows)
    o.to_csv('probeO.csv', index=False)
    L = ['CELLS = %d' % len(o), 'ASSAYS = %d' % o.assay.nunique()]
    L.append('gap pearson = %.3f' % pearsonr(o.s_th, o.s_obs)[0])
    L.append('gap spearman = %.3f' % spearmanr(o.s_th, o.s_obs)[0])
    L.append('gap median obs = %.4f th = %.4f' % (o.s_obs.median(), o.s_th.median()))
    L.append('gap sign agreement = %.1f' % (100 * np.mean(np.sign(o.s_th) == np.sign(o.s_obs))))
    e_th, e_0 = (o.s_obs - o.s_th).abs(), o.s_obs.abs()
    L.append('gap mae theory = %.4f zero = %.4f' % (e_th.mean(), e_0.mean()))
    L.append('gap theory beats zero = %.1f log10p = %.1f'
             % (100 * (e_th < e_0).mean(), np.log10(max(wilcoxon(e_th, e_0).pvalue, 1e-300))))
    L.append('captured pearson = %.3f' % pearsonr(o.cap_th, o.cap_obs)[0])
    L.append('captured spearman = %.3f' % spearmanr(o.cap_th, o.cap_obs)[0])
    L.append('captured median obs = %.4f th = %.4f' % (o.cap_obs.median(), o.cap_th.median()))
    am = o.groupby('assay').median(numeric_only=True)
    L.append('assay-level gap spearman = %.3f n = %d' % (spearmanr(am.s_th, am.s_obs)[0], len(am)))
    for lo, hi in ((0, .6), (.6, .75), (.75, .9), (.9, 1.05), (1.05, 1.4), (1.4, 99)):
        x = o[(o.tau >= lo) & (o.tau < hi)]
        if len(x):
            L.append('tau %.2f-%.2f n=%d obs=%.3f th=%.3f' % (lo, hi, len(x), x.s_obs.median(), x.s_th.median()))
    L.append('PROBE_O_DONE')
    open('probeO_report.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
