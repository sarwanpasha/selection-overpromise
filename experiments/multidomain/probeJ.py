"""
Probe J -- the misspecification gap outside protein fitness.

Domain-agnostic. Takes any table with a continuous target and either numeric
feature columns or a sequence column, and runs the identical pipeline used on
ProteinGym: ridge surrogate, top-K selection by predicted value, gap between
predicted and realized gain, normalized by the achievable ceiling, plus the
tail-compression index tau and isotonic conditional-mean recalibration.

Usage: probeJ.py <spec.json>
  spec: {"name":..., "path":..., "target":..., "features":[...] | "sequence":col,
         "kmer":1}
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, wilcoxon

GAUSS_Q = 3.449049
FRACS   = (0.1, 0.3, 0.6)
KS      = (10, 20, 50)
SEEDS   = 5
CALIB   = 0.30
ROWCAP  = 20000


def tail_index(y):
    q50, q75, q99 = np.percentile(y, [50, 75, 99])
    if q75 - q50 < 1e-9:
        return float('nan')
    return float(((q99 - q50) / (q75 - q50)) / GAUSS_Q)


def iso_fit(x, y):
    m = IsotonicRegression(out_of_bounds='clip').fit(x, y)
    return lambda v: m.predict(np.asarray(v, dtype=float))


def ridge_fit(X, y, lam=None):
    Xm, ym = X.mean(0), float(y.mean())
    Xc = X - Xm
    n, d = Xc.shape
    if lam is None:
        lam = max(1e-6, 1e-2 * d)
    A = Xc.T @ Xc + lam * np.eye(d)
    w = np.linalg.solve(A, Xc.T @ (y - ym))
    return w, Xm, ym


def predict(X, w, Xm, ym):
    return (X - Xm) @ w + ym


def kmer_featurize(seqs, k=1):
    alpha = sorted({c for s in seqs for c in s})
    if k == 1:
        idx = {c: i for i, c in enumerate(alpha)}
        L = max(len(s) for s in seqs)
        X = np.zeros((len(seqs), L * len(alpha)), dtype=np.float32)
        for r, s in enumerate(seqs):
            for p, c in enumerate(s[:L]):
                X[r, p * len(alpha) + idx[c]] = 1.0
        return X
    grams = {}
    for s in seqs:
        for i in range(len(s) - k + 1):
            grams.setdefault(s[i:i + k], len(grams))
    X = np.zeros((len(seqs), len(grams)), dtype=np.float32)
    for r, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            X[r, grams[s[i:i + k]]] += 1.0
    return X


def load(spec):
    df = pd.read_csv(spec['path'])
    y = pd.to_numeric(df[spec['target']], errors='coerce').values.astype(float)
    if spec.get('sequence'):
        s = df[spec['sequence']].astype(str).values
        ok = np.isfinite(y) & np.array([len(x) > 0 for x in s])
        X = kmer_featurize(list(s[ok]), int(spec.get('kmer', 1)))
        y = y[ok]
    else:
        cols = spec.get('features')
        if not cols:
            cols = [c for c in df.columns
                    if c != spec['target'] and pd.api.types.is_numeric_dtype(df[c])]
        X = df[cols].apply(pd.to_numeric, errors='coerce').values.astype(float)
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        X, y = X[ok], y[ok]
    if len(y) > ROWCAP:
        r = np.random.RandomState(0).choice(len(y), ROWCAP, replace=False)
        X, y = X[r], y[r]
    sd = X.std(0); sd[sd < 1e-12] = 1.0
    X = (X - X.mean(0)) / sd
    return X, y


def run(spec):
    X, y = load(spec)
    if len(y) < 200 or y.std() < 1e-9:
        return []
    tau, n, out = tail_index(y), len(y), []
    for tf in FRACS:
        ntr = int(round(tf * n))
        if ntr < 100 or n - ntr < 100:
            continue
        for seed in range(SEEDS):
            rng = np.random.RandomState(1000 * seed + 7)
            idx = rng.permutation(n)
            tr, te = idx[:ntr], idx[ntr:]
            ncal = max(50, int(round(CALIB * ntr)))
            if ntr - ncal < 80:
                continue
            fit_i, cal_i = tr[:-ncal], tr[-ncal:]
            if y[fit_i].std() < 1e-9 or y[cal_i].std() < 1e-9:
                continue
            w, Xm, ym = ridge_fit(X[fit_i], y[fit_i])
            pc, pt = predict(X[cal_i], w, Xm, ym), predict(X[te], w, Xm, ym)
            if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
                continue
            try:    g = iso_fit(pc, y[cal_i])
            except Exception: g = lambda v: np.asarray(v, dtype=float)
            iso_te = np.asarray(g(pt), dtype=float)
            iso_te[~np.isfinite(iso_te)] = 0.0
            base, yte = float(np.median(y[fit_i])), y[te]
            rho = spearmanr(pt, yte).statistic
            for K in KS:
                if K > len(yte):
                    continue
                top, best = np.argsort(-pt)[:K], np.argsort(-yte)[:K]
                ceil = float(yte[best].mean() - base)
                if ceil <= 1e-9:
                    continue
                out.append(dict(domain=spec['name'], frac=tf, seed=seed, K=K, tau=tau,
                                n=n, ceiling=ceil,
                                realized=float(yte[top].mean() - base),
                                fc_raw=float(pt[top].mean() - base),
                                fc_iso=float(iso_te[top].mean() - base),
                                spearman=float(rho) if np.isfinite(rho) else None))
    return out


if __name__ == '__main__':
    specs = json.load(open(sys.argv[1]))
    rows = []
    for sp in specs:
        try:
            r = run(sp)
            print('%-22s %5d rows' % (sp['name'], len(r)), flush=True)
            rows.extend(r)
        except Exception as e:
            print('FAILED %-18s %s %s' % (sp['name'], type(e).__name__, e), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv('probeJ.csv', index=False)
    g = d.groupby(['domain', 'frac', 'K'], as_index=False).median(numeric_only=True)
    f = lambda r: (g['fc_' + r] - g.realized) / g.ceiling
    lines = ['CELLS = %d' % len(g)]
    for dom, x in g.groupby('domain'):
        sr = (x.fc_raw - x.realized) / x.ceiling
        si = (x.fc_iso - x.realized) / x.ceiling
        lines.append('%s n=%d tau=%.3f raw_med=%.4f raw_mab=%.4f iso_med=%.4f '
                     'iso_mab=%.4f rho=%.3f' % (dom, len(x), x.tau.median(),
                     sr.median(), sr.abs().mean(), si.median(), si.abs().mean(),
                     x.spearman.median()))
    ar, ai = f('raw').abs(), f('iso').abs()
    lines.append('ALL raw_mab=%.4f iso_mab=%.4f win=%.1f log10p=%.1f'
                 % (ar.mean(), ai.mean(), 100 * (ai < ar).mean(),
                    np.log10(max(wilcoxon(ai, ar).pvalue, 1e-300))))
    lines.append('PROBE_J_DONE')
    open('probeJ_report.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
