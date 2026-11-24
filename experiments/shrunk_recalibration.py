"""
Shrunk conditional-mean recalibration.

Isotonic alone is noisy when the score carries little signal. Choose a convex
weight w between the linear calibration and the isotonic fit by K-fold CV inside
the calibration slice. A convex blend of two increasing maps is increasing, so
the ranking, the selected batch and the realized gain are all untouched.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/probe')
from probeH import tail_index, iso_fit, bin_fit
from scipy.stats import wilcoxon

RNG_N, KS, SEEDS, CALIB = 20000, (10,20,50), 20, 0.30
DELTAS = (0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0)
RHOS   = (0.5, 0.6, 0.7, 0.8, 0.9)
WGRID  = np.linspace(0, 1, 11)
NFOLD  = 5

def G(z, d): return np.sinh(np.arcsinh(z)/d)

def pick_w(x, y, rng):
    """K-fold CV inside the calibration slice for the blend weight."""
    n = len(x); fold = rng.permutation(n) % NFOLD
    err = np.zeros(len(WGRID))
    for k in range(NFOLD):
        tr, va = fold != k, fold == k
        if tr.sum() < 50 or va.sum() < 10: return 1.0
        A = np.polyfit(x[tr], y[tr], 1)
        lin_va = A[0]*x[va] + A[1]
        iso_va = iso_fit(x[tr], y[tr])(x[va])
        for j, w in enumerate(WGRID):
            err[j] += np.sum((y[va] - (w*iso_va + (1-w)*lin_va))**2)
    return float(WGRID[int(np.argmin(err))])

rows = []
for delta in DELTAS:
    for rho in RHOS:
        for seed in range(SEEDS):
            rg = np.random.default_rng(hash((delta, rho, seed)) % (2**32))
            z = rg.standard_normal(RNG_N); e = rg.standard_normal(RNG_N)
            u = rho*z + np.sqrt(1-rho**2)*e
            y = G(z, delta)
            idx = rg.permutation(RNG_N); ncal = int(CALIB*RNG_N)
            cal, te = idx[:ncal], idx[ncal:]

            A = np.polyfit(u[cal], y[cal], 1)
            raw_te = A[0]*u[te] + A[1]
            iso_te = iso_fit(u[cal], y[cal])(u[te])
            w = pick_w(u[cal], y[cal], rg)
            shr_te = w*iso_te + (1-w)*raw_te

            base = float(np.median(y[cal])); yte = y[te]
            order = np.argsort(-raw_te)
            # monotonicity check: blended ranking must equal raw ranking
            assert np.array_equal(np.argsort(-shr_te)[:50], order[:50]) or w in (0.0,), 'rank changed'
            for K in KS:
                top = order[:K]; best = np.argsort(-yte)[:K]
                ceil = float(yte[best].mean() - base)
                if ceil <= 1e-9: continue
                rows.append(dict(delta=delta, rho=rho, seed=seed, K=K, w=w,
                    tau=tail_index(y[cal]), ceiling=ceil,
                    realized=float(yte[top].mean()-base),
                    fc_raw=float(raw_te[top].mean()-base),
                    fc_iso=float(iso_te[top].mean()-base),
                    fc_shr=float(shr_te[top].mean()-base)))

df = pd.DataFrame(rows); df.to_csv('blend.csv', index=False)
for r in ('raw','iso','shr'):
    df['s_'+r] = (df['fc_'+r]-df.realized)/df.ceiling
print(f"rows={len(df)}  mean w={df.w.mean():.3f}")
print(f"\n{'rule':>5} {'median_s':>10} {'meanabs':>9} {'win_vs_raw%':>12} {'wilcoxon_p':>12}")
ar = df.s_raw.abs()
for r in ('raw','iso','shr'):
    a = df['s_'+r].abs()
    if r=='raw': print(f"{r:>5} {df['s_'+r].median():10.4f} {a.mean():9.4f} {'-':>12} {'-':>12}")
    else: print(f"{r:>5} {df['s_'+r].median():10.4f} {a.mean():9.4f} {100*(a<ar).mean():12.1f} {wilcoxon(a,ar).pvalue:12.3e}")
print(f"\nby accuracy rho:\n{'rho':>5} {'w':>6} {'raw':>8} {'iso':>8} {'shrunk':>8}")
for rh,g in df.groupby('rho'):
    print(f"{rh:5.1f} {g.w.mean():6.2f} {g.s_raw.abs().mean():8.4f} {g.s_iso.abs().mean():8.4f} {g.s_shr.abs().mean():8.4f}")
print(f"\nby tail index tau:\n{'tau':>7} {'raw':>8} {'iso':>8} {'shrunk':>8}")
for d,g in df.groupby('delta'):
    print(f"{g.tau.median():7.3f} {g.s_raw.abs().mean():8.4f} {g.s_iso.abs().mean():8.4f} {g.s_shr.abs().mean():8.4f}")
