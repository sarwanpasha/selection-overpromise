"""
Controlled validation of conditional-mean recalibration across tail shapes.
Mirrors probeH definitions: top-K selection by score, gains vs median of training,
forecast rules fitted on a held-out calibration slice only.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/probe')
from probeH import tail_index, iso_fit, bin_fit, tweedie_fit
from scipy.stats import wilcoxon

RNG_N   = 20000
KS      = (10, 20, 50)
SEEDS   = 20
CALIB   = 0.30
# Jones-Pewsey sinh-arcsinh: delta<1 heavier tail, delta>1 lighter (compressed) tail
DELTAS  = (0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0)
RHOS    = (0.5, 0.6, 0.7, 0.8, 0.9)

def G(z, delta):
    return np.sinh(np.arcsinh(z) / delta)

rows = []
for delta in DELTAS:
    for rho in RHOS:
        for seed in range(SEEDS):
            rg = np.random.default_rng(hash((delta, rho, seed)) % (2**32))
            z  = rg.standard_normal(RNG_N)
            e  = rg.standard_normal(RNG_N)
            u  = rho * z + np.sqrt(1 - rho**2) * e
            y  = G(z, delta)

            # split: calibration slice (held out from the "training" used to set the linear map)
            idx = rg.permutation(RNG_N)
            ncal = int(CALIB * RNG_N)
            cal, te = idx[:ncal], idx[ncal:]

            # linear calibration (the baseline forecast): least squares of y on u, fit on cal
            A = np.polyfit(u[cal], y[cal], 1)
            raw_te = A[0] * u[te] + A[1]

            iso   = iso_fit(u[cal], y[cal])
            binf  = bin_fit(u[cal], y[cal])
            resid = y[cal] - (A[0]*u[cal] + A[1])
            s2    = float(resid.var())
            raw_cal = A[0]*u[cal] + A[1]
            sub = rg.choice(ncal, size=min(2000, ncal), replace=False)
            tw_fn = tweedie_fit(raw_cal[sub], s2)

            base = float(np.median(y[cal]))
            yte  = y[te]
            tau  = tail_index(y[cal])

            order = np.argsort(-raw_te)
            head  = order[:max(KS)]
            tw_head = tw_fn(raw_te[head])
            iso_te  = iso(u[te]); bin_te = binf(u[te])
            for K in KS:
                top  = order[:K]
                best = np.argsort(-yte)[:K]
                ceil = float(yte[best].mean() - base)
                if ceil <= 1e-9:
                    continue
                realized = float(yte[top].mean() - base)
                rows.append(dict(
                    delta=delta, rho=rho, seed=seed, K=K, tau=tau, ceiling=ceil,
                    realized=realized,
                    fc_raw  = float(raw_te[top].mean() - base),
                    fc_iso  = float(iso_te[top].mean() - base),
                    fc_bin  = float(bin_te[top].mean() - base),
                    fc_tweed= float(tw_head[:K].mean() - base),
                ))

df = pd.DataFrame(rows)
df.to_csv('simgrid.csv', index=False)

print(f"rows={len(df)}  shapes={df.delta.nunique()}  rhos={df.rho.nunique()}")
print(f"\n{'rule':>6} {'median_s':>10} {'meanabs_s':>10} {'overpromise%':>13} {'win_vs_raw%':>12} {'wilcoxon_p':>12}")
base_abs = None
for r in ('raw','iso','bin','tweed'):
    s = (df['fc_'+r] - df['realized']) / df['ceiling']
    a = s.abs()
    if r == 'raw':
        base_abs = a
        print(f"{r:>6} {s.median():10.4f} {a.mean():10.4f} {100*(s>0).mean():13.1f} {'-':>12} {'-':>12}")
    else:
        win = 100*(a < base_abs).mean()
        p = wilcoxon(a, base_abs).pvalue
        print(f"{r:>6} {s.median():10.4f} {a.mean():10.4f} {100*(s>0).mean():13.1f} {win:12.1f} {p:12.3e}")

print("\nby tail shape (raw vs iso), s = gap / ceiling:")
print(f"{'delta':>6} {'tau':>7} {'raw_med':>9} {'iso_med':>9} {'raw_absmean':>12} {'iso_absmean':>12}")
for d, g in df.groupby('delta'):
    sr = ((g.fc_raw - g.realized)/g.ceiling); si = ((g.fc_iso - g.realized)/g.ceiling)
    print(f"{d:6.1f} {g.tau.median():7.3f} {sr.median():9.4f} {si.median():9.4f} {sr.abs().mean():12.4f} {si.abs().mean():12.4f}")

print("\nby accuracy rho (raw vs iso):")
print(f"{'rho':>5} {'raw_med':>9} {'iso_med':>9} {'raw_absmean':>12} {'iso_absmean':>12}")
for rh, g in df.groupby('rho'):
    sr = ((g.fc_raw - g.realized)/g.ceiling); si = ((g.fc_iso - g.realized)/g.ceiling)
    print(f"{rh:5.1f} {sr.median():9.4f} {si.median():9.4f} {sr.abs().mean():12.4f} {si.abs().mean():12.4f}")
