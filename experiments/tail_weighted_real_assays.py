"""Probe I -- tail-weighted shrinkage between linear calibration and isotonic.

Identical protocol to probeH (same splits, seeds, fracs, K) so the numbers are
directly comparable. Adds two rules:
  shr_all  : convex weight chosen by 5-fold CV on all calibration points
  shr_tail : same, but the CV objective is restricted to the top scores, the
             region selection actually draws from
A convex blend of two increasing maps is increasing, so the batch is unchanged.
"""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict
from probeH import tail_index, iso_fit, FRACS, KS, SEEDS, CALIB, ROWCAP

OUT     = os.environ.get('PROBEI_OUT', 'probeI.csv')
WORKERS = int(os.environ.get('PROBEI_WORKERS', '8'))
WGRID   = np.linspace(0, 1, 11)
NFOLD   = 5
TAILFRAC = 0.20
MINTAIL  = 25


def pick_w(x, y, tail):
    n = len(x)
    if n < 60:
        return 1.0
    thr = -np.inf
    if tail:
        k = max(MINTAIL, int(round(TAILFRAC * n)))
        if k < n:
            thr = np.partition(x, n - k)[n - k]
    rs = np.random.RandomState(12345)
    fold = rs.permutation(n) % NFOLD
    err = np.zeros(len(WGRID))
    for f in range(NFOLD):
        tr, va = fold != f, fold == f
        if tr.sum() < 40 or va.sum() < 5:
            return 1.0
        try:
            A = np.polyfit(x[tr], y[tr], 1)
            lin = A[0] * x[va] + A[1]
            iso = np.asarray(iso_fit(x[tr], y[tr])(x[va]), dtype=float)
        except Exception:
            return 1.0
        m = x[va] >= thr
        if m.sum() < 3:
            m = np.ones_like(m, dtype=bool)
        for j, w in enumerate(WGRID):
            d = y[va][m] - (w * iso + (1 - w) * lin)[m]
            err[j] += float(np.sum(d * d))
    return float(WGRID[int(np.argmin(err))])


def _run(f):
    name = os.path.basename(f)
    df = pd.read_csv(f)
    sc = 'DMS_score' if 'DMS_score' in df.columns else df.columns[-1]
    mc = 'mutant' if 'mutant' in df.columns else df.columns[0]
    if len(df) > ROWCAP:
        df = df.sample(ROWCAP, random_state=0)
    parsed, keep = [], []
    for i, mm in enumerate(df[mc].astype(str).values):
        if '/' in mm or ':' in mm or ',' in mm:
            continue
        p = parse(mm)
        if p is not None:
            parsed.append(p); keep.append(i)
    if len(parsed) < 200:
        return []
    y = pd.to_numeric(df[sc].values[keep], errors='coerce').astype(float)
    X = featurize(parsed, sorted({p[1] for p in parsed}))
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    if len(y) < 200 or y.std() < 1e-9:
        return []
    tau = tail_index(y)
    n, out = len(y), []
    for tf in FRACS:
        ntr = int(round(tf * n))
        if ntr < 100 or n - ntr < 100:
            continue
        for seed in range(SEEDS):
            try:
                rng = np.random.RandomState(1000 * seed + 7)
                idx = rng.permutation(n)
                tr, te = idx[:ntr], idx[ntr:]
                ncal = max(50, int(round(CALIB * ntr)))
                if ntr - ncal < 80:
                    continue
                fit_i, cal_i = tr[:-ncal], tr[-ncal:]
                if y[fit_i].std() < 1e-9 or y[cal_i].std() < 1e-9:
                    continue
                w_, ym, mu, lam, A_inv, s2 = ridge_fit(X[fit_i], y[fit_i])
                pc, _ = predict(X[cal_i], w_, ym, mu, A_inv, s2)
                pt, _ = predict(X[te], w_, ym, mu, A_inv, s2)
                if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
                    continue
                yc, yte = y[cal_i], y[te]
                base = float(np.median(y[fit_i]))
                try:    iso_te = np.asarray(iso_fit(pc, yc)(pt), dtype=float)
                except Exception: iso_te = pt.astype(float)
                iso_te[~np.isfinite(iso_te)] = 0.0
                wa = pick_w(pc, yc, False)
                wt = pick_w(pc, yc, True)
                shr_a = wa * iso_te + (1 - wa) * pt
                shr_t = wt * iso_te + (1 - wt) * pt
                for K in KS:
                    if K > len(yte):
                        continue
                    top  = np.argsort(-pt)[:K]
                    best = np.argsort(-yte)[:K]
                    ceiling = float(yte[best].mean() - base)
                    out.append(dict(assay=name, frac=tf, seed=seed, K=K, tau=tau,
                        ceiling=ceiling, realized=float(yte[top].mean() - base),
                        w_all=wa, w_tail=wt,
                        fc_raw=float(pt[top].mean() - base),
                        fc_iso=float(iso_te[top].mean() - base),
                        fc_shra=float(shr_a[top].mean() - base),
                        fc_shrt=float(shr_t[top].mean() - base)))
            except Exception as e:
                print('  fit-fail', name, tf, seed, type(e).__name__, e, flush=True)
    return out


def run(f):
    try:
        return _run(f)
    except Exception as e:
        print('FAILED', os.path.basename(f), type(e).__name__, e, flush=True)
        return []


if __name__ == '__main__':
    import glob
    files = sorted(glob.glob(os.path.join(sys.argv[1], '*.csv')))
    print(len(files), 'files,', WORKERS, 'workers', flush=True)
    rows, done = [], 0
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(run, files):
            rows.extend(r); done += 1
            if done % 20 == 0:
                print('%d/%d done, %d rows' % (done, len(files), len(rows)), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    d = d[d.ceiling > 1e-9]
    g = d.groupby(['assay', 'frac', 'K'], as_index=False).median(numeric_only=True)
    from scipy.stats import wilcoxon
    lines = ['CELLS = %d' % len(g), 'ASSAYS = %d' % g.assay.nunique(),
             'W_ALL_MEAN = %.4f' % g.w_all.mean(), 'W_TAIL_MEAN = %.4f' % g.w_tail.mean()]
    ar = ((g.fc_raw - g.realized) / g.ceiling).abs()
    for r in ('raw', 'iso', 'shra', 'shrt'):
        s = (g['fc_' + r] - g.realized) / g.ceiling
        a = s.abs()
        lines.append('%s_median = %.4f' % (r, s.median()))
        lines.append('%s_meanabs = %.4f' % (r, a.mean()))
        if r != 'raw':
            lines.append('%s_win = %.1f' % (r, 100 * (a < ar).mean()))
            lines.append('%s_log10p = %.1f' % (r, np.log10(max(wilcoxon(a, ar).pvalue, 1e-300))))
    ai = ((g.fc_iso - g.realized) / g.ceiling).abs()
    at = ((g.fc_shrt - g.realized) / g.ceiling).abs()
    lines.append('SHRT_vs_ISO_win = %.1f' % (100 * (at < ai).mean()))
    lines.append('SHRT_vs_ISO_log10p = %.1f' % np.log10(max(wilcoxon(at, ai).pvalue, 1e-300)))
    lines.append('PROBE_I_DONE')
    open('probeI_report.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
