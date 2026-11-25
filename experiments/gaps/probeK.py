"""Probe K -- does the gap survive a stronger surrogate?

Same splits, seeds, fractions and budgets as probeH, but the surrogate is swapped
between ridge, random forest, gradient boosting and k-nearest neighbours. Theory
says the gap is a property of the fitness distribution, not of the model, and that
it does not shrink as rank accuracy rises. This tests that directly.

Also sweeps the calibration slice size, since isotonic recalibration is fitted on it.
"""
import os, sys, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict
from probeH import tail_index, iso_fit
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import spearmanr

OUT     = os.environ.get('PROBEK_OUT', 'probeK.csv')
WORKERS = int(os.environ.get('PROBEK_WORKERS', '32'))
ROWCAP  = int(os.environ.get('PROBEK_ROWCAP', '6000'))
FRACS   = (0.1, 0.3, 0.6)
KS      = (10, 20, 50)
SEEDS   = 3
CALIBS  = (0.20, 0.30, 0.45)
MODELS  = ('ridge', 'rf', 'gbm', 'knn')


def fit_predict(name, Xf, yf, Xc, Xt, seed):
    """Return (cal_pred, test_pred) for one surrogate."""
    if name == 'ridge':
        w, ym, mu, lam, A_inv, s2 = ridge_fit(Xf, yf)
        pc, _ = predict(Xc, w, ym, mu, A_inv, s2)
        pt, _ = predict(Xt, w, ym, mu, A_inv, s2)
        return np.asarray(pc, float), np.asarray(pt, float)
    if name == 'rf':
        m = RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                  min_samples_leaf=2, n_jobs=1, random_state=seed)
    elif name == 'gbm':
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                          random_state=seed)
    elif name == 'knn':
        m = KNeighborsRegressor(n_neighbors=min(10, max(1, len(yf) // 10)))
    else:
        raise ValueError(name)
    m.fit(Xf, yf)
    return m.predict(Xc).astype(float), m.predict(Xt).astype(float)


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
    tau, n, out = tail_index(y), len(y), []
    for tf in FRACS:
        ntr = int(round(tf * n))
        if ntr < 100 or n - ntr < 100:
            continue
        for seed in range(SEEDS):
            rng = np.random.RandomState(1000 * seed + 7)
            idx = rng.permutation(n)
            tr, te = idx[:ntr], idx[ntr:]
            for cal_frac in CALIBS:
                ncal = max(50, int(round(cal_frac * ntr)))
                if ntr - ncal < 80:
                    continue
                fit_i, cal_i = tr[:-ncal], tr[-ncal:]
                if y[fit_i].std() < 1e-9 or y[cal_i].std() < 1e-9:
                    continue
                base, yte = float(np.median(y[fit_i])), y[te]
                for mdl in MODELS:
                    try:
                        pc, pt = fit_predict(mdl, X[fit_i], y[fit_i],
                                             X[cal_i], X[te], seed)
                    except Exception as e:
                        print('  fail', name, mdl, type(e).__name__, flush=True)
                        continue
                    if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
                        continue
                    try:    iso_te = np.asarray(iso_fit(pc, y[cal_i])(pt), float)
                    except Exception: iso_te = pt.astype(float)
                    iso_te[~np.isfinite(iso_te)] = 0.0
                    rho = spearmanr(pt, yte).statistic
                    for K in KS:
                        if K > len(yte):
                            continue
                        top  = np.argsort(-pt)[:K]
                        best = np.argsort(-yte)[:K]
                        ceil = float(yte[best].mean() - base)
                        if ceil <= 1e-9:
                            continue
                        out.append(dict(assay=name, model=mdl, frac=tf, seed=seed,
                            calib=cal_frac, K=K, tau=tau, n=n, ceiling=ceil,
                            realized=float(yte[top].mean() - base),
                            fc_raw=float(pt[top].mean() - base),
                            fc_iso=float(iso_te[top].mean() - base),
                            spearman=float(rho) if np.isfinite(rho) else None))
    return out


def run(f):
    try:
        return _run(f)
    except Exception as e:
        print('FAILED', os.path.basename(f), type(e).__name__, e, flush=True)
        return []


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(sys.argv[1], '*.csv')))
    print(len(files), 'files,', WORKERS, 'workers,', len(MODELS), 'models', flush=True)
    rows, done = [], 0
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(run, files):
            rows.extend(r); done += 1
            if done % 10 == 0:
                print('%d/%d done, %d rows' % (done, len(files), len(rows)), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    d = d[d.ceiling > 1e-9]
    lines = ['ROWS = %d' % len(d), 'ASSAYS = %d' % d.assay.nunique()]
    g = d.groupby(['assay', 'model', 'frac', 'K', 'calib'], as_index=False).median(numeric_only=True)
    for mdl, x in g.groupby('model'):
        sr = (x.fc_raw - x.realized) / x.ceiling
        si = (x.fc_iso - x.realized) / x.ceiling
        lines.append('%s cells=%d rho=%.3f raw_med=%.4f raw_mab=%.4f iso_med=%.4f '
                     'iso_mab=%.4f over=%.1f' % (mdl, len(x), x.spearman.median(),
                     sr.median(), sr.abs().mean(), si.median(), si.abs().mean(),
                     100 * (sr > 0).mean()))
    for cf, x in g.groupby('calib'):
        si = ((x.fc_iso - x.realized) / x.ceiling).abs()
        lines.append('calib=%.2f iso_mab=%.4f' % (cf, si.mean()))
    lines.append('PROBE_K_DONE')
    open('probeK_report.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
