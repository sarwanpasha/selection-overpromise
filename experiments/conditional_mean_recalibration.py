"""
Probe H -- does conditional-mean recalibration remove the forecast bias?

The theorem says the shortfall is E[(linear projection - m(u)) | selected], where
m(u) = E[y|u]. So a predictor that estimates m directly should have zero expected
shortfall. This tests that on real assays, against the linear-calibration baseline.

Forecast rules (all fitted on a held-out slice of the TRAINING data only):
  raw   : the fitted ridge prediction (linear calibration)   [BASELINE]
  iso   : isotonic regression of y on yhat
  bin   : binned conditional mean of y given yhat
  tweed : Efron's Tweedie correction  m(v) = v + s2 * dlog f(v)/dv

Isotonic and binned are monotone, so they leave the selected batch unchanged and
only alter the forecast. Tweedie need not be monotone, so it is also evaluated as a
SELECTION rule, which tests whether a conditional-mean score picks a better batch.

Reports per (assay, frac, seed, K): forecast under each rule, realized gain,
ceiling, and the tail index tau, so the forecast bias can be related to tail shape.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict

OUTCSV  = os.environ.get('PROBEH_OUT', 'probeH.csv')
WORKERS = int(os.environ.get('PROBEH_WORKERS', '8'))
ROWCAP  = 20000
FRACS   = (0.1, 0.3, 0.6)
KS      = (10, 20, 50)
SEEDS   = 5
CALIB   = 0.30
GAUSS_Q = 3.449049          # (q99-q50)/(q75-q50) for a Gaussian


def tail_index(y):
    q50, q75, q99 = np.percentile(y, [50, 75, 99])
    if q75 - q50 < 1e-9:
        return float('nan')
    return float(((q99 - q50) / (q75 - q50)) / GAUSS_Q)


def iso_fit(x, y):
    """Isotonic regression, returns a callable that interpolates."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(x, y)
    return lambda v: ir.predict(np.clip(v, x.min(), x.max()))


def bin_fit(x, y, nbins=20):
    """Binned conditional mean with linear interpolation between bin centers."""
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return lambda v: np.full_like(np.asarray(v, dtype=float), float(y.mean()))
    idx = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
    centers, means = [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() >= 5:
            centers.append(x[m].mean()); means.append(y[m].mean())
    if len(centers) < 2:
        return lambda v: np.full_like(np.asarray(v, dtype=float), float(y.mean()))
    centers = np.array(centers); means = np.array(means)
    return lambda v: np.interp(np.asarray(v, dtype=float), centers, means)


def tweedie_fit(x, s2, bw=None):
    """Efron's Tweedie correction m(v) = v + s2 * d log f(v) / dv.

    f is the marginal density of the prediction, estimated by a Gaussian KDE on the
    calibration slice. The score is computed in closed form for the KDE.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    sd = x.std()
    if not np.isfinite(sd) or sd <= 0:
        return lambda v: np.asarray(v, dtype=float)
    h = bw if bw else 1.06 * sd * n ** (-1 / 5.0)
    h = max(h, 1e-6)

    def m(v):
        v = np.asarray(v, dtype=float)
        z = (v[:, None] - x[None, :]) / h
        w = np.exp(-0.5 * z * z)
        den = w.sum(1) + 1e-300
        # d/dv log f(v) for a Gaussian KDE
        score = -(w * z).sum(1) / (den * h)
        return v + s2 * score
    return m


def run_assay(f):
    try:
        return _run_assay(f)
    except Exception as e:
        print('FAILED', os.path.basename(f), type(e).__name__, e, flush=True)
        return []


def _run_assay(f):
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
    positions = sorted({p[1] for p in parsed})
    X = featurize(parsed, positions)
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
                Xf, yf = X[fit_i], y[fit_i]
                Xc, yc = X[cal_i], y[cal_i]
                Xte, yte = X[te], y[te]
                if yf.std() < 1e-9 or yc.std() < 1e-9:
                    continue

                w, ym, mu, lam, A_inv, s2 = ridge_fit(Xf, yf)
                pc, _ = predict(Xc, w, ym, mu, A_inv, s2)
                pt, _ = predict(Xte, w, ym, mu, A_inv, s2)
                if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
                    continue

                base = float(np.median(yf))
                s2c = float(np.var(yc - pc))

                try:    g_iso = iso_fit(pc, yc)
                except Exception: g_iso = lambda v: np.asarray(v, dtype=float)
                g_bin = bin_fit(pc, yc)
                g_tw  = tweedie_fit(pc, s2c)

                cal_iso = np.asarray(g_iso(pt), dtype=float)
                cal_bin = np.asarray(g_bin(pt), dtype=float)
                cal_tw  = np.asarray(g_tw(pt),  dtype=float)
                for a in (cal_iso, cal_bin, cal_tw):
                    a[~np.isfinite(a)] = 0.0

                rho = spearmanr(pt, yte).statistic
                for K in KS:
                    if K > len(yte):
                        continue
                    top  = np.argsort(-pt)[:K]
                    best = np.argsort(-yte)[:K]
                    ceiling  = float(yte[best].mean() - base)
                    realized = float(yte[top].mean() - base)
                    row = dict(assay=name, frac=tf, seed=seed, K=K, tau=tau,
                               n=int(n), base=base, ceiling=ceiling,
                               realized=realized,
                               fc_raw=float(pt[top].mean() - base),
                               fc_iso=float(cal_iso[top].mean() - base),
                               fc_bin=float(cal_bin[top].mean() - base),
                               fc_tweed=float(cal_tw[top].mean() - base),
                               spearman=float(rho) if np.isfinite(rho) else None)
                    # Tweedie as a SELECTION rule (it need not be monotone)
                    tw_top = np.argsort(-cal_tw)[:K]
                    row['realized_tweedsel'] = float(yte[tw_top].mean() - base)
                    row['tweedsel_same'] = int(set(tw_top.tolist()) == set(top.tolist()))
                    out.append(row)
            except Exception as e:
                print('  fit-fail', name, tf, seed, type(e).__name__, e, flush=True)
                continue
    return out


def report(df):
    from scipy.stats import wilcoxon
    d = df[df.ceiling > 1e-9].copy()
    lines = []
    g = d.groupby(['assay', 'frac', 'K'], as_index=False).mean(numeric_only=True)
    lines.append(('TOTAL_ROWS', len(df)))
    lines.append(('CELLS', len(g)))
    lines.append(('ASSAYS', d.assay.nunique()))
    for r in ('raw', 'iso', 'bin', 'tweed'):
        s = (g['fc_' + r] - g['realized']) / g['ceiling']
        lines.append(('SHORTFALL_%s_median' % r, round(float(s.median()), 4)))
        lines.append(('SHORTFALL_%s_meanabs' % r, round(float(s.abs().mean()), 4)))
        lines.append(('OVERPROMISE_PCT_%s' % r, round(100 * float((s > 0).mean()), 1)))
    # does a conditional-mean SCORE pick a better batch?
    dd = (g['realized_tweedsel'] - g['realized']) / g['ceiling']
    try:    pv = wilcoxon(g['realized_tweedsel'], g['realized']).pvalue
    except Exception: pv = float('nan')
    lines.append(('TWEEDSEL_delta_median', round(float(dd.median()), 5)))
    lines.append(('TWEEDSEL_winpct', round(100 * float((dd > 0).mean()), 1)))
    lines.append(('TWEEDSEL_log10p', round(float(np.log10(pv)), 2) if np.isfinite(pv) else 'NA'))
    lines.append(('TWEEDSEL_identical_pct', round(100 * float(g['tweedsel_same'].mean()), 1)))
    txt = '\n'.join('%s = %s' % (k, v) for k, v in lines)
    open('probeH_report.txt', 'w').write(txt)
    print(txt, flush=True)


if __name__ == '__main__':
    dpath = sys.argv[1]
    files = sorted(glob.glob(os.path.join(dpath, '*.csv')))
    JL = os.path.splitext(OUTCSV)[0] + '_rows.jsonl'
    done = set()
    if os.path.exists(JL):
        for line in open(JL):
            try: done.add(json.loads(line)['assay'])
            except Exception: pass
    todo = [f for f in files if os.path.basename(f) not in done]
    print(f'{len(files)} files, {WORKERS} workers; resuming {len(done)}, todo {len(todo)}', flush=True)
    nrows = 0
    with Pool(WORKERS) as pool, open(JL, 'a') as fh:
        for i, rows in enumerate(pool.imap_unordered(run_assay, todo), 1):
            for r in rows: fh.write(json.dumps(r) + '\n')
            fh.flush(); nrows += len(rows)
            if i % 10 == 0 or i == len(todo):
                print(f'  {i}/{len(todo)} done, {nrows} rows', flush=True)
    df = pd.DataFrame([json.loads(l) for l in open(JL)])
    df.to_csv(OUTCSV, index=False)
    print('=== PROBE H ===', flush=True)
    report(df)
    print('PROBE_H_DONE', flush=True)
