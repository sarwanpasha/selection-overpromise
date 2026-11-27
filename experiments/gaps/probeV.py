"""Probe V -- an interval for the gain a round will actually deliver.

A corrected point forecast still says nothing about how far the realized gain may
fall from it.  The calibration reviewer's objection is that a campaign needs an
interval, and that the conformal variants tried so far are crude.

The construction here uses the calibration slice to simulate the selection itself.
Draw a subsample of the calibration points the size of the candidate pool, take its
top K by the surrogate score, and record the discrepancy between the recalibrated
forecast and the realized mean on that pseudo-batch.  Repeating gives a distribution
of batch-level errors under the same selection pressure the real batch faces, and its
quantiles give an interval.  The point is that the error is measured on batches, not
on individual candidates, which is what makes the naive residual interval too narrow.

Reported for nominal 80 and 90 percent: empirical coverage, median width, and the
same quantities for a normal interval built from per-candidate residuals.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict
from probeH import tail_index, iso_fit

OUT     = os.environ.get('PROBEV_OUT', 'probeV.csv')
WORKERS = int(os.environ.get('PROBEV_WORKERS', '20'))
ROWCAP  = int(os.environ.get('PROBEV_ROWCAP', '20000'))
FRACS   = (0.1, 0.3, 0.6)
KS      = (10, 20, 50)
SEEDS   = 5
CALIB   = 0.30
NBOOT   = 300
LEVELS  = (0.80, 0.90)


def batch_errors(pc, yc, g, K, q, nboot, rng):
    """Discrepancies between forecast and realized mean on simulated batches.

    Each draw subsamples the calibration slice so that taking its top K reproduces
    the selection fraction q of the real candidate pool.
    """
    pc = np.asarray(pc, float); yc = np.asarray(yc, float)
    nc = len(pc)
    m = int(round(K / max(q, 1e-9)))
    m = int(np.clip(m, K + 1, nc))
    if m <= K:
        return np.array([])
    errs = np.empty(nboot)
    cal = np.asarray(g(pc), float)
    cal[~np.isfinite(cal)] = 0.0
    for b in range(nboot):
        sub = rng.choice(nc, size=m, replace=True)
        top = sub[np.argsort(-pc[sub])[:K]]
        errs[b] = float(cal[top].mean() - yc[top].mean())
    return errs


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
                rng0 = np.random.RandomState(1000 * seed + 7)
                idx = rng0.permutation(n)
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
                try:    g = iso_fit(pc, yc)
                except Exception: g = lambda v: np.asarray(v, float)
                cal_t = np.asarray(g(pt), float); cal_t[~np.isfinite(cal_t)] = 0.0
                base = float(np.median(yf))
                rho = spearmanr(pt, yte).statistic
                resid = yc - np.asarray(g(pc), float)
                sd_point = float(np.std(resid[np.isfinite(resid)]))
                rng = np.random.default_rng(31337 + seed)
                for K in KS:
                    if K > len(yte):
                        continue
                    q = K / float(len(yte))
                    top = np.argsort(-pt)[:K]
                    fc = float(cal_t[top].mean() - base)
                    realized = float(yte[top].mean() - base)
                    bestK = np.argsort(-yte)[:K]
                    ceiling = float(yte[bestK].mean() - base)
                    errs = batch_errors(pc, yc, g, K, q, NBOOT, rng)
                    row = dict(assay=name, frac=tf, seed=seed, K=K, q=q, tau=tau,
                               ceiling=ceiling, realized=realized, fc=fc,
                               spearman=float(rho) if np.isfinite(rho) else None)
                    for lv in LEVELS:
                        a = (1.0 - lv) / 2.0
                        if len(errs):
                            lo = fc - float(np.quantile(errs, 1 - a))
                            hi = fc - float(np.quantile(errs, a))
                        else:
                            lo = hi = np.nan
                        row['lo_%d' % int(lv * 100)] = lo
                        row['hi_%d' % int(lv * 100)] = hi
                        row['cov_%d' % int(lv * 100)] = int(lo <= realized <= hi) if np.isfinite(lo) else None
                        row['wid_%d' % int(lv * 100)] = (hi - lo) if np.isfinite(lo) else None
                        # naive interval from per-candidate residuals
                        z = norm.ppf(1 - a)
                        half = z * sd_point / np.sqrt(K)
                        row['ncov_%d' % int(lv * 100)] = int(abs(fc - realized) <= half)
                        row['nwid_%d' % int(lv * 100)] = 2 * half
                    out.append(row)
            except Exception as e:
                print('  fit-fail', name, tf, seed, type(e).__name__, e, flush=True)
                continue
    return out


def report(df):
    d = df[df.ceiling.abs() > 1e-9].copy()
    cell = d.groupby(['assay', 'frac', 'K'], as_index=False).median(numeric_only=True)
    a = cell.groupby('assay', as_index=False).mean(numeric_only=True)
    print('ASSAYS %d CELLS %d' % (cell.assay.nunique(), len(cell)), flush=True)
    for lv in LEVELS:
        p = int(lv * 100)
        print('BATCH_%d cov %d width %d' % (p,
              round(100 * float(a['cov_%d' % p].mean())),
              round(1000 * float((cell['wid_%d' % p] / cell.ceiling.abs()).median()))), flush=True)
        print('NAIVE_%d cov %d width %d' % (p,
              round(100 * float(a['ncov_%d' % p].mean())),
              round(1000 * float((cell['nwid_%d' % p] / cell.ceiling.abs()).median()))), flush=True)
    for lo, hi, lab in ((0.0, 0.8, 'lowtau'), (0.8, 1.1, 'midtau'), (1.1, 9.9, 'hightau')):
        sl = cell[(cell.tau >= lo) & (cell.tau < hi)]
        if len(sl):
            print('COV90_%s %d %d' % (lab, round(100 * float(sl.cov_90.mean())), len(sl)), flush=True)


if __name__ == '__main__':
    dpath = sys.argv[1]
    files = sorted(glob.glob(os.path.join(dpath, '*.csv')))
    JL = os.path.splitext(OUT)[0] + '_rows.jsonl'
    done = set()
    if os.path.exists(JL):
        for line in open(JL):
            try: done.add(json.loads(line)['assay'])
            except Exception: pass
    todo = [f for f in files if os.path.basename(f) not in done]
    print('%d files, %d workers; resuming %d, todo %d' % (len(files), WORKERS, len(done), len(todo)), flush=True)
    nrows = 0
    with Pool(WORKERS) as pool, open(JL, 'a') as fh:
        for i, rows in enumerate(pool.imap_unordered(run_assay, todo), 1):
            for r in rows: fh.write(json.dumps(r) + '\n')
            fh.flush(); nrows += len(rows)
            if i % 20 == 0 or i == len(todo):
                print('  %d/%d done, %d rows' % (i, len(todo), nrows), flush=True)
    df = pd.DataFrame([json.loads(l) for l in open(JL)])
    df.to_csv(OUT, index=False)
    print('=== PROBE V ===', flush=True)
    try:
        report(df)
    except Exception as e:
        print('REPORT FAIL', type(e).__name__, e, flush=True)
    print('PROBE_V_DONE', flush=True)
