"""Probe M -- does the gap persist over successive design rounds?

Model-guided design is iterative: select a batch, measure it, retrain, select again.
Round 1 is what every earlier probe measures. This runs three rounds, feeding each
measured batch back into training, and asks whether the gap shrinks as the training
set grows and comes to include the top of the distribution.
"""
import os, sys, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict
from probeH import tail_index, iso_fit
from scipy.stats import spearmanr

OUT     = os.environ.get('PROBEM_OUT', 'probeM.csv')
WORKERS = int(os.environ.get('PROBEM_WORKERS', '32'))
ROWCAP  = 20000
FRAC0   = 0.10
ROUNDS  = 3
K       = 20
SEEDS   = 5
CALIB   = 0.30


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
    if len(parsed) < 300:
        return []
    y = pd.to_numeric(df[sc].values[keep], errors='coerce').astype(float)
    X = featurize(parsed, sorted({p[1] for p in parsed}))
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    n = len(y)
    if n < 300 or y.std() < 1e-9:
        return []
    tau, out = tail_index(y), []
    for seed in range(SEEDS):
        rng = np.random.RandomState(1000 * seed + 7)
        idx = rng.permutation(n)
        ntr = max(120, int(round(FRAC0 * n)))
        if n - ntr < 120:
            continue
        train = list(idx[:ntr])
        pool_i = list(idx[ntr:])
        for rnd in range(1, ROUNDS + 1):
            if len(pool_i) < K + 50 or len(train) < 100:
                break
            tr = np.array(train); po = np.array(pool_i)
            ncal = max(50, int(round(CALIB * len(tr))))
            if len(tr) - ncal < 60:
                break
            fit_i, cal_i = tr[:-ncal], tr[-ncal:]
            if y[fit_i].std() < 1e-9 or y[cal_i].std() < 1e-9:
                break
            try:
                w, ym, mu, lam, A_inv, s2 = ridge_fit(X[fit_i], y[fit_i])
                pc, _ = predict(X[cal_i], w, ym, mu, A_inv, s2)
                pt, _ = predict(X[po], w, ym, mu, A_inv, s2)
            except Exception:
                break
            if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
                break
            try:    iso_te = np.asarray(iso_fit(pc, y[cal_i])(pt), float)
            except Exception: iso_te = pt.astype(float)
            iso_te[~np.isfinite(iso_te)] = 0.0
            base = float(np.median(y[fit_i]))
            ypo  = y[po]
            top  = np.argsort(-pt)[:K]
            best = np.argsort(-ypo)[:K]
            ceil = float(ypo[best].mean() - base)
            rho  = spearmanr(pt, ypo).statistic
            if ceil > 1e-9:
                out.append(dict(assay=name, seed=seed, round=rnd, tau=tau,
                    n_train=len(tr), ceiling=ceil,
                    realized=float(ypo[top].mean() - base),
                    fc_raw=float(pt[top].mean() - base),
                    fc_iso=float(iso_te[top].mean() - base),
                    spearman=float(rho) if np.isfinite(rho) else None))
            # measure the batch and feed it back
            chosen = [po[i] for i in top]
            train.extend(chosen)
            keepset = set(int(c) for c in chosen)
            pool_i = [i for i in pool_i if int(i) not in keepset]
    return out


def run(f):
    try:
        return _run(f)
    except Exception as e:
        print('FAILED', os.path.basename(f), type(e).__name__, e, flush=True)
        return []


if __name__ == '__main__':
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
    g = d.groupby(['assay', 'round'], as_index=False).median(numeric_only=True)
    lines = ['ROWS = %d' % len(d), 'ASSAYS = %d' % g.assay.nunique()]
    for rnd, x in g.groupby('round'):
        sr = (x.fc_raw - x.realized) / x.ceiling
        si = (x.fc_iso - x.realized) / x.ceiling
        lines.append('round=%d n=%d ntrain=%.0f rho=%.3f raw_med=%.4f raw_mab=%.4f '
                     'iso_med=%.4f iso_mab=%.4f over=%.1f'
                     % (rnd, len(x), x.n_train.median(), x.spearman.median(),
                        sr.median(), sr.abs().mean(), si.median(), si.abs().mean(),
                        100 * (sr > 0).mean()))
    lines.append('PROBE_M_DONE')
    open('probeM_report.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
