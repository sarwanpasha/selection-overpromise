"""Probe P -- conformal baselines and leaderboard-strength surrogates.

Same splits as probeH. Surrogates:
  ridge : the additive ridge used throughout
  zs    : a zero-shot protein language model score, mapped to the fitness scale by a
          least-squares line fitted on the training fraction
  aug   : ridge on the additive features with the zero-shot score appended
Forecast rules for each surrogate:
  raw        : the reported score
  iso        : isotonic conditional mean fitted on the calibration slice
  conf_shift : split-conformal recentring, score plus the median calibration residual
  conf_tail  : the same, using only calibration points in the top 20% of scores
  conf_lo    : one-sided 90% split-conformal lower bound
"""
import os, sys, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict
from probeH import tail_index, iso_fit, FRACS, KS, SEEDS, CALIB, ROWCAP
from scipy.stats import spearmanr

WORKERS = int(os.environ.get('PROBEP_WORKERS', '24'))
ZSDIR   = os.environ.get('PROBEP_ZS', '')
ZSMODELS = [m for m in os.environ.get('PROBEP_MODELS', '').split(',') if m]
RULES = ('raw', 'iso', 'conf_shift', 'conf_tail', 'conf_lo')


def zs_scores(name, mutants):
    """Return {model: array aligned to mutants} for zero-shot files that exist."""
    out = {}
    for m in ZSMODELS:
        f = os.path.join(ZSDIR, m, name)
        if not os.path.exists(f):
            continue
        z = pd.read_csv(f)
        num = [c for c in z.columns if c not in ('mutant', 'mutated_sequence', 'DMS_score',
               'DMS_score_bin') and pd.api.types.is_numeric_dtype(z[c])]
        if 'mutant' not in z.columns or not num:
            continue
        s = z.drop_duplicates('mutant').set_index('mutant')[num[-1]]
        v = s.reindex(mutants).values.astype(float)
        if np.isfinite(v).mean() > 0.95:
            v[~np.isfinite(v)] = np.nanmedian(v)
            out[m] = v
    return out


def rules(pc, yc, pt):
    r = {'raw': pt}
    try:    iso = np.asarray(iso_fit(pc, yc)(pt), float)
    except Exception: iso = pt.astype(float)
    iso[~np.isfinite(iso)] = 0.0
    r['iso'] = iso
    res = yc - pc
    r['conf_shift'] = pt + np.median(res)
    k = max(10, int(round(0.2 * len(pc))))
    top = np.argsort(-pc)[:k]
    r['conf_tail'] = pt + np.median(res[top])
    r['conf_lo'] = pt + np.quantile(res, 0.10)
    return r


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
    muts = df[mc].astype(str).values[keep]
    y = pd.to_numeric(df[sc].values[keep], errors='coerce').astype(float)
    X = featurize(parsed, sorted({p[1] for p in parsed}))
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y, muts = X[ok], y[ok], muts[ok]
    if len(y) < 200 or y.std() < 1e-9:
        return []
    zs = zs_scores(name, muts) if ZSDIR else {}
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
            base, yte, yc = float(np.median(y[fit_i])), y[te], y[cal_i]
            preds = {}
            try:
                w, ym, mu, lam, A_inv, s2 = ridge_fit(X[fit_i], y[fit_i])
                preds['ridge'] = (predict(X[cal_i], w, ym, mu, A_inv, s2)[0],
                                  predict(X[te], w, ym, mu, A_inv, s2)[0])
            except Exception:
                pass
            for m, v in zs.items():
                try:
                    A = np.polyfit(v[fit_i], y[fit_i], 1)
                    preds['zs:' + m] = (A[0] * v[cal_i] + A[1], A[0] * v[te] + A[1])
                    sd = v[fit_i].std() or 1.0
                    Xa = np.column_stack([X, 5.0 * (v - v[fit_i].mean()) / sd])
                    w, ym, mu, lam, A_inv, s2 = ridge_fit(Xa[fit_i], y[fit_i])
                    preds['aug:' + m] = (predict(Xa[cal_i], w, ym, mu, A_inv, s2)[0],
                                         predict(Xa[te], w, ym, mu, A_inv, s2)[0])
                except Exception:
                    pass
            for mdl, (pc, pt) in preds.items():
                pc, pt = np.asarray(pc, float), np.asarray(pt, float)
                if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
                    continue
                fr = rules(pc, yc, pt)
                rho = spearmanr(pt, yte).statistic
                for K in KS:
                    if K > len(yte):
                        continue
                    top, best = np.argsort(-pt)[:K], np.argsort(-yte)[:K]
                    ceil = float(yte[best].mean() - base)
                    row = dict(assay=name, model=mdl, frac=tf, seed=seed, K=K, tau=tau,
                               ceiling=ceil, realized=float(yte[top].mean() - base),
                               spearman=float(rho) if np.isfinite(rho) else None)
                    for rn in RULES:
                        row['fc_' + rn] = float(fr[rn][top].mean() - base)
                    out.append(row)
    return out


def run(f):
    try:
        return _run(f)
    except Exception as e:
        print('FAILED', os.path.basename(f), type(e).__name__, e, flush=True)
        return []


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(sys.argv[1], '*.csv')))
    print(len(files), 'files,', WORKERS, 'workers, zs models:', ZSMODELS, flush=True)
    rows, done = [], 0
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(run, files):
            rows.extend(r); done += 1
            if done % 20 == 0:
                print('%d/%d done, %d rows' % (done, len(files), len(rows)), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.environ.get('PROBEP_OUT', 'probeP.csv'), index=False)
    d = d[d.ceiling > 1e-9]
    g = d.groupby(['assay', 'model', 'frac', 'K'], as_index=False).median(numeric_only=True)
    L = ['ROWS = %d' % len(d)]
    for mdl, x in g.groupby('model'):
        L.append('MODEL %s cells=%d assays=%d rho=%.3f captured=%.3f'
                 % (mdl, len(x), x.assay.nunique(), x.spearman.median(),
                    (x.realized / x.ceiling).median()))
        for rn in RULES:
            s = (x['fc_' + rn] - x.realized) / x.ceiling
            L.append('  %s %s med=%.4f mab=%.4f over=%.1f' % (mdl, rn, s.median(), s.abs().mean(), 100 * (s > 0).mean()))
    L.append('PROBE_P_DONE')
    open(os.environ.get('PROBEP_REP', 'probeP_report.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)
