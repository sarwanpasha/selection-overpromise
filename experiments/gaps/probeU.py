"""Probe U -- is there a tighter statistic than tau inside the Hermite spectrum?

The theory reviewer's question: tau compresses the whole Hermite spectrum of the
fitness quantile transform into a single ratio of two quantiles, which must lose
information.  Does a statistic built from the spectrum itself forecast the gap better?

Every candidate statistic here is computed from the training measurements alone, so
each is available before the campaign spends its budget.  They are compared by
leave-one-assay-out forecasting, exactly as the tau rule is in the paper.

  zero  : assume no gap, which is what trusting the surrogate amounts to
  rho   : accuracy alone
  tau   : log tau and rho, the rule used in the paper
  model : the closed form of Theorem 3.3 evaluated on the training quantile function
  spec  : the leading Hermite coefficients c_2 .. c_8 of that quantile function
  both  : the closed form together with log tau
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict
from probeH import tail_index
from probeO import model_gap

OUT     = os.environ.get('PROBEU_OUT', 'probeU.csv')
WORKERS = int(os.environ.get('PROBEU_WORKERS', '20'))
ROWCAP  = int(os.environ.get('PROBEU_ROWCAP', '20000'))
FRACS   = (0.1, 0.3, 0.6)
KS      = (10, 20, 50)
SEEDS   = 5
KMAX    = 8


def hermite_coeffs(ys, kmax=KMAX):
    """c_k = E[G(Z) He_k(Z)] / k! for the empirical quantile transform of ys.

    c_1 is the linear part; the gap of Theorem 3.3 is driven by k >= 2.
    """
    ys = np.sort(np.asarray(ys, float))
    n = len(ys)
    pp = (np.arange(n) + 0.5) / n
    z = norm.ppf(pp)
    out = []
    he_prev, he_cur = np.ones_like(z), z          # He_0, He_1
    fact = 1.0
    for k in range(1, kmax + 1):
        fact *= k
        out.append(float(np.mean(ys * he_cur) / fact))
        he_prev, he_cur = he_cur, z * he_cur - k * he_prev
    return out                                     # [c_1, c_2, ..., c_kmax]


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
                ytr = y[tr]
                if ytr.std() < 1e-9:
                    continue
                w, ym, mu, lam, A_inv, s2 = ridge_fit(X[tr], ytr)
                pt, _ = predict(X[te], w, ym, mu, A_inv, s2)
                if not np.isfinite(pt).all():
                    continue
                yte = y[te]
                rho = spearmanr(pt, yte).statistic
                if not np.isfinite(rho):
                    continue
                rho_star = float(np.clip(2.0 * np.sin(np.pi * rho / 6.0), -0.999, 0.999))
                cs = hermite_coeffs(ytr)
                scale = max(abs(cs[0]), 1e-12)        # scale-free spectrum
                cnorm = [c / scale for c in cs[1:]]
                tau_tr = tail_index(ytr)
                base = float(np.median(ytr))
                ys_sorted = np.sort(ytr)
                for K in KS:
                    if K > len(yte):
                        continue
                    q = K / float(len(yte))
                    try:
                        fc_m, re_m, ce_m = model_gap(ys_sorted, rho_star, q)
                        model_s = (fc_m - re_m) / ce_m if abs(ce_m) > 1e-9 else np.nan
                    except Exception:
                        model_s = np.nan
                    top  = np.argsort(-pt)[:K]
                    best = np.argsort(-yte)[:K]
                    ceiling = float(yte[best].mean() - base)
                    if abs(ceiling) < 1e-9:
                        continue
                    row = dict(assay=name, frac=tf, seed=seed, K=K, q=q,
                               rho=float(rho), tau_train=float(tau_tr),
                               model_s=float(model_s) if np.isfinite(model_s) else None,
                               ceiling=ceiling,
                               realized=float(yte[top].mean() - base),
                               fc_raw=float(pt[top].mean() - base))
                    for j, c in enumerate(cnorm, start=2):
                        row['c%d' % j] = float(c)
                    out.append(row)
            except Exception as e:
                print('  fit-fail', name, tf, seed, type(e).__name__, e, flush=True)
                continue
    return out


def loao(cell, cols):
    """Leave-one-assay-out mean absolute error of a linear rule on `cols`."""
    assays = cell.assay.unique()
    err = []
    for a in assays:
        tr = cell[cell.assay != a]
        te = cell[cell.assay == a]
        A = np.column_stack([tr[c].values for c in cols] + [np.ones(len(tr))])
        b = tr.s_raw.values
        m = np.isfinite(A).all(1) & np.isfinite(b)
        if m.sum() < 20:
            continue
        coef, *_ = np.linalg.lstsq(A[m], b[m], rcond=None)
        B = np.column_stack([te[c].values for c in cols] + [np.ones(len(te))])
        mt = np.isfinite(B).all(1) & np.isfinite(te.s_raw.values)
        if mt.sum() == 0:
            continue
        pred = B[mt] @ coef
        err.append(float(np.mean(np.abs(pred - te.s_raw.values[mt]))))
    return float(np.mean(err)), len(err)


def report(df):
    d = df[df.ceiling.abs() > 1e-9].copy()
    d['s_raw'] = (d.fc_raw - d.realized) / d.ceiling
    cell = d.groupby(['assay', 'frac', 'K'], as_index=False).median(numeric_only=True)
    cell['logtau'] = np.log(cell.tau_train.clip(lower=1e-6))
    cell = cell[np.isfinite(cell.s_raw)]
    print('ASSAYS %d CELLS %d' % (cell.assay.nunique(), len(cell)), flush=True)
    zero = float(np.mean(np.abs(cell.s_raw)))
    print('LOAO_zero %d' % round(1000 * zero), flush=True)
    cs = ['c%d' % j for j in range(2, KMAX + 1)]
    for name, cols in (('rho', ['rho']),
                       ('tau', ['logtau', 'rho']),
                       ('model', ['model_s', 'rho']),
                       ('spec', cs + ['rho']),
                       ('both', ['model_s', 'logtau', 'rho'])):
        try:
            mae, k = loao(cell, cols)
            print('LOAO_%s %d %d' % (name, round(1000 * mae), k), flush=True)
        except Exception as e:
            print('LOAO_%s FAIL %s' % (name, e), flush=True)
    for c in ['logtau', 'model_s'] + cs:
        m = np.isfinite(cell[c]) & np.isfinite(cell.s_raw)
        if m.sum() > 20:
            r = spearmanr(cell[c][m], cell.s_raw[m])
            print('CORR_%s %d %d' % (c, round(1000 * r.statistic),
                  round(np.log10(r.pvalue)) if r.pvalue > 0 else -999), flush=True)


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
    print('=== PROBE U ===', flush=True)
    try:
        report(df)
    except Exception as e:
        print('REPORT FAIL', type(e).__name__, e, flush=True)
    print('PROBE_U_DONE', flush=True)
