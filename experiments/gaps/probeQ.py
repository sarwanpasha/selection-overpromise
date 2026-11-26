"""Probe Q -- recalibration baselines that the obvious critique demands.

probeH compared isotonic recalibration against the uncorrected score, a binned
conditional mean and Efron's Tweedie correction. Three objections follow:

  1. Linear recalibration of y on yhat on the same held-out slice is the simplest
     monotone fix and is missing.  Simulation suggests it beats isotonic at low
     accuracy, which is where most real cells sit.
  2. Self-calibrated (cross-fitted) isotonic and tail-weighted isotonic are the
     natural competitors to plain isotonic.
  3. Cells sharing an assay are not independent, so cell-level p-values overstate
     significance.  Everything here is also reported at assay level.

It also answers two further objections:

  * tau in probeH is computed on the whole assay, which leaks held-out fitness into
    a statistic we call prospective.  tau_train recomputes it from the training
    measurements only.
  * the ceiling-normalised gap s is unstable when the ceiling is small, so the raw
    gap in fitness units is carried alongside it.

Finally it estimates the finite-sample term behind the residual gap: sigma_n, the
standard deviation of the fitted score around its infinite-data limit, estimated by
splitting the fit set in half, together with the Gaussian bound
    E[eps | any set of size qn] <= sigma_n * phi(z_{1-q}) / q .
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict
from probeH import tail_index, iso_fit

OUT     = os.environ.get('PROBEQ_OUT', 'probeQ.csv')
WORKERS = int(os.environ.get('PROBEQ_WORKERS', '24'))
ROWCAP  = int(os.environ.get('PROBEQ_ROWCAP', '20000'))
FRACS   = (0.1, 0.3, 0.6)
KS      = (10, 20, 50)
SEEDS   = 5
CALIB   = 0.30
RULES   = ('raw', 'lin', 'iso', 'cviso', 'tailiso')


def lin_fit(x, y):
    """Least-squares recalibration y ~ a*yhat + b.  Monotone when a > 0."""
    A = np.vstack([np.asarray(x, float), np.ones(len(x))]).T
    coef, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return (lambda v: a * np.asarray(v, float) + b), a


def cviso_fit(x, y, folds=5, seed=0):
    """Self-calibrated isotonic: average of out-of-fold isotonic fits.

    An average of monotone functions is monotone, so the selected batch is
    unchanged, exactly as for plain isotonic.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x))
    parts = np.array_split(idx, folds)
    fits = []
    for f in range(folds):
        tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        if len(tr) >= 10:
            try: fits.append(iso_fit(x[tr], y[tr]))
            except Exception: pass
    if not fits:
        return lambda v: np.asarray(v, float)
    return lambda v: np.mean([np.asarray(g(v), float) for g in fits], axis=0)


def tailiso_fit(x, y, alpha=2.0):
    """Isotonic regression with weights rising exponentially in the score rank."""
    from sklearn.isotonic import IsotonicRegression
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = np.argsort(np.argsort(x)) / max(len(x) - 1, 1)
    w = np.exp(alpha * r)
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(x, y, sample_weight=w)
    lo, hi = x.min(), x.max()
    return lambda v: ir.predict(np.clip(np.asarray(v, float), lo, hi))


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

    tau_full = tail_index(y)
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

                # tau from training measurements only, no held-out fitness
                tau_tr = tail_index(y[tr])

                w, ym, mu, lam, A_inv, s2 = ridge_fit(Xf, yf)
                pc, _ = predict(Xc, w, ym, mu, A_inv, s2)
                pt, _ = predict(Xte, w, ym, mu, A_inv, s2)
                if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
                    continue

                # finite-sample scale of the fitted score: split the fit set in half
                sigma_n = float('nan')
                try:
                    h = len(fit_i) // 2
                    if h >= 60:
                        wa = ridge_fit(X[fit_i[:h]], y[fit_i[:h]])
                        wb = ridge_fit(X[fit_i[h:2 * h]], y[fit_i[h:2 * h]])
                        pa, _ = predict(Xte, wa[0], wa[1], wa[2], wa[4], wa[5])
                        pb, _ = predict(Xte, wb[0], wb[1], wb[2], wb[4], wb[5])
                        if np.isfinite(pa).all() and np.isfinite(pb).all():
                            var_half = float(np.var(pa - pb)) / 2.0
                            sigma_n = float(np.sqrt(max(var_half, 0.0) / 2.0))
                except Exception:
                    pass

                base = float(np.median(yf))
                g_lin, slope = lin_fit(pc, yc)
                try:    g_iso = iso_fit(pc, yc)
                except Exception: g_iso = lambda v: np.asarray(v, float)
                g_cvi = cviso_fit(pc, yc, seed=seed)
                try:    g_tai = tailiso_fit(pc, yc)
                except Exception: g_tai = g_iso

                cal = {'raw': np.asarray(pt, float)}
                for key, g in (('lin', g_lin), ('iso', g_iso),
                               ('cviso', g_cvi), ('tailiso', g_tai)):
                    a = np.asarray(g(pt), float)
                    a[~np.isfinite(a)] = 0.0
                    cal[key] = a

                rho = spearmanr(pt, yte).statistic
                for K in KS:
                    if K > len(yte):
                        continue
                    top  = np.argsort(-pt)[:K]
                    best = np.argsort(-yte)[:K]
                    ceiling  = float(yte[best].mean() - base)
                    realized = float(yte[top].mean() - base)
                    q = K / float(len(yte))
                    bound = (sigma_n * norm.pdf(norm.ppf(1 - q)) / q
                             if np.isfinite(sigma_n) else float('nan'))
                    row = dict(assay=name, frac=tf, seed=seed, K=K,
                               tau=tau_full, tau_train=tau_tr, n=int(n),
                               base=base, ceiling=ceiling, realized=realized,
                               ysd=float(y.std()), lin_slope=slope,
                               sigma_n=sigma_n, fs_bound=bound, q=q,
                               spearman=float(rho) if np.isfinite(rho) else None)
                    for key in RULES:
                        row['fc_' + key] = float(cal[key][top].mean() - base)
                        # monotone rules leave the batch alone; check it
                        row['same_' + key] = int(
                            set(np.argsort(-cal[key])[:K].tolist()) == set(top.tolist()))
                    out.append(row)
            except Exception as e:
                print('  fit-fail', name, tf, seed, type(e).__name__, e, flush=True)
                continue
    return out


def _stats(g, label, lines):
    """Cell-level and assay-level summaries of every rule, plus pairwise tests."""
    from scipy.stats import wilcoxon
    s, raw = {}, {}
    for r in RULES:
        s[r]   = (g['fc_' + r] - g['realized']) / g['ceiling']
        raw[r] = (g['fc_' + r] - g['realized'])
        lines.append(('%s_%s_median_s' % (label, r), round(float(s[r].median()), 4)))
        lines.append(('%s_%s_meanabs_s' % (label, r), round(float(s[r].abs().mean()), 4)))
        lines.append(('%s_%s_medabs_s' % (label, r), round(float(s[r].abs().median()), 4)))
        lines.append(('%s_%s_median_raw' % (label, r), round(float(raw[r].median()), 4)))
        lines.append(('%s_%s_meanabs_raw' % (label, r), round(float(raw[r].abs().mean()), 4)))
        lines.append(('%s_%s_over_pct' % (label, r), round(100 * float((s[r] > 0).mean()), 1)))
        if r != 'raw':
            lines.append(('%s_%s_batch_identical_pct' % (label, r),
                          round(100 * float(g['same_' + r].mean()), 1)))
    for a, b in (('iso', 'raw'), ('lin', 'raw'), ('iso', 'lin'),
                 ('cviso', 'iso'), ('tailiso', 'iso'), ('cviso', 'lin')):
        da, db = s[a].abs(), s[b].abs()
        win = 100 * float((da < db).mean())
        try:    pv = wilcoxon(da, db).pvalue
        except Exception: pv = float('nan')
        lines.append(('%s_%s_vs_%s_winpct' % (label, a, b), round(win, 1)))
        lines.append(('%s_%s_vs_%s_log10p' % (label, a, b),
                      round(float(np.log10(pv)), 2) if np.isfinite(pv) and pv > 0 else 'NA'))


def report(df):
    d = df[df.ceiling > 1e-9].copy()
    lines = [('TOTAL_ROWS', len(df)), ('ASSAYS', d.assay.nunique())]
    cell = d.groupby(['assay', 'frac', 'K'], as_index=False).median(numeric_only=True)
    lines.append(('CELLS', len(cell)))
    _stats(cell, 'CELL', lines)
    assay = cell.groupby('assay', as_index=False).median(numeric_only=True)
    lines.append(('ASSAY_UNITS', len(assay)))
    _stats(assay, 'ASSAY', lines)
    # tau computed on training data only: does it still track the gap?
    cell['s_raw'] = (cell.fc_raw - cell.realized) / cell.ceiling
    for col in ('tau', 'tau_train'):
        m = np.isfinite(cell[col]) & np.isfinite(cell.s_raw)
        r = spearmanr(cell[col][m], cell.s_raw[m])
        lines.append(('TAUCORR_%s_spearman' % col, round(float(r.statistic), 4)))
        lines.append(('TAUCORR_%s_log10p' % col, round(float(np.log10(r.pvalue)), 2)
                      if r.pvalue > 0 else 'NA'))
    m = np.isfinite(cell.tau) & np.isfinite(cell.tau_train)
    lines.append(('TAU_vs_TAUTRAIN_pearson',
                  round(float(np.corrcoef(cell.tau[m], cell.tau_train[m])[0, 1]), 4)))
    # finite-sample term
    for col in ('sigma_n', 'fs_bound'):
        v = cell[col][np.isfinite(cell[col])]
        if len(v):
            lines.append(('FS_%s_median' % col, round(float(v.median()), 4)))
    sub = cell[np.isfinite(cell.fs_bound) & (cell.ceiling > 1e-9)]
    if len(sub):
        lines.append(('FS_bound_over_ceiling_median',
                      round(float((sub.fs_bound / sub.ceiling).median()), 4)))
        lines.append(('FS_bound_exceeds_gap_pct',
                      round(100 * float(((sub.fs_bound / sub.ceiling) >
                                         ((sub.fc_raw - sub.realized) / sub.ceiling)).mean()), 1)))
    for tf in FRACS:
        sl = cell[cell.frac == tf]
        if len(sl):
            lines.append(('BYFRAC_%s_raw_median_s' % tf,
                          round(float(((sl.fc_raw - sl.realized) / sl.ceiling).median()), 4)))
            lines.append(('BYFRAC_%s_sigma_n_median' % tf,
                          round(float(sl.sigma_n.median()), 4)))
    lines.append(('LIN_SLOPE_positive_pct', round(100 * float((cell.lin_slope > 0).mean()), 1)))
    lines.append(('LIN_SLOPE_median', round(float(cell.lin_slope.median()), 4)))
    txt = '\n'.join('%s = %s' % (k, v) for k, v in lines)
    open('probeQ_report.txt', 'w').write(txt)
    print(txt, flush=True)


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
            if i % 10 == 0 or i == len(todo):
                print('  %d/%d done, %d rows' % (i, len(todo), nrows), flush=True)
    df = pd.DataFrame([json.loads(l) for l in open(JL)])
    df.to_csv(OUT, index=False)
    print('=== PROBE Q ===', flush=True)
    report(df)
    print('PROBE_Q_DONE', flush=True)
