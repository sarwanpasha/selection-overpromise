"""Probe R -- does conditional-mean recalibration survive covariate shift?

Random splits make the held-out candidates exchangeable with the calibration slice,
which is exactly the regime where isotonic recalibration is guaranteed to work.  Real
campaigns extrapolate.  This replaces the random split with four shifted designs and
asks whether the fix still removes the gap.

  pos    : positions are split, so every test variant mutates a position with no
           training measurement at all
  aa     : mutant amino acid identities are split, so test substitutions introduce
           residues never seen as a target in training
  low    : training and calibration are drawn from the lower part of the fitness
           range, so the model extrapolates upward, as an early round does
  multi  : training and calibration are single substitutions, test candidates carry
           two or more, which is the combinatorial design case

Features are additive by construction, so a multiple substitution is represented by
the sum of its single-substitution feature vectors and the model transfers without
modification.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict
from probeH import tail_index, iso_fit
from probeQ import lin_fit, cviso_fit, tailiso_fit

OUT     = os.environ.get('PROBER_OUT', 'probeR.csv')
WORKERS = int(os.environ.get('PROBER_WORKERS', '24'))
ROWCAP  = int(os.environ.get('PROBER_ROWCAP', '20000'))
KS      = (10, 20, 50)
SEEDS   = 5
CALIB   = 0.30
TRFRAC  = 0.5
MODES   = ('pos', 'aa', 'low', 'multi')
RULES   = ('raw', 'lin', 'iso', 'cviso', 'tailiso')


def parse_multi(m):
    """'A1B' or 'A1B:C2D' -> list of (wt, pos, mut); None if any part fails."""
    parts = [p for p in m.replace(',', ':').replace('/', ':').split(':') if p]
    if not parts:
        return None
    out = []
    for p in parts:
        q = parse(p)
        if q is None:
            return None
        out.append(q)
    return out


def featurize_multi(groups, positions):
    """Sum of single-substitution feature rows, one row per variant."""
    flat, owner = [], []
    for i, g in enumerate(groups):
        for sub in g:
            flat.append(sub); owner.append(i)
    Xs = featurize(flat, positions)
    X = np.zeros((len(groups), Xs.shape[1]), dtype=np.float32)
    np.add.at(X, np.asarray(owner), Xs)
    return X


def _forecasts(pc, yc, pt, seed):
    g_lin, slope = lin_fit(pc, yc)
    try:    g_iso = iso_fit(pc, yc)
    except Exception: g_iso = lambda v: np.asarray(v, float)
    g_cvi = cviso_fit(pc, yc, seed=seed)
    try:    g_tai = tailiso_fit(pc, yc)
    except Exception: g_tai = g_iso
    cal = {'raw': np.asarray(pt, float)}
    for key, g in (('lin', g_lin), ('iso', g_iso), ('cviso', g_cvi), ('tailiso', g_tai)):
        a = np.asarray(g(pt), float)
        a[~np.isfinite(a)] = 0.0
        cal[key] = a
    return cal, slope


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

    groups, keep = [], []
    for i, mm in enumerate(df[mc].astype(str).values):
        g = parse_multi(mm)
        if g is not None:
            groups.append(g); keep.append(i)
    if len(groups) < 200:
        return []
    yall = pd.to_numeric(df[sc].values[keep], errors='coerce').astype(float)
    nsub = np.array([len(g) for g in groups])
    positions = sorted({s[1] for g in groups for s in g})
    Xall = featurize_multi(groups, positions)
    ok = np.isfinite(yall) & np.isfinite(Xall).all(1)
    Xall, yall, nsub = Xall[ok], yall[ok], nsub[ok]
    groups = [g for g, k in zip(groups, ok) if k]
    if len(yall) < 200 or yall.std() < 1e-9:
        return []

    pos_of = np.array([g[0][1] for g in groups])
    aa_of  = np.array([g[0][2] for g in groups])
    out = []

    for mode in MODES:
        if mode == 'multi':
            si = np.where(nsub == 1)[0]
            mi = np.where(nsub > 1)[0]
            if len(si) < 200 or len(mi) < 100:
                continue
        else:
            si = np.where(nsub == 1)[0]
            if len(si) < 300:
                continue
        for seed in range(SEEDS):
            try:
                rng = np.random.RandomState(1000 * seed + 11)
                if mode == 'multi':
                    tr, te = si.copy(), mi.copy()
                    rng.shuffle(tr)
                elif mode == 'pos':
                    up = np.unique(pos_of[si]); rng.shuffle(up)
                    cut = max(1, int(round(TRFRAC * len(up))))
                    trp = set(up[:cut].tolist())
                    inz = np.array([p in trp for p in pos_of[si]])
                    tr, te = si[inz], si[~inz]
                elif mode == 'aa':
                    ua = np.unique(aa_of[si]); rng.shuffle(ua)
                    cut = max(1, int(round(TRFRAC * len(ua))))
                    tra = set(ua[:cut].tolist())
                    inz = np.array([a in tra for a in aa_of[si]])
                    tr, te = si[inz], si[~inz]
                else:  # low: train on the lower fitness range
                    o = si[np.argsort(yall[si])]
                    cut = int(round(0.6 * len(o)))
                    tr, te = o[:cut].copy(), o[cut:].copy()
                    rng.shuffle(tr)
                if len(tr) < 150 or len(te) < 100:
                    continue

                ncal = max(50, int(round(CALIB * len(tr))))
                if len(tr) - ncal < 80:
                    continue
                fit_i, cal_i = tr[:-ncal], tr[-ncal:]
                Xf, yf = Xall[fit_i], yall[fit_i]
                Xc, yc = Xall[cal_i], yall[cal_i]
                Xte, yte = Xall[te], yall[te]
                if yf.std() < 1e-9 or yc.std() < 1e-9:
                    continue

                w, ym, mu, lam, A_inv, s2 = ridge_fit(Xf, yf)
                pc, _ = predict(Xc, w, ym, mu, A_inv, s2)
                pt, _ = predict(Xte, w, ym, mu, A_inv, s2)
                if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
                    continue

                cal, slope = _forecasts(pc, yc, pt, seed)
                base = float(np.median(yf))
                rho = spearmanr(pt, yte).statistic
                tau_tr = tail_index(yall[tr])
                for K in KS:
                    if K > len(yte):
                        continue
                    top  = np.argsort(-pt)[:K]
                    best = np.argsort(-yte)[:K]
                    ceiling  = float(yte[best].mean() - base)
                    realized = float(yte[top].mean() - base)
                    row = dict(assay=name, mode=mode, seed=seed, K=K,
                               n_tr=int(len(tr)), n_te=int(len(te)),
                               tau_train=tau_tr, base=base, ceiling=ceiling,
                               realized=realized, lin_slope=slope,
                               spearman=float(rho) if np.isfinite(rho) else None)
                    for key in RULES:
                        row['fc_' + key] = float(cal[key][top].mean() - base)
                    out.append(row)
            except Exception as e:
                print('  fit-fail', name, mode, seed, type(e).__name__, e, flush=True)
                continue
    return out


def report(df):
    from scipy.stats import wilcoxon
    d = df[df.ceiling > 1e-9].copy()
    lines = [('TOTAL_ROWS', len(df))]
    for mode in MODES:
        m = d[d['mode'] == mode]
        if not len(m):
            lines.append(('MODE_%s' % mode, 'no_data')); continue
        cell = m.groupby(['assay', 'K'], as_index=False).median(numeric_only=True)
        assay = cell.groupby('assay', as_index=False).median(numeric_only=True)
        lines.append(('%s_ASSAYS' % mode, m.assay.nunique()))
        lines.append(('%s_CELLS' % mode, len(cell)))
        lines.append(('%s_rho_median' % mode, round(float(cell.spearman.median()), 3)))
        s = {}
        for r in RULES:
            s[r] = (assay['fc_' + r] - assay['realized']) / assay['ceiling']
            lines.append(('%s_%s_median_s' % (mode, r), round(float(s[r].median()), 4)))
            lines.append(('%s_%s_meanabs_s' % (mode, r), round(float(s[r].abs().mean()), 4)))
            lines.append(('%s_%s_meanabs_raw' % (mode, r),
                          round(float((assay['fc_' + r] - assay['realized']).abs().mean()), 4)))
        for a, b in (('iso', 'raw'), ('lin', 'raw'), ('iso', 'lin'), ('cviso', 'iso')):
            da, db = s[a].abs(), s[b].abs()
            try:    pv = wilcoxon(da, db).pvalue
            except Exception: pv = float('nan')
            lines.append(('%s_%s_vs_%s_winpct' % (mode, a, b),
                          round(100 * float((da < db).mean()), 1)))
            lines.append(('%s_%s_vs_%s_log10p' % (mode, a, b),
                          round(float(np.log10(pv)), 2) if np.isfinite(pv) and pv > 0 else 'NA'))
    txt = '\n'.join('%s = %s' % (k, v) for k, v in lines)
    open('probeR_report.txt', 'w').write(txt)
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
    print('=== PROBE R ===', flush=True)
    report(df)
    print('PROBE_R_DONE', flush=True)
