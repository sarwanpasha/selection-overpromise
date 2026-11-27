"""Probe X -- a correction that knows when not to correct, and two robustness asks.

Probe T showed that isotonic recalibration helps when the calibration slice resembles
the candidates and hurts when it does not, and that the discrepancy between the two
conditional means predicts which case holds.  That diagnostic used fitness an
experimenter would not have.  This asks whether a quantity they do have works as well.

Four proxies, all computable before any candidate is measured:
  ks        Kolmogorov-Smirnov distance between calibration and candidate scores
  mshift    difference in mean score, in calibration standard deviations
  topshift  how far the selected batch sits above the top of the calibration range
  beyond    fraction of the batch scoring above every calibration point

Isotonic regression is constant outside the calibration range, so a batch that sits
beyond it is being corrected by extrapolation.  The rule shrinks the correction toward
no correction, with forecasts stored on a grid of shrinkage weights so the mapping from
proxy to weight can be fitted leave-one-assay-out afterwards.

Two further asks are answered here.  The gap is recorded in units of the assay's own
standard deviation as well as of the achievable gain, since the ceiling is small for
some assays and makes the normalized gap unstable.  And on random splits the
calibration slice is swept from a tenth to half the training data.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr, ks_2samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict
from probeH import tail_index, iso_fit
from probeR import parse_multi, featurize_multi

OUT      = os.environ.get('PROBEX_OUT', 'probeX.csv')
WORKERS  = int(os.environ.get('PROBEX_WORKERS', '8'))
ROWCAP   = int(os.environ.get('PROBEX_ROWCAP', '20000'))
KS_LIST  = (10, 20, 50)
SEEDS    = 5
CALIB    = 0.30
CAL_SWEEP = (0.10, 0.20, 0.30, 0.45)
TRFRAC   = 0.5
MODES    = ('rand', 'pos', 'aa', 'low', 'multi')
LAMBDAS  = (0.0, 0.25, 0.5, 0.75, 1.0)


def proxies(pc, pt, top):
    """Shift statistics between the calibration scores and the candidate scores."""
    pc = np.asarray(pc, float); pt = np.asarray(pt, float)
    sd = float(pc.std()) or 1e-9
    hi = float(pc.max())
    sel = pt[top]
    try:    ks = float(ks_2samp(pc, pt).statistic)
    except Exception: ks = np.nan
    return dict(ks=ks,
                mshift=float((pt.mean() - pc.mean()) / sd),
                topshift=float((sel.mean() - hi) / sd),
                beyond=float(np.mean(sel > hi)))


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
    si = np.where(nsub == 1)[0]
    mi = np.where(nsub > 1)[0]
    ysd = float(yall.std())
    tau = tail_index(yall)
    out = []

    for mode in MODES:
        if mode == 'multi':
            if len(si) < 200 or len(mi) < 100:
                continue
        elif len(si) < 300:
            continue
        sweep = CAL_SWEEP if mode == 'rand' else (CALIB,)
        for seed in range(SEEDS):
            try:
                rng = np.random.RandomState(1000 * seed + 11)
                if mode == 'multi':
                    tr, te = si.copy(), mi.copy(); rng.shuffle(tr)
                elif mode == 'rand':
                    o = si.copy(); rng.shuffle(o)
                    cut = int(round(TRFRAC * len(o)))
                    tr, te = o[:cut], o[cut:]
                elif mode == 'pos':
                    up = np.unique(pos_of[si]); rng.shuffle(up)
                    keepp = set(up[:max(1, int(round(TRFRAC * len(up))))].tolist())
                    inz = np.array([p in keepp for p in pos_of[si]])
                    tr, te = si[inz], si[~inz]
                elif mode == 'aa':
                    ua = np.unique(aa_of[si]); rng.shuffle(ua)
                    keepa = set(ua[:max(1, int(round(TRFRAC * len(ua))))].tolist())
                    inz = np.array([a in keepa for a in aa_of[si]])
                    tr, te = si[inz], si[~inz]
                else:
                    o = si[np.argsort(yall[si])]
                    cut = int(round(0.6 * len(o)))
                    tr, te = o[:cut].copy(), o[cut:].copy(); rng.shuffle(tr)
                if len(tr) < 150 or len(te) < 100:
                    continue
                for cfrac in sweep:
                    ncal = max(50, int(round(cfrac * len(tr))))
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
                    try:    g = iso_fit(pc, yc)
                    except Exception: g = lambda v: np.asarray(v, float)
                    iso_t = np.asarray(g(pt), float); iso_t[~np.isfinite(iso_t)] = 0.0
                    try:    g_or = iso_fit(pt, yte)
                    except Exception: g_or = lambda v: np.asarray(v, float)
                    orc = np.asarray(g_or(pt), float); orc[~np.isfinite(orc)] = 0.0
                    base = float(np.median(yf))
                    rho = spearmanr(pt, yte).statistic
                    for K in KS_LIST:
                        if K > len(yte):
                            continue
                        top  = np.argsort(-pt)[:K]
                        bestK = np.argsort(-yte)[:K]
                        ceiling = float(yte[bestK].mean() - base)
                        if abs(ceiling) < 1e-9:
                            continue
                        realized = float(yte[top].mean() - base)
                        row = dict(assay=name, mode=mode, seed=seed, K=K,
                                   cfrac=cfrac, tau=tau, ysd=ysd,
                                   n_tr=int(len(tr)), n_te=int(len(te)),
                                   ceiling=ceiling, realized=realized,
                                   spearman=float(rho) if np.isfinite(rho) else None,
                                   drift=float(np.mean(iso_t[top] - orc[top]) / max(ysd, 1e-12)),
                                   fc_oracle=float(orc[top].mean() - base))
                        row.update(proxies(pc, pt, top))
                        for lb in LAMBDAS:
                            mix = lb * iso_t + (1.0 - lb) * pt
                            row['fc_%03d' % int(lb * 100)] = float(mix[top].mean() - base)
                        out.append(row)
            except Exception as e:
                print('  fit-fail', name, mode, seed, type(e).__name__, e, flush=True)
                continue
    return out


def fit_lambda(cell, proxy):
    """Leave-one-assay-out: pick the shrinkage weight from the proxy, then score it."""
    lams = np.array(LAMBDAS)
    cols = ['fc_%03d' % int(l * 100) for l in lams]
    assays = cell.assay.unique()
    err, errs_iso, errs_raw = [], [], []
    for a in assays:
        tr = cell[cell.assay != a]; te = cell[cell.assay == a]
        if len(tr) < 50 or len(te) == 0:
            continue
        # on the training assays, find the best weight within each proxy tercile
        qs = np.quantile(tr[proxy].dropna(), [1/3, 2/3]) if tr[proxy].notna().any() else [0, 0]
        best = {}
        for b in range(3):
            lo = -np.inf if b == 0 else qs[b - 1]
            hi = np.inf if b == 2 else qs[b]
            sl = tr[(tr[proxy] > lo) & (tr[proxy] <= hi)]
            if len(sl) < 10:
                best[b] = 1.0; continue
            sc = [np.mean(np.abs((sl[c] - sl.realized) / sl.ceiling)) for c in cols]
            best[b] = float(lams[int(np.argmin(sc))])
        for _, r in te.iterrows():
            v = r[proxy]
            b = 0 if not np.isfinite(v) or v <= qs[0] else (1 if v <= qs[1] else 2)
            lb = best[b]
            c = 'fc_%03d' % int(lb * 100)
            err.append(abs((r[c] - r.realized) / r.ceiling))
            errs_iso.append(abs((r['fc_100'] - r.realized) / r.ceiling))
            errs_raw.append(abs((r['fc_000'] - r.realized) / r.ceiling))
    return (float(np.median(err)), float(np.median(errs_iso)),
            float(np.median(errs_raw)), len(err))


def report(df):
    d = df[df.ceiling.abs() > 1e-9].copy()
    print('ORDER lambdas: ' + ' '.join(str(l) for l in LAMBDAS), flush=True)
    for mode, m in d.groupby('mode'):
        base = m[m.cfrac == CALIB] if mode == 'rand' else m
        cell = base.groupby(['assay', 'K'], as_index=False).median(numeric_only=True)
        a = cell.groupby('assay', as_index=False).median(numeric_only=True)
        g = lambda v: int(round(1000 * float(np.median(v))))
        print('=== X %s assays %d ===' % (mode, len(a)), flush=True)
        print('Xabs_%s %s' % (mode, ' '.join(
            str(g(((a['fc_%03d' % int(l * 100)] - a.realized) / a.ceiling).abs()))
            for l in LAMBDAS)), flush=True)
        print('Xsd_%s %s' % (mode, ' '.join(
            str(g(((a['fc_%03d' % int(l * 100)] - a.realized) / a.ysd).abs()))
            for l in LAMBDAS)), flush=True)
        for p in ('ks', 'mshift', 'topshift', 'beyond'):
            mm = np.isfinite(cell[p]) & np.isfinite(cell.drift)
            if mm.sum() > 20:
                r = spearmanr(cell[p][mm], cell.drift[mm].abs())
                print('Xproxy_%s_%s %d' % (mode, p, round(1000 * r.statistic)), flush=True)
    # the adaptive rule, pooled over every split
    cell = d[(d.cfrac == CALIB)].groupby(['assay', 'mode', 'K'], as_index=False).median(numeric_only=True)
    for p in ('ks', 'mshift', 'topshift', 'beyond'):
        try:
            e, ei, er, k = fit_lambda(cell, p)
            print('XADAPT_%s %d iso %d raw %d n %d' % (p, round(1000 * e),
                  round(1000 * ei), round(1000 * er), k), flush=True)
        except Exception as ex:
            print('XADAPT_%s FAIL %s' % (p, ex), flush=True)
    # calibration-slice sweep on random splits
    r = d[d['mode'] == 'rand']
    for cf in CAL_SWEEP:
        sl = r[r.cfrac == cf]
        if not len(sl):
            continue
        cl = sl.groupby(['assay', 'K'], as_index=False).median(numeric_only=True)
        a = cl.groupby('assay', as_index=False).median(numeric_only=True)
        print('XSWEEP_%s %d %d' % (cf,
              int(round(1000 * float(((a.fc_100 - a.realized) / a.ceiling).abs().median()))),
              len(a)), flush=True)


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
    print('=== PROBE X ===', flush=True)
    try:
        report(df)
    except Exception as e:
        print('REPORT FAIL', type(e).__name__, e, flush=True)
    print('PROBE_X_DONE', flush=True)
