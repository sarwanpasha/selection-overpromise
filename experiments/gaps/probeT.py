"""Probe T -- when is a calibration slice informative about the batch?

Probe R found that recalibration helps under some shifts and hurts under others.  The
theory says why it should: \\cref{prop:tower} requires the conditional mean under the
distribution the batch is drawn from, and a calibration slice estimates the conditional
mean under the distribution the training data came from.  The two agree when the shift
leaves the score-conditional law of fitness alone, and not otherwise.

This measures that directly.  For every split we fit the conditional mean twice, once
on the calibration slice and once on the held-out candidates using their measured
fitness, which a practitioner cannot do but an audit can.  The discrepancy between the
two on the selected batch is the calibration drift.  The prediction is that drift, not
the size of the shift, decides whether recalibration helps.

It also adds the rule a practitioner would actually use: choose among no correction,
linear recalibration and isotonic recalibration by cross-validation inside the
calibration slice.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict
from probeH import tail_index, iso_fit
from probeQ import lin_fit, cviso_fit
from probeR import parse_multi, featurize_multi

OUT     = os.environ.get('PROBET_OUT', 'probeT.csv')
WORKERS = int(os.environ.get('PROBET_WORKERS', '20'))
ROWCAP  = int(os.environ.get('PROBET_ROWCAP', '20000'))
KS      = (10, 20, 50)
SEEDS   = 5
CALIB   = 0.30
TRFRAC  = 0.5
MODES   = ('rand', 'pos', 'aa', 'low', 'multi')
RULES   = ('raw', 'lin', 'iso', 'auto')


def choose_rule(pc, yc, seed):
    """Pick among no correction, linear and isotonic by 5-fold CV on the slice."""
    pc = np.asarray(pc, float); yc = np.asarray(yc, float)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pc))
    parts = np.array_split(idx, 5)
    err = {'raw': 0.0, 'lin': 0.0, 'iso': 0.0}
    for f in range(5):
        te = parts[f]
        tr = np.concatenate([parts[g] for g in range(5) if g != f])
        if len(tr) < 20 or len(te) < 3:
            continue
        err['raw'] += float(((pc[te] - yc[te]) ** 2).sum())
        try:
            gl, _ = lin_fit(pc[tr], yc[tr])
            err['lin'] += float(((gl(pc[te]) - yc[te]) ** 2).sum())
        except Exception:
            err['lin'] += np.inf
        try:
            gi = iso_fit(pc[tr], yc[tr])
            err['iso'] += float(((np.asarray(gi(pc[te]), float) - yc[te]) ** 2).sum())
        except Exception:
            err['iso'] += np.inf
    return min(err, key=err.get)


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
    out = []

    for mode in MODES:
        if mode == 'multi':
            if len(si) < 200 or len(mi) < 100:
                continue
        elif len(si) < 300:
            continue
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
                    trp = set(up[:max(1, int(round(TRFRAC * len(up))))].tolist())
                    inz = np.array([p in trp for p in pos_of[si]])
                    tr, te = si[inz], si[~inz]
                elif mode == 'aa':
                    ua = np.unique(aa_of[si]); rng.shuffle(ua)
                    tra = set(ua[:max(1, int(round(TRFRAC * len(ua))))].tolist())
                    inz = np.array([a in tra for a in aa_of[si]])
                    tr, te = si[inz], si[~inz]
                else:
                    o = si[np.argsort(yall[si])]
                    cut = int(round(0.6 * len(o)))
                    tr, te = o[:cut].copy(), o[cut:].copy(); rng.shuffle(tr)
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

                g_lin, slope = lin_fit(pc, yc)
                try:    g_iso = iso_fit(pc, yc)
                except Exception: g_iso = lambda v: np.asarray(v, float)
                pick = choose_rule(pc, yc, seed)
                g_auto = {'raw': (lambda v: np.asarray(v, float)),
                          'lin': g_lin, 'iso': g_iso}[pick]
                # the oracle conditional mean on the candidates themselves
                try:    g_oracle = iso_fit(pt, yte)
                except Exception: g_oracle = lambda v: np.asarray(v, float)

                cal = {'raw': np.asarray(pt, float)}
                for k, g in (('lin', g_lin), ('iso', g_iso), ('auto', g_auto)):
                    a = np.asarray(g(pt), float); a[~np.isfinite(a)] = 0.0
                    cal[k] = a
                oracle = np.asarray(g_oracle(pt), float); oracle[~np.isfinite(oracle)] = 0.0

                base = float(np.median(yf))
                rho = spearmanr(pt, yte).statistic
                ysd = float(yall.std())
                for K in KS:
                    if K > len(yte):
                        continue
                    top  = np.argsort(-pt)[:K]
                    bestK = np.argsort(-yte)[:K]
                    ceiling = float(yte[bestK].mean() - base)
                    # calibration drift on the selected batch, in units of assay sd
                    drift = float(np.mean(cal['iso'][top] - oracle[top]) / max(ysd, 1e-12))
                    row = dict(assay=name, mode=mode, seed=seed, K=K, pick=pick,
                               n_tr=int(len(tr)), n_te=int(len(te)),
                               tau_train=tail_index(yall[tr]), base=base,
                               ceiling=ceiling, realized=float(yte[top].mean() - base),
                               drift=drift, ysd=ysd, lin_slope=slope,
                               spearman=float(rho) if np.isfinite(rho) else None)
                    for k in RULES:
                        row['fc_' + k] = float(cal[k][top].mean() - base)
                    row['fc_oracle'] = float(oracle[top].mean() - base)
                    out.append(row)
            except Exception as e:
                print('  fit-fail', name, mode, seed, type(e).__name__, e, flush=True)
                continue
    return out


def report(df):
    from scipy.stats import wilcoxon
    d = df[df.ceiling > 1e-9].copy()
    print('ORDER rules: raw lin iso auto oracle', flush=True)
    for mode, m in d.groupby('mode'):
        cell = m.groupby(['assay', 'K'], as_index=False).median(numeric_only=True)
        a = cell.groupby('assay', as_index=False).median(numeric_only=True)
        s = {k: (a['fc_' + k] - a['realized']) / a['ceiling']
             for k in list(RULES) + ['oracle']}
        g = lambda v: int(round(float(np.median(v)) * 1000))
        print('=== T %s assays %d rho %d drift %d ===' %
              (mode, len(a), g(a.spearman), g(a.drift)), flush=True)
        print('Tmed_%s %s' % (mode, ' '.join(str(g(s[k])) for k in list(RULES) + ['oracle'])), flush=True)
        print('Tabs_%s %s' % (mode, ' '.join(str(g(s[k].abs())) for k in list(RULES) + ['oracle'])), flush=True)
        for x, y in (('iso','raw'), ('auto','raw'), ('auto','iso'), ('auto','lin')):
            da, db = s[x].abs(), s[y].abs()
            try: p = wilcoxon(da, db).pvalue
            except Exception: p = float('nan')
            print('Tcmp_%s_%s_%s %d %d' % (mode, x, y,
                  round(100 * float((da < db).mean())),
                  round(np.log10(p)) if np.isfinite(p) and p > 0 else -999), flush=True)
        pk = cell.pick.value_counts(normalize=True)
        print('Tpick_%s raw %d lin %d iso %d' % (mode,
              round(100 * pk.get('raw', 0.0)), round(100 * pk.get('lin', 0.0)),
              round(100 * pk.get('iso', 0.0))), flush=True)
    # does drift explain where recalibration helps?
    cell = d.groupby(['assay', 'mode', 'K'], as_index=False).median(numeric_only=True)
    gain = ((cell.fc_raw - cell.realized).abs() -
            (cell.fc_iso - cell.realized).abs()) / cell.ceiling
    ok = np.isfinite(gain) & np.isfinite(cell.drift)
    r = spearmanr(cell.drift.abs()[ok], gain[ok])
    print('TDRIFT_vs_gain %d %d' % (round(1000 * r.statistic),
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
    print('=== PROBE T ===', flush=True)
    report(df)
    print('PROBE_T_DONE', flush=True)
