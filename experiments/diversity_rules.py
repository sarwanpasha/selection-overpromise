"""
Probe G -- can DIVERSITY-AWARE batch selection beat top-K-by-prediction?

Motivation (from the Probe E failure + the theorem):
post-hoc recalibration is a monotone map of yhat, so it cannot change WHICH
variants are selected.  To beat the baseline on realized gain we must use
structure the ranking ignores.  Ridge errors are CORRELATED BY POSITION: if the
model misjudges a site, it misjudges all substitutions there together.  Top-K by
prediction therefore concentrates the batch on the single most-overestimated
site.  Capping picks per position hedges that risk.

Rules (all use only training-visible information):
  raw   : top-K by prediction                                [BASELINE / SOTA]
  cap1  : greedy by prediction, at most 1 variant per position
  cap2  : at most 2 per position
  cap3  : at most 3 per position
  half  : at most max(1, K//10) per position

Metrics: realized MEAN gain of the batch, and realized MAX gain (best single
variant found) -- the latter is what a design campaign usually cares about.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recalibration_rules import parse, featurize, ridge_fit, predict   # verified module

OUTCSV  = os.environ.get('PROBEG_OUT', 'probeG.csv')
WORKERS = int(os.environ.get('PROBEG_WORKERS', '8'))
ROWCAP  = 20000
FRACS   = (0.1, 0.3, 0.6)
KS      = (10, 20, 50)
SEEDS   = 5


def pick_capped(pred, pos, K, cap):
    """Greedy by prediction with at most `cap` picks per position."""
    order = np.argsort(-pred)
    used, sel = {}, []
    for i in order:
        p = pos[i]
        if used.get(p, 0) >= cap:
            continue
        used[p] = used.get(p, 0) + 1
        sel.append(i)
        if len(sel) == K:
            break
    if len(sel) < K:                      # not enough positions: top up by pred
        for i in order:
            if i not in sel:
                sel.append(i)
            if len(sel) == K:
                break
    return np.array(sel[:K])


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
    for i, m in enumerate(df[mc].astype(str).values):
        if '/' in m or ':' in m or ',' in m:
            continue
        p = parse(m)
        if p is not None:
            parsed.append(p); keep.append(i)
    if len(parsed) < 200:
        return []

    y = pd.to_numeric(df[sc].values[keep], errors='coerce').astype(float)
    positions = sorted({p[1] for p in parsed})
    X = featurize(parsed, positions)
    posarr = np.array([p[1] for p in parsed])

    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y, posarr = X[ok], y[ok], posarr[ok]
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
                Xtr, ytr = X[tr], y[tr]
                Xte, yte, poste = X[te], y[te], posarr[te]
                if ytr.std() < 1e-9:
                    continue
                w, ym, mu, lam, A_inv, s2 = ridge_fit(Xtr, ytr)
                pt, vt = predict(Xte, w, ym, mu, A_inv, s2)
                if not np.isfinite(pt).all():
                    continue
                base = float(np.median(ytr))
                rho = spearmanr(pt, yte).statistic
                for K in KS:
                    if K > len(yte):
                        continue
                    best = np.argsort(-yte)[:K]
                    ceil_mean = float(yte[best].mean() - base)
                    ceil_max = float(yte.max() - base)
                    rules = {'raw': np.argsort(-pt)[:K]}
                    for cap, tag in ((1,'cap1'), (2,'cap2'), (3,'cap3'),
                                     (max(1, K//10), 'half')):
                        rules[tag] = pick_capped(pt, poste, K, cap)
                    for tag, sel in rules.items():
                        out.append(dict(
                            assay=name, frac=tf, seed=seed, K=K, rule=tag,
                            realized=float(yte[sel].mean() - base),
                            realized_max=float(yte[sel].max() - base),
                            ceiling=ceil_mean, ceiling_max=ceil_max,
                            npos=int(len(set(poste[sel].tolist()))),
                            spearman=float(rho) if np.isfinite(rho) else None))
            except Exception as e:
                print('  fit-fail', name, tf, seed, type(e).__name__, e, flush=True)
                continue
    return out


def report(df):
    from scipy.stats import wilcoxon
    d = df[df.ceiling > 1e-9].copy()
    d['rn'] = d.realized / d.ceiling
    d['rx'] = d.realized_max / d.ceiling_max.where(d.ceiling_max.abs() > 1e-9)
    for metric, label in (('rn', 'MEAN gain / achievable'),
                          ('rx', 'MAX  gain / achievable-max')):
        g = d.groupby(['assay','frac','K','rule'], as_index=False)[metric].mean()
        p = g.pivot_table(index=['assay','frac','K'], columns='rule',
                          values=metric).dropna()
        if len(p) == 0 or 'raw' not in p.columns:
            continue
        print(f'\n--- {label} --- n pairs={len(p)}  raw={p["raw"].mean():.4f}')
        for r in ('cap1','cap2','cap3','half'):
            if r not in p.columns: continue
            dd = p[r] - p['raw']
            try: pv = wilcoxon(p[r], p['raw']).pvalue
            except Exception: pv = float('nan')
            print(f'  {r:5s} median {dd.median():+.5f} mean {dd.mean():+.5f} '
                  f'wins {100*(dd>0).mean():5.1f}%  p={pv:.3g}')
    print('\n--- batch concentration (distinct positions in the batch) ---')
    print(d.groupby(['rule','K']).npos.median().unstack().round(1))


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
    print(f'=== PROBE G === {len(df)} rows, {df.assay.nunique()} assays', flush=True)
    report(df)
    print('PROBE_G_DONE', flush=True)
