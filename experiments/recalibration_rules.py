"""
Probe E -- does a calibrated selection rule beat top-K-by-prediction?

Baseline (universal current practice): pick the K variants with the highest
predicted score.

Candidate rules, all using ONLY training-visible information:
  raw   : top-K by point prediction                      [BASELINE / SOTA]
  cal   : top-K by a calibration model g(yhat, leverage) fitted on a held-out
          slice of the TRAINING data
  lcb   : top-K by yhat - c*sd, c tuned on the calibration slice
  shrink: top-K by yhat shrunk toward the training median in proportion to
          predictive variance

Reports, per (assay, frac, seed, K, rule):
  predicted gain, realized gain, ceiling, overlap with true top-K.

Everything is relative to base = median(y_train), as in Probe C.

Hardening carried over from Probe C: per-assay guard, per-fit guard,
NaN/inf filtering, incremental JSONL writes, resume.
"""
import os, sys, json, glob, hashlib
import numpy as np
import pandas as pd
from multiprocessing import Pool

OUTCSV  = os.environ.get('PROBEE_OUT', 'probeE.csv')
WORKERS = int(os.environ.get('PROBEE_WORKERS', '8'))
ROWCAP  = 20000
FRACS   = (0.1, 0.3, 0.6)
KS      = (10, 20, 50)
SEEDS   = 5
CALIB   = 0.30          # share of training rows held out to calibrate

AAS = 'ACDEFGHIKLMNPQRSTVWY'
AAI = {a: i for i, a in enumerate(AAS)}

KD = dict(zip(AAS, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,
                    1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOL = dict(zip(AAS, [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,
                     162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
CHG = dict(zip(AAS, [0,0,-1,-1,0,0,0.1,0,1,0,0,0,0,0,1,0,0,0,0,0]))

try:
    from Bio.Align import substitution_matrices
    _B = substitution_matrices.load('BLOSUM62')
    def blosum(a, b):
        try: return float(_B[a, b])
        except Exception: return 0.0
except Exception:
    def blosum(a, b): return 1.0 if a == b else -1.0


def parse(m):
    """'A123G' -> (wt, pos, mut); returns None if unparseable."""
    try:
        wt, mut = m[0], m[-1]
        pos = int(m[1:-1])
        if wt in AAI and mut in AAI:
            return wt, pos, mut
    except Exception:
        pass
    return None


def featurize(muts, positions):
    """Additive features: position one-hot + wt + mut + BLOSUM + phys-chem deltas.

    Deliberately additive so it generalises to unseen (pos, aa) pairs --
    the degenerate one-hot(pos, aa) encoding has zero training support on
    single-mutant DMS and collapses to the intercept.
    """
    pidx = {p: i for i, p in enumerate(positions)}
    P, A = len(positions), len(AAS)
    X = np.zeros((len(muts), P + 2 * A + 5), dtype=np.float32)
    for r, parts in enumerate(muts):
        wt, pos, mut = parts
        X[r, pidx[pos]] = 1.0
        X[r, P + AAI[wt]] = 1.0
        X[r, P + A + AAI[mut]] = 1.0
        o = P + 2 * A
        X[r, o + 0] = blosum(wt, mut)
        X[r, o + 1] = KD[mut] - KD[wt]
        X[r, o + 2] = (VOL[mut] - VOL[wt]) / 100.0
        X[r, o + 3] = CHG[mut] - CHG[wt]
        X[r, o + 4] = 1.0 if (mut == 'P') != (wt == 'P') else 0.0
    return X


def ridge_fit(X, y, lams=np.logspace(-2, 4, 13), folds=5):
    """Ridge with CV-chosen lambda. Returns (w, b, lam, A_inv, sigma2).

    A_inv is (X^T X + lam I)^-1 on centred X, used for predictive variance.
    """
    n, d = X.shape
    mu, ym = X.mean(0), y.mean()
    Xc, yc = X - mu, y - ym
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    best, bestlam = np.inf, lams[0]
    if n >= 2 * folds:
        parts = np.array_split(idx, folds)
        for lam in lams:
            err = 0.0
            for f in range(folds):
                te = parts[f]
                tr = np.concatenate([parts[g] for g in range(folds) if g != f])
                Xt, yt = Xc[tr], yc[tr]
                A = Xt.T @ Xt + lam * np.eye(d, dtype=np.float64)
                try:
                    w = np.linalg.solve(A, Xt.T @ yt)
                except np.linalg.LinAlgError:
                    err = np.inf; break
                err += float(((Xc[te] @ w - yc[te]) ** 2).sum())
            if err < best:
                best, bestlam = err, lam
    lam = bestlam
    A = Xc.T @ Xc + lam * np.eye(d, dtype=np.float64)
    w = np.linalg.solve(A, Xc.T @ yc)
    A_inv = np.linalg.inv(A)
    resid = yc - Xc @ w
    dof = max(n - 1, 1)
    sigma2 = float((resid ** 2).sum() / dof)
    return w, ym, mu, lam, A_inv, sigma2


def predict(Xn, w, ym, mu, A_inv, sigma2):
    """Point prediction and predictive variance (leverage term + noise)."""
    Xc = Xn - mu
    pred = Xc @ w + ym
    lev = np.einsum('ij,jk,ik->i', Xc, A_inv, Xc)
    var = sigma2 * (1.0 + np.maximum(lev, 0.0))
    return pred, var


def gains(score, ytest, base, K):
    """Return (predicted-proxy gain, realized gain, ceiling, overlap) for top-K."""
    K = min(K, len(ytest))
    top = np.argsort(-score)[:K]
    best = np.argsort(-ytest)[:K]
    realized = float(ytest[top].mean() - base)
    ceiling = float(ytest[best].mean() - base)
    overlap = len(set(top.tolist()) & set(best.tolist())) / float(K)
    return realized, ceiling, overlap, top


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
        if '/' in m or ':' in m or ',' in m:   # single mutants only
            continue
        p = parse(m)
        if p is not None:
            parsed.append(p); keep.append(i)
    if len(parsed) < 200:
        return []

    y = pd.to_numeric(df[sc].values[keep], errors='coerce').astype(float)
    positions = sorted({p[1] for p in parsed})
    X = featurize(parsed, positions)

    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    if len(y) < 200 or not np.isfinite(y).all() or y.std() < 1e-9:
        return []

    n = len(y)
    out = []
    for tf in FRACS:
        ntr = int(round(tf * n))
        if ntr < 100 or n - ntr < 100:
            continue
        for seed in range(SEEDS):
            try:
                rng = np.random.RandomState(1000 * seed + 7)
                idx = rng.permutation(n)
                tr, te = idx[:ntr], idx[ntr:]

                # inner split of TRAIN into fit / calibration
                ncal = max(50, int(round(CALIB * ntr)))
                if ntr - ncal < 80:
                    continue
                fit_i, cal_i = tr[:-ncal], tr[-ncal:]

                Xf, yf = X[fit_i], y[fit_i]
                Xc_, yc_ = X[cal_i], y[cal_i]
                Xte, yte = X[te], y[te]
                if yf.std() < 1e-9:
                    continue

                w, ym, mu, lam, A_inv, s2 = ridge_fit(Xf, yf)
                pc, vc = predict(Xc_, w, ym, mu, A_inv, s2)
                pt, vt = predict(Xte, w, ym, mu, A_inv, s2)
                if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
                    continue

                base = float(np.median(yf))
                sdc, sdt = np.sqrt(np.maximum(vc, 0)), np.sqrt(np.maximum(vt, 0))

                # --- calibration model g(yhat, sd) fitted on the calib slice ---
                def design(p, s):
                    return np.column_stack([p, s, p * s, np.ones_like(p)])
                Gc, Gt = design(pc, sdc), design(pt, sdt)
                try:
                    coef, *_ = np.linalg.lstsq(Gc, yc_, rcond=None)
                    cal_score = Gt @ coef
                except Exception:
                    cal_score = pt.copy()
                if not np.isfinite(cal_score).all():
                    cal_score = pt.copy()

                # --- LCB with c tuned on the calib slice ---
                bestc, bestv = 0.0, -np.inf
                for c in (0.0, 0.25, 0.5, 1.0, 2.0):
                    s = pc - c * sdc
                    kk = min(20, len(s))
                    v = float(yc_[np.argsort(-s)[:kk]].mean())
                    if v > bestv:
                        bestv, bestc = v, c
                lcb_score = pt - bestc * sdt

                # --- variance shrinkage toward the training median ---
                sh = s2 / (s2 + np.maximum(vt - s2, 0.0) + 1e-12)
                shrink_score = base + sh * (pt - base)

                from scipy.stats import spearmanr
                rho = spearmanr(pt, yte).statistic
                pear = float(np.corrcoef(pt, yte)[0, 1])

                for K in KS:
                    if K > len(yte):
                        continue
                    for rule, score in (('raw', pt), ('cal', cal_score),
                                        ('lcb', lcb_score), ('shrink', shrink_score)):
                        realized, ceiling, overlap, top = gains(score, yte, base, K)
                        promised = float(pt[top].mean() - base)   # what the model advertises
                        out.append(dict(assay=name, frac=tf, seed=seed, K=K, rule=rule,
                                        n=int(n), L=len(positions),
                                        promised=promised, realized=realized,
                                        ceiling=ceiling, overlap=overlap,
                                        spearman=float(rho) if np.isfinite(rho) else None,
                                        pearson=pear if np.isfinite(pear) else None,
                                        lcb_c=bestc, lam=float(lam)))
            except Exception as e:
                print('  fit-fail', name, tf, seed, type(e).__name__, e, flush=True)
                continue
    return out


if __name__ == '__main__':
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(d, '*.csv')))
    JL = os.path.splitext(OUTCSV)[0] + '_rows.jsonl'

    done = set()
    if os.path.exists(JL):
        with open(JL) as fh:
            for line in fh:
                try: done.add(json.loads(line)['assay'])
                except Exception: pass
    todo = [f for f in files if os.path.basename(f) not in done]
    print(f'{len(files)} assay files, {WORKERS} workers, rowcap {ROWCAP}', flush=True)
    print(f'resuming: {len(done)} assays already done, {len(todo)} to go', flush=True)

    nrows = 0
    with Pool(WORKERS) as pool, open(JL, 'a') as fh:
        for i, rows in enumerate(pool.imap_unordered(run_assay, todo), 1):
            for r in rows:
                fh.write(json.dumps(r) + '\n')
            fh.flush()
            nrows += len(rows)
            if i % 10 == 0 or i == len(todo):
                print(f'  {i}/{len(todo)} done, {nrows} rows', flush=True)

    allrows = [json.loads(l) for l in open(JL)]
    df = pd.DataFrame(allrows)
    df.to_csv(OUTCSV, index=False)
    print('=== PROBE E ===', flush=True)
    print(f'{len(df)} rows, {df.assay.nunique()} assays', flush=True)

    # headline: does any rule beat 'raw' on REALIZED gain?
    g = df.groupby(['assay', 'frac', 'K', 'rule'], as_index=False).mean(numeric_only=True)
    piv = g.pivot_table(index=['assay', 'frac', 'K'], columns='rule',
                        values='realized')
    piv = piv.dropna()
    print('\n--- realized gain, paired across (assay,frac,K) ---', flush=True)
    print(f'n pairs = {len(piv)}', flush=True)
    for rule in ('cal', 'lcb', 'shrink'):
        if rule not in piv.columns:
            continue
        d_ = piv[rule] - piv['raw']
        wins = float((d_ > 0).mean())
        from scipy.stats import wilcoxon
        try:
            p = wilcoxon(piv[rule], piv['raw']).pvalue
        except Exception:
            p = float('nan')
        print(f'{rule:7s} vs raw: median delta {d_.median():+.4f}  '
              f'mean {d_.mean():+.4f}  wins {wins:.1%}  wilcoxon p={p:.3g}', flush=True)

    print('\n--- by K ---', flush=True)
    for K in KS:
        s = g[g.K == K]
        pv = s.pivot_table(index=['assay', 'frac'], columns='rule', values='realized').dropna()
        if len(pv) == 0 or 'cal' not in pv.columns:
            continue
        print(f'K={K:3d} n={len(pv):4d} ' + '  '.join(
            f'{r}:{(pv[r]-pv["raw"]).median():+.4f}' for r in ('cal','lcb','shrink') if r in pv.columns), flush=True)

    print('PROBE_E_DONE', flush=True)
