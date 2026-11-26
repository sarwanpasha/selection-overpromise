"""Probe S -- does the gap survive Bayesian-optimization acquisition rules?

Greedy top-K from a fixed library is one design rule among many.  The objection is
that acquisition functions with a calibrated posterior might behave differently.  They
cannot change the argument of Proposition 3.1, which constrains any rule measurable in
the score, but whether the measured gap and its tail dependence persist is empirical.

Rules, all using the ridge posterior mean and predictive standard deviation:
  greedy : top K by posterior mean, the rule used elsewhere in the paper
  ucb1   : top K by mean + 1 sd
  ucb2   : top K by mean + 2 sd
  ei     : top K by expected improvement over the best training measurement
  ts     : Thompson sampling, one posterior draw per candidate, top K of the draw

For each rule the forecast is the mean posterior value of the chosen batch, which is
what a campaign would report, and the corrected forecast applies isotonic recalibration
fitted on the held-out calibration slice.  Realized gain is also reported per rule, so
the batch quality and the forecast quality can be read separately.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, featurize, ridge_fit, predict
from probeH import tail_index, iso_fit
from probeQ import lin_fit

OUT     = os.environ.get('PROBES_OUT', 'probeS.csv')
WORKERS = int(os.environ.get('PROBES_WORKERS', '16'))
ROWCAP  = int(os.environ.get('PROBES_ROWCAP', '20000'))
FRACS   = (0.1, 0.3, 0.6)
KS      = (10, 20, 50)
SEEDS   = 3
CALIB   = 0.30
ACQS    = ('greedy', 'ucb1', 'ucb2', 'ei', 'ts')


def acquisition(name, mean, sd, best, rng):
    if name == 'greedy':
        return mean
    if name == 'ucb1':
        return mean + sd
    if name == 'ucb2':
        return mean + 2.0 * sd
    if name == 'ei':
        z = (mean - best) / np.maximum(sd, 1e-12)
        return (mean - best) * norm.cdf(z) + sd * norm.pdf(z)
    if name == 'ts':
        return mean + sd * rng.standard_normal(len(mean))
    raise ValueError(name)


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
                pt, vt = predict(Xte, w, ym, mu, A_inv, s2)
                if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
                    continue
                sd = np.sqrt(np.maximum(vt, 0.0))
                best = float(yf.max())
                base = float(np.median(yf))
                try:    g_iso = iso_fit(pc, yc)
                except Exception: g_iso = lambda v: np.asarray(v, float)
                g_lin, _ = lin_fit(pc, yc)
                cal_iso = np.asarray(g_iso(pt), float); cal_iso[~np.isfinite(cal_iso)] = 0.0
                cal_lin = np.asarray(g_lin(pt), float); cal_lin[~np.isfinite(cal_lin)] = 0.0
                rho = spearmanr(pt, yte).statistic
                gen = np.random.default_rng(10000 + seed)
                for acq in ACQS:
                    a = acquisition(acq, pt, sd, best, gen)
                    for K in KS:
                        if K > len(yte):
                            continue
                        top  = np.argsort(-a)[:K]
                        bestK = np.argsort(-yte)[:K]
                        out.append(dict(
                            assay=name, acq=acq, frac=tf, seed=seed, K=K, tau=tau,
                            base=base, ceiling=float(yte[bestK].mean() - base),
                            realized=float(yte[top].mean() - base),
                            fc_raw=float(pt[top].mean() - base),
                            fc_iso=float(cal_iso[top].mean() - base),
                            fc_lin=float(cal_lin[top].mean() - base),
                            spearman=float(rho) if np.isfinite(rho) else None))
            except Exception as e:
                print('  fit-fail', name, tf, seed, type(e).__name__, e, flush=True)
                continue
    return out


def report(df):
    from scipy.stats import wilcoxon
    d = df[df.ceiling > 1e-9].copy()
    lines = [('TOTAL_ROWS', len(df)), ('ASSAYS', d.assay.nunique())]
    gre = None
    for acq in ACQS:
        m = d[d.acq == acq]
        cell = m.groupby(['assay', 'frac', 'K'], as_index=False).median(numeric_only=True)
        assay = cell.groupby('assay', as_index=False).median(numeric_only=True)
        lines.append(('%s_ASSAYS' % acq, len(assay)))
        for r in ('raw', 'iso', 'lin'):
            s = (assay['fc_' + r] - assay['realized']) / assay['ceiling']
            lines.append(('%s_%s_median_s' % (acq, r), round(float(s.median()), 4)))
            lines.append(('%s_%s_meanabs_s' % (acq, r), round(float(s.abs().mean()), 4)))
        si = ((assay.fc_iso - assay.realized) / assay.ceiling).abs()
        sr = ((assay.fc_raw - assay.realized) / assay.ceiling).abs()
        try:    pv = wilcoxon(si, sr).pvalue
        except Exception: pv = float('nan')
        lines.append(('%s_iso_vs_raw_winpct' % acq, round(100 * float((si < sr).mean()), 1)))
        lines.append(('%s_iso_vs_raw_log10p' % acq,
                      round(float(np.log10(pv)), 2) if np.isfinite(pv) and pv > 0 else 'NA'))
        cap = assay.realized / assay.ceiling
        lines.append(('%s_captured_median' % acq, round(float(cap.median()), 4)))
        if acq == 'greedy':
            gre = assay.set_index('assay').realized / assay.set_index('assay').ceiling
        else:
            cur = assay.set_index('assay').realized / assay.set_index('assay').ceiling
            j = gre.index.intersection(cur.index)
            lines.append(('%s_vs_greedy_captured_delta' % acq,
                          round(float((cur[j] - gre[j]).median()), 4)))
    txt = '\n'.join('%s = %s' % (k, v) for k, v in lines)
    open('probeS_report.txt', 'w').write(txt)
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
    print('=== PROBE S ===', flush=True)
    report(df)
    print('PROBE_S_DONE', flush=True)
