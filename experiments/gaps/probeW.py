"""Probe W -- does the gap survive a protein language model embedding regressor?

The protein engineering reviewer's sharpest objection: the paper's surrogate is a
ridge on one-hot and amino-acid descriptors, and the "stronger surrogate" test only
appends a zero-shot score to it.  Supervised baselines that regress on language model
representations are stronger than either, and the paper's most counterintuitive claim,
that a better model overpromises more, should be tested against one.

For single substitutions the representation of a variant can be read off one forward
pass of the wild-type sequence: the residue-level hidden state at the mutated position
carries the structural and evolutionary context, and the substitution itself is
described by the identity of the two amino acids.  That makes the embedding cost one
pass per assay rather than one per variant.

Feature vector per variant:
    h[pos]                      language model hidden state at the mutated site
    one-hot of the mutant residue, one-hot of the wild-type residue
    optionally the zero-shot score of the same model

Splits, budgets and seeds match the rest of the paper, so the gap is comparable.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probeE import parse, ridge_fit, predict
from probeH import tail_index, iso_fit

OUT     = os.environ.get('PROBEW_OUT', 'probeW.csv')
ROWCAP  = int(os.environ.get('PROBEW_ROWCAP', '20000'))
MODEL   = os.environ.get('PROBEW_MODEL', 'facebook/esm2_t33_650M_UR50D')
CACHE   = os.environ.get('PROBEW_CACHE', 'esm_cache')
FRACS   = (0.1, 0.3, 0.6)
KS      = (10, 20, 50)
SEEDS   = 5
CALIB   = 0.30
AAS     = 'ACDEFGHIKLMNPQRSTVWY'
AAI     = {a: i for i, a in enumerate(AAS)}


def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL).to(dev).eval()
    print('model on %s' % dev, flush=True)
    return tok, mod, dev


def embed_sequence(seq, tok, mod, dev, chunk=1022):
    """Residue-level hidden states for a sequence, in windows if it is long."""
    import torch
    reps = []
    for start in range(0, len(seq), chunk):
        piece = seq[start:start + chunk]
        enc = tok(piece, return_tensors='pt', add_special_tokens=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            h = mod(**enc).last_hidden_state[0]
        reps.append(h[1:1 + len(piece)].float().cpu().numpy())
    return np.concatenate(reps, 0)


def wt_sequence(ref, name):
    """Target sequence for an assay from the ProteinGym reference file."""
    stem = name[:-4] if name.endswith('.csv') else name
    hit = ref[ref.DMS_id == stem]
    if len(hit) == 0:
        return None
    for col in ('target_seq', 'seq', 'sequence'):
        if col in hit.columns:
            s = str(hit.iloc[0][col])
            if s and s != 'nan':
                return s
    return None


def build_features(parsed, reps):
    n = len(parsed)
    d = reps.shape[1]
    X = np.zeros((n, d + 40), dtype=np.float32)
    for i, (wt, pos, mut) in enumerate(parsed):
        p = pos - 1
        if p < 0 or p >= len(reps):
            return None
        X[i, :d] = reps[p]
        if mut in AAI: X[i, d + AAI[mut]] = 1.0
        if wt in AAI:  X[i, d + 20 + AAI[wt]] = 1.0
    return X


def run_assay(path, ref, tok, mod, dev):
    name = os.path.basename(path)
    df = pd.read_csv(path)
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
    seq = wt_sequence(ref, name)
    if not seq:
        print('NOSEQ', name, flush=True)
        return []
    os.makedirs(CACHE, exist_ok=True)
    cf = os.path.join(CACHE, name.replace('.csv', '.npy'))
    if os.path.exists(cf):
        reps = np.load(cf)
    else:
        reps = embed_sequence(seq, tok, mod, dev)
        np.save(cf, reps.astype(np.float16))
        reps = reps.astype(np.float32)
    reps = reps.astype(np.float32)

    y = pd.to_numeric(df[sc].values[keep], errors='coerce').astype(float)
    X = build_features(parsed, reps)
    if X is None:
        print('POSOUT', name, flush=True)
        return []
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
                rng = np.random.RandomState(1000 * seed + 7)
                idx = rng.permutation(n)
                tr, te = idx[:ntr], idx[ntr:]
                ncal = max(50, int(round(CALIB * ntr)))
                if ntr - ncal < 80:
                    continue
                fit_i, cal_i = tr[:-ncal], tr[-ncal:]
                if y[fit_i].std() < 1e-9 or y[cal_i].std() < 1e-9:
                    continue
                w, ym, mu, lam, A_inv, s2 = ridge_fit(X[fit_i], y[fit_i])
                pc, _ = predict(X[cal_i], w, ym, mu, A_inv, s2)
                pt, _ = predict(X[te], w, ym, mu, A_inv, s2)
                if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
                    continue
                try:    g = iso_fit(pc, y[cal_i])
                except Exception: g = lambda v: np.asarray(v, float)
                cal = np.asarray(g(pt), float); cal[~np.isfinite(cal)] = 0.0
                base = float(np.median(y[fit_i]))
                yte = y[te]
                rho = spearmanr(pt, yte).statistic
                for K in KS:
                    if K > len(yte):
                        continue
                    top = np.argsort(-pt)[:K]
                    bestK = np.argsort(-yte)[:K]
                    out.append(dict(assay=name, frac=tf, seed=seed, K=K, tau=tau,
                                    ceiling=float(yte[bestK].mean() - base),
                                    realized=float(yte[top].mean() - base),
                                    fc_raw=float(pt[top].mean() - base),
                                    fc_iso=float(cal[top].mean() - base),
                                    spearman=float(rho) if np.isfinite(rho) else None))
            except Exception as e:
                print('  fit-fail', name, tf, seed, type(e).__name__, e, flush=True)
                continue
    return out


def report(df):
    from scipy.stats import wilcoxon
    d = df[df.ceiling.abs() > 1e-9].copy()
    cell = d.groupby(['assay', 'frac', 'K'], as_index=False).median(numeric_only=True)
    a = cell.groupby('assay', as_index=False).median(numeric_only=True)
    g = lambda v: int(round(1000 * float(np.median(v))))
    sr = (a.fc_raw - a.realized) / a.ceiling
    si = (a.fc_iso - a.realized) / a.ceiling
    print('W assays %d rho %d' % (len(a), g(a.spearman)), flush=True)
    print('Wmed %d %d' % (g(sr), g(si)), flush=True)
    print('Wabs %d %d' % (g(sr.abs()), g(si.abs())), flush=True)
    print('Wmeanabs %d %d' % (int(round(1000 * sr.abs().mean())),
                              int(round(1000 * si.abs().mean()))), flush=True)
    try:    p = wilcoxon(si.abs(), sr.abs()).pvalue
    except Exception: p = float('nan')
    print('Wcmp_iso_raw %d %d' % (round(100 * float((si.abs() < sr.abs()).mean())),
          round(np.log10(p)) if np.isfinite(p) and p > 0 else -999), flush=True)
    print('Wover_pct %d' % round(100 * float((sr > 0).mean())), flush=True)


if __name__ == '__main__':
    dpath, refpath = sys.argv[1], sys.argv[2]
    ref = pd.read_csv(refpath)
    print('reference rows %d cols %s' % (len(ref), list(ref.columns)[:6]), flush=True)
    files = sorted(glob.glob(os.path.join(dpath, '*.csv')))
    JL = os.path.splitext(OUT)[0] + '_rows.jsonl'
    done = set()
    if os.path.exists(JL):
        for line in open(JL):
            try: done.add(json.loads(line)['assay'])
            except Exception: pass
    todo = [f for f in files if os.path.basename(f) not in done]
    print('%d files; resuming %d, todo %d' % (len(files), len(done), len(todo)), flush=True)
    tok, mod, dev = load_model()
    nrows = 0
    with open(JL, 'a') as fh:
        for i, path in enumerate(todo, 1):
            try:
                rows = run_assay(path, ref, tok, mod, dev)
            except Exception as e:
                print('FAILED', os.path.basename(path), type(e).__name__, e, flush=True)
                rows = []
            for r in rows: fh.write(json.dumps(r) + '\n')
            fh.flush(); nrows += len(rows)
            if i % 10 == 0 or i == len(todo):
                print('  %d/%d done, %d rows' % (i, len(todo), nrows), flush=True)
    df = pd.DataFrame([json.loads(l) for l in open(JL)])
    df.to_csv(OUT, index=False)
    print('=== PROBE W ===', flush=True)
    try:
        report(df)
    except Exception as e:
        print('REPORT FAIL', type(e).__name__, e, flush=True)
    print('PROBE_W_DONE', flush=True)
