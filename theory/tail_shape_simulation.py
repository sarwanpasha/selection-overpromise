"""
Why does the normalized shortfall stay FLAT in rho instead of decaying as (1-rho)?

Gaussian selection theory (derivation):
  Let (yhat, y) be bivariate normal, standardized, correlation rho.
  Select the top-q fraction by yhat. With lam(q) = phi(z_q)/q  (inverse Mills),
      E[yhat | selected] = lam(q)
      E[y    | selected] = rho * lam(q)
      ceiling = E[y | top-q by y] = lam(q)
  => normalized shortfall = (lam - rho*lam)/lam = 1 - rho.          (*)

Observed on 217 ProteinGym assays: flat at ~0.21-0.26 above rho=0.5, and at
rho=0.8-0.9 it is 0.243 where (*) predicts 0.158.

HYPOTHESIS: (*) assumes the prediction and the truth have the SAME upper-tail
shape. Real DMS fitness is compressed at the top (saturation / ceiling effects),
while ridge predictions are near-Gaussian by CLT over many additive features.
A globally-calibrated prediction therefore OVERSHOOTS in the top region, and the
overshoot does not vanish as rho -> 1.

Prediction of the hypothesis: shortfall should scale with UPPER-TAIL COMPRESSION
of the assay, independent of rho. Expression assays (strong saturation) should
overpromise most; binding assays (wide dynamic range at the top) least.
Observed: Expression 0.325, Activity 0.242, Stability 0.235, OrgFitness 0.167,
Binding 0.124 -- at matched rho ~0.61.
"""
import numpy as np
from scipy.stats import norm, rankdata, spearmanr

RNG = np.random.RandomState(0)
N, K, REP = 3000, 20, 400


def truth(kind, n, rng):
    """Draw true fitness with a controlled UPPER-tail shape."""
    if kind == 'gaussian':
        return rng.randn(n)
    if kind == 'ceiling':
        # saturating upper tail: soft cap. Mimics expression/activity assays.
        z = rng.randn(n)
        return np.where(z > 0, np.tanh(z), z)
    if kind == 'hardceiling':
        z = rng.randn(n)
        return np.minimum(z, 0.5)
    if kind == 'bimodal':
        # dead cluster + functional cluster, short upper tail (typical DMS)
        d = rng.rand(n) < 0.45
        return np.where(d, rng.randn(n) * 0.3 - 2.0, rng.randn(n) * 0.5)
    if kind == 'heavytop':
        # LONG upper tail: binding-like, wide dynamic range at the top
        z = rng.randn(n)
        return np.where(z > 0, z ** 1.6, z)
    raise ValueError(kind)


def tail_compression(y):
    """Scale-free upper-tail shape: how compressed the top is vs a Gaussian.
    ratio = (q99-q50)/(q75-q50); divided by the Gaussian value (3.22)."""
    q50, q75, q99 = np.percentile(y, [50, 75, 99])
    if q75 - q50 < 1e-12:
        return np.nan
    return ((q99 - q50) / (q75 - q50)) / 3.2242


def simulate(kind, rho_target, n=N, K=K, rep=REP, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(rep):
        y = truth(kind, n, rng)
        # prediction correlated with y's normal score, then GLOBALLY calibrated
        # by least squares -- which is what a fitted ridge model gives you.
        z = norm.ppf((rankdata(y) - 0.5) / n)
        raw = rho_target * z + np.sqrt(max(1 - rho_target ** 2, 0)) * rng.randn(n)
        b, a = np.polyfit(raw, y, 1)
        yhat = a + b * raw                      # calibrated on average

        base = float(np.median(y))
        top = np.argsort(-yhat)[:K]
        best = np.argsort(-y)[:K]
        promised = float(yhat[top].mean() - base)
        realized = float(y[top].mean() - base)
        ceiling = float(y[best].mean() - base)
        if ceiling <= 1e-9:
            continue
        out.append(((promised - realized) / ceiling,
                    spearmanr(yhat, y).statistic))
    a = np.array(out)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1]))


print('upper-tail compression index (1.0 = Gaussian, <1 = compressed top):')
r = np.random.RandomState(7)
for kind in ('heavytop', 'gaussian', 'bimodal', 'ceiling', 'hardceiling'):
    tc = np.median([tail_compression(truth(kind, N, r)) for _ in range(200)])
    print(f'  {kind:12s} {tc:.3f}')

print('\nnormalized shortfall vs rho   (Gaussian theory = 1-rho)')
hdr = f'{"rho":>6} {"1-rho":>7} ' + ' '.join(f'{k:>11}' for k in
      ('heavytop', 'gaussian', 'bimodal', 'ceiling', 'hardceiling'))
print(hdr)
for rt in (0.2, 0.4, 0.6, 0.75, 0.85, 0.95):
    row, rho_emp = [], None
    for kind in ('heavytop', 'gaussian', 'bimodal', 'ceiling', 'hardceiling'):
        sf, rho_e = simulate(kind, rt, seed=hash(kind) % 1000)
        row.append(sf)
        if kind == 'gaussian':
            rho_emp = rho_e
    print(f'{rho_emp:6.2f} {1-rho_emp:7.3f} ' + ' '.join(f'{v:11.3f}' for v in row))
