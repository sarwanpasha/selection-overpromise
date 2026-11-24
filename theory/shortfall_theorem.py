"""
THEOREM (selection overpromise for a calibrated predictor).

Let true fitness be y = G(z), z ~ N(0,1), G monotone increasing, with Hermite
coefficients c_k = E[G(z) He_k(z)].  Let the predictor be s = rho*z + sqrt(1-rho^2)*eps,
calibrated by least squares onto y:  yhat = a + b*s,  b = Cov(y,s)/Var(s) = rho*c1,
a = mu_y.  Select the top-q fraction by yhat; let R denote the selected s
(truncated normal above z_q).

Since z | s ~ N(rho*s, 1-rho^2), the Hermite reproducing property gives
    E[y | s] = mu_y + sum_{k>=1} (c_k/k!) rho^k He_k(s),
while the calibrated prediction keeps only the k=1 term:
    E[yhat | s] = mu_y + rho*c1*He_1(s).
Therefore

    SHORTFALL(rho,q) = E[yhat|sel] - E[y|sel] = - sum_{k>=2} (c_k/k!) rho^k E[He_k(R)]

CONSEQUENCES
 (i)  G linear (Gaussian fitness) => c_k = 0 for k>=2 => shortfall == 0 at every rho.
 (ii) leading order: shortfall ~ -(c2/2) rho^2 E[He_2(R)].  Its SIGN is the sign of
      -c2: a CONCAVE G (compressed/saturating upper tail) overpromises; a CONVEX G
      (long upper tail) UNDER-promises.
 (iii) as rho -> 1 the shortfall tends to -sum_{k>=2} (c_k/k!) E[He_k(R)] != 0:
      a model with PERFECT ranking still overpromises.  The bias does not vanish
      with model quality -- it saturates.  This is the observed flat curve.

Below: verify the identity and (i)-(iii) numerically.
"""
import math
import numpy as np
from numpy.polynomial.hermite_e import hermeval
from scipy.stats import norm
rng = np.random.RandomState(0)

def G(kind, z):
    if kind=='linear':    return z
    if kind=='concave':   return np.where(z>0, np.tanh(z), z)        # saturating top
    if kind=='convex':    return np.where(z>0, z**1.6, z)            # long top tail
    if kind=='bimodal':
        u = norm.cdf(z)
        return np.where(u<0.45, -2.0+0.3*norm.ppf(np.clip(u/0.45,1e-9,1-1e-9)),
                                 0.5*norm.ppf(np.clip((u-0.45)/0.55,1e-9,1-1e-9)))
    raise ValueError

def He(k, x):
    c = np.zeros(k+1); c[k] = 1.0
    return hermeval(x, c)

# --- Hermite coefficients by Gauss-Hermite quadrature -----------------------
from numpy.polynomial.hermite_e import hermegauss
X, W = hermegauss(200); W = W/W.sum()
def coeffs(kind, kmax=8):
    g = G(kind, X)
    return np.array([float((W*g*He(k,X)).sum()) for k in range(kmax+1)])

KMAX, Q, N, REP = 8, 0.02, 200000, 25

print(f'{"G":9} {"c1":>7} {"c2":>8} {"c3":>8}   sign(-c2)')
C = {}
for kind in ('convex','linear','bimodal','concave'):
    c = coeffs(kind); C[kind] = c
    print(f'{kind:9} {c[1]:7.3f} {c[2]:8.3f} {c[3]:8.3f}   {"+" if -c[2]>0 else "-"}')

print('\nshortfall: THEORY (Hermite series) vs SIMULATION')
print(f'{"G":9} {"rho":>5} {"theory":>9} {"sim":>9} {"lead(k=2)":>10}')
zq = norm.ppf(1-Q)
_z = np.random.RandomState(5).randn(40000000); R = _z[_z>zq]
EHe = {k: float(He(k,R).mean()) for k in range(2,KMAX+1)}
print(f'(truncated-normal moments from {len(R)} draws above z_q={zq:.3f})')
for kind in ('convex','linear','bimodal','concave'):
    c = C[kind]
    for rho in (0.3, 0.6, 0.85, 1.0):
        # --- simulation ---
        sf=[]
        for r in range(REP):
            z = rng.randn(N)
            s = rho*z + np.sqrt(max(1-rho**2,0))*rng.randn(N)
            y = G(kind, z)
            b = np.cov(y,s)[0,1]/np.var(s); a = y.mean()-b*s.mean()
            yhat = a+b*s
            k = int(Q*N); top = np.argpartition(-s, k)[:k]
            sf.append(yhat[top].mean()-y[top].mean())
        sim = float(np.mean(sf))
        # --- theory: E[He_k(R)], R = s | s>zq  (precomputed) ---
        th   = -sum(c[k]/math.factorial(k)*rho**k*EHe[k] for k in range(2,KMAX+1))
        lead = -c[2]/2*rho**2*EHe[2]
        print(f'{kind:9} {rho:5.2f} {th:9.4f} {sim:9.4f} {lead:10.4f}')
