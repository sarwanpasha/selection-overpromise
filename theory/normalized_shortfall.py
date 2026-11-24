"""Does the theory reproduce the OBSERVED FLAT normalized shortfall?

normalized shortfall = SHORTFALL(rho,q) / CEILING(q),
CEILING(q) = E[G(z) | z > z_q] - G(0)      (base = median of y)
"""
import math, numpy as np
from numpy.polynomial.hermite_e import hermeval, hermegauss
from scipy.stats import norm

def G(kind, z):
    if kind=='linear':  return z
    if kind=='concave': return np.where(z>0, np.tanh(z), z)
    if kind=='convex':  return np.where(z>0, np.abs(z)**1.6, z)
    if kind=='bimodal':
        u = norm.cdf(z)
        return np.where(u<0.45, -2.0+0.3*norm.ppf(np.clip(u/0.45,1e-9,1-1e-9)),
                                 0.5*norm.ppf(np.clip((u-0.45)/0.55,1e-9,1-1e-9)))
def He(k,x):
    c=np.zeros(k+1); c[k]=1.0; return hermeval(x,c)
X,W = hermegauss(300); W=W/W.sum()
def coeffs(kind,kmax=12): 
    g=G(kind,X); return np.array([float((W*g*He(k,X)).sum()) for k in range(kmax+1)])

KMAX=12
rs=np.random.RandomState(5); _z=rs.randn(40000000)
print(f'{"G":9} ' + ' '.join(f'{r:>8}' for r in (0.3,0.5,0.7,0.85,0.95,1.0)) + '   <- normalized shortfall')
for Q in (0.01, 0.02):
    zq=norm.ppf(1-Q); R=_z[_z>zq]
    EHe={k: float(He(k,R).mean()) for k in range(2,KMAX+1)}
    ceil_={}
    for kind in ('convex','linear','bimodal','concave'):
        gR=G(kind,R); ceil_[kind]=float(gR.mean()-G(kind,np.array([0.0]))[0])
    print(f'--- top-{Q*100:.0f}% selected ---')
    for kind in ('convex','linear','bimodal','concave'):
        c=coeffs(kind); row=[]
        for rho in (0.3,0.5,0.7,0.85,0.95,1.0):
            sf=-sum(c[k]/math.factorial(k)*rho**k*EHe[k] for k in range(2,KMAX+1))
            row.append(sf/ceil_[kind])
        print(f'{kind:9} ' + ' '.join(f'{v:8.3f}' for v in row) + f'   (ceiling={ceil_[kind]:.3f})')
print("""
OBSERVED (217 ProteinGym assays, top-20 of ~2000-3000 => q ~ 0.01):
  rho 0.5-0.6: 0.260   0.6-0.7: 0.241   0.7-0.8: 0.215   0.8-0.9: 0.243   0.9-1.0: 0.149""")
