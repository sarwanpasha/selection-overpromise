"""
Does tau predict the gap across heterogeneous fitness distributions, or is the
relation an artifact of the sinh-arcsinh family? Tests 12 unrelated shapes.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/probe')
from probeH import tail_index, iso_fit
from scipy.stats import spearmanr, wilcoxon

N, KS, SEEDS, CALIB = 20000, (10,20,50), 15, 0.30
RHOS, WGRID, NFOLD = (0.5,0.6,0.7,0.8,0.9), np.linspace(0,1,11), 5

def shapes(z, rg):
    u01 = 0.5*(1+np.math.erf(0)) if False else None
    from scipy.stats import norm
    p = norm.cdf(z)
    return {
      'gaussian'    : z,
      'lognormal'   : np.exp(0.8*z),
      'exponential' : -np.log1p(-p*0.999999),
      'uniform'     : p,
      'beta_2_5'    : __import__('scipy.stats',fromlist=['beta']).beta.ppf(p,2,5),
      'beta_5_2'    : __import__('scipy.stats',fromlist=['beta']).beta.ppf(p,5,2),
      'gamma_2'     : __import__('scipy.stats',fromlist=['gamma']).gamma.ppf(p,2),
      'student_t3'  : __import__('scipy.stats',fromlist=['t']).t.ppf(p,3),
      'trunc_upper' : np.minimum(z,0.8),
      'bimodal'     : np.where(p<0.45, z-1.6, z+1.6),
      'logistic'    : np.log(p/(1-p)),
      'sqrt_abs'    : np.sign(z)*np.sqrt(np.abs(z)),
    }

def pick_w(x,y,rg):
    n=len(x); fold=rg.permutation(n)%NFOLD; err=np.zeros(len(WGRID))
    for k in range(NFOLD):
        tr,va = fold!=k, fold==k
        if tr.sum()<50 or va.sum()<10: return 1.0
        A=np.polyfit(x[tr],y[tr],1); lin=A[0]*x[va]+A[1]
        iso=iso_fit(x[tr],y[tr])(x[va])
        for j,w in enumerate(WGRID): err[j]+=np.sum((y[va]-(w*iso+(1-w)*lin))**2)
    return float(WGRID[int(np.argmin(err))])

rows=[]
for seed in range(SEEDS):
    for rho in RHOS:
        rg=np.random.default_rng(1000*seed+int(rho*10))
        z=rg.standard_normal(N); e=rg.standard_normal(N)
        u=rho*z+np.sqrt(1-rho**2)*e
        for name,y in shapes(z,rg).items():
            y=np.asarray(y,float)
            if not np.all(np.isfinite(y)): continue
            idx=rg.permutation(N); nc=int(CALIB*N); cal,te=idx[:nc],idx[nc:]
            A=np.polyfit(u[cal],y[cal],1); raw=A[0]*u[te]+A[1]
            iso=iso_fit(u[cal],y[cal])(u[te]); w=pick_w(u[cal],y[cal],rg)
            shr=w*iso+(1-w)*raw
            base=float(np.median(y[cal])); yte=y[te]; order=np.argsort(-raw)
            for K in KS:
                top=order[:K]; ceil=float(yte[np.argsort(-yte)[:K]].mean()-base)
                if ceil<=1e-9: continue
                rows.append(dict(shape=name,rho=rho,seed=seed,K=K,w=w,
                    tau=tail_index(y[cal]),ceiling=ceil,
                    realized=float(yte[top].mean()-base),
                    fc_raw=float(raw[top].mean()-base),
                    fc_iso=float(iso[top].mean()-base),
                    fc_shr=float(shr[top].mean()-base)))

df=pd.DataFrame(rows); df.to_csv('shapes.csv',index=False)
for r in ('raw','iso','shr'): df['s_'+r]=(df['fc_'+r]-df.realized)/df.ceiling
print(f"rows={len(df)} shapes={df.shape[0] and df['shape'].nunique()}")
print(f"\n{'shape':>12} {'tau':>7} {'raw_med':>9} {'shr_med':>9} {'raw_abs':>9} {'iso_abs':>9} {'shr_abs':>9}")
for s,g in df.groupby('shape'):
    print(f"{s:>12} {g.tau.median():7.3f} {g.s_raw.median():9.4f} {g.s_shr.median():9.4f} "
          f"{g.s_raw.abs().mean():9.4f} {g.s_iso.abs().mean():9.4f} {g.s_shr.abs().mean():9.4f}")
pm=df.groupby('shape').agg(tau=('tau','median'),s=('s_raw','median'))
r=spearmanr(pm.tau,pm.s)
print(f"\nacross the 12 shapes: Spearman(tau, median gap) = {r.statistic:.3f}  p = {r.pvalue:.2e}")
ar=df.s_raw.abs()
for nm in ('iso','shr'):
    a=df['s_'+nm].abs()
    print(f"{nm}: meanabs {a.mean():.4f} vs raw {ar.mean():.4f}, win {100*(a<ar).mean():.1f}%, p={wilcoxon(a,ar).pvalue:.2e}")
