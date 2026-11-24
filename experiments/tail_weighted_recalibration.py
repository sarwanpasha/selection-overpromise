"""
The CV weight was chosen by overall squared error, but the gap is a tail
phenomenon. Reweight the CV objective toward the region that actually gets
selected and see whether the shrunk estimator becomes uniformly best.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/probe')
from probeH import tail_index, iso_fit
from scipy.stats import spearmanr, wilcoxon, norm, beta, gamma, t as tdist

N,KS,SEEDS,CALIB = 20000,(10,20,50),15,0.30
RHOS,WGRID,NFOLD = (0.5,0.6,0.7,0.8,0.9),np.linspace(0,1,11),5
QSEL = 0.01   # selection region: top 1 percent of scores

def shapes(z):
    p=norm.cdf(z)
    return {'gaussian':z,'lognormal':np.exp(0.8*z),'exponential':-np.log1p(-p*0.999999),
      'uniform':p,'beta_2_5':beta.ppf(p,2,5),'beta_5_2':beta.ppf(p,5,2),
      'gamma_2':gamma.ppf(p,2),'student_t3':tdist.ppf(p,3),'trunc_upper':np.minimum(z,0.8),
      'bimodal':np.where(p<0.45,z-1.6,z+1.6),'logistic':np.log(p/(1-p)),
      'sqrt_abs':np.sign(z)*np.sqrt(np.abs(z))}

def pick_w(x,y,rg,tail):
    n=len(x); fold=rg.permutation(n)%NFOLD; err=np.zeros(len(WGRID))
    thr=np.quantile(x,1-QSEL) if tail else -np.inf
    for k in range(NFOLD):
        tr,va=fold!=k,fold==k
        if tr.sum()<50 or va.sum()<10: return 1.0
        A=np.polyfit(x[tr],y[tr],1); lin=A[0]*x[va]+A[1]
        iso=iso_fit(x[tr],y[tr])(x[va])
        m = x[va]>=thr
        if m.sum()<5: m=np.ones_like(m,bool)
        for j,w in enumerate(WGRID):
            err[j]+=np.sum((y[va][m]-(w*iso+(1-w)*lin)[m])**2)
    return float(WGRID[int(np.argmin(err))])

rows=[]
for seed in range(SEEDS):
    for rho in RHOS:
        rg=np.random.default_rng(1000*seed+int(rho*10))
        z=rg.standard_normal(N); e=rg.standard_normal(N)
        u=rho*z+np.sqrt(1-rho**2)*e
        for name,y in shapes(z).items():
            y=np.asarray(y,float)
            if not np.all(np.isfinite(y)): continue
            idx=rg.permutation(N); nc=int(CALIB*N); cal,te=idx[:nc],idx[nc:]
            A=np.polyfit(u[cal],y[cal],1); raw=A[0]*u[te]+A[1]
            iso=iso_fit(u[cal],y[cal])(u[te])
            w_t=pick_w(u[cal],y[cal],rg,True)
            shrT=w_t*iso+(1-w_t)*raw
            base=float(np.median(y[cal])); yte=y[te]; order=np.argsort(-raw)
            for K in KS:
                top=order[:K]; ceil=float(yte[np.argsort(-yte)[:K]].mean()-base)
                if ceil<=1e-9: continue
                rows.append(dict(shape=name,rho=rho,K=K,w=w_t,tau=tail_index(y[cal]),
                    ceiling=ceil,realized=float(yte[top].mean()-base),
                    fc_raw=float(raw[top].mean()-base),fc_iso=float(iso[top].mean()-base),
                    fc_shrT=float(shrT[top].mean()-base)))

df=pd.DataFrame(rows); df.to_csv('tailcv.csv',index=False)
for r in ('raw','iso','shrT'): df['s_'+r]=(df['fc_'+r]-df.realized)/df.ceiling
print(f"rows={len(df)}  mean tail-CV weight={df.w.mean():.3f}")
print(f"\n{'shape':>12} {'tau':>7} {'raw_abs':>9} {'iso_abs':>9} {'shrT_abs':>9} {'w':>6}")
for s,g in df.groupby('shape'):
    print(f"{s:>12} {g.tau.median():7.3f} {g.s_raw.abs().mean():9.4f} {g.s_iso.abs().mean():9.4f} {g.s_shrT.abs().mean():9.4f} {g.w.mean():6.2f}")
ar=df.s_raw.abs()
print()
for nm in ('iso','shrT'):
    a=df['s_'+nm].abs()
    print(f"{nm:>5}: meanabs {a.mean():.4f}  median {df['s_'+nm].median():+.4f}  win_vs_raw {100*(a<ar).mean():.1f}%  p={wilcoxon(a,ar).pvalue:.2e}")
ai=df.s_iso.abs(); at=df.s_shrT.abs()
print(f"shrT vs iso: win {100*(at<ai).mean():.1f}%  p={wilcoxon(at,ai).pvalue:.2e}")
