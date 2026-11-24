import json,glob,os,sys
import numpy as np,pandas as pd
from scipy.stats import spearmanr
D=sys.argv[1] if len(sys.argv)>1 else 'DMS_ProteinGym_substitutions'
rows=[]
for f in sorted(glob.glob(os.path.join(D,'*.csv'))):
    try:
        df=pd.read_csv(f); sc='DMS_score' if 'DMS_score' in df else df.columns[-1]
        y=pd.to_numeric(df[sc],errors='coerce').values.astype(float); y=y[np.isfinite(y)]
        if len(y)<200: continue
        q50,q75,q99=np.percentile(y,[50,75,99])
        if q75-q50<1e-12: continue
        rows.append(dict(DMS_id=os.path.basename(f)[:-4],
                         tc=((q99-q50)/(q75-q50))/3.449049))
    except Exception as e: print('skip',os.path.basename(f),e)
t=pd.DataFrame(rows); t.to_csv('tailcomp.csv',index=False)
d=pd.DataFrame(map(json.loads,open('probeC_rows.jsonl')))
d['DMS_id']=d.assay.str.replace('.csv','',regex=False)
g=d.groupby(['DMS_id','frac'],as_index=False).mean(numeric_only=True)
g=g[g.ceiling>1e-9]; g['sfn']=(g.predicted-g.realized)/g.ceiling
r=pd.read_csv('ref.csv')[['DMS_id','coarse_selection_type']]
m=g.merge(t,on='DMS_id').merge(r,on='DMS_id')
print('n=',len(m),'assays=',m.DMS_id.nunique())
print('\nTEST 1  shortfall vs tail compression:',spearmanr(m.tc,m.sfn))
print('  partial (control rho): resid corr',
      spearmanr(m.sfn-np.poly1d(np.polyfit(m.spearman,m.sfn,1))(m.spearman),
                m.tc-np.poly1d(np.polyfit(m.spearman,m.tc,1))(m.spearman)))
print('\nTEST 2  by assay class (predicted order: Expression most compressed):')
print(m.groupby('coarse_selection_type')[['tc','sfn','spearman']].median().round(3)
       .sort_values('tc'))
print('\nTEST 3  do LONG-tailed assays UNDER-promise?')
for lo,hi in [(0,.5),(.5,.8),(.8,1.1),(1.1,1.5),(1.5,99)]:
    s=m[(m.tc>=lo)&(m.tc<hi)]
    if len(s): print(f'  tc {lo}-{hi}: n={len(s):4d} shortfall={s.sfn.median():+.3f} '
                     f'neg={100*(s.sfn<0).mean():.0f}% rho={s.spearman.median():.2f}')
print('TAILTEST_DONE')
