import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
df = pd.read_csv('blend.csv')
for r in ('raw','iso','shr'): df['s_'+r] = (df['fc_'+r]-df.realized)/df.ceiling
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); FS = 17
C = {'raw':'#c0392b','iso':'#1f6fb4','shr':'#1d8348'}
L = {'raw':'reported score','iso':'isotonic','shr':'shrunk isotonic'}

g = df.groupby('delta').agg(tau=('tau','median'), **{r:('s_'+r,'median') for r in C}).sort_values('tau')
ax[0].axhline(0, color='0.6', lw=1.2); ax[0].axvline(1.0, color='0.6', ls='--', lw=1.2)
for r,m in zip(('raw','iso','shr'),('o','s','^')):
    ax[0].plot(g.tau, g[r], m+'-', ms=9, lw=2.6, color=C[r], label=L[r])
ax[0].set_xlabel(r'tail index  $\tau$', fontsize=FS)
ax[0].set_ylabel('gap / ceiling', fontsize=FS)
ax[0].legend(fontsize=FS-4, frameon=False)

h = df.groupby('rho').agg(**{r:('s_'+r, lambda v: v.abs().mean()) for r in C})
for r,m in zip(('raw','iso','shr'),('o','s','^')):
    ax[1].plot(h.index, h[r], m+'-', ms=9, lw=2.6, color=C[r])
ax[1].set_xlabel(r'surrogate accuracy  $\rho$', fontsize=FS)
ax[1].set_ylabel(r'mean $|$gap$|$ / ceiling', fontsize=FS); ax[1].set_ylim(bottom=0)
for a in ax:
    a.tick_params(labelsize=FS-3)
    for sp in ('top','right'): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('recalibration.pdf', bbox_inches='tight')
fig.savefig('recalibration.png', dpi=170, bbox_inches='tight'); print('saved')
