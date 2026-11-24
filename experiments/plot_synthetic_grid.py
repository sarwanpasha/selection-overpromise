import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
df = pd.read_csv('simgrid.csv')
df['s_raw'] = (df.fc_raw - df.realized)/df.ceiling
df['s_iso'] = (df.fc_iso - df.realized)/df.ceiling

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
FS = 17

g = df.groupby('delta').agg(tau=('tau','median'), raw=('s_raw','median'), iso=('s_iso','median')).sort_values('tau')
ax[0].axhline(0, color='0.6', lw=1.2, zorder=1)
ax[0].plot(g.tau, g.raw, 'o-', ms=9, lw=2.6, color='#c0392b', label='reported score', zorder=3)
ax[0].plot(g.tau, g.iso, 's-', ms=9, lw=2.6, color='#1f6fb4', label='conditional mean', zorder=3)
ax[0].axvline(1.0, color='0.6', ls='--', lw=1.2, zorder=1)
ax[0].set_xlabel(r'tail index  $\tau$', fontsize=FS)
ax[0].set_ylabel(r'gap / ceiling', fontsize=FS)
ax[0].legend(fontsize=FS-3, frameon=False, loc='upper right')

h = df.groupby('rho').agg(raw=('s_raw', lambda v: v.abs().mean()), iso=('s_iso', lambda v: v.abs().mean()))
ax[1].plot(h.index, h.raw, 'o-', ms=9, lw=2.6, color='#c0392b')
ax[1].plot(h.index, h.iso, 's-', ms=9, lw=2.6, color='#1f6fb4')
ax[1].set_xlabel(r'surrogate accuracy  $\rho$', fontsize=FS)
ax[1].set_ylabel(r'mean $|$gap$|$ / ceiling', fontsize=FS)
ax[1].set_ylim(bottom=0)

for a in ax:
    a.tick_params(labelsize=FS-3)
    for sp in ('top','right'): a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig('sim_recalibration.pdf', bbox_inches='tight')
fig.savefig('sim_recalibration.png', dpi=170, bbox_inches='tight')
print('saved')
print(g.to_string())
print(h.to_string())
