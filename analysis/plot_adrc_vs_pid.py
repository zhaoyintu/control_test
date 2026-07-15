import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False

C_SV  = '#4a3aa7'
C_PID = '#e34948'   # red
C_ADRC= '#2a78d6'   # blue
GRID  = '#e1e0d9'
AXIS  = '#c3c2b7'
MUTED = '#898781'
INK   = '#0b0b0b'

def style_axes(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK, fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.yaxis.label.set_color(INK)

with open('analysis/adrc_vs_pid.pkl', 'rb') as f:
    d = pickle.load(f)
results = d['results']
t = d['t']

def sv_schedule(t):
    if t < 10: return 100.0
    elif t < 40: return 300.0
    else: return 400.0
sv = np.array([sv_schedule(tt) for tt in t])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                gridspec_kw={'height_ratios': [2.2, 1]})
fig.patch.set_facecolor('#fcfcfb')
ax1.set_facecolor('#fcfcfb')
ax2.set_facecolor('#fcfcfb')

ax1.step(t, sv, where='post', color=C_SV, linestyle='--', linewidth=1.6, label='SV 设定值')
_, y_pid, u_pid = results['PID_single']
_, y_adrc, u_adrc = results['ADRC_final']
ax1.plot(t, y_pid, color=C_PID, linewidth=1.4, label='PID (SIMC重调, 修正死区L=0.2s)')
ax1.plot(t, y_adrc, color=C_ADRC, linewidth=1.4, label='ADRC (线性化+联合搜索, wc=2.0/wo=20/wr=8)')
ax1.axvline(55, color=MUTED, linewidth=0.8, linestyle=':')
ax1.annotate('注入未建模扰动\n(+8等效度)', xy=(55, 385), fontsize=8.5, color=MUTED)
style_axes(ax1, '温度 (illustrative, 非真实炉温标度)')
ax1.legend(loc='lower right', frameon=False, fontsize=9.5, labelcolor=INK)
ax1.set_title('修正死区(L=0.2s)+线性化+联合搜索后：三项指标同时达标，不是权衡',
              color=INK, fontsize=12.5, loc='left', pad=10)

ax2.plot(t, u_pid, color=C_PID, linewidth=1.1, label='MV (PID)')
ax2.plot(t, u_adrc, color=C_ADRC, linewidth=1.1, label='MV (ADRC)')
style_axes(ax2, 'MV (%)')
ax2.set_xlabel('时间 (s)', color=INK, fontsize=11)
ax2.legend(loc='upper right', frameon=False, fontsize=9.5, labelcolor=INK)

fig.tight_layout()
fig.savefig('analysis/adrc_vs_pid_compare.png', dpi=170, facecolor=fig.get_facecolor())
plt.close(fig)
print('saved analysis/adrc_vs_pid_compare.png')
