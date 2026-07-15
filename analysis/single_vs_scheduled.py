import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False

def closed_loop_pi_fopdt(K, tau, L, Kc, Ti, sp=1.0, dt=0.02, T=120, mv_min=0, mv_max=100):
    n = int(T/dt)
    t = np.arange(n)*dt
    y = np.zeros(n); mv = np.zeros(n); integ = 0.0
    Ld = int(round(L/dt))
    mv_hist = np.zeros(n)
    for i in range(1, n):
        e = sp - y[i-1]
        integ += e*dt
        u = Kc*e + (Kc/Ti)*integ
        u = np.clip(u, mv_min, mv_max)
        mv_hist[i] = u
        u_delayed = mv_hist[i-Ld] if i-Ld >= 0 else 0.0
        y[i] = y[i-1] + dt*(K*u_delayed - y[i-1])/tau
    return t, y, mv_hist

K, tau = 15.0, 13.0   # representative gain/tau, held fixed -- isolating the effect of L alone
Kc_cold, Ti_cold = 0.0324, 13.0   # SIMC conservative, tuned FOR L=13.8 (tau_c = tau)
Kc_hot,  Ti_hot  = 0.867,  4.0    # SIMC aggressive, tuned FOR L=0.5  (tau_c = L)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
fig.patch.set_facecolor('#fcfcfb')
scenarios = [
    ('保守整定(按L=13.8s设计) 用在 L=0.5s(热态)', Kc_cold, Ti_cold, 0.5, axes[0,0]),
    ('保守整定(按L=13.8s设计) 用在 L=13.8s(冷启动)', Kc_cold, Ti_cold, 13.8, axes[0,1]),
    ('激进整定(按L=0.5s设计) 用在 L=0.5s(热态)', Kc_hot, Ti_hot, 0.5, axes[1,0]),
    ('激进整定(按L=0.5s设计) 用在 L=13.8s(冷启动)', Kc_hot, Ti_hot, 13.8, axes[1,1]),
]
for title, Kc, Ti, L, ax in scenarios:
    t, y, mv = closed_loop_pi_fopdt(K, tau, L, Kc, Ti)
    ax.set_facecolor('#fcfcfb')
    ax.axhline(1.0, color='#4a3aa7', ls='--', lw=1.3, label='SV')
    ax.plot(t, y, color='#2a78d6', lw=1.6, label='PV(仿真)')
    ax.set_title(title, fontsize=10.5, loc='left')
    ax.grid(True, color='#e1e0d9', lw=0.8); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_ylim(-0.5, 2.5)
for ax in axes[1,:]: ax.set_xlabel('时间 (s)')
for ax in axes[:,0]: ax.set_ylabel('归一化 PV')
fig.suptitle('同一个K,τ，只改死区L：固定PI参数无法同时适配两端', fontsize=13, y=1.00)
fig.tight_layout()
fig.savefig('analysis/single_pid_two_extremes.png', dpi=160, facecolor=fig.get_facecolor())
print('saved analysis/single_pid_two_extremes.png')

# quantify: overshoot / settling for each
for title, Kc, Ti, L, ax in scenarios:
    t, y, mv = closed_loop_pi_fopdt(K, tau, L, Kc, Ti)
    peak = y.max()
    overshoot = max(0, (peak-1.0))*100
    # rough settle time: last time |y-1|>5%
    band = np.where(np.abs(y-1.0) > 0.05)[0]
    settle = t[band[-1]] if len(band) else 0.0
    print(f'{title:<45} peak={peak:.2f}  overshoot={overshoot:.0f}%  settle~{settle:.1f}s  final_mv_swing_std={mv[len(mv)//2:].std():.2f}')
