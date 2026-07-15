import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False
C_PV, C_FIT, C_MV = '#2a78d6', '#eb6834', '#898781'

def load(fname, sheet):
    df = pd.read_excel(fname, sheet_name=sheet, engine='openpyxl')
    df.columns = ['t','pv1','pv2','sv','mv']
    df['t'] = pd.to_datetime(df['t'])
    df['sec'] = (df['t']-df['t'].iloc[0]).dt.total_seconds()
    return df.drop_duplicates(subset='sec').sort_values('sec').reset_index(drop=True)

def simulate_fopdt(t, u, K, tau, L, y0=0.0):
    u_interp = interp1d(t, u, bounds_error=False, fill_value=(u[0], u[-1]))
    ud = u_interp(t - L)
    y = np.empty_like(t, dtype=float); y[0]=y0
    for i in range(1, len(t)):
        dt = t[i]-t[i-1]
        n_sub = max(1, int(np.ceil(dt/(tau/5+1e-9))))
        yi = y[i-1]
        for k in range(n_sub):
            yi = yi + (dt/n_sub)*(K*ud[i]-yi)/tau
        y[i] = yi
    return y

def fit_and_plot(t, mv, pv, mv_base, pv_base, p0, bounds, title, outpath, note=None):
    u = mv - mv_base; y_meas = pv - pv_base
    def resid(p):
        return simulate_fopdt(t, u, *p) - y_meas
    res = least_squares(resid, p0, bounds=bounds, xtol=1e-12, ftol=1e-12, max_nfev=10000)
    pred = simulate_fopdt(t, u, *res.x) + pv_base
    r2 = 1 - np.sum((pv-pred)**2)/np.sum((pv-pv.mean())**2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={'height_ratios':[2,1]})
    fig.patch.set_facecolor('#fcfcfb'); ax1.set_facecolor('#fcfcfb'); ax2.set_facecolor('#fcfcfb')
    ax1.plot(t, pv, color=C_PV, lw=1.6, label='实测 PV1')
    ax1.plot(t, pred, color=C_FIT, lw=1.6, ls='--', label=f'FOPDT拟合 (R²={r2:.2f})')
    ax1.set_ylabel('温度 (℃)'); ax1.legend(frameon=False, fontsize=9.5)
    ax1.grid(True, color='#e1e0d9', lw=0.8); ax1.set_axisbelow(True)
    for s in ['top','right']: ax1.spines[s].set_visible(False)
    ax1.set_title(title, loc='left', fontsize=12.5)
    if note: ax1.text(0.02, 0.05, note, transform=ax1.transAxes, fontsize=9, color='#52514e')
    ax2.plot(t, mv, color=C_MV, lw=1.3, label='MV')
    ax2.set_ylabel('MV (%)'); ax2.set_xlabel('时间 (s)')
    ax2.grid(True, color='#e1e0d9', lw=0.8); ax2.set_axisbelow(True)
    for s in ['top','right']: ax2.spines[s].set_visible(False)
    ax2.legend(frameon=False, fontsize=9.5)
    fig.tight_layout(); fig.savefig(outpath, dpi=160, facecolor=fig.get_facecolor()); plt.close(fig)
    print('saved', outpath, 'K,tau,L=', np.round(res.x,3), 'R2=', round(r2,4))
    return res.x, r2

# good fit: 60% pulse
d2 = load('AIC9_DATA-20260702-185858_203502-数据分析.xlsx', 'data')
t0, t1 = 1186.20, 1260
pre = d2[(d2['sec']>=t0-3)&(d2['sec']<t0)]
seg = d2[(d2['sec']>=t0-1)&(d2['sec']<=t1)].reset_index(drop=True)
fit_and_plot(seg['sec'].values, seg['mv'].values, seg['pv1'].values, pre['mv'].mean(), pre['pv1'].mean(),
             [8,12,1], ([0.1,1,0],[40,60,15]),
             '60%功率脉冲：FOPDT拟合效果良好（100℃附近工作点）',
             'analysis/fit_good_60pct.png')

# bad fit: clean 50% pulse (short, reveals higher-order behavior)
d = load('AIC9_DATA-20260702-185858_203502-数据分析.xlsx', 'MV50%')
seg = d[(d['sec']>=0)&(d['sec']<=90)].reset_index(drop=True)
base = seg[seg['sec']<1.3]
fit_and_plot(seg['sec'].values, seg['mv'].values, seg['pv1'].values, base['mv'].mean(), base['pv1'].mean(),
             [5,12,1], ([0.1,1,0],[30,60,15]),
             '50%功率脉冲：简单FOPDT拟合失败——揭示更高阶动态',
             'analysis/fit_bad_50pct.png',
             note='注意MV已归零后PV仍继续上冲，单一FOPDT结构无法解释')

# cold start
d3 = load('AIC9_DATA-20260702-203502_204340-PID闭环数据.xlsx', 'data')
seg = d3[(d3['sec']>=0)&(d3['sec']<=25)].reset_index(drop=True)
pre = seg[seg['sec']<2]
fit_and_plot(seg['sec'].values, seg['mv'].values, seg['pv1'].values, pre['mv'].mean(), pre['pv1'].mean(),
             [1,8,10], ([0.05,1,0],[10,30,13.8]),
             '冷启动0→100℃：FOPDT同样无法解释（死区+欠调震荡）',
             'analysis/fit_bad_coldstart.png',
             note='闭环欠阻尼震荡 + 长死区，非简单一阶延迟对象所能产生')

# gain-vs-power summary
levels = [20,40,60]
Ks = [12.543, 26.228, 30.883]
taus = [12.62, 12.88, 13.20]
fig, ax = plt.subplots(figsize=(6,4.2))
fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
ax.plot(levels, Ks, 'o-', color='#4a3aa7', lw=2, markersize=7)
ax.set_xlabel('脉冲功率 MV (%)'); ax.set_ylabel('拟合增益 K (℃ / %MV)')
ax.set_title('稳态增益随功率档位显著非线性', loc='left', fontsize=12.5)
ax.grid(True, color='#e1e0d9', lw=0.8); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('analysis/gain_vs_power.png', dpi=160, facecolor=fig.get_facecolor())
print('saved gain_vs_power.png')
