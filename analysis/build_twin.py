"""
全波形输出误差辨识: 用两场真机数据把数字孪生的参数拟合出来
  - 输入: 记录的 MV 序列 (含 PID 顶满80%/一步切0、ADRC振铃、平滑爬升、纯冷却段)
  - 拟合: 开环仿真 PV 轨迹 vs 实测 PV, 最小二乘
  - 参数: q(u) 分段线性 10 个增量 + θ + τ2 + h1 + c   (共 14 个)
  - 验证: 每段 RMSE + 盲考(闭环仿真 wc=8 是否复现 1.6s 振铃)
输出: analysis/twin_params.json + plots/10_数字孪生_辨识验证.png
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = font_manager.FontProperties(
        fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name()
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

DT = 0.01
CSV_AM = 'user_feedback/AIC9_DATA-20260713-053919_060746.csv'
CSV_PM = 'user_feedback/AIC9_DATA-20260713-211745_222036.csv'


def load(path):
    df = pd.read_csv(path)
    df.columns = ['time', 'PV1', 'PV2', 'SV', 'MV'][: len(df.columns)]
    tr = (pd.to_datetime(df['time']) - pd.to_datetime(df['time'].iloc[0])).dt.total_seconds().values
    tg = np.arange(0, tr[-1], DT)
    pv = np.interp(tg, tr, df['PV1'].values.astype(float))
    mv = np.interp(tg, tr, df['MV'].values.astype(float))
    idxp = np.clip(np.searchsorted(tr, tg, side='right') - 1, 0, len(tr) - 1)
    sv = df['SV'].values.astype(float)[idxp]
    return tg, pv, mv, sv

tA, pvA, mvA, svA = load(CSV_AM)   # 上午场
tB, pvB, mvB, svB = load(CSV_PM)   # 晚上场

# ------------------------------------------------ 选段 (t0, t1, 来源, 描述)
def win(src, t0, t1, desc):
    tg, pv, mv, _ = src
    i0, i1 = int(t0 / DT), int(t1 / DT)
    return dict(pv=pv[i0:i1], mv=mv[i0:i1], desc=desc)

A = (tA, pvA, mvA, svA)
B = (tB, pvB, mvB, svB)
# 晚上场 SV 上跳时刻: S1 2351.1 S2 2441.8 S3 2617.6 S4 2739.8 S5 2807.1 S6 3106.2
SEGS = [
    win(B, 2343, 2365, 'S1 旧参 100→200'),
    win(B, 2433, 2458, 'S2 旧参 100→400'),
    win(B, 2609, 2630, 'S3 wc=8 (振铃)'),
    win(B, 2732, 2753, 'S4 wc=2 (振铃)'),
    win(B, 2799, 2822, 'S5 wc=0.9'),
    win(B, 3098, 3119, 'S6 PID (顶满80→切0)'),
    win(B, 2457, 2500, '冷却段 400→ (MV=0)'),
    win(A, 1580, 1640, '上午: 450 稳态保持'),
    win(A, 1655, 1700, '上午: 停机 21.7%→0'),
]

# ------------------------------------------------ 模型 (向量化开环)
U_BP = np.array([0, 5, 10, 15, 22, 30, 40, 55, 70, 85, 100], dtype=float)

def open_loop(useq, y0, qbp, theta, tau2, h1, c):
    t = np.arange(len(useq)) * DT
    u_d = np.interp(t - theta, t, useq)
    q_in = np.interp(u_d, U_BP, qbp)
    a = DT / max(tau2, 1e-4)
    qf0 = float(np.interp(useq[0], U_BP, qbp))
    qf, _ = lfilter([a], [1, -(1 - a)], q_in, zi=[(1 - a) * qf0])
    b = DT * h1
    y, _ = lfilter([DT], [1, -(1 - b)], qf + h1 * c, zi=[(1 - b) * y0])
    return y

def unpack(x):
    d = np.exp(x[:10])                 # 保证增量为正 -> q 单调
    qbp = np.concatenate([[0.0], np.cumsum(d)])
    theta, tau2, h1, c = x[10], np.exp(x[11]), np.exp(x[12]), x[13]
    return qbp, theta, tau2, h1, c

def residuals(x):
    qbp, theta, tau2, h1, c = unpack(x)
    out = []
    for s in SEGS:
        y = open_loop(s['mv'], s['pv'][0], qbp, theta, tau2, h1, c)
        r = (y - s['pv'])[::5]         # 50ms 抽样
        out.append(r / max(np.std(s['pv']), 5.0) * np.sqrt(len(s['pv']) / 2000))
    return np.concatenate(out)

# 权重: 纯冷却段降权 (线性散热对高温辐射欠拟合, 孪生用途是加热过渡段)
W_SEG = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 1.0, 0.6]

def residuals_gain(x, theta, tau2):
    """固定相位参数, 只拟合 q 增量 + h1 + c"""
    d = np.exp(x[:10])
    qbp = np.concatenate([[0.0], np.cumsum(d)])
    h1, c = np.exp(x[10]), x[11]
    out = []
    for w, s in zip(W_SEG, SEGS):
        y = open_loop(s['mv'], s['pv'][0], qbp, theta, tau2, h1, c)
        r = (y - s['pv'])[::5]
        out.append(w * r / max(np.std(s['pv']), 5.0) * np.sqrt(len(s['pv']) / 2000))
    return np.concatenate(out)

# 闭环特征评分: 用四段已知参数的闭环行为选相位
def cl_feature_cost(qbp, theta, tau2, h1, c):
    import twin as twin_mod_
    tw_ = twin_mod_.Twin.__new__(twin_mod_.Twin)
    tw_.u_bp, tw_.q_bp = U_BP, qbp
    tw_.theta, tw_.tau2, tw_.h1, tw_.c, tw_.sig_n = theta, tau2, h1, c, 0.0
    cost = 0.0
    for prm, (r90m, ovm, Tm) in [((3.4, 8.0, 24.0, 3.0, 23), (1.86, 1.9, 1.65)),
                                 ((3.4, 2.0, 24.0, 3.0, 23), (1.81, 2.3, 1.62)),
                                 ((3.4, 0.9, 24.0, 3.0, 23), (3.34, 0.9, np.nan)),
                                 ((10.0, 0.8, 40.0, 10.0, 20), (2.91, 3.2, np.nan))]:
        yy, uu = tw_.closed_loop_adrc(*prm, 100, 400, T=14, noise=0.0)
        import twin as tm
        r90, ov, stl = tm.metrics(yy, 100, 400)
        Tr = ring_period(uu)
        cost += abs(ov - ovm) / 2.0
        cost += abs(r90 - r90m) / 1.0 if not np.isnan(r90) else 2.0
        if not np.isnan(Tm):
            cost += (abs(np.log(Tr / Tm)) if not np.isnan(Tr) else 1.5)
        elif not np.isnan(Tr):
            cost += 0.5
    return cost

def ring_period(u_seg, dt=DT, lo_s=1.0, hi_s=8.0):
    a, b = int(lo_s/dt), min(int(hi_s/dt), len(u_seg))
    x = u_seg[a:b] - pd.Series(u_seg[a:b]).rolling(int(2/dt), center=True, min_periods=1).mean().values
    x = pd.Series(x).rolling(8, center=True, min_periods=1).mean().values
    if np.std(x) < 0.3:
        return np.nan
    m = len(x)
    ac = np.correlate(x, x, 'full')[m-1:]
    ac /= (ac[0] + 1e-12)
    for i in range(int(0.4/dt), len(ac)-1):
        if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0.2:
            return i*dt
    return np.nan

import sys
sys.path.insert(0, 'analysis')

q0 = np.array([0.0502 * u * u + 0.919 * u if u <= 65 else
               254.3 + (327 - 254.3) * (u - 65) / 35 for u in U_BP])
x0g = np.concatenate([np.log(np.maximum(np.diff(q0), 1.0)), [np.log(0.135), 127.0]])
lbg = np.concatenate([np.full(10, -2.0), [np.log(0.02), 30.0]])
ubg = np.concatenate([np.log(np.diff(q0) * 3.0 + 5.0), [np.log(0.60), 250.0]])

print('两级拟合: 外层 (θ,τ2) 网格 x 内层增益最小二乘 ...')
results = []
for theta, tau2 in [(0.10, 0.05), (0.15, 0.05), (0.15, 0.15), (0.20, 0.10),
                    (0.20, 0.20), (0.25, 0.15), (0.30, 0.10), (0.25, 0.25)]:
    sol = least_squares(residuals_gain, x0g, bounds=(lbg, ubg), method='trf',
                        x_scale='jac', max_nfev=250, args=(theta, tau2))
    d = np.exp(sol.x[:10])
    qbp = np.concatenate([[0.0], np.cumsum(d)])
    h1, c = np.exp(sol.x[10]), sol.x[11]
    wf_cost = float(np.sqrt(np.mean(sol.fun**2)))
    cl_cost = cl_feature_cost(qbp, theta, tau2, h1, c)
    results.append((cl_cost + 3.0*wf_cost, cl_cost, wf_cost, theta, tau2, qbp, h1, c))
    print(f'  θ={theta:.2f} τ2={tau2:.2f}: 波形cost={wf_cost:.4f} 闭环cost={cl_cost:.2f}')
results.sort()
_, CLC, WFC, THETA, TAU2, QBP, H1, C = results[0]
print(f'\n选定: θ={THETA} τ2={TAU2} (闭环cost={CLC:.2f}, 波形cost={WFC:.4f})')
print(f'\n===== 辨识结果 =====')
print(f'θ = {THETA:.3f} s    τ2 = {TAU2:.3f} s    h1 = {H1:.4f} /s    c(等效环境温度) = {C:.0f} °C')
print('q(u) 断点表 [°C/s]:')
for u, q in zip(U_BP, QBP):
    print(f'  {u:5.0f}%  ->  {q:7.1f}   (K_loc={q/max(u,1e-9):.2f})' if u > 0 else f'  {u:5.0f}%  ->  {q:7.1f}')

print('\n各段开环复现 RMSE:')
for s in SEGS:
    y = open_loop(s['mv'], s['pv'][0], QBP, THETA, TAU2, H1, C)
    print(f"  {s['desc']:24s} RMSE = {np.sqrt(np.mean((y - s['pv'])**2)):6.2f} °C  "
          f"(段内 PV 摆幅 {np.ptp(s['pv']):.0f}°)")

# 噪声水平
resid_hold = None
s = SEGS[7]
y = open_loop(s['mv'], s['pv'][0], QBP, THETA, TAU2, H1, C)
SIG_N = float(np.std((s['pv'] - y) - pd.Series(s['pv'] - y).rolling(100, center=True, min_periods=1).mean().values))
print(f'\n测量噪声 σ_n ≈ {SIG_N:.3f} °C (450稳态段残差高频部分)')

with open('analysis/twin_params.json', 'w') as f:
    json.dump(dict(u_bp=U_BP.tolist(), q_bp=QBP.tolist(), theta=THETA,
                   tau2=TAU2, h1=H1, c=C, sig_n=SIG_N,
                   note='2026-07-13 两场真机数据全波形辨识; MV>80% 为外推'), f, indent=1)
print('saved analysis/twin_params.json')

# ------------------------------------------------ 闭环盲考
import importlib, sys
sys.path.insert(0, 'analysis')
import twin as twin_mod
importlib.reload(twin_mod)
tw = twin_mod.Twin('analysis/twin_params.json')

def ring_period(u_seg, dt=DT, lo_s=1.0, hi_s=8.0):
    a, b = int(lo_s/dt), min(int(hi_s/dt), len(u_seg))
    x = u_seg[a:b] - pd.Series(u_seg[a:b]).rolling(int(2/dt), center=True, min_periods=1).mean().values
    x = pd.Series(x).rolling(8, center=True, min_periods=1).mean().values
    if np.std(x) < 0.3:
        return np.nan
    m = len(x)
    ac = np.correlate(x, x, 'full')[m-1:]
    ac /= (ac[0] + 1e-12)
    for i in range(int(0.4/dt), len(ac)-1):
        if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0.2:
            return i*dt
    return np.nan

print('\n===== 闭环盲考 (孪生 vs 真机, 模型没见过闭环结构) =====')
print(f"{'工况':18s} {'指标':30s} {'孪生':>16s} {'真机':>16s}")
tests = [('S3 wc=8',  (3.4, 8.0, 24.0, 3.0, 23), (1.86, 1.9, 1.65)),
         ('S4 wc=2',  (3.4, 2.0, 24.0, 3.0, 23), (1.81, 2.3, 1.62)),
         ('S5 wc=0.9',(3.4, 0.9, 24.0, 3.0, 23), (3.34, 0.9, np.nan)),
         ('S2 旧参',   (10.0, 0.8, 40.0, 10.0, 20), (2.91, 3.2, np.nan))]
for name, prm, (r90m, ovm, Tm) in tests:
    yy, uu = tw.closed_loop_adrc(*prm, 100, 400, T=16, noise=0.0)
    r90, ov, stl = twin_mod.metrics(yy, 100, 400)
    Tr = ring_period(uu)
    print(f'{name:18s} rise90/超调/振铃T           '
          f'{r90:5.2f}s/{ov:4.1f}°/{Tr if not np.isnan(Tr) else 0:4.2f}s '
          f'{r90m:5.2f}s/{ovm:4.1f}°/{Tm if not np.isnan(Tm) else 0:4.2f}s')

# ------------------------------------------------ 在孪生上复核推荐参数
print('\n===== 孪生上复核 A+ 及升级档 =====')
for tag, prm in [('A+ wc2 wo9 wr3 nd28', (3.4, 2.0, 9.0, 3.0, 28)),
                 ('升级试探 wr3.5', (3.4, 2.0, 9.0, 3.5, 28)),
                 ('升级试探 wc2.5wr3.5', (3.4, 2.5, 9.0, 3.5, 28))]:
    yy, uu = tw.closed_loop_adrc(*prm, 100, 400, T=18, noise=0.0)
    r90, ov, stl = twin_mod.metrics(yy, 100, 400)
    print(f'  {tag:24s} rise90={r90:.2f}s 超调={ov:.2f}° settle±1={stl:.2f}s MV峰={uu.max():.0f}%')

# ------------------------------------------------ 图
fig, axes = plt.subplots(3, 3, figsize=(16, 10))
for k, s in enumerate(SEGS):
    ax = axes[k // 3][k % 3]
    tt = np.arange(len(s['pv'])) * DT
    y = open_loop(s['mv'], s['pv'][0], QBP, THETA, TAU2, H1, C)
    ax.plot(tt, s['pv'], 'k-', lw=1.1, label='实测 PV')
    ax.plot(tt, y, 'C1--', lw=1.1, label='孪生开环复现')
    ax2 = ax.twinx()
    ax2.plot(tt, s['mv'], color='#1d4ed8', lw=0.6, alpha=0.5)
    ax2.set_ylim(-2, 105)
    rmse = np.sqrt(np.mean((y - s['pv'])**2))
    ax.set_title(f"{s['desc']}  RMSE={rmse:.2f}°C", fontsize=9.5)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle('数字孪生全波形辨识: 灌入记录 MV, 开环复现 PV (蓝细线=MV)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('plots/10_数字孪生_辨识验证.png', dpi=110)
print('\nsaved plots/10_数字孪生_辨识验证.png')
