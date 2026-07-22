#!/usr/bin/env python3
"""元件状态模型正式拟合 -- 7-22 元件测温场 (调节器 v2)

模型 (功率加权连续 c_e, 不是通断布尔 -- v1 失败教训):
    充: u>2% 时 dc_e = (1−c_e)/τ_h(u),  τ_h(u) = τ_h90·(90/u)   (功率越大充越快)
    放: u≤2% 时 dc_e = −c_e/τ_c
    元件指令功率 = q(u)·(1 + β·(1−c_e))
c_e 全场由记录 MV 链式积分 (不做走回判断), 四发干净脉冲联合拟 (q90_v2, β, τ_c, τ_h90)。
5s 发打在高温段 (287→422, 跨缩水区) 不参与冷冲拟合。

★ 定案 (2026-07-22):
  热基准单拟先定基线: θ_v2=0.06s (新调节器纯滞后仅旧 0.12 的一半!), τe=0.244 (与已知
  元件常数吻合 -- 元件未换的交叉验证), q90_v2=264~268 °C/s (≈0.82~0.88x 旧)。
  冷冲族 (θ=0.06 下): β=0.42 (耦合形式: 功率x boost 且 τe/boost -- 冷元件更猛且更快),
  τ_c=103s, 全部窗口 RMSE 3.1~4.1°。斜率级直读 β≈+18% 与耦合形式一致 (定义不同)。
  对 v2 控制卡的直接推论: nDeadSteps 12 -> 6; lrTauF 不变 0.24; aMapV x0.82 待 4 点复核。
用法: python3 analysis/fit_element_0722.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from g2_g3_pipeline import P_WALL

AMB = 29.7
THETA, TE = 0.06, 0.242   # θ=0.06: 新调节器纯滞后仅旧的一半 (热基准脉冲单拟定案)
DT = 0.01
F22 = os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260722-121109_124114.csv')

df = pd.read_csv(F22)
ts = pd.to_datetime(df.iloc[:, 0])
t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
pv = df.iloc[:, 1].to_numpy(float)
mv = df.iloc[:, 4].to_numpy(float)

# 0.05s 栅格的全场 MV (c_e 链式积分用) 与墙观测器
TG = np.arange(0.0, t[-1], 0.05)
MVG = mv[np.clip(np.searchsorted(t, TG, side='right') - 1, 0, len(t) - 1)]
PVG = np.interp(TG, t, pv)


def wall_traj():
    a1, b1, c1, d1, e1 = (P_WALL[k] for k in ('a1', 'b1', 'c1', 'd1', 'e1'))
    n = len(TG)
    w1 = np.empty(n); w2 = np.empty(n)
    w1[0] = w2[0] = AMB
    for i in range(1, n):
        w1[i] = w1[i-1] + 0.05 * (b1 * (PVG[i-1] - w1[i-1]) - c1 * (w1[i-1] - w2[i-1]))
        w2[i] = w2[i-1] + 0.05 * (d1 * (w1[i-1] - w2[i-1]) - e1 * (w2[i-1] - AMB))
    return w1, w2


W1T, W2T = wall_traj()


def ce_traj(tau_h90, tau_c):
    n = len(TG)
    ce = np.empty(n); ce[0] = 0.0        # 静置 13.8min 起点: 全冷 (真已知)
    for i in range(1, n):
        u = MVG[i-1]
        if u > 2.0:
            th = tau_h90 * 90.0 / max(u, 5.0)
            ce[i] = ce[i-1] + 0.05 * (1.0 - ce[i-1]) / th
        else:
            ce[i] = ce[i-1] - 0.05 * ce[i-1] / tau_c
        ce[i] = min(max(ce[i], 0.0), 1.0)
    return ce

PULSES = [(893.0, '热基准'), (930.0, '断电30s'), (1051.0, '断电120s'), (1652.0, '断电600s')]


def sim_pulse(t0, q90, beta, ce0, w10, w20):
    """脉冲窗 −0.3~+2.2s: 记录 MV 回放 (低段功率按 90% 比例缩放近似)"""
    a1, b1, c1, d1, e1 = (P_WALL[k] for k in ('a1', 'b1', 'c1', 'd1', 'e1'))
    i0 = np.searchsorted(t, t0 - 0.3)
    tt = np.arange(t[i0], t[i0] + 2.5, DT)
    idx = np.clip(np.searchsorted(t, tt, side='right') - 1, 0, len(t) - 1)
    useq = mv[idx]; pref = np.interp(tt, t, pv)
    nd = max(1, int(round(THETA / DT)))
    y, w1, w2 = pref[0], w10, w20
    qf, ce = 0.0, ce0
    sim = np.empty(len(tt))
    for i in range(len(tt)):
        ud = float(useq[max(0, i - nd)])
        if ud > 2.0:
            ce = min(ce + DT * (1.0 - ce) / (0.35 * 90.0 / max(ud, 5.0)), 1.0)
        boost = 1.0 + beta * (1.0 - ce)
        q_cmd = q90 * (ud / 90.0) * boost
        qf += DT * (q_cmd - qf) / (TE / boost)     # 冷元件: 更猛且更快 (同一物理根源)
        y += DT * (qf - a1 * (y - w1))
        w1 += DT * (b1 * (y - w1) - c1 * (w1 - w2))
        w2 += DT * (d1 * (w1 - w2) - e1 * (w2 - AMB))
        sim[i] = y
    return sim, pref


def resid(x):
    q90, beta, tau_c = x
    ce = ce_traj(0.35, tau_c)
    r = []
    for t0, nm in PULSES:
        ce0 = float(np.interp(t0 - 0.3, TG, ce))
        w10 = float(np.interp(t0 - 0.3, TG, W1T))
        w20 = float(np.interp(t0 - 0.3, TG, W2T))
        sim, pref = sim_pulse(t0, q90, beta, ce0, w10, w20)
        r.append((sim - pref)[::5])
    return np.concatenate(r)


if __name__ == '__main__':
    res = least_squares(resid, [280., 0.18, 180.], bounds=([200., 0.0, 30.], [360., 0.5, 900.]),
                        diff_step=0.05)
    q90, beta, tau_c = res.x
    rmse = np.sqrt(np.mean(res.fun ** 2))
    print(f'拟合: q90_v2={q90:.0f} °C/s  β={beta:.3f}  τ_c={tau_c:.0f}s  RMSE={rmse:.2f}°')
    ce = ce_traj(0.35, tau_c)
    for t0, nm in PULSES:
        ce0 = float(np.interp(t0 - 0.3, TG, ce))
        sim, pref = sim_pulse(t0, q90, beta, ce0, *[float(np.interp(t0 - 0.3, TG, w)) for w in (W1T, W2T)])
        e = np.sqrt(np.mean((sim - pref) ** 2))
        print(f'  {nm}: c_e0={ce0:.2f}  窗 RMSE={e:.2f}°')
    print(f'对照手算: 热 249 / 全冷 294 -> β≈{294/249-1:.2f}; 新旧调节器 90% 比值 ≈ {q90/305:.2f} (旧暖态~305)')
