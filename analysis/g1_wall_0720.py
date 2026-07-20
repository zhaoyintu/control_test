#!/usr/bin/env python3
"""G1: 慢墙参数辨识 -- 7-20 采集场 (AIC9_DATA-20260720-201947_215552.csv)

本场的两件礼物: ①开头 40min 炉子躺在室温 -> amb=28.5°C 直接测量,
且 t=40.2min 起跑时墙温精确已知 (=amb); ②三段 10min 级长降温 + 冷/热墙 400 锚点。

方法 (解开 7-19 的死结): 观测器-拟合迭代自洽 --
  观测器 dw = b_yw*(PV−w) − b_wa*(w−amb) 由实测 PV 驱动, w(0)=amb (真已知);
  各降温段 w0 一律取观测器值 (不给自由度!), 拟 (a_yw, b_yw, b_wa);
  新参数 -> 重跑观测器 -> 再拟, 至收敛。收敛 = G1 过关的自洽性本身。

G1 过关标准: 降温段 RMSE ≤4° (w0 无自由度) 且 保温锚点交叉验证
  a_yw*(y−w_obs) ≈ s_lo*q_tab(u_hold) 一致 ±15%。

★ 状态 (2026-07-20 第一轮): 未过 -- 且给出结构性结论:
  单层线性墙在"观测器自洽"约束下不收敛 (迭代震荡, RMSE 6~24° 跳动,
  a_yw 在 0.036~0.090 之间打摆)。物理解读: 早段降温要求强耦合 (~0.09),
  400→58 深尾段要求墙贴腔慢放 -- 单层无法兼得。7-19 拟合的 RMSE 1.82° 是
  "每段 w0 自由"给出的假好 (自由度吸收了结构误差), 本场 w0 被真值锚定后露馅。
  下一结构: 两层墙 (炉衬快层 + 炉体慢层), 物理对应 refractory liner + shell;
  数据无需再补 -- 本场覆盖已足 (amb 实测 28.4, 冷起跑墙态已知, 3 段长降温,
  冷/热墙 400 锚点, 4 次 440 大信号往返)。

用法: python3 analysis/g1_wall_0720.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin import Twin

F20 = os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260720-201947_215552.csv')
S_LO = 0.492
TAB = Twin()

df = pd.read_csv(F20)
ts = pd.to_datetime(df.iloc[:, 0])
t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
pv = df.iloc[:, 1].to_numpy(float)
mv = df.iloc[:, 4].to_numpy(float)

AMB = float(pv[t < 2000].mean())

# ---- 降温段 (跳过前 1.5s 元件余热), 0.5s 重采样 ----
lo = mv <= 0.5
d = np.diff(lo.astype(int)); s_ = np.where(d == 1)[0] + 1; e_ = np.where(d == -1)[0] + 1
if lo[0]: s_ = np.r_[0, s_]
if lo[-1]: e_ = np.r_[e_, len(lo)]
falls = []
for a, b in zip(s_, e_):
    if t[b-1] - t[a] < 90 or pv[a] < 150:
        continue
    tt = np.arange(t[a] + 1.5, t[b-1], 0.5)
    falls.append((t[a] + 1.5, tt - tt[0], np.interp(tt, t, pv)))
print(f'amb = {AMB:.1f}°C (前33min 直接测量);  {len(falls)} 段降温 (90s~853s)')

# ---- 0.5s 栅格的观测器与拟合 ----
T5 = np.arange(0, t[-1], 0.5)
PV5 = np.interp(T5, t, pv)


def observer(byw, bwa):
    w = np.empty(len(T5)); w[0] = AMB
    for i in range(1, len(T5)):
        w[i] = w[i-1] + 0.5 * (byw * (PV5[i-1] - w[i-1]) - bwa * (w[i-1] - AMB))
    return w


def sim_fall(ayw, byw, bwa, w0, tt, y0):
    y, w = y0, w0
    out = np.empty(len(tt))
    for i in range(len(tt)):
        out[i] = y
        y += 0.5 * (-ayw * (y - w))
        w += 0.5 * (byw * (y - w) - bwa * (w - AMB))
    return out


p = np.array([0.0836, 0.0042, 0.006])      # a_yw, b_yw, b_wa 初值 (7-19 拟合)
for it in range(5):
    wtraj = observer(p[1], p[2])
    w0s = [float(np.interp(t0, T5, wtraj)) for t0, _, _ in falls]

    def resid(x):
        r = []
        for (t0, tt, yy), w0 in zip(falls, w0s):
            r.append(sim_fall(x[0], x[1], x[2], w0, tt, yy[0]) - yy)
        return np.concatenate(r)

    res = least_squares(resid, p, bounds=([0.01, 1e-4, 1e-5], [0.5, 0.1, 0.05]), xtol=1e-10)
    rmse = np.sqrt(np.mean(res.fun ** 2))
    dp = np.abs(res.x - p) / p
    p = res.x
    print(f'  迭代{it+1}: a_yw={p[0]:.4f} b_yw={p[1]:.5f} b_wa={p[2]:.5f} '
          f'RMSE={rmse:.2f}° 参数变化 {dp.max()*100:.1f}%')
    if dp.max() < 0.01:
        break

wtraj = observer(p[1], p[2])
print(f'\n墙温轨迹节点: 起跑(40min)={np.interp(2412, T5, wtraj):.0f} '
      f'首段长冷前(41.7min)={np.interp(2503, T5, wtraj):.0f} '
      f'440往返后(72.3min)={np.interp(4338, T5, wtraj):.0f} '
      f'尾段400锚(90min)={np.interp(5400, T5, wtraj):.0f}')

# ---- 保温锚点交叉验证 ----
print('\n保温锚点交叉验证 (预测保温功率 a_yw·(y−w) vs 诚实表 s_lo·q_tab(u_hold)):')
anchors = [('冷墙 400', 2440, 2495, 400.), ('440 首保', 3270, 3310, 440.),
           ('热墙 400', 5405, 5455, 400.)]
for nm, ta, tb, T_ in anchors:
    i0 = np.searchsorted(t, ta); i1 = np.searchsorted(t, tb)
    u_h = mv[i0:i1].mean()
    w_h = float(np.interp((ta + tb) / 2, T5, wtraj))
    q_pred = p[0] * (T_ - w_h)
    q_meas = S_LO * float(np.interp(u_h, TAB.u_bp, TAB.q_bp))
    print(f'  {nm}: 墙={w_h:.0f}°C  预测 {q_pred:.1f} vs 表 {q_meas:.1f} °C/s '
          f'({(q_pred/q_meas-1)*100:+.0f}%)  [保温MV {u_h:.1f}%]')
