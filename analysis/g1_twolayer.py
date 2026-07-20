#!/usr/bin/env python3
"""G1 (第二轮): 两层墙联合辨识 -- 炉衬(快) + 炉体(慢), 7-20 采集场

结构 (单层被 7-20 长降温数据否决后的升级):
    腔体:  dy/dt  = qf − a1·(y − w1)
    炉衬:  dw1/dt = b1·(y − w1) − c1·(w1 − w2)
    炉体:  dw2/dt = d1·(w1 − w2) − e1·(w2 − amb),  amb = 28.4 (实测)

自洽构造 (关键改进): 观测器 (由实测 PV 驱动, w 状态从已知冷起点 amb 积分)
放进残差函数内部 -- 每组候选参数先跑全场观测器取各降温段初始 (w1,w2),
再仿真降温段算残差。降温段零自由度; 收敛即自洽。

G1 过关: 降温段 RMSE ≤4°; 保温锚点留作 holdout 验证 (±15%):
    冷墙 400 @17.3% -> 真功率 s_lo·q_tab = 18.4 °C/s
    热墙 400 @16.5% -> 17.3 °C/s

用法: python3 analysis/g1_twolayer.py
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
AMB = 28.4
TAB = Twin()

df = pd.read_csv(F20)
ts = pd.to_datetime(df.iloc[:, 0])
t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
pv = df.iloc[:, 1].to_numpy(float)
mv = df.iloc[:, 4].to_numpy(float)

DTO = 0.25
T_ = np.arange(0.0, t[-1], DTO)
PV_ = np.interp(T_, t, pv)

# 降温段
lo = mv <= 0.5
d = np.diff(lo.astype(int)); s_ = np.where(d == 1)[0] + 1; e_ = np.where(d == -1)[0] + 1
if lo[0]: s_ = np.r_[0, s_]
if lo[-1]: e_ = np.r_[e_, len(lo)]
FALLS = []
for a, b in zip(s_, e_):
    if t[b-1] - t[a] < 90 or pv[a] < 150:
        continue
    tt = np.arange(t[a] + 1.5, t[b-1] - 1.0, 0.5)
    FALLS.append((t[a] + 1.5, tt - tt[0], np.interp(tt, t, pv)))


def observer(p):
    """全场观测器: PV 驱动, 冷起点 (w1=w2=amb) 已知"""
    a1, b1, c1, d1, e1 = p
    n = len(T_)
    w1 = np.empty(n); w2 = np.empty(n)
    w1[0] = w2[0] = AMB
    for i in range(1, n):
        dw1 = b1 * (PV_[i-1] - w1[i-1]) - c1 * (w1[i-1] - w2[i-1])
        dw2 = d1 * (w1[i-1] - w2[i-1]) - e1 * (w2[i-1] - AMB)
        w1[i] = w1[i-1] + DTO * dw1
        w2[i] = w2[i-1] + DTO * dw2
    return w1, w2


def sim_fall(p, y0, w10, w20, n):
    a1, b1, c1, d1, e1 = p
    y, w1, w2 = y0, w10, w20
    out = np.empty(n)
    for i in range(n):
        out[i] = y
        dy = -a1 * (y - w1)
        dw1 = b1 * (y - w1) - c1 * (w1 - w2)
        dw2 = d1 * (w1 - w2) - e1 * (w2 - AMB)
        y += 0.5 * dy; w1 += 0.5 * dw1; w2 += 0.5 * dw2
    return out


def resid(x):
    p = np.exp(x)                      # 对数参数化, 防负值且尺度均衡
    w1t, w2t = observer(p)
    r = []
    for t0, tt, yy in FALLS:
        w10 = float(np.interp(t0, T_, w1t))
        w20 = float(np.interp(t0, T_, w2t))
        r.append(sim_fall(p, yy[0], w10, w20, len(tt)) - yy)
    return np.concatenate(r)


if __name__ == '__main__':
    print(f'{len(FALLS)} 段降温; amb={AMB}; 观测器冷起点已知')
    best = None
    for x0 in ([0.09, 0.03, 0.004, 0.003, 0.0008],
               [0.12, 0.06, 0.010, 0.002, 0.0015],
               [0.06, 0.015, 0.002, 0.005, 0.0005]):
        res = least_squares(resid, np.log(x0), xtol=1e-11, ftol=1e-11, max_nfev=400)
        rmse = np.sqrt(np.mean(res.fun ** 2))
        if best is None or rmse < best[0]:
            best = (rmse, np.exp(res.x))
        print(f'  起点 {x0} -> RMSE {rmse:.2f}°')
    rmse, p = best
    a1, b1, c1, d1, e1 = p
    print(f'\n最优: RMSE={rmse:.2f}°  a1={a1:.4f} b1={b1:.4f} c1={c1:.5f} d1={d1:.5f} e1={e1:.6f}')
    print(f'物理读数: 腔→衬 τ={1/a1:.0f}s | 衬升温速率尺度 b1/a1={b1/a1:.2f} (热容比~{a1/b1:.1f}x)'
          f' | 衬→体 τ={1/c1:.0f}s | 体时间常数 ~{1/(d1+e1)/60:.0f}min')

    w1t, w2t = observer(p)
    print('\n墙温轨迹 (炉衬/炉体):')
    for lab, tm in [('起跑 40min', 2412), ('冷锚中点 41.2min', 2472), ('长冷①末 53min', 3180),
                    ('440×4 后 72.3min', 4338), ('热锚中点 90.4min', 5424), ('尾末 96min', 5760)]:
        print(f'  {lab}: 衬={np.interp(tm, T_, w1t):.0f}°C 体={np.interp(tm, T_, w2t):.0f}°C')

    print('\nHoldout 锚点验证 (不在拟合内):')
    for nm, tm, u_h in [('冷墙 400', 2472, 17.3), ('热墙 400', 5424, 16.5)]:
        w1h = float(np.interp(tm, T_, w1t))
        q_pred = a1 * (400.0 - w1h)
        q_meas = S_LO * float(np.interp(u_h, TAB.u_bp, TAB.q_bp))
        print(f'  {nm}: 衬温={w1h:.0f} -> 预测 {q_pred:.1f} vs 实测 {q_meas:.1f} °C/s ({(q_pred/q_meas-1)*100:+.0f}%)')
