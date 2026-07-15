#!/usr/bin/env python3
"""对 7-15 session 每个上跳 bump 做整段拟合: y' = qf - h1*(y-c), qf 以 τe 趋向 q∞
分离出静态马力 q∞ (进表) / 元件惯性 τe (进孪生 τ2) / 纯滞后 θ"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

DT = 0.01
H1, C = 0.1353, 127.0
df = pd.read_csv('/home/yiz/workspace/src/control_test/AIC9_DATA-20260715-000131_004800.csv')
df.columns = ['time', 'PV1', 'PV2', 'SV', 'MV']
traw = (pd.to_datetime(df['time']) - pd.to_datetime(df['time'].iloc[0])).dt.total_seconds().values
tg = np.arange(0, traw[-1], DT)
pv = np.interp(tg, traw, df['PV1'].values.astype(float))
mv = df['MV'].values.astype(float)[np.clip(np.searchsorted(traw, tg, side='right') - 1, 0, None)]

# (跳变时刻[s], 电平, 保持[s])  -- 取自判读输出, 只取上跳
BEATS = [(226.1, 22.2, 2.0), (311.2, 21.8, 2.0), (405.3, 20.4, 2.0), (486.8, 19.9, 2.0),
         (1162.5, 22.0, 2.0), (1216.0, 22.0, 2.0), (1267.4, 22.0, 2.0), (1341.8, 22.0, 2.0),
         (1395.4, 22.0, 2.0), (1434.4, 22.0, 2.0),
         (1840.4, 30.0, 2.0), (1940.0, 40.0, 1.3),
         (2050.0, 55.0, 0.7), (2176.0, 70.0, 0.5), (2333.2, 85.0, 0.5), (2483.1, 100.0, 0.4)]


def simulate(qinf, tau_e, theta, t0i, n, y0, qpre):
    y = np.empty(n); qf = qpre; yy = y0
    for i in range(n):
        te = i * DT - theta
        target = qinf if te >= 0 else qpre
        qf += DT * (target - qf) / max(tau_e, 0.02)
        yy += DT * (qf - H1 * (yy - C))
        y[i] = yy
    return y


def fit_beat(tc, hold, fix=None):
    i0 = int(round(tc / DT))
    n = int((hold + 0.12) / DT)              # 到保持结束+0.12s (下跳影响还没到)
    y0 = float(np.mean(pv[i0 - 100:i0 - 10]))
    qpre = H1 * (y0 - C)
    seg = pv[i0:i0 + n]

    if fix is None:
        def r(p): return simulate(p[0], p[1], p[2], i0, n, y0, qpre) - seg
        p = least_squares(r, [qpre + 30, 0.25, 0.12],
                          bounds=([0, 0.02, 0.0], [500, 1.5, 0.5])).x
        return p[0], p[1], p[2]
    else:
        te, th = fix
        def r(p): return simulate(p[0], te, th, i0, n, y0, qpre) - seg
        q = least_squares(r, [qpre + 100], bounds=([0], [600])).x[0]
        return q, te, th

print('--- 第一遍: 长保持 (>=1.3s) 自由拟合 (q∞, τe, θ) ---')
long_res = []
for tc, u, hold in BEATS:
    if hold >= 1.3:
        q, te, th = fit_beat(tc, hold)
        long_res.append((u, q, te, th))
        print(f'  u={u:5.1f}%  q∞={q:6.1f}  τe={te:.3f}s  θ={th:.3f}s')
tes = np.median([r[2] for r in long_res]); ths = np.median([r[3] for r in long_res])
print(f'长保持中位: τe={tes:.3f}s  θ={ths:.3f}s')

print('\n--- 第二遍: 短保持电平固定 (τe, θ) 只拟合 q∞ ---')
allpts = [(r[0], r[1]) for r in long_res]
for tc, u, hold in BEATS:
    if hold < 1.3:
        q, _, _ = fit_beat(tc, hold, fix=(tes, ths))
        allpts.append((u, q))
        print(f'  u={u:5.1f}%  q∞={q:6.1f}')

print('\n--- 汇总 (电平聚类中位) ---')
allpts.sort()
us = [p[0] for p in allpts]; qs = [p[1] for p in allpts]
i = 0; out = []
while i < len(us):
    j = i
    while j + 1 < len(us) and us[j + 1] - us[i] <= 2.5:
        j += 1
    out.append((float(np.median(us[i:j + 1])), float(np.median(qs[i:j + 1]))))
    i = j + 1
for u, q in out:
    print(f'  u={u:5.1f}%  q∞={q:6.1f} °C/s')
