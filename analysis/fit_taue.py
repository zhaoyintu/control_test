#!/usr/bin/env python3
"""高温 τe(T) 标定判读: 自动找手动 bump, 对每拍做整段拟合, 分离 (θ, τe, q∞),
按温度分组输出 τe(T) —— 用于 350~430°C 高温动态标定 (元件惯性随温度的变化)。

用法:
    python3 analysis/fit_taue.py <AIC9_DATA-xxx.csv>

判定 bump: MV 单拍跳变 ≥3% 且之后 ≥0.35s 保持在新值 ±1% (上跳/下跳都收)。
每拍拟合模型:  qf 以 τe 一阶趋向 q∞ (跳变后延迟 θ 生效),  y' = qf − h1·(y−c)
跳变前基线拟合趋势线 (漂移容忍, 同 theta_from_bump), qpre = h1·(T0−c) + 漂移速率。
h1/c 取自 twin_params.json。
"""
import sys
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

DT = 0.01
JUMP_MIN = 3.0
HOLD_S = 0.35
HOLD_TOL = 1.0
TREND_S = 2.0
TREND_GAP_S = 0.1
DRIFT_MAX = 2.0

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'twin_params.json')) as f:
    _tw = json.load(f)
H1, C = _tw['h1'], _tw['c']


def simulate(qinf, tau_e, theta, n, y0, qpre):
    y = np.empty(n); qf = qpre; yy = y0
    for i in range(n):
        tgt = qinf if i * DT >= theta else qpre
        qf += DT * (tgt - qf) / max(tau_e, 0.02)
        yy += DT * (qf - H1 * (yy - C))
        y[i] = yy
    return y


def main(path):
    df = pd.read_csv(path)
    df.columns = ['time', 'PV1', 'PV2', 'SV', 'MV'][: len(df.columns)]
    traw = (pd.to_datetime(df['time']) - pd.to_datetime(df['time'].iloc[0])).dt.total_seconds().values
    tg = np.arange(0, traw[-1], DT)
    pv = np.interp(tg, traw, df['PV1'].values.astype(float))
    mv = df['MV'].values.astype(float)[np.clip(np.searchsorted(traw, tg, side='right') - 1, 0, None)]
    n = len(tg); t = tg
    hold_n = int(HOLD_S / DT)
    trend_n = int(TREND_S / DT)
    tgap_n = int(TREND_GAP_S / DT)

    dmv = np.diff(mv)
    cand = np.where(np.abs(dmv) >= JUMP_MIN)[0] + 1
    jumps, last = [], -10 * hold_n
    for i0 in cand:
        if i0 - last < hold_n or i0 + hold_n >= n or i0 < trend_n + tgap_n + 10:
            continue
        if np.max(np.abs(mv[i0: i0 + hold_n] - mv[i0])) > HOLD_TOL:
            continue
        jumps.append(i0); last = i0
    if not jumps:
        print('没找到手动 bump。')
        return

    print(f'找到 {len(jumps)} 次 bump (h1={H1:.4f}, c={C:.0f}; 整段拟合分离 θ/τe/q∞):\n')
    print(f"{'#':>2s} {'t0[s]':>8s} {'T0[°C]':>7s} {'MV: 前->后':>13s} {'方向':>4s} {'保持[s]':>7s} "
          f"{'θ[s]':>6s} {'τe[s]':>6s} {'q∞':>7s}  备注")
    res = []
    for k, i0 in enumerate(jumps):
        k1 = i0 - tgap_n; k0 = k1 - trend_n
        Ab = np.column_stack([t[k0:k1] - t[i0], np.ones(k1 - k0)])
        coef = np.linalg.lstsq(Ab, pv[k0:k1], rcond=None)[0]
        d, T0 = float(coef[0]), float(coef[1])
        fit_rms = float(np.sqrt(np.mean((pv[k0:k1] - Ab @ coef) ** 2)))
        u_pre = float(np.median(mv[i0 - 50:i0 - 2])); u_post = float(mv[i0 + 2])
        du = u_post - u_pre
        void = None
        if abs(d) > DRIFT_MAX: void = f'作废:跳变前漂移 {d:+.1f}°C/s'
        elif fit_rms > 0.15: void = '作废:基线弯折'
        elif T0 - C < 20: void = '作废:温度太低, 散热模型不适用'
        if void:
            print(f'{k+1:2d} {t[i0]:8.1f} {T0:7.1f} {u_pre:5.1f}->{u_post:6.1f} {"—":>4s} {"—":>7s} '
                  f'{"—":>6s} {"—":>6s} {"—":>7s}  {void}')
            continue
        hold_end = i0
        while hold_end < n - 1 and abs(mv[hold_end] - u_post) <= HOLD_TOL:
            hold_end += 1
        hold_s = (hold_end - i0) * DT
        nf = min(int(min(hold_s, 4.0) / DT) + 12, n - i0)   # 拟合窗封顶 4s (停机长尾不可用)
        qpre = H1 * (T0 - C) + d
        seg = pv[i0:i0 + nf]

        # θ 对残差是阶梯函数 (梯度为零), 必须网格扫描; 内层拟合 (q∞, τe)
        q0 = max(qpre + (30.0 if du > 0 else -min(15.0, qpre * 0.5)), 1.0)
        best = None
        for th in np.arange(0.04, 0.32, 0.02):
            def r(p):
                return simulate(p[0], p[1], th, nf, T0, qpre) - seg
            sol = least_squares(r, [q0, 0.3], bounds=([0, 0.05], [600, 1.5]))
            rms = float(np.sqrt(np.mean(sol.fun ** 2)))
            if best is None or rms < best[0]:
                best = (rms, float(th), float(sol.x[0]), float(sol.x[1]))
        rms, th, qinf, taue = best
        note = []
        if rms > 1.0: note.append(f'残差大({rms:.2f})')
        if min(hold_s, 4.0) < 1.0: note.append('保持短,τe/q∞联合不确定')
        if taue > 1.4 or taue < 0.06: note.append('τe触界,弃用')
        ok = rms <= 1.0 and 0.06 <= taue <= 1.4
        if ok:
            res.append((T0, du, qinf, taue, th))
        print(f'{k+1:2d} {t[i0]:8.1f} {T0:7.1f} {u_pre:5.1f}->{u_post:6.1f} '
              f'{"上" if du>0 else "下":>4s} {hold_s:7.2f} {th:6.2f} {taue:6.2f} {qinf:7.1f}  {"; ".join(note)}')

    if not res:
        return
    # ---- τe(T) 汇总: 按温度 ±40°C 聚类 ----
    res.sort()
    print('\nτe(T) 汇总 (温度分组中位数):')
    print(f"{'T区间':>12s} {'n':>3s} {'τe上行':>7s} {'τe下行':>7s}")
    i = 0
    pts = []
    while i < len(res):
        j = i
        while j + 1 < len(res) and res[j + 1][0] - res[i][0] <= 40.0:
            j += 1
        grp = res[i:j + 1]
        Tm = float(np.median([g[0] for g in grp]))
        up = [g[3] for g in grp if g[1] > 0]
        dn = [g[3] for g in grp if g[1] < 0]
        us = f'{np.median(up):7.2f}' if up else f"{'—':>7s}"
        ds = f'{np.median(dn):7.2f}' if dn else f"{'—':>7s}"
        print(f'{res[i][0]:5.0f}~{res[j][0]:4.0f}° {len(grp):3d} {us} {ds}')
        pts.append((Tm, float(np.median([g[3] for g in grp]))))
        i = j + 1
    if len(pts) >= 2:
        T = np.array([p[0] for p in pts]); TE = np.array([p[1] for p in pts])
        A = np.column_stack([np.maximum(T - 300.0, 0.0) / 100.0, np.ones(len(T))])
        sl, b0_ = np.linalg.lstsq(A, TE, rcond=None)[0]
        print(f'\n==> τe(T) ≈ {b0_:.2f} + {sl:.2f}·max(T-300,0)/100   '
              f'(粗拟合, 供孪生 v3 与高温接近策略设计用)')
        print('    下一步: 把本结果发回分析侧 -> 孪生 v3 -> 在 v3 上比较 wr调度 vs 预整形')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
