#!/usr/bin/env python3
"""z2 取证: 控制器复刻喂真机 PV/SV/MV -- 不依赖对象模型的两项检验 (2026-07-21)

① 复刻保真度: 用假设配置 (wc=4.5 wo=10 wr=30 nd=12 tauf=0.24 ckq=1.1 + 标注 kd)
   逐拍算 u_pred, 与记录 MV 对比。RMSE 大 => 机上配置与假设不符 (回放辨识老套路)。
② 真实 z2 轨迹: ESO 由实测 PV 驱动、FIFO 按实际记录 MV 记账 => z2(t) 是"机上
   控制器当时心里的账"。与 G3 仿真的 z2 对照, 定位 440 冲透组分歧。

用法: python3 analysis/z2_forensics.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin import Twin
from g2_g3_pipeline import load, TESTS, F_AM, F_PM

TAB = Twin()
DT = 0.01


def replica(path, t0, svt, kd, wc=4.5, wo=10., wr=30., nd=12, tauf=0.24,
            mv_max=90., ckq=1.1, pre=5.0, T=5.0):
    t, pv, sv, mv = load(path)
    i0 = np.searchsorted(t, t0 - pre)
    i1 = np.searchsorted(t, t0 + T)
    n0 = np.searchsorted(t, t0)
    z1, z2 = float(pv[i0]), 0.0
    v1, v2 = float(pv[i0]), 0.0
    va0 = float(np.interp(float(mv[i0]), TAB.u_bp, TAB.q_bp))
    buf = [va0] * (nd + 1)
    vf = va0
    out = []
    for i in range(i0, i1):
        ym = float(pv[i])
        svr = float(sv[i])
        v1n = v1 + DT * v2
        v2n = v2 + DT * (-2 * wr * v2 - wr * wr * (v1 - svr))
        v1, v2 = v1n, v2n
        vd = buf[nd]
        if tauf:
            vf += DT * (vd - vf) / tauf
            vd = vf
        err = ym - z1
        z1 += DT * (z2 + vd + 2 * wo * err)
        z2 += DT * (wo * wo * err)
        vc = wc * (v1 - z1) - z2 - kd * (vd + z2)
        vc = min(max(vc, 0.0), float(np.interp(mv_max, TAB.u_bp, TAB.q_bp)))
        u_pred = min(max(float(np.interp(vc, TAB.q_bp, TAB.u_bp)), 0.0), mv_max)
        # 记账用"实际发生"的 MV (机上如此)
        u_act = float(mv[i])
        va = float(np.interp(u_act, TAB.u_bp, TAB.q_bp))
        if ckq > 0 and ym > 315. and u_act > 30.:
            g = 1. - ckq * min(max((u_act - 30.) / 30., 0.), 1.) * (ym - 315.) / 100.
            va *= max(g, 0.4)
        buf = [va] + buf[:-1]
        if i >= n0:
            out.append((t[i] - t0, u_pred, u_act, z2, vd + z2, ym))
    return np.array(out)


if __name__ == '__main__':
    print('① 复刻保真度 (阶跃后 0~4s, u_pred vs 记录 MV):')
    print(f'{"工况":9s} {"kd":>4s} {"RMSE%":>6s} {"饱和段外RMSE%":>10s}')
    bads = []
    for path, t0, svt, kd, mr, mo in TESTS:
        r = replica(path, t0, svt, kd)
        m = r[:, 0] <= 4.0
        err = r[m, 1] - r[m, 2]
        rmse = np.sqrt(np.mean(err ** 2))
        ns = m & (r[:, 2] < 88) & (r[:, 2] > 2)     # 非饱和段
        rmse_ns = np.sqrt(np.mean((r[ns, 1] - r[ns, 2]) ** 2)) if ns.any() else np.nan
        flag = ''
        if rmse > 12:
            flag = '  <-- 配置存疑'
            bads.append((path, t0, svt, kd))
        print(f'{r[0,5]:.0f}→{svt:.0f} {kd:4.2f} {rmse:6.1f} {rmse_ns:10.1f}{flag}')

    print('\n② 440 冲透组真实 z2 (对照仿真为何过刹):')
    for path, t0, svt, kd in [(F_AM, 53.4, 440., 1.1), (F_PM, 2685.5, 440., 0.9),
                              (F_PM, 2973.8, 440., 1.0), (F_PM, 3128.3, 440., 1.1)]:
        r = replica(path, t0, svt, kd)
        # 关键时刻: 0.5s / MV 首次跌破 80 / PV 首过 svt-30 / svt
        i_brk = np.argmax((r[:, 2] < 80) & (r[:, 0] > 0.3))
        i_n30 = np.argmax(r[:, 5] >= svt - 30)
        i_cx = np.argmax(r[:, 5] >= svt)
        pts = [('0.5s', np.argmax(r[:, 0] >= 0.5)), (f'松油t={r[i_brk,0]:.2f}', i_brk),
               (f'差30°t={r[i_n30,0]:.2f}', i_n30)]
        if r[i_cx, 5] >= svt:
            pts.append((f'触线t={r[i_cx,0]:.2f}', i_cx))
        s = ' | '.join(f'{nm}: z2={r[i,3]:+.0f} v̂={r[i,4]:.0f}' for nm, i in pts)
        print(f'  {r[0,5]:.0f}→{svt:.0f} kd={kd}: {s}')
