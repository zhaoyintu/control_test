#!/usr/bin/env python3
"""孪生 v3 拟合: τe=0.32(平坦,实测) + q 高段温度缩水 kq_hot
拟合目标: 复现 7-15 晚闭环谱系 (100→400 超调 2.5° / 200→440 超调 10.0°, 第3档参数)
机理: 控制器按静态表反查, 对象实际马力在高温缩水 -> z2 冲刺负账 -> 刹车吐账超调
数据依据: 7-16 高温标定 (τe 350~500°C 平坦 0.30~0.34; 90% 满幅 q 实测比表低 15~20%)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin import Twin

DT = 0.01


def cl_map(tv, wc, wo, wr, nd, tauf, mv_max, y0, svt, T=18., pre=2.):
    """开表闭环: 控制器 v空间(静态表), 对象 q_eff(u,y) 带高温缩水"""
    n_pre = int(pre / DT); N = n_pre + int(T / DT)
    nd_p = max(1, int(round(tv.theta / DT)))
    rng = np.random.default_rng(1)
    v_hi = float(tv.q_of(mv_max))
    q_need = tv.h1 * (y0 - tv.c)
    u_ss = float(np.interp(max(q_need, 0.0), tv.q_bp, tv.u_bp))
    v_ss = float(tv.q_of(u_ss))
    y = y0; qf = q_need; z1, z2 = y0, 0.0; v1, v2 = y0, 0.0
    buf = [v_ss] * (nd + 1); vf = v_ss
    uh = np.full(N + nd_p + 2, u_ss)
    yy = np.zeros(N)
    for i in range(N):
        svr = y0 if i < n_pre else svt
        ym = y + rng.normal(0, tv.sig_n)
        v1n = v1 + DT * v2
        v2n = v2 + DT * (-2 * wr * v2 - wr * wr * (v1 - svr))
        v1, v2 = v1n, v2n
        vd = buf[nd]
        if tauf:
            vf += DT * (vd - vf) / tauf
            vd = vf
        err = ym - z1
        z1 = z1 + DT * (z2 + vd + 2 * wo * err)
        z2 = z2 + DT * (wo * wo * err)
        vc = min(max(wc * (v1 - z1) - z2, 0.0), v_hi)
        u = min(max(float(np.interp(vc, tv.q_bp, tv.u_bp)), 0.0), mv_max)
        buf = [float(tv.q_of(u))] + buf[:-1]        # 控制器记账用静态表 (它不知道缩水)
        uh[i + nd_p] = u
        ud = float(uh[i])
        for _ in range(2):
            h = DT / 2
            qf += h * (tv.q_eff(ud, y) - qf) / max(tv.tau2, 1e-4)
            y += h * (qf - tv.h1 * (y - tv.c))
        yy[i] = y
    yy = yy[n_pre:]
    tt = np.arange(len(yy)) * DT
    reach = tt[np.argmax(yy >= svt)] if (yy >= svt).any() else np.inf
    ov = max(yy.max() - svt, 0)
    return reach, ov


def mkv3(kq):
    tv = Twin()
    tv.tau2 = 0.32
    tv.theta = 0.12
    tv.kq_hot = kq
    tv.kq_tref = 150.0
    return tv


R3 = dict(wc=1.5, wo=10., wr=3.5, nd=12, tauf=0.24, mv_max=60.)

if __name__ == '__main__':
    print('扫描 kq_hot (第3档配置, 复现目标: 400步 ov 2.5° / 440步 ov 10.0°):')
    best = None
    for kq in (0.0, 0.06, 0.09, 0.12, 0.15, 0.18, 0.22):
        tv = mkv3(kq)
        r4, o4 = cl_map(tv, **R3, y0=100., svt=400.)
        r44, o44 = cl_map(tv, **R3, y0=200., svt=440.)
        cost = abs(o4 - 2.5) + abs(o44 - 10.0)
        tag = ''
        if best is None or cost < best[0]:
            best = (cost, kq, r4, o4, r44, o44); tag = ' <--'
        print(f'  kq={kq:.2f}: 400步 {r4:.2f}s/{o4:4.1f}° | 440步 {r44:.2f}s/{o44:4.1f}°{tag}')
    _, KQ, r4, o4, r44, o44 = best
    print(f'\n最佳 kq_hot={KQ:.2f}: 复现 400步 {o4:.1f}°(实测2.5) / 440步 {o44:.1f}°(实测10.0)')
    tv = mkv3(KQ)
    r24, o24 = cl_map(tv, **R3, y0=200., svt=400.)
    print(f'旁证: 200→400 预测 ov {o24:.1f}° (7-16 冷炉实测 5.2~8.0°, 冷炉壁吸热未建模, 预期偏低)')
    q90 = tv.q_eff(90.0, 400.0)
    print(f'旁证: q_eff(90%, 400°C) = {q90:.0f} (7-16 满幅实测 271~299)')
