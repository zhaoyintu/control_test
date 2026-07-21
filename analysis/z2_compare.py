#!/usr/bin/env python3
"""z2 对照 + 配置反演 (2026-07-21)
A. 440 冲透组: 仿真 z2 四点摘要 vs 真实 (z2_forensics) -- 定位收油段第一个分叉;
B. 19:09 200→440 (标注 kd=1.0, 复刻 18.5% 存疑): 网格反演实跑配置。
用法: python3 analysis/z2_compare.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin import Twin
from g2_g3_pipeline import (load, observer, fit_w0, plant_step, q_true,
                            F_AM, F_PM, THETA)
from z2_forensics import replica

TAB = Twin()
DT = 0.01
G2R3 = dict(s_hi=1.080, te_hi=0.112, kq_p=0.35)     # 第三轮拟合 (f_split 已否决)


def sim_capture(path, t0, svt, kd, w1w2, wc=4.5, wo=10., wr=30., nd=12, tauf=0.24,
                mv_max=90., ckq=1.1, T=6.0):
    """闭环仿真 (同 g3_run) 但记录 (t, u, z2, v̂, y)"""
    s_hi, te_hi, kq_p = G2R3['s_hi'], G2R3['te_hi'], G2R3['kq_p']
    t, pv, sv, mv = load(path)
    i0 = np.searchsorted(t, t0)
    w1, w2 = w1w2
    j0 = np.searchsorted(t, t0 - 3.0)
    ttp = np.arange(t[j0], t0, DT)
    idx = np.clip(np.searchsorted(t, ttp, side='right') - 1, 0, len(t) - 1)
    upre = mv[idx]
    rng = np.random.default_rng(1)
    nd_p = max(1, int(round(THETA / DT)))
    yv = float(np.interp(t[j0], t, pv))
    qf = q_true(float(upre[0]), s_hi)
    z1, z2 = yv, 0.0
    buf = [float(np.interp(float(upre[0]), TAB.u_bp, TAB.q_bp))] * (nd + 1)
    vf = buf[0]
    uh = list(np.full(nd_p, float(upre[0])))
    for i in range(len(upre)):
        ym = yv + rng.normal(0, TAB.sig_n)
        u = float(upre[i])
        vd = buf[nd]
        if tauf:
            vf += DT * (vd - vf) / tauf
            vd = vf
        err = ym - z1
        z1 += DT * (z2 + vd + 2 * wo * err)
        z2 += DT * (wo * wo * err)
        va = float(np.interp(u, TAB.u_bp, TAB.q_bp))
        if ckq > 0 and ym > 315. and u > 30.:
            g = 1. - ckq * min(max((u - 30.) / 30., 0.), 1.) * (ym - 315.) / 100.
            va *= max(g, 0.4)
        buf = [va] + buf[:-1]
        uh.append(u)
        yv, w1, w2, qf = plant_step(yv, w1, w2, qf, float(uh[i]), s_hi, te_hi,
                                    kq_p=kq_p)
    v1, v2 = z1, 0.0
    N = int(T / DT)
    out = []
    for i in range(N):
        ym = yv + rng.normal(0, TAB.sig_n)
        v1n = v1 + DT * v2
        v2n = v2 + DT * (-2 * wr * v2 - wr * wr * (v1 - svt))
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
        u = min(max(float(np.interp(vc, TAB.q_bp, TAB.u_bp)), 0.0), mv_max)
        va = float(np.interp(u, TAB.u_bp, TAB.q_bp))
        if ckq > 0 and ym > 315. and u > 30.:
            g = 1. - ckq * min(max((u - 30.) / 30., 0.), 1.) * (ym - 315.) / 100.
            va *= max(g, 0.4)
        buf = [va] + buf[:-1]
        uh.append(u)
        yv, w1, w2, qf = plant_step(yv, w1, w2, qf, float(uh[len(upre) + i]),
                                    s_hi, te_hi, kq_p=kq_p)
        out.append((i * DT, u, z2, vd + z2, yv))
    return np.array(out)


def four_pts(r, svt, ucol=1, zcol=2, vcol=3, ycol=4):
    i_brk = np.argmax((r[:, ucol] < 80) & (r[:, 0] > 0.3))
    i_n30 = np.argmax(r[:, ycol] >= svt - 30)
    i_cx = np.argmax(r[:, ycol] >= svt)
    s = [f'0.5s: z2={r[np.argmax(r[:,0]>=0.5), zcol]:+.0f}',
         f'松油t={r[i_brk,0]:.2f} PV={r[i_brk,ycol]:.0f} z2={r[i_brk,zcol]:+.0f}',
         f'差30°t={r[i_n30,0]:.2f} z2={r[i_n30,zcol]:+.0f} v̂={r[i_n30,vcol]:.0f}']
    if r[i_cx, ycol] >= svt:
        s.append(f'触线t={r[i_cx,0]:.2f}')
    else:
        s.append('未触线')
    return ' | '.join(s)


if __name__ == '__main__':
    print('A. 440 冲透组: 仿真 vs 真实 四点摘要')
    obs_cache = {}
    for path, t0, kd, tag in [(F_AM, 53.4, 1.1, 'AM kd=1.1 (实测冲透1.59s)'),
                              (F_PM, 2685.5, 0.9, 'PM kd=0.9 (实测冲透1.42s)'),
                              (F_PM, 3128.3, 1.1, 'PM kd=1.1 (实测爬行3.33s)')]:
        if path not in obs_cache:
            w10, w20, _ = fit_w0(path)
            obs_cache[path] = observer(path, w10, w20)
        T_, w1t, w2t = obs_cache[path]
        w1w2 = (float(np.interp(t0, T_, w1t)), float(np.interp(t0, T_, w2t)))
        rs = sim_capture(path, t0, 440., kd, w1w2)
        rr = replica(path, t0, 440., kd)
        print(f'  [{tag}]')
        print(f'    真实: {four_pts(rr, 440., ucol=2, zcol=3, vcol=4, ycol=5)}')
        print(f'    仿真: {four_pts(rs, 440.)}')

    print('\nB. 19:09 200→440 配置反演 (复刻 RMSE%, 阶跃后 0~4s):')
    best = []
    for kd in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for wc in (3.0, 4.5, 6.0):
            for ckq in (0.0, 1.1):
                r = replica(F_PM, 2973.8, 440., kd, wc=wc, ckq=ckq)
                m = r[:, 0] <= 4.0
                rmse = np.sqrt(np.mean((r[m, 1] - r[m, 2]) ** 2))
                best.append((rmse, kd, wc, ckq))
    best.sort()
    for rmse, kd, wc, ckq in best[:6]:
        print(f'   kd={kd:.1f} wc={wc:.1f} ckq={ckq:.1f}: RMSE {rmse:.1f}%')
