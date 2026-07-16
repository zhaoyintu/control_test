#!/usr/bin/env python3
"""控制器记账温度修正 [8] 的孪生 v3 验证 (2026-07-16)
设计: 只修 FIFO 记账 (lrVApplied × g(u,PV)), 不修反查 -- 模型估错时方向必为安全侧。
结论: 440/450 超调 8.8/12.6° -> 2.8/3.4°; 对象无缩水而修正常开 -> 仅偏慢 (0.2° 超调)。
用法: python3 analysis/ctrl_kq_verify.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin import Twin

DT = 0.01


def cl_map2(tv, wc, wo, wr, nd, tauf, mv_max, y0, svt, ckq=0.0, ctref=315.0, T=18., pre=2.):
    """开表闭环: 控制器带记账温度修正 (ckq = ST 的 lrKqHot), 对象 = 孪生 v3"""
    n_pre = int(pre / DT); N = n_pre + int(T / DT)
    nd_p = max(1, int(round(tv.theta / DT)))
    rng = np.random.default_rng(1)
    v_hi = float(tv.q_of(mv_max))
    q_need = tv.h1 * (y0 - tv.c)
    u_ss = float(np.interp(max(q_need, 0.), tv.q_bp, tv.u_bp))
    v_ss = float(tv.q_of(u_ss))
    y = y0; qf = q_need; z1, z2 = y0, 0.0; v1, v2 = y0, 0.0
    buf = [v_ss] * (nd + 1); vf = v_ss
    uh = np.full(N + nd_p + 2, u_ss); yy = np.zeros(N)
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
        va = float(tv.q_of(u))
        if ckq > 0 and ym > ctref and u > 30.:
            g = 1.0 - ckq * min(max((u - 30.) / 30., 0.), 1.) * (ym - ctref) / 100.
            va *= max(g, 0.4)
        buf = [va] + buf[:-1]
        uh[i + nd_p] = u
        ud = float(uh[i])
        for _ in range(2):
            h = DT / 2
            qf += h * (tv.q_eff(ud, y) - qf) / max(tv.tau2, 1e-4)
            y += h * (qf - tv.h1 * (y - tv.c))
        yy[i] = y
    yy = yy[n_pre:]; tt = np.arange(len(yy)) * DT
    reach = tt[np.argmax(yy >= svt)] if (yy >= svt).any() else np.inf
    ov = max(yy.max() - svt, 0)
    bad = np.where(np.abs(yy - svt) > 2)[0]
    s2 = tt[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(yy) else np.inf
    return reach, ov, s2


R3 = dict(wc=1.5, wo=10., wr=3.5, nd=12, tauf=0.24, mv_max=60.)
CASES = [('100→400', 100., 400.), ('200→400', 200., 400.),
         ('200→440', 200., 440.), ('100→450', 100., 450.)]

if __name__ == '__main__':
    tv = Twin()
    print('对象=孪生v3 | 第3档参数 | 修正前 vs 修正后 (ckq=1.1/tref=315):')
    for nm, y0, svt in CASES:
        a = cl_map2(tv, **R3, y0=y0, svt=svt, ckq=0.0)
        b = cl_map2(tv, **R3, y0=y0, svt=svt, ckq=1.1)
        print(f'  {nm}: {a[0]:.2f}s/{a[1]:4.1f}° -> {b[0]:.2f}s/{b[1]:4.1f}°')
    print('\n鲁棒性 (对象缩水 vs 控制器认知):')
    for pk, ck, lab in [(0.7, 1.1, '对象0.7/控制器1.1'), (1.5, 1.1, '对象1.5/控制器1.1'),
                        (0.0, 1.1, '对象无缩水/控制器1.1'), (1.1, 0.0, '有缩水/不修正')]:
        tv2 = Twin(); tv2.kq_hot = pk
        r4, o4, _ = cl_map2(tv2, **R3, y0=100., svt=400., ckq=ck)
        r, o, _ = cl_map2(tv2, **R3, y0=200., svt=440., ckq=ck)
        print(f'  {lab}: 400步 {r4:.2f}s/{o4:4.1f}° | 440步 {r:.2f}s/{o:4.1f}°')
