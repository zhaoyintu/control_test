#!/usr/bin/env python3
"""G3 终跑: 元件冷启动冲 (7-22 直测) 接回 v1 数据链

元件模型 (fit_element_0722 定案, 耦合形式):
    boost = 1 + β·(1−c_e), β=0.42;  q_cmd = q_true(u,y)·boost;  τe_eff = 0.242/boost
    c_e: 充 τ_h(u)=0.35·90/u, 放 τ_c=103s; 每测试由其前 520s 记录 MV 链式积分得初值。
流程: ① G2 重拟 (7-20 窗, 仅 s_hi/kq_p 自由, 元件参数锁死) ② G3 17 测试终跑。
机理期望: 400 类冷元件起跑 → boost≈1.4 (更猛更快) ✓ 匹配真实 z2 +70~81;
          440 类暖元件起跑 → boost≈1 ✓ 匹配真实 z2 +18~25。单一参数组两 regime 兼得。
用法: python3 analysis/g3_final.py
"""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin import Twin
from g2_g3_pipeline import (load, observer, fit_w0, F20, F_AM, F_PM,
                            TESTS, HEATS20, heat_windows, P_WALL, S_LO, AMB)

TAB = Twin()
DT = 0.01
THETA_V1, TE = 0.12, 0.242
BETA, TAU_C, TAU_H90 = 0.42, 103.0, 0.35


def blend(u, lo, hi):
    if u <= 25.0:
        return lo
    if u >= 55.0:
        return hi
    return lo + (hi - lo) * (u - 25.0) / 30.0


def q_true(u, s_hi, y=0.0, kq_p=0.0):
    base = float(np.interp(u, TAB.u_bp, TAB.q_bp)) * blend(u, S_LO, s_hi)
    if kq_p > 0.0 and y > 315.0 and u > 30.0:
        ramp = min(max((y - 315.0) / 45.0, 0.0), 1.0)
        base *= 1.0 - kq_p * min(max((u - 30.0) / 30.0, 0.0), 1.0) * ramp
    return base


def ce_step(ce, u, dt):
    if u > 2.0:
        ce += dt * (1.0 - ce) / (TAU_H90 * 90.0 / max(u, 5.0))
    else:
        ce -= dt * ce / TAU_C
    return min(max(ce, 0.0), 1.0)


def ce_init(path, t0):
    """t0 前 520s 记录 MV 链式积分 (5τ_c 洗掉初值)"""
    t, pv, sv, mv = load(path)
    tt = np.arange(max(t0 - 520.0, t[0]), t0, 0.05)
    idx = np.clip(np.searchsorted(t, tt, side='right') - 1, 0, len(t) - 1)
    ce = 0.5
    for u in mv[idx]:
        ce = ce_step(ce, float(u), 0.05)
    return ce


def plant_step(y, w1, w2, qf, ce, ud, s_hi, kq_p, dt=DT):
    a1, b1, c1, d1, e1 = (P_WALL[k] for k in ('a1', 'b1', 'c1', 'd1', 'e1'))
    ce = ce_step(ce, ud, dt)
    boost = 1.0 + BETA * (1.0 - ce)
    q_cmd = q_true(ud, s_hi, y, kq_p) * boost
    qf += dt * (q_cmd - qf) / (TE / boost)
    y += dt * (qf - a1 * (y - w1))
    w1 += dt * (b1 * (y - w1) - c1 * (w1 - w2))
    w2 += dt * (d1 * (w1 - w2) - e1 * (w2 - AMB))
    return y, w1, w2, qf, ce


def g2_refit():
    obs = observer(F20)
    wins = heat_windows(F20, HEATS20, obs)
    ces = [ce_init(F20, t0 - 0.5) for t0 in HEATS20]

    def sim_win(x, w, ce0):
        s_hi, kq_p = x
        useq, pref, w10, w20, upre = w
        y, w1, w2, qf, ce = pref[0], w10, w20, 0.0, ce0
        if upre > 2.0:
            qf = q_true(upre, s_hi) * (1.0 + BETA * (1.0 - ce0))
        nd = max(1, int(round(THETA_V1 / DT)))
        sim = np.empty(len(useq))
        for i in range(len(useq)):
            ud = float(useq[max(0, i - nd)])
            y, w1, w2, qf, ce = plant_step(y, w1, w2, qf, ce, ud, s_hi, kq_p)
            sim[i] = y
        return sim

    def resid(x):
        return np.concatenate([(sim_win(x, w, c) - w[1])[::20] for w, c in zip(wins, ces)])

    res = least_squares(resid, [0.95, 0.2], bounds=([0.7, 0.05], [1.2, 0.35]), diff_step=0.03)
    rmse = np.sqrt(np.mean(res.fun ** 2))
    print(f'G2 重拟 (元件参数锁死): s_hi={res.x[0]:.3f} kq_p={res.x[1]:.3f} RMSE={rmse:.2f}°')
    for k, (w, c) in enumerate(zip(wins, ces)):
        e = sim_win(res.x, w, c) - w[1]
        print(f'    窗{k+1} ({w[1][0]:.0f}→{w[1].max():.0f}, c_e0={c:.2f}): {np.sqrt(np.mean(e**2)):.2f}°')
    return float(res.x[0]), float(res.x[1])


def g3(s_hi, kq_p):
    print('\nG3 终跑: 17 测试')
    obs_cache = {}
    ok = 0; n = 0
    fails = []
    print(f'{"工况":9s} {"kd":>4s} {"ce0":>4s} | {"仿真":>12s} | {"实测":>12s} |')
    for path, t0, svt, kd, mr, mo in TESTS:
        if path not in obs_cache:
            w10, w20, _ = fit_w0(path)
            obs_cache[path] = observer(path, w10, w20)
        T_, w1t, w2t = obs_cache[path]
        t, pv, sv, mv = load(path)
        i0 = np.searchsorted(t, t0)
        w1 = float(np.interp(t0 - 3.0, T_, w1t)); w2 = float(np.interp(t0 - 3.0, T_, w2t))
        ce = ce_init(path, t0 - 3.0)
        ce_rec = ce
        j0 = np.searchsorted(t, t0 - 3.0)
        ttp = np.arange(t[j0], t0, DT)
        idx = np.clip(np.searchsorted(t, ttp, side='right') - 1, 0, len(t) - 1)
        upre = mv[idx]
        rng = np.random.default_rng(1)
        nd, tauf, wc, wo, wr, mvmax, ckq = 12, 0.24, 4.5, 10., 30., 90., 1.1
        nd_p = max(1, int(round(THETA_V1 / DT)))
        yv = float(np.interp(t[j0], t, pv))
        qf = q_true(float(upre[0]), s_hi) * (1.0 + BETA * (1.0 - ce)) if float(upre[0]) > 2. else 0.0
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
            if ym > 315. and u > 30.:
                g = 1. - ckq * min(max((u - 30.) / 30., 0.), 1.) * (ym - 315.) / 100.
                va *= max(g, 0.4)
            buf = [va] + buf[:-1]
            uh.append(u)
            yv, w1, w2, qf, ce = plant_step(yv, w1, w2, qf, ce, float(uh[i]), s_hi, kq_p)
        v1r, v2r = z1, 0.0
        N = int(8.0 / DT)
        yy = np.empty(N)
        for i in range(N):
            ym = yv + rng.normal(0, TAB.sig_n)
            v1n = v1r + DT * v2r
            v2n = v2r + DT * (-2 * wr * v2r - wr * wr * (v1r - svt))
            v1r, v2r = v1n, v2n
            vd = buf[nd]
            if tauf:
                vf += DT * (vd - vf) / tauf
                vd = vf
            err = ym - z1
            z1 += DT * (z2 + vd + 2 * wo * err)
            z2 += DT * (wo * wo * err)
            vc = wc * (v1r - z1) - z2 - kd * (vd + z2)
            vc = min(max(vc, 0.0), float(np.interp(mvmax, TAB.u_bp, TAB.q_bp)))
            u = min(max(float(np.interp(vc, TAB.q_bp, TAB.u_bp)), 0.0), mvmax)
            va = float(np.interp(u, TAB.u_bp, TAB.q_bp))
            if ym > 315. and u > 30.:
                g = 1. - ckq * min(max((u - 30.) / 30., 0.), 1.) * (ym - 315.) / 100.
                va *= max(g, 0.4)
            buf = [va] + buf[:-1]
            uh.append(u)
            yv, w1, w2, qf, ce = plant_step(yv, w1, w2, qf, ce, float(uh[len(upre) + i]), s_hi, kq_p)
            yy[i] = yv
        tt2 = np.arange(N) * DT
        cross = np.where(yy >= svt)[0]
        r_ = tt2[cross[0]] if len(cross) else np.inf
        o_ = max(yy.max() - svt, 0.0)
        hit = (r_ < 2.2) == (mr < 2.2)
        ok += hit; n += 1
        if not hit:
            fails.append(f'{pv[i0]:.0f}→{svt:.0f}@kd{kd}')
        rt = f'{r_:5.2f}s/{o_:4.1f}°' if np.isfinite(r_) else f' 爬行/{o_:4.1f}°'
        print(f'{pv[i0]:.0f}→{svt:.0f} {kd:4.2f} {ce_rec:4.2f} | {rt:>12s} | {mr:5.2f}s/{mo:4.1f}° | {"对" if hit else "错"}')
    print(f'\n方向命中 {ok}/{n}' + (f'  未中: {", ".join(fails)}' if fails else ''))


if __name__ == '__main__':
    s_hi, kq_p = g2_refit()
    g3(s_hi, kq_p)
