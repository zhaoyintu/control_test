#!/usr/bin/env python3
"""G2/G3 终版: 元件温度第四状态 (2026-07-21 z2 取证定案后)

结构: 元件冷启动功率冲 --
    c_e ∈ [0,1] 元件温度代理: u>2% 时 dc_e=(1−c_e)/τ_h (通电快充),
                              u≤2% 时 dc_e=−c_e/τ_c   (断电慢放);
    元件指令功率 = q_true(u,y)·(1 + β·(1−c_e));  qf 一阶 τe=0.242 (小信号值, 不再分档)。
依据: 真实 z2 取证 -- 冷起冲刺 +70~81 vs 暖起 +18~25 (金属元件正温度系数,
    冷元件电阻低多给功率; 10% 保温占空比足以保持元件热)。
拟合: (s_hi, kq_p, β, τ_h, τ_c) 于 7-20 全部 7 窗; c_e 初值 = exp(−断电时长/τ_c)。

★ 状态 (2026-07-21): 第一版 (通断二值 c_e 代理) 失败 -- G3 8/17, 已弃用。
  失败原因: "断电时长"的走回法被接管小脉冲 (100°C 接管时几秒 2~60% MV) 打断,
  元件冷热被误判; 元件温度是功率加权连续量, 不是通断布尔。
  正路 = 专项直测 ("元件测温场", 机上 ~20min): 变间隔满幅双脉冲 --
  断电 {5, 30, 120, 600}s 后各打一次 90%×1s, 初始上升速率族直接描出 β 与 τ_c
  (同"降温段测墙"的哲学: 每个参数配一个专属激励, 不靠闭环窗口盲拟)。
用法: python3 analysis/g23_element_state.py
"""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin import Twin
from g2_g3_pipeline import (load, observer, fit_w0, F20, F_AM, F_PM,
                            TESTS, HEATS20, heat_windows, P_WALL, THETA, S_LO, AMB)

TAB = Twin()
DT = 0.01
TE = 0.242


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


def plant4(y, w1, w2, qf, ce, ud, s_hi, kq_p, beta, tau_h, tau_c, dt=DT):
    a1, b1, c1, d1, e1 = (P_WALL[k] for k in ('a1', 'b1', 'c1', 'd1', 'e1'))
    if ud > 2.0:
        ce = ce + dt * (1.0 - ce) / tau_h
    else:
        ce = ce - dt * ce / tau_c
    q_cmd = q_true(ud, s_hi, y, kq_p) * (1.0 + beta * (1.0 - ce))
    qf += dt * (q_cmd - qf) / TE
    dy = qf - a1 * (y - w1)
    dw1 = b1 * (y - w1) - c1 * (w1 - w2)
    dw2 = d1 * (w1 - w2) - e1 * (w2 - AMB)
    return y + dt * dy, w1 + dt * dw1, w2 + dt * dw2, qf, ce


def t_off_before(path, t0):
    """t0 之前元件断电 (MV≤2%) 的持续时长 [s]"""
    t, pv, sv, mv = load(path)
    i0 = np.searchsorted(t, t0)
    j = i0 - 1
    while j > 0 and mv[j] <= 2.0:
        j -= 1
    return max(t[i0] - t[j], 0.0)


def g2_fit():
    obs = observer(F20)
    wins = heat_windows(F20, HEATS20, obs)
    toffs = [t_off_before(F20, t0) for t0 in HEATS20]

    def sim_win(x, w, toff):
        s_hi, kq_p, beta, tau_h, tau_c = x
        useq, pref, w10, w20, upre = w
        y, w1, w2 = pref[0], w10, w20
        ce = 1.0 if upre > 2.0 else float(np.exp(-toff / tau_c))
        qf = q_true(upre, s_hi) * (1.0 + beta * (1.0 - ce)) if upre > 2.0 else 0.0
        nd = max(1, int(round(THETA / DT)))
        sim = np.empty(len(useq))
        for i in range(len(useq)):
            ud = float(useq[max(0, i - nd)])
            y, w1, w2, qf, ce = plant4(y, w1, w2, qf, ce, ud, s_hi, kq_p, beta, tau_h, tau_c)
            sim[i] = y
        return sim

    def resid(x):
        return np.concatenate([(sim_win(x, w, tf) - w[1])[::20] for w, tf in zip(wins, toffs)])

    res = least_squares(resid, [0.95, 0.2, 0.25, 0.4, 30.],
                        bounds=([0.8, 0.05, 0.0, 0.05, 5.], [1.15, 0.35, 0.6, 2.0, 200.]),
                        diff_step=0.05)
    rmse = np.sqrt(np.mean(res.fun ** 2))
    for k, (w, tf) in enumerate(zip(wins, toffs)):
        e = sim_win(res.x, w, tf) - w[1]
        print(f'    窗{k+1} ({w[1][0]:.0f}→{w[1].max():.0f}, 断电{tf:.0f}s): RMSE {np.sqrt(np.mean(e**2)):.2f}°')
    return tuple(float(v) for v in res.x) + (rmse,)


def g3_run(P):
    s_hi, kq_p, beta, tau_h, tau_c = P
    print('\nG3 终版: 17 测试闭环复现')
    obs_cache = {}
    ok = 0; n = 0
    print(f'{"工况":9s} {"kd":>4s} {"断电":>4s} | {"仿真":>12s} | {"实测":>12s} |')
    for path, t0, svt, kd, mr, mo in TESTS:
        if path not in obs_cache:
            w10, w20, rm = fit_w0(path)
            obs_cache[path] = observer(path, w10, w20)
        T_, w1t, w2t = obs_cache[path]
        t, pv, sv, mv = load(path)
        i0 = np.searchsorted(t, t0)
        w1 = float(np.interp(t0 - 3.0, T_, w1t)); w2 = float(np.interp(t0 - 3.0, T_, w2t))
        j0 = np.searchsorted(t, t0 - 3.0)
        ttp = np.arange(t[j0], t0, DT)
        idx = np.clip(np.searchsorted(t, ttp, side='right') - 1, 0, len(t) - 1)
        upre = mv[idx]
        toff = t_off_before(path, t0 - 3.0)
        ce = 1.0 if float(upre[0]) > 2.0 else float(np.exp(-toff / tau_c))
        rng = np.random.default_rng(1)
        nd, tauf, wc, wo, wr, mvmax, ckq = 12, 0.24, 4.5, 10., 30., 90., 1.1
        nd_p = max(1, int(round(THETA / DT)))
        yv = float(np.interp(t[j0], t, pv))
        qf = q_true(float(upre[0]), s_hi) * (1.0 + beta * (1.0 - ce)) if float(upre[0]) > 2. else 0.0
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
            yv, w1, w2, qf, ce = plant4(yv, w1, w2, qf, ce, float(uh[i]),
                                        s_hi, kq_p, beta, tau_h, tau_c)
        v1, v2 = z1, 0.0
        N = int(8.0 / DT)
        yy = np.empty(N)
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
            vc = min(max(vc, 0.0), float(np.interp(mvmax, TAB.u_bp, TAB.q_bp)))
            u = min(max(float(np.interp(vc, TAB.q_bp, TAB.u_bp)), 0.0), mvmax)
            va = float(np.interp(u, TAB.u_bp, TAB.q_bp))
            if ym > 315. and u > 30.:
                g = 1. - ckq * min(max((u - 30.) / 30., 0.), 1.) * (ym - 315.) / 100.
                va *= max(g, 0.4)
            buf = [va] + buf[:-1]
            uh.append(u)
            yv, w1, w2, qf, ce = plant4(yv, w1, w2, qf, ce, float(uh[len(upre) + i]),
                                        s_hi, kq_p, beta, tau_h, tau_c)
            yy[i] = yv
        tt2 = np.arange(N) * DT
        cross = np.where(yy >= svt)[0]
        r_ = tt2[cross[0]] if len(cross) else np.inf
        o_ = max(yy.max() - svt, 0.0)
        hit = (r_ < 2.2) == (mr < 2.2)
        ok += hit; n += 1
        rt = f'{r_:5.2f}s/{o_:4.1f}°' if np.isfinite(r_) else f' 爬行/{o_:4.1f}°'
        print(f'{pv[i0]:.0f}→{svt:.0f} {kd:4.2f} {t_off_before(path, t0):4.0f} | {rt:>12s} | {mr:5.2f}s/{mo:4.1f}° | {"对" if hit else "错"}')
    print(f'方向命中 {ok}/{n}')


if __name__ == '__main__':
    P = g2_fit()
    s_hi, kq_p, beta, tau_h, tau_c, rmse = P
    print(f'G2 终版: s_hi={s_hi:.3f} kq_p={kq_p:.3f} β={beta:.3f} τ_h={tau_h:.2f}s τ_c={tau_c:.0f}s'
          f'  RMSE={rmse:.2f}° ({"过关" if rmse <= 3.0 else "未过 ≤3°"})')
    g3_run(P[:5])
