#!/usr/bin/env python3
"""G2/G3: 大信号元件辨识 + 13 测试闭环复现 -- 基于两层墙 (G1 已过, RMSE 1.62°)

G2: 7-20 的 7 个加热窗口 (2×400 + 4×440 + 阶梯首段), 记录 MV 回放,
    拟 (s_hi, τe_hi): q_true = q_tab·gain(u), gain: 0.492(u≤25)→s_hi(u≥55);
    τe(u): 0.242(u≤25) → τe_hi(u≥55)。过关: 窗口 RMSE ≤3°。
G3: 7-19 两场 13 个标注 kd 测试, 控制器按机上原样 (旧表记账+ckq=1.1),
    对象 = 元件(大信号) + 两层墙; 各场观测器初值由该场降温段反拟 (参数冻结)。
    过关: 冲透/爬行方向 13/13。

★ 状态 (2026-07-21 第一轮):
  G2 = 8.96° 未过 (s_hi=1.048, τe_hi=0.108s, kq_p 顶界 0.6 -- 结构仍缺一块);
  G3 = 方向 13/18 (从 7/18 大幅上来)。
  已复现 (重要!): 200 靶点全部 5 个 -- 含热炉体 8~18° 大超调的幅度级
    (0.66s/18.6° vs 实测 0.71s/18.1°), "适应性差之谜"的现象已被模型捕获;
    400 靶点 5/7 (两个错的都是真机踩悬崖翻面的边缘例, 掷硬币本不可复现)。
  未复现: 200→440 冲透组 4 例全错 (仿真过阻尼爬行, 实测冲透 3~10.7°) --
    440 冲刺段 (>315°C, 90%满幅) 的热沉积仍未建对; kq_p 顶界提示衬面
    过热层/辐射非线性一类结构缺失。下一步专攻此段, 无需新机上数据。
用法: python3 analysis/g2_g3_pipeline.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin import Twin

S_LO, AMB = 0.492, 28.4
P_WALL = dict(a1=0.0689, b1=0.0074, c1=0.00852, d1=0.00529, e1=0.002930)  # G1 冻结
THETA, TE_LO = 0.12, 0.242
TAB = Twin()
DT = 0.01
F20 = os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260720-201947_215552.csv')
F_AM = os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260719-163352_164130.csv')
F_PM = os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260719-181948_191244.csv')

_c = {}


def load(path):
    if path not in _c:
        df = pd.read_csv(path)
        ts = pd.to_datetime(df.iloc[:, 0])
        t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
        _c[path] = (t, df.iloc[:, 1].to_numpy(float), df.iloc[:, 3].to_numpy(float),
                    df.iloc[:, 4].to_numpy(float))
    return _c[path]


def blend(u, lo, hi):
    if u <= 25.0:
        return lo
    if u >= 55.0:
        return hi
    return lo + (hi - lo) * (u - 25.0) / 30.0


def q_true(u, s_hi, y=0.0, kq_p=0.0):
    """诚实表 × 高段温度缩水 (对象侧; 7-16 实测 90%@380-425 约 0.8×)"""
    base = float(np.interp(u, TAB.u_bp, TAB.q_bp)) * blend(u, S_LO, s_hi)
    if kq_p > 0.0 and y > 315.0 and u > 30.0:
        g = 1.0 - kq_p * min(max((u - 30.0) / 30.0, 0.0), 1.0) * (y - 315.0) / 100.0
        base *= max(g, 0.5)
    return base


def observer(path, w10=AMB, w20=AMB, dto=0.25):
    t, pv, sv, mv = load(path)
    T_ = np.arange(0.0, t[-1], dto)
    PV_ = np.interp(T_, t, pv)
    a1, b1, c1, d1, e1 = (P_WALL[k] for k in ('a1', 'b1', 'c1', 'd1', 'e1'))
    n = len(T_)
    w1 = np.empty(n); w2 = np.empty(n)
    w1[0], w2[0] = w10, w20
    for i in range(1, n):
        w1[i] = w1[i-1] + dto * (b1 * (PV_[i-1] - w1[i-1]) - c1 * (w1[i-1] - w2[i-1]))
        w2[i] = w2[i-1] + dto * (d1 * (w1[i-1] - w2[i-1]) - e1 * (w2[i-1] - AMB))
    return T_, w1, w2


def plant_step(y, w1, w2, qf, ud, s_hi, te_hi, dt=DT, kq_p=0.0):
    a1, b1, c1, d1, e1 = (P_WALL[k] for k in ('a1', 'b1', 'c1', 'd1', 'e1'))
    te = blend(ud, TE_LO, te_hi)
    qf += dt * (q_true(ud, s_hi, y, kq_p) - qf) / te
    dy = qf - a1 * (y - w1)
    dw1 = b1 * (y - w1) - c1 * (w1 - w2)
    dw2 = d1 * (w1 - w2) - e1 * (w2 - AMB)
    return y + dt * dy, w1 + dt * dw1, w2 + dt * dw2, qf


# ---------- G2 ----------
HEATS20 = [2440.8, 3213.0, 3490.2, 3883.2, 4294.8, 4446.0, 5393.4]  # 阶跃时刻(s)


def heat_windows(path, t0s, obs, dur=12.0):
    t, pv, sv, mv = load(path)
    T_, w1t, w2t = obs
    out = []
    for t0 in t0s:
        i0 = np.searchsorted(t, t0 - 0.5)
        tt = np.arange(t[i0], t[i0] + 0.5 + dur, DT)
        idx = np.clip(np.searchsorted(t, tt, side='right') - 1, 0, len(t) - 1)
        out.append((mv[idx], np.interp(tt, t, pv),
                    float(np.interp(t[i0], T_, w1t)), float(np.interp(t[i0], T_, w2t)),
                    float(mv[max(0, i0 - 150):i0].mean())))
    return out


def g2_fit():
    obs = observer(F20)
    wins = heat_windows(F20, HEATS20, obs)

    def resid(x):
        s_hi, te_hi, kq_p = x
        r = []
        for useq, pref, w10, w20, upre in wins:
            y, w1, w2 = pref[0], w10, w20
            qf = q_true(upre, s_hi)
            nd = max(1, int(round(THETA / DT)))
            sim = np.empty(len(useq))
            for i in range(len(useq)):
                ud = float(useq[max(0, i - nd)])
                y, w1, w2, qf = plant_step(y, w1, w2, qf, ud, s_hi, te_hi, kq_p=kq_p)
                sim[i] = y
            r.append((sim - pref)[::20])
        return np.concatenate(r)

    res = least_squares(resid, [0.9, 0.12, 0.2], bounds=([0.5, 0.03, 0.0], [1.2, 0.30, 0.6]),
                        diff_step=0.03)
    rmse = np.sqrt(np.mean(res.fun ** 2))
    return float(res.x[0]), float(res.x[1]), float(res.x[2]), rmse


# ---------- G3 ----------
TESTS = [
    (F_AM, 30.8, 200., 1.1, 0.99, 2.6), (F_AM, 53.4, 440., 1.1, 1.59, 3.0),
    (F_AM, 201.3, 400., 0.9, 3.30, 0.5), (F_AM, 422.9, 400., 0.9, 1.57, 2.5),
    (F_PM, 1484.6, 200., 1.2, 0.89, 5.6), (F_PM, 1497.3, 440., 1.2, 3.12, 0.4),
    (F_PM, 1688.7, 400., 1.2, 3.36, 0.6), (F_PM, 1999.4, 400., 0.95, 3.19, 1.0),
    (F_PM, 2125.1, 400., 0.85, 1.71, 1.0), (F_PM, 2418.8, 400., 0.75, 1.54, 4.2),
    (F_PM, 2510.7, 200., 0.75, 0.71, 18.1), (F_PM, 2523.8, 440., 0.75, 1.38, 10.7),
    (F_PM, 2672.6, 200., 0.9, 0.81, 13.5), (F_PM, 2685.5, 440., 0.9, 1.42, 5.6),
    (F_PM, 2961.1, 200., 1.0, 0.83, 8.4), (F_PM, 2973.8, 440., 1.0, 1.41, 4.4),
    (F_PM, 3115.6, 200., 1.1, 0.82, 8.0), (F_PM, 3128.3, 440., 1.1, 3.33, 0.3),
]


def fit_w0(path):
    """该场观测器初值 (w10=w20=W0) 由其降温段反拟, 墙参数冻结"""
    t, pv, sv, mv = load(path)
    lo = mv <= 0.5
    d = np.diff(lo.astype(int)); s_ = np.where(d == 1)[0] + 1; e_ = np.where(d == -1)[0] + 1
    if lo[0]: s_ = np.r_[0, s_]
    if lo[-1]: e_ = np.r_[e_, len(lo)]
    falls = []
    for a, b in zip(s_, e_):
        if t[b-1] - t[a] < 60 or pv[a] < 150:
            continue
        end = min(t[a] + 151.5, t[b-1] - 1.0)
        tt = np.arange(t[a] + 1.5, end, 0.5)
        falls.append((t[a] + 1.5, tt, np.interp(tt, t, pv)))
    a1, b1, c1, d1, e1 = (P_WALL[k] for k in ('a1', 'b1', 'c1', 'd1', 'e1'))

    def resid(x):
        obs = observer(path, w10=float(x[0]), w20=float(x[1]))
        T_, w1t, w2t = obs
        r = []
        for t0, tt, yy in falls:
            y, w1, w2 = yy[0], float(np.interp(t0, T_, w1t)), float(np.interp(t0, T_, w2t))
            sim = np.empty(len(tt))
            for i in range(len(tt)):
                sim[i] = y
                dy = -a1 * (y - w1)
                dw1 = b1 * (y - w1) - c1 * (w1 - w2)
                dw2 = d1 * (w1 - w2) - e1 * (w2 - AMB)
                y += 0.5 * dy; w1 += 0.5 * dw1; w2 += 0.5 * dw2
            r.append(sim - yy)
        return np.concatenate(r)

    res = least_squares(resid, [120., 80.], bounds=([AMB, AMB], [350., 300.]), diff_step=0.05)
    rmse = np.sqrt(np.mean(res.fun ** 2))
    return float(res.x[0]), float(res.x[1]), rmse


def g3_run(s_hi, te_hi, kq_p):
    print('\nG3: 13+5 测试闭环复现 (控制器=机上原样)')
    obs_cache = {}
    ok = 0; n = 0
    print(f'{"工况":9s} {"kd":>4s} {"衬温":>4s} | {"仿真":>12s} | {"实测":>12s} |')
    for path, t0, svt, kd, mr, mo in TESTS:
        if path not in obs_cache:
            w10, w20, rm = fit_w0(path)
            obs_cache[path] = observer(path, w10, w20)
            print(f'  [{os.path.basename(path)[10:19]}: 观测器初值 衬={w10:.0f} 体={w20:.0f} (该场降温 RMSE {rm:.1f}°)]')
        T_, w1t, w2t = obs_cache[path]
        t, pv, sv, mv = load(path)
        i0 = np.searchsorted(t, t0)
        y = float(pv[i0])
        w1 = float(np.interp(t0, T_, w1t)); w2 = float(np.interp(t0, T_, w2t))
        # 预热 3s: 记录 MV 驱动
        j0 = np.searchsorted(t, t0 - 3.0)
        ttp = np.arange(t[j0], t0, DT)
        idx = np.clip(np.searchsorted(t, ttp, side='right') - 1, 0, len(t) - 1)
        upre = mv[idx]
        rng = np.random.default_rng(1)
        nd, tauf, wc, wo, wr, mvmax, ckq = 12, 0.24, 4.5, 10., 30., 90., 1.1
        nd_p = max(1, int(round(THETA / DT)))
        yp = float(np.interp(t[j0], t, pv))
        yv, qf = yp, q_true(float(upre[0]), s_hi)
        z1, z2 = yp, 0.0
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
            yv, w1, w2, qf = plant_step(yv, w1, w2, qf, float(uh[i]), s_hi, te_hi, kq_p=kq_p)
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
            yv, w1, w2, qf = plant_step(yv, w1, w2, qf, float(uh[len(upre) + i]), s_hi, te_hi, kq_p=kq_p)
            yy[i] = yv
        tt2 = np.arange(N) * DT
        cross = np.where(yy >= svt)[0]
        r_ = tt2[cross[0]] if len(cross) else np.inf
        o_ = max(yy.max() - svt, 0.0)
        hit = (r_ < 2.2) == (mr < 2.2)
        ok += hit; n += 1
        rt = f'{r_:5.2f}s/{o_:4.1f}°' if np.isfinite(r_) else f' 爬行/{o_:4.1f}°'
        print(f'{pv[i0]:.0f}→{svt:.0f} {kd:4.2f} {np.interp(t0, T_, w1t):4.0f} | {rt:>12s} | {mr:5.2f}s/{mo:4.1f}° | {"对" if hit else "错"}')
    print(f'方向命中 {ok}/{n}')


if __name__ == '__main__':
    s_hi, te_hi, kq_p, rmse = g2_fit()
    print(f'G2: s_hi={s_hi:.3f}  τe_hi={te_hi:.3f}s  kq_p={kq_p:.3f}  加热窗 RMSE={rmse:.2f}°  '
          f'({"过关" if rmse <= 3.0 else "未过 ≤3° 线"})')
    g3_run(s_hi, te_hi, kq_p)
