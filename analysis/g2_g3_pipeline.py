#!/usr/bin/env python3
"""G2/G3: 大信号元件辨识 + 13 测试闭环复现 -- 基于两层墙 (G1 已过, RMSE 1.62°)

G2: 7-20 的 7 个加热窗口 (2×400 + 4×440 + 阶梯首段), 记录 MV 回放,
    拟 (s_hi, τe_hi): q_true = q_tab·gain(u), gain: 0.492(u≤25)→s_hi(u≥55);
    τe(u): 0.242(u≤25) → τe_hi(u≥55)。过关: 窗口 RMSE ≤3°。
G3: 7-19 两场 13 个标注 kd 测试, 控制器按机上原样 (旧表记账+ckq=1.1),
    对象 = 元件(大信号) + 两层墙; 各场观测器初值由该场降温段反拟 (参数冻结)。
    过关: 冲透/爬行方向 13/13。

★ 状态 (2026-07-21 第三轮): G2=8.83° 未过, G3=14/18 (上限 17/18: 早场 kd=0.9
  同参数两次翻面是掷硬币, 确定性仿真只能对一个)。
  结构裁决记录: 辐射分流 f_split 被拟合否决 (→0.014≈0); 缩水改台阶饱和型
  (315→360 渐进后平, 依 7-16 直测 380~480 平坦 g≈0.75) -- +1 例翻正, 形状正确。
  剩余failure=440 冲透组 3 例 (仿真过刹爬行): kq_p 仍顶界 0.35, 且 G2 最差窗
  是"复热型"起点 (147→400: 11.5°, 66→401: 14.1°) -- 两条线索并存。
  下一步 (禁止再盲加参数): ①残差形状取证 (逐窗误差-时间曲线分段归因);
  ②控制器回放取证: 把控制器复刻直接喂真机 PV/MV, 提取真实 z2 在 440 冲刺中的
  轨迹, 与仿真 z2 对照 -- 无需对象模型即可分离"对象错"vs"记账复刻错";
  ③7-19 晚场墙参数疑似不迁移 (fit_w0 残差 63°), 允许每场 amb/衬参数微调再验。
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
    """诚实表 × 高段温度缩水 -- 台阶饱和型 (7-16 直测: 380~480°C g≈0.74~0.80 平坦,
       非线性下坡; 315→360 渐进, 之后平在 1−kq_p)"""
    base = float(np.interp(u, TAB.u_bp, TAB.q_bp)) * blend(u, S_LO, s_hi)
    if kq_p > 0.0 and y > 315.0 and u > 30.0:
        ramp = min(max((y - 315.0) / 45.0, 0.0), 1.0)
        g = 1.0 - kq_p * min(max((u - 30.0) / 30.0, 0.0), 1.0) * ramp
        base *= g
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


def plant_step(y, w1, w2, qf, ud, s_hi, te_hi, dt=DT, kq_p=0.0, f_sp=0.0):
    a1, b1, c1, d1, e1 = (P_WALL[k] for k in ('a1', 'b1', 'c1', 'd1', 'e1'))
    te = blend(ud, TE_LO, te_hi)
    qf += dt * (q_true(ud, s_hi, y, kq_p) - qf) / te
    dy = qf * (1.0 - f_sp) - a1 * (y - w1)
    dw1 = b1 * (y - w1) - c1 * (w1 - w2) + f_sp * qf * (b1 / a1)
    dw2 = d1 * (w1 - w2) - e1 * (w2 - AMB)
    return y + dt * dy, w1 + dt * dw1, w2 + dt * dw2, qf


# ---------- G2 ----------
HEATS20 = [2440.8, 3213.0, 3490.2, 3883.2, 4294.8, 4446.0, 5393.4]  # 阶跃时刻(s)
HEATS_CLEAN = [2440.8, 3213.0, 3490.2, 4294.8]   # 冷墙400 + 三个干净440 (复热型窗污染元件参数, 剔出拟合)


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

    def sim_win(x, w):
        s_hi, te_hi, kq_p, f_sp = x
        useq, pref, w10, w20, upre = w
        y, w1, w2 = pref[0], w10, w20
        qf = q_true(upre, s_hi)
        nd = max(1, int(round(THETA / DT)))
        sim = np.empty(len(useq))
        for i in range(len(useq)):
            ud = float(useq[max(0, i - nd)])
            y, w1, w2, qf = plant_step(y, w1, w2, qf, ud, s_hi, te_hi, kq_p=kq_p, f_sp=f_sp)
            sim[i] = y
        return sim

    def resid(x):
        return np.concatenate([(sim_win(x, w) - w[1])[::20] for w in wins])

    res = least_squares(resid, [0.9, 0.12, 0.2, 0.0],
                        bounds=([0.6, 0.03, 0.05, 0.0], [1.2, 0.30, 0.35, 0.001]),
                        diff_step=0.03)
    rmse = np.sqrt(np.mean(res.fun ** 2))
    for k, w in enumerate(wins):
        e = sim_win(res.x, w) - w[1]
        print(f'    窗{k+1} ({w[1][0]:.0f}→{w[1].max():.0f}): RMSE {np.sqrt(np.mean(e**2)):.2f}°')
    return tuple(float(v) for v in res.x) + (rmse,)


# ---------- G3 ----------
TESTS = [
    (F_AM, 30.8, 200., 1.1, 0.99, 2.6), (F_AM, 53.4, 440., 1.1, 1.59, 3.0),
    (F_AM, 201.3, 400., 0.9, 3.30, 0.5), (F_AM, 422.9, 400., 0.9, 1.57, 2.5),
    (F_PM, 1484.6, 200., 1.2, 0.89, 5.6), (F_PM, 1497.3, 440., 1.2, 3.12, 0.4),
    (F_PM, 1688.7, 400., 1.2, 3.36, 0.6), (F_PM, 1999.4, 400., 0.95, 3.19, 1.0),
    (F_PM, 2125.1, 400., 0.85, 1.71, 1.0), (F_PM, 2418.8, 400., 0.75, 1.54, 4.2),
    (F_PM, 2510.7, 200., 0.75, 0.71, 18.1), (F_PM, 2523.8, 440., 0.75, 1.38, 10.7),
    (F_PM, 2672.6, 200., 0.9, 0.81, 13.5), (F_PM, 2685.5, 440., 0.9, 1.42, 5.6),
    (F_PM, 2961.1, 200., 1.0, 0.83, 8.4),
    # (F_PM, 2973.8, 440., 1.0, ...): 复刻18.5%+反演wc≈6/kd≈0.5, 实跑配置≠标注, 剔除

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
    # 诊断: 每段初速率反推衬温 vs 观测器
    obs = observer(path, float(res.x[0]), float(res.x[1]))
    T_, w1t, _ = obs
    diag = []
    for t0, tt, yy in falls:
        sl = (yy[6] - yy[0]) / (tt[6] - tt[0])
        diag.append(f'{t0:.0f}s: 反推衬={yy[0] + sl/P_WALL["a1"]:.0f} 观测器衬={np.interp(t0, T_, w1t):.0f}')
    print('    [w1 对照] ' + ' | '.join(diag))
    return float(res.x[0]), float(res.x[1]), rmse


def g3_run(s_hi, te_hi, kq_p, f_sp):
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
            yv, w1, w2, qf = plant_step(yv, w1, w2, qf, float(uh[i]), s_hi, te_hi, kq_p=kq_p, f_sp=f_sp)
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
            yv, w1, w2, qf = plant_step(yv, w1, w2, qf, float(uh[len(upre) + i]), s_hi, te_hi, kq_p=kq_p, f_sp=f_sp)
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
    s_hi, te_hi, kq_p, f_sp, rmse = g2_fit()
    print(f'G2: s_hi={s_hi:.3f}  τe_hi={te_hi:.3f}s  kq_p={kq_p:.3f}  f_split={f_sp:.3f}  '
          f'加热窗 RMSE={rmse:.2f}°  ({"过关" if rmse <= 3.0 else "未过 ≤3° 线"})')
    g3_run(s_hi, te_hi, kq_p, f_sp)
