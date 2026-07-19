#!/usr/bin/env python3
"""v4-lite: 诚实表 + 冻结墙 —— 用 7-19 数据自洽闭合, 过 G3 闭环复现关卡 (2026-07-19)

三个支柱 (全部直接测量, 不依赖慢墙参数):
  1) 表修正: 断电初始降速 = 真保温功率 -> s_lo = 0.492±0.012 (n=17, 表中低段虚增 2.03x)
     高段 s_hi 由加热窗口回放拟合 (90% 冲刺末段斜率交叉验证)
  2) 免费刹车: B = a_yw·(y − w), a_yw = 0.0836 (16 段降温拟合, RMSE 1.82°)
  3) 冻结墙: 阶跃测试只有 ~10s, 墙温取常数, 由测试前 2s 数据直接量出:
       降温进入 (MV≈0): w0 = y0 + (dPV/dt)/a_yw
       保温进入 (MV=u): w0 = y0 − s_lo·q_tab(u)/a_yw

G3 关卡: 控制器按机上原样 (旧表记账 + ckq=1.1 + 标注 kd) 闭环仿真 13 个测试,
到达/首超与实测对表; 方向 (冲透/爬行) 需 13/13 全对才算过。

★ 关卡状态 (2026-07-19 深夜): 未过 (方向 7/18, 开环窗口 RMSE 30.7°, s_hi 顶界 1.15)。
  诊断: 支柱 1/2/3 均直接测量可信; 失败在冲刺侧 -- 90~100% 大信号下元件的
  等效 (θ, τe) 比小信号值 (0.12/0.242) 快得多 (7-15 早有"大电平 θ 更短"记录),
  诚实表 + 小信号滞后组合出的冲刺比真机慢, 拟合以 s_hi 顶界也补不回。
  结论: 大信号元件动力学需要专门辨识 (墙参数采集场序列 #4 的 440 往返即含
  90% 大信号段; 判读时按幅值分档拟 τe(幅值))。在此之前不认证任何新控制律参数。
用法: python3 analysis/v4lite_g3.py
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin import Twin

DT = 0.01
A_YW = 0.08361
S_LO = 0.492
THETA, TAU2 = 0.12, 0.242
F_AM = os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260719-163352_164130.csv')
F_PM = os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260719-181948_191244.csv')
TAB = Twin()

# (文件, t0, svt, kd 或 None, 实测reach, 实测ov)
TESTS = [
    (F_AM, 30.8, 200., 1.1, 0.99, 2.6), (F_AM, 53.4, 440., 1.1, 1.59, 3.0),
    (F_AM, 201.3, 400., 0.9, 3.30, 0.5), (F_AM, 422.9, 400., 0.9, 1.57, 2.5),
    (F_PM, 14.0, 200., None, 2.00, 1.8), (F_PM, 38.2, 440., None, 2.05, 4.6),
    (F_PM, 222.6, 200., None, 4.19, 1.5), (F_PM, 246.5, 440., None, 4.28, 0.9),
    (F_PM, 450.4, 400., None, 4.30, 0.6), (F_PM, 1176.9, 400., None, 3.88, 1.5),
    (F_PM, 1279.0, 200., None, 3.79, 0.4), (F_PM, 1292.2, 440., None, 3.99, 1.0),
    (F_PM, 1484.6, 200., 1.2, 0.89, 5.6), (F_PM, 1497.3, 440., 1.2, 3.12, 0.4),
    (F_PM, 1688.7, 400., 1.2, 3.36, 0.6), (F_PM, 1999.4, 400., 0.95, 3.19, 1.0),
    (F_PM, 2125.1, 400., 0.85, 1.71, 1.0), (F_PM, 2418.8, 400., 0.75, 1.54, 4.2),
    (F_PM, 2510.7, 200., 0.75, 0.71, 18.1), (F_PM, 2523.8, 440., 0.75, 1.38, 10.7),
    (F_PM, 2672.6, 200., 0.9, 0.81, 13.5), (F_PM, 2685.5, 440., 0.9, 1.42, 5.6),
    (F_PM, 2961.1, 200., 1.0, 0.83, 8.4), (F_PM, 2973.8, 440., 1.0, 1.41, 4.4),
    (F_PM, 3115.6, 200., 1.1, 0.82, 8.0), (F_PM, 3128.3, 440., 1.1, 3.33, 0.3),
]

_cache = {}


def load(path):
    if path not in _cache:
        df = pd.read_csv(path)
        ts = pd.to_datetime(df.iloc[:, 0])
        t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
        _cache[path] = (t, df.iloc[:, 1].to_numpy(float), df.iloc[:, 4].to_numpy(float))
    return _cache[path]


def gain(u, s_hi):
    if u <= 25.0:
        return S_LO
    if u >= 55.0:
        return s_hi
    return S_LO + (s_hi - S_LO) * (u - 25.0) / 30.0


def q_true(u, s_hi):
    return float(np.interp(u, TAB.u_bp, TAB.q_bp)) * gain(u, s_hi)


def pre_state(path, t0, s_hi):
    """测试前 2s -> (y0, w0, u_pre): 冻结墙温直接测量"""
    t, pv, mv = load(path)
    i0 = np.searchsorted(t, t0)
    j0 = np.searchsorted(t, t0 - 2.0); j1 = np.searchsorted(t, t0 - 0.1)
    u_pre = float(mv[j0:j1].mean())
    y0 = float(pv[i0])
    if u_pre < 2.0:
        sl = np.polyfit(t[j0:j1], pv[j0:j1], 1)[0]
        w0 = y0 + sl / A_YW
    else:
        w0 = y0 - q_true(u_pre, s_hi) / A_YW
    return y0, max(w0, 20.0), u_pre


def window_replay(path, t0, s_hi, dur=10.0):
    """记录 MV 开环回放 (对象=v4lite), 返回 (仿真y, 实测y)"""
    t, pv, mv = load(path)
    y0, w0, u_pre = pre_state(path, t0, s_hi)
    i0 = np.searchsorted(t, t0 - 0.5)
    tt = np.arange(t[i0], min(t[i0] + 0.5 + dur, t[-1]), DT)
    idx = np.clip(np.searchsorted(t, tt, side='right') - 1, 0, len(t) - 1)
    useq = mv[idx]; pref = np.interp(tt, t, pv)
    nd = max(1, int(round(THETA / DT)))
    y = pref[0]; qf = q_true(u_pre, s_hi)
    yy = np.empty(len(tt))
    for i in range(len(tt)):
        ud = useq[max(0, i - nd)]
        qf += DT * (q_true(float(ud), s_hi) - qf) / TAU2
        y += DT * (qf - A_YW * (y - w0))
        yy[i] = y
    return yy, pref


def fit_s_hi():
    from scipy.optimize import least_squares

    def resid(x):
        r = []
        for path, t0, svt, kd, mr, mo in TESTS:
            sim, ref = window_replay(path, t0, float(x[0]))
            r.append((sim - ref)[::20])
        return np.concatenate(r)

    res = least_squares(resid, [0.85], bounds=([0.5], [1.15]), diff_step=0.02)
    rmse = np.sqrt(np.mean(res.fun ** 2))
    return float(res.x[0]), rmse


def cl_replica(path, t0, svt, kd, s_hi, wc=4.5, wo=10., wr=30., nd=12, tauf=0.24,
               mv_max=90., ckq=1.1, T=8.0, seed=1):
    """闭环复现: 控制器 = 机上原样 (旧表记账), 对象 = v4lite; 用记录 MV 预热 3s"""
    t, pv, mv = load(path)
    y0, w0, u_pre = pre_state(path, t0, s_hi)
    # --- 预热段: 记录 MV 驱动对象与 ESO ---
    i0 = np.searchsorted(t, t0 - 3.0)
    tt = np.arange(t[i0], t0, DT)
    idx = np.clip(np.searchsorted(t, tt, side='right') - 1, 0, len(t) - 1)
    upre = mv[idx]
    rng = np.random.default_rng(seed)
    nd_p = max(1, int(round(THETA / DT)))
    y = float(np.interp(t[i0], t, pv)); w = w0
    qf = q_true(float(upre[0]), s_hi)
    z1, z2 = y, 0.0
    buf = [float(np.interp(float(upre[0]), TAB.u_bp, TAB.q_bp))] * (nd + 1)
    vf = buf[0]
    uh = list(np.full(nd_p, float(upre[0])))
    for i in range(len(upre)):
        ym = y + rng.normal(0, TAB.sig_n)
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
        ud = uh[i]
        qf += DT * (q_true(float(ud), s_hi) - qf) / TAU2
        y += DT * (qf - A_YW * (y - w))
    # --- 阶跃 + 闭环 ---
    v1, v2 = z1, 0.0
    N = int(T / DT)
    yy = np.empty(N)
    for i in range(N):
        ym = y + rng.normal(0, TAB.sig_n)
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
        ud = uh[len(upre) + i]
        qf += DT * (q_true(float(ud), s_hi) - qf) / TAU2
        y += DT * (qf - A_YW * (y - w))
        yy[i] = y
    tt2 = np.arange(N) * DT
    cross = np.where(yy >= svt)[0]
    r = tt2[cross[0]] if len(cross) else np.inf
    o = max(yy.max() - svt, 0.0)
    return r, o


if __name__ == '__main__':
    s_hi, rmse = fit_s_hi()
    print(f'高段修正 s_hi = {s_hi:.3f}  (26 个加热窗口回放 RMSE = {rmse:.2f}°C)')
    print()
    print('G3 关卡: 13 个标注 kd 测试闭环复现 (控制器=机上原样)')
    print(f'{"工况":10s} {"kd":>4s} {"墙温":>4s} | {"仿真":>12s} | {"实测":>12s} | 方向')
    ok = 0; n = 0
    for path, t0, svt, kd, mr, mo in TESTS:
        if kd is None:
            continue
        y0, w0, _ = pre_state(path, t0, s_hi)
        r, o = cl_replica(path, t0, svt, kd, s_hi)
        punch_s = r < 2.2; punch_m = mr < 2.2
        hit = punch_s == punch_m
        ok += hit; n += 1
        rt = f'{r:5.2f}s/{o:4.1f}°' if np.isfinite(r) else f' 爬行/{o:4.1f}°'
        print(f'{y0:.0f}→{svt:.0f}  {kd:4.2f} {w0:4.0f} | {rt:>12s} | {mr:5.2f}s/{mo:4.1f}° | {"对" if hit else "错"}')
    print(f'\n方向命中 {ok}/{n}')
