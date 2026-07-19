#!/usr/bin/env python3
"""孪生 v4: 双热容(腔体+墙)模型 -- 由 7-19 两场数据辨识 (2026-07-19)

背景 (7-19 kd 全扫描暴露的问题):
  固定 kd 只能整定单一(靶点,炉态); 悬崖随墙温游走 ±0.1; 200 靶点热/冷炉体
  首超 2.6°~18° 天差地别 -- 单热容孪生 (h1,c=127) 无墙状态, 阻尼轴系统性偏软。

模型结构:
  MV --θ--> q_true(u) --τe 一阶--> qf --> 腔体 y --a_yw--> 墙 w --b_wa--> 环境
    dy/dt = qf - a_yw*(y-w)                (腔体损失全部经墙)
    dw/dt = b_yw*(y-w) - b_wa*(w-amb)      (墙慢充放)
  q_true(u) = q_tab(u)*gain(u): 旧表中低段携带旧散热模型的补偿虚增
    (保400: 表推 40°C/s, 实测自由降温仅 ~21°C/s), 高段近实。
    gain(u) = s_lo (u<=25) 线性过渡到 s_hi (u>=55)。

辨识路径与状态 (2026-07-19 晚, ★=WIP 未过关):
  1) 降温段拟合 (不沾 q 表, ≤60s 窗): RMSE 1.82° -- 快耦合 a_yw≈0.0836 可信,
     免费刹车地图 B = a_yw·(y−w) 成立; 但慢参数 b_yw/b_wa 在 60s 窗内不可辨识
     (被各段自由 w0 吸收), 观测器长程积分会把墙"放空"到与拟合 w0 矛盾的水平。
  2) ★ 加热窗口拟 s_lo/s_hi: RMSE 20.4° 未收敛 (慢墙参数错 -> 各窗 w0 错 ->
     增益补偿混乱; 且 v4 plant 尚未并入高段温度缩水)。
  3) ★ validate_cl(): 13 个标注 kd 测试闭环复现 -- 未通过 (全爬行, kd 响应反向)。
     ==> 关卡结论: 在慢墙参数拿到可信数据前, 不在 v4 上做任何控制设计。
  补数据清单 (解锁慢参数): 2~3 段 ≥10min 长降温 (不同墙态起点) + 200/300/400
  各 ≥60s 稳态保温 (锚真实保温功率 vs 墙态); 最好含一次隔夜冷却记录。

用法: python3 analysis/wall_twin_v4.py [heats|cl s_lo s_hi]
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin import Twin

DT = 0.01
F_AM = os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260719-163352_164130.csv')
F_PM = os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260719-181948_191244.csv')

# ---- 降温段拟合结果 (fit_falls; 固定, 见文件头) ----
P_WALL = dict(a_yw=0.08361, b_yw=0.004197, b_wa=0.005965, amb=15.0)


def load(path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df.iloc[:, 0])
    t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    return t, df.iloc[:, 1].to_numpy(float), df.iloc[:, 3].to_numpy(float), df.iloc[:, 4].to_numpy(float)


class TwinWall:
    """v4 对象: 元件(θ/τe/q_true) + 腔体 + 墙。theta/tau2 沿用 v2/v3 闭环锚定值。"""

    def __init__(self, s_lo=0.5, s_hi=0.95, wall=P_WALL, theta=0.12, tau2=0.242):
        base = Twin()
        self.u_bp, self.q_bp = base.u_bp, base.q_bp
        self.theta, self.tau2 = theta, tau2
        self.s_lo, self.s_hi = s_lo, s_hi
        self.a_yw, self.b_yw = wall['a_yw'], wall['b_yw']
        self.b_wa, self.amb = wall['b_wa'], wall['amb']
        self.sig_n = base.sig_n

    def gain(self, u):
        if u <= 25.0:
            return self.s_lo
        if u >= 55.0:
            return self.s_hi
        return self.s_lo + (self.s_hi - self.s_lo) * (u - 25.0) / 30.0

    def q_true(self, u):
        return float(np.interp(u, self.u_bp, self.q_bp)) * self.gain(u)

    def hold_q(self, y, w):
        return self.a_yw * (y - w)

    def steady_u(self, y, w):
        """当前墙温下保 y 所需 MV (真物理)"""
        q = max(self.hold_q(y, w), 0.0)
        # 反解 q_true(u)=q: 粗网格+插值足够
        uu = np.linspace(0, 100, 401)
        qq = np.array([self.q_true(u) for u in uu])
        return float(np.interp(q, qq, uu))

    def replay(self, u_seq, y0, w0, dt=DT, sub=1):
        """开环回放: 记录 MV -> (y, w) 轨迹"""
        n = len(u_seq)
        nd = max(1, int(round(self.theta / dt)))
        y, w = y0, w0
        qf = 0.0
        yy = np.empty(n); ww = np.empty(n)
        for i in range(n):
            ud = u_seq[max(0, i - nd)]
            qf += dt * (self.q_true(ud) - qf) / self.tau2
            dy = qf - self.a_yw * (y - w)
            dw = self.b_yw * (y - w) - self.b_wa * (w - self.amb)
            y += dt * dy
            w += dt * dw
            yy[i] = y; ww[i] = w
        return yy, ww


def wall_traj(t, pv, w0=140.0, wall=P_WALL):
    """墙观测器: 由实测 PV 驱动 (不会发散; 在线可实现 -- 这就是 [10] 的观测器形式)
       dw = b_yw*(PV - w) - b_wa*(w - amb)"""
    dt_ = np.diff(t, prepend=t[0])
    w = np.empty(len(t)); w[0] = w0
    byw, bwa, amb = wall['b_yw'], wall['b_wa'], wall['amb']
    for i in range(1, len(t)):
        w[i] = w[i-1] + dt_[i] * (byw * (pv[i-1] - w[i-1]) - bwa * (w[i-1] - amb))
    return w


def heat_windows(t, sv, min_amp=50.0):
    """所有上行阶跃窗口 (t0, svt, t_end)"""
    j = np.where(np.diff(sv) != 0)[0] + 1
    out = []
    for k, i in enumerate(j):
        if sv[i] - sv[i-1] >= min_amp:
            t_end = t[j[k+1]] if k + 1 < len(j) else t[-1]
            out.append((t[i], sv[i], min(t_end, t[i] + 14.0)))
    return out


def fit_heats():
    """在每个加热窗口内做短回放 (w0 来自墙观测器), 拟 s_lo/s_hi"""
    from scipy.optimize import least_squares
    data = []
    for path in (F_AM, F_PM):
        t, pv, sv, mv = load(path)
        w = wall_traj(t, pv)
        for t0, svt, t_end in heat_windows(t, sv):
            i0 = np.searchsorted(t, t0 - 0.5)   # 提前 0.5s, 覆盖元件预热
            i1 = np.searchsorted(t, t_end)
            tt = np.arange(t[i0], t[i1], DT)
            idx = np.clip(np.searchsorted(t[i0:i1], tt - t[i0] + t[i0], side='right') - 1 + i0, i0, i1 - 1)
            useq = mv[idx]
            pref = np.interp(tt, t[i0:i1], pv[i0:i1])
            data.append((useq, pref, w[i0]))
    print(f'{len(data)} 个加热窗口')

    def resid(x):
        s_lo, s_hi = x
        tw = TwinWall(s_lo=s_lo, s_hi=s_hi)
        r = []
        for useq, pref, w0 in data:
            yy, _ = tw.replay(useq, pref[0], w0)
            r.append((yy - pref)[::20])          # 0.2s 栅格
        return np.concatenate(r)

    res = least_squares(resid, [0.5, 0.95], bounds=([0.2, 0.5], [1.0, 1.3]),
                        diff_step=0.02, xtol=1e-9)
    s_lo, s_hi = res.x
    rmse = np.sqrt(np.mean(res.fun ** 2))
    print(f'加热窗口拟合: RMSE={rmse:.2f}°C  s_lo={s_lo:.3f}  s_hi={s_hi:.3f}')
    return s_lo, s_hi, rmse


def cl_wall(tw, ctrl_tab, wc, wo, wr, nd, tauf, mv_max, y0, w0, svt,
            kd=0.0, ckq=1.1, ctref=315.0, T=10.0, pre=3.0, seed=1, qf0=None, u_pre=None):
    """闭环: 控制器按机上旧表记账 (ctrl_tab=Twin()), 对象 = TwinWall"""
    n_pre = int(pre / DT)
    N = n_pre + int(T / DT)
    nd_p = max(1, int(round(tw.theta / DT)))
    rng = np.random.default_rng(seed)
    v_hi = float(np.interp(mv_max, ctrl_tab.u_bp, ctrl_tab.q_bp))
    u_ss = tw.steady_u(y0, w0) if u_pre is None else u_pre
    v_ss = float(np.interp(u_ss, ctrl_tab.u_bp, ctrl_tab.q_bp))
    y, w = y0, w0
    qf = tw.q_true(u_ss) if qf0 is None else qf0
    z1, z2 = y0, 0.0
    v1, v2 = y0, 0.0
    buf = [v_ss] * (nd + 1)
    vf = v_ss
    uh = np.full(N + nd_p + 2, u_ss)
    yy = np.zeros(N)
    for i in range(N):
        svr = y0 if i < n_pre else svt
        ym = y + rng.normal(0, tw.sig_n)
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
        vc = min(max(wc * (v1 - z1) - z2 - kd * (vd + z2), 0.0), v_hi)
        u = min(max(float(np.interp(vc, ctrl_tab.q_bp, ctrl_tab.u_bp)), 0.0), mv_max)
        va = float(np.interp(u, ctrl_tab.u_bp, ctrl_tab.q_bp))
        if ckq > 0 and ym > ctref and u > 30.0:
            g = 1.0 - ckq * min(max((u - 30.0) / 30.0, 0.0), 1.0) * (ym - ctref) / 100.0
            va *= max(g, 0.4)
        buf = [va] + buf[:-1]
        uh[i + nd_p] = u
        ud = float(uh[i])
        qf += DT * (tw.q_true(ud) - qf) / tw.tau2
        y += DT * (qf - tw.a_yw * (y - w))
        w += DT * (tw.b_yw * (y - w) - tw.b_wa * (w - tw.amb))
        yy[i] = y
    return yy[n_pre:]


def validate_cl(s_lo, s_hi):
    """9 个标注 kd 测试 (+早场 3 个) 的闭环复现 vs 实测"""
    ctrl = Twin()
    tw = TwinWall(s_lo=s_lo, s_hi=s_hi)
    CARD = dict(wc=4.5, wo=10.0, wr=30.0, nd=12, tauf=0.24, mv_max=90.0)
    tests = []
    # (文件, t0, y0标称, svt, kd, 实测reach, 实测ov)
    tests += [(F_AM, 30.8, 100., 200., 1.1, 0.99, 2.6), (F_AM, 53.4, 200., 440., 1.1, 1.59, 3.0),
              (F_AM, 201.3, 100., 400., 0.9, 3.30, 0.5), (F_AM, 422.9, 100., 400., 0.9, 1.57, 2.5)]
    tests += [(F_PM, 1497.3, 200., 440., 1.2, 3.12, 0.4), (F_PM, 1688.7, 100., 400., 1.2, 3.36, 0.6),
              (F_PM, 1999.4, 100., 400., 0.95, 3.19, 1.0), (F_PM, 2125.1, 100., 400., 0.85, 1.71, 1.0),
              (F_PM, 2418.8, 100., 400., 0.75, 1.54, 4.2), (F_PM, 2523.8, 200., 440., 0.75, 1.38, 10.7),
              (F_PM, 2685.5, 200., 440., 0.9, 1.42, 5.6), (F_PM, 2973.8, 200., 440., 1.0, 1.41, 4.4),
              (F_PM, 3128.3, 200., 440., 1.1, 3.33, 0.3)]
    cache = {}
    print(f'{"工况":10s} {"kd":>4s} {"墙温":>5s} | {"仿真":>12s} | {"实测":>12s}')
    for path, t0, y0, svt, kd, mr, mo in tests:
        if path not in cache:
            t, pv, sv, mv = load(path)
            cache[path] = (t, pv, sv, mv, wall_traj(t, pv))
        t, pv, sv, mv, w = cache[path]
        i0 = np.searchsorted(t, t0)
        w0 = w[i0]
        # 元件预热初值: 取阶跃前 0.4~0.1s 的实测 MV 均值近似 qf0
        j0 = np.searchsorted(t, t0 - 0.4); j1 = np.searchsorted(t, t0 - 0.05)
        u_pre = float(mv[j0:j1].mean()) if j1 > j0 else 0.0
        yy = cl_wall(tw, ctrl, **CARD, y0=float(pv[i0]), w0=float(w0), svt=svt,
                     kd=kd, qf0=tw.q_true(u_pre), u_pre=u_pre)
        tt = np.arange(len(yy)) * DT
        cross = np.where(yy >= svt)[0]
        r = tt[cross[0]] if len(cross) else np.inf
        o = max(yy.max() - svt, 0.0)
        rt = f'{r:5.2f}s/{o:4.1f}°' if np.isfinite(r) else f' 爬行/{o:4.1f}°'
        mt = f'{mr:5.2f}s/{mo:4.1f}°'
        print(f'{y0:.0f}→{svt:.0f}  {kd:4.2f} {w0:5.0f} | {rt:>12s} | {mt:>12s}')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'heats'
    if mode == 'heats':
        s_lo, s_hi, _ = fit_heats()
        validate_cl(s_lo, s_hi)
    elif mode == 'cl':
        validate_cl(float(sys.argv[2]), float(sys.argv[3]))
