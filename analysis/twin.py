"""
数字孪生 (Python 仿真系统) —— 由两场真机数据全波形辨识得到, 参数在 twin_params.json
模型结构:
    MV --纯滞后θ--> q(u) 查表(分段线性) --一阶惯性τ2--> 净加热速率 --积分--> PV
    dy/dt = qf - h1*(y - c)          (线性散热; c 为等效环境温度)
    测量: PV = y + N(0, σ_n)
用法:
    from twin import Twin
    tw = Twin()                                   # 读 analysis/twin_params.json
    y = tw.open_loop(u_seq, y0)                   # 开环: 给 MV 序列出 PV
    yy, uu = tw.closed_loop_adrc(b0,wc,wo,wr,nd, y0, sv, T=18)   # 闭环 ADRC
"""
import json
import os
import numpy as np
from scipy.signal import lfilter

DT = 0.01
HERE = os.path.dirname(os.path.abspath(__file__))


class Twin:
    def __init__(self, path=None):
        with open(path or os.path.join(HERE, 'twin_params.json')) as f:
            p = json.load(f)
        self.u_bp = np.array(p['u_bp'])
        self.q_bp = np.array(p['q_bp'])
        self.theta = p['theta']
        self.tau2 = p['tau2']
        self.h1 = p['h1']
        self.c = p['c']
        self.sig_n = p.get('sig_n', 0.0)
        # v3: 高段马力温度缩水 q_eff = q(u)·(1 − kq·s(u)·max(T−tref,0)/100)
        #     s(u): u<=30% 为 0, u>=60% 为 1, 之间线性 (低段锚点实测无缩水, 90%满幅实测 −18%)
        self.kq_hot = p.get('kq_hot', 0.0)
        self.kq_tref = p.get('kq_tref', 150.0)

    def q_of(self, u):
        return np.interp(u, self.u_bp, self.q_bp)

    def q_eff(self, u, y):
        """瞬时马力: 静态表 × 高段温度缩水系数 (kq_hot=0 时退化为 q_of)"""
        base = float(np.interp(u, self.u_bp, self.q_bp))
        if self.kq_hot <= 0.0:
            return base
        s = min(max((u - 30.0) / 30.0, 0.0), 1.0)
        g = 1.0 - self.kq_hot * s * max(y - self.kq_tref, 0.0) / 100.0
        return base * max(g, 0.4)

    def u_of_q(self, v):
        return float(np.interp(v, self.q_bp, self.u_bp))

    def steady_u(self, y):
        return self.u_of_q(self.h1 * (y - self.c))

    def open_loop(self, u_seq, y0, qf0=None, dt=DT):
        """开环仿真: 已知 MV 序列 -> PV 轨迹 (kq_hot=0 时全向量化, 否则逐拍)"""
        t = np.arange(len(u_seq)) * dt
        u_d = np.interp(t - self.theta, t, u_seq)          # 纯滞后
        a = dt / max(self.tau2, 1e-4)
        qf0 = self.q_of(u_seq[0]) if qf0 is None else qf0
        if self.kq_hot <= 0.0:
            q_in = self.q_of(u_d)
            qf, _ = lfilter([a], [1, -(1 - a)], q_in, zi=[(1 - a) * qf0])
            b = dt * self.h1
            y, _ = lfilter([dt], [1, -(1 - b)], qf + self.h1 * self.c, zi=[(1 - b) * y0])
            return y
        qf = qf0; y = y0
        out = np.empty(len(u_seq))
        for i in range(len(u_seq)):
            qf += a * (self.q_eff(u_d[i], y) - qf)
            y += dt * (qf - self.h1 * (y - self.c))
            out[i] = y
        return out

    def closed_loop_adrc(self, b0, wc, wo, wr, nd, y0, svt, T=18.0, pre=6.0,
                         noise=None, seed=1, mv_max=100.0, dt=DT):
        """闭环: FB_ADRC_Base 逐行复刻 + 孪生对象"""
        noise = self.sig_n if noise is None else noise
        n_pre = int(pre / dt)
        N = n_pre + int(T / dt)
        nd_p = max(1, int(round(self.theta / dt)))
        rng = np.random.default_rng(seed)
        u_ss = self.steady_u(y0)
        y = y0
        qf = self.h1 * (y0 - self.c)
        z1, z2 = y0, 0.0
        v1, v2 = y0, 0.0
        buf = [u_ss] * (nd + 1)
        uh = np.full(N + nd_p + 2, u_ss)
        yy = np.zeros(N); uu = np.zeros(N)
        for i in range(N):
            svr = y0 if i < n_pre else svt
            ym = y + (rng.normal(0, noise) if noise else 0.0)
            v1n = v1 + dt * v2
            v2n = v2 + dt * (-2 * wr * v2 - wr * wr * (v1 - svr))
            v1, v2 = v1n, v2n
            vd = buf[nd]; err = ym - z1
            z1 = z1 + dt * (z2 + b0 * vd + 2 * wo * err)
            z2 = z2 + dt * (wo * wo * err)
            u = min(max((wc * (v1 - z1) - z2) / b0, 0.0), mv_max)
            buf = [u] + buf[:-1]
            uh[i + nd_p] = u
            ud = uh[i]
            for _ in range(2):
                h_ = dt / 2
                qf += h_ * (self.q_eff(float(ud), y) - qf) / max(self.tau2, 1e-4)
                y += h_ * (qf - self.h1 * (y - self.c))
            yy[i] = y; uu[i] = u
        return yy[n_pre:], uu[n_pre:]


def metrics(yy, y0, tgt, dt=DT, band=1.0):
    tt = np.arange(len(yy)) * dt
    amp = tgt - y0
    idx = np.where(yy - y0 >= 0.9 * amp)[0]
    r90 = tt[idx[0]] if len(idx) else np.nan
    ov = max(yy.max() - tgt, 0.0)
    bad = np.where(np.abs(yy - tgt) > band)[0]
    stl = tt[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(yy) else np.nan
    return r90, ov, stl
