#!/usr/bin/env python3
"""[10] 包络认证: 刹车预算单一律 在全部实测行为包络上的全局参数扫描

背景: G3 精确复刻路线退役 (五轮结构假设各自被数据证伪 -- 对象含 ≥4 个相互作用的
隐藏状态, 0.5s/2° 分辨率的精确闭环复刻超出经济建模边界)。改用包络认证:
包络成员 = 各轮拟合钉死的行为边界, [10] 参数须全员达标 (应力仿真哲学的模型不确定性版)。

控制律 [10] (诚实表 + 刹车预算阻尼, 单一连续律, 全局一组参数):
    vc = wc·(rv1−z1) − z2 − kd·v̂ − v̂²/(2·B̂),  v̂ = vd+z2,  B̂ = max(−z2, Bmin)
    反查与记账都用诚实表 (s_lo=0.492) => z2 是真 °C/s, B̂ 即免费刹车实时读数。

包络 (v1 时代参数; G5 上机终验在 v2 上做):
    元件 3 态: fast(te=0.11) / slow(te=0.242) / boost(te=0.242, β=0.42 全冷起)
    增益 3 档: ×0.85 / ×1.0 / ×1.15;  炉衬 2 态: 70 / 150 °C
    工况 3 个: 100→400 / 200→440 / 100→200  => 54 对象-工况 per 参数组
判据: 全包络 首超 ≤4°; 400 类可冲透成员的到达中位 ≤1.9s; 失败方向必须=偏慢。

★ 认证结论 (2026-07-22):
  价格表: 全包络首超 ≤3.8° -> 400 到达中位 3.96s (wc=3,kd=0.2,Bmin=10);
          ≤6° -> ~3.3s; ≤9° -> ~2.9s。
  两个硬结论:
  1) 刹车预算项对"墙态/靶点"的自适应完全成立 (衬 70 vs 150 结果几乎重合 --
     当初立项动机已解决); 失败方向全部=偏慢 (安全)。
  2) 但"元件快慢/增益 ±15%"这轴在冲刺前不可测, 迫使全局参数保守 -->
     "零表格 + 全包络保证"的价格 = 速度退回 ~4s 级 (第3档时代)。
  ==> 决策建议: [10] 以"已认证、有价格标签"状态入库封存; 生产维持第一代
      (配方 kd 三个数, 1.5~1.7s/≤3° 典型); 若未来需求把"参数统一"看得比
      速度重 (如多机队免整定运维), [10] 随时可启用 -- 代价已知无需再研究。
用法: python3 analysis/cert10_envelope.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin import Twin
from g2_g3_pipeline import P_WALL, S_LO, AMB

TAB = Twin()
DT = 0.01
S_HI_N = 1.0            # 诚实表高段 (控制器侧)
KQ_N = 0.35             # 缩水 (台阶饱和型), 控制器记账同用


def q_h(u, y=0.0, s_hi=S_HI_N, kq=KQ_N):
    base = float(np.interp(u, TAB.u_bp, TAB.q_bp))
    g = S_LO if u <= 25 else (s_hi if u >= 55 else S_LO + (s_hi - S_LO) * (u - 25) / 30)
    base *= g
    if kq > 0 and y > 315 and u > 30:
        ramp = min(max((y - 315) / 45, 0), 1)
        base *= 1 - kq * min(max((u - 30) / 30, 0), 1) * ramp
    return base


Q_GRID_U = np.linspace(0, 100, 201)


def make_inv(y):
    qq = np.array([q_h(u, y) for u in Q_GRID_U])
    return qq


def cl10(el, gain, w1_0, y0, svt, wc, kd, bmin, T=8.0, seed=1):
    """el: ('fast'|'slow'|'boost'); 对象两层墙 + 元件; 控制器 [10] 诚实表"""
    a1, b1, c1, d1, e1 = (P_WALL[k] for k in ('a1', 'b1', 'c1', 'd1', 'e1'))
    te_p = 0.11 if el == 'fast' else 0.242
    beta = 0.42 if el == 'boost' else 0.0
    rng = np.random.default_rng(seed)
    nd, tauf, wo, wr, mvmax = 12, 0.24, 10.0, 30.0, 90.0
    nd_p = 12
    w1, w2 = w1_0, max(w1_0 - 30, AMB)
    ce = 0.0 if el == 'boost' else 1.0
    # 初始: 闭环稳在 y0 (对象平衡)
    q_need = max(a1 * (y0 - w1), 0.0)
    y = y0
    # 求稳态 u: 真对象功率 = gain*q_h*boost0
    qq0 = np.array([gain * q_h(u, y0) for u in Q_GRID_U])
    u_ss = float(np.interp(q_need, qq0, Q_GRID_U))
    qf = q_need
    z1, z2 = y0, -q_need
    v1, v2 = y0, 0.0
    buf = [q_h(u_ss, y0)] * (nd + 1)
    vf = buf[0]
    uh = list(np.full(nd_p, u_ss))
    N = int(T / DT)
    yy = np.empty(N)
    v_hi_cache = {}
    for i in range(N):
        ym = y + rng.normal(0, TAB.sig_n)
        v1n = v1 + DT * v2
        v2n = v2 + DT * (-2 * wr * v2 - wr * wr * (v1 - svt))
        v1, v2 = v1n, v2n
        vd = buf[nd]
        vf += DT * (vd - vf) / tauf
        vd = vf
        err = ym - z1
        z1 += DT * (z2 + vd + 2 * wo * err)
        z2 += DT * (wo * wo * err)
        vhat = vd + z2
        bb = max(-z2, bmin)
        vc = wc * (v1 - z1) - z2 - kd * vhat - (vhat * abs(vhat)) / (2.0 * bb)
        yk = round(ym / 20) * 20
        if yk not in v_hi_cache:
            v_hi_cache[yk] = np.array([q_h(u, yk) for u in Q_GRID_U])
        qq = v_hi_cache[yk]
        vc = min(max(vc, 0.0), float(qq[-11]))
        u = float(np.interp(vc, qq, Q_GRID_U))
        u = min(max(u, 0.0), mvmax)
        buf = [float(np.interp(u, Q_GRID_U, qq))] + buf[:-1]
        uh.append(u)
        ud = float(uh[i])
        # 对象
        if ud > 2.0:
            ce = min(ce + DT * (1 - ce) / (0.35 * 90 / max(ud, 5)), 1.0)
        else:
            ce = max(ce - DT * ce / 103.0, 0.0)
        boost = 1.0 + beta * (1.0 - ce)
        q_cmd = gain * q_h(ud, y) * boost
        qf += DT * (q_cmd - qf) / (te_p / boost)
        y += DT * (qf - a1 * (y - w1))
        w1 += DT * (b1 * (y - w1) - c1 * (w1 - w2))
        w2 += DT * (d1 * (w1 - w2) - e1 * (w2 - AMB))
        yy[i] = y
    tt = np.arange(N) * DT
    cross = np.where(yy >= svt)[0]
    reach = tt[cross[0]] if len(cross) else np.inf
    ov = max(yy.max() - svt, 0.0)
    return reach, ov


CASES = [(100., 400.), (200., 440.), (100., 200.)]
ELS = ['fast', 'slow', 'boost']
GAINS = [0.85, 1.0, 1.15]
W1S = [70.0, 150.0]

if __name__ == '__main__':
    print('[10] 包络扫描: 54 对象-工况 × 参数网格')
    rows = []
    for wc in (3.0, 4.0, 5.0):
        for kd in (0.2, 0.4):
            for bmin in (5.0, 10.0):
                ovs = []; r400 = []; slowfail = True
                for el in ELS:
                    for g in GAINS:
                        for w1 in W1S:
                            for y0, svt in CASES:
                                r, o = cl10(el, g, w1, y0, svt, wc, kd, bmin)
                                ovs.append(o)
                                if svt == 400.:
                                    r400.append(r)
                    pass
                wov = max(ovs)
                finite = [r for r in r400 if np.isfinite(r)]
                med = np.median(finite) if finite else np.inf
                frac = len(finite) / len(r400)
                rows.append((wc, kd, bmin, wov, med, frac))
                print(f'  wc={wc:.0f} kd={kd:.1f} Bmin={bmin:.0f}: 最坏首超 {wov:5.1f}°'
                      f'  400类到达中位 {med:5.2f}s  冲透率 {frac*100:3.0f}%')
    ok = [r for r in rows if r[3] <= 4.0]
    ok.sort(key=lambda r: r[4])
    if ok:
        wc, kd, bmin, wov, med, frac = ok[0]
        print(f'\n最优全局参数组: wc={wc} kd={kd} Bmin={bmin} -> 最坏首超 {wov:.1f}°, 400 到达中位 {med:.2f}s')
        print('该组逐包络明细 (100→400):')
        for el in ELS:
            for g in GAINS:
                for w1 in W1S:
                    r, o = cl10(el, g, w1, 100., 400., wc, kd, bmin)
                    rt = f'{r:.2f}s' if np.isfinite(r) else '爬行'
                    print(f'   {el:5s} gain={g:.2f} 衬={w1:.0f}: {rt}/{o:.1f}°')
