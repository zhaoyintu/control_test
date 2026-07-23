#!/usr/bin/env python3
"""开表后(v空间)参数爬梯 @ 孪生 v2: 标称扫描 -> 短名单应力测试"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin import Twin, metrics

BASE = Twin()
VMAX = float(BASE.q_of(80.0))     # 机器 MV 限幅 80% 对应的 v 上限


def vplant(gain=1.0, theta=None, tau2=None):
    tv = Twin()
    tv.u_bp = np.array([0.0, 500.0])
    tv.q_bp = np.array([0.0, 500.0 * gain])   # 完美查表 => v 空间增益 gain(=表误差)
    if theta is not None: tv.theta = theta
    if tau2 is not None: tv.tau2 = tau2
    return tv


def run(tv, wc, wo, wr, nd, y0=100.0, svt=400.0, T=20.0, seed=1):
    yy, uu = tv.closed_loop_adrc(1.0, wc, wo, wr, nd, y0, svt, T=T, seed=seed, mv_max=VMAX)
    r90, ov, stl = metrics(yy, y0, svt)
    jit = float(np.std(np.diff(uu[-500:])))   # 末 5s v 指令抖动
    return r90, ov, stl, jit

# ---------- 标称粗扫 ----------
rows = []
for wc in (1.5, 2.0, 2.5, 3.0):
    for wo in (6.0, 8.0, 10.0, 12.0):
        for wr in (3.0, 4.0, 5.0, 6.0):
            for nd in (12, 16, 20, 24, 28):
                r90, ov, stl, jit = run(vplant(), wc, wo, wr, nd)
                if ov <= 1.0 and np.isfinite(stl):
                    rows.append((stl, r90, ov, jit, wc, wo, wr, nd))
rows.sort()
print('标称合格 (ov<=1) 前 12 名  [settle | rise90 | ov | v抖动 | wc wo wr nd]:')
for stl, r90, ov, jit, wc, wo, wr, nd in rows[:12]:
    print(f'  {stl:5.2f}s {r90:5.2f}s {ov:4.2f}° {jit:5.2f}  wc={wc} wo={wo} wr={wr} nd={nd}')

# ---------- 短名单应力 ----------
print('\n应力测试 (增益x1.15 / x0.85 / 全应力 theta=0.18 tau2=0.32 gain=1.15 / 450目标 / 300->400):')
seen = set()
short = []
for row in rows:
    key = row[4:7]          # wc wo wr 不同才算新条目, nd 取各自最优
    if key not in seen:
        seen.add(key)
        short.append(row)
    if len(short) >= 6:
        break
for stl, r90, ov, jit, wc, wo, wr, nd in short:
    a = run(vplant(gain=1.15), wc, wo, wr, nd)
    b = run(vplant(gain=0.85), wc, wo, wr, nd)
    c = run(vplant(gain=1.15, theta=0.18, tau2=0.32), wc, wo, wr, nd)
    d = run(vplant(), wc, wo, wr, nd, y0=100, svt=450)
    e = run(vplant(), wc, wo, wr, nd, y0=300, svt=400)
    worst_ov = max(a[1], b[1], c[1])
    worst_st = max(a[2], b[2], c[2])
    print(f'wc={wc} wo={wo} wr={wr} nd={nd}: 标称 {r90:.2f}/{ov:.1f}°/{stl:.2f}s | '
          f'应力最差 ov={worst_ov:.1f}° settle={worst_st:.2f}s | '
          f'450: {d[0]:.2f}/{d[1]:.1f}°/{d[2]:.2f}s | 300→400: {e[0]:.2f}/{e[1]:.1f}°/{e[2]:.2f}s')
