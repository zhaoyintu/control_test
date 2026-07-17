#!/usr/bin/env python3
"""图 18: MV 上限与到达时间极限 -- "1.5s 到 400" 可行性判定 (2026-07-17)
左: 三条轨迹对比 (物理极限 bang-bang / wr=5.5 提速档 / 第3档);
中: 统一律价格表 (到达 vs 首超 pareto, 物理地板与 1.5s 目标标注);
右: 坏日子税 (冷炉/增益漂移下超调, 与档位选择正交)。
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'analysis'))
from rung4_scan import make_plant, sim_plant_step, bang_bang_floor, cl, mets, DT, R3
from twin import Twin

_f = font_manager.FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = _f.get_name()
plt.rcParams['axes.unicode_minus'] = False

tvp = make_plant()
tvc = Twin()

# ---- 左图数据: 三条轨迹 ----
reach_bb, ts_bb, ov_bb, ycut = bang_bang_floor(tvp, 90.0, 100.0, 400.0)
u_hold = tvp.steady_u(400.0)


_reached = [False]


def u_fn(t, y):
    if t < ts_bb:
        return 90.0
    if y >= 399.5:
        _reached[0] = True
    return u_hold if _reached[0] else 0.0


yy_bb = sim_plant_step(tvp, u_fn, 100.0, T=6.0)
t_bb = np.arange(len(yy_bb)) * DT

cfg_fast = dict(wc=1.5, wo=8.0, wr=5.5, nd=12, tauf=0.24, mv_max=60.0)
yy_f, _ = cl(tvc, tvp, **cfg_fast, y0=100.0, svt=400.0, ckq=1.1)
r_f, o_f, _, _ = mets(yy_f, 400.0)
yy_3, _ = cl(tvc, tvp, **R3, y0=100.0, svt=400.0, ckq=1.1)
r_3, o_3, _, _ = mets(yy_3, 400.0)
t_cl = np.arange(len(yy_3)) * DT

# ---- 中图数据: pareto (rung4_scan B 段的复算, 只跑前沿附近) ----
pareto = []
for wr, wo in [(3.5, 8.0), (3.5, 10.0), (3.5, 12.0), (4.5, 8.0), (4.5, 10.0),
               (5.5, 8.0), (6.5, 8.0), (8.0, 8.0)]:
    yy, _ = cl(tvc, tvp, wc=1.5, wo=wo, wr=wr, nd=12, tauf=0.24, mv_max=60.0,
               y0=100.0, svt=400.0, ckq=1.1)
    r, o, m, _ = mets(yy, 400.0)
    pareto.append((r, max(o, m), wr))
pareto.sort()

# ---- 右图数据: 坏日子税 ----
days = [('标称\n(暖炉)', {}, {}), ('冷炉体\nfw=0.12', {}, dict(fw=0.12)),
        ('增益\n×0.85', dict(gain=0.85), {}), ('冷晨组合', dict(gain=0.9, kq=0.35), dict(fw=0.12))]
bars3, barsf = [], []
for _, pk, ck in days:
    tvs = make_plant(**pk)
    yy, _ = cl(tvc, tvs, **R3, **ck, y0=100.0, svt=400.0, ckq=1.1)
    bars3.append(mets(yy, 400.0)[1])
    yy, _ = cl(tvc, tvs, **cfg_fast, **ck, y0=100.0, svt=400.0, ckq=1.1)
    barsf.append(mets(yy, 400.0)[1])

# ================= 画 =================
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
fig.suptitle('MV 上限与到达极限：1.5s 目标的物理判定（孪生按 7-16 实测 90% 马力修正, 100→400）',
             fontsize=13, y=0.99)

ax = axes[0]
ax.plot(t_bb, yy_bb, color='#888888', lw=1.8,
        label=f'物理极限(理想开关) 到达 {reach_bb:.2f}s')
ax.plot(t_cl, yy_f, color='#d62728', lw=1.8,
        label=f'提速档 wr=5.5: {r_f:.2f}s / 首超 {o_f:.1f}°')
ax.plot(t_cl, yy_3, color='#1f77b4', lw=1.8,
        label=f'第3档 wr=3.5: {r_3:.2f}s / 首超 {o_3:.1f}°')
i_cut = int(ts_bb / DT)
ax.axvspan(ts_bb, reach_bb, color='#bbbbbb', alpha=0.25)
ax.annotate(f'断电点 {ycut:.0f}°C', (ts_bb, yy_bb[i_cut]),
            xytext=(ts_bb + 0.55, yy_bb[i_cut] - 55), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#666666'))
ax.text(ts_bb + (reach_bb - ts_bb) / 2, 148, '滑行段：元件蓄热\n≈80°C 只能滑完',
        ha='center', fontsize=9, color='#555555',
        bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))
ax.axhline(400, color='k', ls=':', lw=0.8)
ax.axvline(1.5, color='#2ca02c', ls='--', lw=1.0)
ax.text(1.52, 120, '目标 1.5s', color='#2ca02c', fontsize=9, rotation=90)
ax.set_xlim(0, 6); ax.set_ylim(90, 430)
ax.set_xlabel('时间 (s)'); ax.set_ylabel('温度 (°C)')
ax.set_title('即使理想开关也要 1.6s——瓶颈是刹车不是马力', fontsize=10.5)
ax.legend(fontsize=8.5, loc='lower right')

ax = axes[1]
rr = [p[0] for p in pareto]; oo = [p[1] for p in pareto]
ax.plot(rr, oo, 'o-', color='#1f77b4', lw=1.5, ms=5)
for r, o, wr in pareto:
    ax.annotate(f'wr={wr:g}', (r, o), xytext=(4, 5), textcoords='offset points', fontsize=8)
ax.axhspan(0, 3, color='#2ca02c', alpha=0.12)
ax.text(2.95, 1.4, '超调 ≤3° 预算区', color='#2ca02c', fontsize=9)
ax.axvline(1.55, color='#888888', ls='--', lw=1.2)
ax.text(1.57, 7.6, '物理地板 1.55~1.61s\n(理想开关, 零裕度)', fontsize=8.5, color='#555555')
ax.axvline(1.5, color='#2ca02c', ls='--', lw=1.0)
ax.text(1.36, 4.2, '目标\n1.5s', color='#2ca02c', fontsize=9, ha='center')
ax.annotate('统一律再快就是悬崖\n(1.38s → 首超 39°)', (1.94, 6.5), xytext=(2.6, 8.0),
            fontsize=8.5, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728'))
ax.set_xlim(1.3, 3.1); ax.set_ylim(0, 9.5)
ax.set_xlabel('到达 400°C 时间 (s)'); ax.set_ylabel('首超 (°C)')
ax.set_title('价格表：买 0.5s 要付 ~3° 首超（超调后单调回落, 无持续振荡）', fontsize=10.5)

ax = axes[2]
x = np.arange(len(days)); w = 0.36
ax.bar(x - w / 2, bars3, w, color='#1f77b4', label='第3档 (2.6s)')
ax.bar(x + w / 2, barsf, w, color='#d62728', label='提速档 wr=5.5 (2.1s)')
for xi, (b3, bf) in enumerate(zip(bars3, barsf)):
    ax.text(xi - w / 2, b3 + 0.25, f'{b3:.1f}', ha='center', fontsize=8.5)
    ax.text(xi + w / 2, bf + 0.25, f'{bf:.1f}', ha='center', fontsize=8.5)
ax.axhline(3, color='#2ca02c', ls='--', lw=1.0)
ax.text(-0.45, 3.3, '3° 预算', color='#2ca02c', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([d[0] for d in days], fontsize=9)
ax.set_ylabel('首超 (°C)')
ax.set_title('坏日子税：冷炉/漂移再加 5~8°, 与档位选择正交', fontsize=10.5)
ax.legend(fontsize=9)

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(HERE, '18_0717_MV上限与到达极限.png')
fig.savefig(out, dpi=140)
print('saved:', out)
