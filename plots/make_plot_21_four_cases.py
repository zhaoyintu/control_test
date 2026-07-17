#!/usr/bin/env python3
"""图 21: 第4档四工况仿真 (100→200 / 100→400 / 200→400 / 200→450, 三炉况)
边界工况附拨盘对照: 200 靶点加阻尼 kd=0.85, 450 拖炉况减阻尼 kd=0.70。
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
from twin import Twin
from rung4_scan import cl, mets, DT
from pid_race_0717 import plant_day

_f = font_manager.FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = _f.get_name()
plt.rcParams['axes.unicode_minus'] = False

tvc = Twin()
BASE = dict(wc=4.5, wo=10.0, wr=30.0, nd=12, tauf=0.24, mv_max=90.0)
DAY3 = [('今晚炉况(拖)', 'tonight', '#1f77b4'), ('标准炉况', 'standard', '#2ca02c'),
        ('暖午炉况', 'warm', '#d62728')]

fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.4))
fig.suptitle('第4档四工况仿真：wc=4.5 wo=10 wr=30 nd=12 lrTauF=0.24 lrKd=0.75 lrMVMax=90'
             '（虚线 = 边界工况的拨盘对照）', fontsize=12.5, y=0.985)

cases = [
    (axes[0][0], 100., 200., '(a) 100→200：低温免费刹车弱 → 首超 3.3~6.0° 超预算', (90, 222)),
    (axes[0][1], 100., 400., '(b) 100→400：主战工况，1.50~1.59s / ≤2.8°', (90, 432)),
    (axes[1][0], 200., 400., '(c) 200→400：1.24~1.30s / ≤2.5°', (188, 432)),
    (axes[1][1], 200., 450., '(d) 200→450：快而稳，唯拖炉况严格碰线慢', (188, 482)),
]
for ax, y0, svt, title, ylim in cases:
    uu_std = None
    for nm, d, col in DAY3:
        yy, uu = cl(tvc, plant_day(d), **BASE, kd=0.75, y0=y0, svt=svt, ckq=1.1)
        r, o, m, _ = mets(yy, svt)
        tt = np.arange(len(yy)) * DT
        rt = f'{r:.2f}s' if np.isfinite(r) else '渐近'
        ax.plot(tt, yy, color=col, lw=1.7, label=f'{nm} {rt}/{max(o,m):.1f}°')
        if d == 'standard':
            uu_std = uu
    # 拨盘对照虚线
    if svt == 200.:
        yy, _ = cl(tvc, plant_day('standard'), **BASE, kd=0.85, y0=y0, svt=svt, ckq=1.1)
        r, o, m, _ = mets(yy, svt)
        ax.plot(np.arange(len(yy)) * DT, yy, color='#2ca02c', lw=1.4, ls='--',
                label=f'拨盘 kd=0.85(标准) {r:.2f}s/{max(o,m):.1f}°')
    if svt == 450.:
        yy, _ = cl(tvc, plant_day('tonight'), **BASE, kd=0.70, y0=y0, svt=svt, ckq=1.1)
        r, o, m, _ = mets(yy, svt)
        ax.plot(np.arange(len(yy)) * DT, yy, color='#1f77b4', lw=1.4, ls='--',
                label=f'拨盘 kd=0.70(拖炉况) {r:.2f}s/{max(o,m):.1f}°')
    ax.axhline(svt, color='k', ls=':', lw=0.8)
    ax.axhline(svt + 3, color='#d62728', ls=':', lw=0.7)
    ax.text(3.75, svt + 3.5, '超调预算 +3°', fontsize=8, color='#d62728')
    ax2 = ax.twinx()
    ax2.fill_between(np.arange(len(uu_std)) * DT, 0, uu_std,
                     color='#ff7f0e', alpha=0.20, step='post')
    ax2.set_ylim(0, 260); ax2.set_yticks([0, 50, 100])
    ax2.set_ylabel('MV (%) [标准炉况]', color='#b25000', fontsize=8.5)
    ax.set_xlim(0, 5); ax.set_ylim(*ylim)
    ax.set_xlabel('阶跃后时间 (s)'); ax.set_ylabel('温度 (°C)')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='lower right')

axes[0][0].annotate('免费刹车(散热)在 200°C 只有 400°C 的 1/3\n→ 同样 kd 阻尼不足 → 超调放大',
                    xy=(1.15, 206), xytext=(1.9, 150), fontsize=8.6,
                    arrowprops=dict(arrowstyle='->', color='0.4'))
axes[1][1].annotate('拖炉况: 散热+高段缩水双重强刹\n→ 动量耗尽, 448 后渐近爬 (±2°带 2.2s 已进)',
                    xy=(2.6, 448.5), xytext=(1.7, 412), fontsize=8.6,
                    arrowprops=dict(arrowstyle='->', color='0.4'))

fig.tight_layout(rect=(0, 0, 1, 0.965))
out = os.path.join(HERE, '21_0717_第4档四工况.png')
fig.savefig(out, dpi=140)
print('saved:', out)
