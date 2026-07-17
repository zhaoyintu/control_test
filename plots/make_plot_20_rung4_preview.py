#!/usr/bin/env python3
"""图 20: 第4档仿真效果预览 (上机前的"预告片")
(a)(b)(c) 三个测试工况: 三炉况 PV + 标准炉况 MV 打法;
(d) 100→400 穿越区放大: 超调预算带 + PID 实测参考。
"""
import os
import sys
import numpy as np
import pandas as pd
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
CARD = dict(wc=4.5, wo=10.0, wr=30.0, nd=12, tauf=0.24, mv_max=90.0, kd=0.75)
DAY3 = [('今晚炉况(拖)', 'tonight', '#1f77b4'), ('标准炉况', 'standard', '#2ca02c'),
        ('暖午炉况', 'warm', '#d62728')]

fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
fig.suptitle('第4档仿真效果预览：wc=4.5 wo=10 wr=30 nd=12 lrTauF=0.24 lrKd=0.75 lrMVMax=90'
             '（lrKqHot=1.1 不动）', fontsize=13, y=0.985)

cases = [(axes[0][0], 100.0, 400.0, '(a) 100→400'),
         (axes[0][1], 200.0, 400.0, '(b) 200→400'),
         (axes[1][0], 200.0, 440.0, '(c) 200→440')]
for ax, y0, svt, lab in cases:
    txt = []
    uu_std = None
    for nm, d, col in DAY3:
        yy, uu = cl(tvc, plant_day(d), **CARD, y0=y0, svt=svt, ckq=1.1)
        r, o, m, _ = mets(yy, svt)
        tt = np.arange(len(yy)) * DT
        ax.plot(tt, yy, color=col, lw=1.7, label=f'{nm} {r:.2f}s/{max(o,m):.1f}°')
        if d == 'standard':
            uu_std = uu
    ax.axhline(svt, color='k', ls=':', lw=0.8)
    ax.axhline(svt + 3, color='#d62728', ls=':', lw=0.7)
    ax2 = ax.twinx()
    ax2.fill_between(np.arange(len(uu_std)) * DT, 0, uu_std,
                     color='#ff7f0e', alpha=0.22, step='post')
    ax2.set_ylim(0, 250); ax2.set_yticks([0, 50, 100])
    ax2.set_ylabel('MV (%)  [标准炉况]', color='#b25000', fontsize=9)
    ax.set_xlim(0, 5); ax.set_ylim(y0 - 15, svt + 35)
    ax.set_xlabel('阶跃后时间 (s)'); ax.set_ylabel('温度 (°C)')
    ax.set_title(f'{lab}：满幅冲刺 → 自动急降 → 低位接住（与 PID 打法同构, 但由单一律生成）',
                 fontsize=9.8)
    ax.legend(fontsize=8.5, loc='lower right')

# (d) 穿越区放大 + PID 参考
ax = axes[1][1]
df = pd.read_csv(os.path.join(HERE, '..', 'user_feedback',
                              'AIC9_DATA-20260717-212412_212746.csv'))
ts = pd.to_datetime(df.iloc[:, 0])
t_all = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
pv_all = df.iloc[:, 1].to_numpy(float)
i0 = np.searchsorted(t_all, 21.0); i1 = np.searchsorted(t_all, 27.0)
ax.plot(t_all[i0:i1] - 21.0, pv_all[i0:i1], color='0.55', lw=1.6, ls='--',
        label='PID 实测(今晚) 1.48s/3.0°')
for nm, d, col in DAY3:
    yy, _ = cl(tvc, plant_day(d), **CARD, y0=100.0, svt=400.0, ckq=1.1)
    tt = np.arange(len(yy)) * DT
    ax.plot(tt, yy, color=col, lw=1.7, label=f'第4档 {nm}')
ax.axhspan(400, 403, color='#2ca02c', alpha=0.13)
ax.axhline(400, color='k', ls=':', lw=0.8)
ax.axhline(403, color='#d62728', ls=':', lw=0.8)
ax.text(3.55, 403.6, '超调预算 +3°', fontsize=8.5, color='#d62728')
ax.text(0.6, 396.5, '±2° 带 1.6~1.9s 进入,\n之后单调收敛无振荡', fontsize=8.8)
ax.set_xlim(0.5, 5); ax.set_ylim(383, 410)
ax.set_xlabel('阶跃后时间 (s)'); ax.set_ylabel('温度 (°C)')
ax.set_title('(d) 100→400 穿越区放大：首超 1.5~2.8°, 全部压在预算带内', fontsize=9.8)
ax.legend(fontsize=8.5, loc='lower right')

fig.tight_layout(rect=(0, 0, 1, 0.965))
out = os.path.join(HERE, '20_0717_第4档仿真效果预览.png')
fig.savefig(out, dpi=140)
print('saved:', out)
