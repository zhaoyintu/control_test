#!/usr/bin/env python3
"""图 19: 7-17 晚 PID 对照场与第4档设计 (lrKd 速度阻尼)
左: PID 实测解剖 (1.48s/3.0°, 满幅-急降-断电-接住);
中: 去 SV 平滑后为什么不能光加 wc (无阻尼欠阻尼 vs lrKd 补阻尼);
右: 第4档在三种炉况下的预测 vs PID 实测参考。
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

# ---- PID 实测段 ----
df = pd.read_csv(os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260717-212412_212746.csv'))
ts = pd.to_datetime(df.iloc[:, 0])
t_all = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
pv_all = df.iloc[:, 1].to_numpy(float)
mv_all = df.iloc[:, 4].to_numpy(float)
t0 = 21.0
i0 = np.searchsorted(t_all, t0 - 0.5); i1 = np.searchsorted(t_all, t0 + 6.0)
t_p = t_all[i0:i1] - t0; pv_p = pv_all[i0:i1]; mv_p = mv_all[i0:i1]

tvc = Twin()
CARD = dict(wc=4.5, wo=10.0, wr=30.0, nd=12, tauf=0.24, mv_max=90.0, kd=0.75)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
fig.suptitle('7-17 晚 PID 对照场（1.48s/3.0°）与第4档设计：wr=30 去平滑 + lrKd 速度阻尼',
             fontsize=13, y=0.99)

# ================= 左: PID 解剖 =================
ax = axes[0]
ax.plot(t_p, pv_p, color='#1f77b4', lw=1.8, label='PV (实测)')
ax.axhline(400, color='k', ls=':', lw=0.8)
ax.axhline(403, color='#d62728', ls=':', lw=0.7)
ax.text(4.0, 405, '超调预算 +3°', fontsize=8, color='#d62728')
ax2 = ax.twinx()
ax2.fill_between(t_p, 0, mv_p, color='#ff7f0e', alpha=0.25, step='post')
ax2.set_ylabel('MV (%)', color='#ff7f0e'); ax2.set_ylim(0, 240)
ax2.set_yticks([0, 50, 100])
ax.annotate('到达 1.48s / 超调 3.0°', xy=(1.48, 400), xytext=(2.3, 330), fontsize=9.5,
            arrowprops=dict(arrowstyle='->', color='0.3'))
for x, y_, s in [(0.45, 60, '满幅90%\n0.9s'), (1.25, 40, '断电\n@356°C'), (2.6, 60, '~22% 接住\n(今晚保400实测值)')]:
    ax2.text(x, y_, s, fontsize=8.5, ha='center', color='#b25000')
ax.set_xlim(-0.5, 6); ax.set_ylim(80, 430)
ax.set_xlabel('阶跃后时间 (s)'); ax.set_ylabel('温度 (°C)')
ax.set_title('PID 参考控制器实测解剖（教科书式冲刺-滑行）', fontsize=10.5)
ax.legend(fontsize=8.5, loc='center right')

# ================= 中: 机理 =================
ax = axes[1]
tvp = plant_day('standard')
cases = [
    ('你们当前卡 wc=1.55 (kd=0)', dict(wc=1.55, wo=10.0, wr=30.0, nd=12, tauf=0.24,
                                    mv_max=90.0, kd=0.0), '#ff7f0e'),
    ('光加 wc=4 (kd=0)', dict(wc=4.0, wo=10.0, wr=30.0, nd=12, tauf=0.24,
                            mv_max=90.0, kd=0.0), '#d62728'),
    ('第4档 wc=4.5 + kd=0.75', CARD, '#2ca02c'),
]
for lab, cfg, col in cases:
    yy, _ = cl(tvc, tvp, **cfg, y0=100.0, svt=400.0, ckq=1.1)
    r, o, m, _ = mets(yy, 400.0)
    tt = np.arange(len(yy)) * DT
    ax.plot(tt, yy, color=col, lw=1.7, label=f'{lab}: {r:.2f}s/{max(o,m):.1f}°')
ax.axhline(400, color='k', ls=':', lw=0.8)
ax.text(2.6, 445, '去掉 SV 平滑后环路失去阻尼:\nζ=(1+kd)/(2√(wc·τ总)), 光加 wc 越加越荡;\nlrKd 用 ESO 速度估计 (vd+z2) 补回阻尼\n—— 即 PID 里 D 项的角色, 两行 ST 代码',
        fontsize=8.8, va='top',
        bbox=dict(fc='#f5f5f5', ec='#cccccc'))
ax.set_xlim(0, 6); ax.set_ylim(90, 460)
ax.set_xlabel('阶跃后时间 (s)'); ax.set_ylabel('温度 (°C)')
ax.set_title('为什么"wc 再加点"不够：缺的是阻尼项（标准炉况仿真）', fontsize=10.5)
ax.legend(fontsize=8.5, loc='lower right')

# ================= 右: 第4档三炉况 vs PID =================
ax = axes[2]
ax.plot(t_p[t_p >= 0], pv_p[t_p >= 0], color='0.55', lw=1.6, ls='--',
        label='PID 实测 (今晚): 1.48s/3.0°')
for (nm, d), col in zip([('今晚炉况(拖)', 'tonight'), ('标准炉况', 'standard'), ('暖午炉况', 'warm')],
                        ['#1f77b4', '#2ca02c', '#d62728']):
    yy, _ = cl(tvc, plant_day(d), **CARD, y0=100.0, svt=400.0, ckq=1.1)
    r, o, m, _ = mets(yy, 400.0)
    tt = np.arange(len(yy)) * DT
    ax.plot(tt, yy, color=col, lw=1.7, label=f'第4档 {nm}: {r:.2f}s/{max(o,m):.1f}°')
ax.axhline(400, color='k', ls=':', lw=0.8)
ax.axhline(403, color='#d62728', ls=':', lw=0.7)
ax.set_xlim(0, 6); ax.set_ylim(90, 430)
ax.set_xlabel('阶跃后时间 (s)'); ax.set_ylabel('温度 (°C)')
ax.set_title('第4档预测：三种炉况 1.50~1.59s / 首超 ≤2.8°（待上机验证）', fontsize=10.5)
ax.legend(fontsize=8.5, loc='lower right')

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(HERE, '19_0717晚_PID对照与第4档设计.png')
fig.savefig(out, dpi=140)
print('saved:', out)
