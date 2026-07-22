#!/usr/bin/env python3
"""图 25: 7-22 元件测温场判读 (新功率调节器 v2 首场)
(a) 全场; (b) 五发脉冲对齐叠加; (c) 冷启动功率冲曲线 (本场主产出);
(d) 结尾异常段: 指令 90% 输出为零 38.5s。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
_f = font_manager.FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = _f.get_name()
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260722-121109_124114.csv'))
ts = pd.to_datetime(df.iloc[:, 0])
t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
pv = df.iloc[:, 1].to_numpy(float)
mv = df.iloc[:, 4].to_numpy(float)

PULSES = [(893.0, '热元件基准', 249, '#1f77b4'), (899.0, '断电5s(高温段,含缩水)', 195, '#999999'),
          (930.0, '断电30s', 257, '#2ca02c'), (1051.0, '断电120s', 278, '#ff7f0e'),
          (1652.0, '断电600s', 294, '#d62728')]

fig = plt.figure(figsize=(16.5, 9.4))
gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1], hspace=0.35, wspace=0.28,
                      left=0.05, right=0.975, top=0.90, bottom=0.07)
fig.suptitle('7-22 元件测温场（新功率调节器 v2 首场）：冷启动冲 +18% 确认；调节器增益 ≈0.82×旧；'
             '结尾出现"指令90%输出为零"异常 38.5s', fontsize=12.5, y=0.965)

# (a) 全场
ax = fig.add_subplot(gs[0, :])
ax.plot(t / 60, pv, color='#1f77b4', lw=1.0)
ax2 = ax.twinx()
ax2.fill_between(t / 60, 0, mv, color='#ff7f0e', alpha=0.20, step='post')
ax2.set_ylim(0, 260); ax2.set_yticks([0, 50, 100]); ax2.set_ylabel('MV (%)', color='#b25000', fontsize=8.5)
for t0, nm, q, c in PULSES:
    ax.annotate(nm, xy=(t0 / 60, np.interp(t0 + 1, t, pv)), xytext=(t0 / 60 - 1.2, 480),
                fontsize=8, color=c, arrowprops=dict(arrowstyle='-', color='0.75', lw=0.6))
ax.axvspan(1750 / 60, 1789 / 60, color='#d62728', alpha=0.18)
ax.text(1755 / 60, 260, '异常段\n(d)', fontsize=8.5, color='#d62728')
ax.text(6, 120, '静置 13.8min\namb=29.7°C', fontsize=8.5, color='#2d6a2d')
ax.set_xlim(0, t[-1] / 60); ax.set_ylim(0, 540)
ax.set_xlabel('时间 (min)'); ax.set_ylabel('温度 (°C)')
ax.set_title('(a) 全场 30.1min：静置 → 稳150 → 五发 90%×1s 脉冲（变断电间隔）→ 回稳 → 异常段', fontsize=9.8)

# (b) 脉冲叠加
ax = fig.add_subplot(gs[1, 0])
for t0, nm, q, c in PULSES:
    i0 = np.searchsorted(t, t0 - 0.2); i1 = np.searchsorted(t, t0 + 2.5)
    ax.plot(t[i0:i1] - t0, pv[i0:i1] - pv[np.searchsorted(t, t0)], color=c, lw=1.6,
            ls='--' if '5s' in nm else '-', label=f'{nm}')
ax.set_xlabel('脉冲后时间 (s)'); ax.set_ylabel('温升 ΔPV (°C)')
ax.set_title('(b) 五发脉冲对齐：越冷起步越猛\n(灰虚线=高温段发, 被缩水压低, 不参与冷冲拟合)', fontsize=9.6)
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.25)

# (c) 冷启动冲曲线
ax = fig.add_subplot(gs[1, 1])
xs = [0.5, 30, 120, 600]
qs = [249, 257, 278, 294]
ax.semilogx(xs, [q / qs[0] * 100 - 100 for q in qs], 'o-', color='#d62728', ms=8, lw=1.8)
for x, q in zip(xs, qs):
    ax.annotate(f'{q}°C/s', (x, q / qs[0] * 100 - 100), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=8.5)
ax.axhline(0, color='k', lw=0.7)
ax.set_xlabel('脉冲前断电时长 (s, 对数轴)'); ax.set_ylabel('相对热元件的功率增幅 (%)')
ax.set_ylim(-4, 24)
ax.set_title('(c) 冷启动功率冲曲线（本场主产出）\nβ≈+18%@全冷, 时间常数量级 2~5min', fontsize=9.6)
ax.grid(alpha=0.25, which='both')

# (d) 异常段
ax = fig.add_subplot(gs[1, 2])
i0 = np.searchsorted(t, 1730.0); i1 = np.searchsorted(t, 1800.0)
ax.plot(t[i0:i1] - 1750, pv[i0:i1], color='#1f77b4', lw=1.6, label='PV (持续下滑)')
ax2 = ax.twinx()
ax2.fill_between(t[i0:i1] - 1750, 0, mv[i0:i1], color='#d62728', alpha=0.30, step='post')
ax2.set_ylim(0, 100); ax2.set_ylabel('MV (%)', color='#d62728', fontsize=8.5)
ax.axvspan(0, 38.5, color='#d62728', alpha=0.08)
ax.text(6, 138, '指令 90%\n炉温照跌 (≈自然冷却速率)\n→ 输出级 38.5s 无功率', fontsize=9, color='#b03030')
ax.set_xlim(-20, 50); ax.set_ylim(85, 160)
ax.set_xlabel('相对异常起点 (s)'); ax.set_ylabel('温度 (°C)')
ax.set_title('(d) 异常段：闭环把 MV 顶到 90%, PV 按无功率速率下滑\n—— 新调节器输出级掉线事件, 需电气排查', fontsize=9.6)
ax.legend(fontsize=8.5, loc='upper right')

out = os.path.join(HERE, '25_0722_元件场与新调节器判读.png')
fig.savefig(out, dpi=140)
print('saved:', out)
