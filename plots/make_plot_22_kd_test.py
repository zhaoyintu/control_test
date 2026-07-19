#!/usr/bin/env python3
"""图 22: 7-19 第4档 lrKd 首测判读 (AIC9_DATA-20260719-163352_164130.csv)
(a) 全场总览; (b) 100→200 kd=1.1; (c) 200→440 kd=1.1;
(d) 两次 100→400 kd=0.9 对齐叠加 -- 同参数结果翻面, 悬崖实锤。
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

df = pd.read_csv(os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260719-163352_164130.csv'))
ts = pd.to_datetime(df.iloc[:, 0])
t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
pv = df.iloc[:, 1].to_numpy(float)
sv = df.iloc[:, 3].to_numpy(float)
mv = df.iloc[:, 4].to_numpy(float)

fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
fig.suptitle('7-19 第4档 lrKd 首测：wc=4.5 wo=10 wr=30 nd=12 lrTauF=0.24 lrMVMax=90'
             '（保温 MV 18.3~19.2% → 标准炉况日）', fontsize=12.5, y=0.985)

# (a) 全场
ax = axes[0][0]
ax.plot(t, pv, color='#1f77b4', lw=1.2, label='PV')
ax.plot(t, sv, color='k', lw=0.8, ls=':', label='SV')
ax2 = ax.twinx()
ax2.fill_between(t, 0, mv, color='#ff7f0e', alpha=0.20, step='post')
ax2.set_ylim(0, 260); ax2.set_yticks([0, 50, 100]); ax2.set_ylabel('MV (%)', color='#b25000', fontsize=8.5)
ax.axvspan(28, 75, color='#2ca02c', alpha=0.07)
ax.axvspan(195, 445, color='#9467bd', alpha=0.06)
ax.text(40, 470, 'kd=1.1', fontsize=10, color='#2ca02c')
ax.text(300, 470, 'kd=0.9', fontsize=10, color='#9467bd')
ax.set_xlim(0, 458); ax.set_ylim(60, 500)
ax.set_xlabel('时间 (s)'); ax.set_ylabel('温度 (°C)')
ax.set_title('(a) 全场：100→200、200→440（kd=1.1）｜100→400 ×2（kd=0.9）', fontsize=10)
ax.legend(fontsize=8.5, loc='upper right')

def zoom(ax, t0, y0, svt, dur, label, col='#1f77b4'):
    i0 = np.searchsorted(t, t0); i1 = np.searchsorted(t, t0 + dur)
    tt = t[i0:i1] - t0; p = pv[i0:i1]; m = mv[i0:i1]
    cross = np.where(p >= svt)[0]
    reach = tt[cross[0]] if len(cross) else np.inf
    ov = max(p.max() - svt, 0)
    ax.plot(tt, p, color=col, lw=1.8, label=label.format(reach=reach, ov=ov))
    ax2 = ax.twinx()
    ax2.fill_between(tt, 0, m, color='#ff7f0e', alpha=0.20, step='post')
    ax2.set_ylim(0, 260); ax2.set_yticks([0, 50, 100])
    ax2.set_ylabel('MV (%)', color='#b25000', fontsize=8.5)
    ax.axhline(svt, color='k', ls=':', lw=0.8)
    ax.axhline(svt + 3, color='#d62728', ls=':', lw=0.7)
    return reach, ov

# (b) 100→200 kd=1.1
ax = axes[0][1]
zoom(ax, 30.8, 100., 200., 8, '实测 {reach:.2f}s / 首超 {ov:.1f}°')
ax.text(4.0, 204.2, '超调预算 +3°', fontsize=8, color='#d62728')
ax.set_xlim(0, 8); ax.set_ylim(90, 215)
ax.set_xlabel('阶跃后时间 (s)'); ax.set_ylabel('温度 (°C)')
ax.set_title('(b) 100→200（kd=1.1）：0.99s / 2.6° —— 达标\n（孪生预测该表现需 kd≈0.85: 真机比孪生"欠阻尼"）', fontsize=9.5)
ax.legend(fontsize=8.5, loc='lower right')

# (c) 200→440 kd=1.1
ax = axes[1][0]
zoom(ax, 53.4, 200., 440., 8, '实测 {reach:.2f}s / 首超 {ov:.1f}°')
ax.text(4.0, 444.5, '超调预算 +3°', fontsize=8, color='#d62728')
ax.set_xlim(0, 8); ax.set_ylim(188, 460)
ax.set_xlabel('阶跃后时间 (s)'); ax.set_ylabel('温度 (°C)')
ax.set_title('(c) 200→440（kd=1.1）：1.59s / 3.0°，±3°带 1.51s 进入 —— 达标\n（MV 无归零段: 阻尼收着进弯, 峰值 443 距 490 预算远）', fontsize=9.5)
ax.legend(fontsize=8.5, loc='lower right')

# (d) 两次 100→400 kd=0.9 叠加
ax = axes[1][1]
for t0, col, tag in [(201.3, '#9467bd', '第1次(t=201s)'), (422.9, '#d62728', '第2次(t=423s)')]:
    i0 = np.searchsorted(t, t0); i1 = np.searchsorted(t, t0 + 8)
    tt = t[i0:i1] - t0; p = pv[i0:i1]
    cross = np.where(p >= 400.)[0]
    reach = tt[cross[0]] if len(cross) else np.inf
    ov = max(p.max() - 400., 0)
    ax.plot(tt, p, color=col, lw=1.8, label=f'{tag}: {reach:.2f}s / {ov:.1f}°')
ax.axhline(400, color='k', ls=':', lw=0.8)
ax.axhline(403, color='#d62728', ls=':', lw=0.7)
ax.set_xlim(0, 8); ax.set_ylim(90, 430)
axins = ax.inset_axes([0.45, 0.15, 0.52, 0.42])
for t0, col in [(201.3, '#9467bd'), (422.9, '#d62728')]:
    i0 = np.searchsorted(t, t0); i1 = np.searchsorted(t, t0 + 8)
    axins.plot(t[i0:i1] - t0, pv[i0:i1], color=col, lw=1.5)
axins.axhline(400, color='k', ls=':', lw=0.7)
axins.axhline(403, color='#d62728', ls=':', lw=0.6)
axins.set_xlim(1.0, 5); axins.set_ylim(393, 405)
axins.tick_params(labelsize=7)
ax.set_xlabel('阶跃后时间 (s)'); ax.set_ylabel('温度 (°C)')
ax.set_title('(d) 100→400 同参数跑两次（kd=0.9）：1.57s/2.5° vs 3.30s/0.5°\n—— kd=0.9 正踩在 400 类的悬崖线上，炉体状态微差即翻面', fontsize=9.5)
ax.legend(fontsize=8.5, loc='center right')

fig.tight_layout(rect=(0, 0, 1, 0.965))
out = os.path.join(HERE, '22_0719_kd首测判读.png')
fig.savefig(out, dpi=140)
print('saved:', out)
