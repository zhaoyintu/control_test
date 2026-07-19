#!/usr/bin/env python3
"""图 23: 7-19 晚 kd 全扫描判读 (AIC9_DATA-20260719-181948_191244.csv)
上: 全场总览, 每个标注测试给 kd + 到达/首超;
下: 三个靶点类的真机拨盘曲线 (reach & 首超 vs kd), 叠加早场(16:41)对照点。
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

df = pd.read_csv(os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260719-181948_191244.csv'))
ts = pd.to_datetime(df.iloc[:, 0])
t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
pv = df.iloc[:, 1].to_numpy(float)
sv = df.iloc[:, 3].to_numpy(float)
mv = df.iloc[:, 4].to_numpy(float)

STEPS = [  # (t0, svt, kd 或 None)
    (14.0, 200., None), (38.2, 440., None), (222.6, 200., None), (246.5, 440., None),
    (450.4, 400., None), (1176.9, 400., None), (1279.0, 200., None), (1292.2, 440., None),
    (1484.6, 200., 1.2), (1497.3, 440., 1.2), (1688.7, 400., 1.2),
    (1999.4, 400., 0.95), (2125.1, 400., 0.85), (2418.8, 400., 0.75),
    (2510.7, 200., 0.75), (2523.8, 440., 0.75), (2672.6, 200., 0.9), (2685.5, 440., 0.9),
    (2961.1, 200., 1.0), (2973.8, 440., 1.0), (3115.6, 200., 1.1), (3128.3, 440., 1.1),
]


def metrics(t0, svt):
    i0 = np.searchsorted(t, t0)
    j = np.where(np.diff(sv[i0 + 5:]) != 0)[0]
    i1 = min(i0 + 5 + (j[0] if len(j) else len(t)), i0 + 1500, len(t) - 1)
    tt = t[i0:i1] - t0; p = pv[i0:i1]
    cross = np.where(p >= svt)[0]
    reach = tt[cross[0]] if len(cross) else np.inf
    ov = max(p.max() - svt, 0)
    return reach, ov


R = {(t0, svt): metrics(t0, svt) for t0, svt, _ in STEPS}

fig = plt.figure(figsize=(16.5, 9.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.34, wspace=0.27,
                      left=0.05, right=0.97, top=0.90, bottom=0.07)
fig.suptitle('7-19 晚 lrKd 全扫描：真机拨盘曲线（wc=4.5 wo=10 wr=30 nd=12 lrTauF=0.24 lrMVMax=90）',
             fontsize=13, y=0.965)

# ---------- (a) 总览 ----------
ax = fig.add_subplot(gs[0, :])
ax.plot(t / 60, pv, color='#1f77b4', lw=1.0)
ax.plot(t / 60, sv, color='k', lw=0.6, ls=':')
ax2 = ax.twinx()
ax2.fill_between(t / 60, 0, mv, color='#ff7f0e', alpha=0.16, step='post')
ax2.set_ylim(0, 300); ax2.set_yticks([0, 50, 100])
ax2.set_ylabel('MV (%)', color='#b25000', fontsize=8.5)
up = 0
for t0, svt, kd in STEPS:
    if svt == 200.:
        continue                       # 总览只标 400/440 大步, 避免拥挤
    reach, ov = R[(t0, svt)]
    rt = f'{reach:.2f}s' if np.isfinite(reach) else '未触线'
    lab = f'kd={kd:g}' if kd else 'kd=?'
    y = 500 if up == 0 else 545
    up = 1 - up
    ax.annotate(f'{lab}\n{rt}/{ov:.1f}°', xy=(t0 / 60, svt), xytext=(t0 / 60 - 0.35, y),
                fontsize=7.8, color=('#333333' if kd else '#999999'),
                arrowprops=dict(arrowstyle='-', color='0.75', lw=0.6))
ax.set_ylim(20, 600); ax.set_xlim(0, t[-1] / 60)
ax.set_xlabel('时间 (min)'); ax.set_ylabel('温度 (°C)')
ax.set_title('(a) 全场 52.9min：前 4 组未标注 kd（灰）；标注组 = 400 类扫 1.2/0.95/0.85/0.75，'
             '440 类扫 1.2/0.75/0.9/1.0/1.1（各带同 kd 的 100→200 小步）', fontsize=9.5)

# ---------- 拨盘曲线 ----------
def dial(ax, svt, title, morning=None, kd_list=None):
    pts = sorted([(kd, R[(t0, s)]) for t0, s, kd in STEPS if s == svt and kd], key=lambda x: x[0])
    kds = [p[0] for p in pts]
    reach = [p[1][0] for p in pts]
    ovs = [p[1][1] for p in pts]
    axb = ax.twinx()
    axb.bar(kds, ovs, width=0.035, color='#d62728', alpha=0.35, zorder=1)
    axb.axhline(3, color='#d62728', ls='--', lw=0.9)
    axb.set_ylim(0, 20); axb.set_ylabel('首超 (°C)', color='#d62728', fontsize=9)
    rr = [r if np.isfinite(r) else np.nan for r in reach]
    ax.plot(kds, rr, 'o-', color='#1f77b4', lw=1.6, ms=6, zorder=3)
    for k, r in zip(kds, reach):
        if not np.isfinite(r) or r > 3.0:
            ax.annotate('爬行', (k, min(r, 4.4) if np.isfinite(r) else 4.4),
                        xytext=(0, 6), textcoords='offset points', fontsize=8, color='#1f77b4', ha='center')
    if morning:
        for (mk, mr, mo) in morning:
            ax.plot([mk], [mr], 'o', mfc='none', mec='#1f77b4', ms=9, mew=1.6, zorder=4)
            axb.plot([mk], [mo], 's', mfc='none', mec='#d62728', ms=8, mew=1.6, zorder=4)
    ax.axhline(1.5, color='#2ca02c', ls=':', lw=0.9)
    ax.set_ylim(0, 4.6); ax.set_xlabel('lrKd'); ax.set_ylabel('到达 (s)', color='#1f77b4', fontsize=9)
    ax.set_title(title, fontsize=9.6)

axd = fig.add_subplot(gs[1, 0])
dial(axd, 400., '(b) 400 类拨盘：kd=0.85 甜点 (1.71s/1.0°)\nkd≥0.95 爬行, kd=0.75 超预算 (4.2°)',
     morning=[(0.9, 1.57, 2.5), (0.9, 3.30, 0.5)])
axd.annotate('空心 = 早场(16:41)\nkd=0.9 两次翻面', xy=(0.9, 3.3), xytext=(0.98, 3.9),
             fontsize=8, arrowprops=dict(arrowstyle='->', color='0.5'))

axd = fig.add_subplot(gs[1, 1])
dial(axd, 440., '(c) 440 类拨盘：悬崖在 1.0~1.1 之间且早晚游走\n晚场冲透档全部 >3° (4.4~10.7°)',
     morning=[(1.1, 1.59, 3.0)])
axd.annotate('早场 kd=1.1: 1.59s/3.0°\n晚场同 kd: 爬行 —— 悬崖游走 ±0.1', xy=(1.1, 3.33),
             xytext=(0.76, 3.75), fontsize=8, arrowprops=dict(arrowstyle='->', color='0.5'))

axd = fig.add_subplot(gs[1, 2])
dial(axd, 200., '(d) 200 类拨盘：晚场炉体热, 墙在"推"而非"刹"\n首超 5.6~18°, 全部超预算 (早场冷炉体仅 2.6°)',
     morning=[(1.1, 0.99, 2.6)])
axd.annotate('同 kd=1.1: 冷炉体 2.6° vs 热炉体 8.0°\n→ 低靶点的免费刹车随墙温变号', xy=(1.1, 0.82),
             xytext=(0.78, 2.1), fontsize=8, arrowprops=dict(arrowstyle='->', color='0.5'))

out = os.path.join(HERE, '23_0719晚_kd全扫描拨盘曲线.png')
fig.savefig(out, dpi=140)
print('saved:', out)
