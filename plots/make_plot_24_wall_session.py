#!/usr/bin/env python3
"""图 24: 7-20 墙参数采集场验收 + G1 第一轮结构证据
(a) 全场分幕总览; (b) 冷/热墙 400 锚点对照 (墙的托力可见);
(c) 长降温半对数图 -- 两时间尺度 = 两层墙的直接证据;
(d) 单层模型"一张脸顾不了两头"的失败展示。
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

df = pd.read_csv(os.path.join(HERE, '..', 'user_feedback', 'AIC9_DATA-20260720-201947_215552.csv'))
ts = pd.to_datetime(df.iloc[:, 0])
t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
pv = df.iloc[:, 1].to_numpy(float)
mv = df.iloc[:, 4].to_numpy(float)
AMB = float(pv[t < 2000].mean())

fig = plt.figure(figsize=(16.5, 9.6))
gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1], hspace=0.35, wspace=0.28,
                      left=0.05, right=0.975, top=0.90, bottom=0.07)
fig.suptitle('7-20 墙参数采集场（96 min）：数据验收通过；G1 第一轮 → 单层墙被否决，结构升级为两层（炉衬+炉体）',
             fontsize=12.5, y=0.965)

# ---------- (a) 全场分幕 ----------
ax = fig.add_subplot(gs[0, :])
ax.plot(t / 60, pv, color='#1f77b4', lw=1.0)
ax2 = ax.twinx()
ax2.fill_between(t / 60, 0, mv, color='#ff7f0e', alpha=0.18, step='post')
ax2.set_ylim(0, 280); ax2.set_yticks([0, 50, 100]); ax2.set_ylabel('MV (%)', color='#b25000', fontsize=8.5)
phases = [(0, 40.2, '静置 40min\namb=28.4°C 直接测得\n起跑墙温=室温 精确已知', 60),
          (40.2, 41.7, '', 0), (41.7, 53.2, '长降温①\n692s, 400→58\n(冷墙态)', 300),
          (53.2, 72.3, '440 往返 ×4\n(大信号 τe 素材,\n间隔 3~6min 合规)', 470),
          (74.1, 76.4, '', 0), (76.4, 89.9, '长降温②\n853s, 200→66\n(热墙态)', 300),
          (89.9, 90.9, '', 0), (90.9, 96, '尾段\n降温③', 200)]
for a_, b_, s, y_ in phases:
    if s:
        ax.axvspan(a_, b_, color='#2ca02c', alpha=0.05)
        ax.text((a_ + b_) / 2, y_, s, ha='center', fontsize=8.2, color='#2d6a2d')
ax.annotate('冷墙 400 锚点 55s\nMV 18.3→17.3%', xy=(41.2, 400), xytext=(44.5, 430),
            fontsize=8.2, arrowprops=dict(arrowstyle='->', color='0.5'))
ax.annotate('热墙 400 锚点 58s\nMV 17.7→16.5%', xy=(90.4, 400), xytext=(81, 440),
            fontsize=8.2, arrowprops=dict(arrowstyle='->', color='0.5'))
ax.set_xlim(0, 96.2); ax.set_ylim(0, 520)
ax.set_xlabel('时间 (min)'); ax.set_ylabel('温度 (°C)')
ax.set_title('(a) 全场分幕：静置 → 冷墙锚点 → 长降温① → 440×4 烤墙 → 阶梯 → 长降温② → 热墙锚点 → 尾段', fontsize=9.8)

# ---------- (b) 冷/热墙锚点对照 ----------
ax = fig.add_subplot(gs[1, 0])
bars = [('冷墙态\n(静置后首烧)', 17.3, '#1f77b4'), ('热墙态\n(440×4 之后)', 16.5, '#d62728')]
for i, (nm, v, c) in enumerate(bars):
    ax.bar(i, v, 0.55, color=c, alpha=0.8)
    ax.text(i, v + 0.06, f'{v:.1f}%', ha='center', fontsize=10)
ax.set_xticks([0, 1]); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
ax.set_ylim(15.5, 18.2); ax.set_ylabel('保 400°C 所需 MV (%)  [末 20s]')
ax.set_title('(b) 同样保 400°C：墙热了少花 0.8% MV\n—— 墙的"托力"第一次被直接看见', fontsize=9.6)
ax.text(0.5, 15.75, '差值 ≈ 1.9 °C/s 加热速率\n= 自适应刹车要在线读的那个量',
        ha='center', fontsize=8.5, color='#555555')

# ---------- (c) 长降温半对数 ----------
ax = fig.add_subplot(gs[1, 1])
falls = [(2503, 690, '① 400 起, 冷墙', '#1f77b4'),
         (4541, 835, '② 200 起, 热墙', '#2ca02c'),
         (5456, 298, '③ 400 起, 热墙', '#d62728')]
for t0, dur, nm, c in falls:
    i0 = np.searchsorted(t, t0 + 1.5); i1 = np.searchsorted(t, t0 + dur)
    tt = t[i0:i1] - t0
    ax.semilogy(tt / 60, pv[i0:i1] - AMB, color=c, lw=1.6, label=nm)
ax.set_xlabel('断电后时间 (min)'); ax.set_ylabel('PV − 室温 (°C, 对数轴)')
ax.set_title('(c) 半对数图上单指数应是直线——\n实测明显两折 = 两个时间尺度 = 两层墙证据', fontsize=9.6)
ax.text(6.5, 150, '①③同起点不同墙态,\n脸型不同 → 墙态决定降温',
        fontsize=8.3, color='#555555')
ax.legend(fontsize=8.5, loc='lower left')
ax.grid(alpha=0.25, which='both')

# ---------- (d) 单层模型失败展示 ----------
ax = fig.add_subplot(gs[1, 2])
t0, dur = 2503, 692
i0 = np.searchsorted(t, t0 + 1.5); i1 = np.searchsorted(t, t0 + dur)
tt = t[i0:i1] - t0; yy = pv[i0:i1]
ax.plot(tt / 60, yy, color='k', lw=2.0, label='实测 (长降温①)')


def sim_single(ayw, byw, bwa, w0, y0, n, dt=0.5):
    y, w = y0, w0
    out = np.empty(n)
    for i in range(n):
        out[i] = y
        y += dt * (-ayw * (y - w))
        w += dt * (byw * (y - w) - bwa * (w - AMB))
    return out


tt5 = np.arange(0, dur - 1.5, 0.5)
for ayw, byw, bwa, w0, nm, c in [
        (0.0836, 0.0042, 0.0060, 150, '单层参数组 A (顾早段)', '#d62728'),
        (0.0434, 0.0051, 0.0015, 256, '单层参数组 B (全局折中)', '#ff7f0e')]:
    ys = sim_single(ayw, byw, bwa, w0, yy[0], len(tt5))
    ax.plot(tt5 / 60, ys, ls='--', color=c, lw=1.5, label=nm)
ax.set_xlabel('断电后时间 (min)'); ax.set_ylabel('温度 (°C)')
ax.set_title('(d) 单层墙怎么调都对不上全程\n（早段/深尾二选一）→ G1 判决: 结构升级', fontsize=9.6)
ax.legend(fontsize=8.3, loc='upper right')

out = os.path.join(HERE, '24_0720_墙场验收与结构证据.png')
fig.savefig(out, dpi=140)
print('saved:', out)
