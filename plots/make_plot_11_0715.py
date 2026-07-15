#!/usr/bin/env python3
"""7-15 bump session: 数据总览 + 标定结果 + 改进原理 (plots/11)"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.gridspec as gridspec

sys.path.insert(0, '/home/yiz/workspace/src/control_test/analysis')
from twin import Twin

f = font_manager.FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = f.get_name()
plt.rcParams['axes.unicode_minus'] = False

DT = 0.01
df = pd.read_csv('/home/yiz/workspace/src/control_test/AIC9_DATA-20260715-000131_004800.csv')
df.columns = ['time', 'PV1', 'PV2', 'SV', 'MV']
traw = (pd.to_datetime(df['time']) - pd.to_datetime(df['time'].iloc[0])).dt.total_seconds().values
tg = np.arange(0, traw[-1], DT)
pv = np.interp(tg, traw, df['PV1'].values.astype(float))
mv = df['MV'].values.astype(float)[np.clip(np.searchsorted(traw, tg, side='right') - 1, 0, None)]

tw = Twin()  # v2

C1, C2, C3, C4 = '#2b6cb0', '#c05621', '#38a169', '#9f2b68'

fig = plt.figure(figsize=(16.5, 11.5))
gs = gridspec.GridSpec(3, 3, height_ratios=[1.05, 1, 1], hspace=0.44, wspace=0.30,
                       left=0.055, right=0.985, top=0.94, bottom=0.055)

# ---------- (0) 全场总览 ----------
ax = fig.add_subplot(gs[0, :])
ax.plot(tg / 60, pv, color=C1, lw=0.8)
ax.set_ylabel('PV [°C]', color=C1)
ax.set_xlabel('时间 [min]')
ax2 = ax.twinx()
ax2.fill_between(tg / 60, mv, color=C2, alpha=0.30, lw=0)
ax2.set_ylabel('MV [%]', color=C2)
ax2.set_ylim(0, 108)
for x0, x1, s in [(3.7, 9.6, 'A组: 300°C\n4×(+8%, 2s)'), (19.3, 24.2, 'B组: 300°C\n6×(→22%, 2s)'),
                  (30.4, 42.0, 'C组: 200°C\nq 扫 30→100%')]:
    ax.axvspan(x0, x1, color='#888888', alpha=0.10)
    ax.text((x0 + x1) / 2, 505, s, ha='center', va='top', fontsize=9)
ax.set_ylim(60, 520)
ax.set_title('7-15 深夜 bump session 全场（46.5 min）：闭环稳温 → 开环一步跳变 → 回闭环', fontsize=11)

# ---------- (1) 22% 拍放大 + 孪生复现 ----------
ax = fig.add_subplot(gs[1, 0])
t0, w0, w1 = 1216.0, 1215.2, 1219.8
m = (tg >= w0) & (tg <= w1)
ax.plot(tg[m] - t0, pv[m], color=C1, lw=1.4, label='实测 PV')
i0 = int(w0 / DT)
ysim = tw.open_loop(mv[i0:int(w1 / DT)], y0=float(pv[i0]))
n = min(m.sum(), len(ysim))
ax.plot(tg[m][:n] - t0, ysim[:n], '--', color=C4, lw=1.4, label='孪生 v2 复现')
# 旧算法短窗斜率线
th, sl = 0.12, 15.9
tt = np.array([th, th + 0.5])
ax.plot(tt, 300.0 + sl * (tt - th), color=C3, lw=2.2, alpha=0.9, label='旧算法: 前0.5s斜率')
ax.annotate('前 0.5s 只看到 15.9°C/s\n→ q=39 (低估 22%)', xy=(0.45, 305.5), fontsize=8.5, color=C3)
ax.annotate('整段拟合: q∞=50.7, τe=0.23s', xy=(1.3, 322), fontsize=8.5, color=C4)
ax3 = ax.twinx()
ax3.step(tg[m] - t0, mv[m], color=C2, lw=0.9, alpha=0.7)
ax3.set_ylim(0, 110)
ax3.set_yticks([])
ax.set_title('B组一拍（12.6→22%）：响应是"弯"的\n= 元件热惯性 τe，短窗斜率必然低估', fontsize=10)
ax.set_xlabel('相对跳变 [s]')
ax.set_ylabel('PV [°C]')
ax.legend(fontsize=8, loc='lower right')

# ---------- (2) 100% 拍放大 ----------
ax = fig.add_subplot(gs[1, 1])
t0, w0, w1 = 2483.1, 2482.7, 2485.8
m = (tg >= w0) & (tg <= w1)
ax.plot(tg[m] - t0, pv[m], color=C1, lw=1.4, label='实测 PV')
i0 = int(w0 / DT)
ysim = tw.open_loop(mv[i0:int(w1 / DT)], y0=float(pv[i0]))
n = min(m.sum(), len(ysim))
ax.plot(tg[m][:n] - t0, ysim[:n], '--', color=C4, lw=1.4, label='孪生 v2 复现')
ax3 = ax.twinx()
ax3.step(tg[m] - t0, mv[m], color=C2, lw=0.9, alpha=0.7)
ax3.set_ylabel('MV [%]', color=C2)
ax3.set_ylim(0, 110)
ax.annotate('MV 已切 0，PV 又涨 ≈58°C\n= 元件"在途热量"倒出来\n(刹车距离的物理来源)',
            xy=(0.62, 252), xytext=(1.35, 225), fontsize=8.5,
            arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))
ax.set_title('C组 100% 拍（0.4s 保持）：断电后的"滑行"\n直接看到 τe —— settle 4s 物理墙的元凶', fontsize=10)
ax.set_xlabel('相对跳变 [s]')
ax.set_ylabel('PV [°C]')
ax.legend(fontsize=8, loc='lower right')

# ---------- (3) q(u) 三条曲线 ----------
ax = fig.add_subplot(gs[1, 2])
v1 = Twin('/home/yiz/workspace/src/control_test/analysis/twin_params_v1.json')
uu = np.linspace(0, 100, 300)
ax.plot(uu, v1.q_of(uu), ':', color='0.45', lw=1.6, label='7-13 模型 v1（高段是外推）')
naive_u = [22, 30, 40, 55, 70, 85, 100]
naive_q = [39.3, 55.3, 88.7, 129.4, 162.8, 176.1, 153.3]
ax.plot(naive_u, naive_q, 's', color=C3, ms=6, label='短窗斜率（含 τe 污染, 弃用）')
qi_u = [0, 8.1, 12.3, 22.0, 30.0, 40.0, 55.0, 70.0, 85.0, 100.0]
qi_q = [0, 9.9, 23.4, 50.4, 85.2, 142.1, 248.4, 335.0, 335.1, 345.7]
ax.plot(qi_u, qi_q, 'o-', color=C4, ms=6, lw=1.8, label='整段拟合 q∞（进表/孪生v2）')
ax.axhline(43.7, color=C1, lw=0.8, ls='--', alpha=0.6)
ax.annotate('7-13 的 450°C 平衡点 q(21.8)=43.8\n（独立佐证 q∞ 而非短窗值）', xy=(46, 40), fontsize=8, color=C1)
ax.set_title('q(u)：短窗斜率低估 40%，q∞ 才是真马力\n65% 以上确认平坦（饱和）', fontsize=10)
ax.set_xlabel('MV [%]')
ax.set_ylabel('加热马力 q [°C/s]')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(0, 104)

# ---------- (4) θ 样本: 幅度与方向 ----------
ax = fig.add_subplot(gs[2, 0])
up_small = [(8.3, .13), (8.9, .12), (8.1, .12), (7.9, .12), (8.9, .14), (9.4, .12),
            (9.8, .10), (10.0, .10), (10.4, .11), (10.4, .12)]
up_big = [(20.7, .09), (31.5, .09), (46.8, .04), (61.9, .07), (76.9, .06), (91.9, .06)]
dn = [(11.7, .19), (11.6, .12), (7.8, .20), (21.8, .20)]
ax.plot(*zip(*up_small), 'o', color=C1, ms=7, label='上行小幅（A/B组）')
ax.plot(*zip(*up_big), '^', color=C3, ms=7, label='上行大幅（C组）')
ax.plot(*zip(*dn), 'v', color=C2, ms=7, label='下行（断电, 含7-13的450°C样本）')
ax.axhline(0.12, color=C1, lw=0.8, ls='--', alpha=0.6)
ax.axhline(0.20, color=C2, lw=0.8, ls='--', alpha=0.6)
ax.text(50, 0.125, '上行 θ≈0.12s', fontsize=8.5, color=C1)
ax.text(50, 0.205, '下行 θ≈0.20s（元件放热）', fontsize=8.5, color=C2)
ax.set_ylim(0, 0.24)
ax.set_title('死区 θ：方向不对称 + 大功率变快\n刹车靠下行 → lrTheta 取 0.24~0.31（含 τe 补偿）', fontsize=10)
ax.set_xlabel('|Δu| [%]')
ax.set_ylabel('θ [s]')
ax.legend(fontsize=8, loc='upper right')

# ---------- (5) 平衡 MV 慢漂 ----------
ax = fig.add_subplot(gs[2, 1])
a300 = [(155, 14.6), (240, 13.1), (325, 12.5), (415, 12.1), (500, 11.9),
        (1030, 14.2), (1175, 12.7), (1230, 12.4), (1280, 12.0), (1355, 11.7)]
a200 = [(1965, 8.1), (2075, 7.9), (2200, 7.9), (2355, 7.9), (2505, 7.9)]
ax.plot([a / 60 for a, _ in a300], [b for _, b in a300], 'o-', color=C1, label='稳住 300°C 所需 MV')
ax.plot([a / 60 for a, _ in a200], [b for _, b in a200], 's-', color=C3, label='稳住 200°C 所需 MV')
ax.axvspan(9.5, 17.2, color='0.85', alpha=0.5)
ax.axvspan(24.2, 29.5, color='0.85', alpha=0.5)
ax.text(13.3, 13.9, '深度冷却', ha='center', fontsize=8, color='0.4')
ax.text(26.8, 13.9, '深度冷却', ha='center', fontsize=8, color='0.4')
ax.annotate('冷透重来 → 需求回到 14.6%\n炉体回暖 → 5min 内滑到 11.9%\n（±10% 增益慢漂, 归 z2 实时吃掉）',
            xy=(8.3, 11.95), xytext=(15.5, 10.6), fontsize=8.5,
            arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))
ax.set_title('同一温度、所需 MV 随炉体热状态漂 ±10%\n→ 表是快照，慢漂必须留给 ESO 的 z2', fontsize=10)
ax.set_xlabel('时间 [min]')
ax.set_ylabel('平衡 MV [%]')
ax.set_ylim(7, 15.5)
ax.legend(fontsize=8, loc='center right')

# ---------- (6) 各方案对比 ----------
ax = fig.add_subplot(gs[2, 2])
names = ['PID\n(7-13实测)', '旧参ADRC\nwc=0.9(实测)', '第1档\n(v2预期)', '第2档\n(v2预期)']
settle = [4.11, 6.7, 5.44, 4.16]
ov = [7.6, 0.9, 0.01, 0.85]
x = np.arange(4)
b1 = ax.bar(x - 0.19, settle, 0.38, color=C1, alpha=0.85, label='settle [s]')
axr = ax.twinx()
b2 = axr.bar(x + 0.19, ov, 0.38, color=C2, alpha=0.85, label='超调 [°C]')
for i in (2, 3):
    b1[i].set_hatch('//'); b2[i].set_hatch('//')
    b1[i].set_edgecolor('white'); b2[i].set_edgecolor('white')
for xi, s, o in zip(x, settle, ov):
    ax.text(xi - 0.19, s + 0.08, f'{s:.1f}', ha='center', fontsize=8.5, color=C1)
    axr.text(xi + 0.19, o + 0.12, f'{o:.1f}', ha='center', fontsize=8.5, color=C2)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel('settle ±1°C [s]', color=C1)
axr.set_ylabel('超调 [°C]', color=C2)
ax.set_ylim(0, 7.6); axr.set_ylim(0, 8.4)
ax.set_title('开表爬梯预期（斜纹=孪生预测, 待实测）\n第2档 ≈ PID 速度、1/9 超调', fontsize=10)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')

fig.suptitle('AIC9 通道1 · 7-15 bump session：q∞ 标定、元件惯性 τe 的发现与开表方案', fontsize=13.5, y=0.985)
out = '/home/yiz/workspace/src/control_test/plots/11_0715标定_数据与改进原理.png'
fig.savefig(out, dpi=130)
print('saved:', out)
