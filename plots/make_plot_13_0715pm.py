#!/usr/bin/env python3
"""7-15 下午真机开表验证: 总览 + 两组放大 + 回放辨识证据 (plots/13)"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.gridspec as gridspec

font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = font_manager.FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams['axes.unicode_minus'] = False

DT = 0.01
df = pd.read_csv('/home/yiz/workspace/src/control_test/user_feedback/AIC9_DATA-20260715-163445_170519.csv')
df.columns = ['time', 'PV1', 'PV2', 'SV', 'MV']
traw = (pd.to_datetime(df['time']) - pd.to_datetime(df['time'].iloc[0])).dt.total_seconds().values
tg = np.arange(0, traw[-1], DT)
pv = np.interp(tg, traw, df['PV1'].values.astype(float))
sv = df['SV'].values.astype(float)[np.clip(np.searchsorted(traw, tg, side='right') - 1, 0, None)]
mv = df['MV'].values.astype(float)[np.clip(np.searchsorted(traw, tg, side='right') - 1, 0, None)]

U_BP = np.array([0.0, 8.1, 12.3, 22.0, 30.0, 40.0, 55.0, 70.0, 100.0])
Q_BP = np.array([0.0, 9.9, 23.4, 50.4, 85.2, 142.1, 248.4, 335.0, 345.7])


def replay(istep, wc, wo, wr, nd, pre=3.0, dur=10.0):
    i0 = istep - int(pre / DT)
    n = int((pre + dur) / DT)
    u0 = mv[i0]
    va0 = float(np.interp(u0, U_BP, Q_BP))
    z1 = pv[i0]; z2 = -va0; v1 = pv[i0]; v2 = 0.0
    buf = [va0] * (nd + 1)
    out = np.zeros(n)
    for k in range(n):
        i = i0 + k
        v1n = v1 + DT * v2
        v2n = v2 + DT * (-2 * wr * v2 - wr * wr * (v1 - sv[i]))
        v1, v2 = v1n, v2n
        err = pv[i] - z1
        z1 = z1 + DT * (z2 + buf[nd] + 2 * wo * err)
        z2 = z2 + DT * (wo * wo * err)
        vc = min(max(wc * (v1 - z1) - z2, 0.0), Q_BP[-1])
        u = min(max(float(np.interp(vc, Q_BP, U_BP)), 0.0), 100.0)
        buf = [float(np.interp(u, U_BP, Q_BP))] + buf[:-1]
        out[k] = u
    j = int(pre / DT)
    return out[j:]


C1, C2, C3, C4 = '#2b6cb0', '#c05621', '#38a169', '#9f2b68'
fig = plt.figure(figsize=(16.5, 12))
gs = gridspec.GridSpec(3, 2, height_ratios=[0.95, 1, 1], hspace=0.42, wspace=0.22,
                       left=0.05, right=0.985, top=0.93, bottom=0.05)

# ---------- (0) 全场总览 ----------
ax = fig.add_subplot(gs[0, :])
ax.plot(tg / 60, pv, color=C1, lw=0.8)
ax2 = ax.twinx()
ax2.fill_between(tg / 60, mv, color=C2, alpha=0.3, lw=0)
ax2.set_ylabel('MV [%]', color=C2); ax2.set_ylim(0, 108)
ax.set_ylabel('PV [°C]', color=C1); ax.set_xlabel('时间 [min]')
ax.set_ylim(0, 520)
for t0, lab in [(180.3, '①第1档'), (384.9, '②第1档'), (565.3, '③第1档'),
                (784.6, '④加码\nwc≈2.2,wr≈5'), (1677.4, '⑤加码\nwc≈1.8,wr≈4.5')]:
    ax.annotate(lab, xy=(t0 / 60, 405), xytext=(t0 / 60, 480), fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='0.4', lw=0.8))
ax.axvspan(565.3 / 60 + 1.2, 784.6 / 60 - 0.2, color='#f0c040', alpha=0.18)
ax.text((565.3 / 60 + 784.6 / 60) / 2 + 0.5, 60, '改参数', fontsize=9, ha='center', color='0.4')
ax.set_title('7-15 下午 真机开表验证全场（30.6 min）：①②③ = 第1档 (wc=1.0/wo=12/wr=3)，④⑤ = 自行加码（回放辨识）', fontsize=11)

# ---------- (1) 第1档一步 放大 ----------
ax = fig.add_subplot(gs[1, 0])
t0 = 180.3; i0 = int(t0 / DT); n = int(14 / DT); tt = np.arange(n) * DT
ax.axhspan(398, 402, color=C3, alpha=0.15, lw=0)
ax.axhline(400, color=C3, lw=0.8, ls='--', alpha=0.7)
ax.plot(tt, pv[i0:i0 + n], color=C1, lw=1.5)
axr = ax.twinx(); axr.fill_between(tt, mv[i0:i0 + n], color=C2, alpha=0.3, lw=0)
axr.set_ylim(0, 108); axr.set_ylabel('MV [%]', color=C2)
ax.set_ylim(95, 430); ax.set_xlabel('相对 SV 阶跃 [s]'); ax.set_ylabel('PV [°C]')
ax.set_title('第1档（第①步）：无振铃、超调 1.0°，但到达 400 要 5.6s\n'
             '孪生预测 2.7s升/0°/5.4s收敛 —— 命中（收尾爬行即"无超调的代价"）', fontsize=10)
ax.annotate('收尾爬行段\n(最后 3° 花了 ~2.5s)', xy=(4.6, 398.5), xytext=(7.5, 340),
            fontsize=8.5, arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))

# ---------- (2) 加码一步 放大 ----------
ax = fig.add_subplot(gs[1, 1])
t0 = 1677.4; i0 = int(t0 / DT); n = int(14 / DT); tt = np.arange(n) * DT
ax.axhspan(398, 402, color=C3, alpha=0.15, lw=0)
ax.axhline(400, color=C3, lw=0.8, ls='--', alpha=0.7)
ax.plot(tt, pv[i0:i0 + n], color=C1, lw=1.5)
axr = ax.twinx(); axr.fill_between(tt, mv[i0:i0 + n], color=C2, alpha=0.3, lw=0)
axr.set_ylim(0, 108); axr.set_ylabel('MV [%]', color=C2)
ax.set_ylim(95, 430); ax.set_xlabel('相对 SV 阶跃 [s]'); ax.set_ylabel('PV [°C]')
ax.set_title('加码（第⑤步，wc≈1.8/wr≈4.5）：2.0s 到达，但超调 12.1°\n'
             '+12°→−9°→+5° 周期 ~2.6s 衰减回摆 = 进弯速度 × 元件在途热量', fontsize=10)
for t_, v_, s_ in [(2.4, 412.1, '+12.1°'), (3.6, 390.8, '−9.2°'), (5.1, 405.2, '+5.2°')]:
    ax.annotate(s_, xy=(t_, v_), fontsize=8.5, color=C4, ha='center',
                xytext=(t_, v_ + (10 if v_ > 400 else -14)))

# ---------- (3)(4) 回放证据 ----------
for col, (t0, idp, lab_id, lab_step) in enumerate([
        (1677.4, (1.8, 6., 4.5, 20), 'wc=1.8, wo=6, wr=4.5, nd=20  (RMSE 1.8%)', '第⑤步'),
        (784.6,  (2.2, 6., 5., 15),  'wc=2.2, wo=6, wr=5, nd=15  (RMSE 6.0%)',  '第④步')]):
    ax = fig.add_subplot(gs[2, col])
    istep = int(t0 / DT); dur = 8.0; tt = np.arange(int(dur / DT)) * DT
    um = mv[istep:istep + int(dur / DT)]
    ax.fill_between(tt, um, color=C2, alpha=0.30, lw=0, label='实测 MV（日志）')
    r_id = replay(istep, *[int(x) if i == 3 else x for i, x in enumerate(idp)])[:len(tt)]
    r_b = replay(istep, 1.5, 8., 3., 24)[:len(tt)]
    ax.plot(tt, r_id, '-', color=C4, lw=1.6, label=f'回放·辨识参数 {lab_id}')
    ax.plot(tt, r_b, '--', color=C1, lw=1.6, label='回放·假设"第2档 wc=1.5/wr=3"')
    ax.set_ylim(0, 108); ax.set_xlabel('相对 SV 阶跃 [s]'); ax.set_ylabel('MV [%]')
    pk_m, pk_b = um.max(), r_b.max()
    ax.annotate(f'实测冲到 {pk_m:.0f}%', xy=(tt[np.argmax(um)], pk_m), xytext=(2.6, 92),
                fontsize=9, color=C2, arrowprops=dict(arrowstyle='->', color=C2, lw=0.9))
    ax.annotate(f'第2档方程最多只会给 {pk_b:.0f}%', xy=(tt[np.argmax(r_b)], pk_b),
                xytext=(2.6, 52), fontsize=9, color=C1,
                arrowprops=dict(arrowstyle='->', color=C1, lw=0.9))
    ax.set_title(f'回放证据（{lab_step}）：同一段实测 PV 喂给控制器方程，比谁发的 MV\n'
                 '紫线（辨识参数）与实测咬合；蓝虚线（第2档假设）幅度差 2~3 倍', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')

fig.suptitle('AIC9 通道1 · 7-15 下午真机验证：第1档通过 · ④⑤为加码参数（回放辨识铁证） · 绿带 = 400±2°C 目标带',
             fontsize=13, y=0.975)
out = '/home/yiz/workspace/src/control_test/plots/13_0715下午_真机验证与回放证据.png'
fig.savefig(out, dpi=130)
print('saved:', out)
