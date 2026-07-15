#!/usr/bin/env python3
"""控制器回放辨识 (开表版): 把实测 PV/SV 逐拍喂给 FB_ADRC_Base 复刻, 比对命令 MV,
确认真机某次阶跃实际在跑的参数 (不依赖对象模型, 只考控制器方程)。

用法:
    python3 analysis/replay_identify.py <csv> <step_t0_s> [wc wo wr nd]
      不带参数 => 扫网格找最匹配的 (wc, wo, wr, nd), 打印前 5
      带 4 个参数 => 只回放这一组, 打印 RMSE 与特征对比 (MV峰值/起步斜率)

先例: 7-15 下午 session, 第①步 1.07% 坐实第1档; 第⑤步 1.76% 坐实 wc≈1.8/wr≈4.5
(而"第2档 wc=1.5/wr=3"假设 RMSE 18.5%, MV 峰 38% vs 实测 92% -- 两个家族的波形)。
"""
import sys
import numpy as np
import pandas as pd

DT = 0.01
# 已下发的 q∞ 表 (adrc_map_enabled.st 默认值)
U_BP = np.array([0.0, 8.1, 12.3, 22.0, 30.0, 40.0, 55.0, 70.0, 100.0])
Q_BP = np.array([0.0, 9.9, 23.4, 50.4, 85.2, 142.1, 248.4, 335.0, 345.7])


def load(path):
    df = pd.read_csv(path)
    df.columns = ['time', 'PV1', 'PV2', 'SV', 'MV'][: len(df.columns)]
    traw = (pd.to_datetime(df['time']) - pd.to_datetime(df['time'].iloc[0])).dt.total_seconds().values
    tg = np.arange(0, traw[-1], DT)
    pv = np.interp(tg, traw, df['PV1'].values.astype(float))
    idx = np.clip(np.searchsorted(traw, tg, side='right') - 1, 0, None)
    sv = df['SV'].values.astype(float)[idx]
    mv = df['MV'].values.astype(float)[idx]
    return pv, sv, mv


def replay(pv, sv, mv, istep, wc, wo, wr, nd, tauf=0.0, pre=3.0, dur=12.0):
    """从阶跃前 pre 秒进入 (稳态 MV 反推 z2 种子), 返回 (预测MV[阶跃后], 实测MV[阶跃后])"""
    i0 = istep - int(pre / DT)
    n = int((pre + dur) / DT)
    va0 = float(np.interp(mv[i0], U_BP, Q_BP))
    z1 = pv[i0]; z2 = -va0; v1 = pv[i0]; v2 = 0.0
    buf = [va0] * (nd + 1); vf = va0
    out = np.zeros(n)
    for k in range(n):
        i = i0 + k
        v1n = v1 + DT * v2
        v2n = v2 + DT * (-2 * wr * v2 - wr * wr * (v1 - sv[i]))
        v1, v2 = v1n, v2n
        vd = buf[nd]
        if tauf > 0:
            vf += DT * (vd - vf) / tauf
            vd = vf
        err = pv[i] - z1
        z1 = z1 + DT * (z2 + vd + 2 * wo * err)
        z2 = z2 + DT * (wo * wo * err)
        vc = min(max(wc * (v1 - z1) - z2, 0.0), Q_BP[-1])
        u = min(max(float(np.interp(vc, Q_BP, U_BP)), 0.0), 100.0)
        buf = [float(np.interp(u, U_BP, Q_BP))] + buf[:-1]
        out[k] = u
    j = int(pre / DT)
    return out[j:], mv[i0 + j: i0 + n]


def main():
    path, t0 = sys.argv[1], float(sys.argv[2])
    pv, sv, mv = load(path)
    istep = int(t0 / DT)
    if len(sys.argv) >= 7:
        wc, wo, wr, nd = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
        up, um = replay(pv, sv, mv, istep, wc, wo, wr, nd)
        rm = float(np.sqrt(np.mean((up - um) ** 2)))
        print(f'wc={wc} wo={wo} wr={wr} nd={nd}: RMSE {rm:.2f}%')
        print(f'  MV峰: 预测 {up.max():.1f}% / 实测 {um.max():.1f}%')
        return
    rows = []
    for wc in (0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5):
        for wo in (6., 8., 10., 12., 16.):
            for wr in (3., 3.5, 4., 4.5, 5.):
                for nd in (12, 15, 20, 24, 31):
                    up, um = replay(pv, sv, mv, istep, wc, wo, wr, nd)
                    rows.append((float(np.sqrt(np.mean((up - um) ** 2))), wc, wo, wr, nd))
    rows.sort()
    print(f'阶跃 t0={t0}s 最匹配的参数 (RMSE% | wc wo wr nd):')
    for r, wc, wo, wr, nd in rows[:5]:
        print(f'  {r:6.2f} | wc={wc} wo={wo:.0f} wr={wr} nd={nd}')
    print('(经验: <2% 坐实; 2~6% 参数在网格点附近; >10% 假设不成立)')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main()
