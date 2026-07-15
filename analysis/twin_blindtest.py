#!/usr/bin/env python3
"""孪生 v2 盲测: 预测 7-13 晚场四组闭环工况 (v2 从未见过的数据), v1 同题对照
结论 (2026-07-15):
  - 温和工况 (S5) 三项指标 ±10% 内全中
  - 危险工况方向正确且偏保守: wc=8 报剧烈振荡 (实测衰减振铃), 振铃周期 1.53s vs 实测 1.65s
  - 真机 1.6s 神秘振铃被 θ+τe 滞后链解释; v1 在 S3/S4 说"没事/无振铃" -- 正是当年事故根源
  - 已知盲区: S2 旧参数实测 13.5s 长尾未复现 (真机噪声/滤波在 wo=40 下反复出带, 孪生噪声太干净)
用法: python3 analysis/twin_blindtest.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin import Twin, metrics

HERE = os.path.dirname(os.path.abspath(__file__))


def ring_period(y, tgt, dt=0.01):
    """到达后误差峰间距的中位数 [s]"""
    e = y - tgt
    i0 = np.argmax(y > tgt - 30)
    e = e[i0:]
    pk = [i for i in range(1, len(e) - 1) if e[i] > e[i-1] and e[i] > e[i+1] and abs(e[i]) > 0.15]
    if len(pk) < 3:
        return np.nan
    return float(np.median(np.diff(pk))) * dt


# 7-13 晚场四组: (标签, b0, wc, wo, wr, nd, 实测 rise/ov/settle, 实测振铃周期)
CASES = [
    ('S2 旧参数',  10.0, 0.8, 40.0, 10.0, 20, (2.91, 3.2, 13.5), None),
    ('S3 wc=8',    3.4, 8.0, 24.0,  3.0, 23, (1.86, 1.9,  8.2), 1.65),
    ('S4 wc=2',    3.4, 2.0, 24.0,  3.0, 23, (1.81, 2.3,  6.0), 1.62),
    ('S5 wc=0.9',  3.4, 0.9, 24.0,  3.0, 23, (3.34, 0.9,  6.7), None),
]

if __name__ == '__main__':
    print(f"{'工况':14s} {'':6s} {'rise[s]':>8s} {'超调[°C]':>8s} {'settle[s]':>9s} {'振铃[s]':>7s}")
    for name, path in [('孪生v2', None), ('孪生v1', os.path.join(HERE, 'twin_params_v1.json'))]:
        tw = Twin(path) if path else Twin()
        print(f'--- {name} 盲测 vs 实测 ---')
        for lab, b0, wc, wo, wr, nd, meas, ringm in CASES:
            yy, uu = tw.closed_loop_adrc(b0, wc, wo, wr, nd, 100.0, 400.0, T=24.0, pre=1.5)
            r90, ov, stl = metrics(yy, 100.0, 400.0)
            rp = ring_period(yy, 400.0)
            rps = f'{rp:7.2f}' if np.isfinite(rp) else '   无'
            rm = f'{ringm:.2f}' if ringm else '无'
            print(f'{lab:12s} 预测: {r90:8.2f} {ov:8.1f} {stl:9.1f} {rps}')
            print(f'{"":12s} 实测: {meas[0]:8.2f} {meas[1]:8.1f} {meas[2]:9.1f} {rm:>7s}')
