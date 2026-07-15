#!/usr/bin/env python3
"""验证 theta_from_bump.py v2 的漂移容忍: 孪生造数 -> 脚本判读 -> 对比真值"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/yiz/workspace/src/control_test/analysis')
from twin import Twin, DT

import tempfile; OUT = tempfile.gettempdir()
tw = Twin()
y0 = 300.0
u_eq = tw.steady_u(y0)


def seg(u, dur):
    return [float(u)] * int(round(dur / DT))


def build(freeze_errs, tag):
    """三次小幅 bump (+8%) + 一拍大电平 (30%), 手动段带冻结误差 -> 漂移"""
    uu = seg(u_eq, 25)                            # "自动"稳态段
    for derr in freeze_errs:
        uu += seg(u_eq + derr, 4)                 # 切手动, 冻结值偏 derr -> 开始漂
        uu += seg(u_eq + derr + 8.0, 2.0)         # 一步 +8%, 保持 2s
        uu += seg(u_eq + derr, 0.5)               # 回 u0 (下跳, 基线是爬升段, 应被作废)
        uu += seg(u_eq, 28)                       # "切回自动"回稳
    uu += seg(u_eq + (freeze_errs[0] if freeze_errs[0] else 0.0) + 0.3, 4)
    uu += seg(30.0, 2.0)                          # 大电平拍
    uu += seg(u_eq, 30)                           # 回稳
    uu += seg(tw.steady_u(450.0), 80)             # 跳到 450 平衡电平并久稳 -> 高温锚点
    u = np.array(uu)
    y = tw.open_loop(u, y0)
    rng = np.random.default_rng(42)
    pv = y + rng.normal(0, tw.sig_n, len(y))
    ts = pd.to_datetime('2026-07-15 10:00:00') + pd.to_timedelta(np.arange(len(u)) * DT, unit='s')
    df = pd.DataFrame({'time': ts.strftime('%Y-%m-%d %H:%M:%S.%f'),
                       'PV1': np.round(pv, 3), 'PV2': np.round(pv, 3),
                       'SV': 300.0, 'MV': np.round(u, 2)})
    path = f'{OUT}/synth_{tag}.csv'
    df.to_csv(path, index=False)
    return path


print(f'孪生真值: u_eq={u_eq:.2f}%  θ=0.15s  q(u_eq+8)={tw.q_of(u_eq+8):.1f}  '
      f'q(30)={tw.q_of(30.0):.1f}  q(u_eq)={tw.q_of(u_eq):.1f} °C/s')
p_drift = build([+0.25, -0.20, +0.35], 'drift')   # 冻结误差 -> 漂移 0.3~0.8°C/s 量级
p_clean = build([0.0, 0.0, 0.0], 'clean')         # 完美恒温对照
print('\n================ 带漂移 (真机场景) ================')
sys.argv = ['x', p_drift]
import importlib
m = importlib.import_module('theta_from_bump')
m.main(p_drift)
print('\n================ 恒温对照 ================')
m.main(p_clean)
