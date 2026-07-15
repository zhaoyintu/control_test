#!/usr/bin/env python3
"""
震荡诊断: 从震荡段 CSV 里提取 震荡周期 / PV幅值 / MV幅值 / MV->PV 相位滞后,
用于区分两种根因:
  A) 纯滞后偏大 (θ_eff 0.3~0.6s):  T_osc ≈ 2~4×θ_eff, 即周期 1~2.5s 量级
  B) 未建模慢惯性 (加热元件热容/传感器套管/PV滤波, τ≈0.5~1.5s):
     T_osc 会到 3~8s 量级, 且 MV->PV 相位滞后远大于 nDeadSteps 补偿的值

用法:
    python3 analysis/osc_diagnose.py <csv> [t_start] [t_end]
    不给 t_start/t_end 时自动找 PV 相对 SV 波动最大的 60s 窗口。
"""
import sys
import numpy as np
import pandas as pd

DT = 0.01


def dominant_period(x, dt):
    """去趋势后用自相关找主周期"""
    x = x - np.mean(x)
    n = len(x)
    ac = np.correlate(x, x, 'full')[n - 1:]
    ac /= (ac[0] + 1e-12)
    # 找第一个正峰 (跳过 0 滞后附近)
    i0 = int(0.15 / dt)
    for i in range(i0, len(ac) - 1):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > 0.2:
            return i * dt, ac[i]
    return np.nan, 0.0


def main():
    path = sys.argv[1]
    df = pd.read_csv(path)
    df.columns = ['time', 'PV1', 'PV2', 'SV', 'MV'][: len(df.columns)]
    pv = df['PV1'].values.astype(float)
    sv = df['SV'].values.astype(float)
    mv = df['MV'].values.astype(float)
    n = len(df)
    t = np.arange(n) * DT

    if len(sys.argv) >= 4:
        i0, i1 = int(float(sys.argv[2]) / DT), int(float(sys.argv[3]) / DT)
    else:
        # 自动找 PV-SV 残差滚动std最大的 60s 窗口 (SV 恒定的段内)
        err = pv - sv
        w = int(60 / DT)
        best, i0 = -1.0, 0
        step = int(5 / DT)
        for s in range(0, n - w, step):
            if np.ptp(sv[s:s + w]) > 0:     # 跨越 SV 变化的窗口跳过
                continue
            v = float(np.std(err[s:s + w]))
            if v > best:
                best, i0 = v, s
        i1 = i0 + w
    i1 = min(i1, n)
    seg_pv, seg_mv, seg_sv = pv[i0:i1], mv[i0:i1], sv[i0:i1]
    print(f'分析窗口: t = {t[i0]:.1f} ~ {t[i1-1]:.1f} s   SV = {seg_sv[0]:.0f}')

    # 幅值
    pv_amp = float(np.ptp(seg_pv)) / 2
    mv_amp = float(np.ptp(seg_mv)) / 2
    pv_std = float(np.std(seg_pv - seg_pv.mean()))
    print(f'PV 振幅 ≈ ±{pv_amp:.1f} °C (std {pv_std:.2f})    MV 振幅 ≈ ±{mv_amp:.1f} %')
    print(f'MV 是否顶到限幅: min={seg_mv.min():.1f}%  max={seg_mv.max():.1f}%'
          f'  {"<-- 打满, 属于硬限幅极限环" if seg_mv.min() < 0.5 or seg_mv.max() > 99.5 else ""}')

    # 主周期
    T_pv, q1 = dominant_period(seg_pv, DT)
    T_mv, q2 = dominant_period(seg_mv, DT)
    print(f'PV 主周期 T_osc ≈ {T_pv:.2f} s (自相关峰 {q1:.2f})')
    print(f'MV 主周期        ≈ {T_mv:.2f} s (自相关峰 {q2:.2f})')

    # MV -> PV 互相关滞后 (取 ±T_osc 范围)
    if not np.isnan(T_pv):
        a = seg_mv - seg_mv.mean()
        b = seg_pv - seg_pv.mean()
        m = int(min(T_pv * 1.2, 8.0) / DT)
        cc = [float(np.dot(a[: len(a) - k], b[k:])) /
              (np.std(a) * np.std(b) * (len(a) - k) + 1e-12) for k in range(m)]
        k_best = int(np.argmax(cc))
        print(f'MV→PV 互相关滞后 ≈ {k_best * DT:.2f} s (相关 {cc[k_best]:.2f})')
        print()
        print('---- 判读 ----')
        print(f'若为"延迟主导"型失稳, 总有效滞后 θ_eff ≈ T_osc/4 ≈ {T_pv/4:.2f} s')
        print('对照: 当前 nDeadSteps 补偿 = lrTheta 值; 差距即未补偿部分。')
        if T_pv >= 3.0:
            print('>> 周期 ≥3s: 更像"慢惯性主导"(元件热容/传感器/PV滤波 ~秒级极点),')
            print('   模型需要加二阶极点, wc 上限由该惯性决定, 单纯加 nDeadSteps 治不了。')
        elif T_pv >= 0.8:
            print('>> 周期 0.8~3s: 延迟主导, θ_eff 明显大于当前补偿 → 实测θ后调 nDeadSteps,')
            print('   wc 上限 ≈ (0.3~0.5)/θ_eff。')
        else:
            print('>> 周期 <0.8s: 高频, 先查 wo 是否过高 / lrCycleT 与任务周期是否一致。')

    print()
    print('请顺带确认并回复三件事:')
    print(' 1) 震荡发生时的整组参数 (lrK/lrWc/lrWo/lrWr/lrTheta)')
    print(' 2) ADRC FB 所在任务的真实执行周期, 与传入的 lrCycleT 是否一致')
    print(' 3) PV1 进 FB 之前有没有仪表/AI通道/程序里的输入滤波, 滤波时间常数多少')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()
