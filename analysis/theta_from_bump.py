#!/usr/bin/env python3
"""
q(u) 标定 session 判读: 从 10ms CSV 自动找出所有手动 bump, 输出
  1) 死区时间 θ (小幅 bump 三连发的中位数 + 一致性检查)
  2) q(u) 实测断点 (每个电平: 起动斜率 + 散热补偿)
  3) 可直接粘贴进 ST 变量表的 aMapU / aMapV / nMapPoints / lrTheta
  4) --write-twin 时把结果写入 analysis/twin_params.json (孪生 v2)

用法:
    python3 analysis/theta_from_bump.py <AIC9_DATA-xxx.csv> [--write-twin]

判定为手动 bump: MV 单拍跳变 ≥3% 且之后 ≥0.35s 保持在新值 ±1%
(ADRC 自动模式的 MV 连续变化, 不会误报)

漂移容忍 (v2): 跳变前不要求恒温 -- 真机反馈开环冻结 MV 拿不住恒温
(工作点附近 1% MV 平衡温差 ≈17°C, 散热时间常数 ≈7s)。
脚本对跳变前 2s 基线拟合直线, 得到漂移速率 d 和外推起点 T0:
  θ:  PV 连续 5 点越出 [趋势线 + 5σ噪声带] 判起动 (漂移被趋势线扣除)
  q:  跳变后初期温度仍≈T0 => slope = q(u_post) - h1*(T0-c)
      => q(u_post) = slope + h1*(T0-c)     (与跳变前是否平衡无关)
  稳态白送点: q(u_pre) = d + h1*(T0-c)     (d=0 退化为原平衡关系)
h1/c 取自 twin_params.json (孪生 v1)。
作废条件: |d|>2°C/s (上一拍没回稳就切手动) 或基线弯折 (窗口跨了模式切换/回程)。
=> 操作要求从"手动恒温 20s"改为: 切手动后停 3~5s 即跳变。

稳态锚点 (v2.1): 自动识别所有"稳住≥30s"的段 (PV 波动<1°C 且 MV 平稳),
按平衡关系折算 q(u_ss)=h1*(T-c), 与低温 bump 表对比给出偏差 -- 用于回答
"200~350°C 测的 q 表在工作温度 450°C 差多少"。只需在 session 结尾
自动模式 SV=450 稳 60s (纯正常工况, 零风险)。锚点只作检查, 不进表。
"""
import sys
import json
import os
import numpy as np
import pandas as pd

DT = 0.01
JUMP_MIN = 3.0       # bump 判定: 最小单拍跳变 [%]
HOLD_S = 0.35        # 跳变后 MV 需保持的最短时长 [s] (100% 电平只保持 0.4s)
HOLD_TOL = 1.0
BASE_S = 8.0         # 噪声σ估计窗口 [s]
BASE_GAP_S = 0.2
TREND_S = 2.0        # 基线趋势拟合窗口 [s] -- 需全落在切手动之后 => 手动停 3~5s 再跳
TREND_GAP_S = 0.1
DRIFT_SLOPE_MAX = 2.0  # 跳变前允许的最大漂移速率 [°C/s]
SEARCH_S = 1.5       # 起动搜索窗口 [s]
RUN_N = 5            # 连续 5 点超阈值
THETA_DU_MAX = 12.0  # |Δu| 不超过此值的 bump 计入 θ 统计 (小幅信噪比合适)

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(HERE, 'twin_params.json')) as f:
        _tw = json.load(f)
    H1, C = _tw['h1'], _tw['c']
except Exception:
    H1, C = 0.1353, 127.0


def find_anchors(pv, mv, t):
    """稳住≥30s 的段 (PV 峰谷差<1°C 且 MV std<1.5) -> (u_med, T_med, t0, t1) 列表"""
    win = int(30 / DT)
    step = int(5 / DT)
    marks = [i for i in range(0, len(t) - win, step)
             if np.ptp(pv[i:i + win]) < 1.0 and float(np.std(mv[i:i + win])) < 1.5]
    anchors = []
    k = 0
    while k < len(marks):
        i0 = marks[k]
        end = i0 + win
        k += 1
        while k < len(marks) and marks[k] <= end:   # 只归并真正连续覆盖的窗口
            end = marks[k] + win
            k += 1
        seg = slice(i0, end)
        anchors.append((float(np.median(mv[seg])), float(np.median(pv[seg])),
                        t[i0], t[end - 1]))
    return anchors


def print_anchors(anchors, u_out, q_out):
    ok = [(u, T, a, b) for u, T, a, b in anchors if T - C >= 20.0]
    nskip = len(anchors) - len(ok)
    if not ok:
        if anchors:
            print(f'\n(找到 {len(anchors)} 段稳态但都在 T<{C+20:.0f}°C, 线性散热模型在低温段不适用, 不作锚点)')
        return
    print('\n稳态锚点 (稳住≥30s 的段, q = h1*(T-c); 检查低温 bump 表在各温度的适用性):')
    print(f"{'t区间[s]':>15s} {'T[°C]':>7s} {'u[%]':>6s} {'q锚点':>7s} {'查表q(u)':>9s} {'偏差':>7s}")
    for u, T, a, b in ok:
        qa = H1 * (T - C)
        if len(u_out) >= 2 and u_out[0] <= u <= u_out[-1]:
            qt = float(np.interp(u, u_out, q_out))
            qts, dev = f'{qt:9.1f}', f'{(qa - qt) / qt * 100:+5.1f}%'
        else:
            qts, dev = f"{'—':>9s}", f"{'—':>7s}"
        print(f'{a:7.0f}~{b:6.0f} {T:7.1f} {u:6.1f} {qa:7.1f} {qts} {dev:>7s}')
    print('  (锚点不进表, 仅作检查; 工作温度锚点偏差 ≤±10% => 温度依赖可忽略, 更大 => 高温段需复核)')
    if nskip:
        print(f'  (另有 {nskip} 段 T<{C+20:.0f}°C 的稳态段跳过: 线性散热模型不适用)')


def main(path, write_twin=False):
    df = pd.read_csv(path)
    df.columns = ['time', 'PV1', 'PV2', 'SV', 'MV'][: len(df.columns)]
    traw = (pd.to_datetime(df['time']) - pd.to_datetime(df['time'].iloc[0])).dt.total_seconds().values
    tg = np.arange(0, traw[-1], DT)
    pv = np.interp(tg, traw, df['PV1'].values.astype(float))
    # MV 是零阶保持信号: 用前值保持重采样, 线性插值会把一步跳变劈成多个小步导致漏检
    mv = df['MV'].values.astype(float)[np.clip(np.searchsorted(traw, tg, side='right') - 1, 0, None)]
    n = len(tg)
    t = tg

    hold_n = int(HOLD_S / DT)
    base_n = int(BASE_S / DT)
    gap_n = int(BASE_GAP_S / DT)
    trend_n = int(TREND_S / DT)
    tgap_n = int(TREND_GAP_S / DT)

    jumps = []
    dmv = np.diff(mv)
    cand = np.where(np.abs(dmv) >= JUMP_MIN)[0] + 1
    last = -10 * hold_n
    for i0 in cand:
        if i0 - last < hold_n or i0 + hold_n >= n or i0 < base_n + gap_n:
            continue
        if np.max(np.abs(mv[i0: i0 + hold_n] - mv[i0])) > HOLD_TOL:
            continue
        jumps.append(i0)
        last = i0
    if not jumps:
        print('没找到手动 bump。确认测试时 MV 是一步跳变而不是爬坡。')
        return

    print(f'找到 {len(jumps)} 次 bump  (散热参数 h1={H1:.4f}, c={C:.0f} 取自孪生v1; 基线漂移自动扣除):\n')
    print(f"{'#':>2s} {'t0[s]':>8s} {'T0[°C]':>7s} {'MV: 前->后':>13s} {'漂移[°C/s]':>9s} {'θ[s]':>7s} "
          f"{'斜率[°C/s]':>10s} {'q(u_post)':>9s}  备注")

    thetas, qpts = [], []      # qpts: (u, q)
    for k, i0 in enumerate(jumps):
        # 噪声 σ: 8s 窗、滚动均值去趋势 (对慢漂移不敏感)
        base = pv[i0 - base_n - gap_n: i0 - gap_n]
        sig = float(np.std(base - pd.Series(base).rolling(50, center=True, min_periods=1).mean().values))
        # 基线趋势: 跳变前 2s 线性拟合 -> 漂移速率 d 与外推到跳变时刻的 T0
        k1 = i0 - tgap_n
        k0 = k1 - trend_n
        Ab = np.column_stack([t[k0:k1] - t[i0], np.ones(k1 - k0)])
        coef = np.linalg.lstsq(Ab, pv[k0:k1], rcond=None)[0]
        d, pv0 = float(coef[0]), float(coef[1])
        fit_rms = float(np.sqrt(np.mean((pv[k0:k1] - Ab @ coef) ** 2)))
        u_pre, u_post = float(np.median(mv[i0 - 50:i0 - 2])), float(mv[i0 + 2])
        du = u_post - u_pre
        sgn = 1.0 if du > 0 else -1.0
        thr = max(5 * sig, 0.10)
        void = None
        if abs(d) > DRIFT_SLOPE_MAX:
            void = f'作废:跳变前在漂移/爬温({d:+.1f}°C/s), 上一拍未回稳'
        elif fit_rms > max(3 * sig, 0.08):
            void = '作废:基线弯折(切手动后不足2s就跳? 停3~5s再跳)'
        if void:
            print(f'{k+1:2d} {t[i0]:8.1f} {pv0:7.1f} {u_pre:5.1f}->{u_post:6.1f} '
                  f'{d:+9.2f} {"—":>7s} {"—":>10s} {"—":>9s}  {void}')
            continue
        theta = None
        lim = min(n, i0 + int(SEARCH_S / DT))
        run = 0
        for i in range(i0, lim):
            if sgn * (pv[i] - (pv0 + d * (t[i] - t[i0]))) > thr:
                run += 1
                if run >= RUN_N:
                    theta = t[i - RUN_N + 1] - t[i0]
                    i_dep = i - RUN_N + 1
                    break
            else:
                run = 0
        if theta is None:
            print(f'{k+1:2d} {t[i0]:8.1f} {pv0:7.1f} {u_pre:5.1f}->{u_post:6.1f} '
                  f'{d:+9.2f} {"—":>7s} {"—":>10s} {"—":>9s}  {SEARCH_S}s 内未起动?')
            continue
        # 斜率窗口: 起动后到 MV 保持结束前 (最长 0.5s)
        hold_end = i0
        while hold_end < n - 1 and abs(mv[hold_end] - u_post) <= HOLD_TOL:
            hold_end += 1
        j1 = min(i_dep + int(0.5 / DT), hold_end + int(theta / DT))  # 延迟窗口内的响应仍属保持期
        j0 = i_dep
        if j1 - j0 < int(0.12 / DT):
            j1 = min(i_dep + int(0.2 / DT), n)
        A = np.column_stack([t[j0:j1], np.ones(j1 - j0)])
        slope = float(np.linalg.lstsq(A, pv[j0:j1], rcond=None)[0][0])
        # 散热按窗口内平均温度逐点补偿 (按起点 T0 补会低估大电平/长保持的 q)
        q_post = slope + H1 * (float(np.mean(pv[j0:j1])) - C)
        note = ''
        if abs(du) <= THETA_DU_MAX:
            thetas.append(theta)
            note = 'θ样本'
        if du > 0:
            qpts.append((u_post, q_post))   # 跳变后斜率点只收上跳 (下跳受加热元件蓄热放热污染)
        if pv0 - C >= 20.0:                 # 低温段线性散热模型不可信, 白送点不收
            qpts.append((u_pre, d + H1 * (pv0 - C)))   # 跳变前稳态白送点: 与跳变方向无关
        print(f'{k+1:2d} {t[i0]:8.1f} {pv0:7.1f} {u_pre:5.1f}->{u_post:6.1f} '
              f'{d:+9.2f} {theta:7.3f} {slope:10.1f} {q_post:9.1f}  {note}')

    # ---------------- θ ----------------
    print()
    if thetas:
        med = float(np.median(thetas))
        rng = float(np.ptp(thetas)) if len(thetas) > 1 else 0.0
        ok = rng <= 0.03 or len(thetas) < 2
        print(f'θ 中位数 = {med:.3f}s  极差 = {rng:.3f}s  ({len(thetas)} 个小幅样本)'
              + ('' if ok else '  << 极差超 0.03s, 建议补测'))
        lr_theta = round(med + 0.02, 2)
        print(f'(上行纯滞后 θ̂+0.02 = {lr_theta:.2f}s; 开查表的 lrTheta 以爬梯档位为准 --'
              f' 元件热惯性 τe≈0.24s 需一并补偿, 见 HTML 落点③: 0.31/0.24)')
    else:
        med, lr_theta = None, None
        print('无 θ 样本 (小幅 bump), 只能沿用现值')

    # ---------------- q(u) 表 ----------------
    anchors = find_anchors(pv, mv, t)
    if not qpts:
        print('无 q 样本 (没有基线干净的上跳 bump)')
        print_anchors(anchors, [], [])
        return
    qpts.append((0.0, 0.0))
    us = np.array([p[0] for p in qpts]); qs = np.array([p[1] for p in qpts])
    # 相近电平 (±2%) 聚类取中位
    order = np.argsort(us)
    us, qs = us[order], qs[order]
    u_out, q_out = [], []
    i = 0
    while i < len(us):
        j = i
        while j + 1 < len(us) and us[j + 1] - us[i] <= 2.0:
            j += 1
        u_out.append(round(float(np.median(us[i:j+1])), 1))
        q_out.append(round(float(np.median(qs[i:j+1])), 1))
        i = j + 1
    mono = all(q_out[i] < q_out[i+1] for i in range(len(q_out)-1))
    print(f'\nq(u) 实测断点 ({len(u_out)} 点){"" if mono else "  << 警告: 不单调, 检查异常行后重测该电平"}:')
    print('  u[%]  :', '  '.join(f'{u:6.1f}' for u in u_out))
    print('  q[°C/s]:', '  '.join(f'{q:6.1f}' for q in q_out))
    if max(u_out) < 99:
        print(f'  (最高实测电平 {max(u_out):.0f}%, 以上区间控制器会按末段斜率外推 -- 尽量补测 100%)')

    print_anchors(anchors, u_out, q_out)

    npts = len(u_out)
    if npts > 16:
        print('断点超过 16 个, 请合并后再填表')
        return
    pad = 16 - npts
    print('\n==> 粘贴进 ST 变量表 (FB_ADRC_Base 输入):')
    print(f"  nMapPoints := {npts};")
    print(f"  aMapU := [{', '.join(f'{u:.1f}' for u in u_out)}"
          + (f", {pad}(0.0)];" if pad else '];'))
    print(f"  aMapV := [{', '.join(f'{q:.1f}' for q in q_out)}"
          + (f", {pad}(0.0)];" if pad else '];'))
    print('  lrK := 1.0;   (* v 空间增益恒 1, 原 lrK/lrTau 均不再使用 *)')
    if lr_theta:
        print(f'  (* lrTheta 按爬梯档位: 第1档 0.31 / 第2档 0.24; 上行纯滞后实测 {lr_theta:.2f} 仅供参考 *)')

    if write_twin:
        out = dict(u_bp=u_out, q_bp=q_out,
                   theta=(med if med else _tw.get('theta', 0.15)),
                   tau2=_tw.get('tau2', 0.05), h1=H1, c=C,
                   sig_n=_tw.get('sig_n', 0.05),
                   note=f'孪生 v2: q(u)/θ 由 {os.path.basename(path)} bump session 实测')
        with open(os.path.join(HERE, 'twin_params.json'), 'w') as f:
            json.dump(out, f, indent=1)
        print('\n已写入 analysis/twin_params.json (孪生 v2)')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], write_twin='--write-twin' in sys.argv)
