#!/usr/bin/env python3
"""第4档可行性: 放开 lrMVMax 后 "1.5s 到 400 / 超调≤3° / 后续振荡≤3°" 能否成立 (2026-07-17)

关键前置修正 -- 高段马力实测覆盖孪生外推:
  7-16 标定场 6 段 90% 满幅冲刺的末段斜率 (qf 已充满 87~96%) 给出
  q_eff(90%, 380~425°C) ≈ 266~284 °C/s, 即 g ≈ 0.80 (对表值 339);
  孪生 v3 的 kq=1.1/下限0.4 在该区间预测 g=0.4 (135) -- 悲观一倍。
  v3 的 kq=1.1 是在第3档闭环包络 (u≤60) 内拟合的, 其中混入了未建模的
  冷炉体吸热; 在 u=90~100 包络按直接测量取 kq=0.2 (g(400)=0.83)。

结论要点 (运行本脚本复现):
  A. bang-bang 物理极限: 理想逐拍开关下 100→400 最快 1.55~1.61s (u=100/90),
     200→400 约 1.2~1.35s。元件蓄热 qf·(θ+τe) ≈ 70~85° 的滑行段无法缩短
     (只热不冷), 断电点差 ±5° 结果天壤之别 -> 1.5s 目标压在物理地板之下。
  B. MV 上限不是瓶颈: 上限 60→90 到达 2.58→2.62s (无增益; 真机第3档冲刺
     MV 峰值实测仅 53~58%, 连 60 都没顶满)。统一律价格表 (标称, 100→400):
     首超≤3° -> 2.6s | 4.4° -> 2.30s | 5.4° -> 2.13s | 6.5° -> 1.94s |
     1.38s -> 39°。前沿全部 wc=1.5, 速度旋钮是 wr。超调后单调回落不振荡,
     "后续振荡≤3°"自动满足, 付钱的只有首超。
  C. 应力: 任何档位冷炉/漂移日再加 5~8° (机理同冷启动, 与档位选择正交)。
  D. 冲刺-滑行分段方案: 刀锋敏感 (k1 差 0.02 -> 1.4s/6.5° ↔ 3.5s/0°;
     增益 +8% 或 τ2 +15% 当天翻脸)。用户已否决分段控制, 本节仅存档为证据。

用法: python3 analysis/rung4_scan.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin import Twin

DT = 0.01
KQ_REAL = 0.2          # 90% 满幅实测 g(400)≈0.83 -> kq=0.2 (孪生 v3 的 1.1 为 u≤60 包络拟合值)


def make_plant(kq=KQ_REAL, gain=1.0, theta=None, tau2=None):
    tv = Twin()
    tv.kq_hot = kq
    tv.q_bp = tv.q_bp * gain
    if theta is not None:
        tv.theta = theta
    if tau2 is not None:
        tv.tau2 = tau2
    return tv


def sim_plant_step(tvp, u_of_t, y0, T, fw=0.0):
    """给 u(t) 出真实轨迹 (含纯滞后/元件惯性/散热/炉体偷热 fw·qf)"""
    N = int(T / DT)
    nd_p = max(1, int(round(tvp.theta / DT)))
    q0 = tvp.h1 * (y0 - tvp.c)
    u_ss = tvp.u_of_q(max(q0, 0.0) / max(1.0 - fw, 1e-6))
    uh = np.full(N + nd_p + 2, u_ss)
    y = y0
    qf = max(q0, 0.0) / max(1.0 - fw, 1e-6)
    yy = np.zeros(N)
    for i in range(N):
        uh[i + nd_p] = u_of_t(i * DT, y)
        ud = float(uh[i])
        for _ in range(2):
            h = DT / 2
            qf += h * (tvp.q_eff(ud, y) - qf) / max(tvp.tau2, 1e-4)
            y += h * (qf * (1.0 - fw) - tvp.h1 * (y - tvp.c))
        yy[i] = y
    return yy


def bang_bang_floor(tvp, mv, y0, svt, ov_budget=3.0):
    """开关最优: u=mv 烧到 ts 后 u=0, 之后到温交回 u_ss。扫 ts 求满足 峰值≤svt+ov 的最快到达"""
    u_hold = tvp.steady_u(svt)
    best = (np.inf, None, None, None)
    for ts in np.arange(0.3, 3.0, 0.01):
        def u_fn(t, y, ts=ts):
            if t < ts:
                return mv
            return u_hold if y >= svt - 0.5 else 0.0
        yy = sim_plant_step(tvp, u_fn, y0, T=6.0)
        peak = yy.max()
        cross = np.where(yy >= svt)[0]
        if peak > svt + ov_budget or not len(cross):
            continue
        reach = cross[0] * DT
        if reach < best[0]:
            best = (reach, ts, peak - svt, yy[int(ts / DT)])
    return best


def cl(tvc, tvp, wc, wo, wr, nd, tauf, mv_max, y0, svt, ckq=KQ_REAL, ctref=315.0,
       fw=0.0, T=14.0, pre=3.0, seed=1):
    """闭环: 控制器 (tvc 的表 + ckq 记账) vs 对象 (tvp 物理 + fw 偷热), 返回 (yy, uu) 阶跃后段"""
    n_pre = int(pre / DT)
    N = n_pre + int(T / DT)
    nd_p = max(1, int(round(tvp.theta / DT)))
    rng = np.random.default_rng(seed)
    v_hi = float(tvc.q_of(mv_max))
    q_need = tvp.h1 * (y0 - tvp.c)
    u_ss = tvp.u_of_q(max(q_need, 0.0) / max(1.0 - fw, 1e-6))
    v_ss = float(tvc.q_of(u_ss))
    y = y0
    qf = max(q_need, 0.0) / max(1.0 - fw, 1e-6)
    z1, z2 = y0, 0.0
    v1, v2 = y0, 0.0
    buf = [v_ss] * (nd + 1)
    vf = v_ss
    uh = np.full(N + nd_p + 2, u_ss)
    yy = np.zeros(N); uu = np.zeros(N)
    for i in range(N):
        svr = y0 if i < n_pre else svt
        ym = y + rng.normal(0, tvp.sig_n)
        v1n = v1 + DT * v2
        v2n = v2 + DT * (-2 * wr * v2 - wr * wr * (v1 - svr))
        v1, v2 = v1n, v2n
        vd = buf[nd]
        if tauf:
            vf += DT * (vd - vf) / tauf
            vd = vf
        err = ym - z1
        z1 = z1 + DT * (z2 + vd + 2 * wo * err)
        z2 = z2 + DT * (wo * wo * err)
        vc = min(max(wc * (v1 - z1) - z2, 0.0), v_hi)
        u = min(max(float(np.interp(vc, tvc.q_bp, tvc.u_bp)), 0.0), mv_max)
        va = float(tvc.q_of(u))
        if ckq > 0 and ym > ctref and u > 30.0:
            g = 1.0 - ckq * min(max((u - 30.0) / 30.0, 0.0), 1.0) * (ym - ctref) / 100.0
            va *= max(g, 0.4)
        buf = [va] + buf[:-1]
        uh[i + nd_p] = u
        ud = float(uh[i])
        for _ in range(2):
            h = DT / 2
            qf += h * (tvp.q_eff(ud, y) - qf) / max(tvp.tau2, 1e-4)
            y += h * (qf * (1.0 - fw) - tvp.h1 * (y - tvp.c))
        yy[i] = y; uu[i] = u
    return yy[n_pre:], uu[n_pre:]


def sprint_cl(tvc, tvp, wc, wo, wr, nd, tauf, mv_max, y0, svt, k1, k2=0.0, vlo=30.0,
              wc_catch=None, ckq=KQ_REAL, ctref=315.0, fw=0.0, T=14.0, pre=3.0, seed=1):
    """冲刺-滑行监督模式原型 (ST 可加 ~20 行实现):
       设定值阶跃且误差>40° 时进入冲刺 (u=mv_max, 每次阶跃只许一次 -- 锁存);
       断电条件 剩余距离 ≤ k1·实测速度 + k2; 滑行 (u=0) 至速度<vlo 或差2°进弯;
       交还后底层 ADRC 用 wc_catch 收尾 (小误差+低动量, 高 wc 不再危险)。
       断电点跟速度走 -> 增益漂移自适应 (漂移改变同温度处的速度, 规则自动移断电点)"""
    n_pre = int(pre / DT)
    N = n_pre + int(T / DT)
    nd_p = max(1, int(round(tvp.theta / DT)))
    rng = np.random.default_rng(seed)
    v_hi = float(tvc.q_of(mv_max))
    q_need = tvp.h1 * (y0 - tvp.c)
    u_ss = tvp.u_of_q(max(q_need, 0.0) / max(1.0 - fw, 1e-6))
    v_ss = float(tvc.q_of(u_ss))
    y = y0
    qf = max(q_need, 0.0) / max(1.0 - fw, 1e-6)
    z1, z2 = y0, 0.0
    v1, v2 = y0, 0.0
    buf = [v_ss] * (nd + 1)
    vf = v_ss
    uh = np.full(N + nd_p + 2, u_ss)
    yy = np.zeros(N); uu = np.zeros(N)
    mode = 0                     # 0=ADRC 1=冲刺 2=滑行
    armed, caught = True, False  # 锁存: 每次设定值阶跃只许冲一次
    v_f, ym_prev = 0.0, y0       # PV 斜率低通 (80ms)
    svr_prev = y0
    for i in range(N):
        svr = y0 if i < n_pre else svt
        if svr != svr_prev:
            armed, caught = True, False
        svr_prev = svr
        ym = y + rng.normal(0, tvp.sig_n)
        v_f += (max(ym - ym_prev, -5.0) / DT - v_f) * DT / 0.08
        ym_prev = ym
        v1n = v1 + DT * v2
        v2n = v2 + DT * (-2 * wr * v2 - wr * wr * (v1 - svr))
        v1, v2 = v1n, v2n
        vd = buf[nd]
        if tauf:
            vf += DT * (vd - vf) / tauf
            vd = vf
        err = ym - z1
        z1 = z1 + DT * (z2 + vd + 2 * wo * err)
        z2 = z2 + DT * (wo * wo * err)
        if mode == 0 and armed and svr - ym > 40.0:
            mode, armed = 1, False
        if mode == 1 and (svt - ym) <= k1 * max(v_f, 0.0) + k2:
            mode = 2
        if mode == 2 and (v_f < vlo or ym >= svt - 2.0):
            mode, caught = 0, True
            v1, v2 = svt, 0.0    # 参考就位, ADRC 只负责收尾与稳住
        if mode == 1:
            u = mv_max
        elif mode == 2:
            u = 0.0
        else:
            wc_eff = wc_catch if (caught and wc_catch) else wc
            vc = min(max(wc_eff * (v1 - z1) - z2, 0.0), v_hi)
            u = min(max(float(np.interp(vc, tvc.q_bp, tvc.u_bp)), 0.0), mv_max)
        va = float(tvc.q_of(u))
        if ckq > 0 and ym > ctref and u > 30.0:
            g = 1.0 - ckq * min(max((u - 30.0) / 30.0, 0.0), 1.0) * (ym - ctref) / 100.0
            va *= max(g, 0.4)
        buf = [va] + buf[:-1]
        uh[i + nd_p] = u
        ud = float(uh[i])
        for _ in range(2):
            h = DT / 2
            qf += h * (tvp.q_eff(ud, y) - qf) / max(tvp.tau2, 1e-4)
            y += h * (qf * (1.0 - fw) - tvp.h1 * (y - tvp.c))
        yy[i] = y; uu[i] = u
    return yy[n_pre:], uu[n_pre:]


def mets(yy, svt):
    """(到达s, 超调°, 到达后最大偏差°, ±2°带最终进入s)"""
    tt = np.arange(len(yy)) * DT
    cross = np.where(yy >= svt)[0]
    if not len(cross):
        return np.inf, 0.0, np.abs(yy - svt).max(), np.inf
    reach = tt[cross[0]]
    ov = max(yy.max() - svt, 0.0)
    maxdev = np.abs(yy[cross[0]:] - svt).max()
    bad = np.where(np.abs(yy - svt) > 2.0)[0]
    s2 = tt[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(yy) else 0.0
    return reach, ov, maxdev, s2


R3 = dict(wc=1.5, wo=10.0, wr=3.5, nd=12, tauf=0.24, mv_max=60.0)

if __name__ == '__main__':
    tvp = make_plant()

    print('=' * 74)
    print('A. 物理极限 (bang-bang 理想开关, 对象=实测修正孪生 kq=0.2, 峰值预算 +3°)')
    print('=' * 74)
    for y0, svt in [(100.0, 400.0), (200.0, 400.0)]:
        for mv in (60.0, 90.0, 100.0):
            reach, ts, ov, ycut = bang_bang_floor(tvp, mv, y0, svt)
            cut = '-' if ts is None else f'{ts:.2f}s@{ycut:.0f}°'
            print(f'  {y0:.0f}→{svt:.0f} u={mv:3.0f}%: 最快到达 {reach:5.2f}s (断电 {cut}, 峰值+{ov:.1f}°)')

    print()
    print('=' * 74)
    print('B. 第4档扫描 (对象同上, mv_max=90, nd=12, tauf=0.24, 记账 ckq=0.2, 100→400)')
    print('    合格线: 超调≤3 且 到达后最大偏差≤3; 按到达时间排序')
    print('=' * 74)
    tvc = Twin()  # 控制器用标称表
    rows = []
    for wc in (1.5, 2.0, 2.5, 3.0, 3.5):
        for wo in (8.0, 10.0, 12.0, 14.0):
            for wr in (3.5, 4.5, 5.5, 6.5, 8.0):
                yy, _ = cl(tvc, tvp, wc, wo, wr, 12, 0.24, 90.0, 100.0, 400.0)
                rows.append((wc, wo, wr) + mets(yy, 400.0))
    ok = [r for r in rows if r[4] <= 3.0 and r[5] <= 3.0 and np.isfinite(r[3])]
    ok.sort(key=lambda r: r[3])
    print(f'  合格 {len(ok)}/{len(rows)}; 前 8:')
    for wc, wo, wr, reach, ov, md, s2 in ok[:8]:
        print(f'   wc={wc:.1f} wo={wo:4.1f} wr={wr:.1f}: 到达 {reach:.2f}s 超调 {ov:.1f}° 后偏差 {md:.1f}° ±2带 {s2:.1f}s')
    near = sorted([r for r in rows if np.isfinite(r[3])], key=lambda r: r[3])
    print('  代价曲线 (pareto 前沿: 每档到达时间的最小超调 -- 统一律下"快"的价格表):')
    front = []
    best_ov = np.inf
    for r in near:
        w = max(r[4], r[5])
        if w < best_ov - 1e-9:
            best_ov = w
            front.append(r)
    for wc, wo, wr, reach, ov, md, s2 in front:
        print(f'   到达 {reach:5.2f}s 超调 {max(ov, md):5.1f}°  (wc={wc:.1f} wo={wo:4.1f} wr={wr:.1f})')

    if ok:
        wc, wo, wr = ok[0][:3]
        cand = dict(wc=wc, wo=wo, wr=wr, nd=12, tauf=0.24, mv_max=90.0)
        print(f'\n  候选第4档: wc={wc} wo={wo} wr={wr} nd=12 tauf=0.24 mv_max=90 ckq=0.2')
        yy, _ = cl(tvc, tvp, **cand, y0=200.0, svt=400.0)
        r, o, m, s = mets(yy, 400.0)
        print(f'  200→400: 到达 {r:.2f}s 超调 {o:.1f}° 后偏差 {m:.1f}°')
        yy, _ = cl(tvc, tvp, **cand, y0=100.0, svt=400.0, ckq=KQ_REAL)
        print(f'  mv_max=100 对比: ', end='')
        yy2, _ = cl(tvc, tvp, wc=wc, wo=wo, wr=wr, nd=12, tauf=0.24, mv_max=100.0,
                    y0=100.0, svt=400.0)
        r2, o2, m2, _ = mets(yy2, 400.0)
        print(f'到达 {r2:.2f}s 超调 {o2:.1f}° (增益有限)')

        print()
        print('=' * 74)
        print('C. 应力测试 (候选第4档, 100→400; 对照第3档同工况)')
        print('=' * 74)
        stress = [
            ('标称', {}, {}),
            ('增益×0.85', dict(gain=0.85), {}),
            ('增益×1.15', dict(gain=1.15), {}),
            ('θ 0.12→0.15', dict(theta=0.15), {}),
            ('τ2 ×1.3', dict(tau2=0.315), {}),
            ('缩水加重 kq=0.5', dict(kq=0.5), {}),
            ('无缩水 kq=0', dict(kq=0.0), {}),
            ('冷炉体偷热 fw=0.12', {}, dict(fw=0.12)),
            ('冷晨组合 (×0.9+fw0.12+kq0.35)', dict(gain=0.9, kq=0.35), dict(fw=0.12)),
        ]
        for name, pk, ck in stress:
            tvs = make_plant(**pk)
            yy4, _ = cl(tvc, tvs, **cand, **ck, y0=100.0, svt=400.0)
            r4, o4, m4, s4 = mets(yy4, 400.0)
            yy3, _ = cl(tvc, tvs, **R3, **ck, y0=100.0, svt=400.0, ckq=1.1)
            r3_, o3_, m3_, _ = mets(yy3, 400.0)
            flag = '' if (o4 <= 3.0 and m4 <= 3.0) else '  <-- 超预算'
            print(f'  {name:26s} 第4档 {r4:5.2f}s/{o4:4.1f}°/偏差{m4:4.1f}°'
                  f'  | 第3档 {r3_:5.2f}s/{o3_:4.1f}°{flag}')

    print()
    print('=' * 74)
    print('D. 冲刺-滑行监督模式原型 (u=90 冲 -> 按实测速度断电 -> 滑行 -> 交还 ADRC)')
    print('   底层 ADRC 第1档口味 (wc=1.5 wo=10 wr=3.5 nd=12 tauf=0.24), 收尾用 wc_catch')
    print('=' * 74)
    BASE = dict(wc=1.5, wo=10.0, wr=3.5, nd=12, tauf=0.24, mv_max=90.0)

    print('  D1. 刀锋展示: 纯滑行 (无接球), k1 细扫 -- 断电点差几度, 结果差一个世界')
    for k1 in (0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.40):
        yy, _ = sprint_cl(tvc, tvp, **BASE, y0=100.0, svt=400.0, k1=k1)
        r, o, m, _ = mets(yy, 400.0)
        rtxt = f'{r:5.2f}s' if np.isfinite(r) else '  不到 '
        print(f'   k1={k1:.2f}: 到达 {rtxt} 超调 {o:4.1f}° 后偏差 {m:4.1f}°')

    print('\n  D2. 表演模式 (k1 卡在刀锋快侧) 的日漂移敏感性:')
    k1_show = 0.30
    for name, pk in [('标称', {}), ('增益×0.92', dict(gain=0.92)), ('增益×1.08', dict(gain=1.08)),
                     ('τ2×1.15', dict(tau2=0.278)), ('冷炉偷热fw=0.08', {})]:
        tvs = make_plant(**pk)
        ck = dict(fw=0.08) if 'fw' in name else {}
        yy, _ = sprint_cl(tvc, tvs, **BASE, y0=100.0, svt=400.0, k1=k1_show, **ck)
        r, o, m, _ = mets(yy, 400.0)
        rtxt = f'{r:5.2f}s' if np.isfinite(r) else '  不到 '
        print(f'   {name:16s} k1={k1_show}: 到达 {rtxt} 超调 {o:4.1f}° 后偏差 {m:4.1f}°')

    print('\n  D3. 稳健版: 滑行故意留短 + wc_catch 接球收尾 (整定网格):')
    grid = []
    for k1 in (0.36, 0.40, 0.44):
        for wcc in (2.5, 3.0, 3.5):
            yy, _ = sprint_cl(tvc, tvp, **BASE, y0=100.0, svt=400.0, k1=k1, wc_catch=wcc, vlo=25.0)
            r, o, m, _ = mets(yy, 400.0)
            tvs = make_plant(tau2=0.315)
            yys, _ = sprint_cl(tvc, tvs, **BASE, y0=100.0, svt=400.0, k1=k1, wc_catch=wcc, vlo=25.0)
            _, o_t, _, _ = mets(yys, 400.0)
            tvg = make_plant(gain=0.85)
            yyg, _ = sprint_cl(tvc, tvg, **BASE, y0=100.0, svt=400.0, k1=k1, wc_catch=wcc, vlo=25.0)
            _, o_g, _, _ = mets(yyg, 400.0)
            grid.append((k1, wcc, r, o, m, max(o_t, o_g)))
            print(f'   k1={k1:.2f} wc_catch={wcc:.1f}: 到达 {r:5.2f}s 超调 {o:4.1f}°'
                  f' | τ2×1.3 超调 {o_t:4.1f}° | 增益×0.85 超调 {o_g:4.1f}°')

    good = [g for g in grid if g[3] <= 2.0 and g[4] <= 3.0 and g[5] <= 4.0 and np.isfinite(g[2])]
    good.sort(key=lambda g: g[2])
    if good:
        k1, wcc = good[0][:2]
        SPR = dict(k1=k1, wc_catch=wcc, vlo=25.0)
        print(f'\n  D4. 选定稳健冲刺档 k1={k1:.2f} wc_catch={wcc:.1f}; 全应力 (对照第3档同工况):')
        for name, pk, ck in stress:
            tvs = make_plant(**pk)
            yy, _ = sprint_cl(tvc, tvs, **BASE, y0=100.0, svt=400.0, **SPR, **ck)
            r, o, m, s2 = mets(yy, 400.0)
            yy3, _ = cl(tvc, tvs, **R3, **ck, y0=100.0, svt=400.0, ckq=1.1)
            r3_, o3_, _, _ = mets(yy3, 400.0)
            flag = '' if (o <= 3.0 and m <= 3.0) else '  <-- 超预算'
            print(f'   {name:26s} 冲刺档 {r:5.2f}s/{o:4.1f}°/偏差{m:4.1f}°'
                  f' | 第3档 {r3_:5.2f}s/{o3_:4.1f}°{flag}')
        for lab, y0_ in [('200→400', 200.0), ('300→400', 300.0)]:
            yy, _ = sprint_cl(tvc, tvp, **BASE, y0=y0_, svt=400.0, **SPR)
            r, o, m, _ = mets(yy, 400.0)
            print(f'   {lab} 标称: 到达 {r:.2f}s 超调 {o:.1f}° 后偏差 {m:.1f}°')
        for sd in (2, 3, 4):
            yy, _ = sprint_cl(tvc, tvp, **BASE, y0=100.0, svt=400.0, **SPR, seed=sd)
            r, o, m, _ = mets(yy, 400.0)
            print(f'   噪声种子{sd}: 到达 {r:.2f}s 超调 {o:.1f}° 后偏差 {m:.1f}°')
