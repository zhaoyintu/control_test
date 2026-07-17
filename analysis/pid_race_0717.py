#!/usr/bin/env python3
"""7-17 晚 PID 对照场判读 + "1.5s 方向" ADRC 整定 (wr=30 去参考平滑路线)

事实 (AIC9_DATA-20260717-212412_212746.csv, PID 参考控制器, MV 上限 90):
  100→400: 到达 1.48s / 超调 3.0° / ±2°带 1.86s;  200→440: 1.58s / 1.3°
  打法 = 满幅 0.9s → 急降 → 断电(356°C)滑行 47° → ~22% 接住。

同 MV 开环回放定位模型误差:
  冲刺段孪生贴合 (±5%, 两场符号相反 -> θ/τe/q 表无系统误差);
  但今晚保 400 需 MV≈22% (孪生 17.5%, 7-15 实测 18.3~19.6%) -> 当晚散热
  比模型大 ~35%, 被动刹车强 -> 滑行短 -> 到达地板比"标准炉况"低。
  => 7-17 早前"物理地板 1.55~1.61s"是标准炉况数字; 地板本身随炉况浮动。
  => PID 死断电点依赖炉况 == "其他温度/其他日子表现不好"的机理本体。

本脚本: A. 三种炉况对象 (今晚拖 / 标准 / 暖午滑) + 今晚地板复算
        B. wr=30 (参考≈阶跃) × wc × wo × 记账 ckq 扫描, 三炉况同评
           -- 结果全灭: 去掉 SV 平滑后环路是欠阻尼二阶 ζ=(1+kd)/(2√(wc·τ总)),
              纯加 wc 越加越荡 (wc=4 可荡 40°), wc=1.55 约 6.5~8.6°
        C. 用户当前卡 (wc=1.55 wo=10 wr=30 ckq=1.1) 的预测
        D. 解法 = [9] lrKd 速度阻尼 (v̂ = vd+z2, ESO 模型速度, 免微分):
           第4档 wc=4.5 wo=10 wr=30 nd=12 tauf=0.24 kd=0.75 mv=90
           三炉况 100→400 1.50~1.59s/≤2.8°, 200→440 1.45s/3.2°, ckq 不敏感;
           拨盘: 首超>4 => kd+0.05, 爬行 => kd-0.05, ≥0.85 悬崖;
           冷晨/漂移税照旧 ~10° (与档位正交, 热机规程管)
用法: python3 analysis/pid_race_0717.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin import Twin
from rung4_scan import cl, mets, bang_bang_floor, make_plant, DT, KQ_REAL


def plant_day(day):
    """炉况三态: 散热线用两个稳态锚点反推 (q(u_hold)=h1'(T-c'))
       tonight: 200@10%->q16.0, 400@22%->q50.4 => h1=0.172, c=107 (7-17 晚实测)
       standard: 孪生 v3 (h1=0.1353, c=127; 保400=17.5%)
       warm:   保400≈18% 的暖午 (h1=0.147, c=127)"""
    tv = Twin()
    tv.kq_hot = KQ_REAL
    if day == 'tonight':
        tv.h1, tv.c = 0.172, 107.0
    elif day == 'warm':
        tv.h1 = 0.147
    return tv


DAYS = [('今晚(拖 22%@400)', 'tonight'), ('标准(17.5%@400)', 'standard'), ('暖午(18%, 散热小)', 'warm')]

if __name__ == '__main__':
    tvc = Twin()

    print('=' * 78)
    print('A. 物理地板随炉况浮动 (bang-bang 理想开关, u=90, 峰值预算+3°)')
    print('=' * 78)
    for nm, d in DAYS:
        tvp = plant_day(d)
        r4, ts4, o4, yc4 = bang_bang_floor(tvp, 90.0, 100.0, 400.0)
        print(f'  {nm:20s} 100→400 地板 {r4:.2f}s (断电 {yc4:.0f}°C)   [PID 实测今晚 1.48s]')

    print()
    print('=' * 78)
    print('B. wr=30 路线扫描: 每组参数在三炉况下的 (到达s/首超°) -- 找日态稳的组合')
    print('    nd=12 tauf=0.24 mv_max=90, 100→400')
    print('=' * 78)
    combos = []
    for wc in (2.0, 3.0, 4.0, 5.0, 6.0):
        for wo in (8.0, 10.0, 14.0):
            for ckq in (0.25, 0.6, 1.1):
                res = []
                for _, d in DAYS:
                    tvp = plant_day(d)
                    yy, _ = cl(tvc, tvp, wc=wc, wo=wo, wr=30.0, nd=12, tauf=0.24,
                               mv_max=90.0, y0=100.0, svt=400.0, ckq=ckq)
                    r, o, m, _ = mets(yy, 400.0)
                    res.append((r, max(o, m)))
                worst_ov = max(x[1] for x in res)
                worst_r = max(x[0] for x in res)
                combos.append((wc, wo, ckq, res, worst_r, worst_ov))
    ok = [c for c in combos if c[5] <= 3.0 and np.isfinite(c[4])]
    ok.sort(key=lambda c: c[4])
    print('  三炉况首超都 ≤3° 的组合 (按最慢炉况的到达排序, 前 10):')
    for wc, wo, ckq, res, wr_, wo_ in ok[:10]:
        s = ' | '.join(f'{r:.2f}s/{o:.1f}°' for r, o in res)
        print(f'   wc={wc:.1f} wo={wo:4.1f} ckq={ckq:.2f}:  {s}')
    print('\n  放宽到最坏首超 ≤4.5° (追速度的代价档, 前 8):')
    ok2 = [c for c in combos if 3.0 < c[5] <= 4.5 and np.isfinite(c[4])]
    ok2.sort(key=lambda c: c[4])
    for wc, wo, ckq, res, wr_, wo_ in ok2[:8]:
        s = ' | '.join(f'{r:.2f}s/{o:.1f}°' for r, o in res)
        print(f'   wc={wc:.1f} wo={wo:4.1f} ckq={ckq:.2f}:  {s}')

    print()
    print('=' * 78)
    print('C. 用户当前卡预测 (wc=1.55 wo=10 wr=30 ckq=1.1, mv=90) + 200→440 附测')
    print('=' * 78)
    for nm, d in DAYS:
        tvp = plant_day(d)
        yy, _ = cl(tvc, tvp, wc=1.55, wo=10.0, wr=30.0, nd=12, tauf=0.24,
                   mv_max=90.0, y0=100.0, svt=400.0, ckq=1.1)
        r, o, m, s2 = mets(yy, 400.0)
        yy2, _ = cl(tvc, tvp, wc=1.55, wo=10.0, wr=30.0, nd=12, tauf=0.24,
                    mv_max=90.0, y0=200.0, svt=440.0, ckq=1.1)
        r2, o2, m2, _ = mets(yy2, 440.0)
        print(f'  {nm:20s} 100→400: {r:.2f}s/{max(o,m):.1f}°  200→440: {r2:.2f}s/{max(o2,m2):.1f}°')

    print()
    print('=' * 78)
    print('D. 第4档 (lrKd 速度阻尼) 验证: wc=4.5 wo=10 wr=30 nd=12 tauf=0.24 kd=0.75 mv=90')
    print('=' * 78)
    CARD = dict(wc=4.5, wo=10.0, wr=30.0, nd=12, tauf=0.24, mv_max=90.0, kd=0.75)
    print('  kd 拨盘 (三炉况最坏, 100→400):')
    for kd in (0.65, 0.70, 0.75, 0.80, 0.85):
        res = []
        for _, d in DAYS:
            yy, _ = cl(tvc, plant_day(d), **dict(CARD, kd=kd), y0=100.0, svt=400.0, ckq=1.1)
            r, o, m, _ = mets(yy, 400.0)
            res.append((r, max(o, m)))
        w_r = max(x[0] for x in res); w_o = max(x[1] for x in res)
        rs = f'{w_r:.2f}s' if np.isfinite(w_r) else ' 爬行 '
        print(f'   kd={kd:.2f}: 最坏 {rs}/{w_o:.1f}°')
    print('  各工况 (kd=0.75, 三炉况最坏):')
    for lab, y0_, svt_ in [('100→400', 100., 400.), ('200→400', 200., 400.),
                           ('200→440', 200., 440.), ('100→440', 100., 440.),
                           ('350→400', 350., 400.)]:
        res = []
        for _, d in DAYS:
            yy, _ = cl(tvc, plant_day(d), **CARD, y0=y0_, svt=svt_, ckq=1.1)
            r, o, m, s2 = mets(yy, svt_)
            res.append((r, max(o, m), s2))
        print(f'   {lab}: 到达 {max(x[0] for x in res):.2f}s / 首超 {max(x[1] for x in res):.1f}°'
              f' / ±2°带 {max(x[2] for x in res):.1f}s')
    print('  应力 (标准炉况基底):')
    for name, pk, ck in [('冷炉偷热 fw=0.12', {}, dict(fw=0.12)),
                         ('增益×0.85', dict(gain=0.85), {}),
                         ('增益×1.15', dict(gain=1.15), {}),
                         ('τ2×1.15', dict(tau2=0.278), {})]:
        tvs = make_plant(**pk)
        yy, _ = cl(tvc, tvs, **CARD, y0=100.0, svt=400.0, ckq=1.1, **ck)
        r, o, m, _ = mets(yy, 400.0)
        rs = f'{r:.2f}s' if np.isfinite(r) else ' 爬行 '
        print(f'   {name:16s} {rs}/{max(o, m):.1f}°')
