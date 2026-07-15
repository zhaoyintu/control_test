"""
第二轮真机数据分析: user_feedback/AIC9_DATA-20260713-211745_222036.csv
测试序列 (用户口述):
  S1: 100->200  旧参数 lrK=10 wc=0.8 wo=40 wr=10 nd=20
  S2: 100->400  旧参数 (同上)
  S3: 100->400  新参数 lrK=3.4 wo=24 wr=3 nd=23, wc=8    <- 震荡
  S4: 100->400  同上, wc=2
  S5: 100->400  同上, wc=0.9
  S6: 100->400  PID (机内原有 PID, 参数未知)
步骤:
  1) 时间戳重采样到均匀 10ms 网格
  2) 每段指标: rise90 / 超调 / settle±1 / 震荡周期与幅值 / MV 行为
  3) 用 S5 (参数全知) 回放, 判定控制周期 10ms vs 15ms
  4) 用 S3 震荡周期 + S4/S5 形态, 在 (θ, τ2, q_scale) 网格上反演对象模型
     模型: dy/dt 由 q(u(t-θ)) 经一阶惯性 τ2 平滑后驱动 - h(y-T_amb)
  5) 在反演出的模型上搜索 无超调-最快 参数组
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ['/mnt/c/Windows/Fonts/msyh.ttc']:
    try:
        font_manager.fontManager.addfont(f)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name()
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

CSV = 'user_feedback/AIC9_DATA-20260713-211745_222036.csv'
DT = 0.01
a_q, b_q = 0.0502, 0.919          # 上一轮闭环拟合的 q(u), 5~65% 有效
H_B, TAMB = 0.1353, 127.0

def q_of(u):
    return np.where(u <= 65.0, a_q*u*u + b_q*u,
                    a_q*65*65 + b_q*65 + (327.0 - (a_q*65*65 + b_q*65))*(u-65.0)/35.0)

def q_scalar(u):
    if u <= 65.0:
        return a_q*u*u + b_q*u
    q65 = a_q*65*65 + b_q*65
    return q65 + (327.0 - q65)*(u-65.0)/35.0

# ---------------------------------------------------------------- 载入+重采样
df = pd.read_csv(CSV)
df.columns = ['time', 'PV1', 'PV2', 'SV', 'MV'][: len(df.columns)]
traw = (pd.to_datetime(df['time']) - pd.to_datetime(df['time'].iloc[0])).dt.total_seconds().values
tg = np.arange(0, traw[-1], DT)
pv = np.interp(tg, traw, df['PV1'].values.astype(float))
mv = np.interp(tg, traw, df['MV'].values.astype(float))
# SV 是阶跃信号: 用"上一个值保持"重采样, 避免插值造成的假中间值
idx_prev = np.clip(np.searchsorted(traw, tg, side='right') - 1, 0, len(traw)-1)
sv = df['SV'].values.astype(float)[idx_prev]
n = len(tg)

# 测试段: 在原始数据上分段 (SV 上跳到 >=200), 再映射到均匀网格
sv_raw = df['SV'].values.astype(float)
chg_raw = np.where(np.diff(sv_raw) != 0)[0] + 1
steps = []
for k in chg_raw:
    if sv_raw[k] > sv_raw[k-1] and sv_raw[k] >= 200:
        # 段终点 = 下一次 SV 变化
        nxt = [c for c in chg_raw if c > k]
        k1 = nxt[0] if nxt else len(sv_raw)-1
        i0 = int(np.searchsorted(tg, traw[k]))
        i1 = int(np.searchsorted(tg, traw[k1]))
        steps.append((i0, i1, sv_raw[k-1], sv_raw[k]))
labels = ['S1 旧参 100→200', 'S2 旧参 100→400', 'S3 wc=8', 'S4 wc=2', 'S5 wc=0.9', 'S6 PID']

def dominant_period(x, dt=DT):
    x = x - x.mean()
    if np.std(x) < 0.05:
        return np.nan, 0.0
    m = len(x)
    ac = np.correlate(x, x, 'full')[m-1:]
    ac /= (ac[0] + 1e-12)
    i0 = int(0.15/dt)
    for i in range(i0, len(ac)-1):
        if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0.25:
            return i*dt, ac[i]
    return np.nan, 0.0

print('===== 各段指标 =====')
print(f"{'段':16s} {'rise90':>7s} {'超调':>7s} {'settle±1':>9s} {'尾段PV±':>8s} {'尾段周期':>8s} {'MV范围':>12s}")
seg_data = []
for k, (i0, i1, y0, tgt) in enumerate(steps):
    yy = pv[i0:i1]; uu = mv[i0:i1]; tt = tg[i0:i1] - tg[i0]
    amp = tgt - y0
    idx = np.where(yy - y0 >= 0.9*amp)[0]
    r90 = tt[idx[0]] if len(idx) else np.nan
    ov = max(yy.max() - tgt, 0.0)
    bad = np.where(np.abs(yy - tgt) > 1.0)[0]
    stl = tt[bad[-1]+1] if len(bad) and bad[-1]+1 < len(yy) else np.nan
    # 尾段(到达目标之后)的振荡
    tail0 = idx[0] if len(idx) else len(yy)//2
    tail = yy[tail0:]
    T_osc, quality = dominant_period(tail)
    osc_amp = np.ptp(tail[len(tail)//4:]) / 2 if len(tail) > 200 else np.nan
    lbl = labels[k] if k < len(labels) else f'S{k+1}'
    print(f'{lbl:16s} {r90:6.2f}s {ov:6.2f}° {str(round(stl,2)) if not np.isnan(stl) else "  未整定":>8s}s '
          f'{osc_amp:7.2f}° {str(round(T_osc,2)) if not np.isnan(T_osc) else "   -":>7s}s '
          f'{uu.min():4.1f}~{uu.max():5.1f}%')
    seg_data.append((lbl, i0, i1, y0, tgt, r90, ov, stl, T_osc, osc_amp))

# ---------------------------------------------------------------- 回放判定控制周期 (S5, 参数全知)
def replay_rmse(i0, i1, dt_c, b0, wc, wo, wr, nd, pre_s=20.0):
    j0 = max(0, i0 - int(pre_s/DT))
    # 重采样到 dt_c 网格
    tt = np.arange(tg[j0], tg[i1-1], dt_c)
    pvx = np.interp(tt, tg, pv); mvx = np.interp(tt, tg, mv); svx = np.interp(tt, tg, sv)
    z1, z2 = pvx[0], 0.0
    v1, v2 = svx[0], 0.0
    buf = [0.0]*40
    up = np.empty(len(tt))
    for i in range(len(tt)):
        v1n = v1 + dt_c*v2
        v2n = v2 + dt_c*(-2*wr*v2 - wr*wr*(v1 - svx[i]))
        v1, v2 = v1n, v2n
        vd = buf[nd]; err = pvx[i] - z1
        z1 = z1 + dt_c*(z2 + b0*vd + 2*wo*err)
        z2 = z2 + dt_c*(wo*wo*err)
        up[i] = min(max((wc*(v1 - z1) - z2)/b0, 0.0), 100.0)
        buf[1:] = buf[:-1]; buf[0] = mvx[i]
    lo = int((tg[i0]-tg[j0])/dt_c)
    sl = slice(lo, len(tt))
    return float(np.sqrt(np.mean((up[sl] - mvx[sl])**2)))

print('\n===== 控制周期判定 (S5, lrK=3.4 wc=0.9 wo=24 wr=3 nd=23) =====')
i0, i1 = seg_data[4][1], seg_data[4][2]
for dt_c in [0.010, 0.015, 0.020]:
    r = replay_rmse(i0, i1, dt_c, 3.4, 0.9, 24.0, 3.0, 23)
    print(f'  假设周期 {dt_c*1000:.0f}ms: 回放 MV RMSE = {r:.3f}%')
print('  (S3 wc=8 同样验证)')
i0, i1 = seg_data[2][1], seg_data[2][2]
for dt_c in [0.010, 0.015]:
    r = replay_rmse(i0, i1, dt_c, 3.4, 8.0, 24.0, 3.0, 23)
    print(f'  假设周期 {dt_c*1000:.0f}ms: 回放 MV RMSE = {r:.3f}%')

# ---------------------------------------------------------------- 对象模型反演
# 模型: dq_f/dt = (q(u(t-θ)) - q_f)/τ2 ;  dy/dt = qs*q_f - h*(y - T_amb)
def sim_adrc_on_plant(b0, wc, wo, wr, nd, dt_c, theta, tau2, qs,
                      y0, svt, T=25.0, pre=5.0):
    n_pre = int(pre/dt_c); N = n_pre + int(T/dt_c)
    nd_p = max(1, int(round(theta/dt_c)))
    y = y0; qf = 0.0
    z1, z2 = y0, 0.0; v1, v2 = y0, 0.0
    buf = [0.0]*(nd+1)
    uh = np.zeros(N + nd_p + 2)
    yy = np.zeros(N); uu = np.zeros(N)
    # 预热段: 求能托住 y0 的稳态 u (二分)
    q_need = H_B*(y0 - TAMB)/max(qs, 1e-6)
    lo_, hi_ = 0.0, 100.0
    for _ in range(40):
        mid = (lo_+hi_)/2
        if q_scalar(mid) < q_need: lo_ = mid
        else: hi_ = mid
    u_ss = (lo_+hi_)/2
    qf = q_need if q_need > 0 else 0.0
    uh[:] = u_ss
    z2 = qs*qf*0 - 0.0   # 让 ESO 自己收敛
    for i in range(N):
        svr = y0 if i < n_pre else svt
        v1n = v1 + dt_c*v2
        v2n = v2 + dt_c*(-2*wr*v2 - wr*wr*(v1 - svr))
        v1, v2 = v1n, v2n
        vd = buf[nd]; err = y - z1
        z1 = z1 + dt_c*(z2 + b0*vd + 2*wo*err)
        z2 = z2 + dt_c*(wo*wo*err)
        u = min(max((wc*(v1 - z1) - z2)/b0, 0.0), 100.0)
        buf = [u] + buf[:-1]
        uh[i + nd_p] = u
        ud = uh[i]
        sub = 5
        for _ in range(sub):
            h_ = dt_c/sub
            qf += h_*(q_scalar(ud) - qf)/max(tau2, 1e-4)
            y += h_*(qs*qf - H_B*(y - TAMB))
        yy[i] = y; uu[i] = u
    return yy[n_pre:], uu[n_pre:]

def seg_features(yy, y0, tgt, dt=DT):
    tt = np.arange(len(yy))*dt
    amp = tgt - y0
    idx = np.where(yy - y0 >= 0.9*amp)[0]
    r90 = tt[idx[0]] if len(idx) else np.nan
    ov = max(yy.max() - tgt, 0.0)
    tail0 = idx[0] if len(idx) else len(yy)//2
    T_osc, _ = dominant_period(yy[tail0:], dt)
    osc_amp = np.ptp(yy[tail0 + len(yy[tail0:])//4:])/2 if len(yy) - tail0 > 200 else 0.0
    return r90, ov, T_osc, osc_amp

# 实测特征 (S2 旧参, S3 wc8, S4 wc2, S5 wc0.9)
def tail_settle(yy, tgt, dt=DT):
    tt = np.arange(len(yy))*dt
    bad = np.where(np.abs(yy - tgt) > 1.0)[0]
    return tt[bad[-1]+1] if len(bad) and bad[-1]+1 < len(yy) else np.nan

f2 = seg_features(pv[seg_data[1][1]:seg_data[1][2]], 100, 400)
s2_stl = tail_settle(pv[seg_data[1][1]:seg_data[1][2]], 400)
f3 = seg_features(pv[seg_data[2][1]:seg_data[2][2]], 100, 400)
f4 = seg_features(pv[seg_data[3][1]:seg_data[3][2]], 100, 400)
f5 = seg_features(pv[seg_data[4][1]:seg_data[4][2]], 100, 400)
print('\n实测特征: S2(旧参) ov=%.1f° stl=%.1fs | S3(wc8) T=%.2fs amp=%.1f° | '
      'S4(wc2) T=%.2fs amp=%.1f° | S5(wc0.9) ov=%.1f° r90=%.2fs'
      % (f2[1], s2_stl, f3[2], f3[3], f4[2], f4[3], f5[1], f5[0]))

DT_C = 0.015   # 回放判定: 控制周期 15ms
ND_MACH = 23   # 机器上设的 nDeadSteps

def model_err(theta, tau2, qs):
    """同时匹配 4 段: S2(旧参) S3(wc8,振荡) S4(wc2,振荡) S5(wc0.9,稳)"""
    e = 0.0
    y2, _ = sim_adrc_on_plant(10.0, 0.8, 40.0, 10.0, 20, DT_C, theta, tau2, qs, 100, 400, T=16)
    g2 = seg_features(y2, 100, 400, DT_C)
    e += abs(g2[1] - f2[1])/3.0                       # S2 超调
    y3, _ = sim_adrc_on_plant(3.4, 8.0, 24.0, 3.0, ND_MACH, DT_C, theta, tau2, qs, 100, 400, T=16)
    g3 = seg_features(y3, 100, 400, DT_C)
    if np.isnan(g3[2]):
        e += 4.0                                       # S3 必须振荡
    else:
        e += abs(np.log(g3[2]/f3[2]))*2 + abs(g3[3]-f3[3])/max(f3[3], 0.5)
    y4, _ = sim_adrc_on_plant(3.4, 2.0, 24.0, 3.0, ND_MACH, DT_C, theta, tau2, qs, 100, 400, T=16)
    g4 = seg_features(y4, 100, 400, DT_C)
    if np.isnan(g4[2]):
        e += 3.0                                       # S4 也在振荡
    else:
        e += abs(np.log(g4[2]/f4[2]))*2 + abs(g4[3]-f4[3])/max(f4[3], 0.5)
    y5, _ = sim_adrc_on_plant(3.4, 0.9, 24.0, 3.0, ND_MACH, DT_C, theta, tau2, qs, 100, 400, T=16)
    g5 = seg_features(y5, 100, 400, DT_C)
    if not np.isnan(g5[2]) and g5[3] > 0.7:
        e += 3.0                                       # S5 不振荡
    e += abs(g5[1] - f5[1])/3.0 + (abs(g5[0]-f5[0]) if not np.isnan(g5[0]) and not np.isnan(f5[0]) else 1.0)/3.0
    return e, (g2, g3, g4, g5)

print('\n===== (θ, τ2, q_scale) 网格反演: 同时匹配 S2/S3/S4/S5 =====')
best = None
for theta in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    for tau2 in [0.05, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]:
        for qs in [0.7, 0.85, 1.0, 1.15]:
            e, gs = model_err(theta, tau2, qs)
            if best is None or e < best[0]:
                best = (e, theta, tau2, qs, gs)
E_F, TH_F, TAU2_F, QS_F, (g2, g3, g4, g5) = best
print(f'最优: θ={TH_F}s  τ2={TAU2_F}s  q_scale={QS_F}   (拟合误差 {E_F:.2f})')
print(f'  S2 旧参:  模型 ov={g2[1]:.1f}°          | 实测 ov={f2[1]:.1f}°')
print(f'  S3 wc8:  模型 T={g3[2]:.2f}s ±{g3[3]:.1f}° | 实测 T={f3[2]:.2f}s ±{f3[3]:.1f}°')
print(f'  S4 wc2:  模型 T={g4[2]:.2f}s ±{g4[3]:.1f}° | 实测 T={f4[2]:.2f}s ±{f4[3]:.1f}°')
print(f'  S5 wc.9: 模型 ov={g5[1]:.1f}° r90={g5[0]:.2f}s | 实测 ov={f5[1]:.1f}° r90={f5[0]:.2f}s')
print('反演误差前 5 名 (看多解性):')
allr = []
for theta in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    for tau2 in [0.05, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]:
        for qs in [0.7, 0.85, 1.0, 1.15]:
            e, _ = model_err(theta, tau2, qs)
            allr.append((e, theta, tau2, qs))
allr.sort()
for e, th, t2, q in allr[:5]:
    print(f'  err={e:.2f}  θ={th}  τ2={t2}  qs={q}')

# ---------------------------------------------------------------- 在反演模型上找 无超调-最快
print('\n===== 反演模型上的参数搜索 (周期15ms, 目标: 超调≤0.5°, 无持续纹波, settle 最短) =====')
rows = []
for wr in [0.6, 0.9, 1.2, 1.8, 2.5]:
    for wcm in [2, 3, 5]:
        wc = wr*wcm
        for wom in [2, 3, 4]:
            wo = min(wc*wom, 26.0)
            for nd in sorted({int(round(TH_F/DT_C)), int(round((TH_F+0.5*TAU2_F)/DT_C)),
                              int(round((TH_F+TAU2_F)/DT_C))}):
                if nd > 39:
                    continue
                yy, uu = sim_adrc_on_plant(3.4, wc, wo, wr, nd, DT_C, TH_F, TAU2_F, QS_F, 100, 400, T=20)
                r90, ov, T_osc, amp = seg_features(yy, 100, 400, DT_C)
                stl = tail_settle(yy, 400, DT_C)
                if np.isnan(stl) or ov > 0.5 or (not np.isnan(T_osc) and amp > 0.5):
                    continue
                rows.append((stl, r90, ov, wc, wo, wr, nd, uu.max()))
rows.sort()
print(f"{'settle':>7s} {'rise90':>7s} {'超调':>6s} {'wc':>5s} {'wo':>5s} {'wr':>4s} {'nd':>3s} {'MV峰':>6s}")
for r in rows[:8]:
    print(f'{r[0]:6.2f}s {r[1]:6.2f}s {r[2]:5.2f}° {r[3]:5.1f} {r[4]:5.1f} {r[5]:4.1f} {r[6]:3d} {r[7]:5.1f}%')
if rows:
    S_, R_, O_, WC_R, WO_R, WR_R, ND_R, MVP = rows[0]
    print(f'\n推荐(待鲁棒复核): wc={WC_R} wo={WO_R} wr={WR_R} nd={ND_R} (15ms 拍) '
          f'-> 预期 rise90={R_:.2f}s ov={O_:.2f}° settle={S_:.2f}s')

# ---------------------------------------------------------------- 画图
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
for k, (lbl, i0, i1, y0, tgt, r90, ov, stl, T_osc, osc_amp) in enumerate(seg_data[:6]):
    ax = axes[k//3][k%3]
    tt = tg[i0:i1] - tg[i0]
    ax.plot(tt, pv[i0:i1], color='#c2410c', lw=1.3, label='PV')
    ax2 = ax.twinx()
    ax2.plot(tt, mv[i0:i1], color='#1d4ed8', lw=0.9, alpha=0.75, label='MV')
    ax2.set_ylim(-2, 105); ax2.set_ylabel('MV [%]', fontsize=8)
    ax.axhline(tgt, color='gray', ls=':', lw=0.8)
    info = f'rise90={r90:.2f}s 超调={ov:.1f}°'
    if not np.isnan(T_osc):
        info += f'\n尾段振荡 T≈{T_osc:.2f}s ±{osc_amp:.1f}°'
    elif not np.isnan(stl):
        info += f' settle={stl:.1f}s'
    ax.set_title(f'{lbl}   {info}', fontsize=10)
    ax.set_xlabel('t [s]'); ax.set_ylabel('PV [°C]')
    ax.set_xlim(0, min(tt[-1], 16))
plt.tight_layout()
plt.savefig('plots/08_第二轮真机测试_各参数对比.png', dpi=110)
print('\nsaved plots/08_第二轮真机测试_各参数对比.png')
