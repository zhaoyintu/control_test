"""
第二轮数据 B 版分析: 振铃周期 + 稳态抖动量化 -> (θ,τ2) 反演 -> 带噪声约束的参数搜索
诊断修正: wc=8/wc=2 不是发散极限环, 而是 (a) 阶跃后 2~3 个衰减振铃 (相位裕度薄)
          (b) 稳态下 MV 持续抖动 (噪声经 ESO 放大, 增益 ∝ wo·wc/b0), PV 纹波 ±1~1.5°
目标: 快 + 无超调 + MV 安静 (纹波收进 ±1°C 带)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = font_manager.FontProperties(
        fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name()
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

CSV = 'user_feedback/AIC9_DATA-20260713-211745_222036.csv'
DT = 0.01                      # 控制周期 (回放判定: 10ms; 本次日志间隔~15ms 是记录掉拍)
ND_MACH = 23
a_q, b_q = 0.0502, 0.919
H_B, TAMB = 0.1353, 127.0

def q_scalar(u):
    if u <= 65.0:
        return a_q*u*u + b_q*u
    q65 = a_q*65*65 + b_q*65
    return q65 + (327.0 - q65)*(u - 65.0)/35.0

# ---------------------------------------------------------------- 载入
df = pd.read_csv(CSV)
df.columns = ['time', 'PV1', 'PV2', 'SV', 'MV'][: len(df.columns)]
traw = (pd.to_datetime(df['time']) - pd.to_datetime(df['time'].iloc[0])).dt.total_seconds().values
tg = np.arange(0, traw[-1], DT)
pv = np.interp(tg, traw, df['PV1'].values.astype(float))
mv = np.interp(tg, traw, df['MV'].values.astype(float))
idx_prev = np.clip(np.searchsorted(traw, tg, side='right') - 1, 0, len(traw)-1)
sv = df['SV'].values.astype(float)[idx_prev]

sv_raw = df['SV'].values.astype(float)
chg_raw = np.where(np.diff(sv_raw) != 0)[0] + 1
steps = []
for k in chg_raw:
    if sv_raw[k] > sv_raw[k-1] and sv_raw[k] >= 200:
        nxt = [c for c in chg_raw if c > k]
        k1 = nxt[0] if nxt else len(sv_raw) - 1
        steps.append((int(np.searchsorted(tg, traw[k])),
                      int(np.searchsorted(tg, traw[k1])), sv_raw[k-1], sv_raw[k]))
labels = ['S1 旧参 100→200', 'S2 旧参 100→400', 'S3 wc=8', 'S4 wc=2', 'S5 wc=0.9', 'S6 PID']

def smooth(x, w):
    w = max(1, int(w))
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values

def ring_period(u_seg, dt=DT, lo_s=1.0, hi_s=8.0):
    """阶跃后 MV 振铃周期: 去掉慢趋势后自相关首峰"""
    a = int(lo_s/dt); b = min(int(hi_s/dt), len(u_seg))
    if b - a < int(2.0/dt):
        return np.nan
    x = u_seg[a:b] - smooth(u_seg[a:b], 2.0/dt)
    x = smooth(x, 0.08/dt)             # 去高频噪声
    if np.std(x) < 0.3:
        return np.nan
    m = len(x)
    ac = np.correlate(x, x, 'full')[m-1:]
    ac /= (ac[0] + 1e-12)
    i0 = int(0.4/dt)
    for i in range(i0, len(ac)-1):
        if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0.2:
            return i*dt
    return np.nan

def metrics(yy, y0, tgt, dt=DT):
    tt = np.arange(len(yy))*dt
    amp = tgt - y0
    idx = np.where(yy - y0 >= 0.9*amp)[0]
    r90 = tt[idx[0]] if len(idx) else np.nan
    ov = max(yy.max() - tgt, 0.0)
    bad = np.where(np.abs(yy - tgt) > 1.0)[0]
    stl = tt[bad[-1]+1] if len(bad) and bad[-1]+1 < len(yy) else np.nan
    return r90, ov, stl

print('===== 各段量化 (振铃 + 稳态抖动) =====')
print(f"{'段':16s} {'rise90':>7s} {'超调':>6s} {'settle':>7s} {'MV振铃T':>8s} "
      f"{'稳态MV std':>10s} {'稳态PV std':>10s}")
feats = {}
for k, (i0, i1, y0, tgt) in enumerate(steps):
    yy, uu = pv[i0:i1], mv[i0:i1]
    r90, ov, stl = metrics(yy, y0, tgt)
    T_ring = ring_period(uu)
    m0 = max(0, len(yy) - int(4.0/DT))          # 最后 4s 当稳态
    mv_std = float(np.std(uu[m0:]))
    pv_std = float(np.std(yy[m0:] - smooth(yy[m0:], 1.0/DT)))
    lbl = labels[k] if k < len(labels) else f'S{k+1}'
    feats[lbl.split()[0]] = dict(r90=r90, ov=ov, stl=stl, T=T_ring,
                                 mv_std=mv_std, pv_std=pv_std, i0=i0, i1=i1,
                                 y0=y0, tgt=tgt)
    print(f'{lbl:16s} {r90:6.2f}s {ov:5.1f}° {str(round(stl,2)) if not np.isnan(stl) else "  未整定":>6s}s '
          f'{str(round(T_ring,2)) if not np.isnan(T_ring) else "    -":>7s}s '
          f'{mv_std:9.2f}% {pv_std:9.3f}°')

# PV 噪声 (阶跃前静止段)
i0_s3 = feats['S3']['i0']
noise_seg = pv[i0_s3 - int(8/DT): i0_s3 - int(1/DT)]
SIG_N = float(np.std(noise_seg - smooth(noise_seg, 0.5/DT)))
print(f'\nPV 测量噪声 σ ≈ {SIG_N:.3f}°C (S3 前静止段)')

# ---------------------------------------------------------------- 模型 + 控制器仿真
def sim_full(b0, wc, wo, wr, nd, theta, tau2, qs, y0, svt,
             T=16.0, pre=6.0, noise=0.0, seed=1, mv_max=100.0):
    n_pre = int(pre/DT); N = n_pre + int(T/DT)
    nd_p = max(1, int(round(theta/DT)))
    rng = np.random.default_rng(seed)
    q_need = H_B*(y0 - TAMB)/max(qs, 1e-6)
    lo_, hi_ = 0.0, 100.0
    for _ in range(40):
        mid = (lo_ + hi_)/2
        if q_scalar(mid)*1.0 < q_need: lo_ = mid
        else: hi_ = mid
    u_ss = (lo_ + hi_)/2 if q_need > 0 else 0.0
    y = y0; qf = max(q_need, 0.0)
    z1, z2 = y0, 0.0; v1, v2 = y0, 0.0
    buf = [u_ss]*(nd + 1)
    uh = np.full(N + nd_p + 2, u_ss)
    yy = np.zeros(N); uu = np.zeros(N)
    for i in range(N):
        svr = y0 if i < n_pre else svt
        ym = y + (rng.normal(0, noise) if noise else 0.0)
        v1n = v1 + DT*v2
        v2n = v2 + DT*(-2*wr*v2 - wr*wr*(v1 - svr))
        v1, v2 = v1n, v2n
        vd = buf[nd]; err = ym - z1
        z1 = z1 + DT*(z2 + b0*vd + 2*wo*err)
        z2 = z2 + DT*(wo*wo*err)
        u = min(max((wc*(v1 - z1) - z2)/b0, 0.0), mv_max)
        buf = [u] + buf[:-1]
        uh[i + nd_p] = u
        ud = uh[i]
        for _ in range(5):
            h_ = DT/5
            qf += h_*(q_scalar(ud) - qf)/max(tau2, 1e-4)
            y += h_*(qs*qf - H_B*(y - TAMB))
        yy[i] = y; uu[i] = u
    return yy[n_pre:], uu[n_pre:]

# ---------------------------------------------------------------- (θ, τ2, qs) 反演
def model_err(theta, tau2, qs, detail=False):
    e, d = 0.0, {}
    # S2 旧参
    yy, uu = sim_full(10.0, 0.8, 40.0, 10.0, 20, theta, tau2, qs, 100, 400)
    r90, ov, stl = metrics(yy, 100, 400)
    d['S2'] = (r90, ov, ring_period(uu))
    e += abs(ov - feats['S2']['ov'])/2.0 + abs(r90 - feats['S2']['r90'])/1.0
    # S3 wc=8: 振铃周期
    yy, uu = sim_full(3.4, 8.0, 24.0, 3.0, ND_MACH, theta, tau2, qs, 100, 400)
    r90, ov, stl = metrics(yy, 100, 400)
    Tr = ring_period(uu)
    d['S3'] = (r90, ov, Tr)
    tgtT = feats['S3']['T']
    if np.isnan(Tr) or np.isnan(tgtT):
        e += 2.0 if (np.isnan(Tr) != np.isnan(tgtT)) else 0.0
    else:
        e += 2.0*abs(np.log(Tr/tgtT))
    e += abs(ov - feats['S3']['ov'])/2.0
    # S4 wc=2
    yy, uu = sim_full(3.4, 2.0, 24.0, 3.0, ND_MACH, theta, tau2, qs, 100, 400)
    r90, ov, stl = metrics(yy, 100, 400)
    Tr = ring_period(uu)
    d['S4'] = (r90, ov, Tr)
    tgtT = feats['S4']['T']
    if not (np.isnan(Tr) or np.isnan(tgtT)):
        e += 2.0*abs(np.log(Tr/tgtT))
    e += abs(ov - feats['S4']['ov'])/2.0
    # S5 wc=0.9
    yy, uu = sim_full(3.4, 0.9, 24.0, 3.0, ND_MACH, theta, tau2, qs, 100, 400)
    r90, ov, stl = metrics(yy, 100, 400)
    d['S5'] = (r90, ov, ring_period(uu))
    e += abs(ov - feats['S5']['ov'])/1.5 + abs(r90 - feats['S5']['r90'])/1.0
    return (e, d) if detail else (e, None)

print('\n===== (θ, τ2, q_scale) 反演: 匹配 S2/S3/S4/S5 =====')
res = []
for theta in [0.10, 0.15, 0.20, 0.28, 0.36]:
    for tau2 in [0.05, 0.2, 0.4, 0.7, 1.0, 1.5]:
        for qs in [0.75, 0.9, 1.05, 1.2]:
            e, _ = model_err(theta, tau2, qs)
            res.append((e, theta, tau2, qs))
res.sort()
for e, th, t2, q in res[:6]:
    print(f'  err={e:5.2f}  θ={th:.2f}  τ2={t2:.2f}  qs={q:.2f}')
E_F, TH_F, TAU2_F, QS_F = res[0]
_, dd = model_err(TH_F, TAU2_F, QS_F, detail=True)
print(f'\n选定 θ={TH_F} τ2={TAU2_F} qs={QS_F}, 模型 vs 实测:')
for s in ['S2', 'S3', 'S4', 'S5']:
    r90, ov, Tr = dd[s]
    f = feats[s]
    print(f"  {s}: 模型 r90={r90:.2f} ov={ov:.1f}° T={Tr if not np.isnan(Tr) else float('nan'):.2f} | "
          f"实测 r90={f['r90']:.2f} ov={f['ov']:.1f}° T={f['T'] if not np.isnan(f['T']) else float('nan'):.2f}")

# ---------------------------------------------------------------- 参数搜索 (转型+噪声双约束)
print('\n===== 搜索: 超调≤0.5° + MV稳态std≤1.2% + settle最短 (模型: 反演最优) =====')
rows = []
for wr in [2.0, 2.5, 3.0, 3.5, 4.5]:
    for wc in [1.2, 1.8, 2.5, 3.5]:
        for wo in [4, 6, 9, 13, 18]:
            for nd in [23, 28, 34]:
                yy, uu = sim_full(3.4, wc, wo, wr, nd, TH_F, TAU2_F, QS_F, 100, 400, T=18)
                r90, ov, stl = metrics(yy, 100, 400)
                if np.isnan(stl) or ov > 0.5:
                    continue
                # 噪声增益: 恒定 SV 段, 实测噪声注入
                yn, un = sim_full(3.4, wc, wo, wr, nd, TH_F, TAU2_F, QS_F, 400, 400,
                                  T=8, pre=4, noise=SIG_N)
                mv_std = float(np.std(un[int(2/DT):]))
                pv_std = float(np.std(yn[int(2/DT):] - smooth(yn[int(2/DT):], 1.0/DT)))
                if mv_std > 1.2:
                    continue
                rows.append((stl, r90, ov, mv_std, pv_std, wc, wo, wr, nd, uu.max()))
rows.sort()
print(f"{'settle':>7s} {'rise90':>7s} {'超调':>6s} {'MVstd':>6s} {'PVstd':>6s} "
      f"{'wc':>4s} {'wo':>4s} {'wr':>4s} {'nd':>3s} {'MV峰':>6s}")
for r in rows[:10]:
    print(f'{r[0]:6.2f}s {r[1]:6.2f}s {r[2]:5.2f}° {r[3]:5.2f}% {r[4]:5.2f}° '
          f'{r[5]:4.1f} {r[6]:4.0f} {r[7]:4.1f} {r[8]:3d} {r[9]:5.1f}%')

# 参照: 实测 PID 与 wc=0.9 段
print(f"\n参照: S6 PID 实测   rise90={feats['S6']['r90']:.2f}s ov={feats['S6']['ov']:.1f}° "
      f"settle={feats['S6']['stl']:.2f}s MVstd={feats['S6']['mv_std']:.2f}%")
print(f"      S5 wc0.9 实测 rise90={feats['S5']['r90']:.2f}s ov={feats['S5']['ov']:.1f}° "
      f"settle={feats['S5']['stl']:.2f}s MVstd={feats['S5']['mv_std']:.2f}%")

# ---------------------------------------------------------------- 校准应力成员 + 候选终评
print('\n===== 校准应力成员: 找能复现 S4 实况(ov≈2.3° 振铃T≈1.6s)的模型摄动 =====')
best_s = None
for th in [0.18, 0.22, 0.26]:
    for t2 in [0.10, 0.18, 0.26]:
        for qs2 in [1.05, 1.15]:
            yy, uu = sim_full(3.4, 2.0, 24.0, 3.0, ND_MACH, th, t2, qs2, 100, 400, T=16)
            r90, ov, stl = metrics(yy, 100, 400)
            Tr = ring_period(uu)
            e = abs(ov - 2.3)/2 + (abs(np.log(Tr/1.62)) if not np.isnan(Tr) else 1.5)
            if best_s is None or e < best_s[0]:
                best_s = (e, th, t2, qs2, ov, Tr)
_, TH_S, TAU2_S, QS_S, ov_s, Tr_s = best_s
print(f'应力成员: θ={TH_S} τ2={TAU2_S} qs={QS_S} -> S4设置下 ov={ov_s:.1f}° 振铃T={Tr_s:.2f}s (实测 2.3°/1.62s)')

print('\n===== 候选终评 (名义 + 校准应力, 双模型都要干净) =====')
print(f"{'候选':26s} {'模型':4s} {'rise90':>7s} {'超调':>6s} {'settle':>7s} {'振铃T':>6s} {'MVstd':>6s}")
for tag, wc, wo, wr, nd in [('A 主推 wc2 wo9 wr3', 2.0, 9.0, 3.0, 23),
                            ('A+ 过补偿 nd28', 2.0, 9.0, 3.0, 28),
                            ('B 进取 wc2.5 wo13 wr3.5', 2.5, 13.0, 3.5, 23),
                            ('B+ 过补偿 nd28', 2.5, 13.0, 3.5, 28)]:
    for mtag, (th, t2, qs2) in [('名义', (TH_F, TAU2_F, QS_F)), ('应力', (TH_S, TAU2_S, QS_S))]:
        yy, uu = sim_full(3.4, wc, wo, wr, nd, th, t2, qs2, 100, 400, T=18)
        r90, ov, stl = metrics(yy, 100, 400)
        Tr = ring_period(uu)
        yn, un = sim_full(3.4, wc, wo, wr, nd, th, t2, qs2, 400, 400, T=8, pre=4, noise=SIG_N)
        mv_std = float(np.std(un[int(2/DT):]))
        print(f'{tag:26s} {mtag:4s} {r90:6.2f}s {ov:5.2f}° '
              f'{str(round(stl,2)) if not np.isnan(stl) else "  未整定":>6s}s '
              f'{str(round(Tr,2)) if not np.isnan(Tr) else "   -":>5s}s {mv_std:5.2f}%')

# ---------------------------------------------------------------- 验证图
# 终选 A+ (名义+应力双干净): wc=2, wo=9, wr=3, nd=28
WC_R, WO_R, WR_R, ND_R = 2.0, 9.0, 3.0, 28
yy_, uu_ = sim_full(3.4, WC_R, WO_R, WR_R, ND_R, TH_F, TAU2_F, QS_F, 100, 400, T=18)
R_, O_, S_ = metrics(yy_, 100, 400)
if True:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # 左: 模型验证 S3
    f = feats['S3']
    tt = tg[f['i0']:f['i1']] - tg[f['i0']]
    axes[0].plot(tt, pv[f['i0']:f['i1']], 'k-', lw=1.2, label='实测 PV (wc=8)')
    ym, um = sim_full(3.4, 8.0, 24.0, 3.0, ND_MACH, TH_F, TAU2_F, QS_F, 100, 400, T=min(14, tt[-1]))
    axes[0].plot(np.arange(len(ym))*DT, ym, 'C0--', lw=1.2, label='反演模型')
    axes[0].set_title(f'模型验证: S3 wc=8 (θ={TH_F}, τ2={TAU2_F}, qs={QS_F})', fontsize=10)
    # 中: 模型验证 S5
    f = feats['S5']
    tt = tg[f['i0']:f['i1']] - tg[f['i0']]
    axes[1].plot(tt, pv[f['i0']:f['i1']], 'k-', lw=1.2, label='实测 PV (wc=0.9)')
    ym, um = sim_full(3.4, 0.9, 24.0, 3.0, ND_MACH, TH_F, TAU2_F, QS_F, 100, 400, T=min(14, tt[-1]))
    axes[1].plot(np.arange(len(ym))*DT, ym, 'C0--', lw=1.2, label='反演模型')
    axes[1].set_title('模型验证: S5 wc=0.9', fontsize=10)
    # 右: 推荐参数预测 vs 实测PID
    f = feats['S6']
    tt = tg[f['i0']:f['i1']] - tg[f['i0']]
    axes[2].plot(tt, pv[f['i0']:f['i1']], color='gray', lw=1.2, label=f"实测 PID (ov={f['ov']:.1f}°)")
    yr, ur = sim_full(3.4, WC_R, WO_R, WR_R, ND_R, TH_F, TAU2_F, QS_F, 100, 400, T=min(14, tt[-1]))
    axes[2].plot(np.arange(len(yr))*DT, yr, 'C2-', lw=1.5,
                 label=f'推荐 wc={WC_R} wo={WO_R:.0f} wr={WR_R} nd={ND_R}\n'
                       f'(rise={R_:.2f}s ov={O_:.2f}° settle={S_:.2f}s)')
    axes[2].set_title('推荐参数(模型预测) vs 实测 PID', fontsize=10)
    for ax in axes:
        ax.axhline(400, color='gray', ls=':', lw=0.7)
        ax.set_xlabel('t [s]'); ax.set_ylabel('PV [°C]')
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 14)
        ax.set_ylim(80, 430)
    plt.tight_layout()
    plt.savefig('plots/09_模型反演与推荐参数.png', dpi=110)
    print('\nsaved plots/09_模型反演与推荐参数.png')
