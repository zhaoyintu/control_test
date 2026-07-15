"""
真机 ADRC 数据 (user_feedback/AIC9_DATA-20260713-*.csv) vs QtSim 仿真 对比
================================================================================
  1) 解析真机 CSV, 分段并计算每个阶跃的指标 (rise90 / 超调 / ±1C整定时间 / MV峰值)
  2) "控制器回放辨识": 把记录的 PV/SV/MV 喂给 Python 精确复刻的 C 版 ADRC,
     网格搜索 (b0, wo, nd, wr) + wc 闭式最小二乘, 反推真机上实际生效的参数
  3) 用 QtSim 的 FOIPDT 对象 (tau*ddy+dy = K*u(t-θ) - h*(y-T_amb)) 以
     "QtSim 默认参数" 与 "反推真机参数" 仿真同一工况, 与真机叠图 → 量化差距
  4) 用真机闭环数据重估对象 (dPV/dt 回归 + 分 MV 段局部 K) → 修正模型
  5) 两套参数方案的鲁棒网格搜索 (指标取模型摄动族内最差):
       方案A: 保持现结构 (纯标量 b0), 搜 (b0, wc, wo, wr)
       方案B: 恢复 g(u) 查表线性化 (adrc.st 中被注释的部分), v 空间 b0=1, 搜 (wc, wo, wr)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 微软雅黑(WSL 挂载的 Windows 字体)一个字体同时覆盖中英文; droid 只有 CJK 字形
for f in ['/mnt/c/Windows/Fonts/msyh.ttc',
          '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
          '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf']:
    try:
        font_manager.fontManager.addfont(f)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name()
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

CSV = 'user_feedback/AIC9_DATA-20260713-053919_060746.csv'
DT = 0.01

K_SIM, THETA_SIM, TAU_SIM = 3.774461, 0.0932, 0.0055
H_LOSS, T_AMB = 0.55, 35.0
QT_B0, QT_WC, QT_WO, QT_WR = 3.774461, 5.0, 30.0, 20.0

# ---------------------------------------------------------------- 1) 载入分段
df = pd.read_csv(CSV)
df.columns = ['time', 'PV1', 'PV2', 'SV', 'MV']
pv = df['PV1'].values.astype(float)
sv = df['SV'].values.astype(float)
mv = df['MV'].values.astype(float)
n = len(df)
t = np.arange(n) * DT

chg = np.where(np.diff(sv) != 0)[0] + 1
seg_bounds = np.concatenate([[0], chg, [n]])

def win_metrics(tt, yy, y0, target):
    amp = target - y0; s = np.sign(amp)
    idx = np.where(s * (yy - y0) >= s * 0.9 * abs(amp))[0]
    rise90 = tt[idx[0]] if len(idx) else np.nan
    peak = yy.max() if s > 0 else yy.min()
    ov = max((peak - target) * s, 0.0)
    bad = np.where(np.abs(yy - target) > 1.0)[0]
    settle = (tt[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(yy)
              else (np.nan if len(bad) else 0.0))
    return rise90, ov, settle

print('===== 真机各阶跃指标 (PV1) =====')
print(f"{'t0':>8s} {'y0':>6s}->{'SV':>5s} {'rise90':>8s} {'超调°C':>7s} {'settle±1':>9s} {'MV峰':>6s}")
steps = []
for k in range(len(chg)):
    i0 = chg[k]
    i1 = seg_bounds[np.searchsorted(seg_bounds, i0 + 1)]
    y0 = pv[i0 - 1]
    if abs(sv[i0] - y0) < 5:
        continue
    r90, ov, stl = win_metrics(t[i0:i1] - t[i0], pv[i0:i1], y0, sv[i0])
    steps.append((i0, i1, y0, sv[i0], dict(rise90=r90, ov_c=ov, settle=stl,
                                           mv_peak=mv[i0:i1].max())))
    print(f"{t[i0]:8.1f} {y0:6.1f}->{sv[i0]:5.0f} {r90:8.2f} {ov:7.2f} {stl:9.2f} "
          f"{mv[i0:i1].max():6.1f}")

up450 = [s for s in steps if s[3] == 450 and s[2] < 120]
print(f"\n100→450 共 {len(up450)} 次: rise90 均值 {np.mean([s[4]['rise90'] for s in up450]):.2f}s, "
      f"超调均值 {np.mean([s[4]['ov_c'] for s in up450]):.2f}°C, "
      f"settle±1 均值 {np.mean([s[4]['settle'] for s in up450]):.1f}s, "
      f"MV峰均值 {np.mean([s[4]['mv_peak'] for s in up450]):.1f}% (从未饱和→非功率极限)")

# ---------------------------------------------------------------- 2) 控制器回放辨识
def replay_eso(pv_seq, mv_seq, b0, wo, nd, dt=DT):
    m = len(pv_seq)
    z1 = np.empty(m); z2 = np.empty(m)
    z1c, z2c = pv_seq[0], 0.0
    buf = [0.0] * 40
    for i in range(m):
        vd = buf[nd]
        err = pv_seq[i] - z1c
        z1c = z1c + dt * (z2c + b0 * vd + 2.0 * wo * err)
        z2c = z2c + dt * (wo * wo * err)
        buf[1:] = buf[:-1]; buf[0] = mv_seq[i]
        z1[i] = z1c; z2[i] = z2c
    return z1, z2

def replay_td(sv_seq, wr, rv0, dt=DT):
    m = len(sv_seq)
    rv1 = np.empty(m)
    v1, v2 = rv0, 0.0
    for i in range(m):
        v1n = v1 + dt * v2
        v2n = v2 + dt * (-2.0 * wr * v2 - wr * wr * (v1 - sv_seq[i]))
        v1, v2 = v1n, v2n
        rv1[i] = v1
    return rv1

def fit_step(i0, i1, pre_s=60, fit_s=20,
             B0G=(3.0, 3.7745, 4.7, 5.9, 7.4, 9.2, 11.5),
             WOG=(1.5, 2, 3, 4.5, 6, 9, 15, 25),
             NDG=(5, 9, 15, 20, 25, 32),
             WRG=(2, 3, 4, 5, 6.5, 8, 10, 14)):
    j0 = max(0, i0 - int(pre_s / DT))
    pv_w, mv_w, sv_w = pv[j0:i1], mv[j0:i1], sv[j0:i1]
    lo = i0 - j0
    sl = slice(lo, min(lo + int(fit_s / DT), len(pv_w)))
    best = None
    for b0 in B0G:
        for wo in WOG:
            for nd in NDG:
                z1, z2 = replay_eso(pv_w, mv_w, b0, wo, nd)
                for wr in WRG:
                    rv1 = replay_td(sv_w, wr, rv0=sv_w[0])
                    e = (rv1 - z1)[sl]
                    tu = (mv_w * b0 + z2)[sl]
                    den = float(np.dot(e, e))
                    if den < 1e-9:
                        continue
                    wc = float(np.dot(e, tu)) / den
                    if wc <= 0:
                        continue
                    up = np.clip((wc * (rv1 - z1) - z2) / b0, 0, 100)[sl]
                    rmse = float(np.sqrt(np.mean((up - mv_w[sl]) ** 2)))
                    if best is None or rmse < best[0]:
                        best = (rmse, b0, wo, nd, wr, wc)
    return best

i0f, i1f = up450[-1][0], up450[-1][1]
RMSE_F, B0_F, WO_F, ND_F, WR_F, WC_F = fit_step(i0f, i1f)
print('\n===== 控制器回放辨识 =====')
print(f"最后一个 100→450: b0={B0_F} wo={WO_F} nd={ND_F} wr={WR_F} wc={WC_F:.3f} "
      f"(MV RMSE={RMSE_F:.3f}%)")
r_a = fit_step(up450[0][0], up450[0][1])
print(f"第一个 100→450:   b0={r_a[1]} wo={r_a[2]} nd={r_a[3]} wr={r_a[4]} wc={r_a[5]:.3f} "
      f"(RMSE={r_a[0]:.3f}%)")

# ---------------------------------------------------------------- 3) 对象重估
w = 101
pv_s = pd.Series(pv).rolling(w, center=True, min_periods=1).mean().values
dpv = np.gradient(pv_s, DT)
sel_base = (t > 30)
scan = []
for th_try in np.arange(0.0, 1.01, 0.05):
    nshift = int(round(th_try / DT))
    u_d = np.roll(mv, nshift); u_d[:nshift] = 0
    sel = sel_base & (np.arange(n) > nshift)
    A = np.column_stack([u_d[sel], -(pv_s[sel]), np.ones(sel.sum())])
    coef, *_ = np.linalg.lstsq(A, dpv[sel], rcond=None)
    pred = A @ coef
    r2 = 1 - np.sum((dpv[sel] - pred) ** 2) / np.sum((dpv[sel] - dpv[sel].mean()) ** 2)
    scan.append((r2, th_try, *coef))
scan.sort(reverse=True)
R2_B, TH_B, K_B, H_B, C_B = scan[0]
TAMB_B = C_B / H_B if H_B > 1e-6 else T_AMB
print('\n===== 真机闭环数据重估对象 =====')
print(f"线性 FOIPDT: θ≈{TH_B:.2f}s  K={K_B:.3f}  h={H_B:.4f}/s  T_amb_eff={TAMB_B:.0f}°C  R²={R2_B:.3f}")
print(f"QtSim 用的:  θ={THETA_SIM}s  K={K_SIM}  h={H_LOSS}/s  T_amb={T_AMB}°C")

# 分 MV 段局部增益 → 静态热输入曲线 q(u) (供方案B查表)
nsh = int(round(TH_B / DT))
u_d = np.roll(mv, nsh)
qdot = dpv + H_B * (pv_s - TAMB_B)          # ≈ q(u_delayed)
pts_u, pts_q = [], []
print('分 MV 段: 局部 K = q(u)/u')
for lo, hi in [(5, 15), (15, 30), (30, 45), (45, 65)]:
    m_sel = (u_d >= lo) & (u_d < hi) & sel_base
    if m_sel.sum() > 100:
        um = np.median(u_d[m_sel]); qm = np.median(qdot[m_sel])
        pts_u.append(um); pts_q.append(qm)
        print(f"  MV {lo:3d}-{hi:3d}%: u_med={um:5.1f}  q={qm:6.1f}°C/s  K_loc={qm/um:.2f}")

# q(u) = a*u^2 + b*u 最小二乘 (过原点: u=0 无加热)
Au = np.column_stack([np.array(pts_u) ** 2, np.array(pts_u)])
(a_q, b_q), *_ = np.linalg.lstsq(Au, np.array(pts_q), rcond=None)
print(f"拟合 q(u) = {a_q:.4f}·u² + {b_q:.3f}·u   "
      f"(K_loc: 10%→{a_q*10+b_q:.2f}, 60%→{a_q*60+b_q:.2f}, 100%→{a_q*100+b_q:.2f})")

def q_of(u):
    return a_q * u * u + b_q * u

def q_inv(v):
    if v <= 0:
        return 0.0
    disc = b_q * b_q + 4 * a_q * v
    return (-b_q + np.sqrt(max(disc, 0.0))) / (2 * a_q)

# ---------------------------------------------------------------- 4) 仿真器
def adrc_c(state, ym, svr, dt, first, b0, wc, wo, wr, nd, use_map=False):
    if first:
        state.update(z1=ym, z2=0.0, rv1=svr, rv2=0.0, buf=[0.0] * (nd + 1))
    v1, v2 = state['rv1'], state['rv2']
    v1n = v1 + dt * v2
    v2n = v2 + dt * (-2 * wr * v2 - wr * wr * (v1 - svr))
    state['rv1'], state['rv2'] = v1n, v2n
    z1, z2 = state['z1'], state['z2']
    ud = state['buf'][nd]
    err = ym - z1
    z1n = z1 + dt * (z2 + b0 * ud + 2 * wo * err)
    z2n = z2 + dt * (wo * wo * err)
    state['z1'], state['z2'] = z1n, z2n
    u0 = wc * (v1n - z1n)
    vcmd = (u0 - z2n) / b0
    if use_map:
        u_raw = q_inv(vcmd)                     # v→MV%, 查表反解
        us = min(max(u_raw, 0.0), 100.0)
        state['buf'] = [q_of(us)] + state['buf'][:-1]   # FIFO 存 v_applied
    else:
        us = min(max(vcmd, 0.0), 100.0)
        state['buf'] = [us] + state['buf'][:-1]
    return us

def sim_plant(params, y0, sv_t, T=20.0, nd_ctrl=None, use_map=False,
              plant='lin', Kp=None, thp=None, hp=None, tambp=None,
              q_scale=1.0, mv_max=100.0, sub=10, pre_s=3.0):
    """plant='lin': dy/dt=K·u(t-θ)-h(y-T)  |  plant='q': dy/dt=q_scale·q(u(t-θ))-h(y-T)
    与 QtSim 一致: 控制器先在 SV=y0 预热 pre_s 秒 (TD/ESO 收敛), 再切 SV → 阶跃;
    返回从阶跃时刻起算的 t/y/u"""
    b0, wc, wo, wr = params
    Kp = K_B if Kp is None else Kp
    thp = TH_B if thp is None else thp
    hp = H_B if hp is None else hp
    tambp = TAMB_B if tambp is None else tambp
    nd_c = max(1, int(round(TH_B / DT)) if nd_ctrl is None else nd_ctrl)
    n_pre = int(pre_s / DT)
    nsteps = n_pre + int(T / DT)
    nd_p = max(1, int(round(thp / DT)))
    y, v = y0, 0.0
    st = {}
    u_hist = np.zeros(nsteps + nd_p + 2)
    tt = np.zeros(nsteps); yy = np.zeros(nsteps); uu = np.zeros(nsteps)
    for i in range(nsteps):
        svr = y0 if i < n_pre else sv_t
        u = adrc_c(st, y, svr, DT, i == 0, b0, wc, wo, wr, nd_c, use_map)
        u = min(u, mv_max)
        u_hist[i + nd_p] = u
        ud = u_hist[i] if i >= nd_p else 0.0
        drive = (Kp * ud) if plant == 'lin' else (q_scale * q_of(ud))
        h_sub = DT / sub
        for _ in range(sub):
            v += h_sub * (-v + drive - hp * (y - tambp)) / TAU_SIM
            y += h_sub * v
        tt[i] = i * DT; yy[i] = y; uu[i] = u
    return tt[n_pre:] - pre_s, yy[n_pre:], uu[n_pre:]

# --- QtSim 模型 + 两组参数 vs 真机 (差距量化) ---
y0_ref = pv[i0f - 1]
real_tt = t[i0f:i1f] - t[i0f]
real_yy = pv[i0f:i1f]
real_uu = mv[i0f:i1f]
r_m = up450[-1][4]
T_SHOW = float(min(26, real_tt[-1]))

t_qt, y_qt, u_qt = sim_plant((QT_B0, QT_WC, QT_WO, QT_WR), y0_ref, 450, T=T_SHOW,
                             nd_ctrl=max(1, int(THETA_SIM / DT)),
                             Kp=K_SIM, thp=THETA_SIM, hp=H_LOSS, tambp=T_AMB)
t_ft, y_ft, u_ft = sim_plant((B0_F, WC_F, WO_F, WR_F), y0_ref, 450, T=T_SHOW,
                             nd_ctrl=ND_F, Kp=K_SIM, thp=THETA_SIM, hp=H_LOSS, tambp=T_AMB)
t_fb, y_fb, u_fb = sim_plant((B0_F, WC_F, WO_F, WR_F), y0_ref, 450, T=T_SHOW,
                             nd_ctrl=ND_F, plant='q')   # 反推参数+重估(非线性)模型
m_qt = win_metrics(t_qt, y_qt, y0_ref, 450)
m_ft = win_metrics(t_ft, y_ft, y0_ref, 450)
m_fb = win_metrics(t_fb, y_fb, y0_ref, 450)

print('\n===== 100→450 同工况对比 =====')
print(f"{'':34s} {'rise90':>8s} {'超调°C':>7s} {'settle±1':>9s} {'MV峰':>6s}")
print(f"{'真机 (用户整定)':34s} {r_m['rise90']:8.2f} {r_m['ov_c']:7.2f} {r_m['settle']:9.2f} {r_m['mv_peak']:6.1f}")
print(f"{'QtSim模型+反推参数':34s} {m_ft[0]:8.2f} {m_ft[1]:7.2f} {m_ft[2]:9.2f} {u_ft.max():6.1f}")
print(f"{'重估q(u)模型+反推参数':34s} {m_fb[0]:8.2f} {m_fb[1]:7.2f} {m_fb[2]:9.2f} {u_fb.max():6.1f}")
print(f"{'QtSim模型+QtSim默认参数':34s} {m_qt[0]:8.2f} {m_qt[1]:7.2f} {m_qt[2]:9.2f} {u_qt.max():6.1f}")

# ---------------------------------------------------------------- 5) 鲁棒搜索
# 物理约束: 超调 ≈ 末段接近速度 × 死区误差 + 跟踪误差。350°阶跃临界阻尼 TD 末段
# (距目标10°C) 接近速度 ≈ 6·wr [°C/s] → wr 直接决定可达的超调下限;
# wc/wo 负责把 PV 紧贴 rv1 (跟踪松了尾巴长、转角处过冲)。
#
# 摄动族取"证据支持"的不确定度, 不再拍脑袋 ±30%/±0.15s:
#  - 7 次 100→450 rise90 = 2.85~2.92s → 真机重复性极好, 增益不确定度主要是
#    q(u) 拟合残差 → q×{0.85, 1.0, 1.15}
#  - θ: 开环专项测得 0.080~0.107s (8 条 MV 幅值下高度一致, 同一套采集链路,
#    已含传感器滞后); 闭环回归的 0.15~0.35 平坦区是 1s 平滑窗造成的偏置,
#    不构成反证 → 取 θ∈{0.08, 0.12, 0.18}, 控制器补偿固定 0.12s
FAMILY = [(qs, thp) for qs in (0.85, 1.0, 1.15) for thp in (0.08, 0.12, 0.18)]
ND_NOM = 12
NOM_MEMBER = (1.0, 0.12)

def robust_eval(params, use_map, T=12.0):
    worst = [0.0, 0.0, 0.0]
    nom = None
    for qs, thp in FAMILY:
        ttx, yyx, _ = sim_plant(params, 100.0, 450.0, T=T, nd_ctrl=ND_NOM,
                                use_map=use_map, plant='q', q_scale=qs, thp=thp)
        r90, ov, stl = win_metrics(ttx, yyx, 100.0, 450.0)
        if np.isnan(stl) or np.isnan(r90):
            return None, None
        if (qs, thp) == NOM_MEMBER:
            nom = (r90, ov, stl)
        worst = [max(worst[0], r90), max(worst[1], ov), max(worst[2], stl)]
    return worst, nom

def search(use_map, b0_grid, label):
    rows = []
    for b0 in b0_grid:
        for wr in [1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
            for wcm in [3, 5, 8]:
                wc = min(wr * wcm, 16.0)
                for wom in [3, 5]:
                    wo = min(wc * wom, 40.0)   # wo·dt ≤ 0.4 保证离散稳定
                    w, nom = robust_eval((b0, wc, wo, wr), use_map)
                    if w:
                        rows.append((w[2], w[0], w[1], nom, b0, wc, wo, wr))
    rows.sort()
    print(f"\n--- {label} ---")
    print(f"{'预算':>6s} {'settle最差':>9s} {'rise90最差':>10s} {'超调最差':>8s} "
          f"{'名义r90/ov/stl':>18s} {'b0':>6s} {'wc':>5s} {'wo':>5s} {'wr':>4s}")
    picks = {}
    for budget in (1.0, 2.0, 5.0):
        feas = [r for r in rows if r[2] <= budget]
        if feas:
            s, r90, ov, nom, b0, wc, wo, wr = feas[0]
            picks[budget] = feas[0]
            print(f"{budget:5.1f}° {s:9.2f} {r90:10.2f} {ov:8.2f} "
                  f"{nom[0]:6.2f}/{nom[1]:4.2f}/{nom[2]:5.2f} {b0:6.2f} {wc:5.1f} {wo:5.1f} {wr:4.1f}")
        else:
            print(f"{budget:5.1f}°  (无可行解)")
    if rows:
        rows_ov = sorted(rows, key=lambda r: r[2])
        print('  超调最小的 3 个候选 (看可达下限):')
        for s, r90, ov, nom, b0, wc, wo, wr in rows_ov[:3]:
            print(f"       {s:9.2f} {r90:10.2f} {ov:8.2f} "
                  f"{nom[0]:6.2f}/{nom[1]:4.2f}/{nom[2]:5.2f} {b0:6.2f} {wc:5.1f} {wo:5.1f} {wr:4.1f}")
    return picks

print('\n===== 鲁棒定向搜索 (对象 = 重估 q(u) 模型族, 按超调预算分档) =====')
w_cur, nom_cur = robust_eval((B0_F, WC_F, WO_F, WR_F), use_map=False)
if w_cur:
    print(f"基准: 当前真机参数(反推 b0={B0_F} wc={WC_F:.2f} wo={WO_F} wr={WR_F}) 在同一族上:"
          f" 最差 r90={w_cur[0]:.2f} ov={w_cur[1]:.2f} stl={w_cur[2]:.2f}"
          f" | 名义 r90={nom_cur[0]:.2f} ov={nom_cur[1]:.2f} stl={nom_cur[2]:.2f}")
picksA = search(False, [2.2, 2.8, 3.4], '方案A: 现结构(标量 b0)')
picksB = search(True, [1.0], '方案B: 恢复 g(u) 查表 (v 空间, b0=1)')
outA2 = picksA.get(2.0) or picksA.get(5.0)
outB2 = picksB.get(2.0) or picksB.get(5.0)

# ---------------------------------------------------------------- 6) 画图
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
ax = axes[0][0]
ax.plot(real_tt, real_yy, 'k-', lw=1.6,
        label=f"真机 (rise90={r_m['rise90']:.1f}s 超调{r_m['ov_c']:.1f}°C settle={r_m['settle']:.0f}s)")
ax.plot(t_fb, y_fb, 'C0--', lw=1.4, label=f'重估模型+反推参数 (rise90={m_fb[0]:.1f}s)')
ax.plot(t_qt, y_qt, 'C3:', lw=1.4, label=f'QtSim模型+QtSim默认 (rise90={m_qt[0]:.1f}s)')
ax.axhline(450, color='gray', ls=':', lw=0.8)
ax.set_xlim(0, T_SHOW); ax.set_title('100→450°C: PV 真机 vs 仿真')
ax.set_xlabel('t [s]'); ax.set_ylabel('PV [°C]'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0][1]
ax.plot(real_tt, real_uu, 'k-', lw=1.2, label='真机 MV')
ax.plot(t_fb, u_fb, 'C0--', lw=1.2, label='重估模型+反推参数')
ax.plot(t_qt, u_qt, 'C3:', lw=1.2, label='QtSim模型+QtSim默认')
ax.set_xlim(0, T_SHOW); ax.set_title('同工况 MV')
ax.set_xlabel('t [s]'); ax.set_ylabel('MV [%]'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1][0]
j0 = max(0, i0f - int(60 / DT))
pv_w, mv_w, sv_w = pv[j0:i1f], mv[j0:i1f], sv[j0:i1f]
lo = i0f - j0
z1r, z2r = replay_eso(pv_w, mv_w, B0_F, WO_F, ND_F)
rv1r = replay_td(sv_w, WR_F, rv0=sv_w[0])
u_pred = np.clip((WC_F * (rv1r - z1r) - z2r) / B0_F, 0, 100)
tw = np.arange(len(pv_w)) * DT - lo * DT
ax.plot(tw, mv_w, 'k-', lw=1.4, label='真机 MV (记录)')
ax.plot(tw, u_pred, 'C1--', lw=1.2,
        label=f'回放重构 (b0={B0_F:.2f} wc={WC_F:.2f} wo={WO_F} wr={WR_F} nd={ND_F})')
ax.set_xlim(-2, 20); ax.set_title(f'控制器回放辨识 (RMSE={RMSE_F:.2f}%MV)')
ax.set_xlabel('t [s]'); ax.set_ylabel('MV [%]'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1][1]
ax.plot(real_tt, real_yy, 'k-', lw=1.6, label='真机 (当前参数)')
# 推荐档 (wr 阶梯, 前提: 第0步热态实测 θ 并按 nd=round(θ/Ts) 补偿):
# 第一档 wr=1.5 → rise≈2.8s ov≈0 settle≈5.5s; 目标档 wr=2.5 → 1.7/0/3.3
# 警示曲线: 若 θ 实为0.25s 而 nd 仍按0.15s 配 → 第一档也极限环 (θ实测不可跳过)
t1, y1, _ = sim_plant((3.4, 4.5, 13.5, 1.5), y0_ref, 450, T=T_SHOW,
                      nd_ctrl=15, plant='q', thp=0.15)
ax.plot(t1, y1, 'C2-', lw=1.5, label='第一档 b0=3.4 wc=4.5 wo=13.5 wr=1.5 (nd按实测θ)')
t2, y2, _ = sim_plant((3.4, 7.5, 22.5, 2.5), y0_ref, 450, T=T_SHOW,
                      nd_ctrl=15, plant='q', thp=0.12)
ax.plot(t2, y2, 'C4-', lw=1.5, label='目标档 wc=7.5 wo=22.5 wr=2.5 (确认θ≤0.15后)')
t1b, y1b, _ = sim_plant((3.4, 4.5, 13.5, 1.5), y0_ref, 450, T=T_SHOW,
                        nd_ctrl=15, plant='q', thp=0.25)
ax.plot(t1b, y1b, 'r--', lw=0.9, alpha=0.6, label='警示: 第一档但θ被低估0.1s → 极限环 (先实测θ!)')
ax.axhline(450, color='gray', ls=':', lw=0.8)
ax.set_xlim(0, T_SHOW); ax.set_title('真机现状 vs 推荐参数 (重估模型上仿真)')
ax.set_xlabel('t [s]'); ax.set_ylabel('PV [°C]'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/07_真机ADRC_vs_仿真对比.png', dpi=110)
print('\nsaved plots/07_真机ADRC_vs_仿真对比.png')
