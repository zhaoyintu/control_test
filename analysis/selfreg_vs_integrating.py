"""
Adjudicate the model-structure question with real burst data:

  A: self-regulating FOPDT           tau*dy/dt = K*u(t-L) - y
  B: integrator + fast lag + delay   G(s) = K'*e^(-Ls) / [s*(1+tau2*s)]

Two hard facts decide it (write-up: analysis/sysid_reference.html section 05):
  1. Free decay. After the 60% burst is cut, PV falls at up to -27 C/s. An
     integrator predicts PV holds level once input returns to zero -- B fitted
     on the rise alone (R2=0.999) collapses to R2=-12.6 when extrapolated over
     the decay; A fits the whole window at R2=0.986.
  2. Balance power. Holding 100/300/400 C steady takes ~2.5 / 15-18 / 21-22 %MV
     (closed-loop file plateaus, printed below). An integrating process can
     hold a constant level only at ZERO input.

Equally important: on the 1-3 s timescale the closed loop actually lives on,
the two models agree exactly -- A's K/tau = 3.487 = B's fitted K' (C/s per
%MV) -- and SIMC yields identical Kc/Ti from either model (the fast-tuning
branch of the Ti min()). The distinction only matters for long holds and for
predicting where PV settles (a static map g(u) exists only if self-regulating).

Also documents the window-contamination fix: the original windows in
sysid_fopdt3.py start exactly when the PID engages, so ~20 s of closed-loop
2-3% MV hold sits inside the "open-loop" fit -- dragging K low and L high
(hence the old L=0.47-0.71 s vs the 0.19-0.21 s seen in closed-loop data).
This script trims to burst+decay only [1206.5, 1260] (burst 1207.1, cut
1208.3): the clean 60% fit gives K=41.3, tau=11.9 s, L=0.32 s -- consistent
with the L=0.2 s used in the final controller designs.

Run from the repo root. Writes plots/06_自衡vs积分_模型对比.png.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

F_OPEN   = 'AIC9_DATA-20260702-185858_203502-数据分析.xlsx'
F_CLOSED = 'AIC9_DATA-20260702-203502_204340-PID闭环数据.xlsx'

def load(fname, sheet):
    df = pd.read_excel(fname, sheet_name=sheet, engine='openpyxl')
    df.columns = ['t','pv1','pv2','sv','mv']
    df['t'] = pd.to_datetime(df['t'])
    df['sec'] = (df['t'] - df['t'].iloc[0]).dt.total_seconds()
    df = df.drop_duplicates(subset='sec').sort_values('sec').reset_index(drop=True)
    return df

def sim_fopdt(t, u, K, tau, L):
    ud = interp1d(t, u, bounds_error=False, fill_value=(u[0], u[-1]))(t - L)
    y = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = t[i]-t[i-1]; n = max(1, int(np.ceil(dt/(tau/5+1e-9))))
        yi = y[i-1]
        for _ in range(n):
            yi += (dt/n) * (K*ud[i] - yi) / tau
        y[i] = yi
    return y

def sim_intlag(t, u, Kp, tau2, L):
    # tau2*dv/dt = u(t-L) - v ;  dy/dt = Kp*v   (integrator driven through a fast lag)
    ud = interp1d(t, u, bounds_error=False, fill_value=(u[0], u[-1]))(t - L)
    y = np.zeros_like(t); v = 0.0
    for i in range(1, len(t)):
        dt = t[i]-t[i-1]; n = max(1, int(np.ceil(dt/(tau2/5+1e-9))))
        yi = y[i-1]
        for _ in range(n):
            v  += (dt/n) * (ud[i] - v) / tau2
            yi += (dt/n) * Kp * v
        y[i] = yi
    return y

def fit(simfn, t, u, ymeas, p0s, lo, hi):
    best = None
    for p0 in p0s:   # multistart: single-start least_squares can stall in a local minimum here
        r = least_squares(lambda p: simfn(t, u, *p) - ymeas, p0,
                          bounds=(lo, hi), xtol=1e-12, ftol=1e-12, max_nfev=8000)
        if best is None or r.cost < best.cost:
            best = r
    pred = simfn(t, u, *best.x)
    ss_res = np.sum((ymeas-pred)**2); ss_tot = np.sum((ymeas-ymeas.mean())**2)
    return best.x, 1-ss_res/ss_tot, pred

# ---- evidence 2: balance MV at each settled plateau of the closed-loop file ----
print('closed-loop plateaus (balance MV = mean MV over the settled last 30%):')
dc = load(F_CLOSED, 'data')
sv = dc['sv'].values
change = np.where(np.diff(sv) != 0)[0]
starts = np.r_[0, change+1]; ends = np.r_[change, len(sv)-1]
for s, e in zip(starts, ends):
    if e - s < 100: continue
    tail = dc.iloc[s + int(0.7*(e-s)) : e]
    err = (tail['pv1'] - sv[s]).abs().mean()
    if sv[s] > 0 and err < 0.5:
        print(f'  SV={sv[s]:5.0f}C  t={dc["sec"].iloc[s]:6.1f}..{dc["sec"].iloc[e]:6.1f}s'
              f'  balance MV={tail["mv"].mean():6.2f}%  (|PV-SV|={err:.2f}C)')

# ---- evidence 1 + timescale equivalence: fit both models on the 60% burst ----
d = load(F_OPEN, 'data')
W0, W1 = 1206.5, 1260.0          # burst at 1207.10, MV cut at 1208.32
seg = d[(d['sec']>=W0)&(d['sec']<=W1)].reset_index(drop=True)
t = seg['sec'].values - W0
u = seg['mv'].values              # absolute MV (balance ~2.5% at 100C is worth ~1C; ignored)
y = seg['pv1'].values - 100.0     # PID had PV pinned at 100.00C before the burst

pA, r2A, prA = fit(sim_fopdt, t, u, y,
                   [[30,13,0.3],[8,10,1.0],[25,9,0.2],[15,20,0.5]],
                   [0.1,1,0], [60,60,5])

mrise = t <= 3.0                  # rise-only fit for B: where an integrating model lives
pB, r2Brise, _ = fit(sim_intlag, t[mrise], u[mrise], y[mrise],
                     [[2.5,0.3,0.1],[4,0.5,0.0],[1.5,0.1,0.3],[3,1.0,0.2]],
                     [0.01,0.03,0], [20,10,5])
prB = sim_intlag(t, u, *pB)
r2Bfull = 1 - np.sum((y-prB)**2)/np.sum((y-y.mean())**2)

print(f'\nA self-reg FOPDT (full-window fit): K={pA[0]:.2f} C/%MV  tau={pA[1]:.2f}s  L={pA[2]:.3f}s'
      f'  R2(full)={r2A:.4f}   K/tau={pA[0]/pA[1]:.3f} C/s/%MV')
print(f"B integrator+lag (rise-only fit):   K'={pB[0]:.3f} C/s/%MV  tau2={pB[1]:.3f}s  L={pB[2]:.3f}s"
      f'  R2(rise)={r2Brise:.4f}  R2(full, extrapolated)={r2Bfull:.4f}')

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
ax.plot(t, y, '.', ms=2.5, color='#999', label='measured (60% burst, then MV=0)')
ax.plot(t, prA, '-',  lw=1.8, color='#1c66b0',
        label=f'A self-reg FOPDT, full fit  R$^2$={r2A:.2f}')
ax.plot(t, prB, '--', lw=1.8, color='#c1670a',
        label=f'B integ.+lag, rise fit -> extrapolated  R$^2$={r2Bfull:.2f}')
ax.axhline(0, color='#ccc', lw=0.8)
ax.set_title('full window: 1.1 s burst + free decay')
ax.set_xlabel('t (s)'); ax.set_ylabel(u'PV − 100 (°C)')
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')
i30 = np.searchsorted(t, 30)
ax.annotate('B: power off -> integrator holds level',
            xy=(30, prB[i30]), xytext=(24, prB[i30]+45), fontsize=9, color='#c1670a',
            arrowprops=dict(arrowstyle='->', color='#c1670a'))
ax.annotate('measured: falls at up to 27 °C/s\n= heat loss = self-regulating',
            xy=(30, y[i30]), xytext=(33, y[i30]+80), fontsize=9, color='#555',
            arrowprops=dict(arrowstyle='->', color='#888'))

ax = axes[1]
m = t <= 4.0
ax.plot(t[m], y[m], 'o', ms=4, color='#999', mfc='none', label='measured')
ax.plot(t[m], prA[m], '-',  lw=1.8, color='#1c66b0', label=f'A: K={pA[0]:.1f}, τ={pA[1]:.1f}s, L={pA[2]:.2f}s')
ax.plot(t[m], prB[m], '--', lw=1.8, color='#c1670a', label=f"B: K'={pB[0]:.2f} °C/s/%MV, τ₂={pB[1]:.2f}s, L={pB[2]:.2f}s")
ax.axvspan(1207.10-W0, 1208.32-W0, color='#c1670a', alpha=0.08)
ax.text(1207.7-W0, -30, 'MV=60%', fontsize=8, color='#a55', ha='center')
ax.set_title('zoom on the rise: the ~1-3 s timescale the loop lives on')
ax.set_xlabel('t (s)'); ax.set_ylabel(u'PV − 100 (°C)')
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='lower right')
fig.tight_layout()
out = 'plots/06_自衡vs积分_模型对比.png'
fig.savefig(out, dpi=110)
print('plot ->', out)
