import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

def load(fname, sheet):
    df = pd.read_excel(fname, sheet_name=sheet, engine='openpyxl')
    df.columns = ['t','pv1','pv2','sv','mv']
    df['t'] = pd.to_datetime(df['t'])
    df['sec'] = (df['t'] - df['t'].iloc[0]).dt.total_seconds()
    return df

def fit_fopdt(t, mv, pv, p0, bounds, mv_base=0.0):
    """Fit K, tau, L, T0 of dy/dt=(K*u(t-L)-y)/tau to pv(t), u=mv-mv_base."""
    t0 = t[0]
    u = mv - mv_base
    u_interp = interp1d(t, u, bounds_error=False, fill_value=(u[0], u[-1]))

    def simulate(params):
        K, tau, L, T0 = params
        def rhs(tt, y):
            td = tt - L
            ud = u_interp(td) if td >= t0 else u_interp(t0)
            return (K*ud - y[0])/tau
        sol = solve_ivp(rhs, (t[0], t[-1]), [0.0], t_eval=t, method='RK45', max_step=(t[-1]-t[0])/300)
        return sol.y[0] + T0

    def resid(params):
        return simulate(params) - pv

    res = least_squares(resid, p0, bounds=bounds, xtol=1e-10, ftol=1e-10)
    pred = simulate(res.x)
    ss_res = np.sum((pv-pred)**2)
    ss_tot = np.sum((pv-pv.mean())**2)
    r2 = 1 - ss_res/ss_tot
    return res.x, r2, pred

segments = []

# 1) MV50% sheet: clean isolated pulse, base ~100C
d = load('AIC9_DATA-20260702-185858_203502-数据分析.xlsx', 'MV50%')
seg = d[(d['sec']>=0)&(d['sec']<=90)]
segments.append(dict(name='50%脉冲 (base~100C)', t=seg['sec'].values, mv=seg['mv'].values,
                      pv=seg['pv1'].values, p0=[5.0, 15, 2, 100], bounds=([0,1,0,90],[20,80,20,110])))

# 2-5) escalating pulses in 'data' sheet, all based ~97-100C
d2 = load('AIC9_DATA-20260702-185858_203502-数据分析.xlsx', 'data')
for name, t0, t1, p0 in [
    ('20%脉冲 (base~97C)', 795, 880, [10, 15, 2, 97]),
    ('40%脉冲 (base~98C)', 973, 1060, [8, 15, 2, 98]),
    ('60%脉冲 (base~98C)', 1176, 1270, [6, 15, 2, 98]),
    ('70%脉冲-长 (base~100C)', 1295, 1600, [4, 15, 2, 100]),
]:
    seg = d2[(d2['sec']>=t0)&(d2['sec']<=t1)]
    segments.append(dict(name=name, t=seg['sec'].values, mv=seg['mv'].values, pv=seg['pv1'].values,
                          p0=p0, bounds=([0,1,0,p0[3]-10],[20,80,20,p0[3]+10])))

# 6) cold start from PID closed-loop data: MV saturated at 100% for ~14s = effectively open-loop
d3 = load('AIC9_DATA-20260702-203502_204340-PID闭环数据.xlsx', 'data')
seg = d3[(d3['sec']>=0)&(d3['sec']<=25)]
segments.append(dict(name='冷启动 0→100C (MV饱和于100%)', t=seg['sec'].values, mv=seg['mv'].values,
                      pv=seg['pv1'].values, p0=[1.0, 5, 10, 35], bounds=([0,0.5,0,30],[10,60,25,40])))

print(f'{"segment":<28}{"K(C/%MV)":>10}{"tau(s)":>8}{"L(s)":>7}{"T0(C)":>8}{"R2":>8}')
results = []
for s in segments:
    (K,tau,L,T0), r2, pred = fit_fopdt(s['t'], s['mv'], s['pv'], s['p0'], s['bounds'])
    print(f"{s['name']:<28}{K:>10.3f}{tau:>8.2f}{L:>7.2f}{T0:>8.1f}{r2:>8.4f}")
    results.append((s['name'], K, tau, L, T0, r2))

import pickle
with open('analysis/fopdt_results.pkl','wb') as f:
    pickle.dump((segments, results), f)
