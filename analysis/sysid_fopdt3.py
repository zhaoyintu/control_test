import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

def load(fname, sheet):
    df = pd.read_excel(fname, sheet_name=sheet, engine='openpyxl')
    df.columns = ['t','pv1','pv2','sv','mv']
    df['t'] = pd.to_datetime(df['t'])
    df['sec'] = (df['t'] - df['t'].iloc[0]).dt.total_seconds()
    df = df.drop_duplicates(subset='sec').sort_values('sec').reset_index(drop=True)
    return df

def simulate_fopdt(t, u, K, tau, L, y0):
    u_interp = interp1d(t, u, bounds_error=False, fill_value=(u[0], u[-1]))
    ud = u_interp(t - L)
    y = np.empty_like(t, dtype=float)
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        n_sub = max(1, int(np.ceil(dt / (tau/5 + 1e-9))))
        yi = y[i-1]
        for k in range(n_sub):
            yi = yi + (dt/n_sub) * (K*ud[i] - yi) / tau
        y[i] = yi
    return y

def fit_fopdt(t, mv, pv, mv_base, pv_base, p0, bounds):
    u = mv - mv_base
    y_meas = pv - pv_base
    def resid(params):
        K, tau, L = params
        return simulate_fopdt(t, u, K, tau, L, y0=0.0) - y_meas
    res = least_squares(resid, p0, bounds=bounds, xtol=1e-12, ftol=1e-12, max_nfev=8000)
    pred = simulate_fopdt(t, u, *res.x, y0=0.0) + pv_base
    ss_res = np.sum((pv-pred)**2); ss_tot = np.sum((pv-pv.mean())**2)
    r2 = 1 - ss_res/ss_tot
    return res.x, r2, pred

jobs = []

# clean isolated 50% pulse; true pre-pulse hold is t<1.3s
d = load('AIC9_DATA-20260702-185858_203502-数据分析.xlsx', 'MV50%')
seg = d[(d['sec']>=0)&(d['sec']<=90)].reset_index(drop=True)
base = seg[seg['sec']<1.3]
jobs.append(dict(name='50%脉冲(base~100C)', seg=seg,
                  mv_base=base['mv'].mean(), pv_base=base['pv1'].mean(),
                  p0=[5,12,1], bounds=([0.1,1,0],[30,60,15])))

# escalating staircase pulses: use short pre-window immediately before each, non-overlapping fit window
d2 = load('AIC9_DATA-20260702-185858_203502-数据分析.xlsx', 'data')
staircase = [
    ('20%脉冲(base~PV)', 804.66, 880),
    ('40%脉冲(base~PV)', 983.25, 1060),
    ('60%脉冲(base~PV)', 1186.20, 1260),
    ('80%脉冲(base~PV)', 1736.28, 1790),
]
for name, t_start, t_end in staircase:
    pre = d2[(d2['sec']>=t_start-3)&(d2['sec']<t_start)]
    seg = d2[(d2['sec']>=t_start-1)&(d2['sec']<=t_end)].reset_index(drop=True)
    jobs.append(dict(name=name, seg=seg, mv_base=pre['mv'].mean(), pv_base=pre['pv1'].mean(),
                      p0=[8,10,1], bounds=([0.1,1,0],[40,60,15])))

# cold start: ONLY the MV-saturated (open-loop) window, t=0 to 14s
d3 = load('AIC9_DATA-20260702-203502_204340-PID闭环数据.xlsx', 'data')
seg = d3[(d3['sec']>=0)&(d3['sec']<=13.9)].reset_index(drop=True)
pre = seg[seg['sec']<2]
jobs.append(dict(name='冷启动0→100C(仅MV饱和窗口)', seg=seg,
                  mv_base=pre['mv'].mean(), pv_base=pre['pv1'].mean(),
                  p0=[1,8,10], bounds=([0.05,1,0],[10,60,14])))

print(f'{"segment":<26}{"mv_base":>8}{"pv_base":>8}{"peakMV":>8}{"peakPV":>8}{"K(C/%MV)":>10}{"tau(s)":>8}{"L(s)":>7}{"R2":>8}')
results = []
for j in jobs:
    seg = j['seg']
    (K,tau,L), r2, pred = fit_fopdt(seg['sec'].values, seg['mv'].values, seg['pv1'].values,
                                     j['mv_base'], j['pv_base'], j['p0'], j['bounds'])
    print(f"{j['name']:<26}{j['mv_base']:>8.2f}{j['pv_base']:>8.1f}{seg['mv'].max():>8.1f}{seg['pv1'].max():>8.1f}{K:>10.3f}{tau:>8.2f}{L:>7.2f}{r2:>8.4f}")
    results.append(dict(name=j['name'], mv_base=j['mv_base'], pv_base=j['pv_base'], K=K, tau=tau, L=L, r2=r2, seg=seg, pred=pred))

import pickle
with open('analysis/fopdt_results3.pkl','wb') as f:
    pickle.dump(results, f)
