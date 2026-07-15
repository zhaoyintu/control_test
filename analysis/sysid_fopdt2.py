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
    """Discrete forward-Euler sim of dy/dt=(K*u(t-L)-y)/tau on irregular grid t."""
    u_interp = interp1d(t, u, bounds_error=False, fill_value=(u[0], u[-1]))
    ud = u_interp(t - L)
    y = np.empty_like(t, dtype=float)
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        # sub-step if dt is large relative to tau, for stability/accuracy
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
        y = simulate_fopdt(t, u, K, tau, L, y0=0.0)
        return y - y_meas
    res = least_squares(resid, p0, bounds=bounds, xtol=1e-12, ftol=1e-12, max_nfev=5000)
    pred = simulate_fopdt(t, u, *res.x, y0=0.0) + pv_base
    ss_res = np.sum((pv-pred)**2); ss_tot = np.sum((pv-pv.mean())**2)
    r2 = 1 - ss_res/ss_tot
    return res.x, r2, pred

segments = []
d = load('AIC9_DATA-20260702-185858_203502-数据分析.xlsx', 'MV50%')
seg = d[(d['sec']>=0)&(d['sec']<=90)]
mv_base = seg['mv'].iloc[:20].mean(); pv_base = seg['pv1'].iloc[:20].mean()
segments.append(dict(name='50%脉冲 (base~100C)', seg=seg, mv_base=mv_base, pv_base=pv_base,
                      p0=[5,15,2], bounds=([0.1,1,0],[30,60,20])))

d2 = load('AIC9_DATA-20260702-185858_203502-数据分析.xlsx', 'data')
for name, t0, t1 in [
    ('20%脉冲 (base~97C)', 795, 880),
    ('40%脉冲 (base~98C)', 973, 1060),
    ('60%脉冲 (base~98C)', 1176, 1270),
    ('70%脉冲-长 (base~100C)', 1295, 1600),
]:
    seg = d2[(d2['sec']>=t0)&(d2['sec']<=t1)].reset_index(drop=True)
    mv_base = seg['mv'].iloc[:20].mean(); pv_base = seg['pv1'].iloc[:20].mean()
    segments.append(dict(name=name, seg=seg, mv_base=mv_base, pv_base=pv_base,
                          p0=[5,12,2], bounds=([0.1,1,0],[30,60,20])))

d3 = load('AIC9_DATA-20260702-203502_204340-PID闭环数据.xlsx', 'data')
seg = d3[(d3['sec']>=0)&(d3['sec']<=25)].reset_index(drop=True)
mv_base = seg['mv'].iloc[:20].mean(); pv_base = seg['pv1'].iloc[:20].mean()
segments.append(dict(name='冷启动 0→100C (MV饱和)', seg=seg, mv_base=mv_base, pv_base=pv_base,
                      p0=[1,5,10], bounds=([0.05,0.5,0],[10,60,25])))

print(f'{"segment":<28}{"mv_base":>8}{"pv_base":>8}{"K(C/%MV)":>10}{"tau(s)":>8}{"L(s)":>7}{"R2":>8}')
results = []
for s in segments:
    seg = s['seg']
    (K,tau,L), r2, pred = fit_fopdt(seg['sec'].values, seg['mv'].values, seg['pv1'].values,
                                     s['mv_base'], s['pv_base'], s['p0'], s['bounds'])
    print(f"{s['name']:<28}{s['mv_base']:>8.2f}{s['pv_base']:>8.1f}{K:>10.3f}{tau:>8.2f}{L:>7.2f}{r2:>8.4f}")
    results.append((s['name'], s['mv_base'], s['pv_base'], K, tau, L, r2, seg, pred))

import pickle
with open('analysis/fopdt_results2.pkl','wb') as f:
    pickle.dump(results, f)
