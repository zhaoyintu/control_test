import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, differential_evolution

def load(fname, sheet):
    df = pd.read_excel(fname, sheet_name=sheet, engine='openpyxl')
    df.columns = ['t','pv1','pv2','sv','mv']
    df['t'] = pd.to_datetime(df['t'])
    df['sec'] = (df['t'] - df['t'].iloc[0]).dt.total_seconds()
    return df.drop_duplicates(subset='sec').sort_values('sec').reset_index(drop=True)

def simulate_sopdt(t, u, K, tau1, tau2, L, y0=0.0):
    u_interp = interp1d(t, u, bounds_error=False, fill_value=(u[0], u[-1]))
    ud = u_interp(t - L)
    x1 = 0.0; y = np.empty_like(t, dtype=float); y[0] = y0
    yi = y0
    for i in range(1, len(t)):
        dt = t[i]-t[i-1]
        n_sub = max(1, int(np.ceil(dt / (min(tau1,tau2)/5 + 1e-9))))
        for k in range(n_sub):
            h = dt/n_sub
            x1 = x1 + h*(K*ud[i] - x1)/tau1
            yi = yi + h*(x1 - yi)/tau2
        y[i] = yi
    return y

def simulate_fopdt(t, u, K, tau, L, y0=0.0):
    u_interp = interp1d(t, u, bounds_error=False, fill_value=(u[0], u[-1]))
    ud = u_interp(t - L)
    y = np.empty_like(t, dtype=float); y[0]=y0
    for i in range(1, len(t)):
        dt = t[i]-t[i-1]
        n_sub = max(1, int(np.ceil(dt/(tau/5+1e-9))))
        yi = y[i-1]
        for k in range(n_sub):
            yi = yi + (dt/n_sub)*(K*ud[i]-yi)/tau
        y[i] = yi
    return y

def r2(pv, pred):
    ss_res = np.sum((pv-pred)**2); ss_tot = np.sum((pv-pv.mean())**2)
    return 1 - ss_res/ss_tot

def fit_model(sim_fn, t, u, pv_offset, pv, p0, bounds, nparam):
    def resid(params):
        y = sim_fn(t, u, *params)
        return y - pv_offset
    res = least_squares(resid, p0, bounds=bounds, xtol=1e-12, ftol=1e-12, max_nfev=10000)
    pred = sim_fn(t, u, *res.x) + (pv - pv_offset)[0]*0  # placeholder, fixed below
    return res.x, res

# ---------- segment A: clean 50% pulse ----------
d = load('AIC9_DATA-20260702-185858_203502-数据分析.xlsx', 'MV50%')
seg = d[(d['sec']>=0)&(d['sec']<=90)].reset_index(drop=True)
base = seg[seg['sec']<1.3]
mv_base, pv_base = base['mv'].mean(), base['pv1'].mean()
t = seg['sec'].values; u = seg['mv'].values - mv_base; pv = seg['pv1'].values; y_meas = pv - pv_base

# FOPDT (for reference, best-effort)
def resid_fo(p):
    K,tau,L = p
    return simulate_fopdt(t,u,K,tau,L) - y_meas
res_fo = least_squares(resid_fo, [5,12,1], bounds=([0.1,1,0],[30,60,15]), xtol=1e-12, ftol=1e-12)
pred_fo = simulate_fopdt(t,u,*res_fo.x) + pv_base
print('A) 50%脉冲  FOPDT :', dict(zip(['K','tau','L'], np.round(res_fo.x,3))), 'R2=', round(r2(pv,pred_fo),4))

# SOPDT
def resid_so(p):
    K,tau1,tau2,L = p
    return simulate_sopdt(t,u,K,tau1,tau2,L) - y_meas
res_so = least_squares(resid_so, [5,2,10,0.5], bounds=([0.1,0.1,0.1,0],[30,30,60,10]), xtol=1e-12, ftol=1e-12, max_nfev=20000)
pred_so = simulate_sopdt(t,u,*res_so.x) + pv_base
print('A) 50%脉冲  SOPDT :', dict(zip(['K','tau1','tau2','L'], np.round(res_so.x,3))), 'R2=', round(r2(pv,pred_so),4))

import pickle
with open('analysis/segA.pkl','wb') as f:
    pickle.dump(dict(t=t, pv=pv, pv_base=pv_base, mv=seg['mv'].values, pred_fo=pred_fo, pred_so=pred_so,
                      fo=res_fo.x, so=res_so.x), f)

# ---------- segment B: cold start, MV-saturated window ----------
d3 = load('AIC9_DATA-20260702-203502_204340-PID闭环数据.xlsx', 'data')
segB = d3[(d3['sec']>=0)&(d3['sec']<=13.9)].reset_index(drop=True)
preB = segB[segB['sec']<2]
mv_baseB, pv_baseB = preB['mv'].mean(), preB['pv1'].mean()
tB = segB['sec'].values; uB = segB['mv'].values - mv_baseB; pvB = segB['pv1'].values; yB = pvB - pv_baseB

def resid_fo_B(p):
    K,tau,L = p
    return simulate_fopdt(tB,uB,K,tau,L) - yB
res_fo_B = least_squares(resid_fo_B, [0.5,3,10], bounds=([0.05,0.5,0],[10,30,13.8]), xtol=1e-13, ftol=1e-13, max_nfev=20000)
pred_fo_B = simulate_fopdt(tB,uB,*res_fo_B.x) + pv_baseB
print('B) 冷启动   FOPDT :', dict(zip(['K','tau','L'], np.round(res_fo_B.x,3))), 'R2=', round(r2(pvB,pred_fo_B),4))

def resid_so_B(p):
    K,tau1,tau2,L = p
    return simulate_sopdt(tB,uB,K,tau1,tau2,L) - yB
res_so_B = least_squares(resid_so_B, [0.5,3,3,5], bounds=([0.05,0.1,0.1,0],[10,20,20,13.8]), xtol=1e-13, ftol=1e-13, max_nfev=30000)
pred_so_B = simulate_sopdt(tB,uB,*res_so_B.x) + pv_baseB
print('B) 冷启动   SOPDT :', dict(zip(['K','tau1','tau2','L'], np.round(res_so_B.x,3))), 'R2=', round(r2(pvB,pred_so_B),4))

with open('analysis/segB.pkl','wb') as f:
    pickle.dump(dict(t=tB, pv=pvB, pv_base=pv_baseB, mv=segB['mv'].values, pred_fo=pred_fo_B, pred_so=pred_so_B,
                      fo=res_fo_B.x, so=res_so_B.x), f)
