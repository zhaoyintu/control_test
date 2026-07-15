import numpy as np

A_Q, B_Q, C_Q = 0.1812, 11.504, 68.86
def y_ss(u): return A_Q*u**2+B_Q*u+C_Q
def y_ss_inv(v):
    # solve A*u^2 + B*u + (C-v) = 0 for the positive root
    disc = B_Q**2 - 4*A_Q*(C_Q - v)
    disc = max(disc, 0.0)
    return (-B_Q + np.sqrt(disc)) / (2*A_Q)
def k_local(u): return 2*A_Q*u+B_Q
TAU=13.0; L=0.5; DT=0.02; ND=max(1,int(round(L/DT)))

def simulate(step_fn, sv_schedule, T_total):
    n=int(T_total/DT); t=np.arange(n)*DT
    y=np.zeros(n); u_hist=np.zeros(n); y[0]=sv_schedule(0.0)
    state={}
    for i in range(1,n):
        sv=sv_schedule(t[i])
        u_cmd,state=step_fn(state,y[i-1],sv,DT,first=(i==1))
        u_cmd=float(np.clip(u_cmd,0,100)); u_hist[i]=u_cmd
        u_delayed=u_hist[i-ND] if i-ND>=0 else 0.0
        n_sub=4; yi=y[i-1]
        for _ in range(n_sub):
            yi=yi+(DT/n_sub)*(y_ss(u_delayed)-yi)/TAU
        y[i]=yi
    return t,y,u_hist

def shape_ref(state, sv_raw, dt, wr):
    v1=state.get('rv1',sv_raw); v2=state.get('rv2',0.0)
    v1n=v1+dt*v2; v2n=v2+dt*(-2*wr*v2-wr*wr*(v1-sv_raw))
    return v1n,v2n

def sv_schedule(t):
    return 300.0 if t<10 else 400.0

def settling_time(t, y, t0, target, band):
    seg = t >= t0
    tt, yy = t[seg], y[seg]
    ok = np.abs(yy - target) <= band
    bad_idx = np.where(~ok)[0]
    if len(bad_idx)==0: return 0.0
    last_bad = bad_idx[-1]
    return float('inf') if last_bad+1>=len(tt) else tt[last_bad+1]-t0

def metrics(t,y,t0,target,amp,band=1.0):
    seg=(t>=t0); tt,yy=t[seg],y[seg]
    y0=target-amp
    idx=np.where(np.sign(amp)*(yy-y0)>=np.sign(amp)*0.9*amp)[0]
    t90=(tt[idx[0]]-t0) if len(idx) else float('nan')
    peak=yy.max() if amp>0 else yy.min()
    ov=(peak-target)/amp*100
    ts=settling_time(t,y,t0,target,band)
    return t90,ov,ts

WC=0.9; WO=3*WC   # ORIGINAL, conservative wo -- not cranked up
WR=5.0

# ---------- baseline: current design (b0 from a single mid-point guess, no linearization) ----------
K_MID = k_local(15.0)
B0 = K_MID/TAU
def adrc_step_baseline(state,y_meas,sv_raw,dt,first=False):
    if first or 'z1' not in state:
        state={'z1':y_meas,'z2':0.0,'ubuf':[0.0]*(ND+1)}
    rv1,rv2=shape_ref(state,sv_raw,dt,WR); sv=rv1
    z1,z2,ubuf=state['z1'],state['z2'],state['ubuf']
    u_delayed=ubuf[-1]; err_o=y_meas-z1
    z1n=z1+dt*(z2+B0*u_delayed+2*WO*err_o)
    z2n=z2+dt*(WO*WO*err_o)
    u0=WC*(sv-z1n)
    u=(u0-z2n)/B0; u_sat=np.clip(u,0,100)
    ubuf=[u_sat]+ubuf[:-1]
    state={'z1':z1n,'z2':z2n,'ubuf':ubuf,'rv1':rv1,'rv2':rv2}
    return u_sat,state

# ---------- improved: Hammerstein-linearized (known static map inverted at the last step) ----------
B0_LIN = 1.0/TAU   # exact now -- no guessing, gain is 1 by construction
def adrc_step_linearized(state,y_meas,sv_raw,dt,first=False):
    if first or 'z1' not in state:
        state={'z1':y_meas,'z2':0.0,'vbuf':[y_ss(0.0)]*(ND+1)}
    rv1,rv2=shape_ref(state,sv_raw,dt,WR); sv=rv1
    z1,z2,vbuf=state['z1'],state['z2'],state['vbuf']
    v_delayed=vbuf[-1]; err_o=y_meas-z1
    z1n=z1+dt*(z2+B0_LIN*v_delayed+2*WO*err_o)
    z2n=z2+dt*(WO*WO*err_o)
    u0=WC*(sv-z1n)
    v_cmd=(u0-z2n)/B0_LIN     # this is now in "temperature-equivalent" units
    u_cmd=y_ss_inv(v_cmd)     # invert the KNOWN static map -> actual MV%
    u_sat=np.clip(u_cmd,0,100)
    v_applied = y_ss(u_sat)   # what v actually resulted after the clamp
    vbuf=[v_applied]+vbuf[:-1]
    state={'z1':z1n,'z2':z2n,'vbuf':vbuf,'rv1':rv1,'rv2':rv2}
    return u_sat,state

print(f"{'design':22s} {'rise90':>8s} {'overshoot':>10s} {'settle+-1':>10s} {'settle+-0.5':>12s}")
for name, fn in [('baseline (guess b0)', adrc_step_baseline), ('linearized (exact b0=1/tau)', adrc_step_linearized)]:
    t,y,u = simulate(fn, sv_schedule, 40.0)
    t90,ov,ts1 = metrics(t,y,10,400,100,1.0)
    _,_,ts05 = metrics(t,y,10,400,100,0.5)
    print(f"{name:22s} {t90:8.2f} {ov:10.2f} {ts1:10.2f} {ts05:12.2f}")

print("\nlinearized design, sweeping wo (combine with the other lever):")
print(f"{'wo/wc':>6s} {'rise90':>8s} {'overshoot':>10s} {'settle+-1':>10s}")
def make_lin_wo(wo):
    def adrc_step(state,y_meas,sv_raw,dt,first=False):
        if first or 'z1' not in state:
            state={'z1':y_meas,'z2':0.0,'vbuf':[y_ss(0.0)]*(ND+1)}
        rv1,rv2=shape_ref(state,sv_raw,dt,WR); sv=rv1
        z1,z2,vbuf=state['z1'],state['z2'],state['vbuf']
        v_delayed=vbuf[-1]; err_o=y_meas-z1
        z1n=z1+dt*(z2+B0_LIN*v_delayed+2*wo*err_o)
        z2n=z2+dt*(wo*wo*err_o)
        u0=WC*(sv-z1n)
        v_cmd=(u0-z2n)/B0_LIN
        u_cmd=y_ss_inv(v_cmd)
        u_sat=np.clip(u_cmd,0,100)
        v_applied=y_ss(u_sat)
        vbuf=[v_applied]+vbuf[:-1]
        state={'z1':z1n,'z2':z2n,'vbuf':vbuf,'rv1':rv1,'rv2':rv2}
        return u_sat,state
    return adrc_step

for mult in [1,2,3,5,7,10]:
    fn = make_lin_wo(mult*WC)
    t,y,u = simulate(fn, sv_schedule, 40.0)
    t90,ov,ts1 = metrics(t,y,10,400,100,1.0)
    print(f"{mult:6d} {t90:8.2f} {ov:10.2f} {ts1:10.2f}")

print("\nfor reference, real system: rise90~1.12s (measured earlier), settle+-1=3.47s")

print("\nlinearized design, wo=5*wc fixed, now sweeping wc itself:")
print(f"{'wc':>5s} {'rise90':>8s} {'overshoot':>10s} {'settle+-1':>10s}")
def make_lin_wc(wc, wo_mult=5.0):
    wo = wo_mult*wc
    def adrc_step(state,y_meas,sv_raw,dt,first=False):
        if first or 'z1' not in state:
            state={'z1':y_meas,'z2':0.0,'vbuf':[y_ss(0.0)]*(ND+1)}
        rv1,rv2=shape_ref(state,sv_raw,dt,WR); sv=rv1
        z1,z2,vbuf=state['z1'],state['z2'],state['vbuf']
        v_delayed=vbuf[-1]; err_o=y_meas-z1
        z1n=z1+dt*(z2+B0_LIN*v_delayed+2*wo*err_o)
        z2n=z2+dt*(wo*wo*err_o)
        u0=wc*(sv-z1n)
        v_cmd=(u0-z2n)/B0_LIN
        u_cmd=y_ss_inv(v_cmd)
        u_sat=np.clip(u_cmd,0,100)
        v_applied=y_ss(u_sat)
        vbuf=[v_applied]+vbuf[:-1]
        state={'z1':z1n,'z2':z2n,'vbuf':vbuf,'rv1':rv1,'rv2':rv2}
        return u_sat,state
    return adrc_step

for wc in [0.9, 1.5, 2.5, 4.0, 6.0, 9.0]:
    fn = make_lin_wc(wc)
    t,y,u = simulate(fn, sv_schedule, 40.0)
    t90,ov,ts1 = metrics(t,y,10,400,100,1.0)
    print(f"{wc:5.1f} {t90:8.2f} {ov:10.2f} {ts1:10.2f}")
