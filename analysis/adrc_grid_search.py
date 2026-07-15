import numpy as np

A_Q, B_Q, C_Q = 0.1812, 11.504, 68.86
def y_ss(u): return A_Q*u**2+B_Q*u+C_Q
def y_ss_inv(v):
    disc = max(B_Q**2 - 4*A_Q*(C_Q - v), 0.0)
    return (-B_Q + np.sqrt(disc)) / (2*A_Q)

TAU = 13.0
L = 0.2          # CORRECTED: real 300->400 dead time is ~0.19-0.21s, not the 0.5s used before
DT = 0.02
ND = max(1, int(round(L/DT)))
B0_LIN = 1.0/TAU

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

def make_lin(wc, wo, wr):
    def adrc_step(state,y_meas,sv_raw,dt,first=False):
        if first or 'z1' not in state:
            state={'z1':y_meas,'z2':0.0,'vbuf':[y_ss(0.0)]*(ND+1)}
        rv1,rv2=shape_ref(state,sv_raw,dt,wr); sv=rv1
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

def sv_schedule(t):
    return 300.0 if t<10 else 400.0

def metrics(t,y,t0,target,amp,band=1.0):
    seg=(t>=t0); tt,yy=t[seg],y[seg]
    y0=target-amp
    idx=np.where(np.sign(amp)*(yy-y0)>=np.sign(amp)*0.9*amp)[0]
    t90=(tt[idx[0]]-t0) if len(idx) else float('nan')
    peak=yy.max() if amp>0 else yy.min()
    ov=(peak-target)/amp*100
    ok=np.abs(yy-target)<=band
    bad=np.where(~ok)[0]
    ts=(tt[bad[-1]+1]-t0) if len(bad) and bad[-1]+1<len(tt) else 0.0
    return t90,ov,ts

print(f"L={L}s (corrected), ND={ND} steps, wr=8.0 fixed")
print(f"{'wc':>5s} {'wo/wc':>6s} {'rise90':>8s} {'overshoot':>10s} {'settle+-1':>10s}")
best = None
for wc in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    for wo_mult in [3, 5, 7, 10, 15]:
        wo = wo_mult*wc
        fn = make_lin(wc, wo, 8.0)
        t,y,u = simulate(fn, sv_schedule, 30.0)
        t90,ov,ts = metrics(t,y,10,400,100,1.0)
        flag = ""
        if ov < 3.0 and t90 < 1.3 and ts < 4.0:
            flag = "  <-- all three good"
            if best is None or (t90+ts) < (best[2]+best[4]):
                best = (wc, wo_mult, t90, ov, ts)
        print(f"{wc:5.1f} {wo_mult:6d} {t90:8.2f} {ov:10.2f} {ts:10.2f}{flag}")
print("\nbest found:", best)
