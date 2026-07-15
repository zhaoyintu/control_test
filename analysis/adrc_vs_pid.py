import numpy as np
import pickle

# ------------------------------------------------------------------
# Illustrative nonlinear plant. A smooth, strictly-monotonic-increasing-
# gain quadratic through the two most trustworthy real closed-loop
# anchors (100C@~2.6%MV, 400C@~21.5%MV) with curvature pinned so the
# local gain at u=40% comes out near the pulse-test secant K(40%)=26.2 --
# i.e. calibrated to the ~2.2-2.5x gain range actually measured. This is
# still NOT a precision-validated model of the real furnace -- it exists
# to test whether a design survives the REAL, measured amount of static-
# gain nonlinearity, for PID vs ADRC, on the same plant.
#
# ---- history of what changed across iterations (kept for context) ----
# 1) First version fed u_prev (1 step old) into the ESO's prediction, silently
#    assuming zero dead time -> real oscillation once wc got aggressive. Fixed
#    with a delay FIFO feeding the ESO the SAME delayed input the plant sees.
# 2) A raw step SV fed straight into a fast loop caused overshoot -- fixed by
#    reference shaping (critically damped 2nd-order TD), same idea as 2-DOF
#    PID setpoint weighting. A first attempt also fed the TD's rate (rv2)
#    forward into u0 (correct for 2nd-order/motion-control ADRC, WRONG for
#    this 1st-order design) -- removing it fixed a real WR-vs-overshoot blowup.
# 3) True settling time (+-1degC) is NOT the same metric as 90%-rise-time --
#    ADRC at wo=3x looked fast (rise90=1.60s) but had a long tail (6.46s to
#    truly settle). Traced to wo (observer bandwidth), not wr.
# 4) Pushing wo alone traded rise90 speed for a shorter tail (a real tradeoff)
#    -- UNTIL two more fixes removed most of that tradeoff:
#      a) L=0.5s was a conservative round number "from the 0.19-0.61s range" --
#         the real 300-400C dead time is specifically ~0.19-0.21s, not 0.5s.
#         Using an inflated L was silently capping the achievable wc.
#      b) A grid search over (wc, wo) jointly -- not one parameter at a time --
#         combined with Hammerstein-linearizing the KNOWN static map g(u) (so
#         the ESO only fights genuinely unknown disturbance, not the actuator's
#         own characterized nonlinearity) found a region where rise90, overshoot,
#         AND true settling are simultaneously excellent. No trade-off needed --
#         the earlier tradeoff curves were an artifact of an inflated L and only
#         ever sweeping one knob at a time.
#    (A bug was also found and fixed along the way: an early version of the
#    validation script's metrics() had no upper time bound, so a "100->300"
#    overshoot measurement was contaminated by the LATER 300->400 peak,
#    reporting a false 50% overshoot. Always window metrics to end before the
#    next setpoint change.)
# ------------------------------------------------------------------
A_Q, B_Q, C_Q = 0.1812, 11.504, 68.86

def y_ss(u):
    return A_Q * u**2 + B_Q * u + C_Q

def y_ss_inv(v):
    disc = max(B_Q**2 - 4 * A_Q * (C_Q - v), 0.0)
    return (-B_Q + np.sqrt(disc)) / (2 * A_Q)

def k_local(u):
    return 2 * A_Q * u + B_Q

TAU = 13.0    # s, from the good pulse fits (12.6-13.2s)
L = 0.2       # s, CORRECTED: matches the real ~0.19-0.21s dead time measured in 100-400C steps
DT = 0.02
ND = max(1, int(round(L / DT)))
NOISE_STD = 0.03   # matches real steady-state PV std (0.019-0.055C)

np.random.seed(2)


def simulate(step_fn, sv_schedule, dist_fn, T_total):
    n = int(T_total / DT)
    t = np.arange(n) * DT
    y = np.zeros(n)
    u_hist = np.zeros(n)
    y[0] = sv_schedule(0.0)
    state = {}
    for i in range(1, n):
        sv = sv_schedule(t[i])
        y_meas = y[i - 1] + np.random.normal(0, NOISE_STD)
        u_cmd, state = step_fn(state, y_meas, sv, DT, first=(i == 1))
        u_cmd = float(np.clip(u_cmd, 0.0, 100.0))
        u_hist[i] = u_cmd
        u_delayed = u_hist[i - ND] if i - ND >= 0 else 0.0
        d = dist_fn(t[i])
        n_sub = 4
        yi = y[i - 1]
        for _ in range(n_sub):
            yi = yi + (DT / n_sub) * (y_ss(u_delayed) + d - yi) / TAU
        y[i] = yi
    return t, y, u_hist


def shape_ref(state, sv_raw, dt, wr):
    v1 = state.get('rv1', sv_raw)
    v2 = state.get('rv2', 0.0)
    v1n = v1 + dt * v2
    v2n = v2 + dt * (-2 * wr * v2 - wr * wr * (v1 - sv_raw))
    return v1n, v2n


# ---------------- PID, SIMC-retuned with the CORRECTED dead time ----------------
K_MID = k_local(15.0)
TC = 0.3
KC = (1.0 / K_MID) * (TAU / (TC + L))
TI = min(TAU, 4 * (TC + L))
WR_PID = 3.0
print(f"PID retuned: K_local={K_MID:.2f}  Kc={KC:.3f}  Ti={TI:.2f}s  (SIMC, corrected L={L}s)")

def pid_step(state, y_meas, sv_raw, dt, first=False):
    if first or 'integ' not in state:
        state = {'integ': 0.0}
    rv1, rv2 = shape_ref(state, sv_raw, dt, WR_PID)
    sv = rv1
    e = sv - y_meas
    state['integ'] += e * dt
    u = KC * e + (KC / TI) * state['integ']
    u_sat = np.clip(u, 0.0, 100.0)
    if u_sat != u:
        state['integ'] -= e * dt
    state['rv1'], state['rv2'] = rv1, rv2
    return u_sat, state


# ---------------- ADRC: linearized + corrected dead time + jointly-searched (wc,wo) ----------------
# b0 = 1/tau exactly -- no guessing, because commanding in v=g(u) space makes the
# v->y path an exact unity-gain FOPDT by construction (see adrc_linearized.py).
B0_LIN = 1.0 / TAU
WC, WO, WR_ADRC = 2.0, 20.0, 8.0
print(f"ADRC final: wc={WC} wo={WO} wr={WR_ADRC}  (linearized, corrected L={L}s)")

def adrc_step(state, y_meas, sv_raw, dt, first=False):
    if first or 'z1' not in state:
        state = {'z1': y_meas, 'z2': 0.0, 'vbuf': [y_ss(0.0)] * (ND + 1)}
    rv1, rv2 = shape_ref(state, sv_raw, dt, WR_ADRC)
    sv = rv1
    z1, z2, vbuf = state['z1'], state['z2'], state['vbuf']
    v_delayed = vbuf[-1]
    err_o = y_meas - z1
    z1n = z1 + dt * (z2 + B0_LIN * v_delayed + 2 * WO * err_o)
    z2n = z2 + dt * (WO * WO * err_o)
    u0 = WC * (sv - z1n)
    v_cmd = (u0 - z2n) / B0_LIN
    u_cmd = y_ss_inv(v_cmd)
    u_sat = np.clip(u_cmd, 0.0, 100.0)
    v_applied = y_ss(u_sat)
    vbuf = [v_applied] + vbuf[:-1]
    state = {'z1': z1n, 'z2': z2n, 'vbuf': vbuf, 'rv1': rv1, 'rv2': rv2}
    return u_sat, state


def sv_schedule(t):
    if t < 10:
        return 100.0
    elif t < 40:
        return 300.0
    else:
        return 400.0

def dist_fn(t):
    return 8.0 if t > 55 else 0.0

T_TOTAL = 90.0

results = {}
for name, fn in [('PID_single', pid_step), ('ADRC_final', adrc_step)]:
    t, y, u = simulate(fn, sv_schedule, dist_fn, T_TOTAL)
    results[name] = (t, y, u)

def metrics(t, y, t0, t1, target, amp):
    # NOTE: t1 upper bound is load-bearing -- omitting it once contaminated a
    # "100->300" overshoot reading with the later 300->400 step's peak (see history above).
    seg = (t >= t0) & (t <= t1)
    tt, yy = t[seg], y[seg]
    y0 = target - amp
    idx = np.where(np.sign(amp) * (yy - y0) >= np.sign(amp) * 0.9 * amp)[0]
    t90 = (tt[idx[0]] - t0) if len(idx) else float('nan')
    peak = yy.max() if amp > 0 else yy.min()
    ov = (peak - target) / amp * 100
    ok = np.abs(yy - target) <= 1.0
    bad = np.where(~ok)[0]
    tset = (tt[bad[-1] + 1] - t0) if len(bad) and bad[-1] + 1 < len(tt) else 0.0
    return t90, ov, tset

print("\n--- 100->300 step (t=10) ---")
for name in results:
    t, y, u = results[name]
    t90, ov, tset = metrics(t, y, 10, 38, 300, 200)
    print(f"{name:12s} rise90={t90:5.2f}s overshoot={ov:6.2f}%  settle+-1={tset:5.2f}s")

print("\n--- 300->400 step (t=40) ---")
for name in results:
    t, y, u = results[name]
    t90, ov, tset = metrics(t, y, 40, 53, 400, 100)
    print(f"{name:12s} rise90={t90:5.2f}s overshoot={ov:6.2f}%  settle+-1={tset:5.2f}s")
print(f"{'真实系统(参考)':12s} rise90~1.12s overshoot~3.1%  settle+-1~3.47s  (from real closed-loop xlsx)")

print("\n--- disturbance rejection (unmodeled step disturbance at t=55, holding SV=400) ---")
for name in results:
    t, y, u = results[name]
    seg = (t >= 55) & (t <= 90)
    dev = y[seg] - 400.0
    peak_idx = np.argmax(np.abs(dev))
    print(f"{name:12s} peak_dev={dev[peak_idx]:6.3f}  final_dev={dev[-1]:6.3f}")

with open('analysis/adrc_vs_pid.pkl', 'wb') as f:
    pickle.dump({'results': results, 't': results['PID_single'][0]}, f)
print("\nsaved analysis/adrc_vs_pid.pkl")
