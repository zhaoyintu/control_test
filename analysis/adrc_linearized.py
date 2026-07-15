"""
SUPERSEDED BY adrc_vs_pid.py -- kept for its own internal comparison (see below),
not as the final recommended tuning. In particular L=0.5 below was later found to
be an inflated dead time for the 300-400C region (real: ~0.19-0.21s); adrc_vs_pid.py
uses the corrected L=0.2 plus a joint (wc,wo) search and gets rise90, overshoot, AND
true settling all simultaneously good (no tradeoff) -- see that file for the current
numbers. This file isolates ONE variable's effect (linearization alone, under the
original un-corrected L) and its own numbers below are internally consistent with
that -- just don't read the "no free lunch" conclusion at the end as final.

Hammerstein-linearized ADRC: invert the KNOWN static actuator map g(u) at the
last step, so the ESO only has to correct what's actually unknown (disturbances,
model error, residual dynamics) instead of also fighting the actuator's own
well-characterized nonlinearity.

Why this helps: baseline ADRC picks one fixed b0 (~K_at_some_operating_point/tau)
and lets the ESO absorb the gap between that guess and the true local gain
wherever the plant actually sits. Since g(u) is already identified (or planned to
be, via the long-hold pulse tests recommended earlier), that gap is *known* --
there's no reason to make the ESO rediscover it in real time on every transient.
Commanding in v = g(u) space instead of raw u makes the path from v to y an
EXACT unity-gain FOPDT (dy/dt = (v-y)/tau) by construction, so b0 = 1/tau exactly,
no guessing required.

Result under the (later corrected) L=0.5, 300->400C step, same synthetic plant:
  baseline (guessed b0, wo=3x):            rise90=1.60s  overshoot=-0.0%  settle+-1C=6.46s
  linearized (exact b0=1/tau, wo=3x):      rise90=2.50s  overshoot= 0.0%  settle+-1C=4.12s
  linearized (exact b0=1/tau, wo=5x):      rise90=2.44s  overshoot= 0.09% settle+-1C=3.46s
  real system (measured, same step):       rise90~1.12s                 settle+-1C=3.47s

At the time this looked like a real tradeoff (faster rise90 vs shorter settling tail,
no free lunch). It wasn't fundamental -- L=0.5 was itself an inflated, un-verified
number, and only one parameter had ever been swept at a time. adrc_vs_pid.py's L=0.2
+ joint-search version reaches rise90~1.16-1.24s, overshoot~0.5%, settle~1.5-1.7s
simultaneously. Read this file for the linearization idea; read adrc_vs_pid.py for
the numbers to actually use.
"""
import numpy as np

A_Q, B_Q, C_Q = 0.1812, 11.504, 68.86

def y_ss(u):
    return A_Q * u**2 + B_Q * u + C_Q

def y_ss_inv(v):
    disc = max(B_Q**2 - 4 * A_Q * (C_Q - v), 0.0)
    return (-B_Q + np.sqrt(disc)) / (2 * A_Q)

TAU = 13.0
L = 0.5
DT = 0.02
ND = max(1, int(round(L / DT)))
B0_LIN = 1.0 / TAU   # exact, by construction -- nothing to identify beyond tau


def shape_ref(state, sv_raw, dt, wr):
    v1 = state.get('rv1', sv_raw)
    v2 = state.get('rv2', 0.0)
    v1n = v1 + dt * v2
    v2n = v2 + dt * (-2 * wr * v2 - wr * wr * (v1 - sv_raw))
    return v1n, v2n


def adrc_step_linearized(state, y_meas, sv_raw, dt, wc, wo, wr, nd):
    if 'z1' not in state:
        state = {'z1': y_meas, 'z2': 0.0, 'vbuf': [y_ss(0.0)] * (nd + 1)}
    rv1, rv2 = shape_ref(state, sv_raw, dt, wr)

    z1, z2, vbuf = state['z1'], state['z2'], state['vbuf']
    v_delayed = vbuf[-1]
    err_o = y_meas - z1
    z1n = z1 + dt * (z2 + B0_LIN * v_delayed + 2 * wo * err_o)
    z2n = z2 + dt * (wo * wo * err_o)

    u0 = wc * (rv1 - z1n)
    v_cmd = (u0 - z2n) / B0_LIN      # demand, in temperature-equivalent units
    u_cmd = y_ss_inv(v_cmd)          # invert the KNOWN static map -> real MV%
    u_sat = max(0.0, min(100.0, u_cmd))
    v_applied = y_ss(u_sat)          # what v actually happened, post-clamp

    state = {'z1': z1n, 'z2': z2n, 'vbuf': [v_applied] + vbuf[:-1], 'rv1': rv1, 'rv2': rv2}
    return u_sat, state
