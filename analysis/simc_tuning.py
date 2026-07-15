"""
SIMC (Skogestad Internal Model Control) tuning: turn an identified FOPDT model
(K, tau, L -- from sysid_fopdt3.py) into PI parameters (Kc, Ti), by choosing one
design parameter tau_c (desired closed-loop time constant) instead of tuning
Kc/Ti by trial and error.

Design idea: choose Kc, Ti such that IF the plant exactly matched the FOPDT
model, the closed loop would collapse to a clean first-order response with
time constant tau_c. The formulas below are the algebraic result of that
requirement (see Skogestad, "Simple analytic rules for model reduction and
PID controller tuning", 2003).
"""

def simc_pi(K, tau, L, tau_c):
    Kc = (1.0 / K) * (tau / (tau_c + L))
    Ti = min(tau, 4 * (tau_c + L))
    return Kc, Ti

if __name__ == '__main__':
    K, TAU, L = 16.94, 13.0, 0.2   # from sysid_fopdt3.py, K at u=15%
    print(f"{'tau_c':>6s} {'Kc':>8s} {'Ti':>8s}")
    for tc in [0.2, 0.3, 0.5, 1.0, 2.0, 5.0]:
        kc, ti = simc_pi(K, TAU, L, tc)
        print(f"{tc:6.1f} {kc:8.3f} {ti:8.2f}")
