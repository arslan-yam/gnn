"""E1e: second-order (topology) term of the two-block threshold.

Theorem 1(v): kappa(t0; rA, rB) = 1 + [log L - 1/2 log log L + 1/2 log(rA + rB) + o(1)] / L,  L = log t0.
(a) validated enclosures of kappa(t0; rho, rho) for moderate t0 (discrete map, core.kappa_enclosure);
(b) the continuum limit dx/dtau = -lam (1+x) exp(-tau (2x + x^2)) (tau = log s), shot for the critical x
    at tau = L up to L = 1e5, to show the residual (kappa-1) L - [log L - 1/2 loglog L + 1/2 log lam] -> 0;
(c) a high-precision check of the barrier inequalities used in the proof (and of their third-order analogue).
Usage: python -m experiments.exp_second_order [--barriers]   (--barriers redoes only (c))
Writes results/theory_E1e.json.
"""
import json
import math
import sys
import time

import numpy as np
from mpmath import exp as mexp, log as mlog, mp, mpf
from scipy.integrate import solve_ivp

from gnnlimit.core import kappa_enclosure

RHOS = [0.1, 0.25, 0.5, 1.0]
T0_ENC = [11.0, 30.0, 100.0, 300.0, 1000.0]
L_GRID = list(np.round(np.exp(np.linspace(np.log(math.log(11.0)), np.log(1e5), 28)), 6))
TAUS = [3.0 * 1.02 ** k for k in range(400)]
T_START = time.monotonic()


def stamp(msg):
    print(f"[{time.monotonic() - T_START:7.1f}s] {msg}", flush=True)


def two_term(L, lam):
    return (math.log(L) - 0.5 * math.log(math.log(L)) + 0.5 * math.log(lam)) / L


def three_term(L, lam):
    """Formal next term (not proved): + (loglog L - log lam + 2) / (4 log L) / L."""
    return two_term(L, lam) + (math.log(math.log(L)) - math.log(lam) + 2.0) / (4.0 * math.log(L)) / L


def continuum_fate(x0, L, lam):
    def f(t, y):
        x = y[0]
        return [-lam * (1 + x) * math.exp(-t * (2 * x + x * x))]

    def merged(t, y):
        return y[0]
    merged.terminal, merged.direction = True, -1

    def safe(t, y):  # x stays >= x/2 forever if the frozen-gap loss bound is below x/2
        x = y[0]
        return -1.0 if x <= 0 else x / 2 - lam * (1 + x) * math.exp(-t * x) / x
    safe.terminal, safe.direction = True, 1
    if safe(L, [x0]) >= 0:
        return 1
    sol = solve_ivp(f, (L, L * 1e9), [x0], events=[merged, safe], rtol=1e-10, atol=1e-14, method="LSODA")
    if sol.t_events[0].size:
        return 0
    if sol.t_events[1].size:
        return 1
    raise RuntimeError(f"unresolved continuum trajectory L={L} lam={lam} x0={x0}")


def continuum_kappa(L, lam):
    a, b = 0.0, 3 * math.log(max(L, math.e)) / L + 1.0
    for _ in range(50):
        m = 0.5 * (a + b)
        if continuum_fate(m, L, lam):
            b = m
        else:
            a = m
    return 1 + 0.5 * (a + b)


def barrier_check(rA, rB, eta, sign, taus, order=2):
    """sign=+1: F(1+X(tau)) >= 1+X(tau'); sign=-1: <=. X = (log tau - loglog tau/2 + c)/tau with
    c = log(lam)/2 + sign*eta (order 2) or c = log(lam)/2 + (loglog tau - log lam + 2 + sign*eta)/(4 log tau) (order 3).
    Both sides differ from 1+X(tau) by O(e^-tau), so the working precision grows with tau (tau/log 10 + 40 digits).
    Returns the largest grid tau where the inequality fails (None if it holds on the whole grid)."""
    worst = None
    for tau in taus:
        with mp.workdps(int(tau / math.log(10)) + 40):
            lam, P = mpf(rA) + mpf(rB), mpf(rA) * mpf(rB)

            def X(t):
                c = mlog(lam) / 2 + (sign * mpf(eta) if order == 2
                                     else (mlog(mlog(t)) - mlog(lam) + 2 + sign * mpf(eta)) / (4 * mlog(t)))
                return (mlog(t) - mlog(mlog(t)) / 2 + c) / t
            t0 = mpf(tau)
            s = mexp(t0)
            t1 = mlog(s + 1)
            d = 1 + X(t0)
            a = s ** (-d * d)
            Fd = d * (1 - P * a * a) / ((1 + rA * a) * (1 + rB * a))
            diff = Fd - (1 + X(t1))
            if (sign > 0 and diff < 0) or (sign < 0 and diff > 0):
                worst = float(tau)
    return worst


def barrier_table(taus):
    checks = []
    for order in (2, 3):
        for rA, rB in [(1, 1), (0.5, 0.5), (0.25, 0.25), (0.1, 0.1), (1, 0.25)]:
            for eta in [0.5, 0.2]:
                row = dict(order=order, rA=rA, rB=rB, eta=eta, tau_max=taus[-1],
                           persist_fails_up_to=barrier_check(rA, rB, eta, +1, taus, order),
                           merge_fails_up_to=barrier_check(rA, rB, eta, -1, taus, order))
                checks.append(row)
                stamp(f"  {row}")
    return checks


if __name__ == "__main__":
    if sys.argv[1:] == ["--barriers"]:  # redo only part (c), keep (a) and (b)
        with open("results/theory_E1e.json") as f:
            out = json.load(f)
        stamp("barrier inequalities (precision grows with tau)")
        out["barrier_checks"] = barrier_table(TAUS)
        with open("results/theory_E1e.json", "w") as f:
            json.dump(out, f, indent=1)
        stamp("done")
        sys.exit(0)
    out = dict(rhos=RHOS, t0_enc=T0_ENC, L_grid=L_GRID)
    stamp("validated enclosures kappa(t0; rho, rho)")
    enc = []
    for rho in RHOS:
        for t0 in T0_ENC:
            klo, khi, est = kappa_enclosure(t0, rho, rho, Tmax=10**6, T_float=10**6)
            enc.append(dict(rho=rho, t0=t0, klo=klo, khi=khi, est=est))
            stamp(f"  rho={rho} t0={t0:g}: [{klo:.5f}, {khi:.5f}]")
    out["enclosures"] = enc

    stamp("continuum limit")
    cont = {}
    for rho in RHOS:
        lam = 2 * rho
        ks = [continuum_kappa(L, lam) for L in L_GRID]
        cont[str(rho)] = dict(kappa=ks, residual=[(k - 1) * L - two_term(L, lam) * L for k, L in zip(ks, L_GRID)],
                              residual3=[(k - 1) * L - three_term(L, lam) * L for k, L in zip(ks, L_GRID)])
        stamp(f"  rho={rho}: residual at L={L_GRID[0]:.2f}: {cont[str(rho)]['residual'][0]:+.3f}, "
              f"at L=1e5: {cont[str(rho)]['residual'][-1]:+.3f} (three-term: {cont[str(rho)]['residual3'][-1]:+.4f})")
    out["continuum"] = cont

    stamp("barrier inequalities (precision grows with tau)")
    out["barrier_checks"] = barrier_table(TAUS)
    with open("results/theory_E1e.json", "w") as f:
        json.dump(out, f, indent=1)
    stamp("done")
