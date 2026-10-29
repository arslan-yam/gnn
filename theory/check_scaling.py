"""Rescue boundary u*(v0) of gap 1 in the three-layer chain, normalised by kappa(t0), compared with
the scaling limit phi(x) = max(x, 1 - x/2) (x = v0/kappa < 1), phi = 1 (x > 1) of Theorem B2.
Float (non-interval) classification:
  merge   if h = u + v/2 < 1 at some time                       (Proposition B0(a), exact);
  persist if u(t) >= k_t, the isolated critical trajectory started from kappa_hi (Theorem 2 + 1(iv)).
kappa_hi is a validated upper enclosure of kappa(t0) where available (E1), else estimate*(1+1e-6)."""
import json, math, os, sys
import numba
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gnnlimit.core import kappa_estimate

KHI = {11.0: 1.53909, 1e3: 1.29517}   # validated upper enclosures from results/theory_E1.json

@numba.njit(cache=True)
def fate(u, v, t0, khi, Tmax):
    k = khi
    for t in range(Tmax):
        L = math.log(t + t0)
        w1 = math.exp(-u * u * L); w2 = math.exp(-v * v * L)
        D1 = 1 + w1; D2 = 1 + w1 + w2; D3 = 1 + w2
        un = u * (1 + w2 - w1 * w1) / (D1 * D2) + w2 * v / D2
        vn = v * (1 + w1 - w2 * w2) / (D2 * D3) + w1 * u / D2
        a = math.exp(-k * k * L)
        k = k * (1 - a) / (1 + a)
        u, v = un, vn
        if u + 0.5 * v < 1.0:
            return 0, t + 1
        if u >= k:
            return 1, t + 1
    return -1, Tmax

def boundary(x, t0, kap, khi, Tmax):
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        f, _ = fate(mid * kap, x * kap, t0, khi, Tmax)
        if f == -1:
            break
        if f == 1:
            hi = mid
        else:
            lo = mid
    return lo, hi

phi = lambda x: max(x, 1 - x / 2) if x < 1 else 1.0
xs = [0.1, 0.3, 0.5, 0.6, 2 / 3, 0.7, 0.8, 0.9, 1.2]
out = {}
for t0 in [11.0, 1e3, 1e5]:
    kap = kappa_estimate(t0, T=10**7)
    khi = KHI.get(t0, kap * (1 + 1e-6))
    rows = [boundary(x, t0, kap, khi, 10**7) for x in xs]
    out[t0] = dict(kappa_est=kap, kappa_hi=khi, xs=xs, ustar_over_kappa=rows)
    print(f"t0={t0:g} kappa~{kap:.5f}: " + " ".join(f"{x:.2f}:[{a:.3f},{b:.3f}]" for x, (a, b) in zip(xs, rows)), flush=True)
print("phi(x):           " + " ".join(f"{x:.2f}:{phi(x):.3f}" for x in xs))
print("exact lower bound (1 - v0/2)/kappa at t0=1e5:", " ".join(f"{(1 - x * out[1e5]['kappa_est'] / 2) / out[1e5]['kappa_est']:.3f}" for x in xs))
json.dump({str(k): v for k, v in out.items()}, open("check_scaling.json", "w"), indent=1)
