"""Numerical sanity checks for the identities used in theory/proofs.tex (three-layer chain).
Units of sqrt(c); s = t + t0, w1 = s^(-u^2), w2 = s^(-v^2)."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gnnlimit.core import chain_step

rng = np.random.default_rng(1)
worst = {"I1": 0, "I2": 0, "I3": 0, "v_lower": 0, "comp": 0}
for _ in range(200000):
    u, v = rng.uniform(0, 3, 2)
    s = np.exp(rng.uniform(0.01, 12))
    t0 = s  # chain_step uses log(t + t0); take t = 0
    w1, w2 = s ** (-u * u), s ** (-v * v)
    D1, D2, D3 = 1 + w1, 1 + w1 + w2, 1 + w2
    un, vn = chain_step(np.array([u, v]), 0, t0)
    F = u * (1 - w1) / (1 + w1)
    worst["I1"] = max(worst["I1"], abs(un - (F + w2 / D2 * (v + w1 * u / D1))))
    h, hn = u + v / 2, un + vn / 2
    worst["I2"] = max(worst["I2"], abs(hn - (h - w1 * (u * (1 / (2 * D2) + 1 / D1) + w2 * v / (2 * D2 * D3)))))
    worst["I3"] = max(worst["I3"], abs((un + vn) - (u + v - w1 * u / D1 - w2 * v / D3)))
    worst["v_lower"] = max(worst["v_lower"], v * (1 - 2 * w2) - vn)          # should be <= 0
    worst["comp"] = max(worst["comp"], F - un)                                 # comparison: <= 0
print({k: float(x) for k, x in worst.items()})
