"""Explicit constants quoted in theory/proofs.tex (Theorems B1, B2). Writes constants.json.
Units of sqrt(c). kappa_+ is the C1 upper bound of Theorem 1(vi) (exp_theory.kappa_upper)."""
import functools, json, math, os, sys
print = functools.partial(print, flush=True)
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments.exp_theory import kappa_upper, kappa_lower

OUT = {}
KHI = {5.0: 1.66879, 11.0: 1.53909, 17.0: 1.49159, 100.0: 1.37280, 1e3: 1.29517, 1e6: 1.20573}  # validated (E1)

# ---------------- Theorem B1 (one-step rescue, every v0 > 0) ----------------
def B1_depth(t0, v0):
    L = math.log(t0)
    w = t0 ** (-v0 * v0)
    p0 = w * v0 / (1 + w)
    dm = 1 - p0
    m = max(dm * dm, 1.5 / L)
    p = w * v0 / (1 + t0 ** (-dm * dm) + w)
    M = 1 + max(4 * m * L - 2, 0.0) * math.exp(-m * L)   # = 1 + sup_{d >= dm} (4 d^2 L - 2)_+ t0^{-d^2}
    return p / M

rows = []
for t0 in [2.0, 5.0, 11.0, 17.0, 100.0, 1e3, 1e6, 1e10]:
    L = math.log(t0)
    vs = np.linspace(0.01, 2.0, 2000)
    R = np.array([B1_depth(t0, v) or 0.0 for v in vs])
    i = int(R.argmax())
    kap = KHI.get(t0, kappa_upper(t0))
    rows.append(dict(t0=t0, v0_opt=float(vs[i]), R=float(R[i]), R_over_kappa_hi=float(R[i] / kap),
                     kappa_hi=kap, asym_0p267_over_sqrtL=0.2668 / math.sqrt(L)))
    print(f"B1 t0={t0:g}: max R = {R[i]:.4f} at v0 = {vs[i]:.3f}; R/kappa_hi = {R[i]/kap:.4f}; 0.267/sqrt(L) = {0.2668/math.sqrt(L):.4f}")
OUT["B1"] = rows

# ---------------- Theorem B2, explicit T* ----------------
TS = np.logspace(2, 300, 900)
_KB = {}
def kbar(t0):
    """C1 upper bound at the grid point <= t0 (valid for all larger t0, since C1 is monotone in t0)."""
    g = TS[TS <= t0 * (1 + 1e-12)][-1] if t0 >= TS[0] else None
    g = float(g)
    if g not in _KB:
        _KB[g] = kappa_upper(g)
    return _KB[g]

def persist_ok(t0, x1, x2, y, a, b, beta, kb):
    if not (b < a < y and b * b < beta < a * a and t0 ** beta + 1 <= t0):
        return False
    e = 2 * t0 ** (beta - a * a)              # >= T t0^{-a^2}
    E = math.exp(-2 ** (-b * b) * t0 ** (beta - b * b))
    X = x2 + 1.0                               # >= (x + y')/1 for y' < 1
    P1 = y - 2 * X * kb * e >= a
    P2 = x2 * kb + X * kb * e <= b
    P3 = y + x1 / 2 - 1 >= 2 * X * e + (x2 / 2) * E
    return P1 and P2 and P3

def race_ok(t0, x1, x2, y, ub, vl, beta, kb):
    # merge for u0 = y' kappa, y' <= y, v0 = x kappa, x in [x1, x2]; requires y kb < ub, ub^2 < beta < vl^2, vl < x1
    if not (ub * ub < beta < vl * vl and vl < x1 and t0 ** beta + 1 <= t0):
        return False
    e = 2 * t0 ** (beta - vl * vl)            # >= T t0^{-vl^2}
    X = x2 + y
    Q1 = y * kb + X * kb * e <= ub
    Q2 = x1 * (1 - 2 * e) >= vl
    Q3 = y * kb * math.exp(-2 ** (-ub * ub) * t0 ** (beta - ub * ub)) + X * kb * e + kb * (x2 / 2 + y / 4) < 1
    return Q1 and Q2 and Q3

def find_Tstar(ok, params_grid, lo=1e2, hi=1e300):
    best = None
    for prm in params_grid:
        # find smallest t0 on a log grid where ok holds (monotone in t0 for fixed params)
        for t0 in TS:
            if ok(t0, *prm, kbar(t0)):
                if best is None or t0 < best[0]:
                    best = (float(t0), prm)
                break
    return best

def grid_persist(x1, x2, y):
    for a in np.linspace(x2 + 0.01, y - 0.005, 12):
        for b in np.linspace(x2 * 1.0001, a - 0.005, 8):
            for f in (0.25, 0.5, 0.75):
                beta = b * b + f * (a * a - b * b)
                yield (x1, x2, y, float(a), float(b), float(beta))

def grid_race(x1, x2, y):
    for vl in np.linspace(y + 0.01, x1 - 0.002, 10):
        for ub in np.linspace(y * 1.0001, vl - 0.005, 8):
            for f in (0.25, 0.5, 0.75):
                beta = ub * ub + f * (vl * vl - ub * ub)
                yield (x1, x2, y, float(ub), float(vl), float(beta))

B2 = []
for (x1, x2, y) in [(0.30, 0.50, 0.90), (0.40, 0.60, 0.85), (0.55, 0.75, 0.80)]:
    r = find_Tstar(persist_ok, list(grid_persist(x1, x2, y)))
    print(f"B2 persist: x in [{x1},{x2}], y >= {y}: T* = {r[0]:.3g} with (a,b,beta) = {tuple(round(z,4) for z in r[1][3:])}" if r else f"B2 persist {x1,x2,y}: none")
    B2.append(dict(kind="persist", x1=x1, x2=x2, y=y, Tstar=r[0] if r else None, params=r[1][3:] if r else None,
                   kbar_at_Tstar=kbar(r[0]) if r else None))
for (x1, x2, y) in [(0.85, 0.95, 0.50), (0.80, 0.95, 0.65)]:
    r = find_Tstar(race_ok, list(grid_race(x1, x2, y)))
    print(f"B2 race-merge: x in [{x1},{x2}], y <= {y}: T* = {r[0]:.3g} with (ub,vl,beta) = {tuple(round(z,4) for z in r[1][3:])}" if r else f"B2 race {x1,x2,y}: none")
    B2.append(dict(kind="race_merge", x1=x1, x2=x2, y=y, Tstar=r[0] if r else None, params=r[1][3:] if r else None))
OUT["B2"] = B2
json.dump(OUT, open("constants.json", "w"), indent=1)
