"""Fully rigorous (mpmath interval arithmetic) verification of the certified statements used in
the paper's computer-assisted theorems. Units of sqrt(c); schedule T_t = 1/log(t + t0).

Run:  python -m gnnlimit.rigorous_mp   (prints a verification log; exits nonzero on any failure)
"""
import json
import sys

from mpmath import iv, mpf

iv.prec = 160


def weight(g, L):
    return iv.exp(-(g * g) * L)


def chain_step(g, t, t0):
    """Exact gap update of the reduced chain (see core.py), evaluated in interval arithmetic."""
    n = len(g)
    L = iv.log(iv.mpf(t) + t0)
    w = [weight(gl, L) for gl in g]
    zero = iv.mpf(0)
    out = []
    for l in range(n):
        wm = w[l - 1] if l > 0 else zero
        wp = w[l + 1] if l < n - 1 else zero
        Dl = 1 + wm + w[l]
        Dr = 1 + w[l] + wp
        N = (1 - w[l]) * (1 + w[l]) + wm + wp + wm * wp
        v = g[l] * N / (Dl * Dr)
        if l < n - 1:
            v = v + wp * g[l + 1] / Dr
        if l > 0:
            v = v + wm * g[l - 1] / Dl
        # gaps are provably >= 0 (comparison lemma); intersect with [0, inf)
        lo = max(v.a, mpf(0))
        out.append(iv.mpf([lo, v.b]))
    return out


def tail(s, delta):
    """Interval upper-bound expression for sum_{u>=0} (s+u)^(-delta^2) <= (s-1)^(1-delta^2)/(delta^2-1)."""
    e = delta * delta - 1
    return iv.exp(-e * iv.log(s - 1)) / e


def persist_ok(g, s, delta):
    """Certificate C1: log g - (1 + 1/(1-q)) tail >= log delta, q = s^(-delta^2)."""
    q = iv.exp(-(delta * delta) * iv.log(s))
    expr = iv.log(g) - (1 + 1 / (1 - q)) * tail(s, delta) - iv.log(delta)
    return expr.a >= 0


def best_level(g, s):
    if g.a <= 1:
        return None
    best = None
    for k in range(1, 100):
        delta = iv.mpf(1 + (float(g.a) - 1) * k / 100.0)
        if persist_ok(g, s, delta):
            best = delta
    return best


def classify(g, s):
    """Return list of labels (1 persist / 0 merge) if the partition is certified, else None."""
    n = len(g)
    lev = [best_level(gl, s) for gl in g]
    labels = [None] * n
    l = 0
    while l < n:
        if lev[l] is not None:
            labels[l] = 1
            l += 1
            continue
        j = l
        diam = iv.mpf(0)
        while j < n and lev[j] is None:
            diam = diam + g[j]
            j += 1
        k = j - l + 1
        qB = k // 2
        deltas = []
        if l > 0:
            deltas.append(lev[l - 1])
        if j < n:
            deltas.append(lev[j])
        push = iv.mpf(0)
        if deltas:
            delta = min(deltas, key=lambda d: d.a)
            push = 2 * (delta + 1) * tail(s, delta)
        if (diam + push - 1 / iv.sqrt(qB)).b > 0:
            return None
        for u in range(l, j):
            labels[u] = 0
        l = j
    return labels


def certify(g0_str, t0, Tmax=5000):
    g = [iv.mpf(x) for x in g0_str]
    t0 = iv.mpf(t0)
    for t in range(Tmax):
        g = chain_step(g, t, t0)
        lab = classify(g, t + 1 + t0)
        if lab is not None:
            return lab, t + 1, [(float(x.a), float(x.b)) for x in g]
    return None, Tmax, [(float(x.a), float(x.b)) for x in g]


def two_layer_certify(d0_str, t0, Tmax):
    d = iv.mpf(d0_str)
    t0 = iv.mpf(t0)
    next_check = 1
    for t in range(Tmax):
        L = iv.log(iv.mpf(t) + t0)
        a = iv.exp(-(d * d) * L)
        d = d * (1 - a) / (1 + a)
        if d.b <= 1:
            return 0, t + 1
        if t + 1 >= next_check:
            if best_level(d, t + 1 + t0) is not None:
                return 1, t + 1
            next_check = t + 1 + max(1, (t + 1) // 4)
    return -1, Tmax


def main():
    log = {"three_layer": [], "five_layer": [], "fail": 0}
    t0 = 11  # r = 3, m = 3: t0 = rm + 2
    cases = [
        (("1.4", "0.1"), [0, 0]),
        (("1.4", "0.5"), [1, 0]),
        (("1.4", "0.95"), [1, 0]),
        (("1.4", "1.5"), [0, 1]),
        (("1.4", "3.0"), [0, 1]),
        (("1.2", "0.95"), [1, 0]),
    ]
    for g0, expect in cases:
        lab, T, enc = certify(g0, t0)
        ok = lab == expect
        log["fail"] += int(not ok)
        print(f"r=3 t0={t0} g0={g0}: labels={lab} (expected {expect}) certified at T={T}  {'OK' if ok else 'FAIL'}")
        log["three_layer"].append({"g0": g0, "labels": lab, "T": T, "enclosure": enc, "ok": ok})
    # the isolated two-layer threshold with the same schedule, kappa(11) in [1.538, 1.540]
    for d0, expect in (("1.538", 0), ("1.540", 1)):
        st, T = two_layer_certify(d0, t0, 400000)
        ok = st == expect
        log["fail"] += int(not ok)
        print(f"two-layer t0={t0} d0={d0}: status={st} (expected {expect}) at T={T}  {'OK' if ok else 'FAIL'}")
        log.setdefault("kappa11", []).append({"d0": d0, "status": st, "T": T, "ok": ok})
    with open("results/rigorous_mp.json", "w") as f:
        json.dump(log, f, indent=1)
    sys.exit(1 if log["fail"] else 0)


if __name__ == "__main__":
    main()
