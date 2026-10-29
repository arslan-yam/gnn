"""Validated simulation of layered Gaussian-attention dynamics under logarithmic cooling.

Units: features are measured in units of sqrt(c), so the schedule is T_t = 1/log(t + t0)
and an unnormalised attention weight across a gap g is w = (t + t0)^(-g^2).

Reduced chain model (layer-constant features, r layers, gaps g_1..g_{r-1} >= 0):
    D_l      = 1 + w_{l-1} + w_l                      (w_0 = w_r = 0)
    g_l'     = g_l (A B - w_l^2) / (D_l D_{l+1}) + w_{l+1} g_{l+1} / D_{l+1} + w_{l-1} g_{l-1} / D_l,
    A = 1 + w_{l-1},  B = 1 + w_{l+1}.
Every term is nonnegative, which keeps interval evaluation tight.

Interval arithmetic: float64 with outward rounding (nextafter after every operation;
exp/log widened by EXP_PAD ulps to cover libm error). Results are enclosures of the exact
real trajectory.  Certificates (Section 4 of the paper):
  * gap persistence (C1): log g_T - (1 + 1/(1-q)) (s_T-1)^(1-delta^2)/(delta^2-1) >= log delta
    with delta > 1, q = s_T^(-delta^2), s_T = T + t0. Valid for any gap of any chain because
    each gap dominates the isolated two-layer recursion (comparison lemma).
  * block merging (C2): a maximal run of layers between certified gaps (min certified level
    delta) merges if  D_B(T) + 2 (delta+1) (s_T-1)^(1-delta^2)/(delta^2-1) <= 1/sqrt(q_B),
    q_B = ceil((k_B - 1)/2), k_B the number of layers in the block.
"""
import math

import numba
import numpy as np

EXP_PAD = 4
INF = np.inf


@numba.njit(inline="always")
def dn(x):
    return np.nextafter(x, -INF)


@numba.njit(inline="always")
def up(x):
    return np.nextafter(x, INF)


@numba.njit(inline="always")
def pad_dn(x, k):
    for _ in range(k):
        x = np.nextafter(x, -INF)
    return x


@numba.njit(inline="always")
def pad_up(x, k):
    for _ in range(k):
        x = np.nextafter(x, INF)
    return x


@numba.njit(inline="always")
def i_log(x):
    # log of a positive point value, enclosure
    y = math.log(x)
    return pad_dn(y, EXP_PAD), pad_up(y, EXP_PAD)


@numba.njit(inline="always")
def i_weight(glo, ghi, Llo, Lhi):
    """Enclosure of exp(-g^2 L) for g in [glo, ghi] (glo >= 0), L in [Llo, Lhi] (> 0)."""
    xlo = dn(dn(glo * glo) * Llo)
    xhi = up(up(ghi * ghi) * Lhi)
    wlo = pad_dn(math.exp(-xhi), EXP_PAD)
    whi = pad_up(math.exp(-xlo), EXP_PAD)
    if wlo < 0.0:
        wlo = 0.0
    if whi > 1.0:
        whi = 1.0
    return wlo, whi


@numba.njit(cache=True)
def chain_step_iv(glo, ghi, t, t0, nglo, nghi, wlo, whi):
    """One step of the reduced chain in gap coordinates, interval version (in place into ng*)."""
    n = glo.shape[0]
    Llo, Lhi = i_log(t + t0)
    for l in range(n):
        wlo[l], whi[l] = i_weight(glo[l], ghi[l], Llo, Lhi)
    for l in range(n):
        wl_lo, wl_hi = wlo[l], whi[l]
        wm_lo = wlo[l - 1] if l > 0 else 0.0
        wm_hi = whi[l - 1] if l > 0 else 0.0
        wp_lo = wlo[l + 1] if l < n - 1 else 0.0
        wp_hi = whi[l + 1] if l < n - 1 else 0.0
        # D_l = 1 + w_{l-1} + w_l ; D_{l+1} = 1 + w_l + w_{l+1}
        Dl_lo = dn(dn(1.0 + wm_lo) + wl_lo)
        Dl_hi = up(up(1.0 + wm_hi) + wl_hi)
        Dr_lo = dn(dn(1.0 + wl_lo) + wp_lo)
        Dr_hi = up(up(1.0 + wl_hi) + wp_hi)
        # numerator N = (1 - w)(1 + w) + w_{l-1} + w_{l+1} + w_{l-1} w_{l+1}
        om_lo = dn(1.0 - wl_hi)
        om_hi = up(1.0 - wl_lo)
        if om_lo < 0.0:
            om_lo = 0.0
        N_lo = dn(dn(dn(om_lo * dn(1.0 + wl_lo)) + wm_lo) + wp_lo)
        N_lo = dn(N_lo + dn(wm_lo * wp_lo))
        N_hi = up(up(up(om_hi * up(1.0 + wl_hi)) + wm_hi) + wp_hi)
        N_hi = up(N_hi + up(wm_hi * wp_hi))
        t1_lo = dn(dn(glo[l] * N_lo) / up(Dl_hi * Dr_hi))
        t1_hi = up(up(ghi[l] * N_hi) / dn(Dl_lo * Dr_lo))
        t2_lo = 0.0
        t2_hi = 0.0
        if l < n - 1:
            t2_lo = dn(dn(wp_lo * glo[l + 1]) / Dr_hi)
            t2_hi = up(up(wp_hi * ghi[l + 1]) / Dr_lo)
        t3_lo = 0.0
        t3_hi = 0.0
        if l > 0:
            t3_lo = dn(dn(wm_lo * glo[l - 1]) / Dl_hi)
            t3_hi = up(up(wm_hi * ghi[l - 1]) / Dl_lo)
        nglo[l] = dn(dn(t1_lo + t2_lo) + t3_lo)
        nghi[l] = up(up(t1_hi + t2_hi) + t3_hi)


@numba.njit(inline="always")
def tail_sum(s, delta):
    """Upper bound for sum_{u >= 0} (s + u)^(-delta^2), via the integral from s - 1."""
    e = delta * delta - 1.0
    return math.exp(-e * math.log(s - 1.0)) / e


@numba.njit(cache=True)
def persist_level(glo_T, s):
    """Largest delta on a grid such that certificate C1 holds for a gap with lower bound glo_T
    at time with s_T = s. Returns 0.0 if no delta works. Uses a 1e-12 relative safety margin."""
    if glo_T <= 1.0:
        return 0.0
    best = 0.0
    lg = math.log(glo_T)
    for k in range(1, 200):
        delta = 1.0 + (glo_T - 1.0) * k / 200.0
        q = math.exp(-delta * delta * math.log(s))
        if q >= 1.0:
            continue
        loss = (1.0 + 1.0 / (1.0 - q)) * tail_sum(s, delta)
        if lg - loss * (1.0 + 1e-12) - 1e-12 >= math.log(delta):
            best = delta
    return best


@numba.njit(cache=True)
def classify_chain(glo, ghi, s, out_labels):
    """Try to certify the limiting partition at time with s = T + t0.
    out_labels[l] = 1 (gap persists) / 0 (gap closes). Returns True if every gap is certified."""
    n = glo.shape[0]
    lev = np.zeros(n)
    for l in range(n):
        lev[l] = persist_level(glo[l], s)
    # blocks: maximal runs of non-persisting gaps
    l = 0
    while l < n:
        if lev[l] > 0.0:
            out_labels[l] = 1
            l += 1
            continue
        j = l
        diam = 0.0
        while j < n and lev[j] == 0.0:
            diam = up(diam + ghi[j])
            j += 1
        # block contains layers l .. j (inclusive) -> k = j - l + 1 layers
        k = j - l + 1
        qB = (k - 1 + 1) // 2
        delta = INF
        if l > 0:
            delta = min(delta, lev[l - 1])
        if j < n:
            delta = min(delta, lev[j])
        push = 0.0
        if delta < INF:
            push = 2.0 * (delta + 1.0) * tail_sum(s, delta)
        if (diam + push) * (1.0 + 1e-12) > 1.0 / math.sqrt(qB):
            return False
        for u in range(l, j):
            out_labels[u] = 0
        l = j
    return True


@numba.njit(cache=True)
def certify_chain(g0, t0, Tmax, check_every):
    """Validated run from exact initial gaps g0 (float64, taken as exact).
    Returns (certified, labels, T_cert, glo, ghi)."""
    n = g0.shape[0]
    glo = g0.copy()
    ghi = g0.copy()
    nglo = np.empty(n)
    nghi = np.empty(n)
    wlo = np.empty(n)
    whi = np.empty(n)
    labels = -np.ones(n, dtype=np.int64)
    next_check = 1
    for t in range(Tmax):
        chain_step_iv(glo, ghi, t, t0, nglo, nghi, wlo, whi)
        for l in range(n):
            glo[l] = max(nglo[l], 0.0)
            ghi[l] = nghi[l]
        if t + 1 >= next_check:
            if classify_chain(glo, ghi, t + 1 + t0, labels):
                return True, labels, t + 1, glo, ghi
            next_check = t + 1 + min(check_every, max(1, (t + 1) // 4))
    return False, labels, Tmax, glo, ghi


@numba.njit(cache=True, parallel=True)
def certify_grid(G0, t0, Tmax, check_every):
    """G0: (B, n) initial gaps. Returns labels (B, n) with -1 for uncertified, and T_cert (B,)."""
    B, n = G0.shape
    L = -np.ones((B, n), dtype=np.int64)
    Tc = np.zeros(B, dtype=np.int64)
    for b in numba.prange(B):
        ok, lab, Tcert, _, _ = certify_chain(G0[b].copy(), t0, Tmax, check_every)
        if ok:
            for l in range(n):
                L[b, l] = lab[l]
        Tc[b] = Tcert
    return L, Tc


# ---------------------------------------------------------------------------
# plain float64 versions (fast, used for trajectories and finite-depth heuristics)
# ---------------------------------------------------------------------------
@numba.njit(cache=True)
def chain_step(g, t, t0):
    n = g.shape[0]
    L = math.log(t + t0)
    w = np.empty(n)
    for l in range(n):
        w[l] = math.exp(-g[l] * g[l] * L)
    out = np.empty(n)
    for l in range(n):
        wm = w[l - 1] if l > 0 else 0.0
        wp = w[l + 1] if l < n - 1 else 0.0
        Dl = 1.0 + wm + w[l]
        Dr = 1.0 + w[l] + wp
        N = (1.0 - w[l]) * (1.0 + w[l]) + wm + wp + wm * wp
        v = g[l] * N / (Dl * Dr)
        if l < n - 1:
            v += wp * g[l + 1] / Dr
        if l > 0:
            v += wm * g[l - 1] / Dl
        out[l] = v
    return out


@numba.njit(cache=True)
def chain_traj(g0, t0, T, cps):
    """Float trajectory, values recorded at the (sorted) checkpoint times cps."""
    g = g0.copy()
    out = np.empty((cps.shape[0], g0.shape[0]))
    k = 0
    if cps[0] == 0:
        out[0] = g
        k = 1
    for t in range(T):
        g = chain_step(g, t, t0)
        if k < cps.shape[0] and cps[k] == t + 1:
            out[k] = g
            k += 1
    return out


@numba.njit(cache=True, parallel=True)
def chain_final(G0, t0, T):
    B, n = G0.shape
    out = np.empty((B, n))
    for b in numba.prange(B):
        g = G0[b].copy()
        for t in range(T):
            g = chain_step(g, t, t0)
        out[b] = g
    return out


# ---------------------------------------------------------------------------
# two layers: exact scalar recursion d' = d (1 - a)/(1 + a), a = s^(-d^2)
# ---------------------------------------------------------------------------
@numba.njit(cache=True)
def persist_level_rho(glo_T, s, lam, P):
    """Certificate C1 for the two-block recursion d' = d (1 - P a^2)/((1+rA a)(1+rB a)),
    lam = rA + rB, P = rA rB:  -log(1 - phi) <= lam a + P a^2/(1 - P a^2)."""
    if glo_T <= 1.0:
        return 0.0
    best = 0.0
    lg = math.log(glo_T)
    for k in range(1, 200):
        delta = 1.0 + (glo_T - 1.0) * k / 200.0
        q = math.exp(-delta * delta * math.log(s))
        if P * q * q >= 1.0:
            continue
        loss = (lam + P * q / (1.0 - P * q * q)) * tail_sum(s, delta)
        if lg - loss * (1.0 + 1e-12) - 1e-12 >= math.log(delta):
            best = delta
    return best


@numba.njit(cache=True)
def two_block_certify(d0, t0, rA, rB, Tmax):
    """Validated run of the two-block recursion. Returns (status, T): 0 = certified merge
    (d_T <= 1), 1 = certified persistence, -1 = unknown. Requires rA * rB <= 1."""
    lo = d0
    hi = d0
    lam = rA + rB
    P = rA * rB
    next_check = 1
    for t in range(Tmax):
        Llo, Lhi = i_log(t + t0)
        alo, ahi = i_weight(lo, hi, Llo, Lhi)
        # psi(a) = (1 - P a^2)/((1+rA a)(1+rB a)) is decreasing in a; f = d psi(a(d)) increasing in d
        num_lo = dn(1.0 - up(P * up(ahi * ahi)))
        num_hi = up(1.0 - dn(P * dn(alo * alo)))
        den_lo = dn(dn(1.0 + dn(rA * alo)) * dn(1.0 + dn(rB * alo)))
        den_hi = up(up(1.0 + up(rA * ahi)) * up(1.0 + up(rB * ahi)))
        nlo = dn(dn(lo * max(num_lo, 0.0)) / den_hi)
        nhi = up(up(hi * num_hi) / den_lo)
        lo = max(nlo, 0.0)
        hi = nhi
        if hi <= 1.0:
            return 0, t + 1
        if t + 1 >= next_check:
            if persist_level_rho(lo, t + 1 + t0, lam, P) > 0.0:
                return 1, t + 1
            next_check = t + 1 + max(1, (t + 1) // 8)
    return -1, Tmax


@numba.njit(cache=True)
def two_layer_certify(d0, t0, Tmax):
    return two_block_certify(d0, t0, 1.0, 1.0, Tmax)


@numba.njit(cache=True)
def two_block_final(d0, t0, rA, rB, T):
    d = d0
    P = rA * rB
    for t in range(T):
        a = math.exp(-d * d * math.log(t + t0))
        d = d * (1.0 - P * a * a) / ((1.0 + rA * a) * (1.0 + rB * a))
    return d


@numba.njit(cache=True)
def two_layer_final(d0, t0, T):
    return two_block_final(d0, t0, 1.0, 1.0, T)


@numba.njit(cache=True)
def two_layer_traj(d0, t0, cps):
    T = cps[-1]
    out = np.empty(cps.shape[0])
    d = d0
    k = 0
    if cps[0] == 0:
        out[0] = d
        k = 1
    for t in range(T):
        a = math.exp(-d * d * math.log(t + t0))
        d = d * (1.0 - a) / (1.0 + a)
        if k < cps.shape[0] and cps[k] == t + 1:
            out[k] = d
            k += 1
    return out


def kappa_estimate(t0, rA=1.0, rB=1.0, T=10**6):
    """Finite-depth threshold: smallest d0 with d_T > 0.5 (bisection)."""
    a, b = 1.0, 4.0
    for _ in range(60):
        mid = 0.5 * (a + b)
        if two_block_final(mid, t0, rA, rB, T) > 0.5:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)


def kappa_enclosure(t0, rA=1.0, rB=1.0, Tmax=10**6, T_float=10**6):
    """Validated enclosure [klo, khi] of the critical gap kappa(t0; rA, rB) (units of sqrt c):
    klo is a certified-merge d0 and khi a certified-persist d0. Valid because d_inf is
    nondecreasing in d0. Returns (klo, khi, finite-depth estimate)."""
    est = kappa_estimate(t0, rA, rB, T_float)
    klo = khi = None
    h = 1e-7
    while klo is None or khi is None:
        if klo is None and two_block_certify(est - h, t0, rA, rB, Tmax)[0] == 0:
            klo = est - h
        if khi is None and two_block_certify(est + h, t0, rA, rB, Tmax)[0] == 1:
            khi = est + h
        h *= 1.5
    return klo, khi, est


def log_checkpoints(n_steps, n_points=200):
    t = np.unique(np.round(np.logspace(0, np.log10(n_steps), n_points)).astype(np.int64))
    return np.concatenate([[0], t[t <= n_steps]]).astype(np.int64)
