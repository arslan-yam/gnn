"""E9: learned (query-key) attention.

Score f(x, y) = x^T A y - 1/2 y^T B y + b^T y (dot-product attention with biases: B = 0; Gaussian attention with a key
bias: B = A = A^T). Attention alpha_ij ~ exp(f(z_i, z_j) log(t + t0) / c) over the closed neighbourhood, so that the
unnormalised weight is s_t^{-m_ij / c} with the directed margin m_ij = f(z_i, z_i) - f(z_i, z_j).

  twoblock   critical margins of the two-block margin map (Theorem 5) in the plane of the two directed margins
  graphs     threshold-graph rule with directed margins on contextual SBMs under dot-product attention
  examples   the oscillating pair (Proposition 7) and the suspended node (Proposition 8)

Usage: python -m experiments.exp_learned [twoblock|graphs|trained ...]   (default: twoblock). Results go to results/learned_*.json.
"""
import json
import math
import os
import sys
import time

import numba
import numpy as np

from gnnlimit import full

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# two blocks
# ---------------------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def margin_step(MA, MB, s, theta, rhoA, rhoB):
    aA = s ** (-MA)
    aB = s ** (-MB)
    pA = rhoA * aA / (1.0 + rhoA * aA)
    pB = rhoB * aB / (1.0 + rhoB * aB)
    k = 1.0 - pA - pB
    d2 = MA + MB
    sym = 0.5 * theta * (pA + pB)
    return (k * (MA - d2 * (sym + (1.0 - theta) * pA)),
            k * (MB - d2 * (sym + (1.0 - theta) * pB)))


@numba.njit(cache=True)
def c1_certificate(MA, MB, s, lam):
    """Certificate C1' at offset s: True if the margins provably stay above some delta^2 > 1 forever."""
    m = min(MA, MB)
    if m <= 1.0:
        return False
    for f in (0.2, 0.4, 0.6, 0.8):
        d2 = 1.0 + f * (m - 1.0)
        R = (MA + MB) / d2
        q = s ** (-d2)
        if R * lam * q >= 1.0:
            continue
        loss = (1.0 + R) * lam / (1.0 - R * lam * q) * (s - 1.0) ** (1.0 - d2) / (d2 - 1.0)
        if math.log(m / d2) >= loss:
            return True
    return False


@numba.njit(cache=True)
def classify(MA, MB, theta, rhoA, rhoB, t0, Tmax):
    """+1: persists (certified by C1'), -1: merges (min margin <= 1, Theorem 5(i)), 0: unresolved after Tmax."""
    lam = rhoA + rhoB
    for t in range(Tmax):
        s = t + t0
        if min(MA, MB) <= 1.0:
            return -1
        if t % 64 == 0 and c1_certificate(MA, MB, s, lam):
            return 1
        MA, MB = margin_step(MA, MB, s, theta, rhoA, rhoB)
    return 0


@numba.njit(cache=True)
def critical_weak_margin(MY, theta, rhoX, rhoY, t0, Tmax, lo, hi, iters):
    """Smallest margin M_X of the weaker block from which the pair persists, partner margin MY fixed (bisection).
    Block X is block A of the map. Returns (lo, hi, n_unresolved)."""
    unres = 0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = classify(mid, MY, theta, rhoX, rhoY, t0, Tmax)
        if r == 0:
            unres += 1
            r = 1 if min(mid, MY) > 1.0 else -1  # trajectory stayed above 1 up to Tmax
        if r == 1:
            hi = mid
        else:
            lo = mid
    return lo, hi, unres


def run_twoblock():
    out = {"rows": []}
    Tmax = 2_000_000
    # symmetric critical margin: diagonal M_A = M_B, equals kappa(t0)^2 by Theorem 5(iv)
    for t0 in (11.0, 1e3):
        lo, hi = 1.0, 6.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            r = classify(mid, mid, 0.0, 1.0, 1.0, t0, Tmax)
            if r == 0:
                r = 1
            hi, lo = (mid, lo) if r == 1 else (hi, mid)
        ksq = hi
        log(f"t0={t0:g}: symmetric critical margin {ksq:.5f} (kappa = {math.sqrt(ksq):.5f})")
        for theta in (0.0, 1.0):
            for MY in np.concatenate([np.linspace(ksq, ksq + 1, 11), [ksq + 1.5, ksq + 2, ksq + 3, ksq + 5, ksq + 8]]):
                lo, hi, unres = critical_weak_margin(float(MY), theta, 1.0, 1.0, t0, Tmax, 1.0, float(MY), 36)
                L = math.log(t0)
                lamX = 1.0 * (1.0 + (1.0 - theta / 2) * (1.0 + MY))
                pred2 = 1 + (2 * math.log(L) - math.log(math.log(L)) + math.log(lamX / 2)) / L if L > math.e else float("nan")
                out["rows"].append(dict(t0=t0, theta=theta, MY=float(MY), MX_lo=lo, MX_hi=hi, unresolved=unres,
                                        kappa_sq=ksq, second_order_frozen_partner=pred2))
                log(f"  theta={theta:.0f} MY={MY:7.3f}: critical MX in [{lo:.5f}, {hi:.5f}]  unres {unres}")
    with open("results/learned_twoblock.json", "w") as f:
        json.dump(out, f, indent=1)


# ---------------------------------------------------------------------------------------------------------------
# graphs: dot-product attention f(x, y) = x^T A y + b^T y on contextual SBMs
# ---------------------------------------------------------------------------------------------------------------
@numba.njit(cache=True, parallel=True)
def qk_step(Z, R, indptr, indices, invT, out):
    """One layer. R[i] = A^T z_i + b is the query of node i, so f(z_i, z_j) = <R[i], z_j>."""
    n, p = Z.shape
    for i in numba.prange(n):
        mx = -1e300
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            f = 0.0
            for k in range(p):
                f += R[i, k] * Z[j, k]
            if f > mx:
                mx = f
        s = 0.0
        for k in range(p):
            out[i, k] = 0.0
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            f = 0.0
            for k in range(p):
                f += R[i, k] * Z[j, k]
            w = math.exp((f - mx) * invT)
            s += w
            for k in range(p):
                out[i, k] += w * Z[j, k]
        for k in range(p):
            out[i, k] /= s


def qk_run(Z0, A, b, indptr, indices, t0, T, record=()):
    Z = np.ascontiguousarray(Z0, dtype=np.float64).copy()
    buf = np.empty_like(Z)
    rec = {0: Z.copy()} if 0 in record else {}
    for t in range(T):
        R = np.ascontiguousarray(Z @ A + b)
        qk_step(Z, R, indptr, indices, math.log(t + t0), buf)
        Z, buf = buf, Z
        if t + 1 in record:
            rec[t + 1] = Z.copy()
    return Z, rec


def directed_margins(Z, A, b, src, dst):
    """m_ij = f(z_i, z_i) - f(z_i, z_j) = <A^T z_i + b, z_i - z_j> for the edges (src, dst)."""
    R = Z @ A + b
    return np.einsum("ek,ek->e", R[src], Z[src] - Z[dst])


def relative_margins(Z, A, b, indptr, indices):
    """m+_ij = max_{k in N[i]} f(z_i, z_k) - f(z_i, z_j) >= m_ij, for every CSR edge (i, j). Since
    s^{-m+_ij} / Nmax <= alpha_ij <= s^{-m+_ij}, this is the margin that sets the attention weight."""
    n = len(indptr) - 1
    src = np.repeat(np.arange(n), np.diff(indptr))
    R = Z @ A + b
    f = np.einsum("ek,ek->e", R[src], Z[indices])
    fmax = np.maximum.reduceat(f, indptr[:-1])
    return fmax[src] - f


def reverse_edges(indptr, indices):
    """rev[e] = index of the edge (j, i) for the CSR edge e = (i, j) of a symmetric graph with sorted rows."""
    from scipy.sparse import csr_matrix
    n = len(indptr) - 1
    E = csr_matrix((np.arange(1, len(indices) + 1, dtype=np.float64), indices, indptr), shape=(n, n))
    ET = E.T.tocsr()
    ET.sort_indices()
    assert np.array_equal(ET.indptr, indptr) and np.array_equal(ET.indices, indices)
    return ET.data.astype(np.int64) - 1


def margin_partition(Z, A, b, indptr, indices, tau2, rule):
    """Components of the edges predicted to merge: rule(m_ij, m_ji) < tau2 (tau2 = kappa^2, a margin).
    Rules: min / max / sym (mean) of the two self-margins m_ij, m_ji; minp: min of the relative margins m+."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(indptr) - 1
    src = np.repeat(np.arange(n), np.diff(indptr))
    dst = indices
    if rule == "minp":
        mp = relative_margins(Z, A, b, indptr, indices)
        val = np.minimum(mp, mp[reverse_edges(indptr, indices)])
    else:
        mij = directed_margins(Z, A, b, src, dst)
        mji = directed_margins(Z, A, b, dst, src)
        val = {"min": np.minimum(mij, mji), "max": np.maximum(mij, mji), "sym": 0.5 * (mij + mji)}[rule]
    keep = val < tau2
    M = csr_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n, n))
    return connected_components(M, directed=False)[1]


GK, GNB, GP, GPIN, GPOUT, GDELTA, GT = 4, 150, 8, 0.2, 0.01, 3.0, 4000
G_SIGMAS = [0.1, 0.2, 0.3]
G_SHIFTS = [0.0, 1.0, 1.5, 2.0, 3.0]
G_SEEDS = range(5)
G_T1 = (0, 5, 20)


def graph_task(sigma, shift, untied, seed, kap):
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(10_000 * seed + int(1000 * sigma) + int(100 * shift) + 7 * untied)
    indptr, indices, y = full.sbm([GNB] * GK, GPIN, GPOUT, rng)
    n = len(y)
    t0 = n + 2.0
    mu = np.zeros((GK, GP))
    mu[np.arange(GK), np.arange(GK)] = GDELTA / math.sqrt(2)
    # shift towards block 0: the key potential |z|^2 then favours block 0, and at the level of block means the margin
    # of every other block towards block 0 is DELTA^2 - sqrt(2) DELTA shift (both directions are DELTA^2 at shift 0)
    v = np.zeros(GP)
    v[0] = shift
    Z0 = mu[y] + sigma * rng.standard_normal((n, GP)) + v
    # A = 2I makes the symmetric part of every margin equal to the squared distance, as in the Gaussian model (E4)
    A = 2.0 * np.eye(GP)
    if untied:
        G = rng.standard_normal((GP, GP))
        A = A + 0.3 * G / np.linalg.norm(G, 2)
    b = np.zeros(GP)
    ZT, rec = qk_run(Z0, A, b, indptr, indices, t0, GT, record=set(G_T1) | {GT // 2, GT - 1})
    osc = float(np.abs(ZT - rec[GT - 1]).max())
    obs, unres = full.observed_partition(ZT, indptr, indices)
    obs_half, _ = full.observed_partition(rec[GT // 2], indptr, indices)
    src = np.repeat(np.arange(n), np.diff(indptr))
    nonloop = src != indices
    row = dict(sigma=sigma, shift=shift, untied=untied, seed=seed, n_obs=int(obs.max() + 1), unresolved=unres,
               stable=float(adjusted_rand_score(obs, obs_half)), last_step=osc,
               ari_labels=float(adjusted_rand_score(y, obs)),
               min_eig_S=float(np.linalg.eigvalsh((A + A.T) / 2).min()))
    for t1 in G_T1:
        for rule in ("minp", "min", "sym", "max"):
            pred = margin_partition(rec[t1], A, b, indptr, indices, kap[t1] ** 2, rule)
            row[f"{rule}_ari_t{t1}"] = float(adjusted_rand_score(pred, obs))
            row[f"{rule}_n_t{t1}"] = int(pred.max() + 1)
            if t1 == 0:
                e = nonloop
                ps = pred[src[e]] != pred[indices[e]]  # predicted to persist (different predicted clusters)
                os_ = obs[src[e]] != obs[indices[e]]
                row[f"{rule}_false_persist"] = float((ps & ~os_).sum() / max(ps.sum(), 1))
                row[f"{rule}_false_merge"] = float((~ps & os_).sum() / max((~ps).sum(), 1))
    return row


def run_graphs():
    from gnnlimit.core import kappa_estimate
    n = GK * GNB
    kap = {t1: kappa_estimate(n + 2.0 + t1, T=10**6) for t1 in G_T1}
    log(f"kappa(t0 + t1) = {kap}")
    rows = []
    for untied in (0, 1):
        for sigma in G_SIGMAS:
            for shift in G_SHIFTS:
                for seed in G_SEEDS:
                    r = graph_task(sigma, shift, untied, seed, kap)
                    rows.append(r)
                log(f"untied={untied} sigma={sigma} shift={shift}: n_obs "
                    f"{np.mean([r['n_obs'] for r in rows[-5:]]):.1f}, ARI min/sym/max "
                    + " / ".join(f"{np.mean([r[f'{k}_ari_t0'] for r in rows[-5:]]):.3f}" for k in ("minp", "min", "sym", "max"))
                    + f", false persist (minp) {np.mean([r["minp_false_persist"] for r in rows[-5:]]):.4f}"
                    + f", stable {np.mean([r['stable'] for r in rows[-5:]]):.3f}, last step "
                    f"{max(r['last_step'] for r in rows[-5:]):.1e}")
    with open("results/learned_graphs.json", "w") as f:
        json.dump(dict(kappa=kap, rows=rows), f, indent=1)


# ---------------------------------------------------------------------------------------------------------------
# trained: the cooled dot-product networks of exp_train_qk.py, run on to 4000 layers
# ---------------------------------------------------------------------------------------------------------------
TR_T, TR_T1 = 4000, (0, 5, 20)


def trained_task(exp, group, K, seed, row, kap):
    """Rerun the selected network's own dynamics (z <- Att(z) z with its J and c, t0 = 11) to TR_T layers."""
    from scipy.sparse import csr_matrix
    from sklearn.metrics import adjusted_rand_score
    from experiments import exp_longrange
    from experiments import exp_train
    from experiments.exp_train_qk import state_path
    st = np.load(state_path(exp, group, K, seed, row["alt"]))
    if exp == "E6b":
        A, X, y, tr, va, te = exp_train.make_data(group, seed)
    else:
        A, X, y, tr, va, te, _ = exp_longrange.make_longrange(seed, **exp_longrange.VARIANTS[group])
    S = csr_matrix(A)
    S.sort_indices()
    indptr, indices = S.indptr.astype(np.int64), S.indices.astype(np.int64)
    c, J = float(st["c"]), st["J"]
    b = np.zeros(J.shape[0])
    Z0 = st["z0"] / math.sqrt(c)  # margins in units of c
    rec_t = set(TR_T1) | {K, TR_T // 2, TR_T - 1}
    ZT, rec = qk_run(Z0, J, b, indptr, indices, 11.0, TR_T, record=rec_t)

    def acc(Z):
        logits = (math.sqrt(c) * Z) @ st["Wout"].T + st["bout"]
        return float((logits.argmax(1)[te] == y[te]).mean())

    n = len(y)
    src = np.repeat(np.arange(n), np.diff(indptr))
    nonloop = src != indices
    d0 = np.linalg.norm(Z0[src] - Z0[indices], axis=1)[nonloop]
    scale = float(np.median(d0)) if d0.size else 1.0
    dT = np.linalg.norm(ZT[src] - ZT[indices], axis=1)

    def partition(Z, tol):
        d = np.linalg.norm(Z[src] - Z[indices], axis=1)
        keep = d < tol
        M = csr_matrix((np.ones(keep.sum()), (src[keep], indices[keep])), shape=(n, n))
        from scipy.sparse.csgraph import connected_components
        return connected_components(M, directed=False)[1]

    tol = 1e-6 * scale
    obs = partition(ZT, tol)
    obs_half = partition(rec[TR_T // 2], tol)
    Js = (J + J.T) / 2
    out = dict(exp=exp, group=group, depth=K, seed=seed, alt=row["alt"], test_reported=row["test"],
               test_at_K=acc(rec[K]), test_at_T=acc(ZT), n_obs=int(obs.max() + 1),
               unresolved=float(((dT >= tol) & (dT <= 1e-2 * scale))[nonloop].mean()),
               stable=float(adjusted_rand_score(obs, obs_half)), ari_labels=float(adjusted_rand_score(y, obs)),
               last_step=float(np.abs(ZT - rec[TR_T - 1]).max()),
               J_sym_min=float(np.linalg.eigvalsh(Js).min()), J_sym_max=float(np.linalg.eigvalsh(Js).max()),
               J_antisym_norm=float(np.linalg.norm((J - J.T) / 2, 2)), c=c)
    for t1 in TR_T1:
        for rule in ("minp", "sym"):
            pred = margin_partition(rec[t1], J, b, indptr, indices, kap[t1] ** 2, rule)
            out[f"{rule}_ari_t{t1}"] = float(adjusted_rand_score(pred, obs))
            out[f"{rule}_n_t{t1}"] = int(pred.max() + 1)
            ps = pred[src[nonloop]] != pred[indices[nonloop]]
            os_ = obs[src[nonloop]] != obs[indices[nonloop]]
            out[f"{rule}_false_persist_t{t1}"] = float((ps & ~os_).sum() / max(ps.sum(), 1))
            out[f"{rule}_false_merge_t{t1}"] = float((~ps & os_).sum() / max((~ps).sum(), 1))
    return out


def run_trained():
    from gnnlimit.core import kappa_estimate
    from gnnlimit.summarize_train import load_all
    from experiments import exp_train_qk
    kap = {t1: kappa_estimate(11.0 + t1, T=10**6) for t1 in TR_T1}
    log(f"kappa(11 + t1) = {kap}")
    rows = []
    for exp, key in (("E6b", "regime"), ("E7", "variant")):
        sel = [r for r in load_all(exp) if r["model"] == "qk_cool"]  # the run reported in Fig 6 and the tables
        for r in sorted(sel, key=lambda r: (r[key], r["depth"], r["seed"])):
            if not os.path.exists(exp_train_qk.state_path(exp, r[key], r["depth"], r["seed"], r["alt"])):
                continue
            o = trained_task(exp, r[key], r["depth"], r["seed"], r, kap)
            rows.append(o)
            log(f"{exp} {r[key]:18s} K={r['depth']:3d} s={r['seed']} alt={int(r['alt'])}: test {o['test_reported']:.3f} "
                f"(rerun {o['test_at_K']:.3f}, at {TR_T} {o['test_at_T']:.3f}), clusters {o['n_obs']}, "
                f"ARI minp {o['minp_ari_t0']:.2f}/{o['minp_ari_t5']:.2f}/{o['minp_ari_t20']:.2f} "
                f"sym {o['sym_ari_t0']:.2f}/{o['sym_ari_t20']:.2f}, Js in [{o['J_sym_min']:.2f}, {o['J_sym_max']:.2f}]")
    with open("results/learned_trained.json", "w") as f:
        json.dump(dict(kappa=kap, rows=rows), f, indent=1)


if __name__ == "__main__":
    modes = sys.argv[1:] or ["twoblock"]
    for m in modes:
        globals()[f"run_{m}"]()
