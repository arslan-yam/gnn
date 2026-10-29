"""Full node-level Gaussian-attention dynamics on arbitrary graphs (vector features).

    z_i <- sum_{j in N[i]} alpha_ij z_j,  alpha_ij ~ exp(-||z_i - z_j||^2 log(t + t0) / c)

(z = W h: a fixed metric W is absorbed by a change of coordinates, see Proposition 4.)
Graphs are CSR arrays (indptr, indices) and must contain self-loops.
"""
import math

import numba
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def sbm(sizes, p_in, p_out, rng):
    """Undirected SBM with self-loops. Returns (indptr, indices, labels)."""
    n = int(sum(sizes))
    y = np.repeat(np.arange(len(sizes)), sizes)
    P = np.where(y[:, None] == y[None, :], p_in, p_out)
    U = rng.random((n, n))
    A = np.triu(U < P, 1)
    A = A | A.T
    np.fill_diagonal(A, True)
    S = csr_matrix(A)
    S.sort_indices()
    return S.indptr.astype(np.int64), S.indices.astype(np.int64), y


@numba.njit(cache=True, parallel=True)
def step(Z, indptr, indices, invT, out):
    n, p = Z.shape
    for i in numba.prange(n):
        s = 0.0
        for k in range(p):
            out[i, k] = 0.0
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            d2 = 0.0
            for k in range(p):
                diff = Z[i, k] - Z[j, k]
                d2 += diff * diff
            w = math.exp(-d2 * invT)
            s += w
            for k in range(p):
                out[i, k] += w * Z[j, k]
        for k in range(p):
            out[i, k] /= s


def run(Z0, indptr, indices, t0, c=1.0, T=3000, cooling=True, record=()):
    """Simulate T steps. record: iterable of times at which a copy of Z is stored."""
    Z = np.ascontiguousarray(Z0, dtype=np.float64).copy()
    buf = np.empty_like(Z)
    rec = {}
    record = set(int(r) for r in record)
    if 0 in record:
        rec[0] = Z.copy()
    for t in range(T):
        invT = (math.log(t + t0) if cooling else 1.0) / c
        step(Z, indptr, indices, invT, buf)
        Z, buf = buf, Z
        if t + 1 in record:
            rec[t + 1] = Z.copy()
    return Z, rec


def edge_dist(Z, indptr, indices):
    src = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    return src, indices, np.linalg.norm(Z[src] - Z[indices], axis=1)


def threshold_partition(Z, indptr, indices, tau):
    """Connected components of the subgraph of edges with ||z_i - z_j|| < tau."""
    src, dst, d = edge_dist(Z, indptr, indices)
    keep = d < tau
    n = len(indptr) - 1
    M = csr_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n, n))
    return connected_components(M, directed=False)[1]


def observed_partition(Z, indptr, indices, merged_tol=0.05, sep_tol=0.9):
    """Limiting partition read off at finite depth (units of sqrt c). Returns (labels, frac_unresolved)
    where unresolved edges have merged_tol <= distance <= sep_tol."""
    _, _, d = edge_dist(Z, indptr, indices)
    unres = float(((d >= merged_tol) & (d <= sep_tol)).mean())
    return threshold_partition(Z, indptr, indices, merged_tol), unres


def effective_rank(Z):
    """Roy-Vetterli effective rank exp(H(p)), p = singular values / sum (rows centred)."""
    X = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-12 * max(s.max(), 1e-300)] if s.size and s.max() > 0 else s[:0]
    if s.size == 0:
        return 1.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def hop_radius(indptr, indices, members):
    """Hop radius of the induced subgraph on `members` (min over centres of max BFS distance);
    returns inf if the induced subgraph is disconnected."""
    idx = {v: k for k, v in enumerate(members)}
    nb = [[idx[u] for u in indices[indptr[v]:indptr[v + 1]] if u in idx and u != v] for v in members]
    best = math.inf
    for s in range(len(members)):
        dist = [-1] * len(members)
        dist[s] = 0
        q = [s]
        for v in q:
            for u in nb[v]:
                if dist[u] < 0:
                    dist[u] = dist[v] + 1
                    q.append(u)
        if min(dist) < 0:
            return math.inf
        best = min(best, max(dist))
        if best <= 1:
            break
    return best


def planted_certificate(Z0, indptr, indices, y, t0, c=1.0):
    """Check the sufficient conditions of Theorem 5 (planted partition) for blocks y.
    Returns dict with the quantities and a boolean 'holds'."""
    n = Z0.shape[0]
    K = int(y.max()) + 1
    mu = np.stack([Z0[y == k].mean(axis=0) for k in range(K)])
    r = max(float(np.linalg.norm(Z0[y == k] - mu[k], axis=1).max()) for k in range(K))
    src = np.repeat(np.arange(n), np.diff(indptr))
    cross = y[src] != y[indices]
    pairs = set(zip(y[src][cross].tolist(), y[indices][cross].tolist()))
    Delta = min((float(np.linalg.norm(mu[a] - mu[b])) for a, b in pairs), default=math.inf)
    e_max = int(np.bincount(src[cross], minlength=n).max()) if cross.any() else 0
    rho = max(hop_radius(indptr, indices, np.where(y == k)[0]) for k in range(K))
    R0 = 2 * float(np.linalg.norm(Z0 - Z0.mean(axis=0), axis=1).max())
    cond_a = 16 * r * r * rho <= c
    sep = (Delta - 4 * r) ** 2 / c - 1 if Delta > 4 * r else -1
    cond_b = sep > 0
    cond_c = False
    if cond_b:
        if e_max == 0:
            cond_c = True
        else:
            for gam in np.linspace(sep / 50, sep, 50):
                if R0 * e_max * (t0 - 1) ** (-gam) / gam <= r:
                    cond_c = True
                    break
    return dict(r=r, Delta=Delta, e_max=e_max, rho=rho, R0=R0, cond_a=bool(cond_a), cond_b=bool(cond_b),
                cond_c=bool(cond_c), holds=bool(cond_a and cond_b and cond_c))


@numba.njit(cache=True, parallel=True)
def step_values(Z, H, indptr, indices, invT, outZ, outH):
    """Attention computed from keys/queries Z, applied to both Z and the values H."""
    n, p = Z.shape
    q = H.shape[1]
    for i in numba.prange(n):
        s = 0.0
        for k in range(p):
            outZ[i, k] = 0.0
        for k in range(q):
            outH[i, k] = 0.0
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            d2 = 0.0
            for k in range(p):
                diff = Z[i, k] - Z[j, k]
                d2 += diff * diff
            w = math.exp(-d2 * invT)
            s += w
            for k in range(p):
                outZ[i, k] += w * Z[j, k]
            for k in range(q):
                outH[i, k] += w * H[j, k]
        for k in range(p):
            outZ[i, k] /= s
        for k in range(q):
            outH[i, k] /= s


def run_values(Z0, H0, indptr, indices, t0, c=1.0, T=3000, record=()):
    """Metric attention: alpha from z = W h, values h. Since z = W h is preserved exactly
    (z' = W h'), passing Z0 = H0 W^T gives the learned-metric model."""
    Z = np.ascontiguousarray(Z0, dtype=np.float64).copy()
    H = np.ascontiguousarray(H0, dtype=np.float64).copy()
    bz, bh = np.empty_like(Z), np.empty_like(H)
    rec = {}
    record = set(int(r) for r in record)
    if 0 in record:
        rec[0] = (Z.copy(), H.copy())
    for t in range(T):
        step_values(Z, H, indptr, indices, math.log(t + t0) / c, bz, bh)
        Z, bz = bz, Z
        H, bh = bh, H
        if t + 1 in record:
            rec[t + 1] = (Z.copy(), H.copy())
    return Z, H, rec
