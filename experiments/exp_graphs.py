"""Experiments E4 (noisy community graphs) and E5 (learned metric) on the full node-level model.
Writes results/graphs_E4.json|npz, results/graphs_E5.json|npz.
"""
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score

from gnnlimit.core import kappa_estimate
from gnnlimit import full

T_START = time.monotonic()


def stamp(msg):
    print(f"[{time.monotonic() - T_START:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# E4: noisy SBM communities, vector features; predict the limiting partition
# ---------------------------------------------------------------------------
K4, NB, P_DIM, P_IN, DELTA = 4, 150, 8, 0.2, 3.0
T4 = 4000
T1S = (0, 5, 20)
SIGMAS = [0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
POUTS = [0.0, 0.005, 0.01, 0.02, 0.05]
SEEDS4 = range(5)


def e4_task(args):
    sigma, p_out, seed, kap = args
    rng = np.random.default_rng(1000 * seed + int(1e4 * sigma) + int(1e5 * p_out))
    indptr, indices, y = full.sbm([NB] * K4, P_IN, p_out, rng)
    n = len(y)
    t0 = n + 2.0
    mu = np.zeros((K4, P_DIM))
    mu[np.arange(K4), np.arange(K4)] = DELTA / math.sqrt(2)  # pairwise distance DELTA
    Z0 = mu[y] + sigma * rng.standard_normal((n, P_DIM))
    ZT, rec = full.run(Z0, indptr, indices, t0, T=T4, record=set(T1S) | {T4 // 2})
    obs, unres = full.observed_partition(ZT, indptr, indices)
    obs_half, _ = full.observed_partition(rec[T4 // 2], indptr, indices)
    out = dict(sigma=sigma, p_out=p_out, seed=seed, n_obs=int(obs.max() + 1), unresolved=unres,
               stable=float(adjusted_rand_score(obs, obs_half)), ari_gt=float(adjusted_rand_score(y, obs)),
               erank0=full.effective_rank(Z0), erankT=full.effective_rank(ZT))
    for t1 in T1S:
        pred = full.threshold_partition(rec[t1], indptr, indices, kap[t1])
        out[f"ari_t{t1}"] = float(adjusted_rand_score(pred, obs))
        out[f"n_pred_t{t1}"] = int(pred.max() + 1)
    cert = full.planted_certificate(Z0, indptr, indices, y, t0)
    out["cert_holds"] = cert["holds"]
    out["cert_r"] = cert["r"]
    out["cert_ok_if_holds"] = (not cert["holds"]) or bool(adjusted_rand_score(y, obs) == 1.0)
    return out


def run_E4():
    stamp("E4 noisy SBM: prediction of the limiting partition")
    n = K4 * NB
    kap = {t1: kappa_estimate(n + 2.0 + t1, T=10**6) for t1 in T1S}
    stamp(f"  kappa(t0 + t1) = {kap}")
    tasks = [(s, p, seed, kap) for s in SIGMAS for p in POUTS for seed in SEEDS4]
    with Pool(8) as pool:
        rows = pool.map(e4_task, tasks)
    stamp(f"  {len(rows)} runs")
    with open("results/graphs_E4.json", "w") as f:
        json.dump(dict(kappa=kap, K=K4, NB=NB, p=P_DIM, p_in=P_IN, Delta=DELTA, T=T4, rows=rows), f, indent=1)


# ---------------------------------------------------------------------------
# E5: learned metric decides which distinctions survive
# ---------------------------------------------------------------------------
K5, NB5 = 3, 120
DEPTHS5 = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
SEEDS5 = range(5)


def make_e5(seed):
    """Labels y (3 classes, homophilous SBM). Features: 2 signal dims (class means at mutual
    distance 1.0 -- below the merging threshold), 2 nuisance dims (binary nuisance factor
    independent of y, separation 3.0), 4 pure-noise dims; isotropic noise 0.15."""
    rng = np.random.default_rng(seed)
    indptr, indices, y = full.sbm([NB5] * K5, 0.15, 0.01, rng)
    n = len(y)
    s = rng.integers(0, 2, n)
    ang = 2 * np.pi * np.arange(K5) / K5
    sig = np.stack([np.cos(ang), np.sin(ang)], 1) / math.sqrt(3)  # pairwise distance 1.0
    X = np.zeros((n, 8))
    X[:, :2] = sig[y]
    X[:, 2] = 3.0 * s
    X += 0.15 * rng.standard_normal((n, 8))
    perm = rng.permutation(n)
    tr, te = perm[: n // 5], perm[n // 5:]
    return indptr, indices, y, s, X, tr, te


def metric_calibrated(W, X, y, tr, kap, margin=2.0):
    """Theory-guided scaling (Section 6.3): scale W so that the smallest W-distance between
    training class means equals margin * kappa (in units of sqrt c)."""
    mus = np.stack([X[tr][y[tr] == k].mean(0) for k in range(K5)])
    dmin = min(np.linalg.norm(W @ (mus[a] - mus[b])) for a in range(K5) for b in range(a + 1, K5))
    return W * (margin * kap / dmin)


def e5_task(args):
    seed, kap = args
    indptr, indices, y, s, X, tr, te = make_e5(seed)
    n = len(y)
    t0 = n + 2.0
    lda = LinearDiscriminantAnalysis(n_components=2).fit(X[tr], y[tr])
    W_lda = lda.scalings_[:, :2].T  # 2 x 8
    W_nuis = np.zeros((2, 8))
    W_nuis[0, 2] = 1.0
    W_nuis[1, 3] = 1.0
    metrics = {
        "identity": np.eye(8),
        "learned (LDA, calibrated)": metric_calibrated(W_lda, X, y, tr, kap),
        "nuisance projection": W_nuis,
        "learned (LDA, too small)": metric_calibrated(W_lda, X, y, tr, kap, margin=0.4),
    }
    res = {}
    for name, W in metrics.items():
        _, _, rec = full.run_values(X @ W.T, X, indptr, indices, t0, T=max(DEPTHS5), record=DEPTHS5)
        acc, er, ncl = [], [], []
        for d in DEPTHS5:
            Z, H = rec[d]
            clf = LogisticRegression(max_iter=2000).fit(H[tr], y[tr])
            acc.append(float(clf.score(H[te], y[te])))
            er.append(full.effective_rank(H))
            ncl.append(int(full.observed_partition(H, indptr, indices)[0].max() + 1))
        res[name] = dict(acc=acc, erank=er, nclusters=ncl)
    return res


def run_E5():
    stamp("E5 learned metric")
    kap = kappa_estimate(K5 * NB5 + 2.0, T=10**6)
    with Pool(5) as pool:
        out = pool.map(e5_task, [(seed, kap) for seed in SEEDS5])
    with open("results/graphs_E5.json", "w") as f:
        json.dump(dict(kappa=kap, depths=DEPTHS5, runs=out), f, indent=1)
    for name in out[0]:
        a = np.array([o[name]["acc"] for o in out])
        stamp(f"  {name:28s} acc@0 {a[:, 0].mean():.3f}  acc@64 {a[:, DEPTHS5.index(64)].mean():.3f}  acc@2048 {a[:, -1].mean():.3f}")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["E4", "E5"]:
        {"E4": run_E4, "E5": run_E5}[w]()
    stamp("done")
