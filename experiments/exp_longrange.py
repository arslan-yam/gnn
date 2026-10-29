"""E7: a long-range node-classification task where depth is necessary.

Each of K classes is a ring lattice of n_b nodes (node i ~ i+-1, i+-2). Only a fraction f of nodes
("anchors") carry class information in their features (x = mu_y + sigma_a eps); all other nodes have
pure-noise features (x = sigma_n eps). Sparse random cross-class edges (each node w.p. p_x) create a
global mixing channel, so propagation that is too long oversmooths across classes.
A non-anchor node is typically ~1/(4f) hops from the nearest anchor of its class, so shallow models
cannot classify it; very deep linear propagation mixes the classes.

Usage: python -m experiments.exp_longrange probe | run   (env: ALT=1 alternative grid, RESUME=1 skip finished tasks)
Writes results/longrange_E7.json.
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

TASK = dict(K=3, n_b=200, f=0.02, p_x=0.05, sigma_a=0.3, sigma_n=1.0, d=16)
VARIANTS = {"sparse_cross": dict(TASK, p_x=0.05), "dense_cross": dict(TASK, p_x=0.2)}
DEPTHS = [2, 8, 16, 32, 64, 128]
MODELS = ["gcn", "gcn_res", "gcn_pairnorm", "gcnii", "appnp", "sgc", "gat", "gat_res", "grand",
          "gauss_const", "gauss_cool"]
SEEDS = range(5)
OUT = "results/longrange_E7.json"
ALT = os.environ.get("ALT") == "1"
if ALT:
    OUT = "results/longrange_E7_alt.json"


def make_longrange(seed, K, n_b, f, p_x, sigma_a, sigma_n, d):
    rng = np.random.default_rng(20_000 + seed)
    n = K * n_b
    y = np.repeat(np.arange(K), n_b)
    A = np.zeros((n, n), dtype=bool)
    for k in range(K):
        base = k * n_b
        for i in range(n_b):
            for s in (1, 2):
                a, b = base + i, base + (i + s) % n_b
                A[a, b] = A[b, a] = True
    for v in np.where(rng.random(n) < p_x)[0]:
        others = np.where(y != y[v])[0]
        u = rng.choice(others)
        A[u, v] = A[v, u] = True
    np.fill_diagonal(A, True)
    mu = rng.standard_normal((K, d))
    mu = 3.0 * mu / np.linalg.norm(mu, axis=1, keepdims=True)
    anchor = np.zeros(n, dtype=bool)
    for k in range(K):
        idx = np.where(y == k)[0]
        anchor[rng.choice(idx, max(1, int(round(f * n_b))), replace=False)] = True
    X = sigma_n * rng.standard_normal((n, d))
    X[anchor] = mu[y[anchor]] + sigma_a * rng.standard_normal((anchor.sum(), d))
    perm = rng.permutation(n)
    ntr, nva = int(0.2 * n), int(0.2 * n)
    return A, X.astype(np.float32), y, perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:], anchor


def hop_to_anchor(A, y, anchor):
    """BFS distance from each node to the nearest same-class anchor inside its own ring."""
    n = len(y)
    dist = np.full(n, -1)
    q = list(np.where(anchor)[0])
    dist[q] = 0
    for v in q:
        for u in np.where(A[v])[0]:
            if dist[u] < 0 and y[u] == y[v]:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


def task(args):
    model, K, seed, tuned = args[:4]
    variant = args[4] if len(args) > 4 else "sparse_cross"
    import torch
    torch.set_num_threads(1)
    from gnnlimit.models import Graph, fit, fit_tuned
    A, X, y, tr, va, te, anchor = make_longrange(seed, **VARIANTS[variant])
    G = Graph(A)
    if tuned:
        r = fit_tuned(model, K, X, y, tr, va, te, G, seed, alt=ALT)
    else:
        r = fit(model, K, X, y, tr, va, te, G, 0.01, 0.0, seed)
    r.update(model=model, depth=K, seed=seed, variant=variant)
    return r


def cost(t):
    from gnnlimit.models import ATTENTION_MODELS
    return t[1] * (10 if t[0] in ATTENTION_MODELS else 1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "probe":
        A, X, y, tr, va, te, anchor = make_longrange(0, **TASK)
        dist = hop_to_anchor(A, y, anchor)
        print("n", len(y), "edges", int(A.sum() - len(y)) // 2, "anchors", int(anchor.sum()),
              "hops to anchor: median", np.median(dist), "p90", np.percentile(dist, 90), "max", dist.max())
        tasks = [(m, K, 0, False) for m in ["mlp"] for K in [0]]
        tasks += [(m, K, 0, False) for m in ["sgc", "appnp", "gcnii", "gauss_cool", "gauss_const"]
                  for K in [2, 8, 32, 128]]
    else:
        tasks = [(m, K, s, True, v) for v in VARIANTS for m in MODELS for K in DEPTHS for s in SEEDS]
        tasks += [("mlp", 0, s, True, v) for v in VARIANTS for s in SEEDS]
    tasks.sort(key=cost, reverse=True)
    rows = []
    if mode == "run" and os.environ.get("RESUME") == "1" and os.path.exists(OUT):  # keep finished tasks
        with open(OUT) as f:
            rows = json.load(f)["rows"]
        done = {(r["model"], r["depth"], r["seed"], True, r["variant"]) for r in rows}
        tasks = [tk for tk in tasks if tk not in done]
        print(f"resuming: {len(rows)} done, {len(tasks)} to run", flush=True)
    t = time.monotonic()
    with Pool(int(os.environ.get("WORKERS", 8))) as pool:
        for i, row in enumerate(pool.imap_unordered(task, tasks)):
            rows.append(row)
            if mode == "probe":
                print(f"  {row['model']:12s} K={row['depth']:3d} val={row['val']:.3f} test={row['test']:.3f} "
                      f"{row['seconds']:.0f}s", flush=True)
            elif (i + 1) % 25 == 0 or i + 1 == len(tasks):
                print(f"[{time.monotonic() - t:7.1f}s] {i + 1}/{len(tasks)}", flush=True)
                with open(OUT, "w") as f:
                    json.dump(dict(variants=VARIANTS, rows=rows), f, indent=1)
    print(f"done in {time.monotonic() - t:.1f}s")
