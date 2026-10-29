"""E6: trained deep GNNs on contextual SBMs (node classification) at depths 2..128.

Models (hidden width 16, linear input embedding, linear readout, matched training budget):
  gcn        K layers  h <- ReLU(A_hat h W_l)
  gat        K layers  single-head GAT, ELU
  sgc        h_K = A_hat^K (X W_in)                      (linear propagation)
  gauss_const  z <- Attn_T(z) z,  alpha ~ exp(-||z_i - z_j||^2 / T),     T learnable
  gauss_cool   same with T_l = c / log(l + t0),  c learnable, t0 = 11   (the model of the paper)
  mlp          depth 0 reference (features only)
Regimes:  homophilous with clean features / homophilous with noisy features / heterophilous.
Writes results/train_E6.json.
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HID, EPOCHS, LR, WD, T0_COOL = 16, 200, 0.01, 5e-4, 11.0
DEPTHS = [2, 8, 16, 32, 64, 128]
MODELS = ["gcn", "gat", "sgc", "gauss_const", "gauss_cool"]
REGIMES = {
    #                     K   n_b  p_in   p_out  feat_dim  mu_sep  noise/dim
    "homophilous_clean": (3, 150, 0.05, 0.005, 16, 3.0, 0.25),
    "homophilous_noisy": (3, 150, 0.05, 0.005, 16, 1.5, 1.0),
    "heterophilous": (3, 150, 0.005, 0.04, 16, 3.0, 1.0),
}
SEEDS = range(5)


def make_data(regime, seed):
    K, nb, p_in, p_out, fd, sep, noise = REGIMES[regime]
    rng = np.random.default_rng(10_000 + seed)
    n = K * nb
    y = np.repeat(np.arange(K), nb)
    P = np.where(y[:, None] == y[None, :], p_in, p_out)
    A = np.triu(rng.random((n, n)) < P, 1)
    A = A | A.T
    np.fill_diagonal(A, True)
    mu = rng.standard_normal((K, fd))
    mu = mu / np.linalg.norm(mu, axis=1, keepdims=True) * sep / math.sqrt(2)
    X = mu[y] + noise * rng.standard_normal((n, fd))
    perm = rng.permutation(n)
    ntr, nva = int(0.2 * n), int(0.2 * n)
    return A, X.astype(np.float32), y, perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]


def build(model, K, n_feat, n_cls, src, dst, n):
    import torch
    import torch.nn as nn

    deg = torch.zeros(n).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    gcn_w = (deg[src] * deg[dst]).rsqrt()

    def spmm(w, h):
        return torch.zeros(n, h.shape[1]).index_add_(0, dst, w[:, None] * h[src])

    def seg_softmax(logit):
        mx = torch.full((n,), -1e30).scatter_reduce(0, dst, logit, reduce="amax", include_self=True)
        e = torch.exp(logit - mx[dst])
        s = torch.zeros(n).index_add_(0, dst, e)
        return e / s[dst]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_feat, HID)
            self.out = nn.Linear(HID, n_cls)
            if model == "gcn":
                self.W = nn.ModuleList([nn.Linear(HID, HID) for _ in range(K)])
            if model == "gat":
                self.W = nn.ModuleList([nn.Linear(HID, HID, bias=False) for _ in range(K)])
                self.a = nn.ParameterList([nn.Parameter(0.1 * torch.randn(2 * HID)) for _ in range(K)])
            if model == "gauss_const":
                self.logT = nn.Parameter(torch.tensor(0.0))
            if model == "gauss_cool":
                self.logc = nn.Parameter(torch.tensor(0.0))

        def propagate(self, h, K_run=None):
            K_ = K if K_run is None else K_run
            for l in range(K_):
                if model == "gcn":
                    h = spmm(gcn_w, self.W[l](h))
                    h = torch.relu(h) if l < K_ - 1 else h
                elif model == "sgc":
                    h = spmm(gcn_w, h)
                elif model == "gat":
                    g = self.W[l](h)
                    a = self.a[l]
                    logit = nn.functional.leaky_relu((g[src] * a[:HID]).sum(1) + (g[dst] * a[HID:]).sum(1), 0.2)
                    h = spmm(seg_softmax(logit), g)
                    h = nn.functional.elu(h) if l < K_ - 1 else h
                else:
                    if model == "gauss_const":
                        invT = torch.exp(-self.logT)
                    else:
                        invT = math.log(l + T0_COOL) * torch.exp(-self.logc)
                    d2 = ((h[src] - h[dst]) ** 2).sum(1)
                    h = spmm(seg_softmax(-d2 * invT), h)
            return h

        def forward(self, x, K_run=None):
            h = self.propagate(self.inp(x), K_run)
            return self.out(h), h

    return Net()


def erank(H):
    X = H - H.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    if s.max() <= 0:
        return 1.0
    s = s[s > 1e-9 * s.max()]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def task(args):
    regime, model, K, seed = args
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    A, X, y, tr, va, te = make_data(regime, seed)
    n = len(y)
    dst_np, src_np = np.nonzero(A)  # edge j -> i stored as (dst=i, src=j)
    src = torch.tensor(src_np)
    dst = torch.tensor(dst_np)
    net = build(model, K, X.shape[1], int(y.max()) + 1, src, dst, n)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    xt, yt = torch.tensor(X), torch.tensor(y)
    best = (-1, 0, None)
    t_start = time.monotonic()
    for ep in range(EPOCHS):
        net.train()
        opt.zero_grad()
        logits, H = net(xt)
        loss = torch.nn.functional.cross_entropy(logits[tr], yt[tr])
        if not torch.isfinite(loss):
            break
        # model selection on the pre-update forward pass (no dropout, so train/eval passes coincide)
        with torch.no_grad():
            pred = logits.argmax(1)
            va_acc = (pred[va] == yt[va]).float().mean().item()
            if va_acc > best[0]:
                te_acc = (pred[te] == yt[te]).float().mean().item()
                best = (va_acc, te_acc, H.detach().numpy().copy(), ep)
        loss.backward()
        opt.step()
    out = dict(regime=regime, model=model, depth=K, seed=seed, val=best[0], test=best[1],
               erank=erank(best[2]) if best[2] is not None else None, seconds=time.monotonic() - t_start)
    if model == "gauss_cool":
        out["c"] = float(torch.exp(net.logc).item())
    if model == "gauss_const":
        out["T"] = float(torch.exp(net.logT).item())
    return out


if __name__ == "__main__":
    regimes = sys.argv[1:] or list(REGIMES)
    tasks = [(r, m, K, s) for r in regimes for K in DEPTHS[::-1] for m in MODELS for s in SEEDS]
    tasks += [(r, "mlp", 0, s) for r in regimes for s in SEEDS]
    t = time.monotonic()
    with Pool(int(os.environ.get("E6_WORKERS", 8))) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(task, tasks)):
            rows.append(row)
            if (i + 1) % 25 == 0:
                print(f"[{time.monotonic() - t:7.1f}s] {i + 1}/{len(tasks)}", flush=True)
                with open("results/train_E6.json", "w") as f:
                    json.dump(rows, f, indent=1)
    with open("results/train_E6.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"done in {time.monotonic() - t:.1f}s")
