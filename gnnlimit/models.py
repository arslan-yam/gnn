"""Model zoo and training loop for the downstream experiments (E6b, E7).

All models: dropout -> linear input embedding (width HID) -> K propagation layers -> dropout -> linear
readout. Propagation layers:
  gcn           h <- ReLU(A h W_l)                                   (Kipf & Welling 2017)
  gcn_res       h <- h + ReLU(A h W_l)
  gcn_pairnorm  h <- PairNorm(ReLU(A h W_l))                          (Zhao & Akoglu 2020)
  gcnii         h <- ReLU(((1-a) A h + a h0)((1-b_l) I + b_l W_l))      (Chen et al. 2020), a=0.1, lam=0.5
  appnp         z <- (1-a) A z + a z0                                  (Gasteiger et al. 2019), a=0.1
  sgc           z <- A z                                               (Wu et al. 2019)
  gat           h <- ELU(GAT_l(h))                                     (Velickovic et al. 2018)
  gat_res       h <- h + ELU(GAT_l(h))
  grand         z <- (1-tau) z + tau Att(z0) z, tau=0.5               (GRAND-l, Chamberlain et al. 2021)
  gauss_const   z <- Gauss_T(z) z,  alpha ~ exp(-||z_i-z_j||^2 / T)     T learnable
  gauss_cool    same with T_l = c / log(l + t0), c learnable, t0 = 11 (this paper)
                (both Gaussian models start from a calibrated temperature, see Net.calibrate)
  qk_cool       z <- Att(z) z,  alpha ~ exp((W_Q z_i)^T (W_K z_j) log(l + t0) / c): dot-product attention
                with query/key maps shared by all layers (score x^T J y, J = W_Q^T W_K, Section 7),
                W_Q = W_K = I at initialisation, c calibrated like the Gaussian models
  mlp           K = 0
A is the symmetrically normalised adjacency with self-loops (dense; ~10x faster than edge lists at
this size). Attention models use edge lists.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HID, T0_COOL = 16, 11.0
GCNII_ALPHA, GCNII_LAMBDA, APPNP_ALPHA, GRAND_TAU = 0.1, 0.5, 0.1, 0.5
ATTENTION_MODELS = {"gat", "gat_res", "grand", "gauss_const", "gauss_cool", "qk_cool"}


class Graph:
    def __init__(self, A):
        """A: dense boolean adjacency with self-loops (numpy)."""
        self.n = A.shape[0]
        dst, src = np.nonzero(A)  # message src -> dst
        self.src = torch.tensor(src)
        self.dst = torch.tensor(dst)
        Ad = torch.tensor(A, dtype=torch.float32)
        deg = Ad.sum(1)
        self.Ahat = Ad / torch.sqrt(deg[:, None] * deg[None, :])

    def seg_softmax(self, logit):
        mx = torch.full((self.n,), -1e30).scatter_reduce(0, self.dst, logit, reduce="amax", include_self=True)
        e = torch.exp(logit - mx[self.dst])
        s = torch.zeros(self.n).index_add_(0, self.dst, e)
        return e / s[self.dst]

    def aggregate(self, w, hs):
        """sum over edges of w_e * hs_e into dst; hs = h[src]."""
        return torch.zeros(self.n, hs.shape[1]).index_add_(0, self.dst, w[:, None] * hs)


def pairnorm(h, s=1.0):
    hc = h - h.mean(0, keepdim=True)
    return s * math.sqrt(h.shape[0]) * hc / (hc.norm() + 1e-9)


# second value of each model's specific hyperparameter (the first value is the module default)
SPECIFIC_ALT = {"gcnii": 0.3, "appnp": 0.3, "grand": 0.1, "gcn_pairnorm": 10.0,
                "gauss_const": 1 / 30, "gauss_cool": 1 / 30, "qk_cool": 1 / 30}


class Net(nn.Module):
    def __init__(self, model, K, n_feat, n_cls, dropout, spec=None):
        """spec: model-specific hyperparameter (GCNII/APPNP alpha, GRAND tau, PairNorm scale,
        Gaussian temperature-init multiplier); None = default."""
        super().__init__()
        self.model, self.K, self.p = model, K, dropout
        defaults = {"gcnii": GCNII_ALPHA, "appnp": APPNP_ALPHA, "grand": GRAND_TAU, "gcn_pairnorm": 1.0,
                    "gauss_const": 1.0, "gauss_cool": 1.0, "qk_cool": 1.0}
        self.spec = defaults.get(model) if spec is None else spec
        self.inp = nn.Linear(n_feat, HID)
        self.out = nn.Linear(HID, n_cls)
        if model in ("gcn", "gcn_res", "gcn_pairnorm", "gcnii"):
            self.W = nn.ModuleList([nn.Linear(HID, HID, bias=(model != "gcnii")) for _ in range(K)])
        if model in ("gat", "gat_res"):
            self.W = nn.ModuleList([nn.Linear(HID, HID, bias=False) for _ in range(K)])
            self.a = nn.ParameterList([nn.Parameter(0.1 * torch.randn(2 * HID)) for _ in range(K)])
        if model == "grand":
            self.Q = nn.Linear(HID, HID, bias=False)
            self.Kk = nn.Linear(HID, HID, bias=False)
        if model == "gauss_const":
            self.logT = nn.Parameter(torch.tensor(0.0))
        if model == "gauss_cool":
            self.logc = nn.Parameter(torch.tensor(0.0))
        if model == "qk_cool":
            self.Wq = nn.Linear(HID, HID, bias=False)
            self.Wk = nn.Linear(HID, HID, bias=False)
            nn.init.eye_(self.Wq.weight)
            nn.init.eye_(self.Wk.weight)
            self.logc = nn.Parameter(torch.tensor(0.0))

    def J(self):
        """Score matrix of qk_cool: f(x, y) = x^T J y."""
        return self.Wq.weight.T @ self.Wk.weight

    @torch.no_grad()
    def calibrate(self, x, G):
        """Data-dependent temperature init for the Gaussian models: the median squared distance
        across (non-self) edges of the initial embedding gets weight e^{-1} at layer 0."""
        if self.model not in ("gauss_const", "gauss_cool", "qk_cool"):
            return
        z = self.inp(x)
        if self.model == "qk_cool":  # |directed margin| m_ij = f(z_i, z_i) - f(z_i, z_j) instead of the squared distance
            q, k = self.Wq(z), self.Wk(z)
            d2 = (q[G.dst] * (k[G.dst] - k[G.src])).sum(1).abs()
        else:
            d2 = ((z[G.src] - z[G.dst]) ** 2).sum(1)
        med = d2[G.src != G.dst].median().clamp_min(1e-6) * self.spec
        if self.model == "gauss_const":
            self.logT.fill_(math.log(med.item()))
        else:
            self.logc.fill_(math.log(med.item() * math.log(T0_COOL)))

    def forward(self, x, G, K_run=None):
        m, K = self.model, (self.K if K_run is None else K_run)
        h = self.inp(F.dropout(x, self.p, self.training))
        if m == "gcnii":
            h = torch.relu(h)
        h0 = h
        if m == "grand":
            q, k = self.Q(h0), self.Kk(h0)
            att = G.seg_softmax((q[G.dst] * k[G.src]).sum(1) / math.sqrt(HID))
        for l in range(K):
            if m in ("gcn", "gcn_res", "gcn_pairnorm"):
                z = G.Ahat @ self.W[l](F.dropout(h, self.p, self.training))
                if m == "gcn":
                    h = torch.relu(z) if l < K - 1 else z
                elif m == "gcn_res":
                    h = h + torch.relu(z)
                else:
                    h = pairnorm(torch.relu(z), self.spec)
            elif m == "gcnii":
                beta = math.log(GCNII_LAMBDA / (l + 1) + 1)
                s = (1 - self.spec) * (G.Ahat @ F.dropout(h, self.p, self.training)) + self.spec * h0
                h = torch.relu((1 - beta) * s + beta * self.W[l](s))
            elif m == "appnp":
                h = (1 - self.spec) * (G.Ahat @ h) + self.spec * h0
            elif m == "sgc":
                h = G.Ahat @ h
            elif m in ("gat", "gat_res"):
                g = self.W[l](F.dropout(h, self.p, self.training))
                a = self.a[l]
                logit = F.leaky_relu((g[G.dst] * a[:HID]).sum(1) + (g[G.src] * a[HID:]).sum(1), 0.2)
                z = G.aggregate(G.seg_softmax(logit), g[G.src])
                if m == "gat":
                    h = F.elu(z) if l < K - 1 else z
                else:
                    h = h + F.elu(z)
            elif m == "grand":
                h = (1 - self.spec) * h + self.spec * G.aggregate(att, h[G.src])
            elif m == "qk_cool":
                invT = math.log(l + T0_COOL) * torch.exp(-self.logc)
                q, k = self.Wq(h), self.Wk(h)
                hs = h[G.src]
                h = G.aggregate(G.seg_softmax((q[G.dst] * k[G.src]).sum(1) * invT), hs)
            elif m in ("gauss_const", "gauss_cool"):
                if m == "gauss_const":
                    invT = torch.exp(-self.logT)
                else:
                    invT = math.log(l + T0_COOL) * torch.exp(-self.logc)
                hs = h[G.src]
                d2 = ((hs - h[G.dst]) ** 2).sum(1)
                h = G.aggregate(G.seg_softmax(-d2 * invT), hs)
            else:
                raise ValueError(m)
        return self.out(F.dropout(h, self.p, self.training)), h


def erank(H):
    X = H - H.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    if s.max() <= 0:
        return 1.0
    s = s[s > 1e-9 * s.max()]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def snapshot(net, xt):
    """Trained qk_cool network as numpy arrays: input embedding z0 (eval mode), score matrix J, temperature scale c
    and the readout, enough to rerun its dynamics to any depth (exp_learned.py trained)."""
    was = net.training
    net.eval()
    st = dict(z0=net.inp(xt).numpy().astype(np.float64), J=net.J().numpy().astype(np.float64),
              c=float(torch.exp(net.logc).item()), Wout=net.out.weight.numpy().copy(), bout=net.out.bias.numpy().copy())
    net.train(was)
    return st


def fit(model, K, X, y, tr, va, te, G, lr, dropout, seed, wd=5e-4, max_epochs=300, patience=60, spec=None):
    """Full-batch Adam; model selection on validation accuracy (eval mode). Returns a dict; for qk_cool the
    entry "state" holds the snapshot of the selected network."""
    torch.manual_seed(seed)
    net = Net(model, K, X.shape[1], int(y.max()) + 1, dropout, spec)
    xt, yt = torch.tensor(X), torch.tensor(y)
    net.calibrate(xt, G)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best = dict(val=-1.0, test=0.0, epoch=-1, H=None)
    t0 = time.monotonic()
    for ep in range(max_epochs):
        net.train()
        opt.zero_grad()
        logits, H = net(xt, G)
        loss = F.cross_entropy(logits[tr], yt[tr])
        if not torch.isfinite(loss):
            break
        if dropout == 0.0:
            ev_logits, ev_H = logits.detach(), H.detach()  # identical to the eval-mode pass
            if model == "qk_cool":
                cand = snapshot(net, xt)  # parameters that produced ev_logits (before the update)
        loss.backward()
        opt.step()
        if dropout > 0.0:
            net.eval()
            with torch.no_grad():
                ev_logits, ev_H = net(xt, G)
            if model == "qk_cool":
                cand = snapshot(net, xt)
        pred = ev_logits.argmax(1)
        va_acc = (pred[va] == yt[va]).float().mean().item()
        if va_acc > best["val"]:
            best = dict(val=va_acc, test=(pred[te] == yt[te]).float().mean().item(), epoch=ep,
                        H=ev_H.numpy().copy())
            if model == "qk_cool":
                best["state"] = cand
        elif ep - best["epoch"] >= patience:
            break
    out = dict(val=best["val"], test=best["test"], best_epoch=best["epoch"], epochs=ep + 1,
               erank=erank(best["H"]) if best["H"] is not None else None,
               seconds=time.monotonic() - t0)
    if model in ("gauss_cool", "qk_cool"):
        out["c"] = float(torch.exp(net.logc).item())
    if model == "qk_cool" and "state" in best:
        out["state"] = best["state"]
        out["c"] = best["state"]["c"]
        out["J_sym_eig"] = [float(v) for v in np.linalg.eigvalsh((best["state"]["J"] + best["state"]["J"].T) / 2)]
    if model == "gauss_const":
        out["T"] = float(torch.exp(net.logT).item())
    return out


GRID = [dict(lr=lr, dropout=p) for lr in (0.01, 0.003) for p in (0.0, 0.5)]


def fit_tuned(model, K, X, y, tr, va, te, G, seed, alt=False):
    """Per-run hyperparameter selection over GRID by validation accuracy. alt=True runs the grid with
    the second value of the model-specific hyperparameter (SPECIFIC_ALT, or weight decay 0 for
    models without one)."""
    runs = []
    for hp in GRID:
        kw = {}
        if alt:
            if model in SPECIFIC_ALT:
                kw["spec"] = SPECIFIC_ALT[model]
            else:
                kw["wd"] = 0.0
        r = fit(model, K, X, y, tr, va, te, G, hp["lr"], hp["dropout"], seed, **kw)
        r.update(hp, alt=alt)
        runs.append(r)
    best = max(runs, key=lambda r: (r["val"], -r["lr"]))
    return dict(best, grid=[{k: r[k] for k in ("lr", "dropout", "val", "test")} for r in runs],
                seconds_total=sum(r["seconds"] for r in runs))
