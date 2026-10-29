"""The model zoo of models.py at real-graph scale (E8).

Differences from models.Net, none of which changes a model's definition:
  * sparse graphs (SparseGraph: edge lists + a sparse normalised adjacency) instead of dense ones;
  * configurable width (models.Net reads the module constant HID, which is set before construction);
  * optional per-layer gradient checkpointing, so that depth-64 attention fits in memory;
  * graph-level tasks: categorical atom features are embedded (sum of per-column embeddings, as the OGB
    AtomEncoder), and node states are mean-pooled per graph before the readout.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import models
from .models import GCNII_LAMBDA, T0_COOL, Graph, Net, pairnorm


class EdgeAdj:
    """Normalised adjacency kept as an edge list; A @ h by gather-scatter (sparse matmul is slow on MPS)."""

    def __init__(self, n, src, dst, w):
        self.n, self.src, self.dst, self.w = n, src, dst, w

    def __matmul__(self, h):
        return torch.zeros(self.n, h.shape[1], device=h.device).index_add_(0, self.dst, self.w[:, None] * h[self.src])


class SparseGraph(Graph):
    """Edge list (src -> dst) that already holds both directions and self-loops. batch: graph id per
    node for block-diagonal batches of graphs (None for a single graph)."""

    def __init__(self, n, src, dst, batch=None, n_graphs=None, device="cpu"):
        self.n, self.device = n, torch.device(device)
        self.src = torch.as_tensor(src, dtype=torch.long).to(self.device)
        self.dst = torch.as_tensor(dst, dtype=torch.long).to(self.device)
        deg = torch.zeros(n, device=self.device).index_add_(0, self.dst, torch.ones(len(self.dst), device=self.device))
        w = (deg[self.src] * deg[self.dst]).rsqrt()
        if self.device.type == "cpu":
            self.Ahat = torch.sparse_coo_tensor(torch.stack([self.dst, self.src]), w, (n, n)).coalesce()
        else:
            self.Ahat = EdgeAdj(n, self.src, self.dst, w)
        self.batch = None if batch is None else torch.as_tensor(batch, dtype=torch.long).to(self.device)
        self.n_graphs = n_graphs
        if self.batch is not None:
            self.counts = torch.zeros(n_graphs, device=self.device).index_add_(
                0, self.batch, torch.ones(n, device=self.device))

    def seg_softmax(self, logit):
        mx = torch.full((self.n,), -1e30, device=self.device).scatter_reduce(
            0, self.dst, logit, reduce="amax", include_self=True)
        e = torch.exp(logit - mx[self.dst])
        s = torch.zeros(self.n, device=self.device).index_add_(0, self.dst, e)
        return e / s[self.dst]

    def aggregate(self, w, hs):
        return torch.zeros(self.n, hs.shape[1], device=self.device).index_add_(0, self.dst, w[:, None] * hs)

    def pool(self, h):
        return torch.zeros(self.n_graphs, h.shape[1], device=self.device).index_add_(0, self.batch, h) / self.counts[:, None]


class AtomEncoder(nn.Module):
    def __init__(self, dims, hidden):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(d, hidden) for d in dims])
        self.out_features = hidden

    def forward(self, x):
        return sum(e(x[:, k]) for k, e in enumerate(self.emb))


class DeepNet(Net):
    def __init__(self, model, K, n_feat, n_cls, dropout, spec=None, hidden=64, atom_dims=None, ckpt=False):
        models.HID = hidden  # read by Net.__init__
        super().__init__(model, K, n_feat, n_cls, dropout, spec)
        self.hid, self.ckpt, self.categorical = hidden, ckpt, atom_dims is not None
        if self.categorical:
            self.inp = AtomEncoder(atom_dims, hidden)

    def embed(self, x):
        if self.categorical:
            return F.dropout(self.inp(x), self.p, self.training)
        return self.inp(F.dropout(x, self.p, self.training))

    @torch.no_grad()
    def calibrate(self, x, G):
        if self.model not in ("gauss_const", "gauss_cool"):
            return
        z = self.inp(x)
        d2 = ((z[G.src] - z[G.dst]) ** 2).sum(1)
        med = d2[G.src != G.dst].median().clamp_min(1e-6) * self.spec
        if self.model == "gauss_const":
            self.logT.fill_(math.log(med.item()))
        else:
            self.logc.fill_(math.log(med.item() * math.log(T0_COOL)))

    def layer(self, l, h, h0, att, G, last):
        """One propagation layer; identical to the loop body of models.Net.forward."""
        m, p, tr, H = self.model, self.p, self.training, self.hid
        if m in ("gcn", "gcn_res", "gcn_pairnorm"):
            z = G.Ahat @ self.W[l](F.dropout(h, p, tr))
            if m == "gcn":
                return z if last else torch.relu(z)
            if m == "gcn_res":
                return h + torch.relu(z)
            return pairnorm(torch.relu(z), self.spec)
        if m == "gcnii":
            beta = math.log(GCNII_LAMBDA / (l + 1) + 1)
            s = (1 - self.spec) * (G.Ahat @ F.dropout(h, p, tr)) + self.spec * h0
            return torch.relu((1 - beta) * s + beta * self.W[l](s))
        if m == "appnp":
            return (1 - self.spec) * (G.Ahat @ h) + self.spec * h0
        if m == "sgc":
            return G.Ahat @ h
        if m in ("gat", "gat_res"):
            g = self.W[l](F.dropout(h, p, tr))
            a = self.a[l]
            logit = F.leaky_relu((g[G.dst] * a[:H]).sum(1) + (g[G.src] * a[H:]).sum(1), 0.2)
            z = G.aggregate(G.seg_softmax(logit), g[G.src])
            if m == "gat":
                return z if last else F.elu(z)
            return h + F.elu(z)
        if m == "grand":
            return (1 - self.spec) * h + self.spec * G.aggregate(att, h[G.src])
        if m in ("gauss_const", "gauss_cool"):
            invT = torch.exp(-self.logT) if m == "gauss_const" else math.log(l + T0_COOL) * torch.exp(-self.logc)
            hs = h[G.src]
            d2 = ((hs - h[G.dst]) ** 2).sum(1)
            return G.aggregate(G.seg_softmax(-d2 * invT), hs)
        raise ValueError(m)

    def forward(self, x, G, K_run=None):
        m, K = self.model, (self.K if K_run is None else K_run)
        h = self.embed(x)
        if m == "gcnii":
            h = torch.relu(h)
        h0, att = h, None
        if m == "grand":
            q, k = self.Q(h0), self.Kk(h0)
            att = G.seg_softmax((q[G.dst] * k[G.src]).sum(1) / math.sqrt(self.hid))
        for l in range(K):
            if self.ckpt and self.training and torch.is_grad_enabled():
                h = checkpoint(self.layer, l, h, h0, att, G, l == K - 1, use_reentrant=False)
            else:
                h = self.layer(l, h, h0, att, G, l == K - 1)
        hr = h if G.batch is None else G.pool(h)
        return self.out(F.dropout(hr, self.p, self.training)), h
