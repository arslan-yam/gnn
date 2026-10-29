"""Loaders for the real-graph benchmarks in data/raw (sources and hashes in data/README.md).

The files are untrusted downloads, so they are read only with np.load(allow_pickle=False) and
torch.load(weights_only=True).
  load_node(name)  -> Roman-empire / Amazon-ratings: one graph, 10 fixed 50/25/25 splits
  load_peptides()  -> Peptides-func: lists of molecular graphs for train / val / test
"""
import os

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
NODE_DATASETS = ("roman_empire", "amazon_ratings")


def symmetrise(n, src, dst):
    """Both directions of every edge plus self-loops, deduplicated and sorted by (src, dst)."""
    key = np.unique(np.concatenate([src * n + dst, dst * n + src, np.arange(n) * (n + 1)]))
    return key // n, key % n


def csr(n, src, dst):
    """(indptr, indices) for edges sorted by src, as used by full.py."""
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, dst.astype(np.int64)


def load_node(name):
    z = np.load(os.path.join(ROOT, name, f"{name}.npz"), allow_pickle=False)
    y = z["node_labels"].astype(np.int64)
    n = len(y)
    E = z["edges"].astype(np.int64)
    src, dst = symmetrise(n, E[:, 0], E[:, 1])
    masks = {k: np.asarray(z[f"{k}_masks"], dtype=bool) for k in ("train", "val", "test")}
    return dict(name=name, X=np.asarray(z["node_features"], dtype=np.float32), y=y, n=n, src=src, dst=dst,
                splits=[{k: np.where(masks[k][i])[0] for k in masks} for i in range(masks["train"].shape[0])])


def load_peptides():
    """Each graph: (x [n, 9] int64 atom features, edge_index [2, m] int64, y [10] float32).
    Bond features are dropped (none of the compared models uses them)."""
    out = {}
    for split in ("train", "val", "test"):
        items = torch.load(os.path.join(ROOT, "peptides_func", "peptidesfunc", f"{split}.pt"), weights_only=True)
        out[split] = [(x.numpy(), ei.numpy(), y.numpy().reshape(-1)) for x, _, ei, y in items]
    allx = np.concatenate([g[0] for s in out.values() for g in s])
    out["atom_dims"] = [int(v) + 1 for v in allx.max(0)]
    return out


def batch_graphs(graphs):
    """Block-diagonal batch with self-loops: (x, src, dst, batch, y)."""
    xs, srcs, dsts, bs, ys, off = [], [], [], [], [], 0
    for g, (x, ei, y) in enumerate(graphs):
        n = x.shape[0]
        s, d = symmetrise(n, ei[0], ei[1])
        xs.append(x)
        srcs.append(s + off)
        dsts.append(d + off)
        bs.append(np.full(n, g, dtype=np.int64))
        ys.append(y)
        off += n
    return (np.concatenate(xs), np.concatenate(srcs), np.concatenate(dsts), np.concatenate(bs),
            np.stack(ys).astype(np.float32))
