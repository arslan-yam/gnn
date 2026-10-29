"""E8: real-graph benchmarks -- Roman-empire, Amazon-ratings (Platonov et al. 2023) and Peptides-func (LRGB).

Modes (each writes results/real_E8_<mode>.jsonl, one JSON row per task, so an interrupted run resumes):
  predict    cluster-survival prediction for the untrained cooled dynamics on the real graphs: the
             threshold-graph rule at t1 in {0, 5, 20} against the partition after 4000 layers, for an
             unsupervised (PCA) metric at three scales and a learned (LDA) metric scaled by the design rule;
             linear-probe accuracy of the propagated features against depth (numba, CPU)
  node       depth sweeps of the model zoo on Roman-empire and Amazon-ratings (GPU if available): split 0
             tunes the learning rate per (model, depth) on validation accuracy, splits 1..2 reuse the choice
  peptides   depth sweeps on Peptides-func (graph level, mean pooling, average precision)
  trained    cluster-survival prediction for the trained gauss_cool networks of `node`, and their test
             accuracy when the trained network is run far beyond its trained depth
  time       per-epoch timings (writes nothing)
Usage: python -m experiments.exp_real MODE        (env SMOKE=1: tiny settings, writes *_smoke.jsonl)
Several workers can share the node and peptides task lists (on the M1 the CPU trains these sparse graphs about
as fast as the GPU): env E8_DEVICE=cpu|mps, E8_THREADS=n; a worker runs a task only after creating its claim
file in results/real_E8_claims, which must be emptied before the workers start (scripts/run_E8.sh does this).
"""
import glob
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from gnnlimit import full
from gnnlimit import models
from gnnlimit.core import kappa_estimate
from gnnlimit.models_real import DeepNet, SparseGraph
from gnnlimit.real_data import NODE_DATASETS, csr, load_node, load_peptides, symmetrise

warnings.filterwarnings("ignore", message=".*Sparse.*")
SMOKE = os.environ.get("SMOKE") == "1"
DEV = os.environ.get("E8_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
if os.environ.get("E8_THREADS"):
    torch.set_num_threads(int(os.environ["E8_THREADS"]))
HIDDEN, WD, CKPT_FROM = 64, 5e-4, 16

NODE_MODELS = ["gcn", "gcn_res", "gcn_pairnorm", "gcnii", "appnp", "sgc", "gat", "gat_res", "grand",
               "gauss_const", "gauss_cool"]
NODE_DEPTHS = [2, 4, 8, 16, 32]
NODE_GRID = [dict(lr=lr, dropout=0.2) for lr in (0.01, 0.003)]  # GPU budget: lr tuned, dropout fixed
NODE_SPLITS = [0, 1, 2]  # of the 10 official 50/25/25 splits; split 0 tunes
MAX_EPOCHS, PATIENCE = 150, 30

PEP_MODELS = ["mlp", "gcn", "gcn_res", "gcnii", "gat_res", "gauss_const", "gauss_cool"]
PEP_DEPTHS = [2, 8, 32]
PEP_SEEDS = [0]  # GPU budget: ~6 s per attention layer-epoch, so one seed and 30 epochs (cosine schedule)
PEP_EPOCHS, PEP_BATCH, PEP_LR, PEP_DROPOUT = 30, 128, 1e-3, 0.1

T1S, T_PRED, PROBE_DEPTHS = (0, 5, 20), 4000, (0, 16, 128, 1024, 4000)
PEP_PRED_MOLECULES, PEP_T0 = 300, 152.0  # t0 = (mean atoms per molecule) + 2
T_TRAINED = 4000

if SMOKE:
    NODE_MODELS, NODE_DEPTHS, NODE_GRID = ["gcn", "gat_res", "gauss_cool"], [2, 16], NODE_GRID[:2]
    NODE_SPLITS, MAX_EPOCHS, PATIENCE = [0, 1], 3, 2
    PEP_MODELS, PEP_DEPTHS, PEP_SEEDS, PEP_EPOCHS = ["gcn", "gauss_cool"], [2, 16], [0], 1
    T1S, T_PRED, PROBE_DEPTHS, PEP_PRED_MOLECULES, T_TRAINED = (0, 5), 40, (0, 40), 5, 40

SUFFIX = "_smoke" if SMOKE else ""
STATES = f"results/real_E8_states{SUFFIX}"
CLAIMS = f"results/real_E8_claims{SUFFIX}"
T_START = time.monotonic()


def stamp(msg):
    print(f"[{time.monotonic() - T_START:8.1f}s] {msg}", flush=True)


def out_path(mode):
    return f"results/real_E8_{mode}{SUFFIX}.jsonl"


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_row(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def claim(mode, key):
    """True for exactly one of the workers that try to claim this task (exclusive file creation)."""
    os.makedirs(CLAIMS, exist_ok=True)
    try:
        os.close(os.open(os.path.join(CLAIMS, f"{mode}__" + "__".join(map(str, key))), os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


# ---------------------------------------------------------------------------------------------------
# node classification (Roman-empire, Amazon-ratings)
# ---------------------------------------------------------------------------------------------------
def node_inputs(name):
    D = load_node(name)
    G = SparseGraph(D["n"], D["src"], D["dst"], device=DEV)
    return D, G, torch.tensor(D["X"]).to(DEV), torch.tensor(D["y"]).to(DEV)


def fit_node(D, G, xt, yt, split, model, K, lr, dropout, seed, state_file=None, max_epochs=None):
    """Full-batch Adam; the epoch with the best validation accuracy (eval mode) is reported."""
    torch.manual_seed(seed)
    idx = {k: torch.tensor(v).to(DEV) for k, v in D["splits"][split].items()}
    net = DeepNet(model, K, xt.shape[1], int(D["y"].max()) + 1, dropout, hidden=HIDDEN, ckpt=K >= CKPT_FROM).to(DEV)
    net.calibrate(xt, G)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=WD)
    best = dict(val=-1.0, test=0.0, epoch=-1, H=None, state=None)
    t0 = time.monotonic()
    ep = -1
    for ep in range(max_epochs or MAX_EPOCHS):
        net.train()
        opt.zero_grad()
        loss = F.cross_entropy(net(xt, G)[0][idx["train"]], yt[idx["train"]])
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            logits, H = net(xt, G)
        pred = logits.argmax(1)
        va = (pred[idx["val"]] == yt[idx["val"]]).float().mean().item()
        if va > best["val"]:
            best = dict(val=va, test=(pred[idx["test"]] == yt[idx["test"]]).float().mean().item(), epoch=ep,
                        H=H.cpu().numpy(),
                        state={k: v.detach().cpu().clone() for k, v in net.state_dict().items()} if state_file else None)
        elif ep - best["epoch"] >= PATIENCE:
            break
    out = dict(val=best["val"], test=best["test"], best_epoch=best["epoch"], epochs=ep + 1,
               erank=models.erank(best["H"]) if best["H"] is not None else None, seconds=time.monotonic() - t0)
    if model == "gauss_cool":
        out["c"] = float(torch.exp(net.logc).item())
    if model == "gauss_const":
        out["T"] = float(torch.exp(net.logT).item())
    if state_file and best["state"] is not None:
        torch.save(best["state"], state_file)
    return out


def node_cells():
    return [("mlp", 0)] + [(m, K) for K in NODE_DEPTHS for m in NODE_MODELS]


def node_tasks(rows):
    """(dataset, model, K, split, cfg, tuning) in run order. cfg is None for a phase-B task whose tuning on
    split 0 is not finished yet."""
    for name in NODE_DATASETS:
        for m, K in node_cells():  # phase A: tuning on split 0
            for cfg in NODE_GRID:
                yield name, m, K, NODE_SPLITS[0], cfg, True
    grid = {(c["lr"], c["dropout"]) for c in NODE_GRID}
    for name in NODE_DATASETS:
        for m, K in node_cells():  # phase B: remaining splits with the configuration chosen on split 0
            cand = [r for r in rows if r["dataset"] == name and r["split"] == NODE_SPLITS[0]
                    and r["model"] == m and r["depth"] == K]
            b = max(cand, key=lambda r: (r["val"], -r["lr"], r["dropout"])) \
                if {(r["lr"], r["dropout"]) for r in cand} >= grid else None
            for split in NODE_SPLITS[1:]:
                yield name, m, K, split, b and dict(lr=b["lr"], dropout=b["dropout"]), False


def run_node():
    path = out_path("node")
    os.makedirs(STATES, exist_ok=True)
    key = lambda r: (r["dataset"], r["model"], r["depth"], r["split"], r["lr"], r["dropout"])
    stamp(f"node: device {DEV}, {len(read_rows(path))} runs already done")
    inputs = {}
    while True:
        rows = read_rows(path)
        done = {key(r) for r in rows}
        row, waiting = None, False
        for name, m, K, split, cfg, tuning in node_tasks(rows):
            if cfg is None:
                waiting = True
                continue
            r = dict(dataset=name, model=m, depth=K, split=split, lr=cfg["lr"], dropout=cfg["dropout"], tuning=tuning)
            if key(r) not in done and claim("node", key(r)):
                row = r
                break
        if row is None:
            if not waiting:
                return
            time.sleep(30)  # another worker is still tuning a cell that a remaining task needs
            continue
        name, m, K, split = row["dataset"], row["model"], row["depth"], row["split"]
        if name not in inputs:
            inputs[name] = node_inputs(name)
        D, G, xt, yt = inputs[name]
        sf = f"{STATES}/{name}_{m}_K{K}_split{split}_lr{row['lr']}_p{row['dropout']}.pt" \
            if m in ("gauss_cool", "gauss_const") else None
        row.update(fit_node(D, G, xt, yt, split, m, K, row["lr"], row["dropout"], seed=split, state_file=sf), device=DEV)
        append_row(path, row)
        stamp(f"  {name} {m:12s} K={K:3d} split {split} lr={row['lr']} p={row['dropout']}: "
              f"val {row['val']:.4f} test {row['test']:.4f} ({row['epochs']} ep, {row['seconds']:.0f}s)")


# ---------------------------------------------------------------------------------------------------
# Peptides-func (graph classification, 10 labels, average precision)
# ---------------------------------------------------------------------------------------------------
def pep_prepare():
    P = load_peptides()
    for split in ("train", "val", "test"):
        graphs = P[split][:512] if SMOKE else P[split]
        P[split] = [(x, *symmetrise(x.shape[0], ei[0], ei[1]), y) for x, ei, y in graphs]
    return P


def pep_batch(graphs):
    sizes = np.array([g[0].shape[0] for g in graphs])
    off = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    x = np.concatenate([g[0] for g in graphs])
    src = np.concatenate([g[1] + o for g, o in zip(graphs, off)])
    dst = np.concatenate([g[2] + o for g, o in zip(graphs, off)])
    batch = np.repeat(np.arange(len(graphs)), sizes)
    G = SparseGraph(len(x), src, dst, batch=batch, n_graphs=len(graphs), device=DEV)
    return torch.tensor(x).to(DEV), G, torch.tensor(np.stack([g[3] for g in graphs])).to(DEV)


def pep_scores(net, batches):
    from sklearn.metrics import average_precision_score
    net.eval()
    ys, ss = [], []
    with torch.no_grad():
        for x, G, y in batches:
            ss.append(torch.sigmoid(net(x, G)[0]).cpu().numpy())
            ys.append(y.cpu().numpy())
    return float(average_precision_score(np.concatenate(ys), np.concatenate(ss), average="macro"))


def fit_pep(P, evals, model, K, seed, max_batches=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = DeepNet(model, K, 1, 10, PEP_DROPOUT, hidden=HIDDEN, atom_dims=P["atom_dims"], ckpt=K >= CKPT_FROM).to(DEV)
    x0, G0, _ = pep_batch(P["train"][:PEP_BATCH])
    net.calibrate(x0, G0)
    opt = torch.optim.Adam(net.parameters(), lr=PEP_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PEP_EPOCHS)
    best = dict(val=-1.0, test=0.0, epoch=-1)
    t0 = time.monotonic()
    n = len(P["train"])
    for ep in range(PEP_EPOCHS):
        net.train()
        perm = rng.permutation(n)
        for b, start in enumerate(range(0, n, PEP_BATCH)):
            if max_batches is not None and b >= max_batches:
                break
            x, G, y = pep_batch([P["train"][i] for i in perm[start:start + PEP_BATCH]])
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(net(x, G)[0], y)
            loss.backward()
            opt.step()
        sched.step()
        if max_batches is not None:
            sync()
            return dict(seconds_per_batch=(time.monotonic() - t0) / max_batches)
        va = pep_scores(net, evals["val"])
        if va > best["val"]:
            best = dict(val=va, test=pep_scores(net, evals["test"]), epoch=ep)
    out = dict(val=best["val"], test=best["test"], best_epoch=best["epoch"], epochs=PEP_EPOCHS,
               seconds=time.monotonic() - t0)
    if model == "gauss_cool":
        out["c"] = float(torch.exp(net.logc).item())
    return out


def run_peptides():
    path = out_path("peptides")
    key = lambda r: (r["model"], r["depth"], r["seed"])
    stamp(f"peptides: device {DEV}, {len(read_rows(path))} runs already done")
    P = None
    # deepest first, so that the workers finish together
    cells = [(m, K) for K in sorted(PEP_DEPTHS, reverse=True) for m in PEP_MODELS if m != "mlp"]
    cells += [("mlp", 0)] if "mlp" in PEP_MODELS else []
    for seed in PEP_SEEDS:
        for m, K in cells:
            row = dict(model=m, depth=K, seed=seed)
            if key(row) in {key(r) for r in read_rows(path)} or not claim("peptides", key(row)):
                continue
            if P is None:
                P = pep_prepare()
                evals = {s: [pep_batch(P[s][i:i + 512]) for i in range(0, len(P[s]), 512)] for s in ("val", "test")}
            row.update(fit_pep(P, evals, m, K, seed), hidden=HIDDEN, lr=PEP_LR, dropout=PEP_DROPOUT, batch=PEP_BATCH,
                       device=DEV)
            append_row(path, row)
            stamp(f"  peptides {m:12s} K={K:3d} seed {seed}: val AP {row['val']:.4f} test AP {row['test']:.4f} "
                  f"({row['seconds']:.0f}s)")


# ---------------------------------------------------------------------------------------------------
# cluster-survival prediction (untrained dynamics, then trained gauss_cool networks)
# ---------------------------------------------------------------------------------------------------
def kappas(t0, t1s):
    return {t1: kappa_estimate(t0 + t1, T=10**6) for t1 in t1s}


def edge_stats(Z0, ZT, indptr, indices, obs, kap0):
    """Edge-level fates (non-self edges) at t1 = 0; observed persist iff the endpoints end in different clusters.
    Predicted persist: "pair", ||z_i - z_j|| >= kappa (the two-block rule edge by edge, which ignores that an
    edge also merges when a path of short edges joins its endpoints); "part", the endpoints lie in different
    components of the threshold graph (the rule of partition_report)."""
    src, dst, d0 = full.edge_dist(Z0, indptr, indices)
    keep = src != dst
    s, t = src[keep], dst[keep]
    pred = full.threshold_partition(Z0, indptr, indices, kap0)
    op = obs[s] != obs[t]
    out = dict(edges=int(keep.sum()), obs_persist=int(op.sum()))
    for tag, pp in (("pair", d0[keep] >= kap0), ("part", pred[s] != pred[t])):
        out.update({f"{tag}_pred_persist": int(pp.sum()), f"{tag}_false_persist": int((pp & ~op).sum()),
                    f"{tag}_false_merge": int((~pp & op).sum())})
    return out


def partition_report(rec, ZT, indptr, indices, kap, t1s):
    from sklearn.metrics import adjusted_rand_score
    obs, unres = full.observed_partition(ZT, indptr, indices)
    out = dict(n_obs=int(obs.max() + 1), unresolved=unres)
    for t1 in t1s:
        pred = full.threshold_partition(rec[t1], indptr, indices, kap[t1])
        out[f"ari_t{t1}"] = float(adjusted_rand_score(pred, obs))
        out[f"n_pred_t{t1}"] = int(pred.max() + 1)
    return obs, out


def probe(H, y, tr, te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return float(clf.fit(H[tr], y[tr]).score(H[te], y[te]))


def run_predict():
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    path = out_path("predict")
    done = {(r["dataset"], r["metric"]) for r in read_rows(path)}
    for name in NODE_DATASETS:
        D = load_node(name)
        n, y = D["n"], D["y"]
        indptr, indices = csr(n, D["src"], D["dst"])
        t0 = n + 2.0
        kap = kappas(t0, T1S)
        stamp(f"predict {name}: t0 = {t0:.0f}, kappa = {kap}")
        H = PCA(64, random_state=0).fit_transform(StandardScaler().fit_transform(D["X"]))
        tr, te = D["splits"][0]["train"], D["splits"][0]["test"]
        C = int(y.max()) + 1
        src, dst, _ = full.edge_dist(H, indptr, indices)
        nonself = src != dst
        W_pca = np.eye(64)[:16]
        med = np.median(np.linalg.norm((H[src] - H[dst])[nonself] @ W_pca.T, axis=1))
        lda = LinearDiscriminantAnalysis(n_components=min(C - 1, 16)).fit(H[tr], y[tr])
        W_lda = lda.scalings_[:, :min(C - 1, 16)].T
        mus = np.stack([H[tr][y[tr] == k].mean(0) for k in range(C)])
        dmin = min(np.linalg.norm(W_lda @ (mus[a] - mus[b])) for a in range(C) for b in range(a + 1, C))
        metrics = {f"pca16 x{s}": W_pca * (s * kap[0] / med) for s in (0.5, 1.0, 2.0)}
        metrics.update({f"lda margin {m}": W_lda * (m * kap[0] / dmin) for m in (2.0, 0.4)})
        for mname, W in metrics.items():
            if (name, mname) in done:
                continue
            ts = time.monotonic()
            ZT, HT, rec = full.run_values(H @ W.T, H, indptr, indices, t0, T=T_PRED,
                                          record=set(T1S) | {T_PRED // 2} | set(PROBE_DEPTHS))
            obs, row = partition_report({t: rec[t][0] for t in T1S}, ZT, indptr, indices, kap, T1S)
            obs_half, _ = full.observed_partition(rec[T_PRED // 2][0], indptr, indices)
            row.update(dataset=name, metric=mname, t0=t0, kappa={str(k): v for k, v in kap.items()}, T=T_PRED,
                       stable=float(adjusted_rand_score(obs, obs_half)), ari_labels=float(adjusted_rand_score(y, obs)),
                       edges=edge_stats(rec[0][0], ZT, indptr, indices, obs, kap[0]),
                       probe={str(d): probe(rec[d][1], y, tr, te) for d in PROBE_DEPTHS},
                       erank={str(d): full.effective_rank(rec[d][1]) for d in PROBE_DEPTHS},
                       seconds=time.monotonic() - ts)
            append_row(path, row)
            stamp(f"  {name} {mname:16s}: n_obs {row['n_obs']}, n_pred@0 {row['n_pred_t0']}, ARI@0 {row['ari_t0']:.3f}, "
                  f"ARI@{T1S[-1]} {row[f'ari_t{T1S[-1]}']:.3f}, unresolved {row['unresolved']:.3f}, "
                  f"probe {row['probe']}")
    if ("peptides_func", "pca8 x1.0") not in done:
        predict_peptides(path, done)


def predict_peptides(path, done):
    from sklearn.decomposition import PCA
    P = load_peptides()
    dims = P["atom_dims"]
    off = np.concatenate([[0], np.cumsum(dims)[:-1]])

    def onehot(x):
        Z = np.zeros((x.shape[0], sum(dims)))
        for k in range(x.shape[1]):
            Z[np.arange(x.shape[0]), off[k] + x[:, k]] = 1.0
        return Z
    rng = np.random.default_rng(0)
    train_atoms = np.concatenate([P["train"][i][0] for i in rng.choice(len(P["train"]), 1000, replace=False)])
    pca = PCA(8, random_state=0).fit(onehot(train_atoms))
    mols = []
    for x, ei, _ in P["test"][:PEP_PRED_MOLECULES]:
        s, d = symmetrise(x.shape[0], ei[0], ei[1])
        mols.append((pca.transform(onehot(x)), *csr(x.shape[0], s, d)))
    med = np.median(np.concatenate([full.edge_dist(Z, ip, ix)[2][full.edge_dist(Z, ip, ix)[0] != ix] for Z, ip, ix in mols]))
    kap = kappas(PEP_T0, T1S)
    for s in (0.5, 1.0, 2.0):
        mname = f"pca8 x{s}"
        if ("peptides_func", mname) in done:
            continue
        ts = time.monotonic()
        per = []
        for Z, ip, ix in mols:
            Z0 = Z * (s * kap[0] / med)
            ZT, rec = full.run(Z0, ip, ix, PEP_T0, T=T_PRED, record=set(T1S))
            obs, r = partition_report(rec, ZT, ip, ix, kap, T1S)
            r["n_atoms"] = Z.shape[0]
            per.append(r)
        row = dict(dataset="peptides_func", metric=mname, t0=PEP_T0, kappa={str(k): v for k, v in kap.items()}, T=T_PRED,
                   molecules=len(per), seconds=time.monotonic() - ts,
                   **{k: float(np.mean([r[k] for r in per])) for k in per[0]},
                   exact={str(t1): float(np.mean([r[f"ari_t{t1}"] > 0.999 for r in per])) for t1 in T1S})
        append_row(path, row)
        stamp(f"  peptides {mname}: mean n_obs {row['n_obs']:.1f}, mean ARI@0 {row['ari_t0']:.3f}, "
              f"exact@0 {row['exact'][str(T1S[0])]:.2f}")


def run_trained():
    from sklearn.metrics import adjusted_rand_score
    path = out_path("trained")
    done = {r["state"] for r in read_rows(path)}
    node_rows = read_rows(out_path("node"))
    for name in NODE_DATASETS:
        files = sorted(glob.glob(f"{STATES}/{name}_gauss_cool_K*.pt"))
        if not files:
            continue
        D = load_node(name)
        n, y = D["n"], D["y"]
        indptr, indices = csr(n, D["src"], D["dst"])
        xt = torch.tensor(D["X"])
        for f in files:
            if os.path.basename(f) in done:
                continue
            parts = os.path.basename(f)[:-3].split("_")
            K = int([p for p in parts if p.startswith("K")][0][1:])
            split = int([p for p in parts if p.startswith("split")][0][5:])
            lr = float([p for p in parts if p.startswith("lr")][0][2:])
            p_drop = float([p for p in parts if p.startswith("p")][-1][1:])
            src_row = [r for r in node_rows if r["dataset"] == name and r["model"] == "gauss_cool" and r["depth"] == K
                       and r["split"] == split and r["lr"] == lr and r["dropout"] == p_drop]
            net = DeepNet("gauss_cool", K, xt.shape[1], int(y.max()) + 1, 0.0, hidden=HIDDEN)
            net.load_state_dict(torch.load(f, weights_only=True))
            net.eval()
            c = float(torch.exp(net.logc))
            with torch.no_grad():
                z0 = net.inp(xt).numpy()
            ts = time.monotonic()
            t1s = sorted({0, K})
            rec_t = set(t1s) | {T_TRAINED // 2}
            ZT, rec = full.run(z0 / math.sqrt(c), indptr, indices, models.T0_COOL, T=T_TRAINED, record=rec_t)
            kap = kappas(models.T0_COOL, t1s)
            obs, row = partition_report(rec, ZT, indptr, indices, kap, t1s)
            obs_half, _ = full.observed_partition(rec[T_TRAINED // 2], indptr, indices)
            te = D["splits"][split]["test"]

            def acc(Z):
                with torch.no_grad():
                    logits = net.out(torch.tensor(Z * math.sqrt(c), dtype=torch.float32))
                return float((logits.argmax(1).numpy()[te] == y[te]).mean())
            row.update(dataset=name, state=os.path.basename(f), depth=K, split=split, lr=lr, dropout=p_drop, c=c,
                       tuning_row_test=src_row[0]["test"] if src_row else None,
                       acc_trained_depth=acc(rec[K]), acc_T=acc(ZT), T=T_TRAINED,
                       stable=float(adjusted_rand_score(obs, obs_half)), ari_labels=float(adjusted_rand_score(y, obs)),
                       edges=edge_stats(rec[0], ZT, indptr, indices, obs, kap[0]), seconds=time.monotonic() - ts)
            append_row(path, row)
            stamp(f"  {name} K={K} split {split}: c {c:.3g}, n_obs {row['n_obs']}, ARI@0 {row['ari_t0']:.3f}, "
                  f"ARI@K {row[f'ari_t{K}']:.3f}, acc@K {row['acc_trained_depth']:.4f} acc@{T_TRAINED} {row['acc_T']:.4f}")


# ---------------------------------------------------------------------------------------------------
def run_time():
    for name in NODE_DATASETS:
        D, G, xt, yt = node_inputs(name)
        for K in (8, 64):
            for m in NODE_MODELS:
                sync()
                r = fit_node(D, G, xt, yt, 0, m, K, 0.01, 0.2, 0, max_epochs=3)
                sync()
                stamp(f"  {name} {m:12s} K={K:3d}: {r['seconds'] / r['epochs']:.2f} s/epoch")
    P = pep_prepare()
    for K in (8, 32):
        for m in ("gcn", "gat_res", "gauss_cool"):
            r = fit_pep(P, None, m, K, 0, max_batches=10)
            stamp(f"  peptides {m:12s} K={K:3d}: {r['seconds_per_batch']:.3f} s/batch, "
                  f"{r['seconds_per_batch'] * math.ceil(len(P['train']) / PEP_BATCH):.0f} s/epoch")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "time"
    {"predict": run_predict, "node": run_node, "peptides": run_peptides, "trained": run_trained,
     "time": run_time}[mode]()
    stamp("done")
