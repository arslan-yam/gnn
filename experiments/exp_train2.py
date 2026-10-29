"""E6b: deep GNNs on the three contextual-SBM regimes of E6, with stronger baselines
(residual, PairNorm, GCNII, APPNP, GRAND) and per-run tuning of lr and dropout on validation.
Writes results/train_E6b.json.  Usage: python -m experiments.exp_train2 [regime ...]   (env: ALT=1 alternative grid, RESUME=1 skip finished tasks)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

from experiments.exp_train import REGIMES, make_data

DEPTHS = [2, 8, 16, 32, 64, 128]
MODELS = ["gcn", "gcn_res", "gcn_pairnorm", "gcnii", "appnp", "sgc", "gat", "gat_res", "grand",
          "gauss_const", "gauss_cool"]
SEEDS = range(5)
OUT = "results/train_E6b.json"


ALT = os.environ.get("ALT") == "1"
if ALT:
    OUT = "results/train_E6b_alt.json"


def task(args):
    regime, model, K, seed = args
    import torch
    torch.set_num_threads(1)
    from gnnlimit.models import Graph, fit_tuned
    A, X, y, tr, va, te = make_data(regime, seed)
    r = fit_tuned(model, K, X, y, tr, va, te, Graph(A), seed, alt=ALT)
    r.update(regime=regime, model=model, depth=K, seed=seed)
    return r


def cost(t):
    _, m, K, _ = t
    from gnnlimit.models import ATTENTION_MODELS
    return K * (10 if m in ATTENTION_MODELS else 1)


if __name__ == "__main__":
    regimes = sys.argv[1:] or list(REGIMES)
    tasks = [(r, m, K, s) for r in regimes for K in DEPTHS for m in MODELS for s in SEEDS]
    tasks += [(r, "mlp", 0, s) for r in regimes for s in SEEDS]
    tasks.sort(key=cost, reverse=True)
    rows = []
    if os.environ.get("RESUME") == "1" and os.path.exists(OUT):  # keep finished tasks of an interrupted run
        with open(OUT) as f:
            rows = json.load(f)
        done = {(r["regime"], r["model"], r["depth"], r["seed"]) for r in rows}
        tasks = [tk for tk in tasks if tk not in done]
        print(f"resuming: {len(rows)} done, {len(tasks)} to run", flush=True)
    t = time.monotonic()
    with Pool(int(os.environ.get("WORKERS", 8))) as pool:
        for i, row in enumerate(pool.imap_unordered(task, tasks)):
            rows.append(row)
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                print(f"[{time.monotonic() - t:7.1f}s] {i + 1}/{len(tasks)}", flush=True)
                with open(OUT, "w") as f:
                    json.dump(rows, f, indent=1)
    print(f"done in {time.monotonic() - t:.1f}s")
