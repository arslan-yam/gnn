"""E6b/E7 for the cooled dot-product attention network qk_cool (Section 7): same tasks, depths, seeds and 8-configuration
tuning as exp_train2.py and exp_longrange.py (base grid with the default temperature init, alternative grid with the
cold start 1/30). The selected network of every run is saved to results/qk_states/ for the analysis in
exp_learned.py trained.

Writes results/train_E6b_qk.json and results/longrange_E7_qk.json (rows of both grids, `alt` marks the grid; merged
with the other models by summarize_train.merge). Usage: python -m experiments.exp_train_qk   (env: WORKERS; finished tasks are kept)
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from experiments import exp_longrange
from experiments import exp_train

MODEL = "qk_cool"
DEPTHS = [2, 8, 16, 32, 64, 128]
SEEDS = range(5)
OUT = {"E6b": "results/train_E6b_qk.json", "E7": "results/longrange_E7_qk.json"}
STATES = "results/qk_states"


def state_path(exp, group, K, seed, alt):
    return os.path.join(STATES, f"{exp}__{group}__K{K}__s{seed}__{'alt' if alt else 'base'}.npz")


def task(args):
    exp, group, K, seed, alt = args
    import torch
    torch.set_num_threads(1)
    from gnnlimit.models import Graph, fit_tuned
    if exp == "E6b":
        A, X, y, tr, va, te = exp_train.make_data(group, seed)
    else:
        A, X, y, tr, va, te, _ = exp_longrange.make_longrange(seed, **exp_longrange.VARIANTS[group])
    r = fit_tuned(MODEL, K, X, y, tr, va, te, Graph(A), seed, alt=alt)
    st = r.pop("state", None)
    if st is not None:
        np.savez(state_path(exp, group, K, seed, alt), **st)
    r.update(model=MODEL, depth=K, seed=seed, exp=exp, **({"regime": group} if exp == "E6b" else {"variant": group}))
    return r


def load_rows(exp):
    if not os.path.exists(OUT[exp]):
        return []
    with open(OUT[exp]) as f:
        return json.load(f)


if __name__ == "__main__":
    os.makedirs(STATES, exist_ok=True)
    tasks = [("E6b", g, K, s, alt) for g in exp_train.REGIMES for K in DEPTHS for s in SEEDS for alt in (False, True)]
    tasks += [("E7", g, K, s, alt) for g in exp_longrange.VARIANTS for K in DEPTHS for s in SEEDS for alt in (False, True)]
    rows = {exp: load_rows(exp) for exp in OUT}
    done = {(r["exp"], r.get("regime", r.get("variant")), r["depth"], r["seed"], r["alt"]) for exp in OUT for r in rows[exp]}
    tasks = [t for t in tasks if t not in done]
    tasks.sort(key=lambda t: t[2], reverse=True)  # deepest first
    print(f"{sum(len(v) for v in rows.values())} done, {len(tasks)} to run", flush=True)
    t = time.monotonic()
    with Pool(int(os.environ.get("WORKERS", 3))) as pool:
        for i, row in enumerate(pool.imap_unordered(task, tasks)):
            rows[row["exp"]].append(row)
            with open(OUT[row["exp"]], "w") as f:
                json.dump(rows[row["exp"]], f, indent=1)
            g = row.get("regime", row.get("variant"))
            print(f"[{time.monotonic() - t:7.1f}s] {i + 1}/{len(tasks)} {row['exp']} {g} K={row['depth']} s={row['seed']} "
                  f"alt={row['alt']}: val {row['val']:.3f} test {row['test']:.3f} c {row.get('c', float('nan')):.3g} "
                  f"({row['seconds_total']:.0f}s)", flush=True)
    print(f"done in {time.monotonic() - t:.1f}s")
