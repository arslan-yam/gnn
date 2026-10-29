"""Tables for E6b (CSBM regimes) and E7 (long-range task): test accuracy mean +- 95% CI over seeds,
for each model and depth, plus the best depth per model. Writes results/tables_train.md."""
import json
from collections import defaultdict

import numpy as np

ORDER = ["mlp", "gcn", "gcn_res", "gcn_pairnorm", "gcnii", "appnp", "sgc", "gat", "gat_res", "grand",
         "gauss_const", "gauss_cool", "qk_cool"]


def ci(v):
    v = np.asarray(v, float)
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), half


def table(rows, group_key, title):
    g = defaultdict(list)
    for r in rows:
        g[(r[group_key], r["model"], r["depth"])].append(r["test"])
    lines = [f"## {title}", ""]
    for grp in sorted({k[0] for k in g}):
        depths = sorted({k[2] for k in g if k[0] == grp and k[2] > 0})
        lines += [f"### {grp}", "", "| model | " + " | ".join(f"K={d}" for d in depths) + " | best |",
                  "|---|" + "---|" * (len(depths) + 1)]
        for m in ORDER:
            if m == "mlp":
                if (grp, m, 0) in g:
                    mu, h = ci(g[(grp, m, 0)])
                    lines.append(f"| mlp (K=0) | {mu:.3f} ± {h:.3f} |" + " |" * len(depths))
                continue
            cells, best = [], (-1, None)
            for d in depths:
                if (grp, m, d) in g:
                    mu, h = ci(g[(grp, m, d)])
                    cells.append(f"{mu:.3f} ± {h:.3f}")
                    if mu > best[0]:
                        best = (mu, d)
                else:
                    cells.append("–")
            if best[1] is not None:
                lines.append(f"| {m} | " + " | ".join(cells) + f" | {best[0]:.3f} (K={best[1]}) |")
        lines.append("")
    return lines


def load(path):
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        return []
    return d["rows"] if isinstance(d, dict) else d


def merge(base, alt, key):
    """Per (group, model, depth, seed): keep the run with the best validation accuracy over the base
    grid and the alternative-hyperparameter grid (8 configurations); ties go to the base run."""
    best = {}
    for r in base + alt:
        k = (r[key], r["model"], r["depth"], r["seed"])
        if k not in best or r["val"] > best[k]["val"]:
            best[k] = r
    return list(best.values())


FILES = {"E6b": ("results/train_E6b.json", "results/train_E6b_alt.json", "results/train_E6b_qk.json", "regime"),
         "E7": ("results/longrange_E7.json", "results/longrange_E7_alt.json", "results/longrange_E7_qk.json", "variant")}


def load_all(exp, root="."):
    """All models of E6b or E7 after merge; the qk_cool rows (both grids in one file, see exp_train_qk.py) are split
    by their `alt` flag so that ties go to the base grid, as for the other models."""
    import os
    base, alt, qk, key = (os.path.join(root, p) if p.endswith(".json") else p for p in FILES[exp])
    q = load(qk)
    return merge(load(base) + [r for r in q if not r["alt"]], load(alt) + [r for r in q if r["alt"]], key)


if __name__ == "__main__":
    out = []
    e6 = load_all("E6b")
    if e6:
        out += table(e6, "regime", "E6b: contextual SBM regimes (tuned over 8 configurations)")
    e7 = load_all("E7")
    if e7:
        out += table(e7, "variant", "E7: long-range rings (tuned over 8 configurations)")
    open("results/tables_train.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))
