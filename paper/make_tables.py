"""LaTeX tables for the paper, generated from the result files (no hand transcription).
Usage (from the repository root):  python3 paper/make_tables.py   -> paper/tables/*.tex
"""
import json
import math
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def t0_tex(t0):
    t0 = float(t0)
    e = math.log10(t0)
    return f"$10^{{{int(round(e))}}}$" if t0 >= 1000 and abs(e - round(e)) < 1e-9 else f"{int(t0)}"


def floor5(x):
    return math.floor(x * 1e5) / 1e5


def ceil5(x):
    return math.ceil(x * 1e5) / 1e5


def kappa_table():
    rows = load("results/theory_E1.json")["E1_kappa"]
    lines = [r"\begin{tabular}{rcccc}", r"\toprule",
             r"$t_0$ & validated enclosure of $\kappa(t_0)$ & lower bound & upper bound & $1+\frac{\log\log t_0}{\log t_0}$\\",
             r"\midrule"]
    for r in rows:
        lines.append(f"{t0_tex(r['t0'])} & $[{floor5(r['klo']):.5f},\\ {ceil5(r['khi']):.5f}]$ & "
                     f"{floor5(r['lower']):.4f} & {ceil5(r['upper']):.4f} & {r['asym']:.4f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tab_kappa.tex"), "w").write("\n".join(lines) + "\n")


def rigorous_table():
    j = load("results/rigorous_mp.json")
    name = {(0, 0): "one cluster", (1, 0): r"$\{1\},\{2,3\}$", (0, 1): r"$\{1,2\},\{3\}$", (1, 1): "three clusters"}
    lines = [r"\begin{tabular}{ccccc}", r"\toprule",
             r"$u_0$ & $v_0$ & proved limit partition & gap 1 & certified at layer $T$\\", r"\midrule"]
    for r in j["three_layer"]:
        lab = tuple(r["labels"])
        lines.append(f"{r['g0'][0]} & {r['g0'][1]} & {name[lab]} & {'persists' if lab[0] else 'merges'} & {r['T']}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tab_rigorous.tex"), "w").write("\n".join(lines) + "\n")


def rho_table():
    j = load("results/theory_E1e.json")
    t0s = j["t0_enc"]
    lines = [r"\begin{tabular}{r" + "c" * len(t0s) + "}", r"\toprule",
             r"$\rho$ & " + " & ".join("$t_0=" + t0_tex(t).strip("$") + "$" for t in t0s) + r"\\", r"\midrule"]
    for rho in j["rhos"]:
        cells = []
        for t0 in t0s:
            e = [r for r in j["enclosures"] if r["rho"] == rho and r["t0"] == t0][0]
            cells.append(f"$[{math.floor(e['klo'] * 1e4) / 1e4:.4f},{math.ceil(e['khi'] * 1e4) / 1e4:.4f}]$")
        lines.append(f"{rho:g} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tab_rho.tex"), "w").write("\n".join(lines) + "\n")


def L_tex(L):
    if L < 1000:
        return f"{L:.3g}"
    m, e = f"{L:.2g}".split("e+")
    return (m + r"\cdot " if m != "1" else "") + f"10^{{{int(e)}}}"


def residual_table():
    """Continuum-limit residuals of the two-term law (proved) and the three-term formula (formal)."""
    j = load("results/theory_E1e.json")
    Ls = j["L_grid"]
    pick = [min(range(len(Ls)), key=lambda i: abs(math.log(Ls[i]) - math.log(x))) for x in (10, 100, 1e3, 1e4, 1e5)]
    lines = [r"\begin{tabular}{l" + "c" * len(pick) + "}", r"\toprule",
             r"$\lambda=2\rho$ & " + " & ".join(f"$L={L_tex(Ls[i])}$" for i in pick) + r"\\", r"\midrule"]
    for rho in j["rhos"]:
        c = j["continuum"][str(rho)]
        lines.append(f"{2 * rho:g}, two terms & " + " & ".join(f"${c['residual'][i]:+.3f}$" for i in pick) + r"\\")
        lines.append(f"{2 * rho:g}, three terms & " + " & ".join(f"${c['residual3'][i]:+.3f}$" for i in pick) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tab_residual.tex"), "w").write("\n".join(lines) + "\n")


MODEL_TEX = {"mlp": "MLP ($K=0$)", "gcn": "GCN", "gcn_res": "GCN + residual", "gcn_pairnorm": "GCN + PairNorm",
             "gcnii": "GCNII", "appnp": "APPNP", "sgc": "SGC", "gat": "GAT", "gat_res": "GAT + residual",
             "grand": "GRAND (linear)", "gauss_const": "Gaussian, constant $T$", "gauss_cool": "Gaussian, cooled",
             "qk_cool": "dot-product, cooled"}


def depth_block(rows, group, title, ncols, metric="test", digits=3, ci=True):
    """Rows of one block of a depth-sweep table: mean and 95% CI half-width over seeds/splits; the best mean
    in each depth column is bold."""
    import numpy as np
    depths = sorted({r["depth"] for r in rows if r["depth"] > 0})
    cell = {}
    for m in MODEL_TEX:
        for d in ([0] if m == "mlp" else depths):
            v = np.array([r[metric] for r in rows if r["model"] == m and r["depth"] == d], float)
            if len(v):
                h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                cell[m, d] = (v.mean(), h)
    best = {d: max(cell[m, d][0] for m in MODEL_TEX if (m, d) in cell) for d in depths}
    fmt = lambda x: f"{x:.{digits}f}".lstrip("0") if x < 1 else f"{x:.{digits}f}"
    pm = lambda h: rf"\,{{\tiny$\pm${fmt(h)}}}" if ci else ""
    lines = [rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{title}}}}}\\"]
    for m, name in MODEL_TEX.items():
        if m == "mlp":
            if (m, 0) in cell:
                mu, h = cell[m, 0]
                lines.append(rf"{name} & \multicolumn{{{len(depths)}}}{{c}}{{{fmt(mu)}{pm(h)}}}\\")
            continue
        if not any((m, d) in cell for d in depths):
            continue
        out = []
        for d in depths:
            if (m, d) not in cell:
                out.append("--")
                continue
            mu, h = cell[m, d]
            v = fmt(mu)
            out.append((rf"\textbf{{{v}}}" if mu >= best[d] - 1e-12 else v) + pm(h))
        lines.append(f"{name} & " + " & ".join(out) + r"\\")
    return depths, lines


def train_tables():
    import sys
    sys.path.insert(0, ROOT)
    from gnnlimit.summarize_train import load_all
    for name, exp, key, groups in [
            ("tab_train_e6b", "E6b", "regime",
             [("heterophilous", "heterophilous"), ("homophilous_clean", "homophilous, clean"),
              ("homophilous_noisy", "homophilous, noisy")]),
            ("tab_train_e7", "E7", "variant",
             [("sparse_cross", "sparse cross-edges ($p_x=0.05$)"), ("dense_cross", "dense cross-edges ($p_x=0.2$)")])]:
        rows = load_all(exp, ROOT)
        body, depths = [], None
        for g, title in groups:
            depths, lines = depth_block([r for r in rows if r[key] == g], g, title, 7)
            body += ([r"\midrule"] if body else []) + lines
        lines = [r"\begin{tabular}{l" + "c" * len(depths) + "}", r"\toprule",
                 "model & " + " & ".join(f"$K={d}$" for d in depths) + r"\\", r"\midrule"] + body + \
                [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(lines) + "\n")


def read_jsonl(path):
    with open(os.path.join(ROOT, path)) as f:
        return [json.loads(line) for line in f if line.strip()]


def predict_real_table():
    """E8: threshold-graph prediction of the partition after 4000 layers on the real graphs."""
    names = {"roman_empire": "Roman-empire", "amazon_ratings": "Amazon-ratings", "peptides_func": "Peptides-func"}
    metric_tex = lambda m: (f"PCA, scale ${m.split('x')[1].rstrip('0').rstrip('.')}$" if m.startswith("pca")
                            else f"LDA, margin ${m.split()[-1].rstrip('0').rstrip('.')}\\kappa$")
    pct = lambda a, b: f"{100 * a / b:.1f}" if b else "--"
    lines = [r"\begin{tabular}{llrrcccc}", r"\toprule",
             r"graph & metric & clusters & predicted & \multicolumn{2}{c}{ARI, $t_1=0$ / $20$} & \multicolumn{2}{c}{false persist / merge (\%)}\\",
             r"\midrule"]
    rows = read_jsonl("results/real_E8_predict.jsonl")
    last = None
    for ds in names:
        for r in [r for r in rows if r["dataset"] == ds]:
            if last is not None and ds != last:
                lines.append(r"\midrule")
            first = ds != last
            last = ds
            if ds == "peptides_func":
                cl, pr, fp, fm = f"{r['n_obs']:.1f}", f"{r['n_pred_t0']:.1f}", "--", "--"
            else:
                e = r["edges"]
                cl, pr = f"{r['n_obs']:,}".replace(",", "{,}"), f"{r['n_pred_t0']:,}".replace(",", "{,}")
                fp = pct(e["part_false_persist"], e["part_pred_persist"])
                fm = pct(e["part_false_merge"], e["edges"] - e["part_pred_persist"])
            lines.append(f"{names[ds] if first else ''} & {metric_tex(r['metric'])} & {cl} & {pr} & {r['ari_t0']:.3f} & "
                         f"{r['ari_t20']:.3f} & {fp} & {fm}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tab_predict_real.tex"), "w").write("\n".join(lines) + "\n")


def real_train_table():
    """E8 depth sweeps on the node graphs: split 0 picks the learning rate per (model, depth) on validation accuracy,
    and the table averages the chosen run with splits 1 and 2."""
    rows = read_jsonl("results/real_E8_node.jsonl")
    best = {}
    for r in rows:
        k = (r["dataset"], r["model"], r["depth"])
        if r["split"] == 0 and (k not in best or r["val"] > best[k]["val"]):
            best[k] = r
    rows = [r for r in rows if r["split"] > 0 or r is best[r["dataset"], r["model"], r["depth"]]]
    body, depths = [], None
    for ds, title in [("roman_empire", "Roman-empire"), ("amazon_ratings", "Amazon-ratings")]:
        depths, lines = depth_block([r for r in rows if r["dataset"] == ds], ds, title, 6)
        body += ([r"\midrule"] if body else []) + lines
    lines = [r"\begin{tabular}{l" + "c" * len(depths) + "}", r"\toprule",
             "model & " + " & ".join(f"$K={d}$" for d in depths) + r"\\", r"\midrule"] + body + \
            [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tab_train_e8.tex"), "w").write("\n".join(lines) + "\n")
    rows = read_jsonl("results/real_E8_peptides.jsonl")
    depths, body = depth_block(rows, "peptides_func", "Peptides-func (average precision, one seed)", 4, ci=False)
    lines = [r"\begin{tabular}{l" + "c" * len(depths) + "}", r"\toprule",
             "model & " + " & ".join(f"$K={d}$" for d in depths) + r"\\", r"\midrule"] + body + \
            [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tab_train_e8_pep.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    predict_real_table()
    real_train_table()
    train_tables()
    kappa_table()
    rigorous_table()
    rho_table()
    residual_table()
    print("tables written to", OUT)
