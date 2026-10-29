"""Figures for the paper (E1-E7). Reads results/*.json|npz and theory/check_*.json, writes paper/figures/*.pdf
(and .png previews).  Usage (from the repository root):  python3 paper/make_figures.py [figure function names, default all]
"""
import json
import math
import os
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# Reference palette (light mode, print).  Categorical slots in fixed order; blue ramp for ordered series.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RAMP = {150: "#b7d3f6", 250: "#86b6ef", 350: "#5598e7", 450: "#2a78d6", 550: "#1c5cab", 600: "#184f95",
        650: "#104281", 700: "#0d366b"}
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FULL = 5.5  # text width in inches

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.3, "lines.markersize": 3.6,
    "legend.frameon": False, "legend.handlelength": 1.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "mathtext.fontset": "dejavusans",
})


def load_json(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def load_npz(path):
    return np.load(os.path.join(ROOT, path))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)


def panel(ax, letter):
    ax.set_title(f"({letter})", loc="left", fontweight="bold", color=INK, pad=3)


def phi(x):
    return np.where(x < 1, np.maximum(x, 1 - x / 2), 1.0)


# ---------------------------------------------------------------------------------------------------------------
# Fig. 1: certified three-layer phase diagram and its normalised boundary
def fig_phase():
    z = load_npz("results/theory_E2.npz")
    e2 = load_json("results/theory_E2.json")
    g, phase = z["E2_grid"], z["E2_phase"]  # phase[i, j]: u0 = g[i] (gap 1), v0 = g[j] (gap 2)
    kap = float(np.mean(e2["E2"]["kappa"]))
    ncl = np.full(phase.shape, np.nan)
    ncl[phase == 0], ncl[(phase == 1) | (phase == 2)], ncl[phase == 3] = 1, 2, 3

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.45), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.32))
    h = g[1] - g[0]
    ext = [g[0] - h / 2, g[-1] + h / 2, g[0] - h / 2, g[-1] + h / 2]
    cmap = ListedColormap([RAMP[150], RAMP[350], RAMP[600]])
    a.imshow(ncl, origin="lower", extent=ext, cmap=cmap, vmin=0.5, vmax=3.5, interpolation="nearest", aspect="equal")
    ii, jj = np.where(phase == -1)
    a.scatter(g[jj], g[ii], s=0.6, color=ORANGE, lw=0, zorder=3)  # not certified (0.36%)
    a.axhline(kap, color=INK, lw=0.8, ls=(0, (3, 2)))
    a.axvline(kap, color=INK, lw=0.8, ls=(0, (3, 2)))
    a.grid(False)
    a.set_xlabel(r"initial gap 2, $v_0/\sqrt{c}$")
    a.set_ylabel(r"initial gap 1, $u_0/\sqrt{c}$")
    a.set_xlim(0, 3), a.set_ylim(0, 3)
    a.text(0.55, 0.55, "one cluster", ha="center", va="center", color=INK, fontsize=7)
    a.text(2.25, 0.55, "{1,2},{3}", ha="center", va="center", color="white", fontsize=7)
    a.text(0.55, 2.3, "{1},{2,3}", ha="center", va="center", color="white", fontsize=7)
    a.text(2.25, 2.3, "three\nclusters", ha="center", va="center", color="white", fontsize=7)
    a.text(kap + 0.04, 2.92, r"$\kappa(t_0)$", color=INK, fontsize=6.8, va="top")
    a.annotate("nonlocal\nrescue", xy=(0.95, 1.3), xytext=(1.75, 1.15), fontsize=6.5, color=INK, ha="left",
               va="center", arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=6))
    panel(a, "a")

    t0s, v0, us = z["E2b_t0"], z["E2b_v0"], z["E2b_ustar"]
    kb = z["E2b_kappa"]
    shades = [RAMP[250], RAMP[350], RAMP[450], RAMP[550], RAMP[700]]
    names = {5: "5", 11: "11", 50: "50", 1000: "10^3", 100000: "10^5"}
    for k in range(len(t0s)):
        x, y = v0[k] / kb[k], us[k] / kb[k]
        m = x < 0.985
        b.plot(x[m], y[m], color=shades[k], lw=1.2, label=rf"$t_0={names[int(t0s[k])]}$")
    xs = np.linspace(0, 1.6, 400)
    b.plot(xs[xs < 1], phi(xs[xs < 1]), color=INK, lw=1.0, ls=(0, (3, 2)), label=r"scaling limit $\varphi$")
    b.plot([1, 1.9], [1, 1], color=INK, lw=1.0, ls=(0, (3, 2)))
    b.axhline(2 / 3, color=MUTED, lw=0.6, ls=":")
    b.text(0.03, 2 / 3 - 0.012, r"$2/3$", color=INK2, fontsize=6.5, va="top")
    b.set_xlim(0, 1.9), b.set_ylim(0.6, 1.06)
    b.set_xlabel(r"neighbouring gap, $v_0/\kappa(t_0)$")
    b.set_ylabel(r"persistence threshold, $u^*(v_0)/\kappa(t_0)$")
    b.legend(loc="lower right", ncol=1)
    panel(b, "b")
    save(fig, "fig1_phase")


# ---------------------------------------------------------------------------------------------------------------
# Fig. 2: the two-block law
def fig_two_block():
    z = load_npz("results/theory_E1.npz")
    j = load_json("results/theory_E1.json")
    fig, axs = plt.subplots(2, 2, figsize=(FULL, 4.1), gridspec_kw=dict(hspace=0.5, wspace=0.3))
    a, b, c, d = axs.ravel()

    # (a) kappa(t0): validated enclosures, explicit bounds, sharp asymptotics
    t = z["E1_t0_dense"]
    a.fill_between(t, z["E1_lower_dense"], z["E1_upper_dense"], color=RAMP[150], lw=0, label="explicit bounds")
    tt = t[t >= 8]
    LL = np.log(tt)
    a.plot(tt, 1 + (np.log(LL) - 0.5 * np.log(np.log(LL)) + 0.5 * np.log(2.0)) / LL, color=INK, lw=0.9,
           ls=(0, (3, 2)), label="two-term asymptotics (v)")
    a.plot(t, z["E1_asym_dense"], color=MUTED, lw=0.8, ls=":", label="leading term only")
    rows = [r for r in j["E1_kappa"]]
    tk = np.array([r["t0"] for r in rows])
    lo, hi = np.array([r["klo"] for r in rows]), np.array([r["khi"] for r in rows])
    a.errorbar(tk, (lo + hi) / 2, yerr=(hi - lo) / 2, fmt="o", color=BLUE, mec="white", mew=0.6, ms=4, elinewidth=1,
               capsize=0, label=r"validated enclosure of $\kappa$")
    a.set_xscale("log")
    a.set_ylim(0.95, 2.2)
    a.axhline(1, color=MUTED, lw=0.6, ls=":")
    a.set_xlabel(r"starting index $t_0$")
    a.set_ylabel(r"critical gap $\kappa(t_0)$  [units of $\sqrt{c}$]")
    a.legend(loc="upper right")
    panel(a, "a")

    # (b) topology at second order: scaled shift (kappa(rho) - kappa(1)) log t0 -> (1/2) log rho
    e = load_json("results/theory_E1e.json")
    Lg = np.array(e["L_grid"])
    k1 = np.array(e["continuum"]["1.0"]["kappa"])
    shades = {0.1: RAMP[250], 0.25: RAMP[450], 0.5: RAMP[650]}
    enc = {(r["rho"], r["t0"]): r for r in e["enclosures"]}
    for rho, col in shades.items():
        kr = np.array(e["continuum"][str(rho)]["kappa"])
        b.plot(Lg, (kr - k1) * Lg, color=col, lw=1.2, label=rf"$\rho={rho:g}$")
        b.axhline(0.5 * np.log(rho), color=col, lw=0.8, ls=(0, (3, 2)))
        for t0 in e["t0_enc"]:
            r, r1 = enc[(rho, t0)], enc[(1.0, t0)]
            L = np.log(t0)
            lo, hi = (r["klo"] - r1["khi"]) * L, (r["khi"] - r1["klo"]) * L
            b.errorbar(L, (lo + hi) / 2, yerr=(hi - lo) / 2, fmt="o", color=col, ms=3, mec="white", mew=0.4,
                       elinewidth=0.8, capsize=0)
    b.plot([], [], color=MUTED, lw=1.2, label="continuum limit")
    b.plot([], [], "o", color=MUTED, ms=3, label="validated (discrete)")
    b.plot([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)), label=r"limit $\frac{1}{2}\log\rho$")
    b.set_xscale("log")
    b.set_ylim(-1.3, 0.42)
    b.set_xlabel(r"$L=\log t_0$")
    b.set_ylabel(r"$[\kappa(t_0;\rho,\rho)-\kappa(t_0;1,1)]\,\log t_0$")
    b.legend(loc="upper right", ncol=2, fontsize=6.0, columnspacing=0.8)
    panel(b, "b")

    # (c) first-order jump at the threshold
    d0, dT, st = z["E1_jump_d0"], z["E1_jump_dT"], z["E1_jump_status"]
    kj = [r for r in j["E1_rho_enclosures_t0_11"] if r["rho"] == 1.0][0]
    c.axvspan(kj["klo"], kj["khi"], color=RAMP[150], lw=0)
    m = st == 0
    c.plot(d0[m], np.zeros(m.sum()), "o", color=ORANGE, ms=2.2, mew=0, label="certified merge")
    c.plot(d0[~m], dT[~m], "o", color=BLUE, ms=2.2, mew=0, label="certified persist")
    c.axhline(1, color=MUTED, lw=0.6, ls=":")
    c.text(0.92, 1.03, r"$\sqrt{c}$", color=INK2, fontsize=6.5, va="bottom")
    c.annotate("", xy=(kj["khi"], dT[~m].min()), xytext=(kj["khi"], 0.0),
               arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=INK, mutation_scale=6, shrinkA=1, shrinkB=1))
    c.text(kj["khi"] + 0.04, 0.6, "jump " + r"$\geq\sqrt{c}$", fontsize=6.5, color=INK, va="center")
    c.set_xlabel(r"initial gap $d_0/\sqrt{c}$  ($t_0=11$)")
    c.set_ylabel(r"gap after $10^6$ layers, $d_T/\sqrt{c}$")
    c.set_ylim(-0.1, 2.15)
    c.legend(loc="upper left")
    panel(c, "c")

    # (d) critical slowing: trajectories just above kappa-hat shadow the critical curve kappa(t0+t)
    cps, tr, hs = z["E1_crit_cps"], z["E1_crit_traj"], z["E1_crit_h"]
    x = cps + 1.0
    shades = {1e-9: RAMP[250], 1e-4: RAMP[450], 1e-2: RAMP[700]}
    for k, h in enumerate(hs):
        if h not in shades:
            continue
        y = tr[k].copy()
        cut = np.argmax(y == 0) if np.any(y == 0) else len(y)
        xx, yy = x[:cut], y[:cut]
        if cut < len(y):  # draw the final drop to zero at the first checkpoint where the gap is gone
            xx, yy = np.append(xx, x[cut]), np.append(yy, 0.0)
        d.plot(xx, yy, color=shades[h], lw=1.2, label=rf"$d_0=\hat\kappa+10^{{{int(np.log10(h))}}}$")
    cc = j["E1_critical_curve"]
    tc = np.array([r["t"] for r in cc]) + 1.0
    clo, chi = np.array([r["klo"] for r in cc]), np.array([r["khi"] for r in cc])
    d.errorbar(tc, (clo + chi) / 2, yerr=(chi - clo) / 2, fmt="s", color=INK, ms=3, elinewidth=1, capsize=0,
               label=r"enclosure of $\kappa(t_0+t)$")
    d.set_xscale("log")
    d.set_ylim(-0.05, 1.7)
    d.set_xlabel(r"layer $t+1$  ($t_0=11$)")
    d.set_ylabel(r"gap $d_t/\sqrt{c}$")
    d.legend(loc="lower left", fontsize=6.3)
    panel(d, "d")
    save(fig, "fig2_two_block")


# ---------------------------------------------------------------------------------------------------------------
# Fig. 3: five-layer chains (E3): one-sided errors and the early-depth rule
def fig_chains():
    z = load_npz("results/theory_E3.npz")
    j = load_json("results/theory_E3.json")["E3"]
    kap = float(np.mean(j["kappa"]))
    G0, L, cert = z["E3_G0"], z["E3_L"], z["E3_cert"].astype(bool)
    g, lab = G0[cert].ravel() / kap, L[cert].ravel()
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.0), gridspec_kw=dict(wspace=0.3))
    bins = np.linspace(0, 2.6, 53)
    a.hist([g[lab == 0], g[lab == 1]], bins=bins, stacked=True, color=[ORANGE, BLUE], label=["merges", "persists"],
           edgecolor="white", lw=0.3)
    a.axvline(1, color=INK, lw=0.8, ls=(0, (3, 2)))
    a.set_ylim(0, 700)
    a.text(1.0, 690, r"isolated-pair rule: persist iff gap $\geq\kappa$", fontsize=6.3, color=INK, va="top", ha="center",
           bbox=dict(fc="white", ec="none", pad=0.5))
    a.set_xlabel(r"initial gap $/\kappa(t_0)$")
    a.set_ylabel("gaps")
    a.set_xlim(-0.1, 3.0)
    a.legend(loc="center right", bbox_to_anchor=(1.0, 0.5), title="final gap", title_fontsize=6.8)
    panel(a, "a")

    t1 = np.array(j["early_t1"]) + 1
    acc = 100 * np.array(j["early_acc"])
    b.plot(t1, acc, "o-", color=BLUE, mec="white", mew=0.5)
    for tt, aa in zip(t1, acc):
        if tt in (1, 3, 11, 1001):
            b.annotate(f"{aa:.2f}%" if aa > 99.9 else f"{aa:.1f}%", (tt, aa), textcoords="offset points", xytext=(2, -9), fontsize=6.3, color=INK2)
    b.set_xscale("log")
    b.set_ylim(90, 100.6)
    b.set_xlabel(r"rule applied after $t_1$ layers ($t_1+1$)")
    b.set_ylabel("gaps predicted correctly (%)")
    panel(b, "b")
    save(fig, "fig3_chains")


# ---------------------------------------------------------------------------------------------------------------
# Fig. 4: noisy contextual SBMs (E4)
def fig_sbm():
    rows = load_json("results/graphs_E4.json")["rows"]
    cert = load_json("theory/check_E4.json")["summary"]
    sig = np.array(sorted({r["sigma"] for r in rows}))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL, 1.95), gridspec_kw=dict(wspace=0.42, width_ratios=[1.35, 1, 1]))

    s = np.array([r["sigma"] for r in cert])
    n = np.array([r["n"] for r in cert], float)
    series = [("survives", "partition survives", BLUE, "o"), ("hull", "certified: hull", ORANGE, "s"),
              ("ball", "certified: ball", AQUA, "^")]
    for key, lab, col, mk in series:
        a.plot(s, 100 * np.array([r[key] for r in cert]) / n, marker=mk, color=col, ms=3, mec="white", mew=0.4,
               label=lab)
    a.set_xlabel(r"feature noise $\sigma$")
    a.set_ylabel("runs (%)")
    a.set_ylim(-4, 104), a.set_xlim(-0.02, 0.75), a.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    a.legend(loc="center right", bbox_to_anchor=(1.02, 0.55))
    panel(a, "a")

    def stat(key):
        m, e = [], []
        for sg in sig:
            v = np.array([r[key] for r in rows if r["sigma"] == sg], float)
            m.append(v.mean()), e.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        return np.array(m), np.array(e)

    for key, lab, col, mk in [("ari_t0", r"rule at $t_1=0$", BLUE, "o"), ("ari_t20", r"rule at $t_1=20$", ORANGE, "s")]:
        m, e = stat(key)
        b.errorbar(sig, m, yerr=e, marker=mk, color=col, ms=3, mec="white", mew=0.4, capsize=0, elinewidth=0.8,
                   label=lab)
    b.set_xlabel(r"feature noise $\sigma$")
    b.set_ylabel("ARI(predicted, final)")
    b.set_ylim(0.45, 1.03)
    b.legend(loc="lower left")
    panel(b, "b")

    m, e = stat("n_obs")
    c.errorbar(sig, m, yerr=e, marker="o", color=BLUE, ms=3, mec="white", mew=0.4, capsize=0, elinewidth=0.8)
    c.axhline(4, color=MUTED, lw=0.6, ls=":")
    c.text(0.0, 5.0, "planted $K=4$", fontsize=6.3, color=INK2)
    c.set_yscale("log")
    c.set_xlabel(r"feature noise $\sigma$")
    c.set_ylabel("final clusters")
    panel(c, "c")
    save(fig, "fig4_sbm")


# ---------------------------------------------------------------------------------------------------------------
# Fig. 5: learned metric (E5)
def fig_metric():
    j = load_json("results/graphs_E5.json")
    depths = j["depths"]
    x = np.arange(len(depths))
    names = [("identity", r"$W=I$", BLUE, "o"), ("nuisance projection", "nuisance projection", ORANGE, "s"),
             ("learned (LDA, too small)", r"LDA, scale $<\kappa$", AQUA, "^"),
             ("learned (LDA, calibrated)", r"LDA, calibrated to $2\kappa$", YELLOW, "D")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.0), gridspec_kw=dict(wspace=0.3))
    for key, lab, col, mk in names:
        acc = np.array([r[key]["acc"] for r in j["runs"]])
        ncl = np.array([r[key]["nclusters"] for r in j["runs"]], float)
        a.plot(x, acc.mean(0), marker=mk, color=col, ms=3, mec="white", mew=0.4, label=lab)
        b.plot(x, np.exp(np.log(ncl).mean(0)), marker=mk, color=col, ms=3, mec="white", mew=0.4, label=lab)
    for ax in (a, b):
        ax.set_xticks(x[::2])
        ax.set_xticklabels([str(d) for d in depths[::2]])
        ax.set_xlabel("depth (layers)")
    a.set_ylabel("node accuracy")
    a.set_ylim(0.25, 1.04)
    b.set_yscale("log")
    b.set_ylabel("feature clusters (geom. mean)")
    b.legend(loc="upper right")
    panel(a, "a"), panel(b, "b")
    save(fig, "fig5_metric")


# ---------------------------------------------------------------------------------------------------------------
# Fig. 6: trained deep networks on the synthetic tasks (E6b, E7)
# One colour and marker per model, in the fixed categorical order, in every depth figure.
MODEL_STYLE = {"gauss_cool": (BLUE, "o", "cooled Gaussian"), "gcn": (ORANGE, "s", "GCN"),
               "gcn_res": (AQUA, "^", "GCN + residual"), "gcn_pairnorm": (YELLOW, "D", "GCN + PairNorm"),
               "gcnii": (MAGENTA, "v", "GCNII"), "gat_res": (GREEN, "P", "GAT + residual"),
               "grand": (VIOLET, "X", "GRAND (linear)"), "qk_cool": (RED, "*", "cooled dot-product")}


def depth_panel(ax, acc, depths, mlp):
    """acc: model -> list of mean test accuracies over depths; mlp: depth-0 baseline."""
    x = np.log2(depths)
    ax.axhline(mlp, color=MUTED, lw=0.9, ls=(0, (4, 2)), zorder=1)
    for m, (col, mk, _) in reversed(list(MODEL_STYLE.items())):
        if m in acc:
            ax.plot(x, acc[m], marker=mk, color=col, ms=4.4 if mk == "*" else 3.4, mec="white", mew=0.4,
                    lw=1.6 if m == "gauss_cool" else 1.0, zorder=4 if m == "gauss_cool" else 3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("depth $K$")


def fig_depth_synthetic():
    sys.path.insert(0, ROOT)
    from gnnlimit.summarize_train import load_all
    groups = [("E6b", "regime", "heterophilous", "CSBM, heterophilous"),
              ("E6b", "regime", "homophilous_clean", "CSBM, homoph., clean"),
              ("E6b", "regime", "homophilous_noisy", "CSBM, homoph., noisy"),
              ("E7", "variant", "sparse_cross", "rings, sparse cross-edges"),
              ("E7", "variant", "dense_cross", "rings, dense cross-edges")]
    rows = {"E6b": load_all("E6b", ROOT), "E7": load_all("E7", ROOT)}
    fig, axs = plt.subplots(2, 3, figsize=(FULL, 3.5), gridspec_kw=dict(wspace=0.28, hspace=0.62))
    for k, (ax, (exp, gkey, g, title)) in enumerate(zip(axs.flat, groups)):
        rs = [r for r in rows[exp] if r[gkey] == g]
        depths = sorted({r["depth"] for r in rs if r["depth"] > 0})
        mean = lambda m, d: np.mean([r["test"] for r in rs if r["model"] == m and r["depth"] == d])
        depth_panel(ax, {m: [mean(m, d) for d in depths] for m in MODEL_STYLE}, depths, mean("mlp", 0))
        ax.set_ylim(0.28, 1.02)
        ax.set_title(f"({'abcde'[k]}) {title}", loc="left", color=INK, pad=3)
        if k % 3 == 0:
            ax.set_ylabel("test accuracy")
    leg = axs.flat[5]
    leg.axis("off")
    handles = [plt.Line2D([], [], color=c, marker=mk, ms=4.4 if mk == "*" else 3.4, mec="white", mew=0.4, lw=1.6 if m == "gauss_cool" else 1.0)
               for m, (c, mk, _) in MODEL_STYLE.items()]
    handles.append(plt.Line2D([], [], color=MUTED, lw=0.9, ls=(0, (4, 2))))
    leg.legend(handles, [lab for _, _, lab in MODEL_STYLE.values()] + ["MLP (no graph)"], loc="center", fontsize=6.8)
    save(fig, "fig6_depth_synthetic")


def fig_learned():
    """E9: (a) critical weaker margin of a two-block pair against the partner margin; (b) directed rule on SBMs."""
    tb = load_json(os.path.join(ROOT, "results/learned_twoblock.json"))["rows"]
    gr = load_json(os.path.join(ROOT, "results/learned_graphs.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 1.95), gridspec_kw=dict(wspace=0.32))
    styles = {(11.0, 0.0): (BLUE, "o", "-"), (11.0, 1.0): (ORANGE, "s", "-"),
              (1e3, 0.0): (BLUE, "o", (0, (3, 1.5))), (1e3, 1.0): (ORANGE, "s", (0, (3, 1.5)))}
    for (t0, om), (col, mk, ls) in styles.items():
        rs = sorted((r for r in tb if r["t0"] == t0 and r["theta"] == om), key=lambda r: r["MY"])
        k2 = rs[0]["kappa_sq"]
        x = [r["MY"] / k2 for r in rs]
        y = [0.5 * (r["MX_lo"] + r["MX_hi"]) / k2 for r in rs]
        a.plot(x, y, color=col, ls=ls, lw=1.2, marker=mk, ms=2.6, mec="white", mew=0.3,
               label=fr"$\omega={om:.0f}$, $t_0={t0:g}$")
    a.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 2)))
    a.set_xscale("log")
    a.set_xticks([1, 1.5, 2, 3, 4, 6])
    a.set_xticks([], minor=True)
    a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    a.set_xlabel(r"partner margin / $\kappa(t_0)^2$")
    a.set_ylabel(r"critical weaker margin / $\kappa(t_0)^2$")
    a.legend(fontsize=6.3, loc="upper left", ncol=2, columnspacing=0.8, handlelength=2.2)
    a.set_ylim(0.9, 1.16)
    panel(a, "a")

    rows = gr["rows"]
    shifts = sorted({r["shift"] for r in rows})
    for rule, col, mk, lab in (("minp", BLUE, "o", "directed"), ("sym", ORANGE, "s", "symmetric")):
        for t1, ls in ((0, (0, (3, 1.5))), (20, "-")):
            y = [np.mean([r[f"{rule}_ari_t{t1}"] for r in rows if r["shift"] == sh]) for sh in shifts]
            b.plot(shifts, y, color=col, ls=ls, lw=1.2, marker=mk, ms=3, mec="white", mew=0.3, label=f"{lab} $t_1{{=}}{t1}$")
    b.set_xticks(shifts)
    k2 = gr["kappa"]["0"] ** 2
    crit = (3.0 ** 2 - k2) / (math.sqrt(2) * 3.0)
    b.axvline(crit, color=MUTED, lw=0.8, ls=(0, (4, 2)))
    b.set_xlabel("shift towards block 1")
    b.set_ylabel("ARI with final partition")
    b.set_ylim(-0.03, 1.03)
    b.set_xlim(-1.0, 3.15)  # room for the legend left of the data
    b.legend(fontsize=5.9, loc="lower left", handlelength=1.6, borderaxespad=0.25, labelspacing=0.3, handletextpad=0.4)
    panel(b, "b")
    save(fig, "fig7_learned")


if __name__ == "__main__":
    figs = [fig_phase, fig_two_block, fig_chains, fig_sbm, fig_metric, fig_depth_synthetic, fig_learned]
    for f in figs:
        if len(sys.argv) == 1 or f.__name__ in sys.argv[1:]:
            f()
    print("figures written to", OUT)
