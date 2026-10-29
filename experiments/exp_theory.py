"""Experiments E1-E3 (theory validation on the reduced models). Writes results/theory.npz + theory.json.

E1  two-block law: validated kappa(t0), non-asymptotic bounds, topology (cut ratio rho),
    discontinuous jump of d_inf, critical slowing down.
E2  three layers: certified phase diagram, rescue boundary, boundary curves for several t0.
E3  five layers: certified partitions for random gaps, isolated rule vs early-depth rule.
"""
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import brentq

from gnnlimit.core import (certify_grid, chain_final, kappa_enclosure, kappa_estimate, log_checkpoints,
                  two_block_certify, two_block_final, two_layer_traj)

T_START = time.monotonic()
OUT = {}
ARR = {}


def stamp(msg):
    print(f"[{time.monotonic() - T_START:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# non-asymptotic bounds for kappa (Theorem 1(v))
# ---------------------------------------------------------------------------
def kappa_lower(t0, rA=1.0, rB=1.0):
    lam_m = (rA + rB) / ((1 + rA / t0) * (1 + rB / t0))
    f = lambda e: e * (2 * e + e * e) - lam_m * t0 ** (-(2 * e + e * e))
    return 1 + brentq(f, 1e-12, 5.0)


def c1_holds(d0, t0, rA, rB):
    lam, P = rA + rB, rA * rB
    for k in range(1, 400):
        delta = 1 + (d0 - 1) * k / 400
        q = t0 ** (-delta * delta)
        tail = (t0 - 1) ** (1 - delta * delta) / (delta * delta - 1)
        loss = (lam + P * q / (1 - P * q * q)) * tail
        if math.log(d0) - loss >= math.log(delta):
            return True
    return False


def kappa_upper(t0, rA=1.0, rB=1.0):
    a, b = 1.0, 10.0
    for _ in range(60):
        m = 0.5 * (a + b)
        if c1_holds(m, t0, rA, rB):
            b = m
        else:
            a = m
    return b


def kappa_asym(t0):
    L1 = math.log(t0)
    return 1 + math.log(L1) / L1


def run_E1():
    stamp("E1a kappa(t0)")
    t0s = [3, 5, 8, 11, 17, 30, 100, 300, 1e3, 1e4, 1e5, 1e6]
    rows = []
    for t0 in t0s:
        Tm = 10**6 if t0 <= 1e3 else 10**7
        klo, khi, est = kappa_enclosure(float(t0), Tmax=Tm, T_float=Tm)
        rows.append(dict(t0=t0, klo=klo, khi=khi, est=est, lower=kappa_lower(t0),
                         upper=kappa_upper(t0), asym=kappa_asym(t0)))
        stamp(f"  t0={t0:g}: [{klo:.6f}, {khi:.6f}] bounds [{rows[-1]['lower']:.4f}, {rows[-1]['upper']:.4f}]")
    OUT["E1_kappa"] = rows
    # dense finite-depth curve + bounds for the figure
    t0d = np.unique(np.round(np.logspace(np.log10(3), 8, 60)))
    ARR["E1_t0_dense"] = t0d
    ARR["E1_lower_dense"] = np.array([kappa_lower(t) for t in t0d])
    ARR["E1_upper_dense"] = np.array([kappa_upper(t) for t in t0d])
    ARR["E1_asym_dense"] = np.array([kappa_asym(t) for t in t0d])

    stamp("E1b topology (cut ratio rho)")
    rhos = [0.1, 0.25, 0.5, 1.0]
    t0b = [11.0, 100.0, 1e3, 1e4, 1e5]
    K = np.zeros((len(rhos), len(t0b)))
    for i, rho in enumerate(rhos):
        for j, t0 in enumerate(t0b):
            K[i, j] = kappa_estimate(t0, rho, rho, T=10**6 if t0 <= 1e3 else 10**7)
    ARR["E1_rhos"], ARR["E1_t0b"], ARR["E1_K_rho"] = np.array(rhos), np.array(t0b), K
    enc = []
    for rho in rhos:
        klo, khi, est = kappa_enclosure(11.0, rho, rho)
        enc.append(dict(rho=rho, klo=klo, khi=khi, est=est))
    asym_rows = []
    for (rA, rB) in [(1.0, 1.0), (0.25, 1.0), (0.1, 0.1)]:
        klo, khi, est = kappa_enclosure(11.0, rA, rB)
        asym_rows.append(dict(rA=rA, rB=rB, klo=klo, khi=khi))
    OUT["E1_rho_enclosures_t0_11"] = enc
    OUT["E1_rho_asym_t0_11"] = asym_rows
    stamp(f"  rho enclosures at t0=11: {[(e['rho'], round(e['klo'], 5), round(e['khi'], 5)) for e in enc]}")

    stamp("E1c jump of d_inf")
    t0 = 11.0
    d0 = np.linspace(0.9, 2.2, 521)
    dT = np.array([two_block_final(x, t0, 1.0, 1.0, 10**6) for x in d0])
    status = np.array([two_block_certify(x, t0, 1.0, 1.0, 10**6)[0] for x in d0])
    ARR["E1_jump_d0"], ARR["E1_jump_dT"], ARR["E1_jump_status"] = d0, dT, status
    above = dT[d0 > OUT["E1_kappa"][3]["khi"]]
    OUT["E1_jump"] = dict(t0=t0, n=len(d0), n_cert_merge=int((status == 0).sum()),
                          n_cert_persist=int((status == 1).sum()), n_unknown=int((status == -1).sum()),
                          min_dT_above_khi=float(above.min()) if above.size else None)

    stamp("E1d critical slowing")
    est = OUT["E1_kappa"][3]["est"]
    cps = log_checkpoints(10**7, 300)
    trajs = {}
    for h in [-1e-9, 1e-9, 1e-6, 1e-4, 1e-2]:
        trajs[h] = two_layer_traj(est + h, t0, cps)
    ARR["E1_crit_cps"] = cps
    ARR["E1_crit_traj"] = np.array([trajs[h] for h in sorted(trajs)])
    ARR["E1_crit_h"] = np.array(sorted(trajs))
    # critical curve: kappa at shifted offsets t0 + t equals the critical trajectory
    shifts = [0, 10, 100, 1000, 10000, 100000]
    cc = []
    for sh in shifts:
        klo, khi, e2 = kappa_enclosure(t0 + sh, Tmax=10**7, T_float=10**7)
        cc.append(dict(t=sh, klo=klo, khi=khi))
    OUT["E1_critical_curve"] = cc
    stamp(f"  critical curve {[(c['t'], round(c['klo'], 5), round(c['khi'], 5)) for c in cc]}")


def run_E2():
    stamp("E2a certified three-layer phase diagram (t0 = 11)")
    t0 = 11.0
    grid = np.round(np.arange(1, 241) * 0.0125, 6)
    U, V = np.meshgrid(grid, grid, indexing="ij")
    G0 = np.stack([U.ravel(), V.ravel()], axis=1)
    t = time.monotonic()
    L, Tc = certify_grid(G0, t0, 200000, 2000)
    stamp(f"  grid {len(grid)}^2 certified in {time.monotonic() - t:.1f}s")
    L = L.reshape(len(grid), len(grid), 2)
    ARR["E2_grid"], ARR["E2_labels"], ARR["E2_Tc"] = grid, L, Tc.reshape(len(grid), len(grid))
    cert = (L[..., 0] >= 0)
    phase = np.full(U.shape, -1)
    phase[cert & (L[..., 0] == 0) & (L[..., 1] == 0)] = 0  # one cluster
    phase[cert & (L[..., 0] == 0) & (L[..., 1] == 1)] = 1  # {1,2},{3}
    phase[cert & (L[..., 0] == 1) & (L[..., 1] == 0)] = 2  # {1},{2,3}
    phase[cert & (L[..., 0] == 1) & (L[..., 1] == 1)] = 3  # all distinct
    ARR["E2_phase"] = phase
    counts = {name: int((phase == k).sum()) for k, name in
              [(-1, "uncertified"), (0, "one cluster"), (1, "{1,2},{3}"), (2, "{1},{2,3}"), (3, "all distinct")]}
    kap = kappa_enclosure(t0)
    # rescue boundary of gap 1 per column v0: largest certified-merge u0 and smallest certified-persist u0
    lo_b = np.full(len(grid), np.nan)
    hi_b = np.full(len(grid), np.nan)
    nonmono = 0
    for j in range(len(grid)):
        lab = L[:, j, 0]
        m = np.where(lab == 0)[0]
        p = np.where(lab == 1)[0]
        if m.size:
            lo_b[j] = grid[m.max()]
        if p.size:
            hi_b[j] = grid[p.min()]
        if m.size and p.size and m.max() > p.min():
            nonmono += 1
    ARR["E2_merge_max"], ARR["E2_persist_min"] = lo_b, hi_b
    jmin = int(np.nanargmin(hi_b))
    OUT["E2"] = dict(t0=t0, grid_step=0.0125, n=int(phase.size), counts=counts,
                     kappa=list(kap[:2]), nonmonotone_columns=nonmono,
                     min_persist_u0=float(hi_b[jmin]), at_v0=float(grid[jmin]),
                     relative_drop=float(1 - hi_b[jmin] / kap[1]),
                     max_Tcert=int(Tc[cert.ravel()].max()))
    stamp(f"  counts {counts}; min certified-persist u0 = {hi_b[jmin]:.4f} at v0 = {grid[jmin]:.4f}")

    stamp("E2b finite-depth rescue boundaries for several t0")
    curves = {}
    for t0b in [5.0, 11.0, 50.0, 1e3, 1e5]:
        T = 10**5 if t0b <= 1e3 else 10**6
        kap_b = kappa_estimate(t0b, T=T)
        v0s = kap_b * np.linspace(0.02, 2.0, 100)
        u_star = np.empty_like(v0s)
        for i, v0 in enumerate(v0s):
            a, b = 0.0, kap_b * 1.2
            for _ in range(30):
                mid = 0.5 * (a + b)
                g = chain_final(np.array([[mid, v0]]), t0b, T)[0]
                if g[0] > 0.5 * min(1.0, mid):
                    b = mid
                else:
                    a = mid
            u_star[i] = 0.5 * (a + b)
        curves[t0b] = (kap_b, v0s, u_star)
        stamp(f"  t0={t0b:g}: kappa={kap_b:.4f}, min u*/kappa = {u_star.min() / kap_b:.3f} at v0/kappa = {v0s[np.argmin(u_star)] / kap_b:.3f}")
    ARR["E2b_t0"] = np.array(sorted(curves))
    ARR["E2b_kappa"] = np.array([curves[k][0] for k in sorted(curves)])
    ARR["E2b_v0"] = np.array([curves[k][1] for k in sorted(curves)])
    ARR["E2b_ustar"] = np.array([curves[k][2] for k in sorted(curves)])
    OUT["E2b"] = [dict(t0=k, kappa=curves[k][0], min_ratio=float(curves[k][2].min() / curves[k][0]),
                       argmin_v0_ratio=float(curves[k][1][np.argmin(curves[k][2])] / curves[k][0]))
                  for k in sorted(curves)]


def run_E3():
    stamp("E3 five layers, random gaps")
    rng = np.random.default_rng(0)
    t0 = 17.0  # r = 5, m = 3
    n = 4000
    G0 = rng.uniform(0.2, 2.6, size=(n, 4))
    L, Tc = certify_grid(G0, t0, 300000, 3000)
    cert = (L >= 0).all(axis=1)
    kap = kappa_enclosure(t0)
    stamp(f"  certified {cert.sum()}/{n}; kappa(17) in [{kap[0]:.5f}, {kap[1]:.5f}]")
    Lc, Gc = L[cert], G0[cert]
    iso_pred = (Gc >= kap[1]).astype(int)
    # gaps in the uncertain window [klo, khi] are excluded from the error count
    amb = (Gc > kap[0]) & (Gc < kap[1])
    err = (iso_pred != Lc) & ~amb
    false_merge = ((iso_pred == 1) & (Lc == 0) & ~amb).sum()
    rescues = ((iso_pred == 0) & (Lc == 1) & ~amb).sum()
    # early-depth rule: after t1 steps compare with kappa(t0 + t1)
    t1s = [0, 1, 2, 5, 10, 20, 50, 100, 1000]
    acc = []
    for t1 in t1s:
        if t1 == 0:
            Gt = Gc
        else:
            Gt = chain_final(Gc, t0, t1)
        # gaps evaluated from time t1 onward = same recursion with offset t0 + t1
        k1 = kappa_estimate(t0 + t1, T=10**6)
        pred = (Gt >= k1).astype(int)
        acc.append(float((pred == Lc).mean()))
        stamp(f"  t1={t1}: kappa(t0+t1)={k1:.4f}, gap-level accuracy {acc[-1]:.5f}, partition-level {(pred == Lc).all(axis=1).mean():.4f}")
    nclust = 1 + Lc.sum(axis=1)
    small_nb = []
    for i, l in zip(*np.where((iso_pred == 0) & (Lc == 1) & ~amb)):
        nb = [Gc[i, k] for k in (l - 1, l + 1) if 0 <= k < 4]
        small_nb.append(min(nb))
    OUT["E3"] = dict(t0=t0, n=n, n_certified=int(cert.sum()), kappa=list(kap[:2]),
                     gap_accuracy_isolated=float(1 - err.mean()), n_gaps=int(Lc.size),
                     n_errors=int(err.sum()), false_merges=int(false_merge), rescues=int(rescues),
                     n_ambiguous=int(amb.sum()),
                     rescued_min_gap=float(Gc[(iso_pred == 0) & (Lc == 1) & ~amb].min()) if rescues else None,
                     rescued_neighbor_gap_range=[float(min(small_nb)), float(max(small_nb))] if small_nb else None,
                     partition_accuracy_isolated=float(((iso_pred == Lc) | amb).all(axis=1).mean()),
                     early_t1=t1s, early_acc=acc,
                     cluster_hist={int(k): int((nclust == k).sum()) for k in range(1, 6)},
                     max_Tcert=int(Tc[cert].max()))
    ARR["E3_G0"], ARR["E3_L"], ARR["E3_cert"] = G0, L, cert


if __name__ == "__main__":
    which = sys.argv[1:] or ["E1", "E2", "E3"]
    for w in which:
        {"E1": run_E1, "E2": run_E2, "E3": run_E3}[w]()
        np.savez_compressed(f"results/theory_{w}.npz", **{k: v for k, v in ARR.items() if k.startswith(w)})
        with open(f"results/theory_{w}.json", "w") as f:
            json.dump({k: v for k, v in OUT.items() if k.startswith(w)}, f, indent=1, default=float)
    stamp("done")
