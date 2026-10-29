"""Check Theorem A (planted partition at the sqrt(c) scale) against experiment E4.
Re-generates the exact E4 initial conditions (same RNG seeds as exp_graphs.py; full.py imported read-only)
and tests the sufficient conditions, in units of sqrt(c) (c = 1 in E4):
  (M) D_k + 2 eta <= 1                       D_k = diam of block k's initial features (hull diameter)
  (S) dist(C_k, C_l) >= delta + 2 eta        for blocks joined by an edge, delta > 1
  (B) e_max * delta * (t0 - 1)^(1 - delta^2) / (delta^2 - 1) <= eta
Two choices of convex sets: C_k = conv(block k) ("hull"; dist lower-bounded by projection onto the
mean-difference direction) and C_k = B(mu_k, r_k) ("ball"). Writes check_E4.json."""
import json, math, os, sys
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import pdist
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gnnlimit import full

K4, NB, P_DIM, P_IN, DELTA = 4, 150, 8, 0.2, 3.0
SIGMAS = [0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
POUTS = [0.0, 0.005, 0.01, 0.02, 0.05]


def delta_min(eta, emax, t0):
    """Smallest delta > 1 with emax*delta*(t0-1)^(1-delta^2)/(delta^2-1) <= eta (LHS decreasing in delta)."""
    if emax == 0:
        return 1.0
    f = lambda d: emax * d * (t0 - 1) ** (1 - d * d) / (d * d - 1) - eta
    lo, hi = 1.0 + 1e-12, 10.0
    if f(hi) > 0:
        return math.inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if f(mid) <= 0 else (mid, hi)
    return hi


def certify(D, sep, emax, t0):
    """Return (holds, eta, delta). sep = min distance between C_k, C_l over adjacent pairs (inf if none)."""
    if D > 1:
        return False, None, None
    etas = np.linspace(0, (1 - D) / 2, 401)[1:] if emax > 0 else [0.0]
    for eta in etas:
        d = delta_min(eta, emax, t0)
        if sep >= d + 2 * eta:
            return True, float(eta), float(d)
    return False, None, None


def run():
    old = {(r["sigma"], r["p_out"], r["seed"]): r for r in json.load(open("../results/graphs_E4.json"))["rows"]}
    rows = []
    for sigma in SIGMAS:
        for p_out in POUTS:
            for seed in range(5):
                rng = np.random.default_rng(1000 * seed + int(1e4 * sigma) + int(1e5 * p_out))
                indptr, indices, y = full.sbm([NB] * K4, P_IN, p_out, rng)
                n = len(y)
                t0 = n + 2.0
                mu = np.zeros((K4, P_DIM))
                mu[np.arange(K4), np.arange(K4)] = DELTA / math.sqrt(2)
                Z0 = mu[y] + sigma * rng.standard_normal((n, P_DIM))
                src = np.repeat(np.arange(n), np.diff(indptr))
                cross = y[src] != y[indices]
                emax = int(np.bincount(src[cross], minlength=n).max()) if cross.any() else 0
                adj = set((min(a, b), max(a, b)) for a, b in zip(y[src][cross], y[indices][cross]))
                mus = np.stack([Z0[y == k].mean(0) for k in range(K4)])
                rk = [float(np.linalg.norm(Z0[y == k] - mus[k], axis=1).max()) for k in range(K4)]
                Dk = [float(pdist(Z0[y == k]).max()) if sigma > 0 else 0.0 for k in range(K4)]
                sep_hull, sep_ball = math.inf, math.inf
                for a, b in adj:
                    e = mus[b] - mus[a]
                    e /= np.linalg.norm(e)
                    sep_hull = min(sep_hull, float((Z0[y == b] @ e).min() - (Z0[y == a] @ e).max()))
                    sep_ball = min(sep_ball, float(np.linalg.norm(mus[a] - mus[b]) - rk[a] - rk[b]))
                h_ok, h_eta, h_del = certify(max(Dk), sep_hull, emax, t0)
                b_ok, b_eta, b_del = certify(2 * max(rk), sep_ball, emax, t0)
                o = old[(sigma, p_out, seed)]
                rows.append(dict(sigma=sigma, p_out=p_out, seed=seed, t0=t0, emax=emax, D=max(Dk), r=max(rk),
                                 sep_hull=sep_hull, sep_ball=sep_ball, hull_holds=h_ok, hull_eta=h_eta,
                                 hull_delta=h_del, ball_holds=b_ok, ball_eta=b_eta, ball_delta=b_del,
                                 old_holds=o["cert_holds"], ari_gt=o["ari_gt"], n_obs=o["n_obs"]))
    by = defaultdict(list)
    for r in rows:
        by[r["sigma"]].append(r)
    summary = []
    print("sigma  D range          r range        old  ball  hull  planted survives (ARI=1)")
    for s in SIGMAS:
        rs = by[s]
        D = [r["D"] for r in rs]; rr = [r["r"] for r in rs]
        line = dict(sigma=s, n=len(rs), D_min=min(D), D_max=max(D), r_min=min(rr), r_max=max(rr),
                    old=sum(r["old_holds"] for r in rs), ball=sum(r["ball_holds"] for r in rs),
                    hull=sum(r["hull_holds"] for r in rs), survives=sum(r["ari_gt"] == 1.0 for r in rs),
                    violations=sum(r["hull_holds"] and r["ari_gt"] != 1.0 for r in rs))
        summary.append(line)
        print(f"{s:5.2f}  [{min(D):.3f},{max(D):.3f}]  [{min(rr):.3f},{max(rr):.3f}]  {line['old']:3d}  {line['ball']:4d}  {line['hull']:4d}  {line['survives']:3d}/25")
    json.dump(dict(summary=summary, rows=rows), open("check_E4.json", "w"), indent=1)


if __name__ == "__main__":
    run()
