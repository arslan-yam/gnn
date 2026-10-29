"""Heuristic 'outlier law' for E4 (conjecture in SUMMARY.md, not a theorem).
A node i at distance rho_i from its block mean, with k_i in-block neighbours, facing a collapsed block, is
treated as a two-block system with rho_A = k_i (the node vs its neighbours) and rho_B ~ 0. It is predicted to
survive as a singleton iff rho_i >= kappa(t0; k_i, ~0). The prediction is compared with the observed number
of extra clusters (n_obs - 4) in results/graphs_E4.json."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gnnlimit import full
from gnnlimit.core import kappa_estimate

K4, NB, P_DIM, P_IN, DELTA = 4, 150, 8, 0.2, 3.0
_kap = {}


def kap(k):
    if k not in _kap:
        _kap[k] = kappa_estimate(602.0, float(k), 1e-3, T=10**5)
    return _kap[k]


def main():
    old = {(r["sigma"], r["p_out"], r["seed"]): r for r in json.load(open("../results/graphs_E4.json"))["rows"]}
    out = {}
    for sigma in [0.2, 0.25, 0.3, 0.4]:
        pred, obs = [], []
        for p_out in [0.0, 0.005, 0.01, 0.02, 0.05]:
            for seed in range(5):
                rng = np.random.default_rng(1000 * seed + int(1e4 * sigma) + int(1e5 * p_out))
                indptr, indices, y = full.sbm([NB] * K4, P_IN, p_out, rng)
                n = len(y)
                mu = np.zeros((K4, P_DIM))
                mu[np.arange(K4), np.arange(K4)] = DELTA / math.sqrt(2)
                Z0 = mu[y] + sigma * rng.standard_normal((n, P_DIM))
                mus = np.stack([Z0[y == k].mean(0) for k in range(K4)])
                rho = np.linalg.norm(Z0 - mus[y], axis=1)
                src = np.repeat(np.arange(n), np.diff(indptr))
                kin = np.bincount(src[(y[src] == y[indices]) & (src != indices)], minlength=n)
                pred.append(int(sum(rho[i] >= kap(int(kin[i])) for i in range(n))))
                obs.append(old[(sigma, p_out, seed)]["n_obs"] - 4)
        out[sigma] = dict(pred_mean=float(np.mean(pred)), obs_mean=float(np.mean(obs)), pred=pred, obs=obs)
        print(f"sigma={sigma}: predicted singletons {np.mean(pred):.1f} ({min(pred)}-{max(pred)}), "
              f"observed extra clusters {np.mean(obs):.1f} ({min(obs)}-{max(obs)})")
    json.dump({str(k): v for k, v in out.items()}, open("check_outliers.json", "w"), indent=1)


if __name__ == "__main__":
    main()
