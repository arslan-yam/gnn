# theory/: what was proved, what is conjectural, and what a referee would flag

Full statements and proofs: `proofs.tex` / `proofs.pdf` (11 pages; `latexmk -pdf proofs.tex`).
Units of √c throughout; t0 is the schedule offset, T_t = c/log(t + t0).
Nothing outside `theory/` was modified. `core.py` and `full.py` are imported read-only.

## Proved

### 0. Universal dichotomy (Prop. 1.5), used by both A and B
On any graph with self-loops and any vector features, every z_i(t) converges, and every edge's
limiting length is either 0 or ≥ √c. This extends Theorem 1(i),(iii) (the jump) from two blocks to every edge of every graph.
- Engine: Touri–Nedić, *Product of random stochastic matrices*, IEEE TAC 59(2) 2014 (arXiv:1110.1751),
  Thm 1, deterministic case. It covers balanced, strongly aperiodic chains and allows arbitrarily small entries. Attention
  matrices qualify with α = γ = 1/N_max because the unnormalised weights are symmetric.
- Hendrickx–Tsitsiklis 2013, Thm 2 (discrete time), does **not** apply: it needs positive entries ≥ α.
  Do not cite it for this.

### A. Planted partition at the √c scale (Thm 2.1, Cors 2.2–2.3, Prop 2.5)
Assume each G[B_k] is connected and each block's initial features lie in a convex set C_k of diameter D_k. If, for some
η ≥ 0 and δ > 1,
- (A1) D_k + 2η ≤ 1,
- (A2) dist(C_k, C_l) ≥ δ + 2η for blocks joined by an edge,
- (A3) e_max · δ · (t0−1)^{1−δ²}/(δ²−1) ≤ η,

then every block collapses to a point, adjacent limits are ≥ δ apart, and the limiting partition is exactly the planted one.
- **Ball form:** r ≤ √c(1/2 − η) and Δ ≥ 2r + (δ+2η)√c. The old condition was r ≤ √c/(4√ρ) and Δ ≥ 4r + …
- **Separation threshold = κ(t0) to leading order:** optimising η gives Γ(t0) = 1 + (1+o(1)) log log t0 / log t0.
  The leading constant is the same as κ(t0; ρ) for every topology, and κ(t0; ρ_A, ρ_B) is necessary for block-regular
  two-block graphs.
- **Mixing constant is sharp (Prop 2.5):** for every D > 1 and t0 there is a connected block of diameter D that splits:
  two k-cliques joined by a perfect matching, with k large. At t0 = 602, D = 1.1 and k = 100 give a limit gap of 1.071.
  So no condition that sees only block diameters (plus connectivity, degrees and separations) can go beyond D ≤ 1.
- **Proof idea (two time scales):** invariant convex sets C_k + B(η) control the total cross-block drift, where weights are
  summable. Within blocks the weights stay ≥ s^{-1}, which is non-summable, so the infinite-flow theorem collapses each block.
  This replaces the Dobrushin path products that produced the factor 1/√ρ.

**E4 check** (`check_E4.py`, regenerates the exact initial conditions; replication verified, max |r − cert_r| = 0):

| σ | max D_k (range) | old cert | ball form | hull form | survives |
|---|---|---|---|---|---|
| 0.00 | 0 | 5 | 25 | 25 | 25 |
| 0.03 | 0.23–0.29 | 23 | 25 | 25 | 25 |
| 0.06 | 0.46–0.55 | 0 | 25 | 25 | 25 |
| 0.10 | 0.79–0.94 | 0 | 11 | **25** | 25 |
| ≥0.15 | ≥1.155 | 0 | 0 | 0 | 25 (σ≤0.25) |

So **σ ≤ 0.1 is now certified (100/100 runs)**, against σ ≤ 0.03 (28 runs) before. There are no violations.
For σ ≥ 0.15, every run has a within-block diameter > √c, so by Prop 2.5 no diameter-based theorem can certify it.

### B. Three-layer chain (Lemma 3.1, Prop 3.2, Thms 3.3 and 3.8, Cor 3.9)
- **Exact criterion (Prop 3.2):** h := u + v/2 is nonincreasing. If u0 + v0 < 2, gap 1 persists ⇔ h(t) ≥ 1 for all t;
  then u∞ = h∞ and v∞ = 0. For **every** t0, gap 1 merges whenever u0 + v0/2 < 1, so u*(v0) ≥ √c − v0/2.
  On the symmetric line u0 = v0 < 1, all three layers merge.
- **One-step rescue (Thm 3.3):** for **every t0 ≥ 2 and every v0 > 0**, gap 1 persists for all
  u0 ≥ κ(t0) − R(t0, v0), with an explicit R > 0. The neighbour pushes u across the critical curve κ(t0+1) in one step,
  and the comparison lemma finishes. Max depth ≈ 0.278/√log t0. Relative to κ that is 5.5% (t0 = 11), 7.5% (t0 = 10³) and 5% (t0 = 10¹⁰).
- **Scaling limit (Thm 3.8, Cor 3.9):** with x = v0/κ(t0) and y = u0/κ(t0), as t0 → ∞ gap 1 persists iff y > φ(x), where
  **φ(x) = max{x, 1 − x/2} for x < 1 and φ(x) = 1 for x > 1**. The normalised boundary converges to φ pointwise, and
  inf over v0 of u*/κ → **2/3**, attained at x = 2/3. This explains E2b's drift 0.770, 0.751, 0.722, 0.695, 0.682 → 0.667.
  - The branch 1 − x/2 is "layers 2, 3 merge to their midpoint, so u gains v/2".
  - The branch x is the race: the smaller gap merges first, at time scale t0^{g²}.
  - Explicit sufficient conditions are given at a finite T*.
  - Check at t0 = 10⁵ (`check_scaling.py`): u*/κ = 0.937, 0.837, 0.737, 0.692, 0.684, 0.703, 0.800, 0.900 against
    φ = 0.950, 0.850, 0.750, 0.700, 0.667, 0.700, 0.800, 0.900, for x = 0.1 … 0.9.

All chain identities were checked against `core.chain_step` to 3e-15 (`check_identities.py`).

## Not proved / conjectural
1. **σ ∈ (0.1, 0.25] in E4.** This is impossible for worst-case (diameter-only) theorems, by Prop 2.5. It needs a
   random-feature theorem using dense blocks and concentration. **Conjecture (outlier law):** the bulk of each block
   collapses when the typical within-block distance σ√(2p) ≲ √c (note σ√(2p) = 1.0 exactly at σ = 0.25). A node at
   distance ρ_i from its block mean, with k_i in-block neighbours, then survives as a singleton iff ρ_i ≥ κ(t0; k_i, ≈0).
   `check_outliers.py` compares the predicted number of singletons with the observed extra clusters: 0 vs 0 (σ ≤ 0.25),
   2.2 vs 2.8 (σ = 0.3), 68.6 vs 63.2 (σ = 0.4). This is heuristic only.
2. **Cut ratio at second order in Thm A.** (A3) uses e_max. Replacing it by ρ_i = e_i/k_i (the post-collapse rate) needs an
   explicit within-block collapse time, which we do not have for general graphs (Remark 2.7).
3. **Explicit T\* in Thm 3.8 is huge near the minimum.** Uniform persistence holds for x ∈ [0.3, 0.5], y ≥ 0.9 from
   t0 ≥ 2·10⁸; for x ∈ [0.4, 0.6], y ≥ 0.85 from 8·10¹⁵; and near the corner x = y = 2/3 the bound is about 10¹¹⁰
   (`constants.py`). At moderate t0, the macroscopic (25–33%) rescue rests only on the validated computations
   (Theorem 3, E2). The analytic one-step rescue is only 5–7.5%.
4. **No rate of convergence to φ.** It is observed to be slow, and on the branch 1 − x/2 the finite-t0 boundary lies slightly
   below φ. The explanation via an effective coupling 5/4 < 2 in the h-identity is plausible but unproved.
5. **Monotonicity in u0** of the persistence set (so that u* is a single threshold) is not proved. Cor 3.9 is stated with the
   two envelopes U₋ ≤ U₊; E2 found 0 non-monotone columns numerically.

## What a referee would flag
- **The dependence on Touri–Nedić.** Their theorem is stated for independent random chains; we use the deterministic
  special case. Their proof goes through Thm 4, which needs "weak aperiodicity"; we note that it follows from strong
  aperiodicity, a step their paper leaves implicit. Their Thm 6 is stated for start time 0 but used for general start
  times. Our reading of their paper is via the arXiv HTML.
- **Theorem 1(v) of the main text** (the sharp asymptotics of κ) is still not written up according to RESULTS_NOTES.
  These notes use it only in sharpness remarks. All proofs use the self-contained Lemma 1.1, whose κ₊ → 1 comes from C1.
- **E4 hull distances** are projection lower bounds, so they are conservative. Separation is not binding at σ ≤ 0.1 anyway.
- **Numerics.** The E2b, scaling and outlier checks use float (non-interval) arithmetic. Only the κ enclosures from E1
  and `rigorous_mp.py` are validated.
- Cor 2.3 needs log(t0 − 1) > e, i.e. t0 > 16.2. Thm 3.3 needs t0 ≥ 2.

## Files
- `proofs.tex`, `proofs.pdf`: statements and proofs.
- `check_identities.py`: chain identities vs `core.chain_step`.
- `check_E4.py` → `check_E4.json`: Thm A on E4.
- `check_scaling.py` → `check_scaling.json`: rescue boundary vs φ.
- `constants.py` → `constants.json`: Thm 3.3 depths and Thm 3.8 values of T\*.
- `check_outliers.py` → `check_outliers.json`: outlier-law heuristic.
