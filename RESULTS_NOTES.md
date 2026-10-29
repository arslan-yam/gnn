# v2 working notes: new theorems and results

Historical log, kept as written. Since 28 Oct 2026 the library modules (`core.py`, `full.py`, `models.py`, ...) live in
`gnnlimit/` and the `exp_*.py` drivers in `experiments/`; see README.md.

Units of sqrt(c). Schedule T_t = c/log(t + t0). Two-block model: blocks A, B, each node of A has
k_A closed in-block neighbours and e_A cross neighbours; rho_A = e_A/k_A, lambda = rho_A + rho_B,
P = rho_A rho_B <= 1. With block-constant features the gap obeys exactly
  d' = d (1 - P a^2)/((1 + rho_A a)(1 + rho_B a)),   a = (t + t0)^(-d^2)
(rho = 1: d' = d (1-a)/(1+a), which is the old two-layer recursion).

## Theorem 1 (sharp two-block law). Proofs are worked out; they still need writing up.
(i) d_inf exists and lies in {0} ∪ [1, ∞). (If 0 < d_inf < 1 then sum a_t = ∞ and the product goes to 0.)
(ii) Merge criterion: d_T <= 1 at any T implies merging. The merge set is open and, by monotonicity of the map, equals (0, kappa).
    kappa > 1.
(iii) d_inf(kappa) = 1 exactly, and d_inf is nondecreasing and upper semicontinuous, hence right-continuous.
    So the limit gap jumps from 0 to sqrt(c): a first-order transition. (If d_inf(kappa) = 1 + 2eta, persistence would be robust under perturbation,
    which contradicts kappa = inf.)
(iv) Critical-curve identity: kappa(t0 + t) = F_{t-1}∘...∘F_0(kappa(t0)), because each F_t is an increasing bijection of (0, ∞).
    So the critical trajectory traces the threshold for later offsets.
(v) Sharp asymptotics (the leading constant is 1 and does not depend on the topology lambda):
    kappa(t0) = 1 + (1 + o(1)) log log t0 / log t0;  equivalently d_crit^2 = c + (2 + o(1)) T0 log(c/T0).
    Lower bound: if 1 + eps persists then eps >= lambda_- t0^{-(2eps+eps^2)}/(2eps+eps^2), with lambda_- = lambda/((1+rho_A/t0)(1+rho_B/t0)).
    Upper bound: bootstrap d >= 1 + theta*eps with theta*K > 1, using the loss bound (lambda + Pq/(1-Pq^2)) (t0-1)^{1-delta^2}/(delta^2-1).
    By (iv), the critical slowing is d_t^crit - 1 ~ log log t / log t.
(vi) Explicit finite-t0 sandwich bounds (kappa_lower / kappa_upper in exp_theory.py).

## Chains: comparison and nonlocality
Lemma (comparison, any r, ordered layer-constant features): g_l' = g_l (AB - w^2)/(D_l D_{l+1})
+ w_{l+1} g_{l+1}/D_{l+1} + w_{l-1} g_{l-1}/D_l >= F_t(g_l). So every gap dominates the isolated two-layer gap.
Theorem 2: g_l(0) >= kappa(t0) implies the gap persists, whatever the other gaps are. More generally, g_l(t) >= kappa(t0 + t) at any t
implies persistence. Neighbours can only rescue a gap; they never destroy one. Upper comparison: u' <= F_t(u) + b(u + v).
This gives Prop.: d_1*(v0) -> kappa as v0 -> ∞.
Certificates: C1 (persistence) and C2 (block merge: D_B + 2(delta+1)tail <= 1/sqrt(q_B)) are in core.py.
Theorem 3 (computer-assisted, fully rigorous mpmath, rigorous_mp.py, all OK, 7 s): r=3, m=3, t0=11, kappa(11) in [1.538, 1.540].
  Fixing u0 = 1.4: v0 = 0.1 gives one cluster; v0 = 0.5 and 0.95 give {1},{2,3}; v0 = 1.5 and 3.0 give {1,2},{3}. So the fate of edge 1-2
  goes merge, persist, merge as v0 grows. u0 = 1.2 (22% below kappa) persists at v0 = 0.95. The point v0 = 1.5 < kappa also persists (rescue in the other direction).
  Mechanism: the neighbour pulls layer 2 across the critical curve within 1–3 steps.

## Numbers so far
- E1 (results/theory_E1.json, log_E1.txt): validated kappa: t0=5 [1.66857,1.66879], 8 [1.58233,1.58266], 11 [1.53859,1.53909],
  17 [1.49084,1.49159], 30 [1.44247,1.44359], 100 [1.37028,1.37280], 300 [1.32458,1.32836], 1e3 [1.28665,1.29517],
  1e6 [1.16258,1.20573]. Sandwich bounds hold (e.g. t0=17: [1.314,1.575]). Topology at t0=11: rho=0.5 gives kappa≈1.4644, (0.25,1) gives ≈1.4881.
  E1b–d (rho sweep, jump, critical curve) were still running when this note was written.
- E2: 240^2 grid at t0=11. 99.64% certified (208 uncertified). Counts: one cluster 10147, {1,2}{3} 16739, {1}{2,3} 16739, all distinct 13767.
  The smallest certified-persist u0 is 1.1625 at v0 = 0.8875 (24% below kappa). Finite-depth min u*/kappa: t0=5 gives 0.770, 11 gives 0.751, 50 gives 0.722, 1e3 gives 0.695.
  The rescue grows relatively stronger as t0 increases.
- E3 (r=5, t0=17, 4000 random gaps in [0.2, 2.6], 3972 certified): the isolated rule has 91.3% gap-level and 68% partition-level accuracy.
  There are 0 false merges, as Theorem 2 guarantees, and all 1387 errors are rescues. The smallest rescued gap is 0.793 < sqrt(c).
  The early-depth rule (threshold g(t1) at kappa(t0+t1)) reaches 98.3% at t1=5, 99.1% at t1=10, and 99.96% at t1=1000.
- E4 (full model, SBM with 4×150 nodes, p=8, Delta=3, t0=n+2, kappa=1.301): the planted partition survives up to sigma=0.25 for every p_out.
  It fragments at 0.3 (≈7 clusters), 0.4 (≈66) and 0.5 (≈213). The init threshold-graph predictor has ARI 1.0 up to 0.25, 0.99 at 0.3, 0.78–0.89 at 0.4 and 0.55–0.68 at 0.5.
  The early rule at t1=20 does a bit better. The Theorem-5 sufficient condition holds only for sigma <= 0.03 (28 runs) and is never violated, so it is very conservative.
- E5 (learned metric; class signal below threshold, nuisance above): with W=I accuracy is 1.0 at depth 64 but 0.315 (chance) at 2048.
  The nuisance projection reaches chance by depth 64. The LDA metric calibrated to 2·kappa keeps 1.0 at depth 2048; the LDA metric scaled too small reaches chance.
- E6 (torch, exp_train.py) was RUNNING in the background and writes results/train_E6.json. Early check: in the noisy-feature
  homophilous regime, Gaussian attention is capped at ≈0.70 while GCN-2 and SGC-8 reach ≈0.97–0.99. This is a failure regime of feature-similarity attention,
  consistent with Ma et al. ICML 2025. GAT-128 collapses to 0.33.

## Paper draft (28 Oct 2026)
v2/paper/main.tex + appendix.tex + references.bib, built with `cd v2/paper && latexmk -pdf main.tex` (21 pages: ~10 main, refs, appendix).
Figures: `python3 paper/make_figures.py`; tables: `python3 paper/make_tables.py` (both from v2/, read results/ and theory/ JSON).
Novelty audit: paper/NOVELTY_AUDIT.md. Numbering in the draft: Thm 1 two-block law (full proof of (v) now in App. A), Lemma 2 + Thm 2
one-sided errors + early-depth monotonicity, Thm 3 rescue law (theory/proofs.tex Task B), Thm 4 computer-assisted, Prop 2 dichotomy,
Thm 5 + Cor 1 + Prop 3 planted partitions (Task A).

## Still to do
- Section 8 (sec:downstream) is drafted for the synthetic runs (Fig. 6 = paper/figures/fig6_depth_synthetic, App. tables
  tab_train_e6b/e7 from make_tables.py, App. E6b/E7 details written). The real-graph depth sweeps are still a TODO paragraph.
  E6b/E7 reading (checked against the per-seed rows): the Gaussian networks end in one of two phases, set by the initial temperature.
  Cold start m = 1/30 gives the frozen phase, which equals the MLP seed by seed (|diff| <= 0.015). Validation picks it in all 30
  heterophilous runs of each Gaussian model, and at K >= 64 in the noisy regime. The default start m = 1 merges: the class partition
  on clean CSBMs (0.999 at all K >= 8), and collapse to chance on noisy ones at K >= 64 (test 0.33, erank 1.5-3); at K = 32, 2 of 5
  seeds still reach the class partition. Mean accuracy is never more than 0.004 below the MLP. Rings: well below PairNorm/GCNII;
  constant T beats cooling by up to 0.10 at K >= 32. Constant vs cooled on the CSBMs: within 0.007.
- Bib entries marked TODO (Gao et al. 2026, Ma et al. 2025 title, ODNet, BIMP, learnable-temperature GAT, Hegarty-Wedin arXiv no.) need verifying.
- Second-order term: PROVED (Thm 1(v), App. A.1, barrier argument): kappa = 1 + [log L - 1/2 loglog L + 1/2 log(rA+rB) + o(1)]/L,
  L = log t0; Cor. "topology enters at second order". Third term (loglog L - log lam + 2)/(4 log L) also PROVED (Prop. "third term",
  same barriers with c = 1/2 log lam + (loglog tau - log lam + 2 +- eta)/(4 log tau)); topology shift = log(lam/lam')/(2L) (1 - 1/(2 log L)).
  Numerics: exp_second_order.py -> results/theory_E1e.json (validated enclosures, continuum residuals, barrier checks; run with
  --barriers to redo only the barrier checks, whose precision now grows with tau -- the first version used 60 digits and was vacuous
  beyond tau ~ 135). Result (tau <= 8100): order-2 barriers hold for tau >= 92 except (0.1,0.1), eta=0.2 (tau >= 1729); order-3
  barriers hold beyond tau* <= 4385 except (0.1,0.1), eta=0.2, whose upper barrier still fails at tau = 8100. That matches the exact
  margin (1-r) - e^{-2 delta}: times log tau it is -0.11 at log tau = 9 and tends to +eta/2 only for log tau >~ 100.
- Real benchmarks: Roman-empire, Amazon-ratings and Peptides-func DOWNLOADED 28 Oct 2026 with the user's OK (data/raw/, sources,
  sizes and SHA-256 in data/README.md; load only with np.load(allow_pickle=False) / torch.load(weights_only=True)).
- E8 real-graph benchmarks: exp_real.py (modes predict | node | peptides | trained | time; results/real_E8_<mode>.jsonl, one row
  per task, resumable; SMOKE=1 for a quick test). real_data.py loads the data; models_real.py runs the models.py zoo on sparse
  graphs (GPU via MPS gather-scatter, per-layer checkpointing from depth 16, width 64), checked to reproduce models.Net exactly
  on CPU and to 3e-6 on MPS. Node protocol (GPU budget, ~0.13 s per attention layer-epoch on Roman-empire, ~0.5 s on
  Amazon-ratings): depths 2-32, lr in {0.01, 0.003} tuned per (model, depth) on split 0, dropout 0.2, wd 5e-4, <= 150 epochs,
  patience 30, splits 0-2 of the official 10. `trained` re-runs each trained gauss_cool network with the numba dynamics (t0 = 11)
  to 4000 layers: it reproduces the trained test accuracy at the trained depth. Peptides protocol (~6 s per attention
  layer-epoch on the M1 GPU at batch 128; larger batches do not help): depths 2/8/32, models mlp gcn gcn_res gcnii gat_res
  gauss_const gauss_cool, 1 seed, 30 epochs, Adam 1e-3 with cosine decay, dropout 0.1, bond features unused.
  run_E8.sh runs predict (CPU) alongside node then peptides (GPU), and trained (CPU) after node; logs results/log_E8_*.txt.
  The 28 Oct shutdown also lost the freshly extracted Peptides .pt files (never flushed); re-extracted from the verified zip,
  hashes match data/README.md.
- The Mac shut down on 28 Oct (battery) and killed the alt runs at E6b-alt 50/1005, E7-alt 225/670; both scripts now take
  RESUME=1, and run_alt_resume.sh restarts them (3 + 3 workers) and rebuilds the tables.
- 28 Oct, 06:46: E8 switched from run_E8.sh (one GPU worker; ~45 GPU-hours left at the measured speed) to run_E8b.sh. In it,
  the GPU and the CPU share the node and peptides task lists through claim files (results/real_E8_claims, cleared by the script;
  env E8_DEVICE, E8_THREADS). On the M1 the CPU (6 threads) trains these sparse graphs about as fast as the GPU: Amazon gcn_res K=32
  3.0 s/epoch on the CPU, Peptides gauss_cool K=32 2.1 s/batch on the CPU vs 2.45 on the GPU, and the GPU is at 99% load. Rows now
  record `device`. The CPU worker runs `trained` after its node share, and once more at the end. Stopping the old queue lost one
  in-progress run (Roman gat_res K=32, ~4 min).
- E8 predict: the node-dataset rows were RERUN (old file kept as results/real_E8_predict_v1.jsonl). The old edge statistics called an
  edge "predicted persist" if it was at least kappa long, which ignores merging through a path of short edges. edge_stats now reports
  "pair_*" (old definition) and "part_*" (threshold-graph components, the rule actually tested). Component-level false persistence:
  Roman 0-1.6%, Amazon 0-5% (the pairwise count had 35% on Amazon x0.5). False merge: up to 40%, so errors are mostly one-sided, as on
  chains. The probe on Roman with LDA margin 2 kappa stays at 0.588 to 4000 layers. PCA x0.5 gives 0.29 and x2 gives 0.53 (majority
  class 0.14). On Amazon the probe is at the majority level (0.37) at every depth. Partition stability (ARI between layers 2000 and
  4000): Roman 0.92-1.00, Amazon 0.61-0.99. This is now Table tab:predict-real and the "Real graphs (E8)" paragraph of Sec. 7.

## E9 / Section 7: learned query-key attention (28 Oct 2026)
- Theory (main.tex Sec. 7, appendix D): score f(x,y) = x^T J y - 1/2 y^T Omega y + b^T y (dot product with biases: Omega = 0;
  Gaussian with metric: J = Omega = 2W^T W). Directed margin m_ij = f(z_i,z_i) - f(z_i,z_j); decomposition = Gaussian part in J_s/2
  + key potential V(z_i) - V(z_j) + rotation; m_ij + m_ji = (z_i - z_j)^T J_s (z_i - z_j).
- Two blocks: closed 2D margin map (eq. margin-map), checked against feature-space dynamics (rel. err < 1e-13, random J, Omega, b in R^3).
  k = (1 - P a_A a_B) / ((1 + rho_A a_A)(1 + rho_B a_B)) > 0 when sigma = e^T J_s e > 0: blocks never cross.
  Theorem: merge iff min margin <= 1 at some layer; symmetric pairs follow the Gaussian map exactly in sqrt(M); bounds and C1';
  critical min margin = 1 + (2+o(1)) log log t0 / log t0 (same leading order as kappa^2).
- Numerics (results/learned_twoblock.json): critical weaker margin / kappa^2 in [0.927, 1.070] for partner margins up to 10,
  t0 in {11, 1e3}, omega in {0, 1}; at every computed point the two critical margins sum to > 2 kappa^2 (symmetric split most robust).
  Diagonal: 2.3674 at t0 = 11 (kappa 1.5386), 1.6574 at t0 = 1e3. Float bisection, up to 2e6 layers, unresolved counted as persisting.
- Oscillation (sigma < 0, z_A = -z_B, rho >= 1): |gap| follows the Gaussian map with cut ratios 1/rho; blocks swap every layer forever
  above kappa(t0; 1/rho, 1/rho). Checked numerically (rel. err 3e-14 in persisting runs).
- Stress test: 20000 random margin-map trajectories, no violation of "min margin <= 1 stays <= 1", no crossings.
- Graphs (results/learned_graphs.json, 150 runs: E4 SBMs, p_out 0.01, sigma 0.1/0.2/0.3, J = 2I tied or 2I + 0.3 G untied, features
  shifted by nu e_1 towards block 1, 4000 layers). Absorption into block 1: 29/30 runs at nu = 3, 4/30 fully at nu = 2.
  Mean ARI (directed rule with relative margins m+ / symmetric part only): t1 = 0: 0.55 / 0.66; t1 = 5: 0.93 / 0.90; t1 = 20: 0.96 / 0.92;
  at nu >= 2, t1 = 20: 0.91 / 0.81. Directed rule at t1 = 0 over-merges (learned margins are first order in noise) but has false persistence
  in only 2/150 runs; symmetric rule has false persistence in 83 runs. No oscillation (max last-layer step 4e-4).
  The plain self-margin rule (min m_ij) is worse at t1 = 0 because a node can prefer a neighbour over itself; m+ fixes part of that.

## E9 trained: cooled dot-product networks (qk_cool), 28 Oct 2026
- Sweep exp_train_qk.py: 300 tasks, 3.6 h. Clean CSBM 0.995-0.998 all depths; hetero 0.82 (K=2) -> 0.60-0.68 (K>=8), cold start picked 28/30, vs MLP/Gaussian 0.85 (no frozen phase: margins can be negative); noisy 0.967/0.960/0.897 at K=2/8/16, ~0.6 from 32; rings peak 0.80 sparse / 0.69 dense at K=16 (Gaussian 0.78/0.60, PairNorm 0.94/0.91).
- exp_learned.py trained (150 reported runs, 4000 layers): rerun matches training accuracy to 3e-8.
  - cold start (46): mean loss 0.004 (max 0.048), all resolved; directed ARI 0.13/0.66/0.97 at t1=0/5/20 (sym 0.00/0.57/0.96); <0.9 in 2 runs at t1=20.
  - default start (104): mean loss 0.29 (max 0.70); 47 unresolved at 4000 (30 sparse rings, 10 dense, 7 clean); resolved 57: directed ARI 0.47/0.71/0.88 (sym 0.49/0.81/0.94); false persistence in 1/57 at t1=20.
  - J_s indefinite in 37 runs (31 rings), lambda_min down to -1.81; no oscillation (max last step 4e-5).

## E8 Peptides-func (done 29 Oct 2026)
- 19 runs (6 models x K in {2,8,32} + MLP), one seed, 30 epochs, test AP: MLP .253; GCN .349/.337/.169; GCN+res .346/.378/.268;
  GCNII .307/.310/.281; GAT+res .373/.421/.379; Gaussian const .253/.247/.247; Gaussian cooled .254/.246/.247.
- Gaussian networks stay at MLP level (max gap .007 AP, K=8) but do not use the graph; the paper's "within 0.005" claim
  is now stated for the seven node tasks, with 0.007 AP on Peptides. Table: paper/tables/tab_train_e8_pep.tex.
