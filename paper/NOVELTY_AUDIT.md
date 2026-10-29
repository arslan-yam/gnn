# Novelty audit (2026-10-27)

Scope: what in our results is already known, what is plausibly new, and how the paper must be positioned.
Sources were checked via web search; several were seen at abstract level only (marked †). Before submission,
read the full texts of the starred items.

## Already known: we must cite these and must NOT claim them

| Our statement | Prior work | Consequence |
|---|---|---|
| Edges with d² < c merge and d² > c persist, asymptotically, under T_t = c/log t | **Hajek 1988**, *Cooling schedules for optimal annealing* (Math. Oper. Res. 13:311–329): annealing with T = c/log t converges iff c ≥ depth d*. Same mechanism: Σ t^{-d/c} diverges iff d ≤ c | The leading-order √c threshold is Hajek-type. It is our motivation, not our contribution. |
| Limiting cluster gaps are ≥ the interaction range (our dichotomy: 0 or ≥ √c) | **Blondel–Hendrickx–Tsitsiklis 2009** (IEEE TAC, Krause model): clusters separated by ≥ R. **Jabin–Motsch 2014**† (JDE): compactly supported kernels give local consensus | The dichotomy is the soft-kernel/cooling analog of a classical bounded-confidence fact. Present it as a lemma with citations. |
| Merging ⇔ non-summable interaction (Σ α = ∞) | **Touri–Nedić 2014** infinite-flow theorem (IEEE TAC 59(2)); **Hendrickx–Tsitsiklis 2013** cut-balanced consensus | This is the engine, not the result (the plan already warned about this). |
| Whether two groups merge depends on other agents (nonlocality) | HK "mediator agents" and equally-spaced-agent studies: Hegarty–Wedin 2014†, Slanina†, the 2R conjecture (Wang–Li–E–Chazelle 2017; Dey–Etesami–Gopalan 2025†) | Qualitative nonlocality is known. Only a *quantitative* law is new. |
| Constant temperature: clusters survive for a time ~ e^{d²/T}, then merge | **Geshkovski–Koubbi–Polyanskiy–Rigollet 2024** (dynamic metastability, exponentially long trapping); **Bruno–Pasqualotto–Agazzi, ICLR 2025** (meta-stable clustering, mean field) | Cooling turns metastability into permanent separation. Frame our constant-T vs cooling contrast against these. |
| Clustering in attention dynamics; threshold in attention sharpness | **Geshkovski et al. NeurIPS 2023**; Karagodin et al. 2024 (causal masking); **Gao–Yang–Chen 2026**† (arXiv 2608.08922, an "attention-condensation transition" above a sharpness threshold) | These are transformers on the sphere, with fixed β and no graph. Cite Gao et al. as the closest "phase transition in sharpness" result. |
| Bounded-confidence / opinion-dynamics message passing mitigates oversmoothing | **ODNet, Lv et al., ICLR 2026**† (arXiv 2310.01272): hard HK cutoffs in message passing. **Wang et al. 2025**† (arXiv 2501.19089, BIMP, "opinion dissensus") | The *method idea* (bounded confidence in GNNs) is published. Our Gaussian cooling is a soft, annealed version. Its value must come from the theory (exact, predictive radius), not from the design. |
| Learnable temperature in graph attention | "Gated GAT with learnable temperature", 2026 preprint† (arXiv 2605.29803); GATS (node-wise temperature for calibration) | Temperature itself is not new. |
| Attention cannot prevent oversmoothing (fixed T) | Wu et al. NeurIPS 2023 | Consistent with our Thm 0.1. |
| When attention helps vs GCN (structure noise vs feature noise) | Ma et al. ICML 2025 | Our noisy-feature failure regime reproduces their prediction. Cite it as agreement. |
| Bifurcations / non-homogeneous states | Turan et al. ICML 2026 | A different mechanism (activation-induced bifurcation). |
| Effective-rank measurement of oversmoothing | Zhang–Deidda–Higham–Tudisco ICLR 2026 | We report erank. |

## Plausibly new (no match found)

1. **Exact finite-depth two-block threshold law.** κ(t0; ρ_A, ρ_B) with a validated enclosure, explicit sandwich bounds,
   a jump of exactly √c at the threshold (first order), the critical-curve identity κ(t0+t) = F_t(κ(t0)), and the asymptotics to second
   order, κ − 1 = [log L − ½ log log L + ½ log(ρ_A+ρ_B) + o(1)]/L with L = log t0 (proved 28 Oct 2026 by a barrier argument). Topology
   enters only at second order and only through ρ_A + ρ_B; in physical units it shifts d²_crit by T0 log(λ/λ'). The third term
   (log log L − log λ + 2)/(4 log L) is proved by the same barriers.
   Hajek gives only the leading-order criterion, and for a different object.
2. **Comparison principle for Gaussian chains:** neighbours can only rescue a gap, never destroy one. Hence the isolated-pair
   rule makes only one-sided errors (0 false merges out of 15,888 gaps).
3. **Quantitative nonlocal rescue law:** the scaling limit φ(x) = max{x, 1 − x/2} for x < 1, with infimum 2/3, plus a
   computer-assisted finite-t0 phase diagram (99.6% certified). No quantitative analog was found in the HK literature.
4. **Planted partition at the √c scale,** with a matching impossibility result (diameter-only conditions cannot go past
   D ≤ √c). Together with the dichotomy on general graphs via infinite flow.
5. **Prediction from initial data:** the early-depth rule predicts the final partition (99.1% of gaps at 10 layers; ARI ≥ 0.99
   on noisy SBMs up to σ = 0.3).
6. Learned-metric reduction (z = Wh). This is simple and should be presented as a remark, not a contribution.

## Positioning for the paper

- Title/claim: **predicting** the cluster structure of deep attention GNNs (exact thresholds, nonlocal corrections, sharpness),
  not "cooling prevents oversmoothing".
- One sentence early in the introduction: "log cooling turns a globally supported Gaussian kernel into an effective
  bounded-confidence model whose confidence radius √c·κ(t0) we compute exactly". This connects to Hajek and HK honestly.
- The related-work section must explicitly say what is classical: Hajek's criterion, the BHT separation, infinite flow,
  qualitative nonlocality.
- The method side (ODNet, learnable temperature) means we should not sell cooling attention as a new architecture. Present it as
  the analysable model, and use E6b/E7 to show where it does and does not help.

## Still to read in full before submission
Jabin–Motsch 2014; ODNet (ICLR 2026) theory section; Gao–Yang–Chen 2026; Dey–Etesami–Gopalan 2025; Hegarty–Wedin 2014
(equally spaced agents: closest to our chain setting).
