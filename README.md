# Predicting cluster formation in deep graph attention networks

Code, results and LaTeX source for the paper *Predicting Cluster Formation in Deep Graph Attention Networks: Phase
Transitions, Topology, and Feature Geometry* (`paper/main.pdf`).

The paper studies graph attention under logarithmic cooling, `T_t = c / log(t + t0)`, and predicts which feature clusters
survive depth. It covers an exact two-block threshold law, one-sided errors and nonlocal rescue in layered chains, planted
partitions on general graphs, and learned query–key attention. It then tests the resulting prediction rule on synthetic
graphs, real graphs and trained networks.

## Layout

```
gnnlimit/        library code
  core.py          two-block and chain dynamics, kappa(t0) estimates and validated enclosures, certificates C1/C2
  full.py          node-level dynamics on arbitrary graphs (vector features), threshold-graph rule
  rigorous_mp.py   160-bit mpmath interval proofs of the computer-assisted statements (Theorem 4)
  models.py        model zoo and training loop (GCN, GCNII, APPNP, SGC, GAT, GRAND, PairNorm, Gaussian and dot-product attention)
  models_real.py   the same zoo on sparse real graphs
  real_data.py     loaders for Roman-empire, Amazon-ratings, Peptides-func (data/raw)
  summarize_train.py  per-run model selection and summaries of the training sweeps
experiments/     one driver per experiment (run as modules from the repository root)
theory/          numerical checks of the identities and constants used in the proofs, and early proof notes
paper/           main.tex, appendix.tex, references.bib, make_figures.py, make_tables.py, figures/, tables/
scripts/         reproduce.sh (everything, in order) and run_E8.sh (real-graph queue on GPU + CPU)
results/         result files (JSON/JSONL/NPZ) and logs; every number in the paper is generated from these
data/            data/README.md lists sources, sizes and SHA-256 of the real-graph files (data/raw is not tracked)
```

## Experiments

| Id | Driver | Paper | Output |
|---|---|---|---|
| E1–E3 | `experiments.exp_theory E1 E2 E3` | Sec. 4–5, Fig. 1–3, Thm. 4 | `results/theory_E1..3.*` |
| E1e | `experiments.exp_second_order` | Sec. 4, Fig. 2b | `results/theory_E1e.json` |
| proofs | `gnnlimit.rigorous_mp` | Thm. 4 (a, b) | log; exits nonzero on failure |
| E4, E5 | `experiments.exp_graphs E4 E5` | Sec. 6, 8, Fig. 4–5 | `results/graphs_E4.json`, `graphs_E5.json` |
| E9 | `experiments.exp_learned twoblock graphs` | Sec. 7, Fig. 7 | `results/learned_twoblock.json`, `learned_graphs.json` |
| E6b | `experiments.exp_train2` (and `ALT=1`) | Sec. 9, Fig. 6 | `results/train_E6b.json`, `train_E6b_alt.json` |
| E7 | `experiments.exp_longrange run` (and `ALT=1`) | Sec. 9, Fig. 6 | `results/longrange_E7.json`, `longrange_E7_alt.json` |
| E6b/E7, dot product | `experiments.exp_train_qk` | Sec. 9 | `results/train_E6b_qk.json`, `longrange_E7_qk.json`, `results/qk_states/` |
| E9, trained | `experiments.exp_learned trained` | Sec. 9 | `results/learned_trained.json` |
| E8 | `experiments.exp_real predict / node / peptides / trained` | Sec. 8–9 | `results/real_E8_*.jsonl` |

`experiments.exp_train` is the first version of the CSBM sweep (E6). It is kept because E6b imports its data generator.

## Running

```bash
python3 -m pip install -r requirements.txt
```

Run every command from this directory, since the drivers read and write `results/` relative to it. For example:

```bash
python3 -m experiments.exp_learned twoblock
```

To rebuild the tables, figures and paper from the existing result files:

```bash
python3 paper/make_tables.py && python3 paper/make_figures.py && (cd paper && latexmk -pdf main.tex)
```

To regenerate everything from scratch:

```bash
./scripts/reproduce.sh
```

From scratch, the training sweeps take about 16 hours on an 8-core laptop and the real-graph queue one to two days. The
sweeps and E8 resume after an interruption (`RESUME=1` for E6b and E7).

The real-graph experiments need the three files in `data/README.md` in `data/raw/`. They are third-party data and are read
only with `np.load(..., allow_pickle=False)` and `torch.load(..., weights_only=True)`.

`results/qk_states/` and `results/real_E8_states/` hold the trained networks that the `trained` analyses rerun. They are not
tracked by git; the training drivers recreate them.
