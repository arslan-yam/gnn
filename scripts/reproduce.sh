#!/bin/zsh
# Reproduces every result file, table and figure of the paper, in dependency order. Each step writes results/*
# and a log; most steps resume after an interruption. Wall times on an 8-core M1 laptop are given for the long steps.
# Run from anywhere: the script moves to the repository root.
set -e
cd "$(dirname "$0")/.."
mkdir -p results
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4}
py() { python3 -W ignore -m "$@"; }

# Theory validation (Sections 4-6): E1-E3 two-block law, three- and five-layer chains; E1e second-order term
py experiments.exp_theory E1 E2 E3        > results/log_theory.txt 2>&1
py experiments.exp_second_order           > results/log_E1e.txt 2>&1
py gnnlimit.rigorous_mp                   > results/log_rigorous.txt 2>&1   # 160-bit interval proofs (Theorem 4)

# Graphs (Sections 6 and 8): E4 noisy community graphs, E5 learned metric
py experiments.exp_graphs E4 E5           > results/log_graphs.txt 2>&1

# Learned attention (Section 7): E9 two-block margin map and contextual SBMs
py experiments.exp_learned twoblock graphs > results/log_learned.txt 2>&1

# Training (Section 9): E6b CSBMs and E7 rings, both temperature grids (~12 h), then the dot-product network (~4 h)
WORKERS=6 py experiments.exp_train2                 > results/log_E6b.txt 2>&1
WORKERS=6 ALT=1 py experiments.exp_train2           > results/log_E6b_alt.txt 2>&1
WORKERS=6 py experiments.exp_longrange run          > results/log_E7.txt 2>&1
WORKERS=6 ALT=1 py experiments.exp_longrange run    > results/log_E7_alt.txt 2>&1
WORKERS=3 py experiments.exp_train_qk               > results/log_train_qk.txt 2>&1
py experiments.exp_learned trained                  > results/log_learned_trained.txt 2>&1

# Real graphs (Sections 8-9): needs data/raw (see data/README.md); GPU + CPU queue (about 1-2 days)
./scripts/run_E8.sh

# Paper
python3 -W ignore paper/make_tables.py
python3 -W ignore paper/make_figures.py
(cd paper && latexmk -pdf -interaction=nonstopmode main.tex)
