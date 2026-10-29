#!/bin/zsh
# E8 queue for one Apple-silicon machine: the GPU (MPS) and the CPU share the node and peptides task lists through
# claim files; the CPU also runs the trained-model predictions after the node sweep, and once more at the end for
# networks the GPU saved last. Every mode resumes. Run from anywhere: the script moves to the repository root.
cd "$(dirname "$0")/.."
rm -rf results/real_E8_claims
python3 -W ignore -m experiments.exp_real predict >> results/log_E8_predict.txt 2>&1 &
(
  E8_DEVICE=mps python3 -W ignore -m experiments.exp_real node >> results/log_E8_node.txt 2>&1
  E8_DEVICE=mps python3 -W ignore -m experiments.exp_real peptides >> results/log_E8_peptides.txt 2>&1
) &
(
  E8_DEVICE=cpu E8_THREADS=6 python3 -W ignore -m experiments.exp_real node >> results/log_E8_node_cpu.txt 2>&1
  python3 -W ignore -m experiments.exp_real trained >> results/log_E8_trained.txt 2>&1
  E8_DEVICE=cpu E8_THREADS=6 python3 -W ignore -m experiments.exp_real peptides >> results/log_E8_peptides_cpu.txt 2>&1
) &
wait
python3 -W ignore -m experiments.exp_real trained >> results/log_E8_trained.txt 2>&1
echo E8_DONE
