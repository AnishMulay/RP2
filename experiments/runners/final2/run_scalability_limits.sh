#!/bin/bash
#SBATCH -J final2_scalability
#SBATCH -o experiments/runners/final2/results/scalability_limits-%j.out
#SBATCH -e experiments/runners/final2/results/scalability_limits-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 16:00:00
#SBATCH -p rtx2060super
#SBATCH --exclude=c34

export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
PATH=/usr/bin:/bin:$PATH conda activate clusterenv

echo "=============================="
echo "Scalability Limits Sweep"
echo "Started: $(date)"
if [ -n "${FINAL2_EXP13_DATASET:-}" ]; then
  echo "Dataset filter: ${FINAL2_EXP13_DATASET}"
fi
echo "=============================="

if [ -n "${FINAL2_EXP13_DATASET:-}" ]; then
  python -u experiments/runners/final2/run_experiments.py --run 13 --exp13-dataset "${FINAL2_EXP13_DATASET}"
else
  python -u experiments/runners/final2/run_experiments.py --run 13
fi

echo "Finished: $(date)"
conda deactivate || true
