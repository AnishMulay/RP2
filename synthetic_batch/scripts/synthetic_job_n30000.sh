#!/bin/bash
#SBATCH -J synthetic_n30000
#SBATCH -o synthetic_batch/logs/synthetic_n30000-%j.out
#SBATCH -e synthetic_batch/logs/synthetic_n30000-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:00:00
#SBATCH -p rtx2060super

export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

# Activate Conda environment (match your existing style)
PATH=/usr/bin:/bin:$PATH conda activate clusterenv

echo "Starting synthetic job for N=30000"
echo "Writing CSV: synthetic_batch/results/results_n30000.csv"

python -u e1_synthetic_vs_exact.py \
  --n_values 30000 \
  --epsilon 0.01 \
  --k 4 \
  --trials 10 \
  --seed 42 \
  --csv "synthetic_batch/results/results_n30000.csv"

echo "Finished synthetic job for N=30000"
