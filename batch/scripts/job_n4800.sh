#!/bin/bash
#SBATCH -J mnist_n4800
#SBATCH -o batch/logs/mnist_n4800-%j.out
#SBATCH -e batch/logs/mnist_n4800-%j.err
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

# Activate Conda environment
PATH=/usr/bin:/bin:$PATH conda activate clusterenv

echo "Starting Job for N=4800"

# Run the script for a SPECIFIC N
# We use --n_values 4800 so it only runs that one size
# We write to a unique CSV for this job
python -u e1_mnist_vs_exact.py \
  --n_values 4800 \
  --epsilon 0.01 \
  --k 4 \
  --trials 10 \
  --seed 42 \
  --csv batch/results/results_n4800.csv

echo "Job N=4800 Complete"

conda deactivate || true
