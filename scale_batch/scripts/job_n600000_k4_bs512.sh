#!/bin/bash
#SBATCH -J synth_n600000_k4_bs512
#SBATCH -o scale_batch/logs/synth_n600000_k4_bs512-%j.out
#SBATCH -e scale_batch/logs/synth_n600000_k4_bs512-%j.err
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

echo "Starting Job for N=600000, K=4, BatchSize=512"

python -u e3_scalability_synthetic_klevel.py \
  --n_values 600000 \
  --dim 2 \
  --epsilon 0.01 \
  --k 4 \
  --batch_size 512 \
  --seed 42 \
  --csv scale_batch/results/results_n600000_k4_bs512.csv

echo "Job N=600000, K=4, BatchSize=512 Complete"

conda deactivate || true
