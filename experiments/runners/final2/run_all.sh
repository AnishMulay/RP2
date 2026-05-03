#!/bin/bash
#SBATCH -J final2_all
#SBATCH -o experiments/runners/final2/results/run_all-%j.out
#SBATCH -e experiments/runners/final2/results/run_all-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -p rtx2060super

export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

PATH=/usr/bin:/bin:$PATH conda activate clusterenv

echo "=============================="
echo "Starting all Final2 experiments"
echo "Started: $(date)"
echo "=============================="

python -u experiments/runners/final2/run_experiments.py --all

echo "=============================="
echo "All experiments complete"
echo "Finished: $(date)"
echo "=============================="

conda deactivate || true
