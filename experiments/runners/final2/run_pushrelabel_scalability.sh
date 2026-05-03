#!/bin/bash
#SBATCH -J final2_pr_scale
#SBATCH -o experiments/runners/final2/results/pushrelabel_scalability-%j.out
#SBATCH -e experiments/runners/final2/results/pushrelabel_scalability-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -p rtx2060super
#SBATCH --exclude=c34

export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
PATH=/usr/bin:/bin:$PATH conda activate clusterenv

echo "=============================="
echo "Push-Relabel Scalability Sweep"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=============================="

python -u experiments/runners/final2/run_experiments.py --run 14

echo "=============================="
echo "Finished: $(date)"
echo "=============================="

conda deactivate || true
