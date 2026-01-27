#!/bin/bash
#SBATCH -J synth_n400000
#SBATCH -o scale_batch/logs/synth_n400000-%j.out
#SBATCH -e scale_batch/logs/synth_n400000-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -p rtx2060super

export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

# Activate Conda environment
PATH=/usr/bin:/bin:$PATH conda activate clusterenv

echo "Starting Job for N=400000 (10 Trials)"

# Run 10 trials for robust statistics
# We shift the seed by the loop index (i)
for i in {0..9}
do
    CURRENT_SEED=$((42 + i))
    echo "Running Trial $((i+1))/10 with Seed $CURRENT_SEED"
    
    python -u e3_scalability_synthetic.py \
      --n_values 400000 \
      --dim 2 \
      --epsilon 0.01 \
      --k 4 \
      --seed $CURRENT_SEED \
      --csv scale_batch/results/results_n400000.csv
done

echo "Job N=400000 Complete"

conda deactivate || true
