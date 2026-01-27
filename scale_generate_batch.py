#!/usr/bin/env python3
import os

# Configuration
min_n = 50000
max_n = 1000000
step = 50000
template_path = "scale_batch/scripts/job_n{}.sh"

# Slurm Header Template
# Changes made:
# 1. Output/Error logs point to scale_batch/logs
# 2. Iterates 10 times (seeds 42-51) to get 10 trials per N
# 3. CSV output points to scale_batch/results
slurm_template = """#!/bin/bash
#SBATCH -J synth_n{n}
#SBATCH -o scale_batch/logs/synth_n{n}-%j.out
#SBATCH -e scale_batch/logs/synth_n{n}-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -p rtx2060super

export PYTHONUNBUFFERED=1

cd "${{SLURM_SUBMIT_DIR:-$PWD}}"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

# Activate Conda environment
PATH=/usr/bin:/bin:$PATH conda activate clusterenv

echo "Starting Job for N={n} (10 Trials)"

# Run 10 trials for robust statistics
# We shift the seed by the loop index (i)
for i in {{0..9}}
do
    CURRENT_SEED=$((42 + i))
    echo "Running Trial $((i+1))/10 with Seed $CURRENT_SEED"
    
    python -u e3_scalability_synthetic.py \\
      --n_values {n} \\
      --dim 2 \\
      --epsilon 0.01 \\
      --k 4 \\
      --seed $CURRENT_SEED \\
      --csv scale_batch/results/results_n{n}.csv
done

echo "Job N={n} Complete"

conda deactivate || true
"""

def generate_scripts():
    count = 0
    # Range is inclusive of max_n
    for n in range(min_n, max_n + 1, step):
        script_content = slurm_template.format(n=n)
        filename = template_path.format(n)
        
        with open(filename, "w") as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(filename, 0o755)
        count += 1
        
    print(f"Successfully generated {count} batch scripts in scale_batch/scripts/")
    print(f"Range: {min_n} to {max_n}, Step: {step}")

if __name__ == "__main__":
    generate_scripts()