#!/usr/bin/env python3
import os

# Configuration
min_n = 50000
max_n = 1000000
step = 50000
k_values = [2, 4, 8]
batch_sizes = [512, 256]
template_path = "scale_batch/scripts/job_n{n}_k{k}_bs{bs}.sh"

# Slurm Header Template
# Changes made:
# 1. Output/Error logs point to scale_batch/logs
# 2. Single trial per script
# 3. CSV output points to scale_batch/results
slurm_template = """#!/bin/bash
#SBATCH -J synth_n{n}_k{k}_bs{bs}
#SBATCH -o scale_batch/logs/synth_n{n}_k{k}_bs{bs}-%j.out
#SBATCH -e scale_batch/logs/synth_n{n}_k{k}_bs{bs}-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:00:00
#SBATCH -p rtx2060super

export PYTHONUNBUFFERED=1

cd "${{SLURM_SUBMIT_DIR:-$PWD}}"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

# Activate Conda environment
PATH=/usr/bin:/bin:$PATH conda activate clusterenv

echo "Starting Job for N={n}, K={k}, BatchSize={bs}"

python -u e3_scalability_synthetic_klevel.py \\
  --n_values {n} \\
  --dim 2 \\
  --epsilon 0.01 \\
  --k {k} \\
  --batch_size {bs} \\
  --seed 42 \\
  --csv scale_batch/results/results_n{n}_k{k}_bs{bs}.csv

echo "Job N={n}, K={k}, BatchSize={bs} Complete"

conda deactivate || true
"""

def generate_scripts():
    count = 0
    # Range is inclusive of max_n
    for n in range(min_n, max_n + 1, step):
        for k in k_values:
            for bs in batch_sizes:
                script_content = slurm_template.format(n=n, k=k, bs=bs)
                filename = template_path.format(n=n, k=k, bs=bs)
                
                with open(filename, "w") as f:
                    f.write(script_content)
                
                # Make the script executable
                os.chmod(filename, 0o755)
                count += 1
        
    print(f"Successfully generated {count} batch scripts in scale_batch/scripts/")
    print(f"Range: {min_n} to {max_n}, Step: {step}")

if __name__ == "__main__":
    generate_scripts()
