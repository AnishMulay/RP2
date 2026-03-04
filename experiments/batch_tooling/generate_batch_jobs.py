import os

# Configuration
min_n = 100
max_n = 5100
step = 100
template_path = "batch/scripts/job_n{}.sh"

# Slurm Header Template (Based on your provided script)
# Note: We direct output to batch/results/results_n{n}.csv to prevent write conflicts
slurm_template = """#!/bin/bash
#SBATCH -J mnist_n{n}
#SBATCH -o batch/logs/mnist_n{n}-%j.out
#SBATCH -e batch/logs/mnist_n{n}-%j.err
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

echo "Starting Job for N={n}"

# Run the script for a SPECIFIC N
# We use --n_values {n} so it only runs that one size
# We write to a unique CSV for this job
python -u e1_mnist_vs_exact.py \\
  --n_values {n} \\
  --epsilon 0.01 \\
  --k 4 \\
  --trials 10 \\
  --seed 42 \\
  --csv batch/results/results_n{n}.csv

echo "Job N={n} Complete"

conda deactivate || true
"""

def generate_scripts():
    count = 0
    # Range is inclusive of max_n, so we add 1 to the stop value
    for n in range(min_n, max_n + 1, step):
        script_content = slurm_template.format(n=n)
        filename = template_path.format(n)
        
        with open(filename, "w") as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(filename, 0o755)
        count += 1
        
    print(f"Successfully generated {count} batch scripts in batch/scripts/")

if __name__ == "__main__":
    generate_scripts()