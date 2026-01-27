#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# -----------------------
# Config (edit if needed)
# -----------------------
MIN_N = 500
MAX_N = 30000
STEP  = 500

TIME_LIMIT = "01:00:00"
PARTITION  = "rtx2060super"
CPUS_PER_TASK = 16
CONDA_ENV = "clusterenv"

# Experiment defaults
EPSILON = 0.01
K_LEVELS = 4
TRIALS = 1
SEED = 42

REPO_ENTRYPOINT = "e1_synthetic_vs_exact.py"

BASE_DIR = Path("synthetic_batch")
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
AGG_DIR = BASE_DIR / "agg"

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -J synthetic_n{n}
#SBATCH -o {logs_dir}/synthetic_n{n}-%j.out
#SBATCH -e {logs_dir}/synthetic_n{n}-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task={cpus}
#SBATCH -t {time_limit}
#SBATCH -p {partition}

export PYTHONUNBUFFERED=1

cd "${{SLURM_SUBMIT_DIR:-$PWD}}"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

# Activate Conda environment (match your existing style)
PATH=/usr/bin:/bin:$PATH conda activate {conda_env}

echo "Starting synthetic job for N={n}"
echo "Writing CSV: {csv_out}"

python -u {entrypoint} \\
  --n_values {n} \\
  --epsilon {epsilon} \\
  --k {k} \\
  --trials {trials} \\
  --seed {seed} \\
  --csv "{csv_out}"

echo "Finished synthetic job for N={n}"
"""

def main():
    # Reset batch directories so we always start clean
    for d in [SCRIPTS_DIR, LOGS_DIR, RESULTS_DIR, AGG_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    ns = list(range(MIN_N, MAX_N + 1, STEP))
    print(f"Generating {len(ns)} job scripts for N in [{MIN_N}, {MAX_N}] step {STEP}")

    for n in ns:
        job_path = SCRIPTS_DIR / f"synthetic_job_n{n}.sh"
        csv_out = RESULTS_DIR / f"results_n{n}.csv"

        script = SLURM_TEMPLATE.format(
            n=n,
            logs_dir=str(LOGS_DIR),
            cpus=CPUS_PER_TASK,
            time_limit=TIME_LIMIT,
            partition=PARTITION,
            conda_env=CONDA_ENV,
            entrypoint=REPO_ENTRYPOINT,
            epsilon=EPSILON,
            k=K_LEVELS,
            trials=TRIALS,
            seed=SEED,
            csv_out=str(csv_out),
        )

        job_path.write_text(script)
        os.chmod(job_path, 0o755)

    print(f"Done. Job scripts written to: {SCRIPTS_DIR}")
    print("Example:")
    ex = SCRIPTS_DIR / f"synthetic_job_n{MIN_N}.sh"
    print(f"  {ex}")

if __name__ == "__main__":
    main()
