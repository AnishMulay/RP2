#!/usr/bin/env python3
"""
Submit all Final2 experiments as one SLURM batch.

Run from the repository root:
  python experiments/runners/final2/submit_batch.py
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


FINAL2_DIR = Path(__file__).resolve().parent

EXP_CONFIG = {
    1:  {"name": "MNIST Equal Proxy",           "time": "12:00:00"},
    2:  {"name": "MNIST Biased Proxy",          "time": "12:00:00"},
    3:  {"name": "EMNIST Equal Proxy",          "time": "12:00:00"},
    4:  {"name": "EMNIST Biased Proxy",         "time": "12:00:00"},
    5:  {"name": "NYC Scalability",             "time": "04:00:00"},
    6:  {"name": "CIFAR SIFT Proxy",            "time": "03:00:00"},
    7:  {"name": "CIFAR SIFT Scalability",      "time": "04:00:00"},
    8:  {"name": "Newsgroups Proxy",            "time": "03:00:00"},
    9:  {"name": "Newsgroups Scalability",      "time": "04:00:00"},
    10: {"name": "MNIST Dissimilar Diagnostic", "time": "01:00:00"},
    11: {"name": "Landmark Density",            "time": "04:00:00"},
}


def _require_repo_root():
    expected = Path.cwd() / "experiments" / "runners" / "final2"
    if expected.resolve() != FINAL2_DIR:
        print(
            "Error: run this script from the repository root "
            "(the directory containing experiments/).",
            file=sys.stderr,
        )
        sys.exit(1)


def _write_experiment_script(script_path, log_dir, run_dir, exp_id, exp_name, time_limit):
    script = f"""#!/bin/bash
#SBATCH -J final2_exp{exp_id:02d}
#SBATCH -o {log_dir}/exp{exp_id:02d}-%j.out
#SBATCH -e {log_dir}/exp{exp_id:02d}-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t {time_limit}
#SBATCH -p rtx2060super

export PYTHONUNBUFFERED=1

cd "${{SLURM_SUBMIT_DIR:-$PWD}}"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

PATH=/usr/bin:/bin:$PATH conda activate clusterenv

echo "=============================="
echo "Starting Exp {exp_id}: {exp_name}"
echo "Run dir: {run_dir}"
echo "=============================="

python -u experiments/runners/final2/run_experiments.py \\
  --run {exp_id} \\
  --results-dir "{run_dir}"

echo "=============================="
echo "Exp {exp_id} complete"
echo "=============================="

conda deactivate || true
"""
    script_path.write_text(script, encoding="utf-8")
    os.chmod(script_path, 0o755)


def _write_aggregate_script(script_path, log_dir, run_dir):
    script = f"""#!/bin/bash
#SBATCH -J final2_aggregate
#SBATCH -o {log_dir}/aggregate-%j.out
#SBATCH -e {log_dir}/aggregate-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -p rtx2060super

export PYTHONUNBUFFERED=1

cd "${{SLURM_SUBMIT_DIR:-$PWD}}"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

PATH=/usr/bin:/bin:$PATH conda activate clusterenv

echo "=============================="
echo "Starting aggregation"
echo "Run dir: {run_dir}"
echo "=============================="

python -u experiments/runners/final2/aggregate_results.py \\
  --results-dir "{run_dir}"

echo "Aggregation complete"

conda deactivate || true
"""
    script_path.write_text(script, encoding="utf-8")
    os.chmod(script_path, 0o755)


def _parse_job_id(stdout):
    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    if not match:
        return None
    return match.group(1)


def _submit(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        print("Error: sbatch command not found.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print("Error: sbatch failed.", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout.strip(), file=sys.stderr)
        if exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        sys.exit(1)

    job_id = _parse_job_id(result.stdout)
    if job_id is None:
        print("Error: could not parse sbatch job ID.", file=sys.stderr)
        print(result.stdout.strip(), file=sys.stderr)
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        sys.exit(1)
    return job_id


def main():
    _require_repo_root()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = FINAL2_DIR / "results" / f"run_{run_id}"
    log_dir = run_dir / "logs"
    scripts_dir = run_dir / "scripts"

    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Final2 run directory: {run_dir.resolve()}")

    script_paths = {}
    for exp_id, config in EXP_CONFIG.items():
        script_path = scripts_dir / f"exp{exp_id:02d}.sh"
        _write_experiment_script(
            script_path=script_path,
            log_dir=log_dir.resolve(),
            run_dir=run_dir.resolve(),
            exp_id=exp_id,
            exp_name=config["name"],
            time_limit=config["time"],
        )
        script_paths[exp_id] = script_path

    agg_script_path = scripts_dir / "aggregate.sh"
    _write_aggregate_script(
        script_path=agg_script_path,
        log_dir=log_dir.resolve(),
        run_dir=run_dir.resolve(),
    )

    submitted = []
    for exp_id in range(1, 12):
        job_id = _submit(["sbatch", str(script_paths[exp_id])])
        submitted.append((exp_id, job_id))

    job_ids = [job_id for _, job_id in submitted]
    dependency_str = "afterok:" + ":".join(job_ids)
    agg_job_id = _submit([
        "sbatch",
        f"--dependency={dependency_str}",
        str(agg_script_path),
    ])

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  Final2 Batch Submitted — Run: {run_id:<23} ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Run dir : {run_dir.resolve()}")
    print(f"  Logs    : {log_dir.resolve()}/")
    print()
    for exp_id, job_id in submitted:
        config = EXP_CONFIG[exp_id]
        print(
            f"  Exp {exp_id:>2}  [job {job_id}]  "
            f"{config['name']:<29} ({config['time']})"
        )
    print(f"  Agg     [job {agg_job_id}]  Aggregation (runs after all above)")
    print()
    first_job_id = submitted[0][1]
    print(f"  Monitor: tail -f {log_dir.resolve()}/exp01-{first_job_id}.out")
    print()
    print(f"  Run dir : {run_dir.resolve()}")


if __name__ == "__main__":
    main()
