#!/bin/bash
# Run from repo root: bash experiments/runners/final2/submit_all.sh

LOGDIR="experiments/runners/final2/results/logs"
mkdir -p "$LOGDIR"

for EXP_ID in 1 2 3 4 5 6 7 8 9 10 11; do
sbatch \
  --job-name="final2_exp${EXP_ID}" \
  --output="${LOGDIR}/exp${EXP_ID}-%j.out" \
  --error="${LOGDIR}/exp${EXP_ID}-%j.err" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --time=12:00:00 \
  --partition=rtx2060super \
  --exclude=c34 \
  --wrap="
export PYTHONUNBUFFERED=1
cd \${SLURM_SUBMIT_DIR:-\$PWD}
if [ -f \$HOME/.bashrc ]; then source \$HOME/.bashrc; fi
PATH=/usr/bin:/bin:\$PATH conda activate clusterenv
echo 'Starting Exp ${EXP_ID} on '\$(hostname)' at '\$(date)
python -u experiments/runners/final2/run_experiments.py --run ${EXP_ID}
echo 'Exp ${EXP_ID} done at '\$(date)
conda deactivate || true
"
echo "Submitted Exp ${EXP_ID}"
done