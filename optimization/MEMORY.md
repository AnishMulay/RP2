# OPTIMIZATION SESSION MEMORY
Last updated: 2026-04-02
Agent: Codex GPT-5

## Experiment Configuration
- Dataset: MNIST (784-dim, L1/Manhattan distance)
- Sizes: n = 1000, 2500, 5000
- Trials: 1 per size (can increase later)
- Epsilon: 0.01, k: 4, seed: 42
- Run command: python -u experiments/runners/e1_mnist_vs_exact.py --epsilon 0.01 --k 4 --trials 1 --seed 42 --csv results_e1_mnist.csv
- HPC: SSH into node, activate clusterenv conda environment, run from repo root RP2/

## Baseline (collected 2026-04-02, sizes 1000 / 5000 / 10000 — NOT the target sizes)
Note: Target sizes are 1000, 2500, 5000. n=2500 baseline is missing. n=10000 included for reference only.

| n     | Exact (s) | 2-Level (s) | k-Level (s) | 2-Level solver (s) | k-Level solver (s) |
|-------|-----------|-------------|-------------|--------------------|--------------------|
| 1000  | 0.11      | 66.58       | 58.52       | 66.19              | 58.38              |
| 5000  | 4.23      | 94.95       | 55.63       | 93.97              | 54.92              |
| 10000 | 22.14     | 221.54      | 67.62       | 215.78             | 62.64              |

Known issues at baseline:
- compute_cost_matrix_L1 in e1_mnist_vs_exact.py uses a Python row loop — killed run at n=15000
- Solver time is essentially 100% of total runtime; clustering is <5%
- Approximate solver is significantly slower than exact at all tested sizes
- k-Level scales better than 2-Level (67s vs 221s at n=10000)

## Current Best
Round 1 candidate:
- Vectorized exact L1 cost-matrix build with `torch.cdist`
- Vectorized matching-cost evaluation in the runner
- Removed always-on solver diagnostics/statistics and per-iteration CUDA cache flushes
- Status: awaiting HPC benchmark on target sizes (1000, 2500, 5000)

## Off-limits (never touch)
- Push-relabel algorithm logic inside bipartite.py
- Clustering phase logic in two_level.py and k_level.py
- Approximation epsilon handling
- CSV output column schema

## Iteration History
### Round 1 (prepared 2026-04-02)
What I changed:
- `experiments/runners/e1_mnist_vs_exact.py`
  - Replaced the Python row loop in `compute_cost_matrix_L1` with `torch.cdist(..., p=1)` on float64 tensors.
  - Replaced Python per-match cost loops for both approximate solvers with a single vectorized L1 reduction on the matched tensors.
  - Removed pre-solve `torch.cuda.empty_cache()` calls and wrapped CUDA timing/memory calls so CPU execution still works.
- `src/clustered_push_relabel/solvers/bipartite.py`
  - Removed always-on cluster analysis / active-center / final-match printing from non-verbose runs.
  - Skipped the expensive cluster statistics computation entirely unless `verbose=True`.
  - Removed `torch.cuda.empty_cache()` from solver setup and from each solve iteration.
  - Kept CUDA synchronizations only where needed and guarded them by device type.
  - Short-circuited `calculate_final_stats()` when not verbose so it no longer computes an extra full distance pass.

Results:
- Local validation only:
  - `python -m compileall experiments/runners/e1_mnist_vs_exact.py src/clustered_push_relabel/solvers/bipartite.py`
  - Tiny synthetic smoke test passed for both `TwoLevelBipartiteSolver` and `KLevelBipartiteSolver`.
- No HPC timing numbers yet.

What I learned:
- The baseline code spends measurable work outside the actual push-relabel logic: Python loops in the runner, unconditional cluster-stat reporting, and repeated allocator/cache management.
- Both solver classes had the same avoidable overhead, so shared cleanup is the highest-confidence first round.

Current status:
- Ready for HPC measurement with the standard MNIST command.
- Need results for n = 1000, 2500, 5000 to determine whether this becomes the new best baseline.

## Active Hypotheses
- Round 1 hypothesis: removing non-algorithmic overhead around CSR construction, solver iteration, and result accounting should reduce both 2-level and k-level wall time without changing costs or matching behavior.
- Expected gain: modest-to-meaningful runtime reduction, with the largest wins likely from eliminating per-iteration `torch.cuda.empty_cache()` and from vectorizing runner-side cost computation.
