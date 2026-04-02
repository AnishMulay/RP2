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
Round 1 accepted (`6476894`):
- Vectorized exact L1 cost-matrix build with `torch.cdist`
- Vectorized matching-cost evaluation in the runner
- Removed always-on solver diagnostics/statistics and per-iteration CUDA cache flushes
- Best measured target-size results so far:
  - n=1000: 2-Level `57.08s`, k-Level `58.79s`
  - n=2500: 2-Level `56.27s`, k-Level `53.04s`
  - n=5000: 2-Level `72.14s`, k-Level `54.22s`
- Compared with the original baseline where directly comparable:
  - 2-Level improved by `14.3%` at n=1000 and `24.0%` at n=5000
  - k-Level regressed by `0.5%` at n=1000 but improved by `2.5%` at n=5000

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
- Local validation:
  - `python -m compileall experiments/runners/e1_mnist_vs_exact.py src/clustered_push_relabel/solvers/bipartite.py`
  - Tiny synthetic smoke test passed for both `TwoLevelBipartiteSolver` and `KLevelBipartiteSolver`.
- HPC timings on target sizes:

| n    | Exact (s) | 2-Level (s) | k-Level (s) | 2-Level solver (s) | k-Level solver (s) |
|------|-----------|-------------|-------------|--------------------|--------------------|
| 1000 | 0.1156    | 57.0807     | 58.7914     | 56.7413            | 58.6672            |
| 2500 | 0.8432    | 56.2723     | 53.0411     | 55.9428            | 52.7820            |
| 5000 | 4.2566    | 72.1450     | 54.2151     | 71.2518            | 53.5459            |

What I learned:
- The baseline code spends measurable work outside the actual push-relabel logic: Python loops in the runner, unconditional cluster-stat reporting, and repeated allocator/cache management.
- Both solver classes had the same avoidable overhead, so shared cleanup was a good first round.
- The round-1 change helped 2-Level substantially and k-Level modestly at larger size, but it did not materially improve the k-Level small-size case.
- Solver time still dominates total runtime by a wide margin; clustering remains under 1s at the measured target sizes.

Current status:
- Round 1 is the current best committed state.
- Next target is hot-loop tensor overhead inside `solve()`.

### Round 2 (prepared 2026-04-02)
What I changed:
- `src/clustered_push_relabel/solvers/bipartite.py`
  - Stored CSR index tensors used for indexing/scatter in `torch.long` once during build instead of converting them during every iteration.
  - Stored `num_active_centers` explicitly and reused it in the resolve phase.
  - Added reusable per-solver buffers for `center_max_yA` and `center_max_L_A` so the solve loop no longer allocates fresh reduction tensors each iteration.

Results:
- Local validation:
  - `python -m compileall src/clustered_push_relabel/solvers/bipartite.py experiments/runners/e1_mnist_vs_exact.py`
  - Tiny synthetic smoke test passed for both `TwoLevelBipartiteSolver` and `KLevelBipartiteSolver`.
- HPC timings: pending.

What I learned:
- The next safe optimization surface is repeated tensor allocation and dtype conversion inside the push/relabel loop, not clustering or the runner.

Current status:
- Ready for HPC measurement for round 2.

## Active Hypotheses
- Round 2 hypothesis: precomputing long-form CSR indices and reusing center-reduction buffers will reduce allocator pressure and per-iteration index-conversion overhead in both solver variants without changing matching behavior.
- Expected gain: modest runtime reduction, likely more visible in k-Level than round 1 if its hot loop was bottlenecked by repeated small tensor setup costs.
