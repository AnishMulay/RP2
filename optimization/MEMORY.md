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
Round 5 accepted (`27c646b`) as final best state:
- Carried the free-blue set forward incrementally instead of rebuilding it from `MB` each iteration
- Best measured target-size results so far:
  - n=1000: 2-Level `48.23s`, k-Level `50.14s`
  - n=2500: 2-Level `47.86s`, k-Level `46.18s`
  - n=5000: 2-Level `66.04s`, k-Level `46.08s`
- Compared with round 4:
  - 2-Level improved by `7.6%` at n=1000, `4.3%` at n=2500, and regressed by `0.5%` at n=5000
  - k-Level improved by `4.1%` at n=1000, `3.5%` at n=2500, and `8.9%` at n=5000
- Compared with the original baseline where directly comparable:
  - 2-Level improved by `27.6%` at n=1000 and `30.4%` at n=5000
  - k-Level improved by `14.3%` at n=1000 and `17.2%` at n=5000
- Final recommendation:
  - Use round 5 / commit `27c646b` as the default mainline state.
  - If a workload is specifically dominated by 2-Level at n around 5000, round 4 remains slightly faster for that one slice (`65.71s` vs `66.04s`).

Round 4 accepted (`48043e0`) as current mainline:
- Added a single-batch push fast path tailored to the target benchmark regime
- Best measured target-size results so far:
  - n=1000: 2-Level `52.22s`, k-Level `52.29s`
  - n=2500: 2-Level `50.01s`, k-Level `47.87s`
  - n=5000: 2-Level `65.71s`, k-Level `50.56s`
- Compared with round 3:
  - 2-Level improved by `4.1%` at n=1000, `3.5%` at n=2500, `4.7%` at n=5000
  - k-Level improved by `5.1%` at n=1000 and `4.2%` at n=5000, but regressed by `0.9%` at n=2500
- Compared with the original baseline where directly comparable:
  - 2-Level improved by `21.6%` at n=1000 and `30.8%` at n=5000
  - k-Level improved by `10.7%` at n=1000 and `9.1%` at n=5000

Round 3 accepted (`72d9c09`) as current mainline:
- Cached push/resolve work buffers to reduce allocator churn inside the solver loop
- Best measured target-size k-Level results so far:
  - n=1000: `55.07s`
  - n=2500: `47.43s`
  - n=5000: `52.78s`
- Best measured target-size 2-Level results so far:
  - n=1000: `54.45s` from round 3
  - n=2500: `51.82s` from round 3
  - n=5000: `68.38s` from round 2
- Compared with round 2:
  - 2-Level improved by `3.0%` at n=1000 and `2.9%` at n=2500, but regressed by `0.9%` at n=5000
  - k-Level improved by `1.7%` at n=1000, `7.8%` at n=2500, and `2.0%` at n=5000

Round 2 accepted (`6f5be97`):
- Stored CSR/scatter index tensors as `torch.long` once during build
- Reused per-iteration center reduction workspaces
- Best measured target-size results so far:
  - n=1000: 2-Level `56.12s`, k-Level `56.04s`
  - n=2500: 2-Level `53.38s`, k-Level `51.43s`
  - n=5000: 2-Level `68.38s`, k-Level `53.86s`
- Compared with round 1:
  - 2-Level improved by `1.7%` at n=1000, `5.1%` at n=2500, `5.2%` at n=5000
  - k-Level improved by `4.7%` at n=1000, `3.0%` at n=2500, `0.7%` at n=5000
- Compared with the original baseline where directly comparable:
  - 2-Level improved by `15.7%` at n=1000 and `28.0%` at n=5000
  - k-Level improved by `4.2%` at n=1000 and `3.2%` at n=5000

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
- HPC timings on target sizes:

| n    | Exact (s) | 2-Level (s) | k-Level (s) | 2-Level solver (s) | k-Level solver (s) |
|------|-----------|-------------|-------------|--------------------|--------------------|
| 1000 | 0.1148    | 56.1217     | 56.0393     | 55.8275            | 55.9144            |
| 2500 | 0.8450    | 53.3835     | 51.4350     | 53.0107            | 51.1595            |
| 5000 | 4.2527    | 68.3806     | 53.8561     | 67.5453            | 53.1858            |

What I learned:
- The next safe optimization surface is repeated tensor allocation and dtype conversion inside the push/relabel loop, not clustering or the runner.
- This round produced consistent gains across all measured target sizes, especially for 2-Level and for k-Level at smaller n.

Current status:
- Round 2 is the current best committed state.
- Next target is repeated `arange` and zero-buffer allocation inside the push and resolve phases.

### Round 3 (prepared 2026-04-02)
What I changed:
- `src/clustered_push_relabel/solvers/bipartite.py`
  - Added reusable cached long buffers for `torch.arange(...)` workspaces used in the push and resolve phases.
  - Added reusable zeroed long buffers for the resolve offset scans, avoiding fresh `torch.zeros(...)` allocations each iteration.
  - Applied the same allocation caching to both `TwoLevelBipartiteSolver` and `KLevelBipartiteSolver`.

Results:
- Local validation:
  - `python -m compileall src/clustered_push_relabel/solvers/bipartite.py experiments/runners/e1_mnist_vs_exact.py`
  - Tiny synthetic smoke test passed for both `TwoLevelBipartiteSolver` and `KLevelBipartiteSolver`.
- HPC timings on target sizes:

| n    | Exact (s) | 2-Level (s) | k-Level (s) | 2-Level solver (s) | k-Level solver (s) |
|------|-----------|-------------|-------------|--------------------|--------------------|
| 1000 | 0.1151    | 54.4472     | 55.0728     | 54.0043            | 54.1523            |
| 2500 | 0.8411    | 51.8214     | 47.4332     | 51.5005            | 47.1846            |
| 5000 | 4.2379    | 68.9653     | 52.7822     | 68.1167            | 52.1155            |

What I learned:
- After round 2, the remaining obvious hot-loop overhead is temporary long-tensor allocation for segment indexing and rank construction.
- The buffer-caching change helped k-Level consistently and substantially at n=2500, but 2-Level at n=5000 became slightly worse than round 2.

Current status:
- Round 3 is the current mainline best overall because it dominates k-Level across all measured target sizes.
- Round 2 remains the best 2-Level result at n=5000.

### Round 4 (prepared 2026-04-02)
What I changed:
- `src/clustered_push_relabel/solvers/bipartite.py`
  - Added a single-batch push fast path for the benchmark regime where `num_free <= push_batch_size`.
  - The fast path skips list accumulation and `torch.cat(...)` overhead and reuses the already-computed full free-point CSR ranges.
  - Kept the original multi-batch path intact for larger problem sizes.

Results:
- Local validation:
  - `python -m compileall src/clustered_push_relabel/solvers/bipartite.py experiments/runners/e1_mnist_vs_exact.py`
  - Tiny synthetic smoke test passed for both `TwoLevelBipartiteSolver` and `KLevelBipartiteSolver`.
- HPC timings on target sizes:

| n    | Exact (s) | 2-Level (s) | k-Level (s) | 2-Level solver (s) | k-Level solver (s) |
|------|-----------|-------------|-------------|--------------------|--------------------|
| 1000 | 0.1143    | 52.2152     | 52.2875     | 51.9118            | 52.1617            |
| 2500 | 0.8462    | 50.0071     | 47.8658     | 49.6810            | 47.6214            |
| 5000 | 4.2592    | 65.7105     | 50.5640     | 64.8738            | 49.9104            |

What I learned:
- For the target sizes in this experiment, the push phase is effectively always single-batch, so the general batching/list path is likely paying overhead without benefit.
- This round delivered the best overall wall-clock results so far despite a small k-Level regression at n=2500.

Current status:
- Round 4 is the current best committed state.
- Next target is the repeated reconstruction of the free-blue set via `nonzero(self.MB == -1)` each iteration.

### Round 5 (prepared 2026-04-02)
What I changed:
- `src/clustered_push_relabel/solvers/bipartite.py`
  - Replaced repeated full-array reconstruction of `B_free` with an incremental update from the monotone free-blue set.
  - Added a reusable boolean keep-mask buffer to filter out newly matched blue vertices without rescanning `MB`.
  - Applied the same free-set carry-forward logic to both solver variants.

Results:
- Local validation:
  - `python -m compileall src/clustered_push_relabel/solvers/bipartite.py experiments/runners/e1_mnist_vs_exact.py`
  - Tiny synthetic smoke test passed for both `TwoLevelBipartiteSolver` and `KLevelBipartiteSolver`.
- HPC timings on target sizes:

| n    | Exact (s) | 2-Level (s) | k-Level (s) | 2-Level solver (s) | k-Level solver (s) |
|------|-----------|-------------|-------------|--------------------|--------------------|
| 1000 | 0.1149    | 48.2260     | 50.1448     | 47.8066            | 50.0216            |
| 2500 | 0.8457    | 47.8582     | 46.1776     | 47.5227            | 45.9263            |
| 5000 | 4.2575    | 66.0440     | 46.0758     | 65.1596            | 45.4324            |

What I learned:
- After round 4, one of the remaining obvious non-algorithmic costs is rebuilding the free-blue index set from scratch every iteration even though matches are monotone.
- That cost mattered: round 5 produced the best k-Level times at all measured target sizes and the best 2-Level times at n=1000 and n=2500.
- The only measured regression relative to round 4 is 2-Level at n=5000, and it is small (`66.04s` vs `65.71s`).

Current status:
- Optimization work concluded at the user's request after round 5 benchmarking.
- Final best overall commit is `27c646b`.

## Active Hypotheses
- Round 6 hypothesis: the bottleneck is inside a single phase of the push-relabel loop.
  Sub-phases A (scatter_reduce maintenance), B (ragged index + slack computation),
  C (argsort/bincount/rank resolve), and D (relabel + free-set update) have never been
  individually timed.  We do not yet know which sub-phase dominates.

### Round 6 (prepared 2026-04-03) — PROFILING ROUND, no algorithmic changes
What changed:
- `src/clustered_push_relabel/solvers/bipartite.py`
  - Added module-level `PROFILING = False` flag.
  - Instrumented both `TwoLevelBipartiteSolver.solve()` and
    `KLevelBipartiteSolver.solve()` with `time.perf_counter()` checkpoints behind
    `if PROFILING:` guards.  Zero overhead when flag is False.
  - Sub-phases timed per iteration:
      A  – scatter_reduce to build center_max_yA / center_max_L_A
      B1 – ragged index construction (repeat_interleave / cumsum)
      B2 – CSR gather (blue_center_indices / blue_levels lookup)
      B3 – slack arithmetic and candidate filter
      C1 – free-red mask filter (MA == -1)
      C2 – argsort / bincount / cumsum / rank computation
      C3 – final slack==0 check + MB/MA write
      D  – B_free update, yB increment, yA decrement
  - At end of solve(), profiling dict stored as `solver._prof` for retrieval.
- `experiments/runners/e1_mnist_profile.py` (new file)
  - Profiling variant of e1_mnist_vs_exact.py.
  - Sets `_bip_module.PROFILING = True` before running solvers.
  - Runs only n=1000 and n=5000 (reduced to limit profiling overhead).
  - Prints a clean table: operation | total(s) | % solver time | avg per iter (ms).
  - Still writes the standard CSV and prints correctness output.
  - Run with:
      python -u experiments/runners/e1_mnist_profile.py --epsilon 0.01 --k 4 --trials 1 --seed 42

Results: pending HPC run.

Next action: paste the profiling table output back to determine which sub-phase to target in round 7.
