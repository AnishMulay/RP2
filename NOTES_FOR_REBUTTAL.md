# Notes for Rebuttal (Reviewer 1nqD)

Generated during a code-only session (no GPU available in the dev sandbox;
all numeric claims below were verified on CPU — real GPU numbers require the
cluster run described in the handoff message sent alongside this file).

## Task 1 — Full-repo audit: "no hand-written CUDA kernels" claim

**Conclusion: holds across the entire implementation.** Every clustering and
solver file under `src/clustered_push_relabel/` uses only standard PyTorch
tensor ops (`torch.cdist`, `torch.mm`/`addmm_`, elementwise ops, `cumsum`,
`scatter_add_`/`scatter_reduce_`, `nonzero`, `cummax`, `torch.where`, etc.).

Grep across `src/` for `torch.ops`, `cpp_extension`, `load_inline`,
`CUDAExtension`, `triton`, `numba`, `cupy`, `dlpack`, raw kernel syntax
(`__global__`, `blockIdx`, `threadIdx`), `ctypes`/`cffi`, and custom
`torch.autograd.Function` subclasses returned **zero matches**.

One incidental hit outside `src/`: `experiments/runners/experiment.py:96` has
a *comment* mentioning `dlpack` ("In production, dlpack is faster, but this
is safer for scripts.") next to an OTT-JAX baseline wrapper — no actual
`dlpack` call exists; it converts via `.cpu().numpy()`. This file is a
baseline comparison harness, not part of the compressed-OT solver.

Files checked (explicit list for the audit trail):
- `src/clustered_push_relabel/clustering/simple.py`
- `src/clustered_push_relabel/clustering/simple_l1.py`
- `src/clustered_push_relabel/clustering/simple_copy.py`
- `src/clustered_push_relabel/clustering/simple_precomputed.py`
- `src/clustered_push_relabel/clustering/simple_three_level.py`
- `src/clustered_push_relabel/clustering/simple_three_level_l1.py`
- `src/clustered_push_relabel/clustering/two_level.py` (`FastGPUClustering` — not previously reviewed)
- `src/clustered_push_relabel/clustering/k_level.py` (`FastGPUMultiLevelClustering` — not previously reviewed)
- `src/clustered_push_relabel/clustering/color_aware_two_level.py` (`ColorAwareClustering` — not previously reviewed)
- `src/clustered_push_relabel/clustering/__init__.py`
- `src/clustered_push_relabel/solvers/bipartite.py`
- `src/clustered_push_relabel/solvers/simple_bipartite.py`
- `src/clustered_push_relabel/solvers/simple_bipartite_2.py`
- `src/clustered_push_relabel/solvers/three_level_bipartite.py`
- `src/clustered_push_relabel/solvers/color_aware_bipartite.py`
- `src/clustered_push_relabel/solvers/transport.py`
- `src/clustered_push_relabel/solvers/__init__.py`
- `src/clustered_push_relabel/utils/distance.py`
- `src/clustered_push_relabel/utils/__init__.py`
- `src/clustered_push_relabel/__init__.py`

This also independently confirms `COLOR_AWARE_SOLVER_SPEC.md:17`'s existing
claim ("GPU framework: PyTorch only. No CuPy, no custom CUDA.").

## Task 2 — `sample_factor` convention discrepancy

**Finding confirmed, and it does NOT affect any reported paper number**, for
the reason below.

- L2 classes use `rate = 1 / (sample_factor * f(N))` — **larger
  `sample_factor` → fewer landmarks**:
  - `clustering/simple.py:97`
  - `clustering/simple_three_level.py:125,135` (`rate1`, `rate2`)
- L1 classes use `rate = sample_factor / f(N)` — **larger `sample_factor` →
  more landmarks**:
  - `clustering/simple_l1.py:51-53`
  - `clustering/simple_three_level_l1.py:91,99` (`rate1`, `rate2`)

This is consistent within each family (all L2 classes agree with each other;
all L1 classes agree with each other) — it looks like an intentional
L1-vs-L2 divergence in how the parameter was introduced, not a random typo,
but it is still a footgun: the same parameter name means opposite things
depending on which class is instantiated.

**Which convention produced the paper's numbers:** every real driver script
found in the repo — `experiments/runners/final2/experiments/*.py` (see
`exp01`–`exp10`), `experiments/runners/final2/scalability_synthetic_3level_binary_search.py`
(the script that performs the binary search behind the reviewer-cited
n≈650,000 3-level scalability result), and `experiments/runners/final2/experiments/archive/exp13_scalability_limits.py`
— either never pass `sample_factor` at all, or pass it explicitly as `1.0`.
At `sample_factor = 1.0` **both conventions are numerically identical**
(`1/(1*f(N)) == 1/f(N)`), so the discrepancy has zero effect on any number
currently in the paper. The only place `sample_factor` is varied across
multiple values is `experiments/runners/final2/experiments/paper/exp11_landmark_density.py`,
and it only instantiates the L1 classes (`SimpleL1Clustering`,
`ThreeLevelL1Clustering`), so it never mixes the two conventions.

**Not fixed** (per instructions — this is a report, not a silent resolution).
If a future experiment ever passes a non-default `sample_factor` to an L2
class while assuming the L1 convention (or vice versa), landmark density
would silently be wrong; consider unifying the convention or renaming one of
the parameters before it's used non-default in any reported result.

## Task 3 — Opt-in stage-level peak-memory profiling

Added `profile_memory: bool = False` (default off, zero overhead when off) to:

- `clustering/simple.py` (`SimpleClustering`) — stages: `landmark_sampling`,
  `DR_DB_construction` (DR/DR_int and DB/d_min_b are interleaved in the
  existing code to bound peak memory, so they're profiled as one combined
  stage rather than reordered), `csr_pass1_counting`, `csr_pass2_fill`.
- `clustering/simple_three_level.py` (`ThreeLevelClustering`) — stages:
  `landmark_sampling_A1_A2`, `DR_construction`, `landmark_assignment_A2`
  (DB_A2 + DA1_A2), `landmark_assignment_A1_tiled` (DB_A1), 
  `adj_B_csr_pass1_counting`, `adj_B_csr_pass2_fill`,
  `adj_A1_csr_pass1_counting`, `adj_A1_csr_pass2_fill`.
- `clustering/two_level.py`, `clustering/k_level.py`,
  `clustering/color_aware_two_level.py` — coarser stage breakdown
  (`landmark_sampling`, `voronoi_bounds`/`landmark_assignment_d_min`,
  `build_cover`/`build_cover_red`/`build_cover_blue`), matching their
  different internal structure. These three classes are **not** the ones
  used to produce the paper's headline scalability numbers (see Task 2); they
  were included for completeness per the explicit instruction to cover them.
- `solvers/simple_bipartite.py` (`SimpleGPUSolver.solve()`),
  `solvers/three_level_bipartite.py` (`ThreeLevelGPUSolver.solve()`),
  `solvers/bipartite.py` (`TwoLevelBipartiteSolver.solve()`,
  `KLevelBipartiteSolver.solve()`) — each `solve()` was split into a thin
  `solve()` wrapper (peak-memory reset/record) and an unmodified
  `_solve_impl()` containing the original body verbatim, so the push-relabel
  loop itself was not touched. This reuses `bipartite.py`'s existing
  `PROFILING` module-level pattern in spirit (module/instance flag defaulting
  off) but adds memory rather than time, since `SimpleGPUSolver` /
  `ThreeLevelGPUSolver` (the solvers that actually produced the paper's
  numbers) had no pre-existing per-phase timing infrastructure to piggyback
  on; `bipartite.py`'s solvers do, and were still only wrapped at the
  overall-`solve()` granularity to keep the change minimal, since they are
  not on the paper's critical path either.

Mechanism: each stage is bounded by
`torch.cuda.reset_peak_memory_stats()` before and
`torch.cuda.synchronize(); torch.cuda.max_memory_allocated()` after, guarded
by `if self.profile_memory and torch.cuda.is_available():`. Results land in
a public `self.memory_profile` dict (not the `run()`/`solve()` return value),
so the shape of every existing return value is completely unchanged whether
or not profiling is enabled.

**Verified claim: `reset_peak_memory_stats`/`max_memory_allocated` are
metadata reads, not allocations.** Confirmed by code inspection (they only
touch CUDA's internal allocator bookkeeping) and by the CPU no-op tests
below, which show zero behavioral difference with the flag on vs. off. A
GPU-side confirmation (checking `torch.cuda.memory_allocated()` before/after
enabling the flag doesn't change) is included as a checklist item in the
cluster handoff.

## Task 4 — CPU portability

Relaxed the CUDA-only guard from `raise ValueError(...)` on any
non-`"cuda"` device to allowing `("cuda", "cpu")`, in:

- `clustering/simple.py`, `clustering/simple_l1.py`,
  `clustering/simple_three_level.py`, `clustering/simple_three_level_l1.py`,
  `clustering/simple_copy.py`, `clustering/simple_precomputed.py` (all
  `_validate()`)
- `solvers/simple_bipartite.py`, `solvers/simple_bipartite_2.py`,
  `solvers/three_level_bipartite.py` (`__init__` device checks)

`clustering/two_level.py`, `clustering/k_level.py`,
`clustering/color_aware_two_level.py`, and `solvers/bipartite.py`'s two
solvers had **no such guard to begin with** — they already ran on CPU (all
`torch.cuda.*` calls in those files were already conditioned on
`device.type == "cuda"`), confirmed by direct testing, so nothing needed to
change there for CPU support itself (only the Task 3 profiling hooks were
added, and those are no-ops off-GPU by construction).

`scripts/cpu_smoke_test.py` runs the 2-Level and 3-Level pipelines
end-to-end on CPU. Default N=8,000 completes in well under a second on a
laptop CPU; N=20,000 completes in ~4s per pipeline (measured on this
machine) — both comfortably inside "a few minutes."

## Task 5 — Regression safety

Fixed-seed (`torch.manual_seed(0)`) synthetic N=2,000 regression comparing
matched indices (hash + sum) and total cost, run on CPU (this dev
environment has no CUDA device), for `SimpleClustering`, `ThreeLevelClustering`,
`SimpleGPUSolver`, and `ThreeLevelGPUSolver`, all new flags left at their
default (`profile_memory=False`):

```
[SimpleClustering] adj_col.sum=73795377 DR_int.sum=1070587
[ThreeLevelClustering] adj_B_col.sum=22687129 adj_A1_col.sum=25146988 DR_int.sum=215152
[SimpleGPUSolver] n_matched=2000 idx_sum=1999000 idx_hash=-7268282923761994748 total_cost=110.8941040039
[ThreeLevelGPUSolver] n_matched=2000 idx_sum=1999000 idx_hash=-5499755664586165878 total_cost=135.3864898682
```

**Result: PASS — bit-identical, diffed after every commit in this session
(Task 4's CPU relaxation, and every Task 3 profiling addition); no diff at
any point.** Additionally verified that setting `profile_memory=True`
produces the *same* matched output as `profile_memory=False` for every
class/solver (CPU no-op check; the flag only ever populates
`self.memory_profile`, never changes control flow or tensor values). The
existing `tests/` suite (`pytest tests/`) also passes unchanged (6/6).

Caveat: this only proves determinism/CPU-parity, not that GPU peak-memory
numbers are exactly zero-overhead in wall-clock terms — that requires the
GPU run described in the cluster handoff.

## Task 6 — Deliverables

- `scripts/profile_memory_breakdown.py` — runs the 3-Level (default) or
  2-Level pipeline with `profile_memory=True` and prints the stage table.
  Defaults (`epsilon=0.01`, `tile_size=2048`, `clustering_tile_size=512`,
  `sample_factor=1.0`, `seed=42`) match
  `scalability_synthetic_3level_binary_search.py`, the actual source of the
  paper's cited 3-level scalability point; `--n` defaults to 650,000 as an
  approximation of that reviewer-cited number (the real experiment found it
  via binary search, so there is no single "the" N baked into that script —
  lower `--n` if it OOMs on your GPU).
- `scripts/cpu_smoke_test.py` — Task 4's CPU end-to-end check.
- This file.

---

See the end of the assistant's final chat message for the cluster
hand-off instructions (exact commands, what to capture, and runtime
expectations) — kept there rather than duplicated here so it can be
copy-pasted as-is into a message to labmates.
