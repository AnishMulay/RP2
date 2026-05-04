# Final2 HiRef Synthetic 2D Benchmark

This directory includes a standalone benchmark for the HiRef paper's synthetic
Half-Moon and S-Curve 2D experiment:

```bash
python experiments/runners/final2/experiment_hiref_synthetic_2d.py
```

The runner compares:

- `RP2_2level_gpu_push_relabel`: RP2's 2-level CUDA push-relabel solver.
- `HiRef_HROT_LR`: a local copy of HiRef's low-rank hierarchical refinement solver.

The experiment generates identical source and target point sets for both
methods using the Half-Moon/S-Curve code from HiRef's
`synthetic_experiments_sample_complexity_bench_GPU.ipynb`, with the notebook's
default JAX random seed `0`. The cost function is Euclidean distance
`||x - y||_2`, matching the notebook's `p = 1` / Table S2 setup.

Results are written to:

```text
experiments/runners/final2/results/hiref_synthetic_2d_<timestamp>.csv
```

Each CSV row records `n`, method, status, primal OT cost, wall-clock runtime,
peak GPU memory, RP2 iterations, and the HiRef rank schedule. The default sweep
uses powers of two from `2^5` through `2^20`, matching the sample sizes in
HiRef's saved synthetic cost table, and stops when both selected methods OOM or
a method errors. Use `--keep-going` to continue after failures.

HiRef's notebook stores primal costs through `compute_OT_cost()` but does not
persist a runtime table for this synthetic sweep. This runner records wall-clock
time around each solver's full setup, solve, and cost evaluation path.

Useful options:

```bash
python experiments/runners/final2/experiment_hiref_synthetic_2d.py --max-power 17
python experiments/runners/final2/experiment_hiref_synthetic_2d.py --methods rp2
python experiments/runners/final2/experiment_hiref_synthetic_2d.py --methods hiref
python experiments/runners/final2/experiment_hiref_synthetic_2d.py --rp2-epsilon 0.01 --batch-size 2048
```

Runtime dependencies are the union of RP2 and HiRef's synthetic notebook stack:
`torch`, `numpy`, `scikit-learn`, `jax`, `scipy`, and `matplotlib`. HiRef source
files needed at runtime are copied into `final2/hiref_src/`, so the runner does
not import from the sibling `HiRef/` repository.
