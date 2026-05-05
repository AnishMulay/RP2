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
HiRef's saved synthetic cost table. By default, the RP2 method is run twice for
each `N`, with `epsilon=0.01` and `epsilon=0.001`, and HiRef is run once. After
each `N`, the runner prints a compact summary table with average cost and wall
time. It stops when all selected method rows OOM or a method errors. Use
`--keep-going` to continue after failures.

HiRef's notebook stores primal costs through `compute_OT_cost()` but does not
persist a runtime table for this synthetic sweep. This runner records wall-clock
time around each solver's full setup, solve, and cost evaluation path.

Useful options:

```bash
python experiments/runners/final2/experiment_hiref_synthetic_2d.py --max-power 17
python experiments/runners/final2/experiment_hiref_synthetic_2d.py --methods rp2
python experiments/runners/final2/experiment_hiref_synthetic_2d.py --methods hiref
python experiments/runners/final2/experiment_hiref_synthetic_2d.py --rp2-epsilons 0.01,0.001 --batch-size 2048
python experiments/runners/final2/experiment_hiref_synthetic_2d.py --rp2-epsilon 0.001
```

Runtime dependencies are the union of RP2 and HiRef's synthetic notebook stack:
`torch`, `numpy`, `scikit-learn`, `jax`, `scipy`, and `matplotlib`. HiRef source
files needed at runtime are copied into `final2/hiref_src/`, so the runner does
not import from the sibling `HiRef/` repository.

## EMNIST Image Benchmark

There is also a high-dimensional image benchmark:

```bash
python experiments/runners/final2/experiment_hiref_emnist.py
```

This compares the same two methods on EMNIST `byclass` images. Each image is
loaded as a `28 x 28` grayscale image, transposed to EMNIST's natural
orientation, flattened to a 784-dimensional vector, scaled to `[0, 1]`, and by
default normalized so its pixel sum is `1`. With the default equal sampling,
both source and target contain exactly `N` images with balanced class counts
over the 62 EMNIST `byclass` classes; source and target images are disjoint.

The reported cost is the average Euclidean matching cost:

```text
(1 / N) * sum_j ||target_j - source_match(j)||_2
```

Results are written to:

```text
experiments/runners/final2/results/hiref_emnist_<timestamp>.csv
```

Useful options:

```bash
python experiments/runners/final2/experiment_hiref_emnist.py --max-power 14
python experiments/runners/final2/experiment_hiref_emnist.py --sampling biased
python experiments/runners/final2/experiment_hiref_emnist.py --normalization pixel01
python experiments/runners/final2/experiment_hiref_emnist.py --methods rp2
```

HiRef's own image notebook uses ImageNet rather than EMNIST. It loads ImageNet
training images with `torchvision.datasets.ImageFolder`, resizes them to
`224 x 224`, feeds them through pretrained ResNet-50 with the final classifier
replaced by identity, and runs HiRef on the resulting 2048-dimensional image
embeddings. The notebook splits the embedding matrix into two equal point clouds
`X` and `Y`, uses Euclidean cost (`p = 1`), computes a HiRef rank schedule with
`hierarchy_depth=6`, `max_Q=2^11`, and `max_rank=64`, then reports
`compute_OT_cost()`.
