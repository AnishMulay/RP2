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

Before either solver is timed, the runner computes the exact joint 2D diameter
of `source ∪ target`, divides both point clouds by that diameter, and passes the
normalized point clouds to both methods. RP2 is called with `diameter=1.0`,
which is the diameter of the normalized coordinates. Reported costs are then
multiplied back by the original diameter, so CSV costs remain in the original
HiRef coordinate scale.

Results are written to:

```text
experiments/runners/final2/results/hiref_synthetic_2d_<timestamp>.csv
experiments/runners/final2/results/hiref_synthetic_2d_<timestamp>.md
```

Each CSV row records `n`, method, status, primal OT cost, wall-clock runtime,
peak GPU memory, normalization diameter, RP2 iterations, and the HiRef rank
schedule. The default sweep uses powers of two from `2^5` through `2^20`,
matching the sample sizes in HiRef's saved synthetic cost table. By default, the
RP2 method is run twice for each `N`, with `epsilon=0.01` and `epsilon=0.001`,
and HiRef is run once. After each `N`, the runner prints a compact summary table
with average cost and wall time, appends the CSV rows, and rewrites the markdown
sidecar with the table accumulated so far. It stops when all selected method
rows OOM or a method errors. Use `--keep-going` to continue after failures.

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
python experiments/runners/final2/experiment_hiref_synthetic_2d.py --markdown-output experiments/runners/final2/results/synthetic_progress.md
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
experiments/runners/final2/results/hiref_emnist_<timestamp>.md
```

The EMNIST runner uses the same incremental output behavior as the synthetic
runner: after each completed `N`, it appends the CSV rows and rewrites the
markdown sidecar table through that `N`.

Useful options:

```bash
python experiments/runners/final2/experiment_hiref_emnist.py --max-power 14
python experiments/runners/final2/experiment_hiref_emnist.py --sampling biased
python experiments/runners/final2/experiment_hiref_emnist.py --normalization pixel01
python experiments/runners/final2/experiment_hiref_emnist.py --methods rp2
python experiments/runners/final2/experiment_hiref_emnist.py --markdown-output experiments/runners/final2/results/emnist_progress.md
```

HiRef's own image notebook uses ImageNet rather than EMNIST. It loads ImageNet
training images with `torchvision.datasets.ImageFolder`, resizes them to
`224 x 224`, feeds them through pretrained ResNet-50 with the final classifier
replaced by identity, and runs HiRef on the resulting 2048-dimensional image
embeddings. The notebook splits the embedding matrix into two equal point clouds
`X` and `Y`, uses Euclidean cost (`p = 1`), computes a HiRef rank schedule with
`hierarchy_depth=6`, `max_Q=2^11`, and `max_rank=64`, then reports
`compute_OT_cost()`.

## MNIST Distribution Benchmark

The MNIST runner compares RP2's 2-level solver against HiRef on the sampling
families used by the earlier final2 MNIST proxy experiments, but with Euclidean
L2 cost instead of L1/Manhattan cost:

```bash
python experiments/runners/final2/experiment_hiref_mnist_distributions.py
```

The default MNIST root is:

```text
experiments/runners/final2/data
```

This matches `helpers/download_mnist.py` and the existing final2 MNIST
experiments. If the data is missing, run:

```bash
python experiments/runners/final2/helpers/download_mnist.py
```

The sampling modes are:

- `equal`: source and target are balanced over digits `0-9`, with disjoint
  images within each digit.
- `biased`: target uses digits `0-4`, source uses digits `5-9`.
- `dissimilar`: target uses digits `1,2,4,7`, source uses digits `8,6,9,3`.

Images are flattened to 784-dimensional vectors, scaled to `[0, 1]`, and by
default normalized so each image sums to `1`. The solver inputs are divided by
the analytic L2 diameter before timing, RP2 is called with `diameter=1.0`, and
reported costs are multiplied back to the original pre-normalized image scale.
HiRef is called with `sq_Euclidean=False`, so it uses Euclidean distance, not
squared Euclidean distance.

By default the runner chooses `N` values from `5,000` upward in `5,000`
increments and adds the largest feasible no-replacement balanced sample size for
the selected sampling mode. Results are written incrementally after each
completed `(sampling, N)`:

```text
experiments/runners/final2/results/hiref_mnist_distributions_<timestamp>.csv
experiments/runners/final2/results/hiref_mnist_distributions_<timestamp>.md
```

Useful options:

```bash
python experiments/runners/final2/experiment_hiref_mnist_distributions.py --sampling all
python experiments/runners/final2/experiment_hiref_mnist_distributions.py --sampling biased --max-n 25000
python experiments/runners/final2/experiment_hiref_mnist_distributions.py --n-values 5000,10000,20000
python experiments/runners/final2/experiment_hiref_mnist_distributions.py --rp2-epsilons 0.01,0.001
python experiments/runners/final2/experiment_hiref_mnist_distributions.py --data-dir experiments/runners/final2/data
```
