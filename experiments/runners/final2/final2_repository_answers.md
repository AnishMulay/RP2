# Final2 Repository Answers

Generated from local repository state at `/Users/anish/Developer/NCSU/RP2` on `2026-05-02T10:33:04`. All file paths are relative to the repository root unless an absolute path is explicitly shown.

## Section 1 — Directory Structure

Full tree of `experiments/runners/final2/`; Python files include line counts.

```text
experiments/runners/final2/
├── experiments/
│   ├── __init__.py (0 lines)
│   ├── exp01_mnist_proxy_equal.py (259 lines)
│   ├── exp02_mnist_proxy_biased.py (234 lines)
│   ├── exp03_emnist_proxy_equal.py (229 lines)
│   ├── exp04_emnist_proxy_biased.py (238 lines)
│   ├── exp05_nyc_scalability.py (296 lines)
│   ├── exp06_cifar_sift_proxy.py (329 lines)
│   ├── exp07_cifar_sift_scalability.py (327 lines)
│   ├── exp08_newsgroups_proxy.py (336 lines)
│   ├── exp09_newsgroups_scalability.py (343 lines)
│   └── exp10_mnist_proxy_dissimilar.py (347 lines)
├── final2_repository_answers.md
├── helpers/
│   └── download_mnist.py (40 lines)
├── results/
├── run_experiments.py (254 lines)
└── shared.py (391 lines)
```

## Section 2 — All Experiment Files

### `experiments/runners/final2/experiments/__init__.py`

This file is empty and defines no experiment constants, no `run()` function, no data-loading functions, no metrics, and no imports from `shared.py`.

### `experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py`

- Full file path: `experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py`
- `EXP_ID` = `1`
- `EXP_NAME` = `MNIST — Exact vs 2L-Proxy vs 3L-Proxy (Equal Sampling)`
- `DATASET` = `MNIST`
- `DATA_DIR` = `FINAL2_DIR / 'data'`
- `N_VALUES` = `[5000, 10000, 15000, 20000, 25000]`
- `EPSILON` = `0.01`
- `BATCH_SIZE` = `512`
- `SEED` = `42`
- `EXACT_N_LIMIT` = `25000`
- Exact `run()` signature: `def run(device, **kwargs):`
- Exact data-loading / sampling function signatures:
  - `def load_mnist_equal(n_samples, seed):`
- Imports from `shared.py`: `compute_ratio, fmt_time, fmt_cost, fmt_ratio, build_two_level_proxy_matrix, build_three_level_proxy_matrix, run_three_level_precomputed`
- Metrics columns from `COL_SPECS`: `[('N', 7), ('Exact Time', 14), ('2L-Prx Time', 14), ('3L-Prx Time', 14), ('Exact Cost', 12), ('2L-Prx Cost', 12), ('3L-Prx Cost', 12), ('2L Ratio', 9), ('3L Ratio', 9)]`
- Result keys: Each row is `{"n": n, "exact": exact, "prx2": prx2, "prx3": prx3}`. Each method result from `_safe` has keys `time_ms`, `cost`, and `status`.
- Dataset loading and DATA_DIR resolution: `torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=False)` and `torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=False)`. It concatenates `train.data`, `test.data`, `train.targets`, and `test.targets`, reshapes images to `(-1, 784)`, converts to `float32 / 255.0`, and normalizes each image row by its own pixel sum.
- `FINAL2_DIR` resolves to `/Users/anish/Developer/NCSU/RP2/experiments/runners/final2`; `DATA_DIR = FINAL2_DIR / "data"` resolves to `/Users/anish/Developer/NCSU/RP2/experiments/runners/final2/data`.
- Sampling/trials: Equal sampling from all 10 MNIST digit classes for both red and blue. For each class, it takes `spc = n_samples // len(classes)` images for red and `spc` different images for blue after shuffling indices with `rng_r = np.random.RandomState(seed)` and `rng_b = np.random.RandomState(seed + 1)`. Because `n_samples` is divisible by 10 for all configured `N_VALUES`, each returned side has exactly `n_samples` rows. Single trial per `n`; fixed `SEED = 42` and no loop over seeds.
- Computes ratio/distortion between exact OT cost and proxy/solver cost: YES.
  - Exact variable/key names used for ratios are the result dictionaries/keys passed to `compute_ratio`: `r["exact"]["cost"]`, `r["prx2"]["cost"]`, `r["prx3"]["cost"]`, or for solver files `r["sol2"]["cost"]`, `r["sol3"]["cost"]`; in row-formatting functions the local variables are `e`, `p2`, `p3`, `s2`, `s3`, `r2`, and `r3` where present.
- Tracks or records peak GPU memory anywhere: NO. It may call `torch.cuda.empty_cache()` or synchronize CUDA, but no `torch.cuda.max_memory_allocated()` or reset peak call appears in this file.
Exact implementation of `load_mnist_equal`:
```python
def load_mnist_equal(n_samples, seed):
    train = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=False)
    test  = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy().reshape(-1, 784)
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()

    classes = np.unique(labels)
    spc = n_samples // len(classes)
    if spc == 0:
        raise ValueError(f"n_samples={n_samples} too small for {len(classes)} classes")

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_parts, blue_parts = [], []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        needed = 2 * spc
        if idx.size < needed:
            warnings.warn(f"Class {cls}: only {idx.size} samples, need {needed}. Skipping.")
            continue
        rng_r.shuffle(idx)
        chosen = idx[:needed]
        red_parts.append(images[chosen[:spc]])
        blue_parts.append(images[chosen[spc:needed]])

    red = np.concatenate(red_parts).astype(np.float32) / 255.0
    blue = np.concatenate(blue_parts).astype(np.float32) / 255.0
    for arr in (red, blue):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each MNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s
    return torch.from_numpy(red), torch.from_numpy(blue)
```


### `experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py`

- Full file path: `experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py`
- `EXP_ID` = `2`
- `EXP_NAME` = `MNIST — Exact vs 2L-Proxy vs 3L-Proxy (Biased: B=0–4, A=5–9)`
- `DATASET` = `MNIST`
- `DATA_DIR` = `FINAL2_DIR / 'data'`
- `N_VALUES` = `[5000, 10000, 15000, 20000, 25000]`
- `EPSILON` = `0.01`
- `BATCH_SIZE` = `512`
- `SEED` = `42`
- `EXACT_N_LIMIT` = `25000`
- Additional exact constants:
- `BLUE_DIGITS` = `list(range(5))`
- `RED_DIGITS` = `list(range(5, 10))`
- Exact `run()` signature: `def run(device, **kwargs):`
- Exact data-loading / sampling function signatures:
  - `def _sample_from_digits(images, labels, digit_set, n_total, rng):`
  - `def load_mnist_biased(n_samples, seed):`
- Imports from `shared.py`: `compute_ratio, fmt_time, fmt_cost, fmt_ratio, build_two_level_proxy_matrix, build_three_level_proxy_matrix`
- Metrics columns from `COL_SPECS`: `[('N', 7), ('Exact Time', 14), ('2L-Prx Time', 14), ('3L-Prx Time', 14), ('Exact Cost', 12), ('2L-Prx Cost', 12), ('3L-Prx Cost', 12), ('2L Ratio', 9), ('3L Ratio', 9)]`
- Result keys: Each row is `{"n": n, "exact": exact, "prx2": prx2, "prx3": prx3}`. Each method result from `_safe` has keys `time_ms`, `cost`, and `status`.
- Dataset loading and DATA_DIR resolution: Same MNIST train/test calls as exp01 with `root=str(DATA_DIR)` and `download=False`; data is reshaped to `(-1, 784)`, converted to `float32 / 255.0`, and row-normalized.
- `FINAL2_DIR` resolves to `/Users/anish/Developer/NCSU/RP2/experiments/runners/final2`; `DATA_DIR = FINAL2_DIR / "data"` resolves to `/Users/anish/Developer/NCSU/RP2/experiments/runners/final2/data`.
- Sampling/trials: Biased MNIST sampling. `BLUE_DIGITS = list(range(5))` (digits 0,1,2,3,4) and `RED_DIGITS = list(range(5, 10))` (digits 5,6,7,8,9). `_sample_from_digits` samples `spc = n_total // len(classes)` per selected digit after shuffling. Single trial per `n`; fixed `SEED = 42` and no loop over seeds.
- Computes ratio/distortion between exact OT cost and proxy/solver cost: YES.
  - Exact variable/key names used for ratios are the result dictionaries/keys passed to `compute_ratio`: `r["exact"]["cost"]`, `r["prx2"]["cost"]`, `r["prx3"]["cost"]`, or for solver files `r["sol2"]["cost"]`, `r["sol3"]["cost"]`; in row-formatting functions the local variables are `e`, `p2`, `p3`, `s2`, `s3`, `r2`, and `r3` where present.
- Tracks or records peak GPU memory anywhere: NO. It may call `torch.cuda.empty_cache()` or synchronize CUDA, but no `torch.cuda.max_memory_allocated()` or reset peak call appears in this file.
Exact implementation of `_sample_from_digits`:
```python
def _sample_from_digits(images, labels, digit_set, n_total, rng):
    """Sample n_total images equally from the specified digits."""
    classes = sorted(digit_set)
    spc = n_total // len(classes)
    if spc == 0:
        raise ValueError(f"n_total={n_total} too small for {len(classes)} classes")
    parts = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        if idx.size < spc:
            warnings.warn(f"Digit {cls}: only {idx.size} available, need {spc}. Skipping.")
            continue
        rng.shuffle(idx)
        parts.append(images[idx[:spc]])
    return np.concatenate(parts)
```

Exact implementation of `load_mnist_biased`:
```python
def load_mnist_biased(n_samples, seed):
    train = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=False)
    test  = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy().reshape(-1, 784)
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_arr  = _sample_from_digits(images, labels, RED_DIGITS,  n_samples, rng_r).astype(np.float32) / 255.0
    blue_arr = _sample_from_digits(images, labels, BLUE_DIGITS, n_samples, rng_b).astype(np.float32) / 255.0

    for arr in (red_arr, blue_arr):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each MNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s

    return torch.from_numpy(red_arr), torch.from_numpy(blue_arr)
```


### `experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py`

- Full file path: `experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py`
- `EXP_ID` = `3`
- `EXP_NAME` = `EMNIST — Exact vs 2L-Proxy vs 3L-Proxy (Equal Sampling)`
- `DATASET` = `EMNIST`
- `DATA_DIR` = `BASE_DIR / 'data'`
- `N_VALUES` = `[5000, 10000, 15000, 20000, 25000]`
- `EPSILON` = `0.01`
- `BATCH_SIZE` = `512`
- `SEED` = `42`
- `EXACT_N_LIMIT` = `25000`
- Additional exact constants:
- `EMNIST_SPLIT` = `byclass`
- Exact `run()` signature: `def run(device, **kwargs):`
- Exact data-loading / sampling function signatures:
  - `def load_emnist_equal(n_samples, seed, split = EMNIST_SPLIT):`
- Imports from `shared.py`: `compute_ratio, fmt_time, fmt_cost, fmt_ratio, build_two_level_proxy_matrix, build_three_level_proxy_matrix`
- Metrics columns from `COL_SPECS`: `[('N', 7), ('Exact Time', 14), ('2L-Prx Time', 14), ('3L-Prx Time', 14), ('Exact Cost', 12), ('2L-Prx Cost', 12), ('3L-Prx Cost', 12), ('2L Ratio', 9), ('3L Ratio', 9)]`
- Result keys: Each row is `{"n": n, "exact": exact, "prx2": prx2, "prx3": prx3}`. Each method result from `_safe` has keys `time_ms`, `cost`, and `status`.
- Dataset loading and DATA_DIR resolution: `torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=True, download=False)` and the same with `train=False`. It concatenates data/targets, reshapes to `(-1, 28, 28)`, transposes axes with `.transpose(0, 2, 1)`, reshapes to `(-1, 784)`, converts to `float32 / 255.0`, and row-normalizes.
- `BASE_DIR` resolves to `/Users/anish/Developer/NCSU/RP2`; `DATA_DIR = BASE_DIR / "data"` resolves to `/Users/anish/Developer/NCSU/RP2/data`.
- Sampling/trials: Equal sampling from all classes present in EMNIST `byclass`. It computes `classes = np.unique(labels)` and `spc = n_samples // len(classes)`. With 62 classes, configured `N_VALUES` are not divisible by 62, so each returned side has `62 * floor(n_samples / 62)` rows, not necessarily exactly `n_samples`. Single trial per `n`; fixed `SEED = 42` and no loop over seeds.
- Computes ratio/distortion between exact OT cost and proxy/solver cost: YES.
  - Exact variable/key names used for ratios are the result dictionaries/keys passed to `compute_ratio`: `r["exact"]["cost"]`, `r["prx2"]["cost"]`, `r["prx3"]["cost"]`, or for solver files `r["sol2"]["cost"]`, `r["sol3"]["cost"]`; in row-formatting functions the local variables are `e`, `p2`, `p3`, `s2`, `s3`, `r2`, and `r3` where present.
- Tracks or records peak GPU memory anywhere: NO. It may call `torch.cuda.empty_cache()` or synchronize CUDA, but no `torch.cuda.max_memory_allocated()` or reset peak call appears in this file.
Exact implementation of `load_emnist_equal`:
```python
def load_emnist_equal(n_samples, seed, split=EMNIST_SPLIT):
    train = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=True,  download=False)
    test  = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    classes = np.unique(labels)
    spc = n_samples // len(classes)
    if spc == 0:
        raise ValueError(f"n_samples={n_samples} too small for {len(classes)} EMNIST classes")

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_parts, blue_parts = [], []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        needed = 2 * spc
        if idx.size < needed:
            warnings.warn(f"Class {cls}: {idx.size} available, need {needed}. Skipping.")
            continue
        rng_r.shuffle(idx)
        chosen = idx[:needed]
        red_parts.append(images[chosen[:spc]])
        blue_parts.append(images[chosen[spc:needed]])

    red  = np.concatenate(red_parts).astype(np.float32)  / 255.0
    blue = np.concatenate(blue_parts).astype(np.float32) / 255.0
    for arr in (red, blue):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each EMNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s

    return torch.from_numpy(red), torch.from_numpy(blue)
```


### `experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py`

- Full file path: `experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py`
- `EXP_ID` = `4`
- `EXP_NAME` = `EMNIST — Exact vs 2L-Proxy vs 3L-Proxy (Biased: B=cls 0–30, A=cls 31–61)`
- `DATASET` = `EMNIST`
- `DATA_DIR` = `BASE_DIR / 'data'`
- `N_VALUES` = `[5000, 10000, 15000, 20000, 25000]`
- `EPSILON` = `0.01`
- `BATCH_SIZE` = `512`
- `SEED` = `42`
- `EXACT_N_LIMIT` = `25000`
- Additional exact constants:
- `EMNIST_SPLIT` = `byclass`
- `BLUE_CLASS_END` = `31`
- `RED_CLASS_START` = `31`
- Exact `run()` signature: `def run(device, **kwargs):`
- Exact data-loading / sampling function signatures:
  - `def _sample_classes(images, labels, class_indices, n_total, rng):`
  - `def load_emnist_biased(n_samples, seed, split = EMNIST_SPLIT):`
- Imports from `shared.py`: `compute_ratio, fmt_time, fmt_cost, fmt_ratio, build_two_level_proxy_matrix, build_three_level_proxy_matrix`
- Metrics columns from `COL_SPECS`: `[('N', 7), ('Exact Time', 14), ('2L-Prx Time', 14), ('3L-Prx Time', 14), ('Exact Cost', 12), ('2L-Prx Cost', 12), ('3L-Prx Cost', 12), ('2L Ratio', 9), ('3L Ratio', 9)]`
- Result keys: Each row is `{"n": n, "exact": exact, "prx2": prx2, "prx3": prx3}`. Each method result from `_safe` has keys `time_ms`, `cost`, and `status`.
- Dataset loading and DATA_DIR resolution: Same EMNIST train/test calls as exp03 with `root=str(DATA_DIR)`, `split=split`, and `download=False`; data is transposed, reshaped to 784, converted to `float32 / 255.0`, and row-normalized.
- `BASE_DIR` resolves to `/Users/anish/Developer/NCSU/RP2`; `DATA_DIR = BASE_DIR / "data"` resolves to `/Users/anish/Developer/NCSU/RP2/data`.
- Sampling/trials: Biased EMNIST byclass sampling. `BLUE_CLASS_END = 31`, so blue classes are `[c for c in all_classes if c < 31]` (0-30). `RED_CLASS_START = 31`, so red classes are `[c for c in all_classes if c >= 31]` (31-61). `_sample_classes` samples `spc = n_total // len(classes)` per class. Single trial per `n`; fixed `SEED = 42` and no loop over seeds.
- Computes ratio/distortion between exact OT cost and proxy/solver cost: YES.
  - Exact variable/key names used for ratios are the result dictionaries/keys passed to `compute_ratio`: `r["exact"]["cost"]`, `r["prx2"]["cost"]`, `r["prx3"]["cost"]`, or for solver files `r["sol2"]["cost"]`, `r["sol3"]["cost"]`; in row-formatting functions the local variables are `e`, `p2`, `p3`, `s2`, `s3`, `r2`, and `r3` where present.
- Tracks or records peak GPU memory anywhere: NO. It may call `torch.cuda.empty_cache()` or synchronize CUDA, but no `torch.cuda.max_memory_allocated()` or reset peak call appears in this file.
Exact implementation of `_sample_classes`:
```python
def _sample_classes(images, labels, class_indices, n_total, rng):
    classes = sorted(class_indices)
    spc = n_total // len(classes)
    if spc == 0:
        raise ValueError(f"n_total={n_total} too small for {len(classes)} classes")
    parts = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        if idx.size < spc:
            warnings.warn(f"Class {cls}: {idx.size} available, need {spc}. Skipping.")
            continue
        rng.shuffle(idx)
        parts.append(images[idx[:spc]])
    return np.concatenate(parts)
```

Exact implementation of `load_emnist_biased`:
```python
def load_emnist_biased(n_samples, seed, split=EMNIST_SPLIT):
    train = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=True,  download=False)
    test  = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    all_classes = np.unique(labels).tolist()
    blue_classes = [c for c in all_classes if c < BLUE_CLASS_END]
    red_classes  = [c for c in all_classes if c >= RED_CLASS_START]

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_arr  = _sample_classes(images, labels, red_classes,  n_samples, rng_r).astype(np.float32) / 255.0
    blue_arr = _sample_classes(images, labels, blue_classes, n_samples, rng_b).astype(np.float32) / 255.0

    for arr in (red_arr, blue_arr):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each EMNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s

    return torch.from_numpy(red_arr), torch.from_numpy(blue_arr)
```


### `experiments/runners/final2/experiments/exp05_nyc_scalability.py`

- Full file path: `experiments/runners/final2/experiments/exp05_nyc_scalability.py`
- `EXP_ID` = `5`
- `EXP_NAME` = `NYC Taxi — Exact vs 2L-Solver vs 3L-Solver (Scalability)`
- `DATASET` = `NYC Taxi`
- `DATA_DIR` = `not defined`
- `N_VALUES` = `[1000, 5000, 10000, 50000, 100000, 200000]`
- `EPSILON` = `0.01`
- `BATCH_SIZE` = `2048`
- `SEED` = `42`
- `EXACT_N_LIMIT` = `10000`
- Additional exact constants:
- `DEFAULT_DATA_PATH` = `pathlib.Path('./nyc_data/yellow_tripdata_2014-01.parquet')`
- Exact `run()` signature: `def run(device, nyc_data_path = None, nyc_day = None, **kwargs):`
- Exact data-loading / sampling function signatures:
  - `def load_taxi(path, day = None):`
  - `def make_points(df, n, rng):`
- Imports from `shared.py`: `compute_ratio, fmt_time, fmt_cost, fmt_ratio, fmt_iters`
- Metrics columns from `COL_SPECS`: `[('N', 7), ('Exact Time', 14), ('2L Time', 12), ('3L Time', 12), ('Exact Cost', 14), ('2L Cost', 12), ('3L Cost', 12), ('2L Ratio', 9), ('3L Ratio', 9), ('2L Iters', 10), ('3L Iters', 10)]`
- Result keys: Each row is `{"n": n, "exact": exact, "sol2": sol2, "sol3": sol3}`. Exact result keys are `time_ms`, `cost`, `status`. Solver result keys are `time_ms`, `cost`, `iters`, `status`.
- Dataset loading and DATA_DIR resolution: `load_taxi(path, day=None)` reads CSV via `pd.read_csv(path)` if the path string ends with `.csv`; otherwise it reads parquet via `pd.read_parquet(path)`. It optionally filters by pickup date, discovers pickup/dropoff lon/lat columns from variant lists, drops missing coordinates, filters to NYC latitude/longitude bounds, renames coordinate columns, and returns a reset-index dataframe.
- `DEFAULT_DATA_PATH = pathlib.Path("./nyc_data/yellow_tripdata_2014-01.parquet")`; from current repo root this resolves to `/Users/anish/Developer/NCSU/RP2/nyc_data/yellow_tripdata_2014-01.parquet`. No `DATA_DIR` constant is defined in this file.
- Sampling/trials: NYC taxi sampling from one dataframe. `make_points` shuffles all rows once per call through `rng.permutation(len(df))`, uses the first `n` pickup coordinates as blue `B_raw`, the next `n` dropoff coordinates as red `A_raw`, projects lon/lat to meters, then normalizes both by a shared diameter. Single pass per `n` with one `rng = np.random.default_rng(SEED)` created before the loop. This advances RNG state across increasing `N_VALUES`, but there is no repeat/trial loop.
- Computes ratio/distortion between exact OT cost and proxy/solver cost: YES.
  - Exact variable/key names used for ratios are the result dictionaries/keys passed to `compute_ratio`: `r["exact"]["cost"]`, `r["prx2"]["cost"]`, `r["prx3"]["cost"]`, or for solver files `r["sol2"]["cost"]`, `r["sol3"]["cost"]`; in row-formatting functions the local variables are `e`, `p2`, `p3`, `s2`, `s3`, `r2`, and `r3` where present.
- Tracks or records peak GPU memory anywhere: NO. It may call `torch.cuda.empty_cache()` or synchronize CUDA, but no `torch.cuda.max_memory_allocated()` or reset peak call appears in this file.
Exact implementation of `load_taxi`:
```python
def load_taxi(path, day=None):
    if pd is None:
        raise RuntimeError("pandas not installed")
    path = pathlib.Path(path)
    if str(path).endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    if day is not None:
        tc = _find_col(df, _PICKUP_TIME, "pickup datetime")
        df[tc] = pd.to_datetime(df[tc], utc=True, errors="coerce")
        if df[tc].dt.tz is None:
            df[tc] = df[tc].dt.tz_localize("America/New_York")
        else:
            df[tc] = df[tc].dt.tz_convert("America/New_York")
        df = df[df[tc].dt.date == pd.Timestamp(day).date()]

    plat = _find_col(df, _PICKUP_LAT, "pickup lat")
    plon = _find_col(df, _PICKUP_LON, "pickup lon")
    dlat = _find_col(df, _DROPOFF_LAT, "dropoff lat")
    dlon = _find_col(df, _DROPOFF_LON, "dropoff lon")

    df = df.dropna(subset=[plat, plon, dlat, dlon])
    df = df[df[plat].between(NYC_LAT_MIN, NYC_LAT_MAX) &
            df[plon].between(NYC_LON_MIN, NYC_LON_MAX) &
            df[dlat].between(NYC_LAT_MIN, NYC_LAT_MAX) &
            df[dlon].between(NYC_LON_MIN, NYC_LON_MAX)]
    df = df.rename(columns={plat: "pickup_latitude", plon: "pickup_longitude",
                             dlat: "dropoff_latitude", dlon: "dropoff_longitude"})
    return df.reset_index(drop=True)
```

Exact implementation of `make_points`:
```python
def make_points(df, n, rng):
    if len(df) < 2 * n:
        return None, None
    idx = rng.permutation(len(df))
    B_raw = df.iloc[idx[:n]][["pickup_longitude", "pickup_latitude"]].values.astype(np.float32)
    A_raw = df.iloc[idx[n:2*n]][["dropoff_longitude","dropoff_latitude"]].values.astype(np.float32)
    A_m = _project(A_raw)
    B_m = _project(B_raw)
    all_pts = np.vstack([A_m, B_m])
    diam = float((all_pts.max(0) - all_pts.min(0)).max())
    diam = max(diam, 1e-6)
    return (A_m / diam).astype(np.float32), (B_m / diam).astype(np.float32), diam
```


### `experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py`

- Full file path: `experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py`
- `EXP_ID` = `6`
- `EXP_NAME` = `CIFAR-10 SIFT — Exact vs 2L-Proxy vs 3L-Proxy`
- `DATASET` = `CIFAR-10 SIFT`
- `DATA_DIR` = `BASE_DIR / 'data' / 'cifar_sift'`
- `N_VALUES` = `[1000, 2000, 3000, 5000, 7000, 10000]`
- `EPSILON` = `0.01`
- `BATCH_SIZE` = `512`
- `SEED` = `42`
- `EXACT_N_LIMIT` = `10000`
- Additional exact constants:
- `TRAIN_DESC_PATH` = `DATA_DIR / 'cifar10_sift_train.pkl.gz'`
- Exact `run()` signature: `def run(device, **kwargs):`
- Exact data-loading / sampling function signatures:
  - `def load_descriptors(path):`
  - `def sample_pair(all_descs, n, seed):`
- Imports from `shared.py`: `compute_ratio, fmt_time, fmt_cost, fmt_ratio, build_two_level_proxy_matrix, build_three_level_proxy_matrix, run_three_level_precomputed`
- Metrics columns from `COL_SPECS`: `[('N', 7), ('Exact Time', 14), ('2L-Prx Time', 14), ('3L-Prx Time', 14), ('Exact Cost', 12), ('2L-Prx Cost', 12), ('3L-Prx Cost', 12), ('2L Ratio', 9), ('3L Ratio', 9)]`
- Result keys: Each row is `{"n": n, "exact": exact, "prx2": prx2, "prx3": prx3}`. Each method result from `_safe` has keys `time_ms`, `cost`, and `status`.
- Dataset loading and DATA_DIR resolution: `load_descriptors(TRAIN_DESC_PATH)` opens `TRAIN_DESC_PATH` with `gzip.open(path, "rb")` and `pickle.load(f)`. `TRAIN_DESC_PATH = DATA_DIR / "cifar10_sift_train.pkl.gz"`.
- `BASE_DIR` resolves to `/Users/anish/Developer/NCSU/RP2`; `DATA_DIR = BASE_DIR / "data" / "cifar_sift"` resolves to `/Users/anish/Developer/NCSU/RP2/data/cifar_sift`.
- Sampling/trials: Descriptor-set pair sampling. `sample_pair(all_descs, n, seed)` creates one `np.random.RandomState(seed)`, permutes descriptor indices, returns red descriptor sets from `perm[:n]` and blue descriptor sets from `perm[n:2*n]`. Single trial per `n`; `sample_pair` is called with the same `SEED = 42` for each `n`, no repeat/trial loop.
- Computes ratio/distortion between exact OT cost and proxy/solver cost: YES.
  - Exact variable/key names used for ratios are the result dictionaries/keys passed to `compute_ratio`: `r["exact"]["cost"]`, `r["prx2"]["cost"]`, `r["prx3"]["cost"]`, or for solver files `r["sol2"]["cost"]`, `r["sol3"]["cost"]`; in row-formatting functions the local variables are `e`, `p2`, `p3`, `s2`, `s3`, `r2`, and `r3` where present.
- Tracks or records peak GPU memory anywhere: NO. It may call `torch.cuda.empty_cache()` or synchronize CUDA, but no `torch.cuda.max_memory_allocated()` or reset peak call appears in this file.
Exact implementation of `load_descriptors`:
```python
def load_descriptors(path):
    with gzip.open(path, "rb") as f:
        descs = pickle.load(f)
    print(f"  Loaded {len(descs):,} descriptor sets from {pathlib.Path(path).name}", flush=True)
    return descs
```

Exact implementation of `sample_pair`:
```python
def sample_pair(all_descs, n, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_descs))
    return ([all_descs[i] for i in perm[:n]],
            [all_descs[i] for i in perm[n:2*n]])
```


### `experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py`

- Full file path: `experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py`
- `EXP_ID` = `7`
- `EXP_NAME` = `CIFAR-10 SIFT — Exact vs 2L-Solver vs 3L-Solver (Scalability)`
- `DATASET` = `CIFAR-10 SIFT`
- `DATA_DIR` = `BASE_DIR / 'data' / 'cifar_sift'`
- `N_VALUES` = `[1000, 2000, 5000, 10000, 20000]`
- `EPSILON` = `0.01`
- `BATCH_SIZE` = `512`
- `SEED` = `42`
- `EXACT_N_LIMIT` = `5000`
- Additional exact constants:
- `TRAIN_DESC_PATH` = `DATA_DIR / 'cifar10_sift_train.pkl.gz'`
- Exact `run()` signature: `def run(device, **kwargs):`
- Exact data-loading / sampling function signatures:
  - `def load_descriptors(path):`
  - `def sample_pair(all_descs, n, seed):`
- Imports from `shared.py`: `compute_ratio, fmt_time, fmt_cost, fmt_ratio, fmt_iters, run_three_level_precomputed`
- Metrics columns from `COL_SPECS`: `[('N', 7), ('Exact Time', 14), ('2L Time', 12), ('3L Time', 12), ('Exact Cost', 12), ('2L Cost', 12), ('3L Cost', 12), ('2L Ratio', 9), ('3L Ratio', 9), ('2L Iters', 10), ('3L Iters', 10)]`
- Result keys: Each row is `{"n": n, "exact": exact, "sol2": sol2, "sol3": sol3}`. Exact result keys are `time_ms`, `cost`, `status`. Solver result keys are `time_ms`, `cost`, `iters`, `status`.
- Dataset loading and DATA_DIR resolution: Same `load_descriptors(path)` as exp06, reading gzip pickle from `TRAIN_DESC_PATH = DATA_DIR / "cifar10_sift_train.pkl.gz"`.
- `BASE_DIR` resolves to `/Users/anish/Developer/NCSU/RP2`; `DATA_DIR = BASE_DIR / "data" / "cifar_sift"` resolves to `/Users/anish/Developer/NCSU/RP2/data/cifar_sift`.
- Sampling/trials: Same descriptor-set pair sampling as exp06: one permutation from `np.random.RandomState(seed)`; red is `perm[:n]`, blue is `perm[n:2*n]`. Single trial per `n`; `sample_pair` uses the same `SEED = 42` for each `n`, no repeat/trial loop.
- Computes ratio/distortion between exact OT cost and proxy/solver cost: YES.
  - Exact variable/key names used for ratios are the result dictionaries/keys passed to `compute_ratio`: `r["exact"]["cost"]`, `r["prx2"]["cost"]`, `r["prx3"]["cost"]`, or for solver files `r["sol2"]["cost"]`, `r["sol3"]["cost"]`; in row-formatting functions the local variables are `e`, `p2`, `p3`, `s2`, `s3`, `r2`, and `r3` where present.
- Tracks or records peak GPU memory anywhere: NO. It may call `torch.cuda.empty_cache()` or synchronize CUDA, but no `torch.cuda.max_memory_allocated()` or reset peak call appears in this file.
Exact implementation of `load_descriptors`:
```python
def load_descriptors(path):
    with gzip.open(path, "rb") as f:
        descs = pickle.load(f)
    print(f"  Loaded {len(descs):,} descriptor sets from {pathlib.Path(path).name}", flush=True)
    return descs
```

Exact implementation of `sample_pair`:
```python
def sample_pair(all_descs, n, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_descs))
    return ([all_descs[i] for i in perm[:n]],
            [all_descs[i] for i in perm[n:2*n]])
```


### `experiments/runners/final2/experiments/exp08_newsgroups_proxy.py`

- Full file path: `experiments/runners/final2/experiments/exp08_newsgroups_proxy.py`
- `EXP_ID` = `8`
- `EXP_NAME` = `20 Newsgroups — Exact vs 2L-Proxy vs 3L-Proxy`
- `DATASET` = `20 Newsgroups (GloVe)`
- `DATA_DIR` = `BASE_DIR / 'data' / 'newsgroups_glove'`
- `N_VALUES` = `[1000, 2000, 3000, 5000, 7000, 10000]`
- `EPSILON` = `0.01`
- `BATCH_SIZE` = `512`
- `SEED` = `42`
- `EXACT_N_LIMIT` = `10000`
- Additional exact constants:
- `MAX_WORDS` = `300`
- `EMBEDDINGS_PATH` = `DATA_DIR / 'newsgroups_embeddings.pkl.gz'`
- Exact `run()` signature: `def run(device, **kwargs):`
- Exact data-loading / sampling function signatures:
  - `def load_embeddings(path):`
  - `def sample_pair(all_embs, n, seed):`
- Imports from `shared.py`: `compute_ratio, fmt_time, fmt_cost, fmt_ratio, build_two_level_proxy_matrix, build_three_level_proxy_matrix, run_three_level_precomputed`
- Metrics columns from `COL_SPECS`: `[('N', 7), ('Exact Time', 14), ('2L-Prx Time', 14), ('3L-Prx Time', 14), ('Exact Cost', 12), ('2L-Prx Cost', 12), ('3L-Prx Cost', 12), ('2L Ratio', 9), ('3L Ratio', 9)]`
- Result keys: Each row is `{"n": n, "exact": exact, "prx2": prx2, "prx3": prx3}`. Each method result from `_safe` has keys `time_ms`, `cost`, and `status`.
- Dataset loading and DATA_DIR resolution: `load_embeddings(EMBEDDINGS_PATH)` opens `EMBEDDINGS_PATH` with `gzip.open(path, "rb")` and `pickle.load(f)`. `EMBEDDINGS_PATH = DATA_DIR / "newsgroups_embeddings.pkl.gz"`.
- `BASE_DIR` resolves to `/Users/anish/Developer/NCSU/RP2`; `DATA_DIR = BASE_DIR / "data" / "newsgroups_glove"` resolves to `/Users/anish/Developer/NCSU/RP2/data/newsgroups_glove`.
- Sampling/trials: Document embedding pair sampling. `sample_pair(all_embs, n, seed)` creates one `np.random.RandomState(seed)`, permutes indices, returns red embeddings from `perm[:n]` and blue embeddings from `perm[n:2*n]`. Single trial per `n`; `sample_pair` uses the same `SEED = 42` for each `n`, no repeat/trial loop.
- Computes ratio/distortion between exact OT cost and proxy/solver cost: YES.
  - Exact variable/key names used for ratios are the result dictionaries/keys passed to `compute_ratio`: `r["exact"]["cost"]`, `r["prx2"]["cost"]`, `r["prx3"]["cost"]`, or for solver files `r["sol2"]["cost"]`, `r["sol3"]["cost"]`; in row-formatting functions the local variables are `e`, `p2`, `p3`, `s2`, `s3`, `r2`, and `r3` where present.
- Tracks or records peak GPU memory anywhere: NO. It may call `torch.cuda.empty_cache()` or synchronize CUDA, but no `torch.cuda.max_memory_allocated()` or reset peak call appears in this file.
Exact implementation of `load_embeddings`:
```python
def load_embeddings(path):
    with gzip.open(path, "rb") as f:
        embs = pickle.load(f)
    print(f"  Loaded {len(embs):,} document embeddings from {Path(path).name}", flush=True)
    return embs
```

Exact implementation of `sample_pair`:
```python
def sample_pair(all_embs, n, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_embs))
    return ([all_embs[i] for i in perm[:n]],
            [all_embs[i] for i in perm[n:2*n]])
```


### `experiments/runners/final2/experiments/exp09_newsgroups_scalability.py`

- Full file path: `experiments/runners/final2/experiments/exp09_newsgroups_scalability.py`
- `EXP_ID` = `9`
- `EXP_NAME` = `20 Newsgroups — Exact vs 2L-Solver vs 3L-Solver (Scalability)`
- `DATASET` = `20 Newsgroups (GloVe)`
- `DATA_DIR` = `BASE_DIR / 'data' / 'newsgroups_glove'`
- `N_VALUES` = `[1000, 2000, 5000, 7000, 9000]`
- `EPSILON` = `0.01`
- `BATCH_SIZE` = `512`
- `SEED` = `42`
- `EXACT_N_LIMIT` = `5000`
- Additional exact constants:
- `MAX_WORDS` = `300`
- `EMBEDDINGS_PATH` = `DATA_DIR / 'newsgroups_embeddings.pkl.gz'`
- Exact `run()` signature: `def run(device, **kwargs):`
- Exact data-loading / sampling function signatures:
  - `def load_embeddings(path):`
  - `def sample_pair(all_embs, n, seed):`
- Imports from `shared.py`: `compute_ratio, fmt_time, fmt_cost, fmt_ratio, fmt_iters, run_three_level_precomputed`
- Metrics columns from `COL_SPECS`: `[('N', 7), ('Exact Time', 14), ('2L Time', 12), ('3L Time', 12), ('Exact Cost', 12), ('2L Cost', 12), ('3L Cost', 12), ('2L Ratio', 9), ('3L Ratio', 9), ('2L Iters', 10), ('3L Iters', 10)]`
- Result keys: Each row is `{"n": n, "exact": exact, "sol2": sol2, "sol3": sol3}`. Exact result keys are `time_ms`, `cost`, `status`. Solver result keys are `time_ms`, `cost`, `iters`, `status`.
- Dataset loading and DATA_DIR resolution: Same `load_embeddings(path)` as exp08, reading gzip pickle from `EMBEDDINGS_PATH = DATA_DIR / "newsgroups_embeddings.pkl.gz"`.
- `BASE_DIR` resolves to `/Users/anish/Developer/NCSU/RP2`; `DATA_DIR = BASE_DIR / "data" / "newsgroups_glove"` resolves to `/Users/anish/Developer/NCSU/RP2/data/newsgroups_glove`.
- Sampling/trials: Same document embedding pair sampling as exp08: one permutation from `np.random.RandomState(seed)`; red is `perm[:n]`, blue is `perm[n:2*n]`. Single trial per `n`; `sample_pair` uses the same `SEED = 42` for each `n`, no repeat/trial loop.
- Computes ratio/distortion between exact OT cost and proxy/solver cost: YES.
  - Exact variable/key names used for ratios are the result dictionaries/keys passed to `compute_ratio`: `r["exact"]["cost"]`, `r["prx2"]["cost"]`, `r["prx3"]["cost"]`, or for solver files `r["sol2"]["cost"]`, `r["sol3"]["cost"]`; in row-formatting functions the local variables are `e`, `p2`, `p3`, `s2`, `s3`, `r2`, and `r3` where present.
- Tracks or records peak GPU memory anywhere: NO. It may call `torch.cuda.empty_cache()` or synchronize CUDA, but no `torch.cuda.max_memory_allocated()` or reset peak call appears in this file.
Exact implementation of `load_embeddings`:
```python
def load_embeddings(path):
    with gzip.open(path, "rb") as f:
        embs = pickle.load(f)
    print(f"  Loaded {len(embs):,} document embeddings from {Path(path).name}", flush=True)
    return embs
```

Exact implementation of `sample_pair`:
```python
def sample_pair(all_embs, n, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_embs))
    return ([all_embs[i] for i in perm[:n]],
            [all_embs[i] for i in perm[n:2*n]])
```


### `experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py`

- Full file path: `experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py`
- `EXP_ID` = `10`
- `EXP_NAME` = `MNIST — Exact vs 2L-Proxy vs 3L-Proxy (Dissimilar: B=1,2,4,7; A=8,6,9,3)`
- `DATASET` = `MNIST`
- `DATA_DIR` = `FINAL2_DIR / 'data'`
- `N_VALUES` = `[500, 1000, 2000, 3000, 5000]`
- `EPSILON` = `0.01`
- `BATCH_SIZE` = `512`
- `SEED` = `42`
- `EXACT_N_LIMIT` = `10000`
- Additional exact constants:
- `BLUE_DIGITS` = `[1, 2, 4, 7]`
- `RED_DIGITS` = `[8, 6, 9, 3]`
- Exact `run()` signature: `def run(device, **kwargs):`
- Exact data-loading / sampling function signatures:
  - `def _sample_from_digits(images, labels, digit_set, n_total, rng):`
  - `def load_mnist_dissimilar(n_samples, seed):`
- Imports from `shared.py`: `compute_ratio, fmt_time, fmt_cost, fmt_ratio, build_two_level_proxy_matrix, build_three_level_proxy_matrix`
- Metrics columns from `COL_SPECS`: `[('N', 7), ('Exact Time', 14), ('2L-Prx Time', 14), ('3L-Prx Time', 14), ('Exact Cost', 12), ('2L-Prx Cost', 12), ('3L-Prx Cost', 12), ('2L Ratio', 9), ('3L Ratio', 9)]`
- Additional diagnostic columns from `DIAG_COL_SPECS`: `[('N', 7), ('Exact >1', 10), ('Exact Min/Med/Max', 20), ('2L >1', 10), ('2L Min/Med/Max', 20), ('3L >1', 10), ('3L Min/Med/Max', 20)]`
- Result keys: Each row is `{"n": n, "exact": exact, "prx2": prx2, "prx3": prx3}`. Each method result from `_safe` has keys `time_ms`, `cost`, `frac_gt_1`, `cost_min`, `cost_median`, `cost_max`, and `status`.
- Dataset loading and DATA_DIR resolution: Same MNIST train/test calls as exp01 with `root=str(DATA_DIR)` and `download=False`; data is reshaped to `(-1, 784)`, converted to `float32 / 255.0`, and row-normalized.
- `FINAL2_DIR` resolves to `/Users/anish/Developer/NCSU/RP2/experiments/runners/final2`; `DATA_DIR = FINAL2_DIR / "data"` resolves to `/Users/anish/Developer/NCSU/RP2/experiments/runners/final2/data`.
- Sampling/trials: Biased/dissimilar MNIST sampling. `BLUE_DIGITS = [1, 2, 4, 7]`; `RED_DIGITS = [8, 6, 9, 3]`. `_sample_from_digits` sorts each set before sampling, so blue class order is `[1, 2, 4, 7]` and red class order is `[3, 6, 8, 9]`; it samples `spc = n_total // len(classes)` per digit after shuffling. Single trial per `n`; fixed `SEED = 42` and no loop over seeds.
- Computes ratio/distortion between exact OT cost and proxy/solver cost: YES.
  - Exact variable/key names used for ratios are the result dictionaries/keys passed to `compute_ratio`: `r["exact"]["cost"]`, `r["prx2"]["cost"]`, `r["prx3"]["cost"]`, or for solver files `r["sol2"]["cost"]`, `r["sol3"]["cost"]`; in row-formatting functions the local variables are `e`, `p2`, `p3`, `s2`, `s3`, `r2`, and `r3` where present.
- Tracks or records peak GPU memory anywhere: NO. It may call `torch.cuda.empty_cache()` or synchronize CUDA, but no `torch.cuda.max_memory_allocated()` or reset peak call appears in this file.
Exact implementation of `_sample_from_digits`:
```python
def _sample_from_digits(images, labels, digit_set, n_total, rng):
    """Sample n_total images equally from the specified digits."""
    classes = sorted(digit_set)
    spc = n_total // len(classes)
    if spc == 0:
        raise ValueError(f"n_total={n_total} too small for {len(classes)} classes")
    parts = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        if idx.size < spc:
            warnings.warn(f"Digit {cls}: only {idx.size} available, need {spc}. Skipping.")
            continue
        rng.shuffle(idx)
        parts.append(images[idx[:spc]])
    return np.concatenate(parts)
```

Exact implementation of `load_mnist_dissimilar`:
```python
def load_mnist_dissimilar(n_samples, seed):
    train = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=False)
    test  = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy().reshape(-1, 784)
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_arr  = _sample_from_digits(images, labels, RED_DIGITS,  n_samples, rng_r).astype(np.float32) / 255.0
    blue_arr = _sample_from_digits(images, labels, BLUE_DIGITS, n_samples, rng_b).astype(np.float32) / 255.0

    for arr in (red_arr, blue_arr):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each MNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s

    return torch.from_numpy(red_arr), torch.from_numpy(blue_arr)
```


## Section 3 — shared.py

Complete contents of `experiments/runners/final2/shared.py`:

```python
#!/usr/bin/env python3
"""
Shared utilities for final2 experiments.

Provides:
  - Formatting helpers
  - Three-level precomputed clustering (from D_rr / D_br matrices)
  - Three-level proxy cost-matrix builder
  - Terminal table printing
  - Markdown result writing
"""

import datetime
import math
from pathlib import Path

import numpy as np
import torch


# ── Formatting ────────────────────────────────────────────────────────────────

def is_nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def fmt_time(v):
    return "N/A" if is_nan(v) else f"{float(v):.1f} ms"


def fmt_cost(v):
    return "N/A" if is_nan(v) else f"{float(v):.4f}"


def fmt_ratio(v):
    return "N/A" if is_nan(v) else f"{float(v):.4f}"


def fmt_iters(v):
    return "N/A" if is_nan(v) else f"{int(v):,}"


def fmt_bytes(v):
    if is_nan(v):
        return "N/A"
    v = float(v)
    if v >= 1024 ** 3:
        return f"{v / 1024**3:.2f} GiB"
    if v >= 1024 ** 2:
        return f"{v / 1024**2:.1f} MiB"
    if v >= 1024:
        return f"{v / 1024:.1f} KiB"
    return f"{v:.0f} B"


def compute_ratio(exact, approx):
    if is_nan(exact) or is_nan(approx) or float(exact) == 0.0:
        return math.nan
    return float(approx) / float(exact)


# ── Three-level precomputed clustering ───────────────────────────────────────
# Builds a dict compatible with ThreeLevelGPUSolver(precomputed_clustering=...)
# from precomputed (N x N) distance matrices D_rr and D_br.


def _group_offsets(b_idx: torch.Tensor) -> torch.Tensor:
    """0-based intra-group offset for each element in a sorted integer tensor."""
    m = b_idx.numel()
    if m <= 1:
        return torch.zeros(m, dtype=torch.long, device=b_idx.device)
    same = torch.zeros(m, dtype=torch.long, device=b_idx.device)
    same[1:] = (b_idx[1:] == b_idx[:-1]).long()
    cumsum = same.cumsum(0)
    start_vals = cumsum.masked_fill(same.bool(), 0)
    baseline = torch.cummax(start_vals, dim=0).values
    return cumsum - baseline


def _build_csr_adjacency(dist_mat, threshold_per_row, N_rows, N_cols,
                          epsilon, tile_size, device):
    """
    Build CSR adjacency where entry (i, j) is included iff
    dist_mat[i, j] < threshold_per_row[i].

    dist_mat   : (N_rows, N_cols) float32 on device
    threshold  : (N_rows,) float32 on device

    Returns (ptr, col, dist_int, dist_float).
    """
    thr = threshold_per_row.unsqueeze(1)  # (N_rows, 1)
    counts = torch.zeros(N_rows, dtype=torch.long, device=device)
    for start in range(0, N_cols, tile_size):
        end = min(start + tile_size, N_cols)
        counts.add_((dist_mat[:, start:end] < thr).sum(dim=1))

    ptr = torch.zeros(N_rows + 1, dtype=torch.long, device=device)
    ptr[1:] = counts.cumsum(0)
    M = int(ptr[-1].item())
    col = torch.empty(M, dtype=torch.long, device=device)
    dist_int = torch.empty(M, dtype=torch.int32, device=device)
    dist_float = torch.empty(M, dtype=torch.float32, device=device)
    del counts

    if M > 0:
        cursor = ptr[:-1].clone()
        for start in range(0, N_cols, tile_size):
            end = min(start + tile_size, N_cols)
            tile = dist_mat[:, start:end]
            mask = tile < thr
            ri, ti = mask.nonzero(as_tuple=True)
            if ri.numel() == 0:
                continue
            wp = cursor[ri] + _group_offsets(ri)
            col[wp] = (ti + start).long()
            d = tile[ri, ti]
            dist_float[wp] = d
            dist_int[wp] = (d / epsilon).ceil_().to(torch.int32)
            cursor.scatter_add_(0, ri, torch.ones_like(ri))
        del cursor

    return ptr, col, dist_int, dist_float


def run_three_level_precomputed(D_rr, D_br, epsilon, tile_size=512):
    """
    Build a ThreeLevelGPUSolver-compatible clustering dict from precomputed
    pairwise distance matrices.

    Parameters
    ----------
    D_rr : (N, N) float32 CUDA tensor — red-to-red distances
    D_br : (N, N) float32 CUDA tensor — blue-to-red distances
    epsilon : float
    tile_size : int

    Returns
    -------
    dict with the same field names as ThreeLevelClustering.run() output.
    """
    device = D_rr.device
    N = D_rr.shape[0]
    eps = epsilon

    # 1. Sample A1 ⊆ A at rate N^(-1/3)  →  E[|A1|] ≈ N^(2/3)
    rate1 = 1.0 / (float(N) ** (1.0 / 3.0))
    mask_A1 = torch.rand(N, device=device) < rate1
    if not mask_A1.any():
        mask_A1[torch.randint(N, (1,), device=device)] = True
    sampled_idx_A1 = mask_A1.nonzero(as_tuple=True)[0]
    S1 = sampled_idx_A1.shape[0]

    # 2. Sample A2 ⊆ A1 at rate N^(-1/3)  →  E[|A2|] ≈ N^(1/3)
    rate2 = 1.0 / (float(N) ** (1.0 / 3.0))
    mask_A2 = torch.rand(S1, device=device) < rate2
    if not mask_A2.any():
        mask_A2[torch.randint(S1, (1,), device=device)] = True
    local_idx_A2 = mask_A2.nonzero(as_tuple=True)[0]
    sampled_idx_A2 = sampled_idx_A1[local_idx_A2]
    S2 = sampled_idx_A2.shape[0]

    # 3. DR: (S2, N) distances from A2 centers to all reds
    DR = D_rr[sampled_idx_A2, :]
    DR_int = (DR / eps).ceil_().to(torch.int32)

    # 4. Nearest A2 center for each blue
    DB_to_A2 = D_br[:, sampled_idx_A2]          # (N, S2)
    d_min_b_A2, nearest_s2 = DB_to_A2.min(dim=1)
    del DB_to_A2
    d_min_b_A2_int = (d_min_b_A2 / eps).ceil_().to(torch.int32)

    # 5. Nearest A2 for each A1 center
    D_A1_to_A2 = D_rr[sampled_idx_A1][:, sampled_idx_A2]  # (S1, S2)
    d_min_A1_A2, nearest_s2_A1 = D_A1_to_A2.min(dim=1)
    del D_A1_to_A2
    d_min_A1_A2_int = (d_min_A1_A2 / eps).ceil_().to(torch.int32)

    # 6. Nearest A1 center for each blue
    DB_to_A1 = D_br[:, sampled_idx_A1]          # (N, S1)
    d_min_b_A1, nearest_s1 = DB_to_A1.min(dim=1)
    del DB_to_A1
    d_min_b_A1_int = (d_min_b_A1 / eps).ceil_().to(torch.int32)

    D_A1_to_reds = D_rr[sampled_idx_A1, :]      # (S1, N)

    # 7. Adj_B: {a : d(b, a) < d_min_b_A1[b]}
    adj_B_ptr, adj_B_col, adj_B_dist_int, adj_B_dist_float = _build_csr_adjacency(
        D_br, d_min_b_A1, N, N, eps, tile_size, device
    )

    # 8. Adj_A1: {a : d(a1, a) < d_min_A1_A2[a1]}
    adj_A1_ptr, adj_A1_col, adj_A1_dist_int, adj_A1_dist_float = _build_csr_adjacency(
        D_A1_to_reds, d_min_A1_A2, S1, N, eps, tile_size, device
    )
    del D_A1_to_reds

    return {
        "sampled_idx_A1": sampled_idx_A1,
        "sampled_idx_A2": sampled_idx_A2,
        "A1_sampled": None,
        "A2_sampled": None,
        "DR": DR,
        "DR_int": DR_int,
        "d_min_b_A1": d_min_b_A1,
        "d_min_b_A1_int": d_min_b_A1_int,
        "nearest_s1": nearest_s1,
        "d_min_b_A2": d_min_b_A2,
        "d_min_b_A2_int": d_min_b_A2_int,
        "nearest_s2": nearest_s2,
        "d_min_A1_A2": d_min_A1_A2,
        "d_min_A1_A2_int": d_min_A1_A2_int,
        "nearest_s2_A1": nearest_s2_A1,
        "adj_B_ptr": adj_B_ptr,
        "adj_B_col": adj_B_col,
        "adj_B_dist_int": adj_B_dist_int,
        "adj_B_dist_float": adj_B_dist_float,
        "adj_A1_ptr": adj_A1_ptr,
        "adj_A1_col": adj_A1_col,
        "adj_A1_dist_int": adj_A1_dist_int,
        "adj_A1_dist_float": adj_A1_dist_float,
    }


def build_three_level_proxy_matrix(clustering, N, device):
    """
    Build the full N×N proxy cost matrix from a three-level clustering dict.

    Proxy priority (each level overwrites the coarser estimate):
      Level 2 (base):  C[b,a] = d_min_b_A2[b] + DR[nearest_s2[b], a]
      Level 1 (refine):C[b,a] = d_min_b_A1[b] + d(s1_b, a)  for a ∈ Adj_A1(s1_b)
      Level 0 (direct):C[b,a] = d(b, a)                      for a ∈ Adj_B(b)

    Returns (N, N) float64 numpy array for ot.emd().
    """
    DR = clustering["DR"]
    d_min_b_A2 = clustering["d_min_b_A2"]
    nearest_s2 = clustering["nearest_s2"]
    d_min_b_A1 = clustering["d_min_b_A1"]
    nearest_s1 = clustering["nearest_s1"]
    adj_B_ptr = clustering["adj_B_ptr"]
    adj_B_col = clustering["adj_B_col"]
    adj_B_dist_float = clustering["adj_B_dist_float"]
    adj_A1_ptr = clustering["adj_A1_ptr"]
    adj_A1_col = clustering["adj_A1_col"]
    adj_A1_dist_float = clustering["adj_A1_dist_float"]
    S1 = int(adj_A1_ptr.shape[0]) - 1

    # Level 2 base
    C = d_min_b_A2.unsqueeze(1) + DR[nearest_s2, :]   # (N, N) float32

    # Level 1 — group blues by nearest A1 center, scatter-overwrite
    sorted_order = torch.argsort(nearest_s1)
    group_counts = torch.bincount(nearest_s1, minlength=S1)
    group_ptr = torch.zeros(S1 + 1, dtype=torch.long, device=device)
    group_ptr[1:] = group_counts.cumsum(0)

    for a1_i in range(S1):
        a1_s = int(adj_A1_ptr[a1_i].item())
        a1_e = int(adj_A1_ptr[a1_i + 1].item())
        if a1_s == a1_e:
            continue
        a_cols = adj_A1_col[a1_s:a1_e]
        a_dists = adj_A1_dist_float[a1_s:a1_e]
        g_s = int(group_ptr[a1_i].item())
        g_e = int(group_ptr[a1_i + 1].item())
        if g_s == g_e:
            continue
        blues = sorted_order[g_s:g_e]
        C[blues.unsqueeze(1), a_cols.unsqueeze(0)] = (
            d_min_b_A1[blues].unsqueeze(1) + a_dists.unsqueeze(0)
        )

    # Level 0 — direct
    if adj_B_col.numel() > 0:
        b_idx = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            adj_B_ptr[1:] - adj_B_ptr[:-1],
        )
        C[b_idx, adj_B_col] = adj_B_dist_float

    return C.cpu().to(torch.float64).numpy()


# ── Two-level proxy cost-matrix builder (from SimplePrecomputedClustering) ───

def build_two_level_proxy_matrix(clustering, N, device):
    """
    Build N×N proxy cost matrix from SimplePrecomputedClustering output.
    For each (b, a): if a ∈ adj(b) use exact dist; else d_min_b[b] + DR[nearest_s[b], a].
    Returns float64 numpy array.
    """
    DR = clustering["DR"]
    d_min_b = clustering["d_min_b"]
    nearest_s = clustering["nearest_s"]
    adj_ptr = clustering["adj_ptr"]
    adj_col = clustering["adj_col"]
    adj_dist_float = clustering["adj_dist_float"]

    C = d_min_b.unsqueeze(1) + DR[nearest_s, :]
    if adj_col.numel() > 0:
        b_idx = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            adj_ptr[1:] - adj_ptr[:-1],
        )
        C[b_idx, adj_col] = adj_dist_float
    return C.cpu().to(torch.float64).numpy()


# ── Terminal table printer ────────────────────────────────────────────────────

def print_results_table(title, col_specs, rows, fmt_fns):
    """
    Print a bordered results table to stdout.

    col_specs : list of (header_str, width_int)
    rows      : list of dicts
    fmt_fns   : dict mapping column header → callable(row) → str
    """
    widths = [max(len(h), w) for h, w in col_specs]
    headers = [h for h, _ in col_specs]

    top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    hdr = "│" + "│".join(f" {h:^{w}} " for h, w in zip(headers, widths)) + "│"
    mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    if title:
        print(f"\n{title}")
    print(top)
    print(hdr)
    print(mid)
    for row in rows:
        cells = [fmt_fns[h](row) for h in headers]
        line = "│" + "│".join(f" {c:>{w}} " for c, w in zip(cells, widths)) + "│"
        print(line)
    print(bot)


# ── Markdown writer ───────────────────────────────────────────────────────────

def get_results_path(results_dir: Path) -> Path:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return results_dir / f"results_{ts}.md"


def _md_table(col_specs, rows, fmt_fns):
    headers = [h for h, _ in col_specs]
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---:" for _ in headers) + " |"
    lines = [header_row, separator]
    for row in rows:
        cells = [fmt_fns[h](row) for h in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(filepath: Path, sections: list):
    """
    Write a markdown results file.

    sections : list of dicts with keys:
        title     : str
        subtitle  : str (optional)
        col_specs : list of (header, width)
        rows      : list of row dicts
        fmt_fns   : dict header → callable(row) → str
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Experiment Results",
        f"",
        f"Generated: {ts}",
        f"",
    ]
    for sec in sections:
        lines.append(f"## {sec['title']}")
        if sec.get("subtitle"):
            lines.append(f"")
            lines.append(f"_{sec['subtitle']}_")
        lines.append(f"")
        if sec.get("rows"):
            lines.append(_md_table(sec["col_specs"], sec["rows"], sec["fmt_fns"]))
        else:
            lines.append("_No results collected._")
        lines.append(f"")
    filepath.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to: {filepath}", flush=True)
```

## Section 4 — run_experiments.py

Top-level runner files found in `experiments/runners/final2/`: `run_experiments.py`. No other top-level runner file exists in this directory.

Complete contents of `experiments/runners/final2/run_experiments.py`:

```python
#!/usr/bin/env python3
"""
Experiment control file — Final2
=================================
Presents an interactive checklist of all 10 experiments, runs selected ones
with live terminal progress, prints a final summary table, and writes every
result set to a timestamped Markdown file under results/.

Usage:
  python run_experiments.py                     # interactive menu
  python run_experiments.py --run 1,3,5         # run specific experiments
  python run_experiments.py --run all           # run everything
  python run_experiments.py --nyc-data /path/to/yellow_tripdata.parquet
"""

import argparse
import datetime
import math
import sys
import time
from pathlib import Path

import torch

FINAL2_DIR = Path(__file__).resolve().parent
if str(FINAL2_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL2_DIR))

from shared import get_results_path, write_markdown

# ── Import all experiment modules ────────────────────────────────────────────
from experiments import (
    exp01_mnist_proxy_equal   as exp01,
    exp02_mnist_proxy_biased  as exp02,
    exp03_emnist_proxy_equal  as exp03,
    exp04_emnist_proxy_biased as exp04,
    exp05_nyc_scalability     as exp05,
    exp06_cifar_sift_proxy    as exp06,
    exp07_cifar_sift_scalability as exp07,
    exp08_newsgroups_proxy    as exp08,
    exp09_newsgroups_scalability as exp09,
    exp10_mnist_proxy_dissimilar as exp10,
)

ALL_MODULES = [exp01, exp02, exp03, exp04, exp05, exp06, exp07, exp08, exp09, exp10]

RESULTS_DIR = FINAL2_DIR / "results"

# ── Terminal helpers ──────────────────────────────────────────────────────────

WIDTH = 70

def _bar(char="─"):
    return char * WIDTH


def _header(text, char="═"):
    pad = max(0, WIDTH - len(text) - 4)
    left = pad // 2
    right = pad - left
    return f"╔{'═' * (WIDTH - 2)}╗\n║ {' ' * left}{text}{' ' * right} ║\n╚{'═' * (WIDTH - 2)}╝"


def _section(text):
    return f"\n{_bar()}\n  {text}\n{_bar()}"


def _check(done):
    return "✓" if done else " "


def print_banner(device):
    print(_header("OT Experiment Runner — Final2"))
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Device : cuda — {name}  ({mem:.1f} GiB)")
    else:
        print(f"  Device : cpu")
    print(f"  Results: {RESULTS_DIR}")
    print()


def print_menu(completed):
    print("  Available experiments:")
    for mod in ALL_MODULES:
        mark = _check(completed.get(mod.EXP_ID, False))
        print(f"    [{mark}] {mod.EXP_ID:>2}.  {mod.EXP_NAME}")
    print()


def ask_selection(completed):
    while True:
        raw = input("  Enter experiment numbers to run  (e.g. 1,2,3  or  all): ").strip()
        if raw.lower() == "all":
            return list(range(1, 11))
        try:
            ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
            valid = [i for i in ids if 1 <= i <= 10]
            if not valid:
                raise ValueError
            return valid
        except ValueError:
            print("  Invalid input. Please enter comma-separated numbers 1–10 or 'all'.")


# ── Table printer for final summary ──────────────────────────────────────────

def _fmt_row(mod, row):
    return [fn(row) for fn in mod.FMT_FNS.values()]


def _print_table(col_specs, fmt_fns_by_header, rows):
    headers = [h for h, _ in col_specs]
    widths  = [max(len(h), w) for h, w in col_specs]
    fmt_fns = [fmt_fns_by_header[h] for h in headers]

    top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    hdr = "│" + "│".join(f" {h:^{w}} " for h, w in zip(headers, widths)) + "│"
    mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    print(top)
    print(hdr)
    print(mid)
    for row in rows:
        cells = [fn(row) for fn in fmt_fns]
        line  = "│" + "│".join(f" {c:>{w}} " for c, w in zip(cells, widths)) + "│"
        print(line)
    print(bot)


def print_results_table(mod, rows):
    if not rows:
        print("  (no results)")
        return
    _print_table(mod.COL_SPECS, mod.FMT_FNS, rows)
    if hasattr(mod, "DIAG_COL_SPECS") and hasattr(mod, "DIAG_FMT_FNS"):
        print()
        _print_table(mod.DIAG_COL_SPECS, mod.DIAG_FMT_FNS, rows)


# ── Markdown section builder ──────────────────────────────────────────────────

def _md_sections(mod, rows):
    sections = [{
        "title":     f"Experiment {mod.EXP_ID}: {mod.EXP_NAME}",
        "subtitle":  f"Dataset: {mod.DATASET}",
        "col_specs": mod.COL_SPECS,
        "rows":      rows,
        "fmt_fns":   mod.FMT_FNS,
    }]
    if hasattr(mod, "DIAG_COL_SPECS") and hasattr(mod, "DIAG_FMT_FNS"):
        sections.append({
            "title":     f"Experiment {mod.EXP_ID}: {mod.EXP_NAME} Diagnostics",
            "subtitle":  f"Dataset: {mod.DATASET}",
            "col_specs": mod.DIAG_COL_SPECS,
            "rows":      rows,
            "fmt_fns":   mod.DIAG_FMT_FNS,
        })
    return sections


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Final2 OT experiments and save results to Markdown."
    )
    parser.add_argument("--run",      default=None,
                        help="Comma-separated experiment IDs or 'all'.")
    parser.add_argument("--nyc-data", default=None,
                        help="Path to NYC taxi parquet file (for Exp 5).")
    parser.add_argument("--nyc-day",  default=None,
                        help="Date filter for NYC taxi data (YYYY-MM-DD).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print_banner(device)
    completed = {}

    # ── Select experiments ────────────────────────────────────────────────────
    if args.run is not None:
        if args.run.lower() == "all":
            selected_ids = list(range(1, 11))
        else:
            selected_ids = [int(x.strip()) for x in args.run.split(",") if x.strip()]
    else:
        print_menu(completed)
        selected_ids = ask_selection(completed)

    selected_mods = [m for m in ALL_MODULES if m.EXP_ID in selected_ids]
    print(f"\n  Will run {len(selected_mods)} experiment(s): "
          f"{', '.join(str(m.EXP_ID) for m in selected_mods)}\n")

    # ── Run each experiment ───────────────────────────────────────────────────
    all_results = {}
    md_sections = []
    overall_start = time.perf_counter()

    for idx, mod in enumerate(selected_mods, 1):
        print(_section(
            f"[{idx}/{len(selected_mods)}]  Exp {mod.EXP_ID}: {mod.EXP_NAME}"
        ), flush=True)

        kwargs = {}
        if mod.EXP_ID == 5:
            if args.nyc_data:
                kwargs["nyc_data_path"] = args.nyc_data
            if args.nyc_day:
                kwargs["nyc_day"] = args.nyc_day

        t0 = time.perf_counter()
        try:
            rows = mod.run(device, **kwargs)
        except KeyboardInterrupt:
            print("\n  Interrupted by user. Saving results collected so far.", flush=True)
            rows = []
        elapsed_s = time.perf_counter() - t0

        completed[mod.EXP_ID] = True
        all_results[mod.EXP_ID] = rows
        md_sections.extend(_md_sections(mod, rows))

        print(f"\n  ── Results: Exp {mod.EXP_ID} ({elapsed_s:.1f}s total) ──", flush=True)
        print_results_table(mod, rows)

    # ── Grand total timing ────────────────────────────────────────────────────
    total_s = time.perf_counter() - overall_start
    print(f"\n{_bar()}", flush=True)
    print(f"  All selected experiments completed in {total_s:.1f}s", flush=True)
    print(_bar(), flush=True)

    # ── Final consolidated summary ────────────────────────────────────────────
    if len(selected_mods) > 1:
        print("\n\n" + _header("FINAL RESULTS SUMMARY"))
        for mod in selected_mods:
            rows = all_results.get(mod.EXP_ID, [])
            if rows:
                print(f"\n  Exp {mod.EXP_ID}: {mod.EXP_NAME}")
                print_results_table(mod, rows)

    # ── Write Markdown ────────────────────────────────────────────────────────
    if md_sections:
        md_path = get_results_path(RESULTS_DIR)
        write_markdown(md_path, md_sections)
    else:
        print("\n  No results to write.", flush=True)


if __name__ == "__main__":
    main()
```

## Section 5 — Clustering Classes

- `SimpleL1Clustering` in `src/clustered_push_relabel/clustering/simple_l1.py`: `__init__(self, epsilon: float, tile_size: int = 2048, sample_factor: float = 1.0):`
- `ThreeLevelL1Clustering` in `src/clustered_push_relabel/clustering/simple_three_level_l1.py`: `__init__(self, epsilon: float, tile_size: int = 2048, sample_factor: float = 1.0):`
- `SimpleClustering` in `src/clustered_push_relabel/clustering/simple.py`: `__init__(self, epsilon: float, tile_size: int = 2048, sample_factor: float = 1.0):`
- `SimpleClustering` in `src/clustered_push_relabel/clustering/simple_copy.py`: `__init__(self, epsilon: float, tile_size: int = 2048, sample_factor: float = 1.0):`
- `FastGPUMultiLevelClustering` in `src/clustered_push_relabel/clustering/k_level.py`: `__init__(self, epsilon, k = 4, batch_size = 2048, metric = 'L2'):`

Search results for FourLevel/FiveLevel/k-level clustering classes in `src/`:

```text
src/clustered_push_relabel/clustering/k_level.py:4:class FastGPUMultiLevelClustering:
src/clustered_push_relabel/clustering/k_level.py:128:    Partitions two point clouds into a K-level unified hierarchy of spatial cells.
src/clustered_push_relabel/solvers/transport.py:258:    Solves discrete Optimal Transport using K-level clustered push-relabel.
src/clustered_push_relabel/solvers/bipartite.py:735:class KLevelBipartiteSolver:
src/clustered_push_relabel/solvers/bipartite.py:737:    Bipartite matching solver using a K-level hierarchical decomposition.
src/clustered_push_relabel/solvers/bipartite.py:1412:    Solves min-cost bipartite matching using K-level clustered push-relabel.
src/clustered_push_relabel/solvers/bipartite.py:1433:        solver = KLevelBipartiteSolver(x, y, epsilon, k=k, batch_size=batch_size, metric=metric)
src/clustered_push_relabel/clustering/simple_three_level_l1.py:16:class ThreeLevelL1Clustering:
src/cluster_search/clustering.py:11:    """Samples point hierarchy levels using the current k-level clustering logic."""
src/clustered_push_relabel/clustering/simple_three_level.py:15:class ThreeLevelClustering:
src/clustered_push_relabel/solvers/three_level_bipartite.py:41:            clustering_class = ThreeLevelClustering
```

A generic k-level clustering implementation exists at `src/clustered_push_relabel/clustering/k_level.py` as `FastGPUMultiLevelClustering` with default `k=4`, so a 4-level implementation exists via the default constructor and a 5-level implementation exists by calling the same class with `k=5`. No class literally named `FourLevel...` or `FiveLevel...` exists anywhere in `src/` based on the search above.

## Section 6 — GPU and Hardware Config

Repository-wide search command used: `rg -n -i "A100|H100|V100|RTX|Tesla|GPU|CUDA_VISIBLE_DEVICES|cuda device|get_device_name|total_memory|max_memory|memory_reserved|memory_allocated|SLURM|SBATCH|gres|partition|cluster configuration|cluster config|GiB|MiB" . --glob "!**/.git/**"`. Relevant lines found:

```text
./src/clustered_push_relabel/solvers/bipartite.py:4:from ..clustering.two_level import FastGPUClustering
./src/clustered_push_relabel/solvers/bipartite.py:5:from ..clustering.k_level import FastGPUMultiLevelClustering
./src/clustered_push_relabel/solvers/bipartite.py:47:    This engine partitions points into a single layer of spatial cells 
./src/clustered_push_relabel/solvers/bipartite.py:75:        cluster_engine = FastGPUClustering(
./src/clustered_push_relabel/solvers/bipartite.py:285:            print(f"         Red Entries: {self.red_indices.numel()} (GPU)")
./src/clustered_push_relabel/solvers/bipartite.py:286:            print(f"         Blue Entries: {self.blue_center_indices.numel()} (GPU)")
./src/clustered_push_relabel/solvers/bipartite.py:307:            alloc = torch.cuda.memory_allocated() / 1024**2
./src/clustered_push_relabel/solvers/bipartite.py:308:            peak = torch.cuda.max_memory_allocated() / 1024**2
./src/clustered_push_relabel/solvers/bipartite.py:769:        cluster_engine = FastGPUMultiLevelClustering(
./src/clustered_push_relabel/solvers/bipartite.py:1423:        batch_size (int, optional): GPU batch size for clustering. Defaults to None.
./src/clustered_push_relabel/solvers/three_level_bipartite.py:8:    SimpleGPUSolver,
./src/clustered_push_relabel/solvers/three_level_bipartite.py:16:class ThreeLevelGPUSolver(SimpleGPUSolver):
./src/clustered_push_relabel/solvers/three_level_bipartite.py:18:    Epsilon-approximate GPU bipartite matcher over the ThreeLevelClustering graph.
./src/clustered_push_relabel/solvers/three_level_bipartite.py:103:                raise ValueError("ThreeLevelGPUSolver requires CUDA tensors")
./src/clustered_push_relabel/solvers/three_level_bipartite.py:218:        def _print_progress(iteration, free_before, free_after, status):
./src/clustered_push_relabel/solvers/three_level_bipartite.py:233:                    f"ThreeLevelGPUSolver exceeded max_iters={self.max_iters}"
./src/clustered_push_relabel/solvers/three_level_bipartite.py:340:                _print_progress(iteration, num_free, num_free, "no_proposals")
./src/clustered_push_relabel/solvers/three_level_bipartite.py:382:                _print_progress(iteration, num_free, num_free, "no_accepts")
./src/clustered_push_relabel/solvers/three_level_bipartite.py:420:            _print_progress(iteration, num_free, F_B_new.numel(), "ok")
./src/clustered_push_relabel/solvers/three_level_bipartite.py:434:    def _set1_eligible_mask(self, B_free):
./src/clustered_push_relabel/solvers/three_level_bipartite.py:461:        eligible = self._set1_eligible_mask(B_free)
./src/clustered_push_relabel/solvers/three_level_bipartite.py:462:        eligible_pos = torch.nonzero(eligible, as_tuple=True)[0]
./src/clustered_push_relabel/solvers/three_level_bipartite.py:465:        if eligible_pos.numel() == 0:
./src/clustered_push_relabel/solvers/three_level_bipartite.py:471:        B_eligible = B_free[eligible_pos]
./src/clustered_push_relabel/solvers/three_level_bipartite.py:472:        free_s = self.nearest_s2[B_eligible]
./src/clustered_push_relabel/solvers/three_level_bipartite.py:473:        free_t = self.d_min_b_A2_int[B_eligible] + 1 - self.y_B[B_eligible]
./src/clustered_push_relabel/solvers/three_level_bipartite.py:484:        eligible_pair_inverse = torch.empty_like(inverse_sorted)
./src/clustered_push_relabel/solvers/three_level_bipartite.py:485:        eligible_pair_inverse[order] = inverse_sorted
./src/clustered_push_relabel/solvers/three_level_bipartite.py:489:        pair_inverse[eligible_pos] = eligible_pair_inverse
./src/clustered_push_relabel/solvers/three_level_bipartite.py:516:        eligible = self._set1_eligible_mask(B_free)
./src/clustered_push_relabel/solvers/three_level_bipartite.py:517:        B_eligible = B_free[eligible]
./src/clustered_push_relabel/solvers/three_level_bipartite.py:518:        if B_eligible.numel() == 0:
./src/clustered_push_relabel/solvers/three_level_bipartite.py:523:        free_s = self.nearest_s2[B_eligible]
./src/clustered_push_relabel/solvers/three_level_bipartite.py:524:        free_t = self.d_min_b_A2_int[B_eligible] + 1 - self.y_B[B_eligible]
./src/clustered_push_relabel/solvers/three_level_bipartite.py:535:        eligible_pair_inverse = torch.empty_like(inverse_sorted)
./src/clustered_push_relabel/solvers/three_level_bipartite.py:536:        eligible_pair_inverse[order] = inverse_sorted
./src/clustered_push_relabel/solvers/three_level_bipartite.py:539:        pair_inverse[eligible] = eligible_pair_inverse
./src/clustered_push_relabel/solvers/three_level_bipartite.py:876:            set1_eligible = self._set1_eligible_mask(B_free)
./src/clustered_push_relabel/solvers/three_level_bipartite.py:877:            if set1_eligible.any().item():
./src/clustered_push_relabel/solvers/three_level_bipartite.py:888:                min_slack1_per_blue[set1_eligible] = (
./src/clustered_push_relabel/solvers/three_level_bipartite.py:889:                    target2[set1_eligible]
./src/clustered_push_relabel/solvers/three_level_bipartite.py:890:                    - v_pair_row_max[pair_inverse[set1_eligible]]
./src/clustered_push_relabel/solvers/simple_bipartite.py:25:class SimpleGPUSolver:
./src/clustered_push_relabel/solvers/simple_bipartite.py:27:    Epsilon-approximate GPU bipartite matcher over the SimpleClustering graph.
./src/clustered_push_relabel/solvers/simple_bipartite.py:106:                raise ValueError("SimpleGPUSolver requires CUDA tensors")
./src/clustered_push_relabel/solvers/simple_bipartite.py:197:        self._debug_last_set1_eligible_by_b = None
./src/clustered_push_relabel/solvers/simple_bipartite.py:218:        def _print_progress(iteration, free_before, free_after, status):
./src/clustered_push_relabel/solvers/simple_bipartite.py:245:                set1_eligible = self._set1_eligible_mask(B_free)
./src/clustered_push_relabel/solvers/simple_bipartite.py:294:                        set1_eligible,
./src/clustered_push_relabel/solvers/simple_bipartite.py:321:                _print_progress(iteration, num_free, num_free, "no_proposals")
./src/clustered_push_relabel/solvers/simple_bipartite.py:349:                    set1_eligible,
./src/clustered_push_relabel/solvers/simple_bipartite.py:389:                _print_progress(iteration, num_free, num_free, "no_accepts")
./src/clustered_push_relabel/solvers/simple_bipartite.py:427:            _print_progress(iteration, num_free, F_B_new.numel(), "ok")
./src/clustered_push_relabel/solvers/simple_bipartite.py:442:    def _set1_eligible_mask(self, B_free):
./src/clustered_push_relabel/solvers/simple_bipartite.py:469:        eligible = self._set1_eligible_mask(B_free)
./src/clustered_push_relabel/solvers/simple_bipartite.py:470:        eligible_pos = torch.nonzero(eligible, as_tuple=True)[0]
./src/clustered_push_relabel/solvers/simple_bipartite.py:473:        if eligible_pos.numel() == 0:
./src/clustered_push_relabel/solvers/simple_bipartite.py:479:        B_eligible = B_free[eligible_pos]
./src/clustered_push_relabel/solvers/simple_bipartite.py:480:        free_s = self.nearest_s[B_eligible]
./src/clustered_push_relabel/solvers/simple_bipartite.py:481:        free_t = self.d_min_b_int[B_eligible] + 1 - self.y_B[B_eligible]
./src/clustered_push_relabel/solvers/simple_bipartite.py:492:        eligible_pair_inverse = torch.empty_like(inverse_sorted)
./src/clustered_push_relabel/solvers/simple_bipartite.py:493:        eligible_pair_inverse[order] = inverse_sorted
./src/clustered_push_relabel/solvers/simple_bipartite.py:497:        pair_inverse[eligible_pos] = eligible_pair_inverse
./src/clustered_push_relabel/solvers/simple_bipartite.py:524:        eligible = self._set1_eligible_mask(B_free)
./src/clustered_push_relabel/solvers/simple_bipartite.py:525:        B_eligible = B_free[eligible]
./src/clustered_push_relabel/solvers/simple_bipartite.py:526:        if B_eligible.numel() == 0:
./src/clustered_push_relabel/solvers/simple_bipartite.py:531:        free_s = self.nearest_s[B_eligible]
./src/clustered_push_relabel/solvers/simple_bipartite.py:532:        free_t = self.d_min_b_int[B_eligible] + 1 - self.y_B[B_eligible]
./src/clustered_push_relabel/solvers/simple_bipartite.py:543:        eligible_pair_inverse = torch.empty_like(inverse_sorted)
./src/clustered_push_relabel/solvers/simple_bipartite.py:544:        eligible_pair_inverse[order] = inverse_sorted
./src/clustered_push_relabel/solvers/simple_bipartite.py:547:        pair_inverse[eligible] = eligible_pair_inverse
./src/clustered_push_relabel/solvers/simple_bipartite.py:588:            set1_eligible = self._set1_eligible_mask(B_free)
./src/clustered_push_relabel/solvers/simple_bipartite.py:589:            if set1_eligible.any().item():
./src/clustered_push_relabel/solvers/simple_bipartite.py:598:                min_slack1_per_blue[set1_eligible] = (
./src/clustered_push_relabel/solvers/simple_bipartite.py:599:                    target1[set1_eligible]
./src/clustered_push_relabel/solvers/simple_bipartite.py:600:                    - v_pair_row_max[pair_inverse[set1_eligible]]
./src/clustered_push_relabel/solvers/simple_bipartite.py:849:        set1_eligible,
./src/clustered_push_relabel/solvers/simple_bipartite.py:866:        set1_eligible_by_b = torch.zeros(N, device=device, dtype=torch.bool)
./src/clustered_push_relabel/solvers/simple_bipartite.py:872:        set1_eligible_by_b[B_free] = set1_eligible
./src/clustered_push_relabel/solvers/simple_bipartite.py:888:        self._debug_last_set1_eligible_by_b = set1_eligible_by_b
./src/clustered_push_relabel/solvers/simple_bipartite.py:1001:        if self._debug_last_set1_eligible_by_b is None:
./src/clustered_push_relabel/solvers/simple_bipartite.py:1003:        if not bool(self._debug_last_set1_eligible_by_b[b].item()):
./src/clustered_push_relabel/solvers/simple_bipartite.py:1026:        if self._debug_last_set1_eligible_by_b is None:
./src/clustered_push_relabel/solvers/simple_bipartite.py:1031:        if not in_B_free or not bool(self._debug_last_set1_eligible_by_b[b].item()):
./src/clustered_push_relabel/solvers/simple_bipartite.py:1177:        set1_eligible = self._debug_bool_at(self._debug_last_set1_eligible_by_b, b)
./src/clustered_push_relabel/solvers/simple_bipartite.py:1237:            f"set1_eligible={set1_eligible} set1_count={set1_count} "
./src/clustered_push_relabel/solvers/__init__.py:5:from .simple_bipartite import SimpleGPUSolver
./src/clustered_push_relabel/solvers/__init__.py:6:from .three_level_bipartite import ThreeLevelGPUSolver
./src/clustered_push_relabel/solvers/transport.py:5:from ..clustering.k_level import FastGPUMultiLevelClustering
./src/clustered_push_relabel/solvers/transport.py:7:class GPUClusteredOTSolver:
./src/clustered_push_relabel/solvers/transport.py:21:        batch_size (int, optional): CPU/GPU Batch parameter. Defaults to 2048.
./src/clustered_push_relabel/solvers/transport.py:38:        cluster_engine = FastGPUMultiLevelClustering(epsilon, k=k, batch_size=self.batch_size, metric=metric)
./src/clustered_push_relabel/solvers/transport.py:270:        batch_size (int, optional): GPU batch size for clustering. Defaults to 2048.
./src/clustered_push_relabel/solvers/transport.py:279:    solver = GPUClusteredOTSolver(x, y, mass_x, mass_y, epsilon, k=k, batch_size=batch_size, metric=metric)
./src/clustered_push_relabel/solvers/simple_bipartite_2.py:26:class SimpleGPUSolver2:
./src/clustered_push_relabel/solvers/simple_bipartite_2.py:30:    This solver uses the same clustering outputs as the original SimpleGPUSolver,
./src/clustered_push_relabel/solvers/simple_bipartite_2.py:64:            raise ValueError("SimpleGPUSolver2 requires CUDA tensors")
./src/clustered_push_relabel/solvers/simple_bipartite_2.py:163:            # Determine which free blues are eligible for Set 1 in this phase:
./src/clustered_push_relabel/solvers/simple_bipartite_2.py:180:                        "SimpleGPUSolver2: no proposals and no positive delta."
./src/clustered_push_relabel/solvers/simple_bipartite_2.py:214:                        "SimpleGPUSolver2: no proposal vertices and no positive delta."
./src/clustered_push_relabel/solvers/simple_bipartite_2.py:237:                        "SimpleGPUSolver2: no accepted proposals and no positive delta."
./src/clustered_push_relabel/solvers/simple_bipartite_2.py:259:                        "SimpleGPUSolver2: non-positive delta after matching update."
./src/clustered_push_relabel/clustering/simple.py:7:def _gpu_mem(label):
./src/clustered_push_relabel/clustering/simple.py:9:        alloc = torch.cuda.memory_allocated() / 1024**3
./src/clustered_push_relabel/clustering/simple.py:10:        res   = torch.cuda.memory_reserved()  / 1024**3
./src/clustered_push_relabel/clustering/simple.py:104:        _gpu_mem("before DR cdist")
./src/clustered_push_relabel/clustering/simple.py:111:        _gpu_mem("after DB deleted")
./src/clustered_push_relabel/clustering/simple.py:114:        _gpu_mem("after DR deleted")
./src/clustered_push_relabel/clustering/simple.py:125:        # Keeping these alive outside the loop means zero GPU allocations in
./src/clustered_push_relabel/clustering/simple.py:127:        _gpu_mem("before tile buffers")
./src/clustered_push_relabel/clustering/simple.py:130:        _gpu_mem("after tile buffers")
./src/clustered_push_relabel/clustering/simple.py:143:        _gpu_mem("after pass 1")
./src/clustered_push_relabel/clustering/simple.py:151:        _gpu_mem("after adj arrays allocated")
./src/clustered_push_relabel/clustering/simple.py:156:            _gpu_mem("before return")
./src/clustered_push_relabel/clustering/simple.py:188:        _gpu_mem("after pass 2")
./src/clustered_push_relabel/clustering/simple.py:191:        _gpu_mem("before return")
./src/clustered_push_relabel/clustering/simple_copy.py:113:        # Keeping these alive outside the loop means zero GPU allocations in
./src/clustered_push_relabel/clustering/simple_three_level.py:7:def _gpu_mem(label: str) -> None:
./src/clustered_push_relabel/clustering/simple_three_level.py:9:        alloc = torch.cuda.memory_allocated() / 1024 ** 3
./src/clustered_push_relabel/clustering/simple_three_level.py:10:        res   = torch.cuda.memory_reserved()  / 1024 ** 3
./src/clustered_push_relabel/clustering/simple_three_level.py:130:        _gpu_mem(f"sampled A1={S1}, A2={S2}")
./src/clustered_push_relabel/clustering/simple_three_level.py:142:        _gpu_mem("after DR")
./src/clustered_push_relabel/clustering/simple_three_level.py:150:        _gpu_mem("after DB_A2")
./src/clustered_push_relabel/clustering/simple_three_level.py:158:        _gpu_mem("after DA1_A2")
./src/clustered_push_relabel/clustering/simple_three_level.py:182:        _gpu_mem("after DB_A1 tiled min")
./src/clustered_push_relabel/clustering/simple_three_level.py:200:        _gpu_mem("Adj_B pass 1 done")
./src/clustered_push_relabel/clustering/simple_three_level.py:209:        _gpu_mem(f"Adj_B allocated MB={MB}")
./src/clustered_push_relabel/clustering/simple_three_level.py:230:        _gpu_mem("Adj_B pass 2 done")
./src/clustered_push_relabel/clustering/simple_three_level.py:253:        _gpu_mem("Adj_A1 pass 1 done")
./src/clustered_push_relabel/clustering/simple_three_level.py:262:        _gpu_mem(f"Adj_A1 allocated MA1={MA1}")
./src/clustered_push_relabel/clustering/simple_three_level.py:283:        _gpu_mem("Adj_A1 pass 2 done")
./src/clustered_push_relabel/clustering/simple_three_level.py:285:        _gpu_mem("before return")
./src/clustered_push_relabel/clustering/k_level.py:4:class FastGPUMultiLevelClustering:
./src/clustered_push_relabel/clustering/k_level.py:6:    Implements the Multi-Level Hierarchical Clustering (Decomposition) on GPU.
./src/clustered_push_relabel/clustering/k_level.py:8:    This stateful engine partitions point clouds into a hierarchy of clusters, 
./src/clustered_push_relabel/clustering/k_level.py:14:        batch_size (int, optional): GPU batch size for processing centers. Defaults to 2048.
./src/clustered_push_relabel/clustering/k_level.py:128:    Partitions two point clouds into a K-level unified hierarchy of spatial cells.
./src/clustered_push_relabel/clustering/k_level.py:135:        batch_size (int, optional): GPU batch size for distance calculations. Defaults to 2048.
./src/clustered_push_relabel/clustering/k_level.py:145:    model = FastGPUMultiLevelClustering(epsilon=epsilon, k=k, batch_size=batch_size, metric=metric)
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:6:def _gpu_mem(label: str) -> None:
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:8:        alloc = torch.cuda.memory_allocated() / 1024 ** 3
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:9:        res = torch.cuda.memory_reserved() / 1024 ** 3
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:113:        _gpu_mem(f"sampled A1={S1}, A2={A2.shape[0]}")
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:117:        _gpu_mem("after DR")
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:123:        _gpu_mem("after DB_A2")
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:129:        _gpu_mem("after DA1_A2")
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:144:        _gpu_mem("after DB_A1 tiled min")
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:156:        _gpu_mem("Adj_B pass 1 done")
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:165:        _gpu_mem(f"Adj_B allocated MB={MB}")
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:185:        _gpu_mem("Adj_B pass 2 done")
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:197:        _gpu_mem("Adj_A1 pass 1 done")
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:206:        _gpu_mem(f"Adj_A1 allocated MA1={MA1}")
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:226:        _gpu_mem("Adj_A1 pass 2 done")
./src/clustered_push_relabel/clustering/simple_three_level_l1.py:228:        _gpu_mem("before return")
./src/clustered_push_relabel/clustering/two_level.py:4:class FastGPUClustering:
./src/clustered_push_relabel/clustering/simple_precomputed.py:7:def _gpu_mem(label):
./src/clustered_push_relabel/clustering/simple_precomputed.py:9:        alloc = torch.cuda.memory_allocated() / 1024**3
./src/clustered_push_relabel/clustering/simple_precomputed.py:10:        res = torch.cuda.memory_reserved() / 1024**3
./src/clustered_push_relabel/clustering/simple_precomputed.py:61:        _gpu_mem("before DR/DB slice")
./src/clustered_push_relabel/clustering/simple_precomputed.py:66:        _gpu_mem("after DB deleted")
./src/clustered_push_relabel/clustering/simple_precomputed.py:71:        _gpu_mem("before pass 1")
./src/clustered_push_relabel/clustering/simple_precomputed.py:79:        _gpu_mem("after pass 1")
./src/clustered_push_relabel/clustering/simple_precomputed.py:88:        _gpu_mem("after adj arrays allocated")
./src/clustered_push_relabel/clustering/simple_precomputed.py:91:            _gpu_mem("before return")
./src/clustered_push_relabel/clustering/simple_precomputed.py:123:        _gpu_mem("after pass 2")
./src/clustered_push_relabel/clustering/simple_precomputed.py:124:        _gpu_mem("before return")
./DOCS_SDD.md:51:        batch_size (int, optional): GPU batch size for clustering. Defaults to None.
./DOCS_SDD.md:91:3. **Review Core Classes:** Add brief class-level docstrings to the underlying `GPUClusteredSolver` and `GPUClusteredOTSolver` noting that they are the underlying stateful engines.
./README.md:8:A high-performance GPU library for discrete Optimal Transport and Bipartite Matching, accelerated by K-level spatial clustering.
./README.md:41:- **Clustering**: GPU-accelerated spatial decomposition routines (`k_level_cluster`).
./README.md:53:> **Status:** Active Research / Work in Progress
./mkdocs.yml:2:site_description: High-performance GPU library for discrete Optimal Transport and Bipartite Matching.
./COLOR_AWARE_SOLVER_SPEC.md:17:- GPU framework: PyTorch only. No CuPy, no custom CUDA.
./COLOR_AWARE_SOLVER_SPEC.md:76:  This matches the convention used by `FastGPUClustering` in `two_level.py`.
./COLOR_AWARE_SOLVER_SPEC.md:432:#### 3.4.8 Structures 5+6 — ball_sizes, d_max, max_list (vectorized GPU init)
./COLOR_AWARE_SOLVER_SPEC.md:434:All three are built with vectorized GPU ops. No Python loops.
./COLOR_AWARE_SOLVER_SPEC.md:586:        Step 7 is fully vectorized on GPU:
./COLOR_AWARE_SOLVER_SPEC.md:633:- `peak_memory_allocated_mb()`
./COLOR_AWARE_SOLVER_SPEC.md:661:    peak_mem   = peak_memory_allocated_mb(device)
./COLOR_AWARE_SOLVER_SPEC.md:693:solver_time_s, peak_gpu_mem_mb, cost, abs_error, rel_error`
./tests/test_clustering.py:3:from clustered_push_relabel.clustering.k_level import FastGPUMultiLevelClustering
./tests/test_clustering.py:4:from clustered_push_relabel.clustering.two_level import FastGPUClustering
./tests/test_clustering.py:16:    clustering = FastGPUMultiLevelClustering(epsilon, k=k)
./tests/test_clustering.py:33:    clustering = FastGPUClustering(epsilon)
./optimization/MEMORY.md:36:  - 2-Level improved by `7.6%` at n=1000, `4.3%` at n=2500, and regressed by `0.5%` at n=5000
./optimization/MEMORY.md:53:  - k-Level improved by `5.1%` at n=1000 and `4.2%` at n=5000, but regressed by `0.9%` at n=2500
./optimization/MEMORY.md:69:  - 2-Level improved by `3.0%` at n=1000 and `2.9%` at n=2500, but regressed by `0.9%` at n=5000
./optimization/MEMORY.md:96:  - k-Level regressed by `0.5%` at n=1000 but improved by `2.5%` at n=5000
./optimization/MEMORY.md:215:- This round delivered the best overall wall-clock results so far despite a small k-Level regression at n=2500.
./optimization/MEMORY.md:243:- The only measured regression relative to round 4 is 2-Level at n=5000, and it is small (`66.04s` vs `65.71s`).
./src/cluster_search/clustering.py:132:    # centers.  Computed in tiles to stay within GPU memory.
./docs/index.md:7:A high-performance GPU library for discrete Optimal Transport and Bipartite Matching, accelerated by K-level spatial clustering.
./docs/index.md:40:- **Clustering**: GPU-accelerated spatial decomposition routines (`k_level_cluster`).
./docs/index.md:52:> **Status:** Active Research / Work in Progress
./experiments/batch_tooling/generate_batch_jobs.py:9:# Slurm Header Template (Based on your provided script)
./experiments/batch_tooling/generate_batch_jobs.py:11:slurm_template = """#!/bin/bash
./experiments/batch_tooling/generate_batch_jobs.py:12:#SBATCH -J mnist_n{n}
./experiments/batch_tooling/generate_batch_jobs.py:13:#SBATCH -o batch/logs/mnist_n{n}-%j.out
./experiments/batch_tooling/generate_batch_jobs.py:14:#SBATCH -e batch/logs/mnist_n{n}-%j.err
./experiments/batch_tooling/generate_batch_jobs.py:15:#SBATCH -N 1
./experiments/batch_tooling/generate_batch_jobs.py:16:#SBATCH -n 1
./experiments/batch_tooling/generate_batch_jobs.py:17:#SBATCH --cpus-per-task=16
./experiments/batch_tooling/generate_batch_jobs.py:18:#SBATCH -t 01:00:00
./experiments/batch_tooling/generate_batch_jobs.py:19:#SBATCH -p rtx2060super
./experiments/batch_tooling/generate_batch_jobs.py:23:cd "${{SLURM_SUBMIT_DIR:-$PWD}}"
./experiments/batch_tooling/generate_batch_jobs.py:64:        script_content = slurm_template.format(n=n)
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:14:PARTITION  = "rtx2060super"
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:32:SLURM_TEMPLATE = """#!/bin/bash
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:33:#SBATCH -J synthetic_n{n}
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:34:#SBATCH -o {logs_dir}/synthetic_n{n}-%j.out
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:35:#SBATCH -e {logs_dir}/synthetic_n{n}-%j.err
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:36:#SBATCH -N 1
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:37:#SBATCH -n 1
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:38:#SBATCH --cpus-per-task={cpus}
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:39:#SBATCH -t {time_limit}
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:40:#SBATCH -p {partition}
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:44:cd "${{SLURM_SUBMIT_DIR:-$PWD}}"
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:81:        script = SLURM_TEMPLATE.format(
./experiments/batch_tooling/synthetic_generate_batch_jobs.py:86:            partition=PARTITION,
./experiments/runners/e4_clustering_coverage.py:271:            # Cleanup GPU memory between iterations
./experiments/batch_tooling/scale_generate_batch.py:12:# Slurm Header Template
./experiments/batch_tooling/scale_generate_batch.py:17:slurm_template = """#!/bin/bash
./experiments/batch_tooling/scale_generate_batch.py:18:#SBATCH -J synth_n{n}_k{k}_bs{bs}
./experiments/batch_tooling/scale_generate_batch.py:19:#SBATCH -o scale_batch/logs/synth_n{n}_k{k}_bs{bs}-%j.out
./experiments/batch_tooling/scale_generate_batch.py:20:#SBATCH -e scale_batch/logs/synth_n{n}_k{k}_bs{bs}-%j.err
./experiments/batch_tooling/scale_generate_batch.py:21:#SBATCH -N 1
./experiments/batch_tooling/scale_generate_batch.py:22:#SBATCH -n 1
./experiments/batch_tooling/scale_generate_batch.py:23:#SBATCH --cpus-per-task=16
./experiments/batch_tooling/scale_generate_batch.py:24:#SBATCH -t 01:00:00
./experiments/batch_tooling/scale_generate_batch.py:25:#SBATCH -p rtx2060super
./experiments/batch_tooling/scale_generate_batch.py:29:cd "${{SLURM_SUBMIT_DIR:-$PWD}}"
./experiments/batch_tooling/scale_generate_batch.py:60:                script_content = slurm_template.format(n=n, k=k, bs=bs)
./experiments/runners/experiment_mnist.py:6:# CRITICAL: Prevent JAX from hogging all GPU memory
./experiments/runners/experiment_mnist.py:54:        return torch.cuda.max_memory_allocated() / (1024 ** 2)
./experiments/runners/experiment_7.py:24:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/experiment_7.py:172:        solver = SimpleGPUSolver(P_red, P_blue, EPSILON, batch_size=BATCH_SIZE, verbose=False)
./experiments/runners/experiment_7.py:183:        solver = SimpleGPUSolver(P_red, P_blue, EPSILON, batch_size=BATCH_SIZE, verbose=False)
./experiments/runners/final2/run_experiments.py:6:with live terminal progress, prints a final summary table, and writes every
./experiments/runners/final2/run_experiments.py:75:        name = torch.cuda.get_device_name(0)
./experiments/runners/final2/run_experiments.py:76:        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
./experiments/runners/final2/run_experiments.py:77:        print(f"  Device : cuda — {name}  ({mem:.1f} GiB)")
./experiments/batch_tooling/process_e1_mnist_results.py:26:    numeric_cols = ['total_time_s', 'peak_gpu_mem_mb', 'cost', 'abs_error', 'rel_error']
./experiments/batch_tooling/process_e1_mnist_results.py:40:            'peak_gpu_mem_mb': ['mean', 'std']
./experiments/runners/e1_synthetic_vs_exact.py:9:# Prevent JAX from preallocating GPU memory (in case we use Sinkhorn here)
./experiments/runners/e1_synthetic_vs_exact.py:17:# Import custom GPU solvers
./experiments/runners/e1_synthetic_vs_exact.py:75:               "total_time_s", "cluster_time_s", "solver_time_s", "peak_gpu_mem_mb",
./experiments/runners/e1_synthetic_vs_exact.py:98:            # Move to GPU for our methods (POT will use CPU numpy)
./experiments/runners/e1_synthetic_vs_exact.py:206:                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
./experiments/runners/e1_synthetic_vs_exact.py:215:                # Clean up solver to free GPU memory
./experiments/runners/e1_synthetic_vs_exact.py:251:                peak_mem_k = torch.cuda.max_memory_allocated() / (1024**2)
./experiments/runners/final2/shared.py:51:        return f"{v / 1024**3:.2f} GiB"
./experiments/runners/final2/shared.py:53:        return f"{v / 1024**2:.1f} MiB"
./experiments/runners/final2/shared.py:66:# Builds a dict compatible with ThreeLevelGPUSolver(precomputed_clustering=...)
./experiments/runners/final2/shared.py:130:    Build a ThreeLevelGPUSolver-compatible clustering dict from precomputed
./experiments/runners/hnsw_recall_experiment.py:53:        raise RuntimeError("This experiment requires CUDA because the dataset and clustering are specified to run on GPU.")
./experiments/runners/experiment_6.py:20:from clustered_push_relabel.clustering.two_level import FastGPUClustering
./experiments/runners/experiment_6.py:136:            "FastGPUClustering",
./experiments/runners/experiment_6.py:137:            lambda: FastGPUClustering(epsilon=EPSILON, batch_size=BATCH_SIZE),
./experiments/runners/experiment_6.py:151:        ("FastGPUClustering", 19),
./experiments/runners/experiment_6.py:167:            f"{row['FastGPUClustering']:>16.1f} ms | "
./experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:119:def clustering_gpu_bytes(c):
./experiments/runners/final2/experiments/exp05_nyc_scalability.py:36:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/final2/experiments/exp05_nyc_scalability.py:37:from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
./experiments/runners/final2/experiments/exp05_nyc_scalability.py:187:    solver = SimpleGPUSolver(red, blue, EPSILON, batch_size=BATCH_SIZE, verbose=False, diameter=1.0)
./experiments/runners/final2/experiments/exp05_nyc_scalability.py:201:    solver = ThreeLevelGPUSolver(red, blue, EPSILON, batch_size=BATCH_SIZE, verbose=False, diameter=1.0)
./experiments/runners/e1_mnist_vs_exact.py:102:def peak_memory_allocated_mb(device):
./experiments/runners/e1_mnist_vs_exact.py:105:    return torch.cuda.max_memory_allocated() / (1024**2)
./experiments/runners/e1_mnist_vs_exact.py:142:               "total_time_s", "cluster_time_s", "solver_time_s", "peak_gpu_mem_mb",
./experiments/runners/e1_mnist_vs_exact.py:159:            # Move to GPU for our solvers
./experiments/runners/e1_mnist_vs_exact.py:245:                peak_mem = peak_memory_allocated_mb(device)
./experiments/runners/e1_mnist_vs_exact.py:282:                peak_memK = peak_memory_allocated_mb(device)
./experiments/runners/experiment_a.py:13:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/experiment_a.py:24:    mem = torch.cuda.memory_allocated() / 1024**3
./experiments/runners/experiment_a.py:25:    reserved = torch.cuda.memory_reserved() / 1024**3
./experiments/runners/experiment_a.py:36:        f"{'Iterations':>10} | {'Avg Cost':>8} | {'Peak GPU (GB)':>13}"
./experiments/runners/experiment_a.py:83:            ckpt(n, "3/8  starting SimpleGPUSolver init")
./experiments/runners/experiment_a.py:85:            # SimpleGPUSolver.__init__ calls SimpleClustering.run internally.
./experiments/runners/experiment_a.py:89:            solver = SimpleGPUSolver(
./experiments/runners/experiment_a.py:113:            peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
./experiments/runners/experiment_a.py:134:            print(f"    Peak GPU   : {peak_gb:.2f} GB", flush=True)
./experiments/runners/experiment_a.py:142:                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
./experiments/runners/experiment_a.py:147:                f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
./experiments/runners/experiment_a.py:168:        print("WARNING: No CUDA device found. This experiment is designed for GPU.")
./experiments/runners/clustered_taxi_experiment.py:116:# PART 1: KERNELS (GPU)
./experiments/runners/clustered_taxi_experiment.py:142:class FastGPUMultiLevelClustering:
./experiments/runners/clustered_taxi_experiment.py:223:class GPUClusteredSolver:
./experiments/runners/clustered_taxi_experiment.py:257:        cluster_engine = FastGPUMultiLevelClustering(epsilon, k=k, batch_size=self.batch_size)
./experiments/runners/clustered_taxi_experiment.py:619:    solver = GPUClusteredSolver(
./experiments/runners/glove_recall_experiment.py:62:def _download_with_progress(url: str, dest: pathlib.Path) -> None:
./experiments/runners/glove_recall_experiment.py:66:    def _progress(block_count, block_size, total_size):
./experiments/runners/glove_recall_experiment.py:77:        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
./experiments/runners/glove_recall_experiment.py:91:    _download_with_progress(GLOVE_URL, path)
./experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:32:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:33:from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
./experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:213:    solver = SimpleGPUSolver(None, None, epsilon=EPSILON, batch_size=BATCH_SIZE,
./experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:230:    solver = ThreeLevelGPUSolver(None, None, epsilon=EPSILON, batch_size=BATCH_SIZE,
./experiments/runners/experiment_runner.py:27:        print("[!] CUDA not available. This experiment requires a GPU.")
./experiments/runners/sift_recall_experiment.py:168:        query = queries[q_idx : q_idx + 1]                      # (1, D) on GPU
./experiments/runners/sift_recall_experiment.py:326:    # move to GPU
./experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:31:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:32:from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
./experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:229:    solver = SimpleGPUSolver(None, None, epsilon=EPSILON, batch_size=BATCH_SIZE,
./experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:246:    solver = ThreeLevelGPUSolver(None, None, epsilon=EPSILON, batch_size=BATCH_SIZE,
./experiments/runners/experiment_8.py:24:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/experiment_8.py:194:        solver = SimpleGPUSolver(
./experiments/runners/experiment_8.py:212:        solver = SimpleGPUSolver(
./experiments/runners/e2_color_aware_vs_exact.py:130:def peak_memory_allocated_mb(device):
./experiments/runners/e2_color_aware_vs_exact.py:133:    return torch.cuda.max_memory_allocated() / (1024**2)
./experiments/runners/e2_color_aware_vs_exact.py:179:               "total_time_s", "cluster_time_s", "solver_time_s", "peak_gpu_mem_mb",
./experiments/runners/e2_color_aware_vs_exact.py:256:                peak_mem = peak_memory_allocated_mb(device)
./experiments/runners/sift_benchmark.py:114:    # Batch size controls GPU memory usage during construction
./experiments/runners/sift_benchmark.py:119:        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
./experiments/runners/sift_benchmark.py:137:    # GPU warm-up (not measured)
./experiments/runners/e3_scalability_synthetic.py:36:               "total_time_s", "cluster_time_s", "solver_time_s", "peak_gpu_mem_mb", "cost"]
./experiments/runners/e3_scalability_synthetic.py:72:            peak_mem = torch.cuda.max_memory_allocated()/(1024**2)
./experiments/runners/e3_scalability_synthetic.py:119:            peak_memK = torch.cuda.max_memory_allocated()/(1024**2)
./experiments/runners/e3_scalability_synthetic_klevel.py:35:               "total_time_s", "cluster_time_s", "solver_time_s", "peak_gpu_mem_mb", "cost"]
./experiments/runners/e3_scalability_synthetic_klevel.py:69:            peak_memK = torch.cuda.max_memory_allocated()/(1024**2)
./experiments/runners/e4_ratio_distribution.py:23:  - kNN via torch.cdist on GPU: (N x N) output only, no (N,N,D) intermediate.
./experiments/runners/experiment.py:5:# CRITICAL: Prevent JAX from hogging all GPU memory, allowing PyTorch to run too.
./experiments/runners/experiment.py:78:        return torch.cuda.max_memory_allocated() / (1024 ** 2)
./experiments/runners/experiment.py:93:    Fast, JIT-compiled, GPU-native.
./experiments/runners/e1_mnist_profile.py:114:def peak_memory_allocated_mb(device):
./experiments/runners/e1_mnist_profile.py:117:    return torch.cuda.max_memory_allocated() / (1024 ** 2)
./experiments/runners/e1_mnist_profile.py:183:               "total_time_s", "cluster_time_s", "solver_time_s", "peak_gpu_mem_mb",
./experiments/runners/e1_mnist_profile.py:246:                peak_mem = peak_memory_allocated_mb(device)
./experiments/runners/e1_mnist_profile.py:294:                peak_memK = peak_memory_allocated_mb(device)
./experiments/runners/final/experiment_scalability_three_level.py:19:from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
./experiments/runners/final/experiment_scalability_three_level.py:33:def gpu_mem_mb() -> tuple[float, float]:
./experiments/runners/final/experiment_scalability_three_level.py:38:        torch.cuda.memory_allocated() / 1024**2,
./experiments/runners/final/experiment_scalability_three_level.py:39:        torch.cuda.memory_reserved() / 1024**2,
./experiments/runners/final/experiment_scalability_three_level.py:43:def gpu_mem_str() -> str:
./experiments/runners/final/experiment_scalability_three_level.py:44:    alloc, res = gpu_mem_mb()
./experiments/runners/final/experiment_scalability_three_level.py:95:    alloc, res = gpu_mem_mb()
./experiments/runners/final/experiment_scalability_three_level.py:167:        solver = ThreeLevelGPUSolver(
./experiments/runners/final/experiment_scalability_three_level.py:253:        alloc_cleanup, res_cleanup = gpu_mem_mb()
./experiments/runners/final/experiment_sift.py:24:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/final/experiment_sift.py:150:        solver = SimpleGPUSolver(
./experiments/runners/final/experiment_sift.py:169:        solver = SimpleGPUSolver(
./experiments/runners/experiment_pot.py:11:# Import the GPU OT solvers (assumes these modules are in the same directory or installed)
./experiments/runners/experiment_pot.py:82:        # 4. Solve with Two-Level Push-Relabel (approximate OT on GPU)
./experiments/runners/experiment_pot.py:114:        # Clean up solver objects to free GPU memory
./experiments/runners/final/experiment_emnist_proxy.py:157:def clustering_gpu_bytes(clustering):
./experiments/runners/final/experiment_emnist_proxy.py:202:    struct_bytes = clustering_gpu_bytes(clustering)
./experiments/runners/final/experiment_emnist_proxy.py:262:        return f"{value / (1024.0 ** 2):.1f} MiB"
./experiments/runners/final/experiment_emnist_proxy.py:263:    return f"{value / (1024.0 ** 3):.2f} GiB"
./experiments/runners/final/experiment_emnist_proxy.py:281:        ("GPU Struct Size", col_widths["struct_size"], ">"),
./experiments/runners/experiment_10.py:22:from clustered_push_relabel.solvers.simple_bipartite_2 import SimpleGPUSolver2
./experiments/runners/experiment_10.py:47:    diameter is a Python float for passing to SimpleGPUSolver2.
./experiments/runners/experiment_10.py:84:    NOTE: Allocates an (N x N) float32 tensor on GPU. Only valid for N up to
./experiments/runners/experiment_10.py:270:        solver = SimpleGPUSolver2(
./experiments/runners/experiment_10.py:285:    solver = SimpleGPUSolver2(
./experiments/runners/final/experiment_nyc.py:15:def gpu_mem(device, label):
./experiments/runners/final/experiment_nyc.py:17:        alloc = torch.cuda.memory_allocated(device) / 1024**3
./experiments/runners/final/experiment_nyc.py:18:        res   = torch.cuda.memory_reserved(device)  / 1024**3
./experiments/runners/final/experiment_nyc.py:33:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/final/experiment_nyc.py:207:    Time the full wall time (clustering + solve) for SimpleGPUSolver.
./experiments/runners/final/experiment_nyc.py:213:        gpu_mem(device, "before init")
./experiments/runners/final/experiment_nyc.py:214:        solver = SimpleGPUSolver(
./experiments/runners/final/experiment_nyc.py:221:        gpu_mem(device, "after init / before solve")
./experiments/runners/final/experiment_nyc.py:224:        gpu_mem(device, "after solve")
./experiments/runners/final/experiment_nyc.py:233:        gpu_mem(device, "before init")
./experiments/runners/final/experiment_nyc.py:234:        solver = SimpleGPUSolver(
./experiments/runners/final/experiment_nyc.py:241:        gpu_mem(device, "after init / before solve")
./experiments/runners/final/experiment_nyc.py:245:        gpu_mem(device, "after solve")
./experiments/runners/final/experiment_nyc.py:349:        description="Benchmark SimpleGPUSolver vs exact OT on NYC yellow taxi data."
./experiments/runners/final/experiment_emnist.py:26:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/final/experiment_emnist.py:185:        solver = SimpleGPUSolver(
./experiments/runners/final/experiment_emnist.py:203:        solver = SimpleGPUSolver(
./experiments/runners/final/download_glove.py:27:def make_progress_callback():
./experiments/runners/final/download_glove.py:39:                f"  Download progress: {state['last_printed']}%",
./experiments/runners/final/download_glove.py:59:            word, sep, values = line.partition(" ")
./experiments/runners/final/download_glove.py:90:            word, _, _ = line.partition(" ")
./experiments/runners/final/download_glove.py:105:        urllib.request.urlretrieve(GLOVE_URL, ZIP_PATH, reporthook=make_progress_callback())
./experiments/runners/final/experiment_sift_three_level.py:24:from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
./experiments/runners/final/experiment_sift_three_level.py:148:        solver = ThreeLevelGPUSolver(
./experiments/runners/final/experiment_sift_three_level.py:168:        solver = ThreeLevelGPUSolver(
./experiments/runners/final/experiment_glove_proxy.py:164:def benchmark_proxy_exact(P_red_gpu, P_blue_gpu, device):
./experiments/runners/final/experiment_glove_proxy.py:165:    if P_red_gpu.shape[0] > 10_000:
./experiments/runners/final/experiment_glove_proxy.py:170:    n = P_red_gpu.shape[0]
./experiments/runners/final/experiment_glove_proxy.py:173:    red_cpu = P_red_gpu.detach().cpu()
./experiments/runners/final/experiment_glove_proxy.py:174:    blue_cpu = P_blue_gpu.detach().cpu()
./experiments/runners/final/experiment_glove_proxy.py:186:        clustering = cluster_engine.run(P_red_gpu, P_blue_gpu)
./experiments/runners/final/experiment_glove_proxy.py:200:        clustering = cluster_engine.run(P_red_gpu, P_blue_gpu)
./experiments/runners/final/experiment_glove_proxy.py:237:def run_proxy_exact(P_red_gpu, P_blue_gpu, device):
./experiments/runners/final/experiment_glove_proxy.py:239:        time_ms, cost = benchmark_proxy_exact(P_red_gpu, P_blue_gpu, device)
./experiments/runners/final/experiment_glove_proxy.py:336:            P_red_gpu = P_red_cpu.to(device)
./experiments/runners/final/experiment_glove_proxy.py:337:            P_blue_gpu = P_blue_cpu.to(device)
./experiments/runners/final/experiment_glove_proxy.py:346:        proxy_result = run_proxy_exact(P_red_gpu, P_blue_gpu, device)
./experiments/runners/final/experiment_glove_proxy.py:350:        del P_red_cpu, P_blue_cpu, P_red_gpu, P_blue_gpu
./experiments/runners/final/experiment_emnist_three_level.py:28:from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
./experiments/runners/final/experiment_emnist_three_level.py:187:        solver = ThreeLevelGPUSolver(
./experiments/runners/final/experiment_emnist_three_level.py:206:        solver = ThreeLevelGPUSolver(
./experiments/runners/final/experiment_newsgroups_proxy.py:149:    SIFT keypoint sets. Reduce `tile_a` if this runs out of GPU memory.
./experiments/runners/final/experiment_glove.py:24:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/final/experiment_glove.py:144:        solver = SimpleGPUSolver(
./experiments/runners/final/experiment_glove.py:164:        solver = SimpleGPUSolver(
./experiments/runners/final/experiment_newsgroups.py:28:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/final/experiment_newsgroups.py:125:    SIFT keypoint sets. Reduce `tile_a` if this runs out of GPU memory.
./experiments/runners/final/experiment_newsgroups.py:418:    then run SimpleGPUSolver with the precomputed clustering.
./experiments/runners/final/experiment_newsgroups.py:432:    solver = SimpleGPUSolver(
./experiments/runners/final/experiment_cifar_sift.py:28:from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
./experiments/runners/final/experiment_cifar_sift.py:113:    Reduce `tile_a` if this runs out of GPU memory.
./experiments/runners/final/experiment_cifar_sift.py:184:    Reduce `tile` if this runs out of GPU memory.
./experiments/runners/final/experiment_cifar_sift.py:363:    then run SimpleGPUSolver with the precomputed clustering.
./experiments/runners/final/experiment_cifar_sift.py:377:    solver = SimpleGPUSolver(
./experiments/runners/final/experiment_cifar_sift_proxy.py:147:    Reduce `tile_a` if this runs out of GPU memory.
./experiments/runners/final/experiment_cifar_sift_proxy.py:218:    Reduce `tile` if this runs out of GPU memory.
./experiments/runners/final/experiment_emnist_three_level_proxy.py:125:def clustering_gpu_bytes(clustering):
./experiments/runners/final/experiment_emnist_three_level_proxy.py:169:    # Pre-group blues by their nearest A1 center so the inner loop does no GPU
./experiments/runners/final/experiment_emnist_three_level_proxy.py:170:    # nonzero calls — only cheap CPU-side ptr slicing + vectorised GPU scatter.
./experiments/runners/final/experiment_emnist_three_level_proxy.py:253:    struct_bytes = clustering_gpu_bytes(clustering)
./experiments/runners/final/experiment_emnist_three_level_proxy.py:311:        return f"{value / (1024.0 ** 2):.1f} MiB"
./experiments/runners/final/experiment_emnist_three_level_proxy.py:312:    return f"{value / (1024.0 ** 3):.2f} GiB"
./experiments/runners/final/experiment_emnist_three_level_proxy.py:332:        ("GPU Struct Size",       col_widths["struct_size"], ">"),
```

## Section 7 — Results Directory

`experiments/runners/final2/results/` exists. Current full listing:

```text
experiments/runners/final2/results
```

No markdown results files exist yet in `experiments/runners/final2/results/`.

## Section 8 — Data Directory

### final2 MNIST DATA_DIR: `/Users/anish/Developer/NCSU/RP2/experiments/runners/final2/data`

Directory does not exist.

### BASE_DIR/data for EMNIST and descriptor datasets: `/Users/anish/Developer/NCSU/RP2/data`

Directory does not exist.

### NYC default relative data directory from repo root: `/Users/anish/Developer/NCSU/RP2/nyc_data`

Directory does not exist.

Dataset readiness: MNIST in `experiments/runners/final2/data` is not downloaded because that directory is absent; EMNIST in `data/` is not downloaded because `data/` is absent; CIFAR SIFT at `data/cifar_sift/cifar10_sift_train.pkl.gz` is not present; 20 Newsgroups GloVe at `data/newsgroups_glove/newsgroups_embeddings.pkl.gz` is not present; NYC taxi default file `nyc_data/yellow_tripdata_2014-01.parquet` is not present.

## Section 9 — Gamma (γ) Computation

Search output for transport-plan access, local/non-local/gamma-like terms:

```text
experiments/runners/final2/shared.py:83:def _build_csr_adjacency(dist_mat, threshold_per_row, N_rows, N_cols,
experiments/runners/final2/shared.py:86:    Build CSR adjacency where entry (i, j) is included iff
experiments/runners/final2/shared.py:161:    local_idx_A2 = mask_A2.nonzero(as_tuple=True)[0]
experiments/runners/final2/shared.py:162:    sampled_idx_A2 = sampled_idx_A1[local_idx_A2]
experiments/runners/final2/shared.py:190:    adj_B_ptr, adj_B_col, adj_B_dist_int, adj_B_dist_float = _build_csr_adjacency(
experiments/runners/final2/shared.py:195:    adj_A1_ptr, adj_A1_col, adj_A1_dist_int, adj_A1_dist_float = _build_csr_adjacency(
experiments/runners/final2/shared.py:216:        "adj_B_ptr": adj_B_ptr,
experiments/runners/final2/shared.py:217:        "adj_B_col": adj_B_col,
experiments/runners/final2/shared.py:218:        "adj_B_dist_int": adj_B_dist_int,
experiments/runners/final2/shared.py:219:        "adj_B_dist_float": adj_B_dist_float,
experiments/runners/final2/shared.py:220:        "adj_A1_ptr": adj_A1_ptr,
experiments/runners/final2/shared.py:221:        "adj_A1_col": adj_A1_col,
experiments/runners/final2/shared.py:222:        "adj_A1_dist_int": adj_A1_dist_int,
experiments/runners/final2/shared.py:223:        "adj_A1_dist_float": adj_A1_dist_float,
experiments/runners/final2/shared.py:236:    Returns (N, N) float64 numpy array for ot.emd().
experiments/runners/final2/shared.py:243:    adj_B_ptr = clustering["adj_B_ptr"]
experiments/runners/final2/shared.py:244:    adj_B_col = clustering["adj_B_col"]
experiments/runners/final2/shared.py:245:    adj_B_dist_float = clustering["adj_B_dist_float"]
experiments/runners/final2/shared.py:246:    adj_A1_ptr = clustering["adj_A1_ptr"]
experiments/runners/final2/shared.py:247:    adj_A1_col = clustering["adj_A1_col"]
experiments/runners/final2/shared.py:248:    adj_A1_dist_float = clustering["adj_A1_dist_float"]
experiments/runners/final2/shared.py:249:    S1 = int(adj_A1_ptr.shape[0]) - 1
experiments/runners/final2/shared.py:261:        a1_s = int(adj_A1_ptr[a1_i].item())
experiments/runners/final2/shared.py:262:        a1_e = int(adj_A1_ptr[a1_i + 1].item())
experiments/runners/final2/shared.py:265:        a_cols = adj_A1_col[a1_s:a1_e]
experiments/runners/final2/shared.py:266:        a_dists = adj_A1_dist_float[a1_s:a1_e]
experiments/runners/final2/shared.py:277:    if adj_B_col.numel() > 0:
experiments/runners/final2/shared.py:280:            adj_B_ptr[1:] - adj_B_ptr[:-1],
experiments/runners/final2/shared.py:282:        C[b_idx, adj_B_col] = adj_B_dist_float
experiments/runners/final2/shared.py:298:    adj_ptr = clustering["adj_ptr"]
experiments/runners/final2/shared.py:299:    adj_col = clustering["adj_col"]
experiments/runners/final2/shared.py:300:    adj_dist_float = clustering["adj_dist_float"]
experiments/runners/final2/shared.py:303:    if adj_col.numel() > 0:
experiments/runners/final2/shared.py:306:            adj_ptr[1:] - adj_ptr[:-1],
experiments/runners/final2/shared.py:308:        C[b_idx, adj_col] = adj_dist_float
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:139:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:141:    match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:158:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:160:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:177:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:179:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:133:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:135:    match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:152:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:154:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:171:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:173:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:135:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:137:    match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:156:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:158:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:175:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:177:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp05_nyc_scalability.py:111:            df[tc] = df[tc].dt.tz_localize("America/New_York")
experiments/runners/final2/experiments/exp05_nyc_scalability.py:178:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp05_nyc_scalability.py:180:    match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:131:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:133:    match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:150:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:152:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:169:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:171:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:216:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:218:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:235:    plan = ot.emd(a, b, C.T, numItermax=10**6)
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:237:    match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:254:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:256:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:202:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:204:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:200:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:202:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:220:    plan = ot.emd(a, b, C.T, numItermax=10**6)
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:222:    match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:239:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:241:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:218:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:220:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:178:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:180:    match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:197:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:199:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:216:    plan = ot.emd(a, b, C, numItermax=10**6)
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:218:    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
```

Conclusion: no code in `experiments/runners/final2/` computes a quantity named `gamma`, `non_local`, `local_mass`, or `local_cost`. No code checks whether matched red points are in a blue point adjacency list. All `plan = ot.emd(...)` uses in the experiment files access the full plan only via `plan.argmax(...)` to derive a hard matching; no code sums transport-plan masses, iterates nonzero transport entries, or computes local/non-local transport mass. `shared.py` builds adjacency/proxy matrices (`adj_B_*`, `adj_A1_*`, `adj_*`) but does not inspect matched pairs from an OT plan.

## Section 10 — Multi-Trial Infrastructure

Search output for trial/seed/aggregation terms:

```text
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:107:    rng_r = np.random.RandomState(seed)
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:108:    rng_b = np.random.RandomState(seed + 1)
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:142:    cost = (blue.cpu() - red.cpu()[match]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:161:    cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:180:    cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:101:    rng_r = np.random.RandomState(seed)
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:102:    rng_b = np.random.RandomState(seed + 1)
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:136:    cost = (blue.cpu() - red.cpu()[match]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:155:    cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:174:    cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:87:    rng_r = np.random.RandomState(seed)
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:88:    rng_b = np.random.RandomState(seed + 1)
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:140:    cost = (bc - rc[match]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:159:    cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:178:    cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp05_nyc_scalability.py:165:    return torch.norm(blue - matched, p=2, dim=1).mean().item()
experiments/runners/final2/experiments/exp05_nyc_scalability.py:251:    rng = np.random.default_rng(SEED)
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:88:    rng_r = np.random.RandomState(seed)
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:89:    rng_b = np.random.RandomState(seed + 1)
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:134:    cost = (blue.cpu() - red.cpu()[match]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:153:    cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:172:    cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:93:    rng = np.random.RandomState(seed)
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:219:    cost = D_cpu[torch.arange(n), match].mean().item() * diameter
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:238:    cost = D_cpu[torch.arange(n), match].mean().item() * diameter
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:257:    cost = D_cpu[torch.arange(n), match].mean().item() * diameter
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:98:    rng = np.random.RandomState(seed)
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:205:    cost = D_cpu[torch.arange(n), match].mean().item() * diameter
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:220:    cost = D_cpu[torch.arange(n), solver.match_B.cpu()].mean().item() * diameter
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:237:    cost = D_cpu[torch.arange(n), solver.match_B.cpu()].mean().item() * diameter
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:94:    rng = np.random.RandomState(seed)
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:203:    cost = D_cpu[torch.arange(n), match].mean().item() * diameter
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:223:    cost = D_cpu[torch.arange(n), match].mean().item() * diameter
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:242:    cost = D_cpu[torch.arange(n), match].mean().item() * diameter
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:98:    rng = np.random.RandomState(seed)
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:221:    cost = D_cpu[torch.arange(n), match].mean().item() * diameter
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:236:    cost = D_cpu[torch.arange(n), solver.match_B.cpu()].mean().item() * diameter
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:253:    cost = D_cpu[torch.arange(n), solver.match_B.cpu()].mean().item() * diameter
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:131:    rng_r = np.random.RandomState(seed)
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:132:    rng_b = np.random.RandomState(seed + 1)
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:156:        "frac_gt_1": float((values > 1.0).mean()),
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:158:        "cost_median": float(np.median(values)),
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:165:    return elapsed, float(pair_costs.mean().item()), stats
```

Conclusion: no multi-trial infrastructure exists in `experiments/runners/final2/`. The files use fixed `SEED = 42` and construct RNGs for single samples. There is no `n_trials`, `num_trials`, `seeds` list, repeated-run loop, or aggregation across trials using mean/std/percentiles. The only `np.median` occurrence is exp10 per-pair cost diagnostics, not trial aggregation.

## Section 11 — Peak Memory Tracking

Search command: `rg -n -C 5 "max_memory_allocated|memory_reserved|memory_allocated|reset_peak_memory_stats|max_memory_reserved|peak" experiments/runners/final2`. Output:

```text
No matches found.
```

Conclusion: no call to `torch.cuda.max_memory_allocated()`, `torch.cuda.memory_reserved()`, `torch.cuda.memory_allocated()`, `torch.cuda.max_memory_reserved()`, or `torch.cuda.reset_peak_memory_stats()` exists inside `experiments/runners/final2/`. No peak memory is recorded in result rows or markdown.

## Section 12 — sample_factor / Landmark Density

Exact clustering-engine instantiation and `sample_factor` search lines:

```text
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:151:    engine = SimpleL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:170:    engine = ThreeLevelL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:145:    engine = SimpleL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:164:    engine = ThreeLevelL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:149:    engine = SimpleL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:168:    engine = ThreeLevelL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:143:    engine = SimpleL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:162:    engine = ThreeLevelL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:229:    c = SimplePrecomputedClustering(epsilon=EPSILON, tile_size=BATCH_SIZE).run(D_rr_norm, D_br_norm)
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:210:    c = SimplePrecomputedClustering(epsilon=EPSILON, tile_size=BATCH_SIZE).run(D_rr_norm, D_br_norm)
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:213:    engine = SimplePrecomputedClustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:226:    c = SimplePrecomputedClustering(epsilon=EPSILON, tile_size=BATCH_SIZE).run(D_rr_norm, D_br_norm)
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:190:    engine = SimpleL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:209:    engine = ThreeLevelL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
```

Conclusion: no experiment in `experiments/runners/final2/experiments/` passes `sample_factor` to a clustering engine constructor. No experiment varies `sample_factor` across multiple values. All clustering constructor calls rely on each class default `sample_factor=1.0` where that parameter exists.

## Section 13 — Proxy Matrix Builders

### `build_two_level_proxy_matrix` complete implementation

```python
def build_two_level_proxy_matrix(clustering, N, device):
    """
    Build N×N proxy cost matrix from SimplePrecomputedClustering output.
    For each (b, a): if a ∈ adj(b) use exact dist; else d_min_b[b] + DR[nearest_s[b], a].
    Returns float64 numpy array.
    """
    DR = clustering["DR"]
    d_min_b = clustering["d_min_b"]
    nearest_s = clustering["nearest_s"]
    adj_ptr = clustering["adj_ptr"]
    adj_col = clustering["adj_col"]
    adj_dist_float = clustering["adj_dist_float"]

    C = d_min_b.unsqueeze(1) + DR[nearest_s, :]
    if adj_col.numel() > 0:
        b_idx = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            adj_ptr[1:] - adj_ptr[:-1],
        )
        C[b_idx, adj_col] = adj_dist_float
    return C.cpu().to(torch.float64).numpy()
```

Keys read from clustering dict: `DR`, `d_min_b`, `nearest_s`, `adj_ptr`, `adj_col`, `adj_dist_float`. It returns `C.cpu().to(torch.float64).numpy()`, a NumPy `float64` array with shape `(N, N)`. Before conversion `C` is a torch tensor on `device` with the dtype produced by the clustering distances, normally `torch.float32`.

### `build_three_level_proxy_matrix` complete implementation

```python
def build_three_level_proxy_matrix(clustering, N, device):
    """
    Build the full N×N proxy cost matrix from a three-level clustering dict.

    Proxy priority (each level overwrites the coarser estimate):
      Level 2 (base):  C[b,a] = d_min_b_A2[b] + DR[nearest_s2[b], a]
      Level 1 (refine):C[b,a] = d_min_b_A1[b] + d(s1_b, a)  for a ∈ Adj_A1(s1_b)
      Level 0 (direct):C[b,a] = d(b, a)                      for a ∈ Adj_B(b)

    Returns (N, N) float64 numpy array for ot.emd().
    """
    DR = clustering["DR"]
    d_min_b_A2 = clustering["d_min_b_A2"]
    nearest_s2 = clustering["nearest_s2"]
    d_min_b_A1 = clustering["d_min_b_A1"]
    nearest_s1 = clustering["nearest_s1"]
    adj_B_ptr = clustering["adj_B_ptr"]
    adj_B_col = clustering["adj_B_col"]
    adj_B_dist_float = clustering["adj_B_dist_float"]
    adj_A1_ptr = clustering["adj_A1_ptr"]
    adj_A1_col = clustering["adj_A1_col"]
    adj_A1_dist_float = clustering["adj_A1_dist_float"]
    S1 = int(adj_A1_ptr.shape[0]) - 1

    # Level 2 base
    C = d_min_b_A2.unsqueeze(1) + DR[nearest_s2, :]   # (N, N) float32

    # Level 1 — group blues by nearest A1 center, scatter-overwrite
    sorted_order = torch.argsort(nearest_s1)
    group_counts = torch.bincount(nearest_s1, minlength=S1)
    group_ptr = torch.zeros(S1 + 1, dtype=torch.long, device=device)
    group_ptr[1:] = group_counts.cumsum(0)

    for a1_i in range(S1):
        a1_s = int(adj_A1_ptr[a1_i].item())
        a1_e = int(adj_A1_ptr[a1_i + 1].item())
        if a1_s == a1_e:
            continue
        a_cols = adj_A1_col[a1_s:a1_e]
        a_dists = adj_A1_dist_float[a1_s:a1_e]
        g_s = int(group_ptr[a1_i].item())
        g_e = int(group_ptr[a1_i + 1].item())
        if g_s == g_e:
            continue
        blues = sorted_order[g_s:g_e]
        C[blues.unsqueeze(1), a_cols.unsqueeze(0)] = (
            d_min_b_A1[blues].unsqueeze(1) + a_dists.unsqueeze(0)
        )

    # Level 0 — direct
    if adj_B_col.numel() > 0:
        b_idx = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            adj_B_ptr[1:] - adj_B_ptr[:-1],
        )
        C[b_idx, adj_B_col] = adj_B_dist_float

    return C.cpu().to(torch.float64).numpy()
```

Keys read from clustering dict: `DR`, `d_min_b_A2`, `nearest_s2`, `d_min_b_A1`, `nearest_s1`, `adj_B_ptr`, `adj_B_col`, `adj_B_dist_float`, `adj_A1_ptr`, `adj_A1_col`, `adj_A1_dist_float`. It returns `C.cpu().to(torch.float64).numpy()`, a NumPy `float64` array with shape `(N, N)`. Before conversion `C` is a torch tensor on `device` with the dtype produced by the clustering distances, normally `torch.float32`.

Other `build_*_proxy_matrix` or `build_*_cost_matrix` functions in `shared.py`: none. Search output:

```text
227:def build_three_level_proxy_matrix(clustering, N, device):
289:def build_two_level_proxy_matrix(clustering, N, device):
```

## Section 14 — EMNIST Experiments

### `experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py`

- Full file path: `experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py`
- `N_VALUES` = `[5000, 10000, 15000, 20000, 25000]`
- `EXACT_N_LIMIT` = `25000`
- Sampling: Equal sampling from all classes present in EMNIST `byclass`. It computes `classes = np.unique(labels)` and `spc = n_samples // len(classes)`. With 62 classes, configured `N_VALUES` are not divisible by 62, so each returned side has `62 * floor(n_samples / 62)` rows, not necessarily exactly `n_samples`.
Exact data loading function `load_emnist_equal` in full:
```python
def load_emnist_equal(n_samples, seed, split=EMNIST_SPLIT):
    train = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=True,  download=False)
    test  = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    classes = np.unique(labels)
    spc = n_samples // len(classes)
    if spc == 0:
        raise ValueError(f"n_samples={n_samples} too small for {len(classes)} EMNIST classes")

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_parts, blue_parts = [], []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        needed = 2 * spc
        if idx.size < needed:
            warnings.warn(f"Class {cls}: {idx.size} available, need {needed}. Skipping.")
            continue
        rng_r.shuffle(idx)
        chosen = idx[:needed]
        red_parts.append(images[chosen[:spc]])
        blue_parts.append(images[chosen[spc:needed]])

    red  = np.concatenate(red_parts).astype(np.float32)  / 255.0
    blue = np.concatenate(blue_parts).astype(np.float32) / 255.0
    for arr in (red, blue):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each EMNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s

    return torch.from_numpy(red), torch.from_numpy(blue)
```

### `experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py`

- Full file path: `experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py`
- `N_VALUES` = `[5000, 10000, 15000, 20000, 25000]`
- `EXACT_N_LIMIT` = `25000`
- Sampling: Biased EMNIST byclass sampling. `BLUE_CLASS_END = 31`, so blue classes are `[c for c in all_classes if c < 31]` (0-30). `RED_CLASS_START = 31`, so red classes are `[c for c in all_classes if c >= 31]` (31-61). `_sample_classes` samples `spc = n_total // len(classes)` per class.
Exact data loading function `load_emnist_biased` in full:
```python
def load_emnist_biased(n_samples, seed, split=EMNIST_SPLIT):
    train = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=True,  download=False)
    test  = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    all_classes = np.unique(labels).tolist()
    blue_classes = [c for c in all_classes if c < BLUE_CLASS_END]
    red_classes  = [c for c in all_classes if c >= RED_CLASS_START]

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_arr  = _sample_classes(images, labels, red_classes,  n_samples, rng_r).astype(np.float32) / 255.0
    blue_arr = _sample_classes(images, labels, blue_classes, n_samples, rng_b).astype(np.float32) / 255.0

    for arr in (red_arr, blue_arr):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each EMNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s

    return torch.from_numpy(red_arr), torch.from_numpy(blue_arr)
```

## Section 15 — Distortion Ratio Reporting

Exact ratio-related lines:

```text
experiments/runners/final2/shared.py-57-
experiments/runners/final2/shared.py-58-
experiments/runners/final2/shared.py:59:def compute_ratio(exact, approx):
experiments/runners/final2/shared.py-60-    if is_nan(exact) or is_nan(approx) or float(exact) == 0.0:
experiments/runners/final2/shared.py-61-        return math.nan
--
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-33-from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-34-from shared import (
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:35:    compute_ratio, fmt_time, fmt_cost, fmt_ratio,
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-36-    build_two_level_proxy_matrix, build_three_level_proxy_matrix,
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-37-)
--
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-61-    ("2L-Prx Cost", 12),
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-62-    ("3L-Prx Cost", 12),
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:63:    ("2L Ratio",     9),
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:64:    ("3L Ratio",     9),
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-65-]
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-66-
--
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-73-    "2L-Prx Cost":  lambda r: fmt_cost(r["prx2"]["cost"]),
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-74-    "3L-Prx Cost":  lambda r: fmt_cost(r["prx3"]["cost"]),
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:75:    "2L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx2"]["cost"])),
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:76:    "3L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx3"]["cost"])),
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-77-}
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-78-
--
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-235-        e, p2, p3 = r["exact"], r["prx2"], r["prx3"]
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py-236-        print(f"N={r['n']:>7,} | exact={fmt_time(e['time_ms'])} cost={fmt_cost(e['cost'])} "
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:237:              f"| 2L ratio={fmt_ratio(compute_ratio(e['cost'], p2['cost']))} "
experiments/runners/final2/experiments/exp04_emnist_proxy_biased.py:238:              f"| 3L ratio={fmt_ratio(compute_ratio(e['cost'], p3['cost']))}")
--
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-33-from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-34-from shared import (
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:35:    compute_ratio, fmt_time, fmt_cost, fmt_ratio,
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-36-    build_two_level_proxy_matrix, build_three_level_proxy_matrix,
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-37-)
--
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-59-    ("2L-Prx Cost", 12),
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-60-    ("3L-Prx Cost", 12),
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:61:    ("2L Ratio",     9),
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:62:    ("3L Ratio",     9),
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-63-]
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-64-
--
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-71-    "2L-Prx Cost":  lambda r: fmt_cost(r["prx2"]["cost"]),
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-72-    "3L-Prx Cost":  lambda r: fmt_cost(r["prx3"]["cost"]),
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:73:    "2L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx2"]["cost"])),
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:74:    "3L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx3"]["cost"])),
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-75-}
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-76-
--
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-228-    for r in results:
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-229-        e, p2, p3 = r["exact"], r["prx2"], r["prx3"]
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:230:        r2 = compute_ratio(e["cost"], p2["cost"])
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:231:        r3 = compute_ratio(e["cost"], p3["cost"])
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py-232-        print(f"N={r['n']:>6,} | exact={fmt_time(e['time_ms'])} cost={fmt_cost(e['cost'])} "
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:233:              f"| 2L={fmt_time(p2['time_ms'])} cost={fmt_cost(p2['cost'])} ratio={fmt_ratio(r2)} "
experiments/runners/final2/experiments/exp02_mnist_proxy_biased.py:234:              f"| 3L={fmt_time(p3['time_ms'])} cost={fmt_cost(p3['cost'])} ratio={fmt_ratio(r3)}")
--
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-33-from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-34-from shared import (
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:35:    compute_ratio, fmt_time, fmt_cost, fmt_ratio,
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-36-    build_two_level_proxy_matrix, build_three_level_proxy_matrix,
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-37-    run_three_level_precomputed,
--
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-57-    ("2L-Prx Cost", 12),
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-58-    ("3L-Prx Cost", 12),
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:59:    ("2L Ratio",     9),
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:60:    ("3L Ratio",     9),
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-61-]
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-62-
--
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-69-    "2L-Prx Cost":  lambda r: fmt_cost(r["prx2"]["cost"]),
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-70-    "3L-Prx Cost":  lambda r: fmt_cost(r["prx3"]["cost"]),
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:71:    "2L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx2"]["cost"])),
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:72:    "3L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx3"]["cost"])),
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-73-}
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-74-
--
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-228-def _fmt_row_terminal(row):
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-229-    e, p2, p3 = row["exact"], row["prx2"], row["prx3"]
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:230:    r2 = compute_ratio(e["cost"], p2["cost"])
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:231:    r3 = compute_ratio(e["cost"], p3["cost"])
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-232-    return [
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-233-        f"{row['n']:>7,}",
--
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-246-    headers = ["       N", "    Exact Time", "  2L-Prx Time",
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-247-               "  3L-Prx Time", "  Exact Cost", " 2L-Prx Cost",
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py:248:               " 3L-Prx Cost", " 2L Ratio", " 3L Ratio"]
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-249-    sep = "-+-".join("-" * len(h) for h in headers)
experiments/runners/final2/experiments/exp01_mnist_proxy_equal.py-250-    print("\n" + " | ".join(headers))
--
experiments/runners/final2/experiments/exp05_nyc_scalability.py-36-from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
experiments/runners/final2/experiments/exp05_nyc_scalability.py-37-from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
experiments/runners/final2/experiments/exp05_nyc_scalability.py:38:from shared import compute_ratio, fmt_time, fmt_cost, fmt_ratio, fmt_iters
experiments/runners/final2/experiments/exp05_nyc_scalability.py-39-
experiments/runners/final2/experiments/exp05_nyc_scalability.py-40-EXP_ID = 5
--
experiments/runners/final2/experiments/exp05_nyc_scalability.py-68-    ("2L Cost",    12),
experiments/runners/final2/experiments/exp05_nyc_scalability.py-69-    ("3L Cost",    12),
experiments/runners/final2/experiments/exp05_nyc_scalability.py:70:    ("2L Ratio",    9),
experiments/runners/final2/experiments/exp05_nyc_scalability.py:71:    ("3L Ratio",    9),
experiments/runners/final2/experiments/exp05_nyc_scalability.py-72-    ("2L Iters",   10),
experiments/runners/final2/experiments/exp05_nyc_scalability.py-73-    ("3L Iters",   10),
--
experiments/runners/final2/experiments/exp05_nyc_scalability.py-82-    "2L Cost":    lambda r: fmt_cost(r["sol2"]["cost"]),
experiments/runners/final2/experiments/exp05_nyc_scalability.py-83-    "3L Cost":    lambda r: fmt_cost(r["sol3"]["cost"]),
experiments/runners/final2/experiments/exp05_nyc_scalability.py:84:    "2L Ratio":   lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["sol2"]["cost"])),
experiments/runners/final2/experiments/exp05_nyc_scalability.py:85:    "3L Ratio":   lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["sol3"]["cost"])),
experiments/runners/final2/experiments/exp05_nyc_scalability.py-86-    "2L Iters":   lambda r: fmt_iters(r["sol2"].get("iters", math.nan)),
experiments/runners/final2/experiments/exp05_nyc_scalability.py-87-    "3L Iters":   lambda r: fmt_iters(r["sol3"].get("iters", math.nan)),
--
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-33-from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-34-from shared import (
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:35:    compute_ratio, fmt_time, fmt_cost, fmt_ratio,
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-36-    build_two_level_proxy_matrix, build_three_level_proxy_matrix,
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-37-)
--
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-57-    ("2L-Prx Cost", 12),
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-58-    ("3L-Prx Cost", 12),
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:59:    ("2L Ratio",     9),
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:60:    ("3L Ratio",     9),
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-61-]
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-62-
--
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-69-    "2L-Prx Cost":  lambda r: fmt_cost(r["prx2"]["cost"]),
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-70-    "3L-Prx Cost":  lambda r: fmt_cost(r["prx3"]["cost"]),
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:71:    "2L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx2"]["cost"])),
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:72:    "3L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx3"]["cost"])),
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-73-}
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-74-
--
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-226-        e, p2, p3 = r["exact"], r["prx2"], r["prx3"]
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py-227-        print(f"N={r['n']:>7,} | exact={fmt_time(e['time_ms'])} cost={fmt_cost(e['cost'])} "
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:228:              f"| 2L={fmt_time(p2['time_ms'])} ratio={fmt_ratio(compute_ratio(e['cost'], p2['cost']))} "
experiments/runners/final2/experiments/exp03_emnist_proxy_equal.py:229:              f"| 3L={fmt_time(p3['time_ms'])} ratio={fmt_ratio(compute_ratio(e['cost'], p3['cost']))}")
--
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-30-from clustered_push_relabel.clustering.simple_precomputed import SimplePrecomputedClustering
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-31-from shared import (
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:32:    compute_ratio, fmt_time, fmt_cost, fmt_ratio,
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-33-    build_two_level_proxy_matrix, build_three_level_proxy_matrix,
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-34-    run_three_level_precomputed,
--
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-56-    ("2L-Prx Cost", 12),
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-57-    ("3L-Prx Cost", 12),
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:58:    ("2L Ratio",     9),
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:59:    ("3L Ratio",     9),
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-60-]
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-61-
--
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-68-    "2L-Prx Cost":  lambda r: fmt_cost(r["prx2"]["cost"]),
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-69-    "3L-Prx Cost":  lambda r: fmt_cost(r["prx3"]["cost"]),
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:70:    "2L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx2"]["cost"])),
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:71:    "3L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx3"]["cost"])),
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-72-}
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-73-
--
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-333-        e, p2, p3 = r["exact"], r["prx2"], r["prx3"]
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py-334-        print(f"N={r['n']:>6,} exact={fmt_time(e['time_ms'])} cost={fmt_cost(e['cost'])} "
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:335:              f"| 2L ratio={fmt_ratio(compute_ratio(e['cost'],p2['cost']))} "
experiments/runners/final2/experiments/exp08_newsgroups_proxy.py:336:              f"| 3L ratio={fmt_ratio(compute_ratio(e['cost'],p3['cost']))}")
--
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-33-from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-34-from shared import (
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:35:    compute_ratio, fmt_time, fmt_cost, fmt_ratio, fmt_iters,
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-36-    run_three_level_precomputed,
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-37-)
--
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-57-    ("2L Cost",    12),
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-58-    ("3L Cost",    12),
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:59:    ("2L Ratio",    9),
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:60:    ("3L Ratio",    9),
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-61-    ("2L Iters",   10),
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-62-    ("3L Iters",   10),
--
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-71-    "2L Cost":    lambda r: fmt_cost(r["sol2"]["cost"]),
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-72-    "3L Cost":    lambda r: fmt_cost(r["sol3"]["cost"]),
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:73:    "2L Ratio":   lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["sol2"]["cost"])),
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py:74:    "3L Ratio":   lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["sol3"]["cost"])),
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-75-    "2L Iters":   lambda r: fmt_iters(r["sol2"].get("iters", math.nan)),
experiments/runners/final2/experiments/exp07_cifar_sift_scalability.py-76-    "3L Iters":   lambda r: fmt_iters(r["sol3"].get("iters", math.nan)),
--
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-32-from clustered_push_relabel.clustering.simple_precomputed import SimplePrecomputedClustering
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-33-from shared import (
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:34:    compute_ratio, fmt_time, fmt_cost, fmt_ratio,
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-35-    build_two_level_proxy_matrix, build_three_level_proxy_matrix,
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-36-    run_three_level_precomputed,
--
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-57-    ("2L-Prx Cost", 12),
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-58-    ("3L-Prx Cost", 12),
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:59:    ("2L Ratio",     9),
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:60:    ("3L Ratio",     9),
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-61-]
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-62-
--
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-69-    "2L-Prx Cost":  lambda r: fmt_cost(r["prx2"]["cost"]),
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-70-    "3L-Prx Cost":  lambda r: fmt_cost(r["prx3"]["cost"]),
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:71:    "2L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx2"]["cost"])),
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:72:    "3L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx3"]["cost"])),
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-73-}
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-74-
--
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-326-        e, p2, p3 = r["exact"], r["prx2"], r["prx3"]
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py-327-        print(f"N={r['n']:>6,} exact={fmt_time(e['time_ms'])} cost={fmt_cost(e['cost'])} "
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:328:              f"| 2L={fmt_time(p2['time_ms'])} ratio={fmt_ratio(compute_ratio(e['cost'],p2['cost']))} "
experiments/runners/final2/experiments/exp06_cifar_sift_proxy.py:329:              f"| 3L={fmt_time(p3['time_ms'])} ratio={fmt_ratio(compute_ratio(e['cost'],p3['cost']))}")
--
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-32-from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-33-from shared import (
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:34:    compute_ratio, fmt_time, fmt_cost, fmt_ratio, fmt_iters,
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-35-    run_three_level_precomputed,
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-36-)
--
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-57-    ("2L Cost",    12),
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-58-    ("3L Cost",    12),
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:59:    ("2L Ratio",    9),
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:60:    ("3L Ratio",    9),
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-61-    ("2L Iters",   10),
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-62-    ("3L Iters",   10),
--
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-71-    "2L Cost":    lambda r: fmt_cost(r["sol2"]["cost"]),
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-72-    "3L Cost":    lambda r: fmt_cost(r["sol3"]["cost"]),
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:73:    "2L Ratio":   lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["sol2"]["cost"])),
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py:74:    "3L Ratio":   lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["sol3"]["cost"])),
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-75-    "2L Iters":   lambda r: fmt_iters(r["sol2"].get("iters", math.nan)),
experiments/runners/final2/experiments/exp09_newsgroups_scalability.py-76-    "3L Iters":   lambda r: fmt_iters(r["sol3"].get("iters", math.nan)),
--
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-34-from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-35-from shared import (
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:36:    compute_ratio, fmt_time, fmt_cost, fmt_ratio,
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-37-    build_two_level_proxy_matrix, build_three_level_proxy_matrix,
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-38-)
--
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-60-    ("2L-Prx Cost",   12),
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-61-    ("3L-Prx Cost",   12),
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:62:    ("2L Ratio",       9),
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:63:    ("3L Ratio",       9),
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-64-]
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-65-
--
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-91-    "2L-Prx Cost":      lambda r: fmt_cost(r["prx2"]["cost"]),
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-92-    "3L-Prx Cost":      lambda r: fmt_cost(r["prx3"]["cost"]),
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:93:    "2L Ratio":         lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx2"]["cost"])),
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:94:    "3L Ratio":         lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx3"]["cost"])),
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-95-}
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-96-
--
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-294-def _fmt_main_row_terminal(row):
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-295-    e, p2, p3 = row["exact"], row["prx2"], row["prx3"]
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:296:    r2 = compute_ratio(e["cost"], p2["cost"])
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:297:    r3 = compute_ratio(e["cost"], p3["cost"])
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-298-    return [
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-299-        f"{row['n']:>7,}",
--
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-325-    main_headers = ["       N", "    Exact Time", "  2L-Prx Time",
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-326-                    "  3L-Prx Time", "  Exact Cost", " 2L-Prx Cost",
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py:327:                    " 3L-Prx Cost", " 2L Ratio", " 3L Ratio"]
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-328-    main_sep = "-+-".join("-" * len(h) for h in main_headers)
experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py-329-    print("\n" + " | ".join(main_headers))
```

Per-file reporting interpretation: every experiment imports or uses `compute_ratio` in `FMT_FNS`, so ratios are included in final tables printed by `run_experiments.py` and in timestamped markdown results generated by `shared.write_markdown()` through those `FMT_FNS`. When an experiment is executed directly via its `if __name__ == "__main__"` block, ratios are printed inline for exp01 through exp04, exp06, exp08, exp10 main tables/lines; exp05, exp07, and exp09 direct blocks print times/iters but do not print ratio in the direct block even though `FMT_FNS` includes ratio for runner tables and markdown. Ratios are computed once per row/N from one run result, not per trial, because no multi-trial infrastructure exists.

## Section 16 — Open Questions for Planning

1. Largest `N` currently tested across all experiments: `200,000`. This comes from `experiments/runners/final2/experiments/exp05_nyc_scalability.py`, where `N_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]`.

2. At `N=25,000` with `EXACT_N_LIMIT=25,000`, `ot.emd()` will return a `25,000 x 25,000` dense plan if it succeeds. At float64 this is `25,000 * 25,000 * 8 = 5,000,000,000` bytes, about `4.66 GiB` (often called ~5 GB). Existing `final2` code does not handle the plan beyond `plan.argmax(...)`. Since the plan is returned by POT as a NumPy array on CPU, GPU setup is not the direct storage location for the plan; the memory problem is CPU RAM for the plan plus CPU/GPU memory for the cost matrix. The current runner only prints GPU name and total memory and has no peak tracking or plan streaming. If any future code also moves this plan to GPU, storing it would consume ~4.66 GiB of GPU memory by itself, before cost matrices and clustering tensors.

3. Experiment that saves per-pair cost distributions: no experiment saves full per-pair cost distributions to a file. `exp10_mnist_proxy_dissimilar.py` computes per-pair costs transiently and records only summary keys `frac_gt_1`, `cost_min`, `cost_median`, and `cost_max` in result rows/markdown.

4. Most recent diagnostic experiment with digit groups `{1,2,4,7}` vs `{3,6,8,9}`: `experiments/runners/final2/experiments/exp10_mnist_proxy_dissimilar.py`; exact constants are `BLUE_DIGITS = [1, 2, 4, 7]` and `RED_DIGITS = [8, 6, 9, 3]`. The red set is the same group as `{3,6,8,9}` but in file order `[8, 6, 9, 3]`; `_sample_from_digits` sorts before sampling.

5. Does any experiment currently use `sample_factor != 1.0`? No. No experiment passes `sample_factor`; clustering class defaults are used.

6. `helpers/` directory inside final2: yes, `experiments/runners/final2/helpers/` exists and contains: `['experiments/runners/final2/helpers/download_mnist.py']`.

7. Exact import path used by experiment files to import from `shared.py`: all experiment files use `from shared import ...` after inserting `FINAL2_DIR` into `sys.path`. `run_experiments.py` also uses `from shared import get_results_path, write_markdown`. No experiment uses `from runners.final2.shared import ...`.

8. Does `run_experiments.py` select which experiments to run? Yes. It supports `--run 1,3,5`, `--run all`, and an interactive menu when `--run` is omitted. It does not always run all experiments. Selection code is in `main()` with `parser.add_argument("--run", ...)`, `ask_selection(completed)`, and `selected_mods = [m for m in ALL_MODULES if m.EXP_ID in selected_ids]`.
