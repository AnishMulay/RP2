#!/usr/bin/env python3
"""
Head-to-head benchmark: RP2 2-level GPU push-relabel vs HiRef on MNIST.

This runner reuses the MNIST sampling families from the final2 proxy
experiments, but uses Euclidean L2 cost instead of L1/Manhattan cost:

  - equal: source and target are balanced over all digits 0-9
  - biased: target digits 0-4, source digits 5-9
  - dissimilar: target digits 1,2,4,7, source digits 8,6,9,3

Default preprocessing matches the prior image-histogram experiments: pixels are
scaled to [0, 1] and each image is normalized to sum to 1. Distances are then
Euclidean distances between the 784-dimensional image vectors, not squared
Euclidean distances.
"""

import argparse
import csv
import gc
import math
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
RP2_SRC_DIR = REPO_ROOT / "src"
HIREF_SRC_DIR = SCRIPT_DIR / "hiref_src"
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"

for path in (RP2_SRC_DIR, HIREF_SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver

DATASET = "MNIST flattened images"
DEFAULT_SEED = 42
DEFAULT_RP2_EPSILONS = "0.01"
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_ITERS = 999_999_999
DEFAULT_HIREF_HIERARCHY_DEPTH = 6
DEFAULT_HIREF_MAX_Q = 2**11
DEFAULT_HIREF_MAX_RANK = 64
DEFAULT_MIN_N = 5_000
DEFAULT_N_STEP = 5_000
DEFAULT_SAMPLING = "equal"

SAMPLING_DIGITS = {
    "equal": {
        "source": list(range(10)),
        "target": list(range(10)),
        "disjoint_within_digit": True,
        "note": "source and target balanced over all digits 0-9",
    },
    "biased": {
        "source": list(range(5, 10)),
        "target": list(range(5)),
        "disjoint_within_digit": False,
        "note": "target digits 0-4, source digits 5-9",
    },
    "dissimilar": {
        "source": [8, 6, 9, 3],
        "target": [1, 2, 4, 7],
        "disjoint_within_digit": False,
        "note": "target digits 1,2,4,7, source digits 8,6,9,3",
    },
}

CSV_FIELDS = [
    "timestamp",
    "dataset",
    "sampling",
    "sampling_note",
    "source_digits",
    "target_digits",
    "seed",
    "n",
    "dim",
    "method",
    "status",
    "primal_ot_cost",
    "wall_time_s",
    "peak_gpu_gb",
    "peak_gpu_extra_gb",
    "input_gpu_baseline_gb",
    "iterations",
    "rank_schedule",
    "epsilon",
    "batch_size",
    "normalization",
    "normalization_diameter",
    "cost_function",
    "dtype",
    "device",
    "data_dir",
    "error",
]


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def aggressive_cache_flush(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def current_gpu_gb(device):
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / 1024**3


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_gb(device):
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def set_torch_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tile_size(n):
    if n <= 8192:
        return 2048
    if n <= 32768:
        return 1024
    if n <= 131072:
        return 512
    return 256


def normalization_diameter(normalization):
    if normalization == "probability":
        return math.sqrt(2.0)
    if normalization == "l2":
        return 2.0
    if normalization == "pixel01":
        return math.sqrt(784.0)
    raise ValueError(f"unknown normalization {normalization!r}")


def load_mnist_arrays(data_dir, download):
    import torchvision

    try:
        train = torchvision.datasets.MNIST(
            root=str(data_dir), train=True, download=download
        )
        test = torchvision.datasets.MNIST(
            root=str(data_dir), train=False, download=download
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not load MNIST from {data_dir}. Run "
            "experiments/runners/final2/helpers/download_mnist.py or pass "
            "--download-missing."
        ) from exc

    images = torch.cat([train.data, test.data], dim=0).numpy().reshape(-1, 784)
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.astype(np.float32) / 255.0
    return images, labels


def normalize_images(arr, normalization):
    if normalization == "probability":
        sums = arr.sum(axis=1, keepdims=True)
        np.maximum(sums, 1e-8, out=sums)
        arr = arr / sums
    elif normalization == "l2":
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        np.maximum(norms, 1e-8, out=norms)
        arr = arr / norms
    elif normalization == "pixel01":
        pass
    else:
        raise ValueError(f"unknown normalization {normalization!r}")
    return arr.astype(np.float32, copy=False)


def class_counts(labels):
    return {
        int(cls): int((labels == cls).sum())
        for cls in sorted(np.unique(labels).tolist())
    }


def balanced_counts(classes, n, capacities, seed):
    classes = list(classes)
    base = n // len(classes)
    rem = n % len(classes)
    if base == 0:
        raise ValueError(f"n={n} is smaller than the {len(classes)} classes")
    too_small = [cls for cls in classes if capacities.get(cls, 0) < base]
    if too_small:
        raise ValueError(f"classes {too_small} cannot supply base count {base}")
    counts = {cls: base for cls in classes}
    if rem:
        eligible = [cls for cls in classes if capacities.get(cls, 0) > base]
        if len(eligible) < rem:
            raise ValueError(f"only {len(eligible)} classes can supply remainder {rem}")
        rng = np.random.RandomState(seed)
        rng.shuffle(eligible)
        for cls in eligible[:rem]:
            counts[cls] += 1
    return counts


def can_sample_balanced(classes, n, capacities):
    try:
        balanced_counts(classes, n, capacities, seed=0)
        return True
    except ValueError:
        return False


def max_balanced_n(classes, capacities):
    lo, hi = 0, sum(capacities.get(cls, 0) for cls in classes)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid >= len(classes) and can_sample_balanced(classes, mid, capacities):
            lo = mid
        else:
            hi = mid - 1
    return lo


def max_feasible_n(labels, sampling):
    spec = SAMPLING_DIGITS[sampling]
    counts = class_counts(labels)
    if spec["disjoint_within_digit"]:
        capacities = {cls: counts.get(cls, 0) // 2 for cls in spec["source"]}
        return max_balanced_n(spec["source"], capacities)
    source_max = max_balanced_n(spec["source"], counts)
    target_max = max_balanced_n(spec["target"], counts)
    return min(source_max, target_max)


def auto_n_values(labels, sampling, min_n, step, max_n):
    feasible = max_feasible_n(labels, sampling)
    if max_n is not None:
        feasible = min(feasible, max_n)
    if feasible < min_n:
        return [feasible] if feasible > 0 else []
    values = list(range(min_n, feasible + 1, step))
    if not values or values[-1] != feasible:
        values.append(feasible)
    return values


def parse_int_list(raw):
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("at least one N value is required")
    return values


def parse_sampling(raw):
    if raw.strip().lower() == "all":
        return list(SAMPLING_DIGITS)
    values = []
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in SAMPLING_DIGITS:
            raise ValueError(f"unknown sampling {part!r}; use equal,biased,dissimilar,all")
        if key not in values:
            values.append(key)
    if not values:
        raise ValueError("at least one sampling value is required")
    return values


def parse_methods(raw):
    aliases = {
        "rp2": "rp2",
        "pushrelabel": "rp2",
        "push-relabel": "rp2",
        "hiref": "hiref",
        "hrot": "hiref",
    }
    methods = []
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"unknown method {part!r}; use rp2,hiref")
        value = aliases[key]
        if value not in methods:
            methods.append(value)
    if not methods:
        raise ValueError("at least one method is required")
    return methods


def parse_float_list(raw):
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0.0:
            raise ValueError("epsilon values must be positive")
        values.append(value)
    if not values:
        raise ValueError("at least one epsilon is required")
    return values


def sample_class_parts(images, labels, classes, n, seed, capacity_scale=1):
    capacities = {
        cls: int((labels == cls).sum()) // capacity_scale
        for cls in classes
    }
    counts = balanced_counts(classes, n, capacities, seed)
    rng = np.random.RandomState(seed)
    parts = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        if idx.size < counts[cls]:
            raise ValueError(f"digit {cls} has {idx.size} images, need {counts[cls]}")
        rng.shuffle(idx)
        parts.append(images[idx[: counts[cls]]])
    return np.concatenate(parts)


def sample_mnist(images, labels, sampling, n, seed):
    spec = SAMPLING_DIGITS[sampling]
    if spec["disjoint_within_digit"]:
        counts = class_counts(labels)
        capacities = {cls: counts.get(cls, 0) // 2 for cls in spec["source"]}
        per_class = balanced_counts(spec["source"], n, capacities, seed + 10_000)
        rng = np.random.RandomState(seed)
        source_parts, target_parts = [], []
        for cls in spec["source"]:
            idx = np.flatnonzero(labels == cls).copy()
            needed = 2 * per_class[cls]
            if idx.size < needed:
                raise ValueError(f"digit {cls} has {idx.size} images, need {needed}")
            rng.shuffle(idx)
            chosen = idx[:needed]
            source_parts.append(images[chosen[: per_class[cls]]])
            target_parts.append(images[chosen[per_class[cls] : needed]])
        return np.concatenate(source_parts), np.concatenate(target_parts)

    source = sample_class_parts(images, labels, spec["source"], n, seed)
    target = sample_class_parts(images, labels, spec["target"], n, seed + 1)
    return source, target


def make_points(images, labels, sampling, n, args, device, dtype):
    source_np, target_np = sample_mnist(images, labels, sampling, n, args.seed)
    source_np = normalize_images(source_np, args.normalization)
    target_np = normalize_images(target_np, args.normalization)
    diameter = normalization_diameter(args.normalization)
    source_np = source_np / diameter
    target_np = target_np / diameter
    source = torch.from_numpy(source_np).to(device=device, dtype=dtype)
    target = torch.from_numpy(target_np).to(device=device, dtype=dtype)
    return source, target, diameter


def average_l2_matching_cost(source, target, match_target_to_source, diameter):
    match = match_target_to_source.to(device=source.device, dtype=torch.long)
    return float(torch.norm(target - source[match], p=2, dim=1).mean().item()) * diameter


def short_error(exc):
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def run_rp2_two_level(source, target, args, device, epsilon):
    solver = None
    cleanup(device)
    aggressive_cache_flush(device)
    baseline = current_gpu_gb(device)
    try:
        set_torch_seed(args.seed)
        reset_peak(device)
        sync_if_cuda(device)
        t0 = time.perf_counter()
        solver = SimpleGPUSolver(
            source,
            target,
            epsilon=epsilon,
            batch_size=args.batch_size,
            tile_size=get_tile_size(source.shape[0]),
            verbose=args.verbose,
            max_iters=args.max_iters,
            diameter=1.0,
        )
        match = solver.solve()
        cost = average_l2_matching_cost(source, target, match, args.current_diameter)
        sync_if_cuda(device)
        wall = time.perf_counter() - t0
        peak = peak_gpu_gb(device)
        return {
            "method": "RP2_2level_gpu_push_relabel",
            "status": "ok",
            "primal_ot_cost": cost,
            "wall_time_s": wall,
            "peak_gpu_gb": peak,
            "peak_gpu_extra_gb": max(0.0, peak - baseline),
            "input_gpu_baseline_gb": baseline,
            "iterations": getattr(solver, "iterations", math.nan),
            "rank_schedule": "",
            "epsilon": epsilon,
            "batch_size": args.batch_size,
            "error": "",
        }
    except Exception as exc:
        peak = peak_gpu_gb(device)
        return {
            "method": "RP2_2level_gpu_push_relabel",
            "status": "oom" if is_oom(exc) else "error",
            "primal_ot_cost": math.nan,
            "wall_time_s": math.nan,
            "peak_gpu_gb": peak,
            "peak_gpu_extra_gb": max(0.0, peak - baseline),
            "input_gpu_baseline_gb": baseline,
            "iterations": getattr(solver, "iterations", math.nan) if solver else math.nan,
            "rank_schedule": "",
            "epsilon": epsilon,
            "batch_size": args.batch_size,
            "error": short_error(exc),
        }
    finally:
        del solver
        cleanup(device)


def run_hiref(source, target, args, device):
    import HR_OT
    import rank_annealing

    hrot = None
    rank_schedule = []
    cleanup(device)
    aggressive_cache_flush(device)
    baseline = current_gpu_gb(device)
    try:
        set_torch_seed(args.seed)
        rank_schedule = rank_annealing.optimal_rank_schedule(
            source.shape[0],
            hierarchy_depth=args.hiref_hierarchy_depth,
            max_Q=args.hiref_max_q,
            max_rank=args.hiref_max_rank,
        )
        reset_peak(device)
        sync_if_cuda(device)
        t0 = time.perf_counter()
        hrot = HR_OT.HierarchicalRefinementOT.init_from_point_clouds(
            source,
            target,
            rank_schedule,
            base_rank=1,
            device=device,
            sq_Euclidean=False,
        )
        hrot.run(return_as_coupling=False)
        cost = float(hrot.compute_OT_cost().item()) * args.current_diameter
        sync_if_cuda(device)
        wall = time.perf_counter() - t0
        peak = peak_gpu_gb(device)
        return {
            "method": "HiRef_HROT_LR",
            "status": "ok",
            "primal_ot_cost": cost,
            "wall_time_s": wall,
            "peak_gpu_gb": peak,
            "peak_gpu_extra_gb": max(0.0, peak - baseline),
            "input_gpu_baseline_gb": baseline,
            "iterations": math.nan,
            "rank_schedule": " ".join(str(v) for v in rank_schedule),
            "epsilon": "",
            "batch_size": "",
            "error": "",
        }
    except Exception as exc:
        peak = peak_gpu_gb(device)
        return {
            "method": "HiRef_HROT_LR",
            "status": "oom" if is_oom(exc) else "error",
            "primal_ot_cost": math.nan,
            "wall_time_s": math.nan,
            "peak_gpu_gb": peak,
            "peak_gpu_extra_gb": max(0.0, peak - baseline),
            "input_gpu_baseline_gb": baseline,
            "iterations": math.nan,
            "rank_schedule": " ".join(str(v) for v in rank_schedule),
            "epsilon": "",
            "batch_size": "",
            "error": short_error(exc),
        }
    finally:
        del hrot
        cleanup(device)


def csv_value(value):
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.10g}"
    return value


def append_rows(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, "")) for k in CSV_FIELDS})


def md_escape(value):
    return str(value).replace("\n", " ").replace("|", "\\|")


def md_float(value, digits=6):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def method_label(row):
    method = row.get("method", "")
    if method.startswith("RP2_"):
        return f"RP2 eps={float(row['epsilon']):.3g}"
    if method.startswith("HiRef"):
        return "HiRef"
    return method


def write_incremental_markdown(md_path, metadata, rows):
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HiRef MNIST Distribution Benchmark",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Setup",
    ]
    for key, value in metadata:
        lines.append(f"- **{md_escape(key)}:** {md_escape(value)}")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Sampling | N | Method | Status | Avg Cost | Time (s) | Peak GPU (GB) | Diameter | Error |",
            "|---|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(row.get("sampling", "")),
                    f"{int(row.get('n', 0)):,}",
                    md_escape(method_label(row)),
                    md_escape(row.get("status", "")),
                    md_float(row.get("primal_ot_cost"), 6),
                    md_float(row.get("wall_time_s"), 2),
                    md_float(row.get("peak_gpu_gb"), 3),
                    md_float(row.get("normalization_diameter"), 6),
                    md_escape(row.get("error", "")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_result(row):
    cost = row["primal_ot_cost"]
    cost_s = "N/A" if not math.isfinite(float(cost)) else f"{float(cost):.6f}"
    wall = row["wall_time_s"]
    wall_s = "N/A" if not math.isfinite(float(wall)) else f"{float(wall):.2f}s"
    peak = row["peak_gpu_gb"]
    peak_s = "N/A" if not math.isfinite(float(peak)) else f"{float(peak):.3f}GB"
    print(
        f"  {row['method']}: {row['status']} | cost={cost_s} | "
        f"time={wall_s} | peak={peak_s}",
        flush=True,
    )


def print_summary_table(sampling, n, rows, diameter):
    if not rows:
        return
    headers = ("Method", "Status", "Avg Cost", "Time")
    body = []
    for row in rows:
        cost = row["primal_ot_cost"]
        wall = row["wall_time_s"]
        body.append(
            (
                method_label(row),
                row["status"],
                "N/A" if not math.isfinite(float(cost)) else f"{float(cost):.6f}",
                "N/A" if not math.isfinite(float(wall)) else f"{float(wall):.2f}s",
            )
        )
    widths = [max(len(headers[i]), *(len(r[i]) for r in body)) for i in range(len(headers))]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(f"\nSummary for sampling={sampling}, N={n:,}", flush=True)
    print(f"Normalized by analytic diameter: {diameter:.6f}", flush=True)
    print(sep, flush=True)
    print("| " + " | ".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers))) + " |", flush=True)
    print(sep, flush=True)
    for row in body:
        print("| " + " | ".join(f"{row[i]:>{widths[i]}}" for i in range(len(row))) + " |", flush=True)
    print(sep, flush=True)


def add_common_fields(row, args, sampling, n, device, diameter):
    spec = SAMPLING_DIGITS[sampling]
    row.update(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": DATASET,
            "sampling": sampling,
            "sampling_note": spec["note"],
            "source_digits": " ".join(str(v) for v in spec["source"]),
            "target_digits": " ".join(str(v) for v in spec["target"]),
            "seed": args.seed,
            "n": n,
            "dim": 784,
            "normalization": args.normalization,
            "normalization_diameter": diameter,
            "cost_function": "Euclidean L2 on 784-D image vectors",
            "dtype": args.dtype,
            "device": str(device),
            "data_dir": str(args.data_dir),
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RP2 2-level solver against HiRef on MNIST sampling distributions."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--sampling", default=DEFAULT_SAMPLING, help="equal, biased, dissimilar, or all")
    parser.add_argument("--n-values", default=None, help="Comma-separated N values. Default: auto to feasible max.")
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--n-step", type=int, default=DEFAULT_N_STEP)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument(
        "--normalization",
        choices=("probability", "l2", "pixel01"),
        default="probability",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--methods", default="rp2,hiref")
    parser.add_argument(
        "--rp2-epsilons",
        default=DEFAULT_RP2_EPSILONS,
        help="Comma-separated RP2 epsilon values to run. Default: 0.01.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--hiref-hierarchy-depth", type=int, default=DEFAULT_HIREF_HIERARCHY_DEPTH)
    parser.add_argument("--hiref-max-q", type=int, default=DEFAULT_HIREF_MAX_Q)
    parser.add_argument("--hiref-max-rank", type=int, default=DEFAULT_HIREF_MAX_RANK)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Incremental markdown output path. Default: same as --output with .md.",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for both benchmarked GPU methods.")

    methods = parse_methods(args.methods)
    samplings = parse_sampling(args.sampling)
    rp2_epsilons = parse_float_list(args.rp2_epsilons)
    device = torch.device("cuda")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = Path(args.output) if args.output else RESULTS_DIR / f"hiref_mnist_distributions_{timestamp}.csv"
    md_path = Path(args.markdown_output) if args.markdown_output else csv_path.with_suffix(".md")
    markdown_rows = []

    warnings.filterwarnings("default")
    images, labels = load_mnist_arrays(args.data_dir, args.download_missing)

    print(f"Dataset: {DATASET}", flush=True)
    print(f"Data dir: {args.data_dir}", flush=True)
    print(f"MNIST images: {images.shape[0]:,}", flush=True)
    print(f"Normalization: {args.normalization}", flush=True)
    print(f"Cost: Euclidean L2, not squared Euclidean", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Output: {csv_path}", flush=True)
    print(f"Markdown: {md_path}", flush=True)
    if "rp2" in methods:
        print(f"RP2 epsilons: {', '.join(str(v) for v in rp2_epsilons)}", flush=True)

    markdown_metadata = [
        ("Dataset", DATASET),
        ("Data dir", args.data_dir),
        ("Samplings", ",".join(samplings)),
        ("Normalization", args.normalization),
        ("Seed", args.seed),
        ("Methods", ",".join(methods)),
        ("RP2 epsilons", ",".join(str(v) for v in rp2_epsilons) if "rp2" in methods else "N/A"),
        ("Cost function", "Euclidean L2 on 784-D image vectors"),
        ("Diameter", "Analytic preprocessing diameter computed before timing; inputs normalized by this value"),
        ("CSV", csv_path),
    ]

    for sampling in samplings:
        if args.n_values:
            n_values = parse_int_list(args.n_values)
        else:
            n_values = auto_n_values(labels, sampling, args.min_n, args.n_step, args.max_n)
        print(f"\nSampling: {sampling} ({SAMPLING_DIGITS[sampling]['note']})", flush=True)
        print(f"N values: {n_values}", flush=True)

        for n in n_values:
            print(f"\nN={n:,} | sampling={sampling} | dim=784", flush=True)
            cleanup(device)
            try:
                source, target, diameter = make_points(images, labels, sampling, n, args, device, dtype)
                args.current_diameter = diameter
                sync_if_cuda(device)
                print(f"  Normalization diameter: {diameter:.6f}", flush=True)
            except Exception as exc:
                row = {
                    "method": "data_generation",
                    "status": "error",
                    "primal_ot_cost": math.nan,
                    "wall_time_s": math.nan,
                    "peak_gpu_gb": peak_gpu_gb(device),
                    "peak_gpu_extra_gb": math.nan,
                    "input_gpu_baseline_gb": current_gpu_gb(device),
                    "iterations": math.nan,
                    "rank_schedule": "",
                    "epsilon": "",
                    "batch_size": "",
                    "error": short_error(exc),
                }
                add_common_fields(row, args, sampling, n, device, "")
                append_rows(csv_path, [row])
                markdown_rows.append(row)
                write_incremental_markdown(md_path, markdown_metadata, markdown_rows)
                print_result(row)
                if not args.keep_going:
                    return
                continue

            rows = []
            if "rp2" in methods:
                for epsilon in rp2_epsilons:
                    rows.append(run_rp2_two_level(source, target, args, device, epsilon))
                    print_result(rows[-1])
            if "hiref" in methods:
                rows.append(run_hiref(source, target, args, device))
                print_result(rows[-1])

            for row in rows:
                add_common_fields(row, args, sampling, n, device, diameter)
            append_rows(csv_path, rows)
            markdown_rows.extend(rows)
            write_incremental_markdown(md_path, markdown_metadata, markdown_rows)
            print_summary_table(sampling, n, rows, diameter)

            both_oom_seen = rows and all(row["status"] == "oom" for row in rows)
            del source, target
            cleanup(device)
            if both_oom_seen and not args.keep_going:
                print("All selected method rows OOMed; moving to next sampling.", flush=True)
                break
            if any(row["status"] == "error" for row in rows) and not args.keep_going:
                print("A selected method errored; stopping.", flush=True)
                return

    print(f"\nWrote results to {csv_path}", flush=True)
    print(f"Wrote markdown to {md_path}", flush=True)


if __name__ == "__main__":
    main()
