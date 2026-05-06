#!/usr/bin/env python3
"""
Head-to-head benchmark: RP2 2-level GPU push-relabel vs HiRef on EMNIST.

Each EMNIST image is represented as a 784-dimensional flattened pixel vector.
The default preprocessing follows the existing final2 EMNIST experiments:
pixels are scaled to [0, 1] and each image is normalized to sum to 1.
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


def verify_gpu_backends(abort_on_failure=True):
    """
    Confirm that both PyTorch and JAX are running on GPU.
    Prints a clear confirmation or failure message for each backend.
    If abort_on_failure=True, raises RuntimeError if either backend
    is not on GPU, so the experiment fails loudly rather than silently
    producing CPU-based timing results.
    """
    print("\n" + "="*60, flush=True)
    print("  GPU BACKEND VERIFICATION", flush=True)
    print("="*60, flush=True)

    failures = []

    # --- PyTorch check ---
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  [PyTorch] OK — CUDA available: {gpu_name}", flush=True)
        print(f"  [PyTorch] Device count: {torch.cuda.device_count()}",
              flush=True)
    else:
        msg = "[PyTorch] FAIL — CUDA not available. " \
              "2-Level solver will run on CPU."
        print(f"  {msg}", flush=True)
        failures.append(msg)

    # --- JAX check ---
    try:
        import jax
        jax_devices = jax.devices()
        jax_backend = jax.default_backend()
        print(f"  [JAX] Backend: {jax_backend}", flush=True)
        print(f"  [JAX] Devices: {jax_devices}", flush=True)
        if jax_backend != "gpu":
            msg = (
                f"[JAX] FAIL — JAX backend is '{jax_backend}', not 'gpu'. "
                f"HiRef will run on {jax_backend.upper()}. "
                f"This makes runtime comparisons with the GPU 2-Level "
                f"solver invalid."
            )
            print(f"  {msg}", flush=True)
            failures.append(msg)
        else:
            # Run a tiny JAX op on GPU to confirm it actually executes
            import jax.numpy as jnp
            _ = jnp.ones((4, 4), dtype=jnp.float32) @ jnp.ones((4, 4), dtype=jnp.float32)
            _.block_until_ready()
            print(f"  [JAX] OK — test op executed successfully on GPU",
                  flush=True)
    except Exception as e:
        msg = f"[JAX] FAIL — could not verify JAX backend: {e}"
        print(f"  {msg}", flush=True)
        failures.append(msg)

    # --- Cross-backend device name reconciliation ---
    # Confirm both backends see the same physical GPU
    try:
        import torch
        import jax
        torch_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        jax_device_str = str(jax.devices()[0]) if jax.devices() else "N/A"
        print(f"  [Cross-check] PyTorch GPU: {torch_gpu}", flush=True)
        print(f"  [Cross-check] JAX device:  {jax_device_str}", flush=True)
    except Exception as e:
        print(f"  [Cross-check] Could not reconcile devices: {e}", flush=True)

    print("="*60 + "\n", flush=True)

    if failures and abort_on_failure:
        raise RuntimeError(
            "GPU backend verification failed. Aborting experiment to prevent "
            "invalid CPU/GPU mixed comparisons.\n" +
            "\n".join(failures)
        )

    return len(failures) == 0


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
RP2_SRC_DIR = REPO_ROOT / "src"
HIREF_SRC_DIR = SCRIPT_DIR / "hiref_src"
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_DATA_DIR = REPO_ROOT / "data"

for path in (RP2_SRC_DIR, HIREF_SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver

DATASET = "EMNIST byclass flattened images"
DEFAULT_MIN_POWER = 10
DEFAULT_MAX_POWER = 16
DEFAULT_SEED = 42
DEFAULT_RP2_EPSILON = 0.01
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_ITERS = 999_999_999
DEFAULT_HIREF_HIERARCHY_DEPTH = 6
DEFAULT_HIREF_MAX_Q = 2**11
DEFAULT_HIREF_MAX_RANK = 64
CSV_FIELDS = [
    "timestamp",
    "dataset",
    "emnist_split",
    "sampling",
    "normalization",
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
    "normalization_diameter",
    "cost_function",
    "dtype",
    "device",
    "pytorch_device",
    "jax_backend",
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


def sample_sizes(min_power, max_power):
    return [2**p for p in range(min_power, max_power + 1)]


def get_tile_size(n):
    if n <= 8192:
        return 2048
    elif n <= 32768:
        return 1024
    elif n <= 131072:
        return 512
    else:
        return 256


def normalization_diameter(normalization):
    if normalization == "probability":
        return math.sqrt(2.0)
    if normalization == "l2":
        return 2.0
    if normalization == "pixel01":
        return math.sqrt(784.0)
    raise ValueError(f"unknown normalization {normalization!r}")


def load_emnist_arrays(data_dir, split):
    import torchvision

    train = torchvision.datasets.EMNIST(
        root=str(data_dir), split=split, train=True, download=False
    )
    test = torchvision.datasets.EMNIST(
        root=str(data_dir), split=split, train=False, download=False
    )
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()

    # EMNIST images are stored transposed relative to their natural orientation.
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)
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


def balanced_class_counts(classes, n, rng):
    classes = list(classes)
    base = n // len(classes)
    rem = n % len(classes)
    if base == 0:
        raise ValueError(f"n={n} is smaller than the {len(classes)} EMNIST classes")
    order = classes.copy()
    rng.shuffle(order)
    counts = {cls: base for cls in classes}
    for cls in order[:rem]:
        counts[cls] += 1
    return counts


def sample_equal(images, labels, n, seed):
    classes = sorted(np.unique(labels).tolist())
    rng_counts = np.random.RandomState(seed + 10_000)
    counts = balanced_class_counts(classes, n, rng_counts)
    rng = np.random.RandomState(seed)
    red_parts, blue_parts = [], []
    for cls in classes:
        count = counts[cls]
        idx = np.flatnonzero(labels == cls).copy()
        needed = 2 * count
        if idx.size < needed:
            raise ValueError(f"class {cls} has {idx.size} images, need {needed}")
        rng.shuffle(idx)
        chosen = idx[:needed]
        red_parts.append(images[chosen[:count]])
        blue_parts.append(images[chosen[count:needed]])
    return np.concatenate(red_parts), np.concatenate(blue_parts)


def sample_biased(images, labels, n, seed):
    classes = sorted(np.unique(labels).tolist())
    midpoint = len(classes) // 2
    blue_classes = classes[:midpoint]
    red_classes = classes[midpoint:]
    rng_red_counts = np.random.RandomState(seed + 20_000)
    rng_blue_counts = np.random.RandomState(seed + 30_000)
    red_counts = balanced_class_counts(red_classes, n, rng_red_counts)
    blue_counts = balanced_class_counts(blue_classes, n, rng_blue_counts)
    rng_red = np.random.RandomState(seed)
    rng_blue = np.random.RandomState(seed + 1)

    red_parts, blue_parts = [], []
    for cls in red_classes:
        count = red_counts[cls]
        idx = np.flatnonzero(labels == cls).copy()
        if idx.size < count:
            raise ValueError(f"red class {cls} has {idx.size} images, need {count}")
        rng_red.shuffle(idx)
        red_parts.append(images[idx[:count]])
    for cls in blue_classes:
        count = blue_counts[cls]
        idx = np.flatnonzero(labels == cls).copy()
        if idx.size < count:
            raise ValueError(f"blue class {cls} has {idx.size} images, need {count}")
        rng_blue.shuffle(idx)
        blue_parts.append(images[idx[:count]])
    return np.concatenate(red_parts), np.concatenate(blue_parts)


def make_emnist_points(args, device, dtype):
    images, labels = load_emnist_arrays(args.data_dir, args.split)
    if args.sampling == "equal":
        source_np, target_np = sample_equal(images, labels, args.current_n, args.seed)
    elif args.sampling == "biased":
        source_np, target_np = sample_biased(images, labels, args.current_n, args.seed)
    else:
        raise ValueError(f"unknown sampling {args.sampling!r}")

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
    text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return text.replace("\n", " ")


def run_rp2_two_level(source, target, args, device):
    solver = None
    cleanup(device)
    aggressive_cache_flush(device)
    baseline = current_gpu_gb(device)
    try:
        set_torch_seed(args.seed)
        reset_peak(device)
        # Confirm 2-Level solver inputs are on CUDA before timing
        assert source.is_cuda, \
            f"2-Level solver input 'source' is on {source.device}, not CUDA"
        assert target.is_cuda, \
            f"2-Level solver input 'target' is on {target.device}, not CUDA"
        print(f"  [2-Level] inputs confirmed on {source.device}",
              flush=True)
        sync_if_cuda(device)
        t0 = time.perf_counter()
        solver = SimpleGPUSolver(
            source,
            target,
            epsilon=args.rp2_epsilon,
            batch_size=args.batch_size,
            tile_size=get_tile_size(source.shape[0]),
            verbose=args.verbose,
            max_iters=args.max_iters,
            diameter=1.0,
        )
        match = solver.solve()
        sync_if_cuda(device)
        wall = time.perf_counter() - t0
        peak = peak_gpu_gb(device)
        cost = average_l2_matching_cost(source, target, match, args.current_diameter)
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
            "epsilon": args.rp2_epsilon,
            "batch_size": args.batch_size,
            "pytorch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "jax_backend": "N/A",
            "error": "",
        }
    except torch.cuda.OutOfMemoryError as exc:
        peak = peak_gpu_gb(device)
        return {
            "method": "RP2_2level_gpu_push_relabel",
            "status": "oom",
            "primal_ot_cost": math.nan,
            "wall_time_s": math.nan,
            "peak_gpu_gb": peak,
            "peak_gpu_extra_gb": max(0.0, peak - baseline),
            "input_gpu_baseline_gb": baseline,
            "iterations": getattr(solver, "iterations", math.nan) if solver else math.nan,
            "rank_schedule": "",
            "epsilon": args.rp2_epsilon,
            "batch_size": args.batch_size,
            "pytorch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "jax_backend": "N/A",
            "error": short_error(exc),
        }
    except Exception as exc:
        return {
            "method": "RP2_2level_gpu_push_relabel",
            "status": "error",
            "primal_ot_cost": math.nan,
            "wall_time_s": math.nan,
            "peak_gpu_gb": peak_gpu_gb(device),
            "peak_gpu_extra_gb": math.nan,
            "input_gpu_baseline_gb": baseline,
            "iterations": getattr(solver, "iterations", math.nan) if solver else math.nan,
            "rank_schedule": "",
            "epsilon": args.rp2_epsilon,
            "batch_size": args.batch_size,
            "pytorch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "jax_backend": "N/A",
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
        import jax
        import jax.numpy as jnp
        # Confirm JAX is still on GPU immediately before HiRef runs
        _jax_backend = jax.default_backend()
        assert _jax_backend == "gpu", \
            f"JAX backend is '{_jax_backend}' immediately before HiRef call. " \
            f"Expected 'gpu'. Aborting to prevent invalid comparison."
        print(f"  [HiRef] JAX backend confirmed: {_jax_backend}", flush=True)
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
            "pytorch_device": "N/A",
            "jax_backend": jax.default_backend(),
            "error": "",
        }
    except torch.cuda.OutOfMemoryError as exc:
        peak = peak_gpu_gb(device)
        return {
            "method": "HiRef_HROT_LR",
            "status": "oom",
            "primal_ot_cost": math.nan,
            "wall_time_s": math.nan,
            "peak_gpu_gb": peak,
            "peak_gpu_extra_gb": max(0.0, peak - baseline),
            "input_gpu_baseline_gb": baseline,
            "iterations": math.nan,
            "rank_schedule": " ".join(str(v) for v in rank_schedule),
            "epsilon": "",
            "batch_size": "",
            "pytorch_device": "N/A",
            "jax_backend": jax.default_backend(),
            "error": short_error(exc),
        }
    except Exception as exc:
        return {
            "method": "HiRef_HROT_LR",
            "status": "error",
            "primal_ot_cost": math.nan,
            "wall_time_s": math.nan,
            "peak_gpu_gb": peak_gpu_gb(device),
            "peak_gpu_extra_gb": math.nan,
            "input_gpu_baseline_gb": baseline,
            "iterations": math.nan,
            "rank_schedule": " ".join(str(v) for v in rank_schedule),
            "epsilon": "",
            "batch_size": "",
            "pytorch_device": "N/A",
            "jax_backend": jax.default_backend(),
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


def write_incremental_markdown(md_path, title, metadata, rows):
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
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
            "| N | Method | Status | Avg Cost | Time (s) | Peak GPU (GB) | Diameter | Error |",
            "|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
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


def main():
    verify_gpu_backends(abort_on_failure=True)

    parser = argparse.ArgumentParser(
        description="Benchmark RP2 2-level solver against HiRef on flattened EMNIST images."
    )
    parser.add_argument("--min-power", type=int, default=DEFAULT_MIN_POWER)
    parser.add_argument("--max-power", type=int, default=DEFAULT_MAX_POWER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--methods", default="rp2,hiref")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="byclass")
    parser.add_argument("--sampling", choices=("equal", "biased"), default="equal")
    parser.add_argument(
        "--normalization",
        choices=("probability", "l2", "pixel01"),
        default="probability",
    )
    parser.add_argument("--rp2-epsilon", type=float, default=DEFAULT_RP2_EPSILON)
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
    device = torch.device("cuda")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = Path(args.output) if args.output else RESULTS_DIR / f"hiref_emnist_{timestamp}.csv"
    md_path = Path(args.markdown_output) if args.markdown_output else csv_path.with_suffix(".md")
    markdown_rows = []

    warnings.filterwarnings("default")
    print(f"Dataset: {DATASET}", flush=True)
    print(f"Split: {args.split} | sampling={args.sampling}", flush=True)
    print(f"Normalization: {args.normalization}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Data dir: {args.data_dir}", flush=True)
    print(f"Output: {csv_path}", flush=True)
    print(f"Markdown: {md_path}", flush=True)
    print(f"Sample sizes: {sample_sizes(args.min_power, args.max_power)}", flush=True)

    markdown_metadata = [
        ("Dataset", DATASET),
        ("Split", args.split),
        ("Sampling", args.sampling),
        ("Normalization", args.normalization),
        ("Seed", args.seed),
        ("Sample sizes", sample_sizes(args.min_power, args.max_power)),
        ("Methods", ",".join(methods)),
        ("RP2 epsilon", args.rp2_epsilon if "rp2" in methods else "N/A"),
        ("Cost function", "Euclidean L2 on 784-D image vectors"),
        ("Diameter", "Analytic preprocessing diameter computed before timing; inputs normalized by this value"),
        ("CSV", csv_path),
    ]

    for n in sample_sizes(args.min_power, args.max_power):
        print(f"\nN={n:,} | dim=784", flush=True)
        cleanup(device)
        args.current_n = n
        try:
            source, target, diameter = make_emnist_points(args, device, dtype)
            args.current_diameter = diameter
            sync_if_cuda(device)
            print(f"  Normalization diameter: {diameter:.6f}", flush=True)
        except Exception as exc:
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "dataset": DATASET,
                "emnist_split": args.split,
                "sampling": args.sampling,
                "normalization": args.normalization,
                "seed": args.seed,
                "n": n,
                "dim": 784,
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
                "normalization_diameter": "",
                "cost_function": "Euclidean L2 on 784-D image vectors",
                "dtype": args.dtype,
                "device": str(device),
                "error": short_error(exc),
            }
            append_rows(csv_path, [row])
            markdown_rows.append(row)
            write_incremental_markdown(
                md_path,
                "HiRef EMNIST Benchmark",
                markdown_metadata,
                markdown_rows,
            )
            print_result(row)
            if not args.keep_going:
                break
            continue

        rows = []
        if "rp2" in methods:
            rows.append(run_rp2_two_level(source, target, args, device))
            print_result(rows[-1])
        if "hiref" in methods:
            rows.append(run_hiref(source, target, args, device))
            print_result(rows[-1])

        for row in rows:
            row.update(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "dataset": DATASET,
                    "emnist_split": args.split,
                    "sampling": args.sampling,
                    "normalization": args.normalization,
                    "seed": args.seed,
                    "n": n,
                    "dim": 784,
                    "cost_function": "Euclidean L2 on 784-D image vectors",
                    "dtype": args.dtype,
                    "device": str(device),
                    "normalization_diameter": diameter,
                }
            )
        append_rows(csv_path, rows)
        markdown_rows.extend(rows)
        write_incremental_markdown(
            md_path,
            "HiRef EMNIST Benchmark",
            markdown_metadata,
            markdown_rows,
        )

        statuses = {row["method"]: row["status"] for row in rows}
        both_oom_seen = rows and all(row["status"] == "oom" for row in rows)
        del source, target
        cleanup(device)
        if both_oom_seen and not args.keep_going:
            print("Both selected methods OOMed; stopping sweep.", flush=True)
            break
        if any(status == "error" for status in statuses.values()) and not args.keep_going:
            print("A selected method errored; stopping sweep.", flush=True)
            break

    print(f"\nWrote results to {csv_path}", flush=True)
    print(f"Wrote markdown to {md_path}", flush=True)


if __name__ == "__main__":
    main()
