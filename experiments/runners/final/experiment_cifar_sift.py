#!/usr/bin/env python3

import gzip
import math
import pathlib
import pickle
import statistics
import sys
import time

import numpy as np
import torch

try:
    import ot
except ImportError:
    ot = None


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR  = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clustered_push_relabel.clustering.simple import SimpleClustering
from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver


DATA_DIR = BASE_DIR / "data" / "cifar_sift"
TRAIN_DESC_PATH = DATA_DIR / "cifar10_sift_train.pkl.gz"

N_VALUES_EXACT = [1_000, 2_000, 5_000]
N_VALUES_SCALE = [10_000, 20_000]
N_VALUES = N_VALUES_EXACT + N_VALUES_SCALE
EXACT_N_LIMIT = 5_000
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
WARMUP_RUNS = 0
TIMED_RUNS = 1


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def is_available(value):
    return value == value


def fmt_time(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.1f} ms"


def fmt_cost(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.4f}"


def fmt_iter(value):
    if not is_available(value):
        return "N/A"
    return f"{int(value):,}"


def normalize_points(A, B):
    """
    Normalize two [N, d] tensors by their joint bounding-box diameter.
    all_pts = torch.cat([A, B], dim=0)
    diameter = float((all_pts.max(dim=0).values -
                      all_pts.min(dim=0).values).max().item())
    diameter = max(diameter, 1e-6)
    return A / diameter, B / diameter, diameter
    """
    all_pts = torch.cat([A, B], dim=0)
    diameter = float(
        (all_pts.max(dim=0).values - all_pts.min(dim=0).values).max().item()
    )
    diameter = max(diameter, 1e-6)
    return A / diameter, B / diameter, diameter


def load_cifar_sift_descriptors(path):
    """Load pkl.gz -> list of N numpy arrays, each (K_i, 128)."""
    with gzip.open(path, "rb") as f:
        descs = pickle.load(f)
    print(
        f"  Loaded {len(descs):,} SIFT descriptor sets from {path.name}",
        flush=True,
    )
    return descs


def compute_mean_descriptors(descs_list):
    """
    Compute mean SIFT descriptor per image.
    For each image i: mean_desc[i] = descs_list[i].mean(axis=0)  shape (128,)
    Images with 0 keypoints (shape (1,128) zero fallback) return zero vector.
    Returns numpy array of shape (N, 128) float32.
    """
    means = np.zeros((len(descs_list), 128), dtype=np.float32)
    for i, desc in enumerate(descs_list):
        arr = np.asarray(desc, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 128:
            raise ValueError(
                f"Descriptor set {i} has shape {arr.shape}; expected (K, 128)."
            )
        if arr.shape[0] == 0:
            continue
        means[i] = arr.mean(axis=0).astype(np.float32, copy=False)
    return means


def sample_pair(all_descs, n_samples, seed):
    """
    Sample two non-overlapping sets of n_samples.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_descs))
    red_descs  = [all_descs[i] for i in perm[:n_samples]]
    blue_descs = [all_descs[i] for i in perm[n_samples:2*n_samples]]
    Returns red_descs, blue_descs (lists of numpy arrays)
    """
    if 2 * n_samples > len(all_descs):
        raise ValueError(
            f"Need at least {2 * n_samples:,} images, got {len(all_descs):,}."
        )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_descs))
    red_descs = [all_descs[i] for i in perm[:n_samples]]
    blue_descs = [all_descs[i] for i in perm[n_samples : 2 * n_samples]]
    return red_descs, blue_descs


def _pad_descriptors(descs_list, device):
    """
    Pad all descriptor sets to the same size and stack into one tensor.
    """
    n = len(descs_list)
    dim = descs_list[0].shape[1]
    max_K = max(max(desc.shape[0], 1) for desc in descs_list)

    padded = torch.zeros((n, max_K, dim), dtype=torch.float32, device=device)
    lengths = torch.empty(n, dtype=torch.int64, device=device)

    for i, desc in enumerate(descs_list):
        K_i = desc.shape[0]
        if K_i == 0:
            lengths[i] = 1
            continue
        padded[i, :K_i, :] = torch.as_tensor(
            desc, dtype=torch.float32, device=device
        )
        lengths[i] = K_i

    return padded, lengths


def compute_chamfer_matrix(descs_B, descs_A, device, tile_b=50, tile_a=200):
    """
    Compute D[b, a] = chamfer(descs_B[b], descs_A[a]) for all b, a.
    Returns (N, N) float32 tensor on device.

    Reduce `tile_a` if this runs out of GPU memory.
    """
    n = len(descs_B)
    if len(descs_A) != n:
        raise ValueError("descs_B and descs_A must have the same length")

    padded_B, lengths_B = _pad_descriptors(descs_B, device)
    padded_A, lengths_A = _pad_descriptors(descs_A, device)
    max_KB = padded_B.shape[1]
    max_KA = padded_A.shape[1]

    valid_B_all = (
        torch.arange(max_KB, device=device).unsqueeze(0) < lengths_B.unsqueeze(1)
    )
    valid_A_all = (
        torch.arange(max_KA, device=device).unsqueeze(0) < lengths_A.unsqueeze(1)
    )
    D = torch.zeros((n, n), dtype=torch.float32, device=device)

    for b_start in range(0, n, tile_b):
        b_end = min(b_start + tile_b, n)
        tb = b_end - b_start

        B_tile = padded_B[b_start:b_end]
        lens_B = lengths_B[b_start:b_end]
        valid_B = valid_B_all[b_start:b_end]
        B_flat = B_tile.reshape(tb * max_KB, B_tile.shape[2])

        for a_start in range(0, n, tile_a):
            a_end = min(a_start + tile_a, n)
            ta = a_end - a_start

            A_tile = padded_A[a_start:a_end]
            lens_A = lengths_A[a_start:a_end]
            valid_A = valid_A_all[a_start:a_end]
            A_flat = A_tile.reshape(ta * max_KA, A_tile.shape[2])

            dists = torch.cdist(
                B_flat,
                A_flat,
                p=2,
                compute_mode="use_mm_for_euclid_dist_if_necessary",
            )
            dists = dists.reshape(tb, max_KB, ta, max_KA)

            dists.masked_fill_(~valid_A.view(1, 1, ta, max_KA), float("inf"))
            fwd_min = dists.min(dim=3).values
            fwd_min = fwd_min.masked_fill(~valid_B.unsqueeze(2), 0.0)
            fwd = (
                fwd_min.sum(dim=1)
                / lens_B.to(torch.float32).unsqueeze(1).clamp(min=1.0)
            )

            dists.masked_fill_(~valid_B.view(tb, max_KB, 1, 1), float("inf"))
            bwd_min = dists.min(dim=1).values
            bwd_min = bwd_min.masked_fill(~valid_A.unsqueeze(0), 0.0)
            bwd = (
                bwd_min.sum(dim=2)
                / lens_A.to(torch.float32).unsqueeze(0).clamp(min=1.0)
            )

            D[b_start:b_end, a_start:a_end] = fwd + bwd

        if b_start % (tile_b * 4) == 0 or b_end == n:
            print(
                f"    Computing Chamfer [blue->red]: "
                f"{b_end}/{n} rows done...",
                flush=True,
            )

    synchronize_if_cuda(device)
    print("    Computing Chamfer [blue->red]: complete.", flush=True)
    del padded_B, padded_A
    return D


def compute_chamfer_matrix_symmetric(descs, device, tile=50):
    """
    Compute symmetric N x N Chamfer matrix for a single set of images.
    D[i,j] = D[j,i], D[i,i] = 0.

    Reduce `tile` if this runs out of GPU memory.
    """
    n = len(descs)
    padded, lengths = _pad_descriptors(descs, device)
    max_K = padded.shape[1]
    valid_all = torch.arange(max_K, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    D = torch.zeros((n, n), dtype=torch.float32, device=device)

    for i_start in range(0, n, tile):
        i_end = min(i_start + tile, n)
        ti = i_end - i_start

        I_tile = padded[i_start:i_end]
        lens_I = lengths[i_start:i_end]
        valid_I = valid_all[i_start:i_end]
        I_flat = I_tile.reshape(ti * max_K, I_tile.shape[2])

        for j_start in range(i_start, n, tile):
            j_end = min(j_start + tile, n)
            tj = j_end - j_start

            J_tile = padded[j_start:j_end]
            lens_J = lengths[j_start:j_end]
            valid_J = valid_all[j_start:j_end]
            J_flat = J_tile.reshape(tj * max_K, J_tile.shape[2])

            dists = torch.cdist(
                I_flat,
                J_flat,
                p=2,
                compute_mode="use_mm_for_euclid_dist_if_necessary",
            )
            dists = dists.reshape(ti, max_K, tj, max_K)

            dists.masked_fill_(~valid_J.view(1, 1, tj, max_K), float("inf"))
            fwd_min = dists.min(dim=3).values
            fwd_min = fwd_min.masked_fill(~valid_I.unsqueeze(2), 0.0)
            fwd = (
                fwd_min.sum(dim=1)
                / lens_I.to(torch.float32).unsqueeze(1).clamp(min=1.0)
            )

            dists.masked_fill_(~valid_I.view(ti, max_K, 1, 1), float("inf"))
            bwd_min = dists.min(dim=1).values
            bwd_min = bwd_min.masked_fill(~valid_J.unsqueeze(0), 0.0)
            bwd = (
                bwd_min.sum(dim=2)
                / lens_J.to(torch.float32).unsqueeze(0).clamp(min=1.0)
            )

            block = fwd + bwd
            if i_start == j_start:
                block = block.clone()
                diag_idx = torch.arange(ti, device=device)
                block[diag_idx, diag_idx] = 0.0
                D[i_start:i_end, j_start:j_end] = block
            else:
                D[i_start:i_end, j_start:j_end] = block
                D[j_start:j_end, i_start:i_end] = block.transpose(0, 1)

        if i_start % (tile * 4) == 0 or i_end == n:
            print(
                f"    Computing Chamfer [red->red]: "
                f"{i_end}/{n} rows done...",
                flush=True,
            )

    synchronize_if_cuda(device)
    print("    Computing Chamfer [red->red]: complete.", flush=True)
    del padded
    return D


def benchmark_exact_chamfer(red_descs, blue_descs, device):
    """
    Compute true Chamfer distance matrix, run ot.emd.
    Guard: raise RuntimeError if N > EXACT_N_LIMIT or ot is None.

    Steps:
    1. D = compute_chamfer_matrix(blue_descs, red_descs, device)
       # D[b, a] = chamfer(blue_b, red_a), shape (N, N) float32 on device
    2. C = D.cpu().to(torch.float64).numpy()
    3. a = b = np.full(n, 1/n, float64)
    4. Time ONLY the ot.emd call.
    5. matching = plan.argmax(axis=0)
    6. cost = D.cpu()[matching, range(N)].mean().item()
       # true Chamfer cost of the matching

    Also wrap ot.emd in try/except MemoryError — fill with nan and warn.
    Returns (time_ms, avg_chamfer_cost, D_for_reuse)
    Return D so the caller can reuse it without recomputing.
    Print: "    Chamfer matrix computed in X ms" (time the cdist step separately)
    """
    n = len(red_descs)
    if n > EXACT_N_LIMIT:
        raise RuntimeError(f"Exact OT skipped for N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    try:
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        D = compute_chamfer_matrix(blue_descs, red_descs, device)
        synchronize_if_cuda(device)
        t1 = time.perf_counter()
        print(
            f"    Chamfer matrix computed in {(t1 - t0) * 1000.0:.1f} ms",
            flush=True,
        )
    except Exception as exc:
        print(f"Warning: Chamfer matrix computation failed: {exc}", flush=True)
        empty_cache_if_cuda(device)
        raise RuntimeError(f"Chamfer matrix computation failed: {exc}") from exc

    try:
        D_cpu = D.detach().cpu()
        C = D_cpu.to(torch.float64).numpy()
        a = np.full(n, 1.0 / n, dtype=np.float64)
        b = np.full(n, 1.0 / n, dtype=np.float64)
        red_indices = torch.arange(n, dtype=torch.long)

        for _ in range(WARMUP_RUNS):
            plan = ot.emd(a, b, C, numItermax=10**6)
            del plan

        times_ms = []
        costs = []
        for _ in range(TIMED_RUNS):
            t_emd0 = time.perf_counter()
            plan = ot.emd(a, b, C, numItermax=10**6)
            t_emd1 = time.perf_counter()
            matching = torch.from_numpy(plan.argmax(axis=0).astype(np.int64, copy=False))
            cost = D_cpu[matching, red_indices].mean().item()
            times_ms.append((t_emd1 - t_emd0) * 1000.0)
            costs.append(cost)
            del plan, matching
    except MemoryError as exc:
        print(f"Warning: Exact OT failed due to memory pressure: {exc}", flush=True)
        empty_cache_if_cuda(device)
        raise RuntimeError(f"Exact OT failed due to memory pressure: {exc}") from exc

    return statistics.median(times_ms), statistics.median(costs), D


def benchmark_simple(P_red_norm, P_blue_norm, device, diameter, D_chamfer=None):
    """
    Run SimpleGPUSolver on normalized mean descriptor vectors.
    Times the full wall time (clustering + solve).

    P_red_norm:  (N, 128) float32 CUDA — normalized mean SIFT descriptors
    P_blue_norm: (N, 128) float32 CUDA — normalized mean SIFT descriptors
    diameter:    float — for converting normalized cost back to original scale
    D_chamfer:   (N, N) float32 CPU tensor, optional — if provided, compute
                 the TRUE Chamfer cost of the push-relabel matching post-hoc.
                 If None, report L2 cost only.

    Steps:
    1. solver = SimpleGPUSolver(
           P_red_norm, P_blue_norm,
           epsilon=EPSILON,
           batch_size=BATCH_SIZE,
           verbose=False,
           diameter=1.0,
           clustering_class=SimpleClustering,
       )
    2. solver.solve()
    3. If D_chamfer is not None:
           matching = solver.match_B.cpu()  # (N,) int64 — match_B[b] = matched red
           cost = D_chamfer[range(N), matching].mean().item()
           # true Chamfer cost of push-relabel matching
       Else:
           cost = torch.norm(P_blue_norm - P_red_norm[solver.match_B],
                             p=2, dim=1).mean().item() * diameter
           # L2 cost in original scale

    Returns (time_ms, cost, iterations)
    Print a note: "    Cost type: true Chamfer" or "    Cost type: L2 (Chamfer unavailable at this N)"
    """
    if D_chamfer is not None:
        print("    Cost type: true Chamfer", flush=True)
    else:
        print("    Cost type: L2 (Chamfer unavailable at this N)", flush=True)

    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        solver = SimpleGPUSolver(
            P_red_norm,
            P_blue_norm,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
            clustering_class=SimpleClustering,
        )
        solver.solve()
        synchronize_if_cuda(device)
        del solver

    times_ms = []
    costs = []
    iterations_list = []
    for _ in range(TIMED_RUNS):
        empty_cache_if_cuda(device)
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        solver = SimpleGPUSolver(
            P_red_norm,
            P_blue_norm,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
            clustering_class=SimpleClustering,
        )
        solver.solve()
        synchronize_if_cuda(device)
        t1 = time.perf_counter()

        if D_chamfer is not None:
            matching = solver.match_B.detach().cpu()
            blue_indices = torch.arange(matching.shape[0], dtype=torch.long)
            cost = D_chamfer[blue_indices, matching].mean().item()
            del matching
        else:
            cost = (
                torch.norm(P_blue_norm - P_red_norm[solver.match_B], p=2, dim=1)
                .mean()
                .item()
                * diameter
            )

        times_ms.append((t1 - t0) * 1000.0)
        costs.append(cost)
        iterations_list.append(solver.iterations)
        del solver

    return (
        statistics.median(times_ms),
        statistics.median(costs),
        statistics.median(iterations_list),
    )


def run_exact(red_descs, blue_descs, device):
    """
    Wraps benchmark_exact_chamfer with try/except.
    Returns {"time_ms", "cost", "D": D_chamfer_tensor_or_None, "status"}
    D is returned for reuse in run_simple.
    On any exception: print warning, return nan for time/cost, None for D.
    """
    try:
        time_ms, cost, D = benchmark_exact_chamfer(red_descs, blue_descs, device)
        return {"time_ms": time_ms, "cost": cost, "D": D, "status": "success"}
    except Exception as exc:
        print(f"Warning: Exact failed: {exc}", flush=True)
        empty_cache_if_cuda(device)
        return {"time_ms": math.nan, "cost": math.nan, "D": None, "status": "fail"}


def run_simple(P_red_norm, P_blue_norm, device, diameter, D_chamfer=None):
    """
    Wraps benchmark_simple with try/except.
    Returns {"time_ms", "cost", "iterations", "cost_type", "status"}
    cost_type is "chamfer" if D_chamfer was provided, "l2" otherwise.
    On any exception: print warning, empty_cache_if_cuda, return nans.
    """
    cost_type = "chamfer" if D_chamfer is not None else "l2"
    try:
        time_ms, cost, iterations = benchmark_simple(
            P_red_norm,
            P_blue_norm,
            device,
            diameter,
            D_chamfer=D_chamfer,
        )
        return {
            "time_ms": time_ms,
            "cost": cost,
            "iterations": iterations,
            "cost_type": cost_type,
            "status": "success",
        }
    except Exception as exc:
        print(f"Warning: Simple failed: {exc}", flush=True)
        empty_cache_if_cuda(device)
        return {
            "time_ms": math.nan,
            "cost": math.nan,
            "iterations": math.nan,
            "cost_type": cost_type,
            "status": "fail",
        }


def _fmt_cost_type(value):
    if value == "chamfer":
        return "Chamfer"
    if value == "l2":
        return "L2"
    return "N/A"


def print_results_table(rows):
    col_widths = {
        "n": 7,
        "exact_time": 14,
        "simple_time": 14,
        "exact_cost": 19,
        "simple_cost": 16,
        "cost_type": 10,
        "simple_iters": 14,
    }
    headers = [
        ("N", col_widths["n"], ">"),
        ("Exact Time", col_widths["exact_time"], ">"),
        ("Simple Time", col_widths["simple_time"], ">"),
        ("Exact Chamfer Cost", col_widths["exact_cost"], ">"),
        ("Simple Cost", col_widths["simple_cost"], ">"),
        ("Cost Type", col_widths["cost_type"], ">"),
        ("Simple Iters", col_widths["simple_iters"], ">"),
    ]

    header_line = " | ".join(
        f"{label:{align}{width}}" for label, width, align in headers
    )
    separator = "-+-".join("-" * width for _, width, _ in headers)

    print(header_line, flush=True)
    print(separator, flush=True)
    for row in rows:
        exact = row["exact"]
        simple = row["simple"]
        cells = [
            f"{row['n']:>{col_widths['n']},}",
            f"{fmt_time(exact['time_ms']):>{col_widths['exact_time']}}",
            f"{fmt_time(simple['time_ms']):>{col_widths['simple_time']}}",
            f"{fmt_cost(exact['cost']):>{col_widths['exact_cost']}}",
            f"{fmt_cost(simple['cost']):>{col_widths['simple_cost']}}",
            f"{_fmt_cost_type(simple['cost_type']):>{col_widths['cost_type']}}",
            f"{fmt_iter(simple['iterations']):>{col_widths['simple_iters']}}",
        ]
        print(" | ".join(cells), flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("=" * 60, flush=True)
    print("Experiment: CIFAR-10 SIFT — Exact OT vs Push-Relabel", flush=True)
    print("=" * 60, flush=True)
    print(f"Device  : {device}", flush=True)
    print(f"Epsilon : {EPSILON}  batch_size={BATCH_SIZE}", flush=True)
    print(f"Exact OT (Chamfer): N ≤ {EXACT_N_LIMIT:,} only", flush=True)
    print(
        "Push-Relabel: all N values (uses mean SIFT descriptor, L2 clustering)",
        flush=True,
    )
    print(
        f"Cost reported: true Chamfer for N ≤ {EXACT_N_LIMIT:,}, "
        f"L2 for larger N",
        flush=True,
    )

    if not TRAIN_DESC_PATH.exists():
        print(f"\nERROR: {TRAIN_DESC_PATH.name} not found.", flush=True)
        print("Run download_cifar_sift.py first.", flush=True)
        return

    print("\n[Data] Loading SIFT descriptors...", flush=True)
    try:
        all_descs = load_cifar_sift_descriptors(TRAIN_DESC_PATH)
    except Exception as exc:
        print(f"ERROR loading data: {exc}", flush=True)
        return
    print(f"[Data] Total images: {len(all_descs):,}", flush=True)

    rows = []
    for n in N_VALUES:
        P_red = None
        P_blue = None
        P_red_raw = None
        P_blue_raw = None
        P_red_norm_cpu = None
        P_blue_norm_cpu = None
        red_descs = None
        blue_descs = None
        red_means = None
        blue_means = None
        D_chamfer_cpu = None
        try:
            print(f"\n{'=' * 40}", flush=True)
            print(f"N = {n:,}", flush=True)
            print(f"{'=' * 40}", flush=True)

            if 2 * n > len(all_descs):
                print(
                    f"  Not enough images (need {2 * n:,}, have {len(all_descs):,}). "
                    f"Skipping.",
                    flush=True,
                )
                continue

            print("  [1/3] Sampling descriptors...", flush=True)
            try:
                red_descs, blue_descs = sample_pair(all_descs, n, SEED)
            except Exception as exc:
                print(f"  Sampling failed: {exc}", flush=True)
                continue

            print("  [2/3] Computing mean descriptors...", flush=True)
            try:
                red_means = compute_mean_descriptors(red_descs)
                blue_means = compute_mean_descriptors(blue_descs)
                P_red_raw = torch.from_numpy(red_means).float()
                P_blue_raw = torch.from_numpy(blue_means).float()
                P_red_norm_cpu, P_blue_norm_cpu, diameter = normalize_points(
                    P_red_raw,
                    P_blue_raw,
                )
                P_red = P_red_norm_cpu.to(device)
                P_blue = P_blue_norm_cpu.to(device)
                print(f"  Mean descriptor diameter: {diameter:.4f}", flush=True)
            except Exception as exc:
                print(f"  Mean descriptor computation failed: {exc}", flush=True)
                continue

            print("  [3/3] Running solvers...", flush=True)

            D_chamfer_cpu = None
            if n <= EXACT_N_LIMIT:
                print("    Running Exact OT (true Chamfer)...", flush=True)
                exact_result = run_exact(red_descs, blue_descs, device)
                if exact_result["D"] is not None:
                    D_chamfer_cpu = exact_result["D"].detach().cpu()
                    exact_result["D"] = None
                    empty_cache_if_cuda(device)
            else:
                print("    Exact OT skipped (N > EXACT_N_LIMIT)", flush=True)
                exact_result = {
                    "time_ms": math.nan,
                    "cost": math.nan,
                    "D": None,
                    "status": "skipped",
                }

            print(
                "    Running Push-Relabel (mean SIFT, L2 clustering)...",
                flush=True,
            )
            simple_result = run_simple(
                P_red,
                P_blue,
                device,
                diameter,
                D_chamfer=D_chamfer_cpu,
            )

            rows.append(
                {
                    "n": n,
                    "exact": exact_result,
                    "simple": simple_result,
                }
            )

            print(f"\n--- Results so far (after N={n:,}) ---", flush=True)
            print_results_table(rows)
            print(f"--- End of intermediate results ---\n", flush=True)
        except Exception as exc:
            print(f"  Unexpected failure at N={n:,}: {exc}", flush=True)
            empty_cache_if_cuda(device)
            continue
        finally:
            if P_red is not None:
                del P_red
            if P_blue is not None:
                del P_blue
            if P_red_raw is not None:
                del P_red_raw
            if P_blue_raw is not None:
                del P_blue_raw
            if P_red_norm_cpu is not None:
                del P_red_norm_cpu
            if P_blue_norm_cpu is not None:
                del P_blue_norm_cpu
            if red_descs is not None:
                del red_descs
            if blue_descs is not None:
                del blue_descs
            if red_means is not None:
                del red_means
            if blue_means is not None:
                del blue_means
            if D_chamfer_cpu is not None:
                del D_chamfer_cpu
            empty_cache_if_cuda(device)

    print(f"\n\n{'=' * 60}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(
        f"Note: Simple Cost = true Chamfer for N ≤ {EXACT_N_LIMIT:,}; "
        f"L2 for larger N",
        flush=True,
    )
    print_results_table(rows)


if __name__ == "__main__":
    main()
