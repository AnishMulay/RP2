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


DATA_DIR = BASE_DIR / "data" / "newsgroups_glove"
EMBEDDINGS_PATH = DATA_DIR / "newsgroups_embeddings.pkl.gz"

N_VALUES_EXACT = [1_000, 2_000, 5_000]
N_VALUES_SCALE = [7_000, 9_000]
N_VALUES = N_VALUES_EXACT + N_VALUES_SCALE
EXACT_N_LIMIT = 5_000
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
MAX_WORDS_PER_DOC = 300
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


def load_embeddings(path):
    """Load pkl.gz -> list of numpy arrays. Same as in newsgroups_proxy."""
    with gzip.open(path, "rb") as f:
        embs = pickle.load(f)
    print(
        f"  Loaded {len(embs):,} document embedding sets from {path.name}",
        flush=True,
    )
    return embs


def compute_mean_embeddings(descs_list):
    """
    Compute mean word embedding per document.
    For each document i: mean_emb[i] = descs_list[i][:MAX_WORDS_PER_DOC].mean(axis=0)
    Documents with 0 valid words (shape (1, 300) zero fallback) return zero vector.
    Returns numpy array of shape (N, 300) float32.
    """
    means = np.zeros((len(descs_list), 300), dtype=np.float32)
    for i, desc in enumerate(descs_list):
        arr = np.asarray(desc, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 300:
            raise ValueError(
                f"Embedding set {i} has shape {arr.shape}; expected (K, 300)."
            )
        arr = arr[:MAX_WORDS_PER_DOC]
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
            f"Need at least {2 * n_samples:,} documents, got {len(all_descs):,}."
        )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_descs))
    red_descs = [all_descs[i] for i in perm[:n_samples]]
    blue_descs = [all_descs[i] for i in perm[n_samples : 2 * n_samples]]
    return red_descs, blue_descs


def _pad_embeddings(descs_list, device):
    """
    Pad all document embedding sets to max_K and stack.
    """
    descs_list = [
        d[:MAX_WORDS_PER_DOC] if len(d) > MAX_WORDS_PER_DOC else d
        for d in descs_list
    ]

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
            desc,
            dtype=torch.float32,
            device=device,
        )
        lengths[i] = K_i

    return padded, lengths


def compute_chamfer_matrix(descs_B, descs_A, device, tile_b=50, tile_a=50):
    """
    Compute D[b, a] = chamfer(descs_B[b], descs_A[a]) for all b, a.
    Returns (N, N) float32 tensor on device.

    For 300d word embeddings, document vocabularies can be much larger than
    SIFT keypoint sets. Reduce `tile_a` if this runs out of GPU memory.
    """
    N = len(descs_B)
    if len(descs_A) != N:
        raise ValueError("descs_B and descs_A must have the same length")

    order_B = sorted(range(len(descs_B)), key=lambda i: len(descs_B[i]))
    order_A = sorted(range(len(descs_A)), key=lambda i: len(descs_A[i]))
    descs_B_sorted = [descs_B[i] for i in order_B]
    descs_A_sorted = [descs_A[i] for i in order_A]

    D_sorted = torch.zeros((N, N), dtype=torch.float32, device=device)

    for b_start in range(0, N, tile_b):
        b_end = min(b_start + tile_b, N)
        B_tile_descs = descs_B_sorted[b_start:b_end]
        padded_B_tile, lengths_B_tile = _pad_embeddings(B_tile_descs, device)
        tb = b_end - b_start
        local_max_KB = padded_B_tile.shape[1]
        B_flat = padded_B_tile.reshape(tb * local_max_KB, padded_B_tile.shape[2])
        valid_B = (
            torch.arange(local_max_KB, device=device).unsqueeze(0)
            < lengths_B_tile.unsqueeze(1)
        )

        for a_start in range(0, N, tile_a):
            a_end = min(a_start + tile_a, N)
            A_tile_descs = descs_A_sorted[a_start:a_end]
            padded_A_tile, lengths_A_tile = _pad_embeddings(A_tile_descs, device)
            ta = a_end - a_start
            local_max_KA = padded_A_tile.shape[1]
            A_flat = padded_A_tile.reshape(ta * local_max_KA, padded_A_tile.shape[2])
            valid_A = (
                torch.arange(local_max_KA, device=device).unsqueeze(0)
                < lengths_A_tile.unsqueeze(1)
            )

            dists = torch.cdist(
                B_flat,
                A_flat,
                p=2,
                compute_mode="use_mm_for_euclid_dist_if_necessary",
            )
            dists = dists.reshape(tb, local_max_KB, ta, local_max_KA)

            dists.masked_fill_(~valid_A.view(1, 1, ta, local_max_KA), float("inf"))
            fwd_min = dists.min(dim=3).values
            fwd_min = fwd_min.masked_fill(~valid_B.unsqueeze(2), 0.0)
            fwd = (
                fwd_min.sum(dim=1)
                / lengths_B_tile.to(torch.float32).unsqueeze(1).clamp(min=1.0)
            )

            dists.masked_fill_(~valid_B.view(tb, local_max_KB, 1, 1), float("inf"))
            bwd_min = dists.min(dim=1).values
            bwd_min = bwd_min.masked_fill(~valid_A.unsqueeze(0), 0.0)
            bwd = (
                bwd_min.sum(dim=2)
                / lengths_A_tile.to(torch.float32).unsqueeze(0).clamp(min=1.0)
            )

            D_sorted[b_start:b_end, a_start:a_end] = fwd + bwd
            del padded_A_tile, A_flat, dists, fwd_min, bwd_min, fwd, bwd, valid_A

        if b_start % (tile_b * 4) == 0 or b_end == N:
            print(
                f"    Computing Chamfer [blue->red]: "
                f"{b_end}/{N} rows done...",
                flush=True,
            )
        del padded_B_tile, B_flat, valid_B

    inv_order_B = torch.zeros(N, dtype=torch.long, device=device)
    inv_order_A = torch.zeros(N, dtype=torch.long, device=device)
    for i, o in enumerate(order_B):
        inv_order_B[o] = i
    for i, o in enumerate(order_A):
        inv_order_A[o] = i

    D = D_sorted[inv_order_B][:, inv_order_A]
    del D_sorted
    synchronize_if_cuda(device)
    print("    Computing Chamfer [blue->red]: complete.", flush=True)
    return D


def compute_chamfer_matrix_symmetric(descs, device, tile=50):
    """
    Symmetric N x N Chamfer matrix for a single set of documents.
    D[i, j] = D[j, i], D[i, i] = 0.
    """
    N = len(descs)

    order = sorted(range(len(descs)), key=lambda i: len(descs[i]))
    descs_sorted = [descs[i] for i in order]

    D_sorted = torch.zeros((N, N), dtype=torch.float32, device=device)

    for i_start in range(0, N, tile):
        i_end = min(i_start + tile, N)
        I_tile_descs = descs_sorted[i_start:i_end]
        padded_I_tile, lengths_I_tile = _pad_embeddings(I_tile_descs, device)
        ti = i_end - i_start
        local_max_KI = padded_I_tile.shape[1]
        I_flat = padded_I_tile.reshape(ti * local_max_KI, padded_I_tile.shape[2])
        valid_I = (
            torch.arange(local_max_KI, device=device).unsqueeze(0)
            < lengths_I_tile.unsqueeze(1)
        )

        for j_start in range(i_start, N, tile):
            j_end = min(j_start + tile, N)
            J_tile_descs = descs_sorted[j_start:j_end]
            padded_J_tile, lengths_J_tile = _pad_embeddings(J_tile_descs, device)
            tj = j_end - j_start
            local_max_KJ = padded_J_tile.shape[1]
            J_flat = padded_J_tile.reshape(tj * local_max_KJ, padded_J_tile.shape[2])
            valid_J = (
                torch.arange(local_max_KJ, device=device).unsqueeze(0)
                < lengths_J_tile.unsqueeze(1)
            )

            dists = torch.cdist(
                I_flat,
                J_flat,
                p=2,
                compute_mode="use_mm_for_euclid_dist_if_necessary",
            )
            dists = dists.reshape(ti, local_max_KI, tj, local_max_KJ)

            dists.masked_fill_(~valid_J.view(1, 1, tj, local_max_KJ), float("inf"))
            fwd_min = dists.min(dim=3).values
            fwd_min = fwd_min.masked_fill(~valid_I.unsqueeze(2), 0.0)
            fwd = (
                fwd_min.sum(dim=1)
                / lengths_I_tile.to(torch.float32).unsqueeze(1).clamp(min=1.0)
            )

            dists.masked_fill_(~valid_I.view(ti, local_max_KI, 1, 1), float("inf"))
            bwd_min = dists.min(dim=1).values
            bwd_min = bwd_min.masked_fill(~valid_J.unsqueeze(0), 0.0)
            bwd = (
                bwd_min.sum(dim=2)
                / lengths_J_tile.to(torch.float32).unsqueeze(0).clamp(min=1.0)
            )

            block = fwd + bwd
            if i_start == j_start:
                block = block.clone()
                diag_idx = torch.arange(ti, device=device)
                block[diag_idx, diag_idx] = 0.0
                D_sorted[i_start:i_end, j_start:j_end] = block
            else:
                D_sorted[i_start:i_end, j_start:j_end] = block
                D_sorted[j_start:j_end, i_start:i_end] = block.transpose(0, 1)
            del padded_J_tile, J_flat, dists, fwd_min, bwd_min, fwd, bwd, valid_J

        if i_start % (tile * 4) == 0 or i_end == N:
            print(
                f"    Computing Chamfer [red->red]: "
                f"{i_end}/{N} rows done...",
                flush=True,
            )
        del padded_I_tile, I_flat, valid_I

    inv_order = torch.zeros(N, dtype=torch.long, device=device)
    for i, o in enumerate(order):
        inv_order[o] = i

    D = D_sorted[inv_order][:, inv_order]
    del D_sorted
    D.fill_diagonal_(0.0)
    synchronize_if_cuda(device)
    print("    Computing Chamfer [red->red]: complete.", flush=True)
    return D


def benchmark_exact_chamfer(red_descs, blue_descs, device):
    """
    Compute true Chamfer distance matrix, run ot.emd.
    Guard: raise RuntimeError if N > EXACT_N_LIMIT or ot is None.
    """
    n = len(red_descs)
    if n > EXACT_N_LIMIT:
        raise RuntimeError(f"Exact OT skipped for N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    try:
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        D = compute_chamfer_matrix(
            blue_descs,
            red_descs,
            device,
            tile_b=50,
            tile_a=50,
        )
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
    Run SimpleGPUSolver on normalized mean embedding vectors.
    Times the full wall time (clustering + solve).
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
    print("Experiment: 20 Newsgroups — Exact OT vs Push-Relabel", flush=True)
    print("=" * 60, flush=True)
    print(f"Device  : {device}", flush=True)
    print(f"Epsilon : {EPSILON}  batch_size={BATCH_SIZE}", flush=True)
    print(f"Exact OT (Chamfer): N ≤ {EXACT_N_LIMIT:,} only", flush=True)
    print(
        "Push-Relabel: all N values (uses mean word embedding, L2 clustering)",
        flush=True,
    )
    print(
        f"Cost reported: true Chamfer for N ≤ {EXACT_N_LIMIT:,}, "
        f"L2 for larger N",
        flush=True,
    )

    if not EMBEDDINGS_PATH.exists():
        print(f"\nERROR: {EMBEDDINGS_PATH.name} not found.", flush=True)
        print("Run download_newsgroups_glove.py first.", flush=True)
        return

    print("\n[Data] Loading document embeddings...", flush=True)
    try:
        all_descs = load_embeddings(EMBEDDINGS_PATH)
    except Exception as exc:
        print(f"ERROR loading data: {exc}", flush=True)
        return
    print(f"[Data] Total documents: {len(all_descs):,}", flush=True)

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
                    f"  Not enough documents (need {2 * n:,}, have {len(all_descs):,}). "
                    f"Skipping.",
                    flush=True,
                )
                continue

            print("  [1/3] Sampling document embeddings...", flush=True)
            try:
                red_descs, blue_descs = sample_pair(all_descs, n, SEED)
            except Exception as exc:
                print(f"  Sampling failed: {exc}", flush=True)
                continue

            print("  [2/3] Computing mean embeddings...", flush=True)
            try:
                red_means = compute_mean_embeddings(red_descs)
                blue_means = compute_mean_embeddings(blue_descs)
                P_red_raw = torch.from_numpy(red_means).float()
                P_blue_raw = torch.from_numpy(blue_means).float()
                P_red_norm_cpu, P_blue_norm_cpu, diameter = normalize_points(
                    P_red_raw,
                    P_blue_raw,
                )
                P_red = P_red_norm_cpu.to(device)
                P_blue = P_blue_norm_cpu.to(device)
                print(f"  Mean embedding diameter: {diameter:.4f}", flush=True)
            except Exception as exc:
                print(f"  Mean embedding computation failed: {exc}", flush=True)
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
                "    Running Push-Relabel (mean word embedding, L2 clustering)...",
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
