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
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clustered_push_relabel.clustering.simple_precomputed import (
    SimplePrecomputedClustering,
)


DATA_DIR = BASE_DIR / "data" / "newsgroups_glove"
EMBEDDINGS_PATH = DATA_DIR / "newsgroups_embeddings.pkl.gz"
LABELS_PATH = DATA_DIR / "newsgroups_labels.npy"

N_VALUES = [1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000]
EPSILON = 0.01
BATCH_SIZE = 512
MAX_WORDS_PER_DOC = 300
# Documents with more unique words are truncated to this limit.
# Average words/doc in 20 Newsgroups is ~65; max is 2509 (outlier).
# Capping at 300 covers >99% of documents without truncation and
# is standard preprocessing in the WMD literature.
SEED = 42
WARMUP_RUNS = 0
TIMED_RUNS = 1
EXACT_N_LIMIT = 10_000


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


def fmt_ratio(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.4f}"


def load_document_embeddings(path):
    """
    Load gzip-pickled document embedding list.
    Returns a Python list of numpy arrays, one per doc, each (K_i, 300).
    """
    with gzip.open(path, "rb") as f:
        embeddings = pickle.load(f)
    print(
        f"  Loaded {len(embeddings):,} document embedding sets from {path.name}",
        flush=True,
    )
    return embeddings


def sample_document_pair(all_embeddings, n_samples, seed):
    """
    Sample two non-overlapping sets of n_samples documents.
    """
    if len(all_embeddings) < 2 * n_samples:
        raise ValueError(
            f"Need at least {2 * n_samples:,} documents, got "
            f"{len(all_embeddings):,}."
        )

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_embeddings))
    red_idx = perm[:n_samples]
    blue_idx = perm[n_samples : 2 * n_samples]
    red_descs = [all_embeddings[i] for i in red_idx]
    blue_descs = [all_embeddings[i] for i in blue_idx]
    return red_descs, blue_descs


def _pad_embeddings(descs_list, device):
    """
    Pad all document embedding sets to max_K and stack.
    """
    # Truncate to MAX_WORDS_PER_DOC: bounds tile memory to
    # tile_b * MAX_WORDS_PER_DOC * tile_a * MAX_WORDS_PER_DOC * 4 bytes
    # = 50 * 300 * 100 * 300 * 4 = ~1.8 GB per tile (manageable)
    descs_list = [d[:MAX_WORDS_PER_DOC] if len(d) > MAX_WORDS_PER_DOC else d for d in descs_list]

    n = len(descs_list)
    max_K = max(desc.shape[0] for desc in descs_list)
    dim = descs_list[0].shape[1]

    padded = torch.zeros((n, max_K, dim), dtype=torch.float32, device=device)
    lengths = torch.empty(n, dtype=torch.int64, device=device)

    for i, desc in enumerate(descs_list):
        K_i = desc.shape[0]
        padded[i, :K_i, :] = torch.as_tensor(
            desc,
            dtype=torch.float32,
            device=device,
        )
        lengths[i] = K_i

    return padded, lengths


# tile_a=50: with MAX_WORDS_PER_DOC=300, each tile uses
# 50*300 * 50*300 * 4 = ~900MB. Reduce further if OOM persists.
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

    # Sort by word count: tiles of short docs use local max_K~50 instead
    # of global max_K=300, reducing cdist size by up to 36x for short tiles.
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

    # Sort by word count: tiles of short docs use local max_K~50 instead
    # of global max_K=300, reducing cdist size by up to 36x for short tiles.
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


def normalize_chamfer_matrices(D_blue_to_red, D_red_to_red):
    """
    Normalize both matrices by their joint maximum value.
    """
    diameter = max(
        float(torch.maximum(D_blue_to_red.max(), D_red_to_red.max()).item()),
        1e-6,
    )
    return D_blue_to_red / diameter, D_red_to_red / diameter, diameter


def build_proxy_cost_matrix(clustering, N, device):
    """
    Build full N x N float proxy cost matrix from SimplePrecomputedClustering.
    """
    DR = clustering["DR"]
    d_min_b = clustering["d_min_b"]
    nearest_s = clustering["nearest_s"]
    adj_ptr = clustering["adj_ptr"]
    adj_col = clustering["adj_col"]
    adj_dist_float = clustering["adj_dist_float"]

    C = d_min_b.unsqueeze(1) + DR[nearest_s, :]
    if adj_col.numel() > 0:
        b_indices = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            adj_ptr[1:] - adj_ptr[:-1],
        )
        C[b_indices, adj_col] = adj_dist_float
    return C.cpu().to(torch.float64).numpy()


def benchmark_exact(D_blue_to_red_norm, device):
    """
    Run ot.emd with true normalized Chamfer distances.
    """
    N = D_blue_to_red_norm.shape[0]
    if N > EXACT_N_LIMIT:
        raise RuntimeError(f"Exact OT skipped for N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    D_cpu = D_blue_to_red_norm.detach().cpu()
    a = np.full(N, 1.0 / N, np.float64)
    b = np.full(N, 1.0 / N, np.float64)

    try:
        # POT expects rows for the first marginal. We transpose so
        # plan.argmax(axis=0) returns the matched red document per blue document.
        C = D_cpu.to(torch.float64).numpy().T
    except MemoryError as exc:
        raise RuntimeError(
            f"Exact skipped at N={N}: CPU memory insufficient "
            f"for {N}x{N} float64 matrix"
        ) from exc

    for _ in range(WARMUP_RUNS):
        try:
            plan = ot.emd(a, b, C, numItermax=10**6)
        except MemoryError as exc:
            raise RuntimeError(
                f"Exact skipped at N={N}: CPU memory insufficient "
                f"for {N}x{N} float64 matrix"
            ) from exc
        del plan

    times_ms = []
    costs = []
    blue_indices = torch.arange(N, dtype=torch.long)
    for _ in range(TIMED_RUNS):
        try:
            t0 = time.perf_counter()
            plan = ot.emd(a, b, C, numItermax=10**6)
            t1 = time.perf_counter()
        except MemoryError as exc:
            raise RuntimeError(
                f"Exact skipped at N={N}: CPU memory insufficient "
                f"for {N}x{N} float64 matrix"
            ) from exc

        matching = torch.from_numpy(plan.argmax(axis=0).astype(np.int64, copy=False))
        cost = D_cpu[blue_indices, matching].mean().item()
        times_ms.append((t1 - t0) * 1000.0)
        costs.append(cost)
        del plan, matching

    return statistics.median(times_ms), statistics.median(costs)


def benchmark_proxy_exact(D_blue_to_red_norm, D_red_to_red_norm, device):
    """
    Run SimplePrecomputedClustering then ot.emd with proxy distances.
    """
    N = D_blue_to_red_norm.shape[0]
    if N > EXACT_N_LIMIT:
        raise RuntimeError(f"Proxy-Exact OT skipped for N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    a = np.full(N, 1.0 / N, np.float64)
    b = np.full(N, 1.0 / N, np.float64)
    D_rr = D_red_to_red_norm.to(device)
    D_br = D_blue_to_red_norm.to(device)
    D_cpu = D_blue_to_red_norm.detach().cpu()
    blue_indices = torch.arange(N, dtype=torch.long)

    clustering_times_ms = []
    emd_times_ms = []
    costs = []

    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        cluster_engine = SimplePrecomputedClustering(
            epsilon=EPSILON,
            tile_size=BATCH_SIZE,
        )
        clustering = cluster_engine.run(D_rr, D_br)
        try:
            C_proxy = build_proxy_cost_matrix(clustering, N, device)
            plan = ot.emd(a, b, C_proxy.T, numItermax=10**6)
        except MemoryError as exc:
            raise RuntimeError(
                f"ProxyExact skipped at N={N}: CPU memory insufficient "
                f"for {N}x{N} float64 matrix"
            ) from exc
        del cluster_engine, clustering, C_proxy, plan

    for _ in range(TIMED_RUNS):
        empty_cache_if_cuda(device)
        cluster_engine = SimplePrecomputedClustering(
            epsilon=EPSILON,
            tile_size=BATCH_SIZE,
        )

        synchronize_if_cuda(device)
        t_cluster0 = time.perf_counter()
        clustering = cluster_engine.run(D_rr, D_br)
        synchronize_if_cuda(device)
        t_cluster1 = time.perf_counter()
        clustering_times_ms.append((t_cluster1 - t_cluster0) * 1000.0)

        try:
            C_proxy = build_proxy_cost_matrix(clustering, N, device)
            t0 = time.perf_counter()
            plan = ot.emd(a, b, C_proxy.T, numItermax=10**6)
            t1 = time.perf_counter()
        except MemoryError as exc:
            raise RuntimeError(
                f"ProxyExact skipped at N={N}: CPU memory insufficient "
                f"for {N}x{N} float64 matrix"
            ) from exc

        matching = torch.from_numpy(plan.argmax(axis=0).astype(np.int64, copy=False))
        cost = D_cpu[blue_indices, matching].mean().item()
        emd_times_ms.append((t1 - t0) * 1000.0)
        costs.append(cost)

        del cluster_engine, clustering, C_proxy, plan, matching

    print(
        f"    Clustering time (excluded): "
        f"{statistics.median(clustering_times_ms):.1f} ms",
        flush=True,
    )
    return statistics.median(emd_times_ms), statistics.median(costs)


def run_exact(D_blue_to_red_norm, device):
    try:
        time_ms, cost = benchmark_exact(D_blue_to_red_norm, device)
        return {"time_ms": time_ms, "cost": cost, "status": "success"}
    except Exception as exc:
        print(f"Warning: Exact failed: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


def run_proxy_exact(D_blue_to_red_norm, D_red_to_red_norm, device):
    try:
        time_ms, cost = benchmark_proxy_exact(
            D_blue_to_red_norm,
            D_red_to_red_norm,
            device,
        )
        return {"time_ms": time_ms, "cost": cost, "status": "success"}
    except Exception as exc:
        print(f"Warning: ProxyExact failed: {exc}", flush=True)
        empty_cache_if_cuda(device)
        return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


def compute_cost_ratio(exact_cost, proxy_cost):
    if math.isnan(exact_cost) or math.isnan(proxy_cost) or exact_cost == 0.0:
        return math.nan
    return proxy_cost / exact_cost


def print_results_table(rows):
    col_widths = {
        "n": 7,
        "exact_time": 14,
        "proxy_time": 19,
        "exact_cost": 12,
        "proxy_cost": 17,
        "ratio": 7,
    }
    headers = [
        ("N", col_widths["n"], ">"),
        ("Exact Time", col_widths["exact_time"], ">"),
        ("ProxyExact Time", col_widths["proxy_time"], ">"),
        ("Exact Cost", col_widths["exact_cost"], ">"),
        ("ProxyExact Cost", col_widths["proxy_cost"], ">"),
        ("Ratio", col_widths["ratio"], ">"),
    ]

    header_line = " | ".join(
        f"{label:{align}{width}}" for label, width, align in headers
    )
    separator = "-+-".join("-" * width for _, width, _ in headers)

    print(header_line, flush=True)
    print(separator, flush=True)
    for row in rows:
        exact = row["exact"]
        proxy = row["proxy_exact"]
        ratio = compute_cost_ratio(exact["cost"], proxy["cost"])
        cells = [
            f"{row['n']:>{col_widths['n']},}",
            f"{fmt_time(exact['time_ms']):>{col_widths['exact_time']}}",
            f"{fmt_time(proxy['time_ms']):>{col_widths['proxy_time']}}",
            f"{fmt_cost(exact['cost']):>{col_widths['exact_cost']}}",
            f"{fmt_cost(proxy['cost']):>{col_widths['proxy_cost']}}",
            f"{fmt_ratio(ratio):>{col_widths['ratio']}}",
        ]
        print(" | ".join(cells), flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("=" * 60, flush=True)
    print("Experiment: 20 Newsgroups Document Chamfer Proxy Quality", flush=True)
    print("=" * 60, flush=True)
    print(f"Device  : {device}", flush=True)
    print(f"Epsilon : {EPSILON}  batch_size={BATCH_SIZE}", flush=True)
    print(
        f"Exact OT: attempted up to N={EXACT_N_LIMIT:,} "
        f"(skipped on MemoryError)",
        flush=True,
    )
    print("Distance: Symmetric Chamfer in GloVe 300d space", flush=True)

    for path in [EMBEDDINGS_PATH]:
        if not path.exists():
            print(f"\nERROR: {path.name} not found.", flush=True)
            print("Run download_newsgroups_glove.py first.", flush=True)
            return

    print("\n[Data] Loading document embeddings...", flush=True)
    all_embeddings = load_document_embeddings(EMBEDDINGS_PATH)
    print(
        f"[Data] Total documents available: {len(all_embeddings):,}",
        flush=True,
    )

    rows = []
    for n in N_VALUES:
        print(f"\n{'=' * 40}", flush=True)
        print(f"N = {n:,}", flush=True)
        print(f"{'=' * 40}", flush=True)

        if 2 * n > len(all_embeddings):
            print(
                f"  Not enough documents (need {2 * n:,}, "
                f"have {len(all_embeddings):,}). Skipping.",
                flush=True,
            )
            continue

        print(
            f"  [1/4] Sampling {n:,} red and {n:,} blue documents...",
            flush=True,
        )
        try:
            red_descs, blue_descs = sample_document_pair(
                all_embeddings,
                n,
                SEED,
            )
        except Exception as exc:
            print(f"  Sampling failed: {exc}", flush=True)
            continue

        print(f"  [2/4] Computing Chamfer distance matrices on {device}...", flush=True)
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        D_blue_to_red = compute_chamfer_matrix(blue_descs, red_descs, device)
        D_red_to_red = compute_chamfer_matrix_symmetric(red_descs, device)
        D_red_to_red.fill_diagonal_(0.0)
        synchronize_if_cuda(device)
        chamfer_time = (time.perf_counter() - t0) * 1000.0
        print(
            f"  Chamfer matrices computed in {chamfer_time:.1f} ms",
            flush=True,
        )

        print("  [3/4] Normalizing...", flush=True)
        D_br_norm, D_rr_norm, diameter = normalize_chamfer_matrices(
            D_blue_to_red,
            D_red_to_red,
        )
        print(f"  Diameter (max Chamfer): {diameter:.4f}", flush=True)
        del D_blue_to_red, D_red_to_red

        print("  [4/4] Running solvers...", flush=True)
        print("    Running Exact OT...", flush=True)
        exact_result = run_exact(D_br_norm, device)

        print("    Running Proxy-Exact OT...", flush=True)
        proxy_result = run_proxy_exact(D_br_norm, D_rr_norm, device)

        rows.append(
            {
                "n": n,
                "exact": exact_result,
                "proxy_exact": proxy_result,
            }
        )

        del D_br_norm, D_rr_norm
        empty_cache_if_cuda(device)

    print(f"\n\n{'=' * 60}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print_results_table(rows)


if __name__ == "__main__":
    main()
