#!/usr/bin/env python3
"""SIFT10K recall and distance-ratio experiment.

Loads siftsmall from the siftsmall/ directory at the repo root, builds the
clustering cover and HNSW index, then reports:
  - recall@k  (k = 10, 20, 50, 100) for HNSW alone and clustering-aided
  - mean / max distance ratio per rank (returned dist / true dist)
  - a recall-vs-k plot and a distance-ratio-vs-rank plot
  - a CSV with all per-query per-k numbers

Usage (on the cluster):
    python experiments/runners/sift_recall_experiment.py
    python experiments/runners/sift_recall_experiment.py --no_normalize
    python experiments/runners/sift_recall_experiment.py --k_search_multiplier 4
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import hnswlib
except ImportError as exc:
    raise ImportError("hnswlib is required: pip install hnswlib") from exc

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
SIFT_DIR = BASE_DIR / "siftsmall"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cluster_search import CoverIndex, build_cover, cluster_search
from clustered_push_relabel.utils.distance import TiledEuclideanKernel

K_VALUES = [10, 20, 50, 100]
EF_CONSTRUCTION = 200


# ---------------------------------------------------------------------------
# SIFT file loaders
# ---------------------------------------------------------------------------

def read_fvecs(path: pathlib.Path) -> np.ndarray:
    """Reads a .fvecs file and returns a float32 array of shape (N, D)."""
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    # each vector: 1 int32 (dim) + d float32 values
    n = len(data) // (1 + d)
    data = data.reshape(n, 1 + d)
    return data[:, 1:].view(np.float32)


def read_ivecs(path: pathlib.Path) -> np.ndarray:
    """Reads an .ivecs file and returns an int32 array of shape (N, K)."""
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    k = data[0]
    n = len(data) // (1 + k)
    data = data.reshape(n, 1 + k)
    return data[:, 1:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SIFT10K recall + distance-ratio experiment")
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--k_cluster", type=int, default=4,
                   help="Number of hierarchy levels for build_cover")
    p.add_argument("--M", type=int, default=16,
                   help="HNSW M parameter (base-layer connections)")
    p.add_argument("--k_search_multiplier", type=int, default=2,
                   help="Seeds passed to cluster_search = k * this multiplier")
    p.add_argument("--no_normalize", action="store_true",
                   help="Skip L2 normalization of the dataset and queries")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Index helpers  (mirrors hnsw_recall_experiment.py exactly)
# ---------------------------------------------------------------------------

def build_hnsw_index(dataset_cpu: np.ndarray, m: int) -> hnswlib.Index:
    n, dim = dataset_cpu.shape
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=m)
    index.add_items(dataset_cpu, np.arange(n, dtype=np.int32))
    return index


def query_hnsw(index: hnswlib.Index, query_vector: np.ndarray, k_search: int) -> list[int]:
    index.set_ef(max(k_search * 2, 50))
    labels, _ = index.knn_query(query_vector, k=k_search)
    return [int(label) for label in labels[0]]


def topk_from_distances(
    distances_sq: torch.Tensor,
    k: int,
    candidate_ids: list[int] | None = None,
) -> list[int]:
    if candidate_ids is None:
        k_eff = min(k, distances_sq.numel())
        return torch.topk(distances_sq, k=k_eff, largest=False).indices.tolist()
    if not candidate_ids:
        return []
    candidate_tensor = torch.tensor(candidate_ids, device=distances_sq.device, dtype=torch.long)
    candidate_distances = distances_sq.index_select(0, candidate_tensor)
    k_eff = min(k, candidate_tensor.numel())
    topk_local = torch.topk(candidate_distances, k=k_eff, largest=False).indices
    return candidate_tensor[topk_local].tolist()


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    dataset: torch.Tensor,
    queries: torch.Tensor,
    queries_cpu: np.ndarray,
    cover_index: CoverIndex,
    hnsw_index: hnswlib.Index,
    epsilon: float,
    m: int,
    k_search_multiplier: int,
) -> tuple[dict, dict, dict, dict, list[dict]]:
    """
    Returns:
        hnsw_recall        : {k: mean recall}
        cluster_recall     : {k: mean recall}
        ratio_mean_per_rank: {k: list of mean ratios indexed by rank 0..k-1}
        ratio_max_per_rank : {k: list of max  ratios indexed by rank 0..k-1}
        rows               : list of per-query dicts for CSV export
    """
    kernel = TiledEuclideanKernel(chunk_size=4096)
    workspace = kernel.prepare_workspace(dataset)

    n_queries = queries.shape[0]

    # accumulators
    hnsw_hits   = {k: 0.0 for k in K_VALUES}
    cluster_hits = {k: 0.0 for k in K_VALUES}
    ratio_sum   = {k: np.zeros(k) for k in K_VALUES}
    ratio_max   = {k: np.zeros(k) for k in K_VALUES}
    rows: list[dict] = []

    for q_idx in range(n_queries):
        if q_idx % 10 == 0:
            print(f"  query {q_idx+1}/{n_queries}")

        query = queries[q_idx : q_idx + 1]                      # (1, D) on GPU
        query_cpu = queries_cpu[q_idx : q_idx + 1]              # (1, D) CPU numpy

        # squared L2 distances from this query to every dataset point
        distances_sq = kernel.compute_dist_tile(query, workspace).squeeze(1)  # (N,)

        for k in K_VALUES:
            k_search = k * k_search_multiplier

            # --- brute-force ground truth (from distances_sq) ---
            true_topk_ids = topk_from_distances(distances_sq, k)          # list[int], length k
            true_topk_set = set(true_topk_ids)
            true_dists = torch.sqrt(distances_sq[
                torch.tensor(true_topk_ids, device=distances_sq.device)
            ]).cpu().numpy()                                               # (k,) true L2 distances

            # --- HNSW alone ---
            hnsw_result = query_hnsw(hnsw_index, query_cpu, k_search)[:k]
            hnsw_hits[k] += len(true_topk_set.intersection(hnsw_result)) / k

            # --- clustering-aided ---
            seeds = query_hnsw(hnsw_index, query_cpu, k_search=m)
            candidates = cluster_search(
                seeds, cover_index, dataset, kernel, k_prime=k_search, epsilon=epsilon
            )
            cluster_result = topk_from_distances(distances_sq, k, candidates)
            cluster_hits[k] += len(true_topk_set.intersection(cluster_result)) / k

            # --- distance ratio per rank ---
            ret_ids = cluster_result[:k]
            for rank, pid in enumerate(ret_ids):
                ret_dist = float(torch.sqrt(distances_sq[pid]).cpu())
                true_dist = float(true_dists[rank]) if rank < len(true_dists) else float("inf")
                ratio = ret_dist / true_dist if true_dist > 1e-12 else 1.0
                ratio_sum[k][rank] += ratio
                ratio_max[k][rank] = max(ratio_max[k][rank], ratio)

            rows.append({
                "query_idx": q_idx,
                "k": k,
                "k_search": k_search,
                "hnsw_recall": len(true_topk_set.intersection(hnsw_result)) / k,
                "cluster_recall": len(true_topk_set.intersection(cluster_result)) / k,
                "n_candidates": len(candidates),
            })

    hnsw_recall   = {k: hnsw_hits[k]   / n_queries for k in K_VALUES}
    cluster_recall = {k: cluster_hits[k] / n_queries for k in K_VALUES}
    ratio_mean_per_rank = {k: (ratio_sum[k] / n_queries).tolist() for k in K_VALUES}
    ratio_max_per_rank  = {k: ratio_max[k].tolist()                for k in K_VALUES}

    return hnsw_recall, cluster_recall, ratio_mean_per_rank, ratio_max_per_rank, rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_recall_plot(
    hnsw_recall: dict,
    cluster_recall: dict,
    out_dir: pathlib.Path,
    m: int,
) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K_VALUES, [hnsw_recall[k]   for k in K_VALUES], marker="o", label=f"HNSW (M={m})")
    ax.plot(K_VALUES, [cluster_recall[k] for k in K_VALUES], marker="s", label="Clustering-aided")
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(K_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("SIFT10K: Recall@k")
    fig.tight_layout()
    path = out_dir / "sift_recall_vs_k.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_ratio_plot(
    ratio_mean: dict,
    ratio_max: dict,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    axes = axes.flatten()
    for ax, k in zip(axes, K_VALUES):
        ranks = list(range(1, k + 1))
        ax.plot(ranks, ratio_mean[k], label="mean ratio")
        ax.plot(ranks, ratio_max[k],  label="max ratio",  linestyle="--")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("rank")
        ax.set_ylabel("dist ratio")
        ax.set_title(f"k={k}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("SIFT10K: returned dist / true dist per rank")
    fig.tight_layout()
    path = out_dir / "sift_distance_ratio_per_rank.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_csv(rows: list[dict], out_dir: pathlib.Path) -> pathlib.Path:
    path = out_dir / "sift_results.csv"
    fieldnames = ["query_idx", "k", "k_search", "hnsw_recall", "cluster_recall", "n_candidates"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_summary(
    hnsw_recall: dict,
    cluster_recall: dict,
    ratio_mean: dict,
    ratio_max: dict,
) -> None:
    print()
    print(f"{'k':>4} | {'HNSW recall':>11} | {'Cluster recall':>14} | {'mean ratio':>10} | {'max ratio':>9}")
    print("-" * 60)
    for k in K_VALUES:
        mean_r = float(np.mean(ratio_mean[k]))
        max_r  = float(np.max(ratio_max[k]))
        print(f"{k:>4} | {hnsw_recall[k]:>11.4f} | {cluster_recall[k]:>14.4f} | {mean_r:>10.4f} | {max_r:>9.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for build_cover and the distance kernel.")
    device = torch.device("cuda")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- load SIFT ---
    print("Loading SIFT10K...")
    base_path  = SIFT_DIR / "siftsmall_base.fvecs"
    query_path = SIFT_DIR / "siftsmall_query.fvecs"
    for p in (base_path, query_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected SIFT file not found: {p}")

    base_np  = read_fvecs(base_path)   # (10000, 128) float32
    query_np = read_fvecs(query_path)  # (100,   128) float32
    print(f"  base:  {base_np.shape}, query: {query_np.shape}")

    # move to GPU
    dataset = torch.from_numpy(base_np).to(device)
    queries = torch.from_numpy(query_np).to(device)

    if not args.no_normalize:
        print("  L2-normalizing dataset and queries...")
        dataset = torch.nn.functional.normalize(dataset, p=2, dim=1)
        queries = torch.nn.functional.normalize(queries, p=2, dim=1)

    dataset_cpu = dataset.cpu().numpy().astype(np.float32, copy=False)
    queries_cpu = queries.cpu().numpy().astype(np.float32, copy=False)

    # --- build cover ---
    print("Building clustering cover...")
    t0 = time.perf_counter()
    cover = build_cover(dataset, epsilon=args.epsilon, k=args.k_cluster)
    torch.cuda.synchronize()
    cover_time = time.perf_counter() - t0
    cover_index = CoverIndex(cover)
    print(f"  done in {cover_time:.2f}s  |  {cover_index}")

    # --- build HNSW ---
    print("Building HNSW index...")
    t0 = time.perf_counter()
    hnsw_index = build_hnsw_index(dataset_cpu, args.M)
    hnsw_time = time.perf_counter() - t0
    print(f"  done in {hnsw_time:.2f}s")

    # --- evaluate ---
    print("Evaluating...")
    t0 = time.perf_counter()
    hnsw_recall, cluster_recall, ratio_mean, ratio_max, rows = evaluate(
        dataset=dataset,
        queries=queries,
        queries_cpu=queries_cpu,
        cover_index=cover_index,
        hnsw_index=hnsw_index,
        epsilon=args.epsilon,
        m=args.M,
        k_search_multiplier=args.k_search_multiplier,
    )
    eval_time = time.perf_counter() - t0
    print(f"  done in {eval_time:.2f}s")

    print_summary(hnsw_recall, cluster_recall, ratio_mean, ratio_max)

    # --- save outputs ---
    out_dir = BASE_DIR / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = save_recall_plot(hnsw_recall, cluster_recall, out_dir, args.M)
    p2 = save_ratio_plot(ratio_mean, ratio_max, out_dir)
    p3 = save_csv(rows, out_dir)

    print(f"Saved recall plot    -> {p1}")
    print(f"Saved ratio plot     -> {p2}")
    print(f"Saved CSV            -> {p3}")
    print()
    print(f"Timing: cover={cover_time:.2f}s  hnsw={hnsw_time:.2f}s  eval={eval_time:.2f}s")


if __name__ == "__main__":
    main()
