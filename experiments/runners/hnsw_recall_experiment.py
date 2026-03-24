#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import hnswlib
except ImportError as exc:
    raise ImportError("hnswlib is required for this experiment.") from exc

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cluster_search import CoverIndex
from cluster_search import build_cover
from cluster_search import cluster_search
from clustered_push_relabel.utils.distance import TiledEuclideanKernel


K_SEARCH_VALUES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024]
NOISE_STD = 0.5
EF_CONSTRUCTION = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HNSW recall experiment with clustering-aided reranking.")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--k_cluster", type=int, default=4)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA because the dataset and clustering are specified to run on GPU.")
    return torch.device("cuda")


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def sample_points_from_centroids(centroids: torch.Tensor, n_points: int) -> torch.Tensor:
    n_clusters, dim = centroids.shape
    assignments = torch.randint(0, n_clusters, (n_points,), device=centroids.device)
    points = centroids[assignments] + NOISE_STD * torch.randn(n_points, dim, device=centroids.device)
    return torch.nn.functional.normalize(points, p=2, dim=1)


def generate_dataset_and_queries(
    n: int,
    dim: int,
    n_clusters: int,
    n_queries: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    centroids = torch.randn(n_clusters, dim, device=device)
    dataset = sample_points_from_centroids(centroids, n)
    query_points = sample_points_from_centroids(centroids, n_queries)
    return dataset, query_points


def build_hnsw_index(dataset_cpu: np.ndarray, m: int) -> hnswlib.Index:
    n, dim = dataset_cpu.shape
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=m)
    index.add_items(dataset_cpu, np.arange(n, dtype=np.int32))
    return index


def query_hnsw(index: hnswlib.Index, query_vector: np.ndarray, k_search: int) -> list[int]:
    index.set_ef(max(k_search, 50))
    labels, _ = index.knn_query(query_vector, k=k_search)
    return [int(label) for label in labels[0][:k_search]]


def topk_from_distances(
    distances_sq: torch.Tensor,
    k: int,
    candidate_ids: list[int] | None = None,
) -> list[int]:
    if candidate_ids is None:
        k_eff = min(k, distances_sq.numel())
        if k_eff <= 0:
            return []
        return torch.topk(distances_sq, k=k_eff, largest=False).indices.tolist()

    if not candidate_ids:
        return []

    candidate_tensor = torch.tensor(candidate_ids, device=distances_sq.device, dtype=torch.long)
    candidate_distances = distances_sq.index_select(0, candidate_tensor)
    k_eff = min(k, candidate_tensor.numel())
    topk_local = torch.topk(candidate_distances, k=k_eff, largest=False).indices
    return candidate_tensor[topk_local].tolist()


def evaluate(
    dataset: torch.Tensor,
    query_points: torch.Tensor,
    query_points_cpu: np.ndarray,
    cover_index: CoverIndex,
    index: hnswlib.Index,
    epsilon: float,
    m: int,
) -> tuple[dict[int, float], dict[int, float], float]:
    kernel = TiledEuclideanKernel(chunk_size=4096)
    workspace = kernel.prepare_workspace(dataset)

    hnsw_hits = {k: 0.0 for k in K_SEARCH_VALUES}
    clustering_hits = {k: 0.0 for k in K_SEARCH_VALUES}
    total_candidate_count = 0.0

    for query_idx in range(query_points.shape[0]):
        query = query_points[query_idx : query_idx + 1]
        distances_sq = kernel.compute_dist_tile(query, workspace).squeeze(1)
        brute_force_top = {
            k_search: set(topk_from_distances(distances_sq, k_search))
            for k_search in K_SEARCH_VALUES
        }

        query_vector = query_points_cpu[query_idx : query_idx + 1]
        seeds = query_hnsw(index, query_vector, k_search=m)

        for k_search in K_SEARCH_VALUES:
            print(f"query {query_idx + 1}/{query_points.shape[0]}, k_search={k_search}")
            hnsw_result = query_hnsw(index, query_vector, k_search)

            candidates = cluster_search(
                seeds,
                cover_index,
                dataset,
                kernel,
                k_search,
                epsilon,
            )
            total_candidate_count += len(candidates)

            if candidates:
                candidate_tensor = torch.tensor(candidates, device=distances_sq.device, dtype=torch.long)
                candidate_distances = distances_sq.index_select(0, candidate_tensor)
                sorted_local = torch.argsort(candidate_distances)
                clustering_result = candidate_tensor[sorted_local].tolist()
            else:
                clustering_result = []

            hnsw_hits[k_search] += len(brute_force_top[k_search].intersection(hnsw_result)) / k_search
            clustering_hits[k_search] += len(brute_force_top[k_search].intersection(clustering_result)) / k_search

    n_queries = float(query_points.shape[0])
    hnsw_recall = {k: hnsw_hits[k] / n_queries for k in K_SEARCH_VALUES}
    clustering_recall = {k: clustering_hits[k] / n_queries for k in K_SEARCH_VALUES}
    n_evals = n_queries * len(K_SEARCH_VALUES)
    avg_candidate_count = total_candidate_count / n_evals if n_evals else 0.0
    return hnsw_recall, clustering_recall, avg_candidate_count


def save_plot(hnsw_recall: dict[int, float], clustering_recall: dict[int, float], m: int) -> pathlib.Path:
    output_dir = BASE_DIR / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "recall_vs_k.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_SEARCH_VALUES, [hnsw_recall[k] for k in K_SEARCH_VALUES], marker="o", label=f"HNSW (M={m})")
    ax.plot(K_SEARCH_VALUES, [clustering_recall[k] for k in K_SEARCH_VALUES], marker="s", label="Clustering-aided")
    ax.axvline(m, color="gray", linestyle="--", linewidth=1.5)
    ax.text(m + 0.5, 0.05, f"M={m}", color="gray", rotation=90, va="bottom")
    ax.set_xlabel("k_search")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(K_SEARCH_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def print_results_table(
    hnsw_recall: dict[int, float],
    clustering_recall: dict[int, float],
    avg_candidate_count: float,
) -> None:
    print()
    print(f"{'k_search':>8} | {'HNSW recall':>11} | {'Clustering-aided recall':>24}")
    print("-" * 50)
    for k_search in K_SEARCH_VALUES:
        print(f"{k_search:>8d} | {hnsw_recall[k_search]:>11.4f} | {clustering_recall[k_search]:>24.4f}")
    print()
    print(f"Average candidate set size: {avg_candidate_count:.1f}")


def main() -> None:
    args = parse_args()
    device = require_cuda()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"Generating dataset on {device}...")
    dataset, query_points = generate_dataset_and_queries(
        args.n,
        args.dim,
        args.n_clusters,
        args.n_queries,
        device,
    )
    sync_if_needed(device)

    print("Building k-level cover...")
    cover = build_cover(dataset, epsilon=args.epsilon, k=args.k_cluster)
    cover_index = CoverIndex(cover)

    print("Preparing HNSW index...")
    dataset_cpu = dataset.detach().cpu().numpy().astype(np.float32, copy=False)
    query_points_cpu = query_points.detach().cpu().numpy().astype(np.float32, copy=False)
    index = build_hnsw_index(dataset_cpu, args.M)

    print("Evaluating recall...")
    hnsw_recall, clustering_recall, avg_candidate_count = evaluate(
        dataset=dataset,
        query_points=query_points,
        query_points_cpu=query_points_cpu,
        cover_index=cover_index,
        index=index,
        epsilon=args.epsilon,
        m=args.M,
    )

    print_results_table(hnsw_recall, clustering_recall, avg_candidate_count)
    output_path = save_plot(hnsw_recall, clustering_recall, args.M)
    print()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
