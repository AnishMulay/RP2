#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cluster_search import CoverIndex
from cluster_search import build_cover
from clustered_push_relabel.utils.distance import TiledEuclideanKernel


N_POINTS = 10_000
DIM = 128
N_CLUSTERS = 50
NOISE_STD = 0.5
EPSILON = 0.01
K_CLUSTER = 4
PAIR_SAMPLE_COUNT = 50_000
PROXY_SAMPLE_COUNT = 10_000

QUANTILES: list[tuple[float, str]] = [
    (0.01, "1st"),
    (0.05, "5th"),
    (0.10, "10th"),
    (0.25, "25th"),
    (0.50, "50th"),
    (0.75, "75th"),
    (0.90, "90th"),
    (0.95, "95th"),
    (0.99, "99th"),
]
RATIO_QUANTILES: list[tuple[float, str]] = QUANTILES + [(1.00, "100th")]


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic script requires CUDA.")
    return torch.device("cuda")


def sample_points_from_centroids(centroids: torch.Tensor, n_points: int) -> torch.Tensor:
    n_clusters, dim = centroids.shape
    assignments = torch.randint(0, n_clusters, (n_points,), device=centroids.device)
    points = centroids[assignments] + NOISE_STD * torch.randn(n_points, dim, device=centroids.device)
    return torch.nn.functional.normalize(points, p=2, dim=1)


def generate_dataset_and_query(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    centroids = torch.randn(N_CLUSTERS, DIM, device=device)
    dataset = sample_points_from_centroids(centroids, N_POINTS)
    query = sample_points_from_centroids(centroids, 1)
    return dataset, query


def sample_distinct_pairs(n_points: int, n_pairs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    first = np.random.randint(0, n_points, size=n_pairs).astype(np.int64, copy=False)
    second = np.random.randint(0, n_points, size=n_pairs).astype(np.int64, copy=False)
    equal_mask = first == second
    while equal_mask.any():
        second[equal_mask] = np.random.randint(0, n_points, size=int(equal_mask.sum()))
        equal_mask = first == second
    return (
        torch.from_numpy(first).to(device=device, dtype=torch.long),
        torch.from_numpy(second).to(device=device, dtype=torch.long),
    )


def summarize_distribution(
    values: torch.Tensor,
    quantiles: list[tuple[float, str]],
) -> dict[str, object]:
    values_cpu = values.detach().to(dtype=torch.float64, device="cpu")
    probs = torch.tensor([prob for prob, _ in quantiles], dtype=torch.float64)
    quantile_values = torch.quantile(values_cpu, probs).tolist()
    return {
        "mean": values_cpu.mean().item(),
        "std": values_cpu.std(unbiased=False).item(),
        "min": values_cpu.min().item(),
        "max": values_cpu.max().item(),
        "quantiles": list(zip((label for _, label in quantiles), quantile_values)),
    }


def print_distribution_summary(
    values: torch.Tensor,
    quantiles: list[tuple[float, str]],
) -> None:
    summary = summarize_distribution(values, quantiles)
    print(f"  mean: {summary['mean']:.6f}")
    print(f"  std: {summary['std']:.6f}")
    print(f"  min: {summary['min']:.6f}")
    print(f"  max: {summary['max']:.6f}")
    print("  quantiles:")
    for label, value in summary["quantiles"]:
        print(f"    {label}: {value:.6f}")


def compute_proxy_distance(
    cover_index: CoverIndex,
    point_x: int,
    point_y: int,
    epsilon: float,
    shell_cache: dict[int, set[tuple[int, int]]],
) -> float | None:
    shells_x = shell_cache.setdefault(point_x, set(cover_index.get_shells(point_x)))
    shells_y = shell_cache.setdefault(point_y, set(cover_index.get_shells(point_y)))
    shared_shells = shells_x.intersection(shells_y)
    if not shared_shells:
        return None
    smallest_level = min(level_id for _, level_id in shared_shells)
    return 2.0 * float(smallest_level) * epsilon


def run_diagnostic_1(dataset: torch.Tensor) -> None:
    print("Diagnostic 1 - Pairwise distance distribution of the dataset")
    pair_i, pair_j = sample_distinct_pairs(N_POINTS, PAIR_SAMPLE_COUNT, dataset.device)
    pairwise_distances = torch.norm(
        dataset.index_select(0, pair_i) - dataset.index_select(0, pair_j),
        dim=1,
    )
    print(f"  sampled pairs: {PAIR_SAMPLE_COUNT}")
    print_distribution_summary(pairwise_distances, QUANTILES)
    print()


def run_diagnostic_2(
    dataset: torch.Tensor,
    query: torch.Tensor,
    kernel: TiledEuclideanKernel,
    workspace: dict[str, torch.Tensor],
) -> None:
    print("Diagnostic 2 - Query-to-dataset distance distribution")
    distances_sq = kernel.compute_dist_tile(query, workspace).squeeze(1)
    distances = torch.sqrt(distances_sq)
    sorted_distances, _ = torch.sort(distances)

    print(f"  query count: {query.shape[0]}")
    print_distribution_summary(distances, QUANTILES)
    print(f"  true nearest neighbor distance: {sorted_distances[0].item():.6f}")
    print(f"  10th nearest neighbor distance: {sorted_distances[9].item():.6f}")
    print(f"  100th nearest neighbor distance: {sorted_distances[99].item():.6f}")
    print(f"  1000th nearest neighbor distance: {sorted_distances[999].item():.6f}")
    print()


def run_proxy_diagnostics(dataset: torch.Tensor, cover_index: CoverIndex) -> None:
    pair_i, pair_j = sample_distinct_pairs(N_POINTS, PROXY_SAMPLE_COUNT, dataset.device)
    true_distances = torch.norm(
        dataset.index_select(0, pair_i) - dataset.index_select(0, pair_j),
        dim=1,
    ).detach().cpu()

    shell_cache: dict[int, set[tuple[int, int]]] = {}
    ratios: list[float] = []
    no_shared_shell = 0

    for point_x, point_y, true_distance in zip(pair_i.cpu().tolist(), pair_j.cpu().tolist(), true_distances.tolist()):
        proxy_distance = compute_proxy_distance(cover_index, point_x, point_y, EPSILON, shell_cache)
        if proxy_distance is None:
            no_shared_shell += 1
            continue

        if true_distance <= 0.0:
            ratio = float("inf") if proxy_distance > 0.0 else 1.0
        else:
            ratio = proxy_distance / true_distance
        ratios.append(ratio)

    print("Diagnostic 3 - Shared-shell coverage for proxy distances")
    print(f"  sampled pairs: {PROXY_SAMPLE_COUNT}")
    print(f"  pairs with no shared shell: {no_shared_shell}")
    print(f"  fraction with no shared shell: {no_shared_shell / PROXY_SAMPLE_COUNT:.6f}")
    print()

    print("Diagnostic 4 - Distance proxy approximation quality")
    if not ratios:
        print("  no ratios were available because none of the sampled pairs shared a shell")
        print()
        return

    ratio_tensor = torch.tensor(ratios, dtype=torch.float64)
    print(f"  pairs with a shared shell: {ratio_tensor.numel()}")
    print(f"  minimum ratio observed: {ratio_tensor.min().item():.6f}")
    print_distribution_summary(ratio_tensor, RATIO_QUANTILES)

    below_one_fraction = (ratio_tensor < 1.0).to(torch.float64).mean().item()
    if below_one_fraction > 0.0:
        print("  warning: at least one proxy ratio is below 1.0")
        print(f"  fraction of ratios below 1: {below_one_fraction:.6f}")

    exceeds_three_fraction = (ratio_tensor > 3.0).to(torch.float64).mean().item()
    print(f"  fraction of ratios exceeding 3: {exceeds_three_fraction:.6f}")
    print()


def main() -> None:
    device = require_cuda()

    print(f"Generating dataset and query on {device}...")
    dataset, query = generate_dataset_and_query(device)

    print("Building cover...")
    cover = build_cover(dataset, epsilon=EPSILON, k=K_CLUSTER)
    cover_index = CoverIndex(cover)
    print(f"Cover index: {cover_index}")
    print()

    kernel = TiledEuclideanKernel(chunk_size=4096)
    workspace = kernel.prepare_workspace(dataset)

    run_diagnostic_1(dataset)
    run_diagnostic_2(dataset, query, kernel, workspace)
    run_proxy_diagnostics(dataset, cover_index)


if __name__ == "__main__":
    main()
