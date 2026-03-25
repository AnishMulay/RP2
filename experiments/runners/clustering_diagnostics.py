#!/usr/bin/env python3

from __future__ import annotations

import math
import pathlib
import sys
import heapq

import hnswlib
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
from cluster_search.searcher import cluster_search
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
NEAREST_NEIGHBOR_RATIO_QUANTILES: list[tuple[float, str]] = [
    (0.01, "p1"),
    (0.05, "p5"),
    (0.10, "p10"),
    (0.25, "p25"),
    (0.50, "p50"),
    (0.75, "p75"),
    (0.90, "p90"),
    (0.95, "p95"),
    (0.99, "p99"),
    (1.00, "p100"),
]


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
    shell_cache: dict[int, dict[int, int]],
) -> float | None:
    center_to_level_x = shell_cache.setdefault(point_x, dict(cover_index.get_shells(point_x)))
    center_to_level_y = shell_cache.setdefault(point_y, dict(cover_index.get_shells(point_y)))
    shared_centers = center_to_level_x.keys() & center_to_level_y.keys()
    if not shared_centers:
        return None

    best_center = min(
        shared_centers,
        key=lambda center_id: max(center_to_level_x[center_id], center_to_level_y[center_id]),
    )
    return float(center_to_level_x[best_center] + center_to_level_y[best_center]) * epsilon


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

    shell_cache: dict[int, dict[int, int]] = {}
    ratios: list[float] = []
    no_shared_center = 0

    for point_x, point_y, true_distance in zip(pair_i.cpu().tolist(), pair_j.cpu().tolist(), true_distances.tolist()):
        proxy_distance = compute_proxy_distance(cover_index, point_x, point_y, EPSILON, shell_cache)
        if proxy_distance is None:
            no_shared_center += 1
            continue

        if true_distance <= 0.0:
            ratio = float("inf") if proxy_distance > 0.0 else 1.0
        else:
            ratio = proxy_distance / true_distance
        ratios.append(ratio)

    print("Diagnostic 3 - Shared-center coverage for proxy distances")
    print(f"  sampled pairs: {PROXY_SAMPLE_COUNT}")
    print(f"  pairs with no shared center: {no_shared_center}")
    print(f"  fraction with no shared center: {no_shared_center / PROXY_SAMPLE_COUNT:.6f}")
    print()

    print("Diagnostic 4 - Distance proxy approximation quality")
    if not ratios:
        print("  no ratios were available because none of the sampled pairs shared a center")
        print()
        return

    ratio_tensor = torch.tensor(ratios, dtype=torch.float64)
    print(f"  pairs with a shared center: {ratio_tensor.numel()}")
    print(f"  minimum ratio observed: {ratio_tensor.min().item():.6f}")
    print_distribution_summary(ratio_tensor, NEAREST_NEIGHBOR_RATIO_QUANTILES)

    below_one_fraction = (ratio_tensor < 1.0).to(torch.float64).mean().item()
    if below_one_fraction > 0.0:
        print("  warning: at least one proxy ratio is below 1.0")
        print(f"  fraction of ratios below 1: {below_one_fraction:.6f}")

    exceeds_three_fraction = (ratio_tensor > 3.0).to(torch.float64).mean().item()
    print(f"  fraction of ratios exceeding 3: {exceeds_three_fraction:.6f}")
    print()


def run_diagnostic_5(dataset: torch.Tensor, cover_index: CoverIndex) -> None:
    sqrt_n = int(math.sqrt(N_POINTS))
    shell_cache: dict[int, dict[int, int]] = {}
    ratios: list[float] = []
    no_shared_center = 0
    total_pairs = 0

    print("Diagnostic 5 - Distance proxy approximation for sqrt(N) nearest neighbors")
    print(f"  sqrt(N): {sqrt_n}")

    for point_x in range(N_POINTS):
        true_distances = torch.norm(dataset - dataset[point_x], dim=1)
        sorted_indices = torch.argsort(true_distances)
        neighbor_indices = sorted_indices[sorted_indices != point_x][:sqrt_n]

        for point_y in neighbor_indices.tolist():
            total_pairs += 1
            true_distance = true_distances[point_y].item()
            proxy_distance = compute_proxy_distance(cover_index, point_x, point_y, EPSILON, shell_cache)
            if proxy_distance is None:
                no_shared_center += 1
                continue

            if true_distance <= 0.0:
                ratio = float("inf") if proxy_distance > 0.0 else 1.0
            else:
                ratio = proxy_distance / true_distance
            ratios.append(ratio)

    print(f"  processed pairs: {total_pairs}")
    print(f"  pairs with no shared center: {no_shared_center}")
    print(f"  fraction with no shared center: {no_shared_center / total_pairs:.6f}")

    if not ratios:
        print("  no ratios were available because none of the nearest-neighbor pairs shared a center")
        print()
        return

    ratio_tensor = torch.tensor(ratios, dtype=torch.float64)
    print(f"  pairs with a shared center: {ratio_tensor.numel()}")
    print(f"  minimum ratio observed: {ratio_tensor.min().item():.6f}")
    print_distribution_summary(ratio_tensor, RATIO_QUANTILES)

    below_one_fraction = (ratio_tensor < 1.0).to(torch.float64).mean().item()
    if below_one_fraction > 0.0:
        print("  warning: at least one proxy ratio is below 1.0")
        print(f"  fraction of ratios below 1: {below_one_fraction:.6f}")

    exceeds_one_plus_epsilon_fraction = (ratio_tensor > (1.0 + EPSILON)).to(torch.float64).mean().item()
    print(f"  fraction of ratios exceeding 1 + epsilon (1.01): {exceeds_one_plus_epsilon_fraction:.6f}")
    print()


def run_diagnostic_6(
    dataset: torch.Tensor,
    query: torch.Tensor,
    cover_index: CoverIndex,
    kernel: TiledEuclideanKernel,
) -> None:
    print("Diagnostic 6 - Heap vs proxy brute force sanity check")

    dataset_cpu = dataset.detach().to(device="cpu", dtype=torch.float32).contiguous()
    query_cpu = query.detach().to(device="cpu", dtype=torch.float32).contiguous()

    index = hnswlib.Index(space="l2", dim=dataset_cpu.shape[1])
    index.init_index(max_elements=dataset_cpu.shape[0], ef_construction=200, M=16)
    index.add_items(dataset_cpu.numpy())
    index.set_ef(1)

    labels, _distances = index.knn_query(query_cpu.numpy(), k=1)
    point_v = int(labels[0, 0])
    print(f"  hnsw nearest neighbor seed v: {point_v}")

    shell_cache: dict[int, dict[int, int]] = {}
    proxy_distances: list[tuple[float, int]] = []
    for point_p in range(dataset.shape[0]):
        proxy_distance = compute_proxy_distance(cover_index, point_v, point_p, EPSILON, shell_cache)
        if proxy_distance is None:
            proxy_distances.append((float("inf"), point_p))
        else:
            proxy_distances.append((proxy_distance, point_p))
    proxy_distances.sort()
    proxy_distance_by_point = {point_id: proxy_distance for proxy_distance, point_id in proxy_distances}

    for k in [10, 50, 100, 200, 500]:
        method_1_results = {point_id for _distance, point_id in proxy_distances[:k]}
        method_2_results = set(
            cluster_search(
                seeds=[point_v],
                cover_index=cover_index,
                dataset=dataset,
                kernel=kernel,
                k_prime=k,
                epsilon=EPSILON,
            )
        )
        intersection_size = len(method_1_results & method_2_results)
        recovered_fraction = intersection_size / k

        print(
            "  "
            f"k={k}, "
            f"method_1_size={len(method_1_results)}, "
            f"method_2_size={len(method_2_results)}, "
            f"intersection_size={intersection_size}, "
            f"recovered_fraction={recovered_fraction:.6f}"
        )
        if k == 50:
            method_1_only = sorted(
                (
                    (proxy_distance_by_point[point_id], point_id)
                    for point_id in method_1_results
                    if point_id not in method_2_results
                ),
                key=lambda item: (item[0], item[1]),
            )
            if not method_1_only:
                print("  k=50 targeted trace: no missed point in method_1 \\ method_2")
                continue

            missed_point = method_1_only[0][1]
            print(f"  k=50 targeted trace: missed_point={missed_point}")

            v_shells = list(cover_index.get_shells(point_v))
            missed_point_shells = list(cover_index.get_shells(missed_point))
            v_center_to_level = dict(v_shells)
            missed_center_to_level = dict(missed_point_shells)

            print("  v shells:")
            for center_id, level_id in v_shells:
                print(
                    "    "
                    f"(center_id={center_id}, level_id={level_id}), "
                    f"v_proxy_contribution={level_id * EPSILON:.6f}"
                )

            print("  missed_point shells:")
            for center_id, level_id in missed_point_shells:
                print(f"    (center_id={center_id}, level_id={level_id})")

            shared_centers = sorted(v_center_to_level.keys() & missed_center_to_level.keys())
            if not shared_centers:
                print("  shared centers: none")
                continue

            print("  shared centers:")
            for center_id in shared_centers:
                v_level = v_center_to_level[center_id]
                p_level = missed_center_to_level[center_id]
                proxy_distance = (v_level + p_level) * EPSILON
                max_level = max(v_level, p_level)
                print(
                    "    "
                    f"center_id={center_id}, "
                    f"v_level={v_level}, "
                    f"missed_point_level={p_level}, "
                    f"proxy_distance={proxy_distance:.6f}, "
                    f"max_level={max_level}"
                )

            best_center = min(
                shared_centers,
                key=lambda center_id: (max(v_center_to_level[center_id], missed_center_to_level[center_id]), center_id),
            )
            best_v_level = v_center_to_level[best_center]
            best_p_level = missed_center_to_level[best_center]
            best_proxy_distance = (best_v_level + best_p_level) * EPSILON
            best_max_level = max(best_v_level, best_p_level)
            print(
                "  optimal path: "
                f"center_id={best_center}, "
                f"v_level={best_v_level}, "
                f"missed_point_level={best_p_level}, "
                f"proxy_distance={best_proxy_distance:.6f}, "
                f"max_level={best_max_level}"
            )

            target_shell = (best_center, best_p_level)
            print(
                "  heap trace toward optimal shell: "
                f"target_shell=(center_id={target_shell[0]}, level_id={target_shell[1]})"
            )

            heap: list[tuple[float, int, int, float]] = []
            results: set[int] = set()
            for center_id, v_level in v_shells:
                delta_c = EPSILON * v_level
                heapq.heappush(heap, (delta_c, center_id, 0, delta_c))

            target_shell_popped = False
            while len(results) < 50 and heap:
                proxy_distance, center_id, level_id, delta_c = heapq.heappop(heap)

                if level_id == 0:
                    if center_id not in results:
                        results.add(center_id)
                else:
                    for point_id in cover_index.get_shell_members(center_id, level_id):
                        results.add(point_id)

                print(
                    "    "
                    f"popped_shell=(center_id={center_id}, level_id={level_id}), "
                    f"proxy_distance={proxy_distance:.6f}, "
                    f"results_collected={len(results)}"
                )

                if (center_id, level_id) == target_shell:
                    target_shell_popped = True
                    break

                next_level = level_id + 1
                if next_level <= cover_index.get_max_level(center_id):
                    next_proxy = delta_c + EPSILON * next_level
                    heapq.heappush(heap, (next_proxy, center_id, next_level, delta_c))

            print(
                "  optimal shell visited before collecting 50 results: "
                f"{'yes' if target_shell_popped else 'no'}"
            )

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
    # workspace = kernel.prepare_workspace(dataset)

    # run_diagnostic_1(dataset)
    # run_diagnostic_2(dataset, query, kernel, workspace)
    # run_proxy_diagnostics(dataset, cover_index)
    # run_diagnostic_5(dataset, cover_index)
    run_diagnostic_6(dataset, query, cover_index, kernel)


if __name__ == "__main__":
    main()
