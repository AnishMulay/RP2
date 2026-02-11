import torch
import time
import tracemalloc
from k_level_index import KLevelVectorIndex
from custom_vector_search import BruteForceSearcher, KLevelSearcher, FaissSearcher

def generate_mock_embeddings(n_samples=20000, dim=384, n_clusters=50, device='cuda'):
    """Generates synthetic data that mimics semantic clustering of real embeddings."""
    print(f"[*] Generating {n_samples} mock embeddings (dim={dim})...")
    centroids = torch.randn(n_clusters, dim, device=device)
    
    # Assign each point to a random cluster and add noise
    cluster_assignments = torch.randint(0, n_clusters, (n_samples,), device=device)
    noise = torch.randn(n_samples, dim, device=device) * 0.5
    
    data = centroids[cluster_assignments] + noise
    
    # L2 Normalize (Standard for cosine similarity / euclidean embedding spaces)
    data = torch.nn.functional.normalize(data, p=2, dim=1)
    return data

def calculate_recall(ground_truth, predictions):
    """Calculates Recall@K: What percentage of true nearest neighbors were found?"""
    recalls = []
    for gt, pred in zip(ground_truth, predictions):
        gt_set = set(gt.cpu().numpy())
        pred_set = set(pred.cpu().numpy())
        intersection = gt_set.intersection(pred_set)
        recalls.append(len(intersection) / len(gt_set))
    return sum(recalls) / len(recalls)

def run_search_with_memory(label, search_fn):
    tracemalloc.start()
    t0 = time.time()
    indices = search_fn()
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"    - {label} done in {elapsed:.3f}s (peak tracemalloc: {peak / (1024 * 1024):.2f} MiB)")
    return indices, elapsed

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")
    
    # 1. Setup Data
    N = 50000
    DIM = 128 # Smaller dim for faster testing, scale up as needed
    K_NEIGHBORS = 10
    
    dataset = generate_mock_embeddings(n_samples=N, dim=DIM, device=device)
    queries = generate_mock_embeddings(n_samples=500, dim=DIM, device=device)

    # 2. Build K-Level Index
    k_level_idx = KLevelVectorIndex(epsilon=0.8, k=4)
    k_level_idx.build_index(dataset)
    num_centroids = len(k_level_idx.centroids)
    
    # 3. Initialize Searchers
    brute_searcher = BruteForceSearcher(dataset)
    klevel_searcher = KLevelSearcher(k_level_idx, n_probe=2)
    faiss_searcher = FaissSearcher(dataset, n_centroids=num_centroids, n_probe=2)

    # 4. Run Queries & Measure Accuracy
    print("\n[*] Running Queries...")

    gt_indices, _ = run_search_with_memory(
        "Brute Force",
        lambda: brute_searcher.search(queries, top_k=K_NEIGHBORS),
    )
    faiss_indices, faiss_time = run_search_with_memory(
        "FAISS",
        lambda: faiss_searcher.search(queries, top_k=K_NEIGHBORS),
    )
    faiss_recall = calculate_recall(gt_indices, faiss_indices)

    # 5. Sweep radius-level pruning for custom K-Level ANN.
    radius_sweep = [1, 2, 4, 8, None]
    sweep_results = []
    for radius_level in radius_sweep:
        radius_label = "None" if radius_level is None else str(radius_level)
        klevel_indices, elapsed = run_search_with_memory(
            f"K-Level (radius={radius_label})",
            lambda radius_level=radius_level: klevel_searcher.search(
                queries,
                top_k=K_NEIGHBORS,
                search_radius_level=radius_level,
            ),
        )
        recall = calculate_recall(gt_indices, klevel_indices)
        sweep_results.append((radius_label, elapsed, recall * 100.0))

    print("\n" + "="*52)
    print(" EXPERIMENT RESULTS: RECALL@10")
    print("="*52)
    print(f" FAISS (IVFFlat) Recall: {faiss_recall * 100:.2f}%")
    print(f" FAISS Time:             {faiss_time:.3f}s")
    print("="*52)
    print(" K-Level Radius Sweep")
    print("="*52)
    print(f"{'Radius Level':<14} | {'Time (s)':>9} | {'Recall (%)':>10}")
    print("-" * 52)
    for radius_label, elapsed, recall_pct in sweep_results:
        print(f"{radius_label:<14} | {elapsed:>9.3f} | {recall_pct:>10.2f}")
    print("="*52)

if __name__ == "__main__":
    main()
