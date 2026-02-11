import torch
import time
from k_level_index import KLevelVectorIndex
from custom_vector_search import BruteForceSearcher, KLevelSearcher, FaissSearcher

def generate_mock_embeddings(n_samples=20000, dim=128, n_clusters=50, device='cuda'):
    """Generates synthetic data with cluster structure."""
    print(f"[*] Generating {n_samples} mock embeddings (dim={dim})...")
    centroids = torch.randn(n_clusters, dim, device=device)
    cluster_assignments = torch.randint(0, n_clusters, (n_samples,), device=device)
    noise = torch.randn(n_samples, dim, device=device) * 0.5
    data = centroids[cluster_assignments] + noise
    return torch.nn.functional.normalize(data, p=2, dim=1)

def calculate_recall(ground_truth, prediction):
    """Recall@K for a single query."""
    gt_set = set(ground_truth.cpu().tolist())
    pred_set = set(prediction.cpu().tolist())
    intersection = gt_set.intersection(pred_set)
    return len(intersection) / len(gt_set) if len(gt_set) > 0 else 0.0

def main():
    if not torch.cuda.is_available():
        print("[!] CUDA not available. This experiment requires a GPU.")
        return
    
    device = torch.device('cuda')
    print(f"[*] Using device: {device}")
    
    # 1. Setup Data
    N = 50000
    DIM = 128
    K_NEIGHBORS = 10
    NUM_QUERIES = 1000
    
    dataset = generate_mock_embeddings(n_samples=N, dim=DIM, device=device)
    queries = generate_mock_embeddings(n_samples=NUM_QUERIES, dim=DIM, device=device)

    # 2. Build K-Level Index
    # Note: Epsilon controls the number of centroids. Lower epsilon = More centroids.
    # Try 0.5 - 0.9 range.
    k_level_idx = KLevelVectorIndex(epsilon=0.6, k=4)
    k_level_idx.build_index(dataset)
    num_centroids = k_level_idx.centroids.shape[0]
    
    # 3. Initialize Searchers
    brute_searcher = BruteForceSearcher(dataset)
    klevel_searcher = KLevelSearcher(k_level_idx)
    # Fair comparison: FAISS gets same number of centroids and nprobe=1
    faiss_searcher = FaissSearcher(dataset, n_centroids=num_centroids, n_probe=1)

    print(f"\n[*] Starting Benchmark Loop ({NUM_QUERIES} queries, 1-by-1)...")
    
    klevel_times = []
    faiss_times = []
    klevel_recalls = []
    faiss_recalls = []

    # 4. The Loop
    for i in range(NUM_QUERIES):
        q = queries[i]
        
        # A. Ground Truth
        gt = brute_searcher.search(q, top_k=K_NEIGHBORS)
        
        # B. Custom K-Level Search
        torch.cuda.synchronize()
        t0 = time.time()
        res_klevel = klevel_searcher.search_one(q, top_k=K_NEIGHBORS)
        torch.cuda.synchronize()
        klevel_times.append(time.time() - t0)
        
        # C. FAISS Search
        torch.cuda.synchronize()
        t0 = time.time()
        res_faiss = faiss_searcher.search_one(q, top_k=K_NEIGHBORS)
        torch.cuda.synchronize()
        faiss_times.append(time.time() - t0)
        
        # D. Metrics
        klevel_recalls.append(calculate_recall(gt, res_klevel))
        faiss_recalls.append(calculate_recall(gt, res_faiss))
        
        if (i+1) % 100 == 0:
            print(f"    Processed {i+1}/{NUM_QUERIES} queries...")

    # 5. Results
    avg_time_k = sum(klevel_times) / len(klevel_times) * 1000 # ms
    avg_time_f = sum(faiss_times) / len(faiss_times) * 1000 # ms
    avg_rec_k = sum(klevel_recalls) / len(klevel_recalls) * 100
    avg_rec_f = sum(faiss_recalls) / len(faiss_recalls) * 100

    print("\n" + "="*50)
    print(" SINGLE-QUERY BENCHMARK RESULTS")
    print("="*50)
    print(f" {'Metric':<15} | {'FAISS (IVFFlat)':<15} | {'Custom K-Level':<15}")
    print("-" * 50)
    print(f" {'Latency (ms)':<15} | {avg_time_f:<15.4f} | {avg_time_k:<15.4f}")
    print(f" {'Recall@10 (%)':<15} | {avg_rec_f:<15.2f} | {avg_rec_k:<15.2f}")
    print("="*50)

if __name__ == "__main__":
    main()