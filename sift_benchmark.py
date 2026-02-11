import os
import torch
import numpy as np
import time
from k_level_index import KLevelVectorIndex
from custom_vector_search import BruteForceSearcher, KLevelSearcher, FaissSearcher

def read_fvecs(filename):
    print(f"    Loading {filename}...")
    with open(filename, 'rb') as f:
        # Read dimensions
        x = np.fromfile(f, dtype='int32', count=1)
        if x.size == 0: return None
        d = x[0]
        
        # Determine number of vectors
        f.seek(0, 2)
        filesize = f.tell()
        f.seek(0)
        
        row_size = 4 + d * 4
        count = filesize // row_size
        
        # Read data
        data_raw = np.fromfile(f, dtype='int32', count=count * (d + 1))
        data_raw = data_raw.reshape(count, d + 1)
        vectors = data_raw[:, 1:].copy().view('float32')
        return torch.from_numpy(vectors)

def read_ivecs(filename):
    print(f"    Loading {filename}...")
    with open(filename, 'rb') as f:
        x = np.fromfile(f, dtype='int32', count=1)
        if x.size == 0: return None
        d = x[0]
        
        f.seek(0, 2)
        filesize = f.tell()
        f.seek(0)
        row_size = 4 + d * 4
        count = filesize // row_size
        
        data_raw = np.fromfile(f, dtype='int32', count=count * (d + 1))
        data_raw = data_raw.reshape(count, d + 1)
        return torch.from_numpy(data_raw[:, 1:].copy())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    # Check for SIFT10K (Small) or SIFT1M (Big)
    if os.path.exists("./siftsmall"):
        print("[*] Detected SIFT10K (Small Dataset)")
        data_dir = "./siftsmall"
        base_file = "siftsmall_base.fvecs"
        query_file = "siftsmall_query.fvecs"
        gt_file = "siftsmall_groundtruth.ivecs"
        MAX_N = 10000
    elif os.path.exists("./sift"):
        print("[*] Detected SIFT1M (Big Dataset)")
        data_dir = "./sift"
        base_file = "sift_base.fvecs"
        query_file = "sift_query.fvecs"
        gt_file = "sift_groundtruth.ivecs"
        MAX_N = 50000 # Limit for speed
    else:
        print("[!] No data found!")
        print("    Please upload the 'siftsmall' folder to this directory.")
        return

    # Load Data
    dataset = read_fvecs(os.path.join(data_dir, base_file))
    queries = read_fvecs(os.path.join(data_dir, query_file))
    gt_indices = read_ivecs(os.path.join(data_dir, gt_file))
    
    # Slice to MAX_N if needed
    if dataset.shape[0] > MAX_N:
        dataset = dataset[:MAX_N]
        # If we slice the dataset, the pre-computed Ground Truth is invalid!
        # We must regenerate it.
        regenerate_gt = True
    else:
        # For SIFT10K, we use the whole thing, so GT is valid.
        regenerate_gt = False

    dataset = dataset.to(device)
    queries = queries.to(device)
    gt_indices = gt_indices.long().to(device)

    print(f"[*] Dataset: {dataset.shape}")
    print(f"[*] Queries: {queries.shape}")

    if regenerate_gt:
        print("[*] Regenerating Ground Truth (since we sliced the dataset)...")
        brute = BruteForceSearcher(dataset)
    else:
        print("[*] Using provided Ground Truth...")

    # 2. Build Custom Index
    print("\n[*] Building Custom Index (All-Points)...")
    t0 = time.time()
    # Batch size controls GPU memory usage during construction
    idx = KLevelVectorIndex(batch_size=1024) 
    idx.build_index(dataset)
    print(f"    Build Time: {time.time()-t0:.2f}s")
    
    custom_searcher = KLevelSearcher(idx)
    
    # 3. Build FAISS
    print("\n[*] Building FAISS...")
    n_centroids = int(np.sqrt(dataset.shape[0]))
    faiss_searcher = FaissSearcher(dataset, n_centroids=n_centroids, n_probe=5)

    # 4. Benchmark
    print("\n[*] Running Benchmark...")
    c_recalls, f_recalls = [], []
    c_times, f_times = [], []
    
    # Run only 100 queries for speed
    num_queries_to_run = min(100, queries.shape[0])
    
    for i in range(num_queries_to_run):
        q = queries[i]
        
        # Get Truth
        if regenerate_gt:
            gt = brute.search(q, top_k=10)
        else:
            gt = gt_indices[i, :10]

        # Custom Search
        torch.cuda.synchronize()
        t0 = time.time()
        res_c = custom_searcher.search_one(q, top_k=10)
        torch.cuda.synchronize()
        c_times.append(time.time() - t0)
        
        # FAISS Search
        torch.cuda.synchronize()
        t0 = time.time()
        res_f = faiss_searcher.search_one(q, top_k=10)
        torch.cuda.synchronize()
        f_times.append(time.time() - t0)
        
        # Recall
        gt_set = set(gt.cpu().tolist())
        c_set = set(res_c.cpu().tolist())
        f_set = set(res_f.cpu().tolist())
        
        c_recalls.append(len(c_set.intersection(gt_set)) / 10.0)
        f_recalls.append(len(f_set.intersection(gt_set)) / 10.0)

    print("\n" + "="*50)
    print(f" BENCHMARK RESULTS ({dataset.shape[0]} vectors)")
    print("="*50)
    print(f" Metric        | FAISS         | Custom")
    print("-" * 50)
    print(f" Recall@10     | {np.mean(f_recalls)*100:.2f}%        | {np.mean(c_recalls)*100:.2f}%")
    print(f" Latency (ms)  | {np.mean(f_times)*1000:.2f}         | {np.mean(c_times)*1000:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()