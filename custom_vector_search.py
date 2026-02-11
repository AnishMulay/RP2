import torch
import faiss
import time

class BruteForceSearcher:
    """Ground truth exact nearest neighbors."""
    def __init__(self, dataset):
        self.dataset = dataset

    def search(self, query, top_k=10):
        # Ensure query is (1, D)
        if query.dim() == 1:
            query = query.unsqueeze(0)
            
        dists = torch.cdist(query, self.dataset, p=2)
        top_dists, top_indices = torch.topk(dists, top_k, largest=False, dim=1)
        return top_indices.squeeze(0)

class KLevelSearcher:
    """
    Simplified Flattened Searcher.
    Strictly NO batching. Searches one query at a time on GPU.
    """
    def __init__(self, index):
        self.index = index

    def search_one(self, query, top_k=10):
        """
        Performs the 2-step lookup for a SINGLE query vector.
        query: (D,) Tensor on GPU
        """
        if query.dim() == 1:
            query = query.unsqueeze(0) # (1, D)

        # Step 1: Find the nearest Top-Level Centroid (The Pivot)
        # Global Search over K centroids
        dists_global = torch.cdist(query, self.index.centroids, p=2) # (1, K)
        best_centroid_idx = torch.argmin(dists_global).item()

        # Step 2: Retrieve the specific cluster (The Bucket)
        start = self.index.offsets[best_centroid_idx]
        end = self.index.offsets[best_centroid_idx + 1]
        
        # Zero-copy slice from VRAM
        local_points = self.index.sorted_dataset[start:end]
        
        # Step 3: Local Search within the bucket
        # If bucket is empty or smaller than k, handle gracefully
        if local_points.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=query.device)

        dists_local = torch.cdist(query, local_points, p=2) # (1, ClusterSize)
        
        k_actual = min(top_k, local_points.shape[0])
        local_top_k = torch.topk(dists_local, k_actual, largest=False, dim=1).indices.squeeze(0)
        
        # Step 4: Map back to Global IDs
        # We found the index in the 'sorted' dataset (offset by start)
        sorted_indices = local_top_k + start
        global_ids = self.index.original_indices[sorted_indices]
        
        return global_ids

class FaissSearcher:
    """Industry standard FAISS IVFFlat wrapper for single-query comparison."""
    def __init__(self, dataset, n_centroids, n_probe=1):
        self.d = dataset.shape[1]
        quantizer = faiss.IndexFlatL2(self.d)
        self.index = faiss.IndexIVFFlat(quantizer, self.d, n_centroids, faiss.METRIC_L2)
        
        # FAISS requires CPU numpy arrays for build
        ds_np = dataset.cpu().numpy()
        print(f"[*] Training FAISS with {n_centroids} centroids...")
        self.index.train(ds_np)
        self.index.add(ds_np)
        self.index.nprobe = n_probe

    def search_one(self, query, top_k=10):
        # FAISS expects (1, D) numpy array
        q_np = query.cpu().numpy().reshape(1, -1)
        D, I = self.index.search(q_np, top_k)
        return torch.tensor(I[0], device=query.device)