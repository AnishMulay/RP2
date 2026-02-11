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
    Searcher for all-points Voronoi index.
    Strictly NO batching. Searches one query at a time on GPU.
    """
    def __init__(self, index):
        self.index = index

    def search_one(self, query, top_k=10):
        """
        Performs all-points pivot scan + CSR cluster refinement for one query.
        query: (D,) Tensor on GPU
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Step A: Global scan over all points
        dists_all = torch.cdist(query, self.index.dataset, p=2).squeeze(0)  # (N,)

        # Step B: Pivot = nearest point globally
        best_pivot = torch.argmin(dists_all)

        # Step C: Retrieve that pivot's cluster from CSR
        start = int(self.index.crow_indices[best_pivot].item())
        end = int(self.index.crow_indices[best_pivot + 1].item())
        cluster_ids = self.index.col_indices[start:end]

        if cluster_ids.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=query.device)

        # Step D: Reuse precomputed global distances (no recomputation)
        cluster_dists = dists_all[cluster_ids]
        k_actual = min(top_k, cluster_ids.numel())
        local_top_k = torch.topk(cluster_dists, k_actual, largest=False).indices
        return cluster_ids[local_top_k]

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
