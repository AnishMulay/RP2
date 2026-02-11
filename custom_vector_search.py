import torch
import faiss

class BruteForceSearcher:
    """Ground truth exact nearest neighbors."""
    def __init__(self, dataset):
        self.dataset = dataset

    def search(self, query, top_k=10):
        if query.dim() == 1: query = query.unsqueeze(0)
        dists = torch.cdist(query, self.dataset)
        return torch.topk(dists, top_k, largest=False).indices.squeeze(0)

class KLevelSearcher:
    """
    All-Points Searcher.
    Step 1: Scan ALL points to find the Pivot (p_best).
    Step 2: Retrieve C_{p_best} (points closer to p_best than to S).
    Step 3: Scan C_{p_best}.
    """
    def __init__(self, index):
        self.index = index

    def search_one(self, query, top_k=10):
        if query.dim() == 1: query = query.unsqueeze(0)

        # Step 1: Find the nearest Pivot (Global Scan over P)
        # Note: This is O(N), same as Brute Force.
        # We do this to validate the CLUSTERING logic (Recall), not speed.
        dists_global = torch.cdist(query, self.index.dataset)
        pivot_idx = torch.argmin(dists_global).item()

        # Step 2: Retrieve the Cluster (The Bucket)
        start = self.index.crow_indices[pivot_idx]
        end = self.index.crow_indices[pivot_idx + 1]
        
        # If cluster is empty (rare, but possible if pivot is a landmark), handle it
        if end == start:
            return torch.tensor([pivot_idx], device=query.device)

        candidate_indices = self.index.col_indices[start:end]
        local_points = self.index.dataset[candidate_indices]

        # Step 3: Local Scan
        dists_local = torch.cdist(query, local_points)
        
        k_actual = min(top_k, local_points.shape[0])
        local_top_k = torch.topk(dists_local, k_actual, largest=False).indices.squeeze(0)
        
        # Map back to global IDs
        global_ids = candidate_indices[local_top_k]
        
        return global_ids

class FaissSearcher:
    """FAISS Wrapper."""
    def __init__(self, dataset, n_centroids, n_probe=1):
        self.d = dataset.shape[1]
        quantizer = faiss.IndexFlatL2(self.d)
        self.index = faiss.IndexIVFFlat(quantizer, self.d, n_centroids, faiss.METRIC_L2)
        self.index.train(dataset.cpu().numpy())
        self.index.add(dataset.cpu().numpy())
        self.index.nprobe = n_probe

    def search_one(self, query, top_k=10):
        q_np = query.cpu().numpy().reshape(1, -1)
        _, I = self.index.search(q_np, top_k)
        return torch.tensor(I[0], device=query.device)