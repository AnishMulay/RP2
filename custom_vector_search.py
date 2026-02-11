import torch
import faiss
import time

class BruteForceSearcher:
    """Ground truth exact nearest neighbors."""
    def __init__(self, dataset):
        self.dataset = dataset

    def search(self, queries, top_k=10):
        # Using pure cdist for absolute exactness
        dists = torch.cdist(queries, self.dataset, p=2)
        top_dists, top_indices = torch.topk(dists, top_k, largest=False, dim=1)
        return top_indices

class KLevelSearcher:
    """ANN Search using your custom hierarchical index."""
    def __init__(self, index, n_probe=1):
        self.index = index
        self.n_probe = n_probe # How many centroids to scan

    def search(self, queries, top_k=10):
        results = []
        for q in queries:
            q = q.unsqueeze(0)
            
            # Step A: Coarse Quantization (Find nearest centroid)
            centroid_dists = torch.cdist(q, self.index.centroids, p=2).squeeze(0)
            best_centroid_idxs = torch.topk(centroid_dists, self.n_probe, largest=False)[1].cpu().numpy()
            
            # Step B: Gather cluster candidates
            candidate_idxs = set()
            for c_idx in best_centroid_idxs:
                candidate_idxs.update(self.index.centroid_to_points[c_idx])
            
            candidate_idxs = list(candidate_idxs)
            if len(candidate_idxs) < top_k:
                # Fallback if cluster is too small
                candidate_idxs = list(range(self.index.dataset.shape[0]))
                
            candidate_vectors = self.index.dataset[candidate_idxs]
            
            # Step C: Fine Search
            fine_dists = torch.cdist(q, candidate_vectors, p=2).squeeze(0)
            best_fine_idxs = torch.topk(fine_dists, min(top_k, len(candidate_idxs)), largest=False)[1]
            
            # Map back to global IDs
            global_best = [candidate_idxs[i.item()] for i in best_fine_idxs]
            results.append(global_best)
            
        return torch.tensor(results, device=queries.device)

class FaissSearcher:
    """Industry standard FAISS IVFFlat for comparison."""
    def __init__(self, dataset, n_centroids, n_probe=1):
        self.d = dataset.shape[1]
        quantizer = faiss.IndexFlatL2(self.d)
        self.index = faiss.IndexIVFFlat(quantizer, self.d, n_centroids, faiss.METRIC_L2)
        
        # FAISS requires CPU numpy arrays
        ds_np = dataset.cpu().numpy()
        print(f"[*] Training FAISS with {n_centroids} centroids...")
        self.index.train(ds_np)
        self.index.add(ds_np)
        self.index.nprobe = n_probe

    def search(self, queries, top_k=10):
        q_np = queries.cpu().numpy()
        D, I = self.index.search(q_np, top_k)
        return torch.tensor(I, device=queries.device)