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
    Leveled all-points searcher.
    Step 1: Global scan to pick pivot.
    Step 2: Gather candidates from:
      - ForwardIndex[pivot]
      - All ForwardIndex[center] where center in InvertedIndex[pivot]
    Step 3: Exact re-score on unique candidates.
    """
    def __init__(self, index):
        self.index = index

    def search_one(self, query, top_k=10):
        if query.dim() == 1: query = query.unsqueeze(0)

        # Step 1: Find nearest pivot across whole dataset
        dists_global = torch.cdist(query, self.index.dataset)
        pivot_idx = torch.argmin(dists_global, dim=1).squeeze(0)
        pivot_idx_long = pivot_idx.to(torch.long)
        device = query.device

        # Step 2A: forward expansion from pivot-as-center
        start = int(self.index.crow_indices[pivot_idx_long].item())
        end = int(self.index.crow_indices[pivot_idx_long + 1].item())
        candidates_forward = self.index.col_indices[start:end]

        # Step 2B: inverted expansion from pivot-as-member (already sorted by level)
        p_start = int(self.index.point_crow_indices[pivot_idx_long].item())
        p_end = int(self.index.point_crow_indices[pivot_idx_long + 1].item())
        member_centers = self.index.point_col_indices[p_start:p_end]

        if member_centers.numel() > 0:
            c_starts = self.index.crow_indices[member_centers]
            c_ends = self.index.crow_indices[member_centers + 1]
            lengths = c_ends - c_starts
            total = int(lengths.sum().item())

            if total > 0:
                offsets = torch.cumsum(lengths, dim=0) - lengths
                flat_pos = torch.arange(total, device=device, dtype=torch.long)
                repeated_starts = torch.repeat_interleave(c_starts, lengths)
                repeated_offsets = torch.repeat_interleave(offsets, lengths)
                flat_idx = repeated_starts + (flat_pos - repeated_offsets)
                candidates_inverted = self.index.col_indices[flat_idx]
            else:
                candidates_inverted = torch.empty(0, dtype=torch.long, device=device)
        else:
            candidates_inverted = torch.empty(0, dtype=torch.long, device=device)

        # Step 2C: merge candidate sources (+ pivot fallback)
        pivot_tensor = pivot_idx_long.unsqueeze(0)
        all_candidates = torch.cat([candidates_forward, candidates_inverted, pivot_tensor], dim=0)
        unique_candidates = torch.unique(all_candidates, sorted=False)

        # Step 3: exact re-scoring on unique candidates only
        local_points = self.index.dataset[unique_candidates]
        dists_local = torch.cdist(query, local_points)
        k_actual = min(top_k, unique_candidates.numel())
        local_top_k = torch.topk(dists_local, k_actual, largest=False).indices.squeeze(0)
        return unique_candidates[local_top_k]

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
