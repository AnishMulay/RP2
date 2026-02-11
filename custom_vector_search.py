import torch
import faiss
import time
import math

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

    def _resolve_pruned_end(self, center_id, row_start, row_end, radius_level):
        if radius_level is None or self.index.bucket_offsets is None:
            return row_end

        bucket_meta = self.index.bucket_offsets
        row_ptr = bucket_meta["row_ptr"]
        levels = bucket_meta["levels"]
        ends = bucket_meta["ends"]

        bucket_start = int(row_ptr[center_id].item())
        bucket_end = int(row_ptr[center_id + 1].item())
        if bucket_end <= bucket_start:
            return row_end

        level_segment = levels[bucket_start:bucket_end]
        end_segment = ends[bucket_start:bucket_end]
        keep_count = int(torch.searchsorted(level_segment, radius_level, right=True).item())
        if keep_count == 0:
            return row_start

        pruned_end = int(end_segment[keep_count - 1].item())
        return min(pruned_end, row_end)

    def search(self, queries, top_k=10, search_radius_level=None):
        if queries.shape[0] == 0:
            return torch.empty((0, 0), dtype=torch.long, device=queries.device)

        radius_level = None
        if search_radius_level is not None:
            radius_value = float(search_radius_level)
            if not math.isinf(radius_value):
                radius_level = int(radius_value)

        # Step A: Batched coarse quantization for all queries.
        centroid_dists = torch.cdist(queries, self.index.centroids, p=2)
        best_centroid_idxs = torch.topk(
            centroid_dists, self.n_probe, largest=False, dim=1
        ).indices

        # Step B: Build ragged candidate sets from CSR, then pad to a dense tensor.
        crow = self.index.crow_indices
        col = self.index.col_indices
        dataset_size = self.index.dataset.shape[0]
        full_dataset_idxs = torch.arange(dataset_size, device=queries.device)

        candidate_lists = []
        candidate_sizes = []
        for q_centroids in best_centroid_idxs:
            starts = crow[q_centroids]
            ends = crow[q_centroids + 1]

            query_slices = []
            for center_id, start, end in zip(q_centroids.tolist(), starts.tolist(), ends.tolist()):
                pruned_end = self._resolve_pruned_end(
                    center_id=center_id,
                    row_start=start,
                    row_end=end,
                    radius_level=radius_level,
                )
                if pruned_end > start:
                    query_slices.append(col[start:pruned_end])

            if query_slices:
                candidate_idxs = torch.unique(torch.cat(query_slices))
            else:
                candidate_idxs = torch.empty(0, dtype=torch.long, device=queries.device)

            if candidate_idxs.numel() < top_k:
                candidate_idxs = full_dataset_idxs

            candidate_lists.append(candidate_idxs)
            candidate_sizes.append(candidate_idxs.numel())

        max_cluster_size = max(candidate_sizes) if candidate_sizes else 0
        candidate_vectors = torch.zeros(
            (queries.shape[0], max_cluster_size, queries.shape[1]),
            device=queries.device,
            dtype=self.index.dataset.dtype,
        )
        candidate_ids_padded = torch.full(
            (queries.shape[0], max_cluster_size),
            -1,
            device=queries.device,
            dtype=torch.long,
        )
        valid_mask = torch.zeros(
            (queries.shape[0], max_cluster_size),
            device=queries.device,
            dtype=torch.bool,
        )

        for q_idx, candidate_idxs in enumerate(candidate_lists):
            count = candidate_idxs.numel()
            if count == 0:
                continue
            candidate_vectors[q_idx, :count] = self.index.dataset[candidate_idxs]
            candidate_ids_padded[q_idx, :count] = candidate_idxs
            valid_mask[q_idx, :count] = True

        # Step C: Batched fine search over padded candidate matrix.
        fine_dists = torch.cdist(queries.unsqueeze(1), candidate_vectors, p=2).squeeze(1)
        fine_dists = fine_dists.masked_fill(~valid_mask, float("inf"))
        k_eff = min(top_k, max_cluster_size)
        best_fine_local = torch.topk(fine_dists, k_eff, largest=False, dim=1).indices
        global_best = torch.gather(candidate_ids_padded, 1, best_fine_local)

        total_candidates_evaluated = int(sum(candidate_sizes))
        candidate_ratio = total_candidates_evaluated / (queries.shape[0] * dataset_size)
        print(
            f"[*] Candidate Ratio: {candidate_ratio:.4f} "
            f"({total_candidates_evaluated} / {queries.shape[0] * dataset_size})"
        )
        if candidate_ratio > 0.5:
            print("[!] Candidate Ratio exceeds 50%; top-level clusters are likely too large.")

        return global_best

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
