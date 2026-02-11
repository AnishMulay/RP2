import math
import torch

class TiledEuclideanKernel:
    """Computes distances ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>"""
    def __init__(self, chunk_size=4096):
        self.chunk_size = chunk_size

    def prepare_workspace(self, P):
        return {
            "P": P,
            "P_T": P.t(),
            "P_norms_sq": (P ** 2).sum(dim=1, keepdim=True)
        }

    def compute_dist_tile(self, query_points, workspace):
        P = workspace["P"]
        P_norms_sq = workspace["P_norms_sq"]
        Q_norms_sq = (query_points ** 2).sum(dim=1, keepdim=True).t()
        
        dists_sq = P_norms_sq + Q_norms_sq
        dists_sq.addmm_(P, query_points.t(), beta=1.0, alpha=-2.0)
        return torch.clamp(dists_sq, min=0.0)

class FastGPUMultiLevelClustering:
    """Core hierarchical clustering adapted for a single vector dataset."""
    def __init__(self, epsilon, k=4, batch_size=2048):
        self.epsilon = epsilon
        self.k = k
        self.batch_size = batch_size
        self.kernel = TiledEuclideanKernel(chunk_size=batch_size)

    def _sample_disjoint_hierarchy(self, n, device):
        prob = n ** (-1.0 / self.k)
        levels = torch.zeros(n, dtype=torch.long, device=device)
        for _ in range(self.k - 1):
            mask = torch.rand(n, device=device) <= prob
            levels += mask.long()
            
        if (levels == self.k - 1).sum() == 0:
            levels[torch.randint(0, n, (1,), device=device)] = self.k - 1
        return levels

    def _process_level(self, targets, centers, center_indices, bounds, workspace):
        n_centers = centers.shape[0]
        new_bounds = bounds.clone()
        chunk_center_ids, chunk_point_ids = [], []
        
        for start in range(0, n_centers, self.batch_size):
            end = min(start + self.batch_size, n_centers)
            batch_centers = centers[start:end]
            batch_indices = center_indices[start:end]
            
            dists = self.kernel.compute_dist_tile(batch_centers, workspace).t()
            batch_min, _ = dists.min(dim=0)
            new_bounds = torch.minimum(new_bounds, batch_min)
            
            mask = dists < bounds.unsqueeze(0)
            if not mask.any(): continue
                
            valid_indices = torch.nonzero(mask)
            if valid_indices.numel() > 0:
                rows, cols = valid_indices[:, 0], valid_indices[:, 1]
                global_c = batch_indices[rows]
                chunk_center_ids.append(global_c.cpu())
                chunk_point_ids.append(cols.cpu())
            del dists, mask, valid_indices
        
        if chunk_center_ids:
            edges = (torch.cat(chunk_center_ids), torch.cat(chunk_point_ids))
        else:
            edges = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        return edges, new_bounds

    def run(self, X):
        n = X.shape[0]
        workspace = self.kernel.prepare_workspace(X)
        levels = self._sample_disjoint_hierarchy(n, X.device)
        
        bounds = torch.full((n,), float('inf'), device=X.device)
        all_c, all_p = [], []
        
        for i in range(self.k - 1, -1, -1):
            mask_i = (levels == i)
            if not mask_i.any(): continue
            idx_i = torch.nonzero(mask_i).squeeze(1)
            pts_i = X[idx_i]
            
            (c_cpu, p_cpu), bounds = self._process_level(X, pts_i, idx_i, bounds, workspace)
            all_c.append(c_cpu)
            all_p.append(p_cpu)
            
        return torch.cat(all_c), torch.cat(all_p), levels


class KLevelVectorIndex:
    """
    Index for the advisor-defined 2-level Voronoi experiment:
    - Sample sqrt(N) landmarks S
    - For every pivot p in dataset, define cluster C_p = {y | dist(y, p) <= dist(y, S)}
    - Store all clusters in CSR with one row per pivot point.
    """
    def __init__(self, epsilon, k=2, batch_size=2048, pivot_batch_size=1000, landmark_batch_size=4096):
        self.epsilon = epsilon
        self.k = k
        self.batch_size = batch_size
        self.pivot_batch_size = pivot_batch_size
        self.landmark_batch_size = landmark_batch_size

        # Main data for search
        self.dataset = None              # (N, D)
        self.crow_indices = None         # (N + 1,)
        self.col_indices = None          # (num_edges,)

        # Diagnostic/compatibility fields
        self.landmark_indices = None     # (sqrt(N),)
        self.nearest_landmark_sq = None  # (N,)
        self.centroids = None            # alias to sampled landmarks

    def _sample_landmarks(self, n, device):
        n_landmarks = max(1, int(math.sqrt(n)))
        return torch.randperm(n, device=device)[:n_landmarks]

    def _squared_distances(self, targets, queries, targets_norms_sq, query_norms_sq):
        dists_sq = targets_norms_sq + query_norms_sq.view(1, -1)
        dists_sq.addmm_(targets, queries.t(), beta=1.0, alpha=-2.0)
        return torch.clamp(dists_sq, min=0.0)

    def _compute_nearest_landmark_sq(self, X, landmark_indices, X_norms_sq):
        landmarks = X[landmark_indices]
        n_landmarks = landmarks.shape[0]
        nearest_sq = torch.full((X.shape[0],), float("inf"), device=X.device)

        for start in range(0, n_landmarks, self.landmark_batch_size):
            end = min(start + self.landmark_batch_size, n_landmarks)
            landmark_batch = landmarks[start:end]
            landmark_norms_sq = (landmark_batch ** 2).sum(dim=1)
            dists_sq = self._squared_distances(X, landmark_batch, X_norms_sq, landmark_norms_sq)
            nearest_sq = torch.minimum(nearest_sq, dists_sq.min(dim=1).values)
            del dists_sq

        return nearest_sq

    def build_index(self, X):
        print(f"[*] Building all-points Voronoi index on {X.shape[0]} vectors (GPU)...")

        with torch.no_grad():
            if not X.is_floating_point():
                X = X.to(torch.float32)

            self.dataset = X.contiguous()
            n = self.dataset.shape[0]
            device = self.dataset.device

            if n == 0:
                self.landmark_indices = torch.empty(0, dtype=torch.long, device=device)
                self.centroids = torch.empty(0, self.dataset.shape[1], dtype=self.dataset.dtype, device=device)
                self.nearest_landmark_sq = torch.empty(0, dtype=self.dataset.dtype, device=device)
                self.crow_indices = torch.zeros(1, dtype=torch.long, device=device)
                self.col_indices = torch.empty(0, dtype=torch.long, device=device)
                return

            self.landmark_indices = self._sample_landmarks(n, device)
            self.centroids = self.dataset[self.landmark_indices]
            print(f"[*] Sampled {self.landmark_indices.numel()} landmarks (~sqrt(N)).")

            dataset_norms_sq = (self.dataset ** 2).sum(dim=1, keepdim=True)
            self.nearest_landmark_sq = self._compute_nearest_landmark_sq(
                self.dataset, self.landmark_indices, dataset_norms_sq
            )

            threshold_sq = self.nearest_landmark_sq.unsqueeze(1)
            row_counts_cpu = torch.zeros(n, dtype=torch.long)
            row_ids_cpu = []
            col_ids_cpu = []

            pivot_chunk = max(1, min(self.pivot_batch_size, n))
            for start in range(0, n, pivot_chunk):
                end = min(start + pivot_chunk, n)
                pivot_batch = self.dataset[start:end]
                pivot_norms_sq = dataset_norms_sq[start:end, 0]

                dists_sq = self._squared_distances(
                    self.dataset, pivot_batch, dataset_norms_sq, pivot_norms_sq
                )
                membership = dists_sq <= threshold_sq
                nz = torch.nonzero(membership, as_tuple=False)

                if nz.numel() > 0:
                    local_rows = nz[:, 1]
                    row_counts_cpu[start:end] += torch.bincount(
                        local_rows, minlength=end - start
                    ).cpu()
                    row_ids_cpu.append((local_rows + start).cpu())
                    col_ids_cpu.append(nz[:, 0].cpu())

                del dists_sq, membership, nz

            crow_cpu = torch.zeros(n + 1, dtype=torch.long)
            crow_cpu[1:] = torch.cumsum(row_counts_cpu, dim=0)

            if row_ids_cpu:
                row_ids = torch.cat(row_ids_cpu, dim=0)
                col_ids = torch.cat(col_ids_cpu, dim=0)
                order = torch.argsort(row_ids)
                col_sorted_cpu = col_ids[order]
            else:
                col_sorted_cpu = torch.empty(0, dtype=torch.long)

            self.crow_indices = crow_cpu.to(device=device)
            self.col_indices = col_sorted_cpu.to(device=device)

            cluster_sizes = row_counts_cpu.float()
            print(
                "[*] CSR ready: "
                f"rows={n}, "
                f"edges={int(self.col_indices.numel())}, "
                f"min_cluster={int(cluster_sizes.min().item())}, "
                f"max_cluster={int(cluster_sizes.max().item())}, "
                f"avg_cluster={cluster_sizes.mean().item():.1f}"
            )
