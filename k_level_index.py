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
    A Flattened K-Level Index that resides entirely on GPU.
    It uses the clustering hierarchy to find centroids, then builds a flat Partitioned Index.
    """
    def __init__(self, epsilon, k=4, batch_size=2048):
        self.cluster_engine = FastGPUMultiLevelClustering(epsilon, k, batch_size)
        self.centroids = None      # (K, D) Tensor
        self.sorted_dataset = None # (N, D) Tensor
        self.offsets = None        # (K+1,) Tensor
        self.original_indices = None # (N,) Tensor (to map back to global IDs)

    def build_index(self, X):
        print(f"[*] Building K-Level Index on {X.shape[0]} vectors (GPU)...")
        
        # 1. Run Clustering to identify the hierarchy
        _, _, levels = self.cluster_engine.run(X)
        
        # 2. Extract Top-Level Centroids (Level k-1)
        # These act as the 'GPS Satellites' for our search
        top_level_mask = (levels == self.cluster_engine.k - 1)
        top_level_indices = torch.nonzero(top_level_mask).squeeze(1)
        self.centroids = X[top_level_indices].clone() # Keep on GPU
        
        num_centroids = self.centroids.shape[0]
        print(f"[*] Identified {num_centroids} top-level centroids.")

        # 3. Assign every point in X to the nearest Top-Level Centroid
        # This creates the "Partition" for the flattened search.
        # We process in chunks to avoid OOM on huge datasets
        assignments = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        chunk_size = 4096
        
        for i in range(0, X.shape[0], chunk_size):
            end = min(i + chunk_size, X.shape[0])
            batch_X = X[i:end]
            # Compute dists to centroids
            dists = torch.cdist(batch_X, self.centroids)
            # Assign to closest
            assignments[i:end] = torch.argmin(dists, dim=1)

        # 4. Sort the dataset by Cluster Assignment
        # This groups all points for Centroid 0 together, then Centroid 1, etc.
        sort_indices = torch.argsort(assignments)
        
        self.sorted_dataset = X[sort_indices].clone()
        self.original_indices = sort_indices.clone() # Map back to global ID
        
        sorted_assignments = assignments[sort_indices]
        
        # 5. Build Offsets (Where does Cluster i begin and end?)
        # counts = number of points in each cluster
        counts = torch.bincount(sorted_assignments, minlength=num_centroids)
        
        # offsets = cumsum of counts
        self.offsets = torch.zeros(num_centroids + 1, dtype=torch.long, device=X.device)
        self.offsets[1:] = torch.cumsum(counts, dim=0)

        # Statistics
        cluster_sizes = counts.float()
        print(
            "[*] Index Structure (Flattened): "
            f"min_cluster={int(cluster_sizes.min().item())}, "
            f"max_cluster={int(cluster_sizes.max().item())}, "
            f"avg_cluster={cluster_sizes.mean().item():.1f}"
        )