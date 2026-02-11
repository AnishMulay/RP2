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
        } #

    def compute_dist_tile(self, query_points, workspace):
        P = workspace["P"]
        P_norms_sq = workspace["P_norms_sq"]
        Q_norms_sq = (query_points ** 2).sum(dim=1, keepdim=True).t()
        
        dists_sq = P_norms_sq + Q_norms_sq
        dists_sq.addmm_(P, query_points.t(), beta=1.0, alpha=-2.0)
        return torch.clamp(dists_sq, min=0.0) #

class FastGPUMultiLevelClustering:
    """Core hierarchical clustering adapted for a single vector dataset."""
    def __init__(self, epsilon, k=4, batch_size=2048):
        self.epsilon = epsilon
        self.k = k
        self.batch_size = batch_size
        self.kernel = TiledEuclideanKernel(chunk_size=batch_size) #

    def _sample_disjoint_hierarchy(self, n, device):
        prob = n ** (-1.0 / self.k)
        levels = torch.zeros(n, dtype=torch.long, device=device)
        for _ in range(self.k - 1):
            mask = torch.rand(n, device=device) <= prob
            levels += mask.long()
            
        if (levels == self.k - 1).sum() == 0:
            levels[torch.randint(0, n, (1,), device=device)] = self.k - 1
        return levels #

    def _process_level(self, targets, centers, center_indices, bounds, workspace):
        n_centers = centers.shape[0]
        new_bounds = bounds.clone()
        chunk_center_ids, chunk_point_ids = [], []
        
        for start in range(0, n_centers, self.batch_size):
            end = min(start + self.batch_size, n_centers)
            batch_centers = centers[start:end]
            batch_indices = center_indices[start:end]
            
            dists = self.kernel.compute_dist_tile(batch_centers, workspace).t() #
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
        return edges, new_bounds #

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
    """Wraps the clustering engine into an Inverted File (IVF) style vector database index."""
    def __init__(self, epsilon, k=4, batch_size=2048):
        self.cluster_engine = FastGPUMultiLevelClustering(epsilon, k, batch_size)
        self.centroids = None
        self.col_indices = None
        self.crow_indices = None
        self.dataset = None

    def build_index(self, X):
        print(f"[*] Building K-Level Index on {X.shape[0]} vectors...")
        self.dataset = X
        
        # Run clustering
        centers, points, levels = self.cluster_engine.run(X)
        
        # Isolate Level (k-1) centroids (our "Landmarks")
        top_level_mask = (levels == self.cluster_engine.k - 1)
        top_level_indices = torch.nonzero(top_level_mask).squeeze(1)
        self.centroids = X[top_level_indices]

        # Build CSR inverted lists:
        # - crow_indices: centroid row offsets (size M+1)
        # - col_indices: flattened point IDs for all centroid rows
        num_centroids = top_level_indices.numel()
        top_level_indices_cpu = top_level_indices.to(device="cpu", dtype=torch.long)
        centers_cpu = centers.to(device="cpu", dtype=torch.long)
        points_cpu = points.to(device="cpu", dtype=torch.long)

        global_to_local = torch.full((X.shape[0],), -1, dtype=torch.long)
        if num_centroids > 0:
            global_to_local[top_level_indices_cpu] = torch.arange(num_centroids, dtype=torch.long)

        local_center_ids = global_to_local[centers_cpu]
        valid_mask = local_center_ids >= 0
        local_center_ids = local_center_ids[valid_mask]
        point_ids = points_cpu[valid_mask]

        if point_ids.numel() > 0:
            pairs = torch.stack((local_center_ids, point_ids), dim=1)
            unique_pairs = torch.unique(pairs, dim=0)
            sort_order = torch.argsort(unique_pairs[:, 0])
            sorted_centers = unique_pairs[sort_order, 0]
            sorted_points = unique_pairs[sort_order, 1]
            counts = torch.bincount(sorted_centers, minlength=num_centroids)
        else:
            sorted_points = torch.empty(0, dtype=torch.long)
            counts = torch.zeros(num_centroids, dtype=torch.long)

        crow_indices = torch.zeros(num_centroids + 1, dtype=torch.long)
        if num_centroids > 0:
            crow_indices[1:] = torch.cumsum(counts, dim=0)

        self.col_indices = sorted_points.to(X.device)
        self.crow_indices = crow_indices.to(X.device)

        cluster_sizes = (crow_indices[1:] - crow_indices[:-1]).to(torch.float32)
        if cluster_sizes.numel() > 0:
            print(
                "[*] Cluster size stats (top-level): "
                f"min={int(cluster_sizes.min().item())}, "
                f"max={int(cluster_sizes.max().item())}, "
                f"mean={cluster_sizes.mean().item():.2f}, "
                f"median={cluster_sizes.median().item():.2f}"
            )

        print(f"[*] Index built with {len(self.centroids)} top-level centroids.")
