import torch
from ..utils.distance import TiledEuclideanKernel, TiledManhattanKernel

class FastGPUMultiLevelClustering:
    """
    Implements the Multi-Level Hierarchical Clustering (Decomposition) on GPU.

    This stateful engine partitions point clouds into a hierarchy of clusters, 
    forming a spatial tree to accelerate distance and flow computations.

    Args:
        epsilon (float): The base radius/scale parameter for clustering discretization.
        k (int, optional): Number of levels in the clustering hierarchy. Defaults to 4.
        batch_size (int, optional): GPU batch size for processing centers. Defaults to 2048.
        metric (str, optional): Distance metric to use ("L2" or "L1"). Defaults to "L2".
    """
    def __init__(self, epsilon, k=4, batch_size=2048, metric="L2", profile_memory=False):
        self.epsilon = epsilon
        self.k = k
        self.batch_size = batch_size
        self.metric = metric
        self.profile_memory = bool(profile_memory)
        self.memory_profile = {}
        if metric == "L1":
            self.kernel = TiledManhattanKernel(chunk_size=batch_size)
        else:
            self.kernel = TiledEuclideanKernel(chunk_size=batch_size)

    # ── Opt-in stage-level peak-memory profiling (default OFF, zero overhead
    # when disabled). No-op on CPU. ─────────────────────────────────────────
    def _prof_reset(self):
        if self.profile_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _prof_record(self, label):
        if self.profile_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.memory_profile[label] = torch.cuda.max_memory_allocated() / 1024 ** 3

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
        n_targets = targets.shape[0]
        
        chunk_center_ids = []
        chunk_point_ids = []
        chunk_levels = []
        
        new_bounds = bounds.clone()
        
        for start in range(0, n_centers, self.batch_size):
            end = min(start + self.batch_size, n_centers)
            batch_centers = centers[start:end]
            batch_indices = center_indices[start:end]
            
            dists = self.kernel.compute_dist_tile(batch_centers, workspace).t()
            batch_min, _ = dists.min(dim=0)
            new_bounds = torch.minimum(new_bounds, batch_min)
            
            mask = dists < bounds.unsqueeze(0)
            if not mask.any():
                continue
                
            valid_indices = torch.nonzero(mask)
            
            if valid_indices.numel() > 0:
                rows = valid_indices[:, 0]
                cols = valid_indices[:, 1]
                
                valid_dists_raw = dists[rows, cols]
                if self.metric == "L2":
                    valid_dists = torch.sqrt(valid_dists_raw)
                else:
                    valid_dists = valid_dists_raw
                
                levels = torch.ceil(valid_dists / self.epsilon).to(torch.long)
                global_c = batch_indices[rows]
                
                chunk_center_ids.append(global_c.cpu())
                chunk_point_ids.append(cols.cpu())
                chunk_levels.append(levels.cpu())
                
            del dists, mask, valid_indices
        
        if chunk_center_ids:
            edges = (torch.cat(chunk_center_ids), torch.cat(chunk_levels), torch.cat(chunk_point_ids))
        else:
            edges = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
            
        return edges, new_bounds

    def run(self, P_red, P_blue):
        if self.profile_memory:
            self.memory_profile = {}

        P_all = torch.cat([P_red, P_blue], dim=0)
        n = P_all.shape[0]
        workspace = self.kernel.prepare_workspace(P_all)

        self._prof_reset()
        levels_red = self._sample_disjoint_hierarchy(P_red.shape[0], P_red.device)
        levels_blue = self._sample_disjoint_hierarchy(P_blue.shape[0], P_blue.device)
        self._prof_record("landmark_sampling")

        def build_cover(centers_source, levels_source):
            all_c, all_l, all_p = [], [], []
            bounds = torch.full((n,), float('inf'), device=P_all.device)
            
            for i in range(self.k - 1, -1, -1):
                mask_i = (levels_source == i)
                if not mask_i.any():
                    continue
                
                idx_i = torch.nonzero(mask_i).squeeze(1)
                pts_i = centers_source[idx_i]
                
                (c_cpu, l_cpu, p_cpu), bounds = self._process_level(
                    P_all, pts_i, idx_i, bounds, workspace
                )
                
                all_c.append(c_cpu)
                all_l.append(l_cpu)
                all_p.append(p_cpu)
                
            if not all_c:
                return (torch.empty(0, dtype=torch.long),)*3
            return (torch.cat(all_c), torch.cat(all_l), torch.cat(all_p))

        self._prof_reset()
        red_coo = build_cover(P_red, levels_red)
        blue_coo = build_cover(P_blue, levels_blue)
        self._prof_record("build_cover")
        return blue_coo, red_coo, levels_red, levels_blue

def k_level_cluster(x, y, epsilon, k=4, batch_size=2048, metric="L2"):
    """
    Public functional interface for K-Level Clustering.

    Partitions two point clouds into a K-level unified hierarchy of spatial cells.

    Args:
        x (torch.Tensor): Source point cloud of shape (N, D).
        y (torch.Tensor): Target point cloud of shape (M, D).
        epsilon (float): Discretization base radius parameter.
        k (int, optional): Number of hierarchy levels. Defaults to 4.
        batch_size (int, optional): GPU batch size for distance calculations. Defaults to 2048.
        metric (str, optional): Distance metric ("L2" or "L1"). Defaults to "L2".

    Returns:
        dict: A dictionary containing:
            - 'blue_cover' (tuple): COO tensor representation of edges (centers, levels, points) for the target (blue) points.
            - 'red_cover' (tuple): COO tensor representation of edges (centers, levels, points) for the source (red) points.
            - 'levels_red' (torch.Tensor): Sampled discrete level assignments for each source point.
            - 'levels_blue' (torch.Tensor): Sampled discrete level assignments for each target point.
    """
    model = FastGPUMultiLevelClustering(epsilon=epsilon, k=k, batch_size=batch_size, metric=metric)
    blue_coo, red_coo, levels_red, levels_blue = model.run(x, y)
    return {
        "blue_cover": blue_coo,
        "red_cover": red_coo,
        "levels_red": levels_red,
        "levels_blue": levels_blue
    }

