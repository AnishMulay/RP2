import torch
from ..utils.distance import TiledEuclideanKernel, TiledManhattanKernel

class FastGPUMultiLevelClustering:
    """
    Implements the Multi-Level Hierarchical Clustering (Decomposition) on GPU.
    """
    def __init__(self, epsilon, k=4, batch_size=2048, metric="L2"):
        self.epsilon = epsilon
        self.k = k
        self.batch_size = batch_size
        self.metric = metric
        if metric == "L1":
            self.kernel = TiledManhattanKernel(chunk_size=batch_size)
        else:
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
        P_all = torch.cat([P_red, P_blue], dim=0)
        n = P_all.shape[0]
        workspace = self.kernel.prepare_workspace(P_all)
        
        levels_red = self._sample_disjoint_hierarchy(P_red.shape[0], P_red.device)
        levels_blue = self._sample_disjoint_hierarchy(P_blue.shape[0], P_blue.device)
        
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

        red_coo = build_cover(P_red, levels_red)
        blue_coo = build_cover(P_blue, levels_blue)
        return blue_coo, red_coo, levels_red, levels_blue

def k_level_cluster(x, y, epsilon, k=4, batch_size=2048, metric="L2"):
    """Public functional interface for K-Level Clustering."""
    model = FastGPUMultiLevelClustering(epsilon=epsilon, k=k, batch_size=batch_size, metric=metric)
    blue_coo, red_coo, levels_red, levels_blue = model.run(x, y)
    return {
        "blue_cover": blue_coo,
        "red_cover": red_coo,
        "levels_red": levels_red,
        "levels_blue": levels_blue
    }

