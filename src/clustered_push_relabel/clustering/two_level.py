import torch
from ..utils.distance import TiledEuclideanKernel, TiledManhattanKernel

class FastGPUClustering:
    """
    Optimized Clustering using Voronoi constraints and minimal shells.
    """
    def __init__(self, epsilon, batch_size=1024, micro_batch_size=32, metric="L2", profile_memory=False):
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
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

    def _sample_landmarks(self, n, device):
        prob = n ** (-0.5)
        mask = torch.rand(n, device=device) < prob
        if not mask.any(): mask[torch.randint(0, n, (1,), device=device)] = True
        return torch.nonzero(mask).squeeze(1), mask

    def _compute_voronoi_bounds(self, targets, landmarks, workspace):
        n_targets = targets.shape[0]
        n_landmarks = landmarks.shape[0]
        D_y = torch.full((n_targets,), float('inf'), device=targets.device)
        
        for i in range(0, n_landmarks, self.batch_size):
            end = min(i + self.batch_size, n_landmarks)
            batch = landmarks[i:end]
            dists = self.kernel.compute_dist_tile(batch, workspace)
            batch_min, _ = dists.min(dim=1)
            D_y = torch.min(D_y, batch_min)
        return D_y

    def _build_cover_bulk(self, centers, center_mask_P1, targets_ws, D_voronoi):
        n_centers = centers.shape[0]
        n_targets = targets_ws["P"].shape[0]
        
        chunk_center_ids = []
        chunk_level_ids = []
        chunk_point_ids = []

        for start_q in range(0, n_centers, self.batch_size):
            end_q = min(start_q + self.batch_size, n_centers)
            q_batch = centers[start_q:end_q]
            curr_bs = q_batch.shape[0]
            
            is_landmark_batch = center_mask_P1[start_q:end_q].view(curr_bs, 1)

            for start_mb in range(0, curr_bs, self.micro_batch_size):
                end_mb = min(start_mb + self.micro_batch_size, curr_bs)
                q_micro = q_batch[start_mb:end_mb]
                micro_is_landmark = is_landmark_batch[start_mb:end_mb]
                
                dists = self.kernel.compute_dist_tile(q_micro, targets_ws).t()
                dv_row = D_voronoi.view(1, n_targets)
                
                mask = dists < dv_row
                mask = torch.where(micro_is_landmark, torch.tensor(True, device=mask.device), mask)
                
                if not mask.any(): 
                    continue
                
                valid_indices = torch.nonzero(mask)
                valid_dists_raw = dists[valid_indices[:, 0], valid_indices[:, 1]]
                if self.metric == "L2":
                    valid_dists = torch.sqrt(valid_dists_raw)
                else:
                    valid_dists = valid_dists_raw
                levels = torch.ceil(valid_dists / self.epsilon).to(torch.long)
                
                indices_cpu = valid_indices.cpu()
                levels_cpu = levels.cpu()
                
                local_c = indices_cpu[:, 0]
                global_c = local_c + (start_q + start_mb)
                
                chunk_center_ids.append(global_c)
                chunk_level_ids.append(levels_cpu)
                chunk_point_ids.append(indices_cpu[:, 1])

        if not chunk_center_ids:
            return (torch.empty(0, dtype=torch.long), 
                    torch.empty(0, dtype=torch.long), 
                    torch.empty(0, dtype=torch.long))

        return (torch.cat(chunk_center_ids), 
                torch.cat(chunk_level_ids), 
                torch.cat(chunk_point_ids))

    def run(self, P_red, P_blue):
        if self.profile_memory:
            self.memory_profile = {}

        P_all = torch.cat([P_red, P_blue], dim=0)
        workspace = self.kernel.prepare_workspace(P_all)

        self._prof_reset()
        red_idx, red_mask = self._sample_landmarks(P_red.shape[0], P_red.device)
        blue_idx, blue_mask = self._sample_landmarks(P_blue.shape[0], P_blue.device)
        self._prof_record("landmark_sampling")

        self._prof_reset()
        D_red = self._compute_voronoi_bounds(P_all, P_red[red_idx], workspace)
        D_blue = self._compute_voronoi_bounds(P_all, P_blue[blue_idx], workspace)
        self._prof_record("voronoi_bounds")

        self._prof_reset()
        b_c, b_l, b_p = self._build_cover_bulk(P_blue, blue_mask, workspace, D_blue)
        r_c, r_l, r_p = self._build_cover_bulk(P_red, red_mask, workspace, D_red)
        self._prof_record("build_cover")

        return (b_c, b_l, b_p), (r_c, r_l, r_p), red_mask.cpu(), blue_mask.cpu()
