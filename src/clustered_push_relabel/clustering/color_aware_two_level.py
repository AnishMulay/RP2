import math
import torch
from ..utils.distance import TiledEuclideanKernel, TiledManhattanKernel


class ColorAwareClustering:
    def __init__(self, epsilon, batch_size=None, metric='L2', profile_memory=False):
        self.epsilon = epsilon
        self.metric = metric
        if batch_size is None:
            batch_size = 256 if metric == 'L1' else 2048
        self.batch_size = batch_size
        self.profile_memory = bool(profile_memory)
        self.memory_profile = {}

    # ── Opt-in stage-level peak-memory profiling (default OFF, zero overhead
    # when disabled). No-op on CPU. ─────────────────────────────────────────
    def _prof_reset(self):
        if self.profile_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _prof_record(self, label):
        if self.profile_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.memory_profile[label] = torch.cuda.max_memory_allocated() / 1024 ** 3

    def run(self, P_red_norm, P_blue_norm):
        if self.profile_memory:
            self.memory_profile = {}
        if self.metric == 'L1':
            kernel = TiledManhattanKernel(chunk_size=self.batch_size)
        else:
            kernel = TiledEuclideanKernel(chunk_size=self.batch_size)

        def get_dists(query, ws_all):
            if self.metric == 'L1':
                return kernel.compute_dist_tile(query, ws_all)
            return torch.sqrt(kernel.compute_dist_tile(query, ws_all))

        device = P_red_norm.device
        N = P_red_norm.shape[0]
        n_total = 2 * N
        P_all = torch.cat([P_red_norm, P_blue_norm], dim=0)
        ws_all = kernel.prepare_workspace(P_all)

        self._prof_reset()
        p_sample = 1.0 / math.sqrt(n_total)
        sampled_mask = torch.bernoulli(torch.full((n_total,), p_sample, device=device)).bool()
        sampled_red_mask = sampled_mask[:N]
        sampled_blue_mask = sampled_mask[N:]
        self._prof_record("landmark_sampling")

        self._prof_reset()
        sampled_red_pts = P_red_norm[sampled_red_mask]
        sampled_blue_pts = P_blue_norm[sampled_blue_mask]

        if sampled_red_pts.numel() > 0:
            d_min_red = get_dists(sampled_red_pts, ws_all).min(dim=1).values
        else:
            d_min_red = torch.full((n_total,), float('inf'), device=device)

        if sampled_blue_pts.numel() > 0:
            d_min_blue = get_dists(sampled_blue_pts, ws_all).min(dim=1).values
        else:
            d_min_blue = torch.full((n_total,), float('inf'), device=device)
        self._prof_record("landmark_assignment_d_min")

        self._prof_reset()
        red_c_list, red_l_list, red_p_list = [], [], []
        for batch_start in range(0, N, self.batch_size):
            batch_end = min(batch_start + self.batch_size, N)
            c_ids = torch.arange(batch_start, batch_end, device=device, dtype=torch.long)
            c_pts = P_red_norm[c_ids]
            dists = get_dists(c_pts, ws_all)

            is_sampled = sampled_red_mask[c_ids]
            non_sampled_member = dists < d_min_red.unsqueeze(1)
            sampled_member = is_sampled.unsqueeze(0).expand(n_total, -1)
            member = sampled_member | non_sampled_member

            levels = (dists / self.epsilon).long()

            p_idx, c_local = torch.where(member)
            if p_idx.numel() > 0:
                red_c_list.append(c_ids[c_local])
                red_l_list.append(levels[p_idx, c_local])
                red_p_list.append(p_idx)
            del dists, member, levels
        self._prof_record("build_cover_red")

        self._prof_reset()
        blue_c_list, blue_l_list, blue_p_list = [], [], []
        for batch_start in range(0, N, self.batch_size):
            batch_end = min(batch_start + self.batch_size, N)
            c_ids = torch.arange(batch_start, batch_end, device=device, dtype=torch.long)
            c_pts = P_blue_norm[c_ids]
            dists = get_dists(c_pts, ws_all)

            is_sampled = sampled_blue_mask[c_ids]
            non_sampled_member = dists < d_min_blue.unsqueeze(1)
            sampled_member = is_sampled.unsqueeze(0).expand(n_total, -1)
            member = sampled_member | non_sampled_member

            levels = (dists / self.epsilon).long()

            p_idx, c_local = torch.where(member)
            if p_idx.numel() > 0:
                blue_c_list.append(c_ids[c_local])
                blue_l_list.append(levels[p_idx, c_local])
                blue_p_list.append(p_idx)
            del dists, member, levels
        self._prof_record("build_cover_blue")

        def _cat_or_empty(lst, device):
            if not lst:
                return torch.empty(0, dtype=torch.long, device=device)
            return torch.cat(lst)

        r_c = _cat_or_empty(red_c_list, device)
        r_l = _cat_or_empty(red_l_list, device)
        r_p = _cat_or_empty(red_p_list, device)
        b_c = _cat_or_empty(blue_c_list, device)
        b_l = _cat_or_empty(blue_l_list, device)
        b_p = _cat_or_empty(blue_p_list, device)
        return (b_c, b_l, b_p), (r_c, r_l, r_p)
