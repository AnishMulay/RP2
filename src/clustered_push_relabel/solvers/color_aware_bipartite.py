import torch
import math
import gc
from ..clustering.color_aware_two_level import ColorAwareClustering


def _ensure_long_arange(owner, attr_name, size, device):
    buf = getattr(owner, attr_name, None)
    if buf is None or buf.numel() < size:
        buf = torch.arange(size, device=device, dtype=torch.long)
        setattr(owner, attr_name, buf)
    return buf[:size]


def _ensure_zero_long_buffer(owner, attr_name, size, device):
    buf = getattr(owner, attr_name, None)
    if buf is None or buf.numel() < size:
        buf = torch.empty(size, device=device, dtype=torch.long)
        setattr(owner, attr_name, buf)
    buf = buf[:size]
    buf.zero_()
    return buf


def _ensure_bool_buffer(owner, attr_name, size, device):
    buf = getattr(owner, attr_name, None)
    if buf is None or buf.numel() < size:
        buf = torch.empty(size, device=device, dtype=torch.bool)
        setattr(owner, attr_name, buf)
    return buf[:size]


class ColorAwareTwoLevelSolver:
    def __init__(self, P_red, P_blue, epsilon, batch_size=None, metric='L2', verbose=False):
        self.device = P_red.device
        self.N = P_red.shape[0]
        self.epsilon = epsilon
        self.metric = metric
        self.verbose = verbose
        self.batch_size = 1024 if batch_size is None else batch_size

        P_all = torch.cat([P_red, P_blue], dim=0)
        if metric == 'L1':
            delta = (P_all.max(dim=0).values - P_all.min(dim=0).values).sum()
        else:
            delta = ((P_all.max(dim=0).values - P_all.min(dim=0).values).pow(2).sum()).sqrt()
        delta = delta.clamp(min=1e-8)
        self.delta = delta
        P_red_norm = P_red.float() / delta
        P_blue_norm = P_blue.float() / delta

        cluster_engine = ColorAwareClustering(epsilon, batch_size=self.batch_size, metric=metric)
        blue_coo, red_coo = cluster_engine.run(P_red_norm, P_blue_norm)

        self._build_csr_and_inv(blue_coo, red_coo)

        del blue_coo, red_coo, cluster_engine, P_red_norm, P_blue_norm, P_all
        gc.collect()
        self.yA = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.yB = torch.full((self.N,), 1, device=self.device, dtype=torch.int32)
        self.MA = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)
        self.MB = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)

    def _build_csr_and_inv(self, blue_coo, red_coo):
        b_c, b_l, b_p = blue_coo
        r_c, r_l, r_p = red_coo
        N = self.N

        b_c_shifted = b_c + N
        all_centers = torch.cat([b_c_shifted, r_c])
        all_levels = torch.cat([b_l, r_l])
        all_points = torch.cat([b_p, r_p])
        is_red_point = all_points < N

        centers_with_red = torch.unique(all_centers[is_red_point])
        centers_with_blue = torch.unique(all_centers[~is_red_point])
        if centers_with_red.numel() == 0 or centers_with_blue.numel() == 0:
            raise ValueError("No valid clusters (empty intersection).")
        valid_centers = centers_with_red[torch.isin(centers_with_red, centers_with_blue)]
        if valid_centers.numel() == 0:
            raise ValueError("No valid clusters (empty intersection).")

        mask_valid = torch.isin(all_centers, valid_centers)
        all_centers = all_centers[mask_valid]
        all_levels = all_levels[mask_valid]
        all_points = all_points[mask_valid]
        is_red_point = is_red_point[mask_valid]

        center_map = torch.searchsorted(valid_centers, all_centers)
        self.num_active_centers = int(valid_centers.numel())

        max_level_global = int(all_levels.max().item())
        self.max_level_global = max_level_global
        L = max_level_global + 1
        K = self.num_active_centers

        red_mask = is_red_point
        red_centers = center_map[red_mask]
        red_points = all_points[red_mask]
        red_levels = all_levels[red_mask].long()

        red_bucket_ids = red_centers * L + red_levels

        perm_r = torch.argsort(red_bucket_ids)
        self.shell_red_indices = red_points[perm_r].to(device=self.device, dtype=torch.long)
        self.shell_red_levels = red_levels[perm_r].to(device=self.device, dtype=torch.long)
        sorted_red_buckets = red_bucket_ids[perm_r]

        r_counts = torch.bincount(sorted_red_buckets, minlength=K * L)
        r_counts_long = r_counts.to(device=self.device, dtype=torch.long)
        self.shell_red_offsets = torch.cat([
            torch.zeros(1, device=self.device, dtype=torch.long),
            torch.cumsum(r_counts_long, 0)
        ])

        self.shell_red_expand_bucket_ids = torch.repeat_interleave(
            torch.arange(K * L, device=self.device, dtype=torch.long), r_counts_long
        )

        center_has_red = red_centers
        level_per_entry = red_levels
        max_level_per_center = torch.zeros(K, device=self.device, dtype=torch.long)
        max_level_per_center.scatter_reduce_(
            0, center_has_red.to(self.device), level_per_entry.to(self.device),
            reduce="amax", include_self=True
        )
        self.max_level_per_center = max_level_per_center

        blue_mask = ~is_red_point
        blue_centers = center_map[blue_mask]
        blue_points = all_points[blue_mask] - N
        blue_levels = all_levels[blue_mask].long()

        blue_bucket_ids = blue_centers * L + blue_levels

        perm_b = torch.argsort(blue_points)
        self.inv_b_bucket_ids = blue_bucket_ids[perm_b].to(device=self.device, dtype=torch.long)
        self.inv_b_levels = blue_levels[perm_b].to(device=self.device, dtype=torch.long)
        sorted_b_pts = blue_points[perm_b]

        b_counts = torch.bincount(sorted_b_pts, minlength=N)
        b_counts_long = b_counts.to(device=self.device, dtype=torch.long)
        self.inv_b_offsets = torch.cat([
            torch.zeros(1, device=self.device, dtype=torch.long),
            torch.cumsum(b_counts_long, 0)
        ])

        perm_a = torch.argsort(red_points)
        self.inv_a_bucket_ids = red_bucket_ids[perm_a].to(device=self.device, dtype=torch.long)
        self.inv_a_levels = red_levels[perm_a].to(device=self.device, dtype=torch.long)
        sorted_a_pts = red_points[perm_a]

        a_counts = torch.bincount(sorted_a_pts, minlength=N)
        a_counts_long = a_counts.to(device=self.device, dtype=torch.long)
        self.inv_a_offsets = torch.cat([
            torch.zeros(1, device=self.device, dtype=torch.long),
            torch.cumsum(a_counts_long, 0)
        ])

        num_buckets = K * L
        self.d_max = torch.full((num_buckets,), -1, device=self.device, dtype=torch.int32)

        nonempty_mask = r_counts_long > 0
        self.d_max[nonempty_mask] = 0

        for center in range(K):
            for lv in range(1, L):
                bid = center * L + lv
                bid_prev = center * L + (lv - 1)
                if self.d_max[bid_prev].item() >= 0 and self.d_max[bid].item() < 0:
                    self.d_max[bid] = self.d_max[bid_prev]

        ball_sizes = torch.zeros(num_buckets, device=self.device, dtype=torch.long)
        for center in range(K):
            cumulative = 0
            for lv in range(L):
                bid = center * L + lv
                cumulative += r_counts_long[bid].item()
                ball_sizes[bid] = cumulative

        self.ball_sizes = ball_sizes

        self.max_list_offsets = torch.cat([
            torch.zeros(1, device=self.device, dtype=torch.long),
            torch.cumsum(ball_sizes, 0)
        ])

        total_max_list_capacity = int(ball_sizes.sum().item())
        self.max_list_values = torch.full(
            (total_max_list_capacity,), -1, device=self.device, dtype=torch.long
        )
        self.max_list_count = torch.zeros(num_buckets, device=self.device, dtype=torch.long)

        for center in range(K):
            for lv in range(L):
                bid = center * L + lv
                start = int(self.shell_red_offsets[bid].item())
                ball_start_shell = center * L + 0
                ball_end_shell = center * L + lv
                ball_a_pts = self.shell_red_indices[
                    self.shell_red_offsets[center * L]:
                    self.shell_red_offsets[center * L + lv + 1]
                ]
                count = ball_a_pts.numel()
                if count > 0:
                    ml_start = int(self.max_list_offsets[bid].item())
                    self.max_list_values[ml_start: ml_start + count] = ball_a_pts
                    self.max_list_count[bid] = count

    def solve(self):
        N = self.N
        K = self.num_active_centers
        L = self.max_level_global + 1
        device = self.device
        B_free = torch.arange(N, device=device, dtype=torch.long)
        iteration = 0

        while True:
            num_free = B_free.numel()
            if num_free <= self.epsilon * N:
                break
            if iteration > 50000:
                break
            iteration += 1
            if iteration % 100 == 0:
                print(f"    [Iter {iteration}] Free B: {num_free}", flush=True)

            starts_b = self.inv_b_offsets[B_free]
            ends_b = self.inv_b_offsets[B_free + 1]
            lengths_b = ends_b - starts_b
            total_inv = int(lengths_b.sum().item())

            if total_inv == 0:
                self.yB[B_free] += 1
                continue

            cum_b = torch.cumsum(lengths_b, 0)
            seg_b = cum_b - lengths_b
            g_range = _ensure_long_arange(self, "_inv_arange", total_inv, device)
            rep_starts_b = torch.repeat_interleave(starts_b, lengths_b)
            offsets_b = g_range - torch.repeat_interleave(seg_b, lengths_b)
            inv_edge_idx = rep_starts_b + offsets_b

            active_b_ids = torch.repeat_interleave(B_free, lengths_b)
            active_bkt_ids = self.inv_b_bucket_ids[inv_edge_idx]
            active_kb = self.inv_b_levels[inv_edge_idx]

            target = 2 * active_kb - self.yB[active_b_ids].long()
            dmax_vals = self.d_max[active_bkt_ids].long()
            is_candidate = (dmax_vals == target) & (dmax_vals >= 0)

            if not is_candidate.any():
                self.yB[B_free] += 1
                continue

            cand_b = active_b_ids[is_candidate]
            cand_bkt = active_bkt_ids[is_candidate]
            cand_kb = active_kb[is_candidate]

            ml_counts = self.max_list_count[cand_bkt]

            perm_c = torch.argsort(cand_b)
            cand_b_s = cand_b[perm_c]
            cand_bkt_s = cand_bkt[perm_c]
            cand_kb_s = cand_kb[perm_c]
            ml_counts_s = ml_counts[perm_c]

            cand_counts_per_b = torch.bincount(cand_b_s, minlength=N)
            has_cand = cand_counts_per_b[B_free] > 0
            b_with_cand = B_free[has_cand]

            b_cand_offsets = torch.cat([
                torch.zeros(1, device=device, dtype=torch.long),
                torch.cumsum(cand_counts_per_b[b_with_cand], 0)
            ])

            num_b_cand = b_with_cand.numel()
            chosen_bkt = torch.empty(num_b_cand, device=device, dtype=torch.long)
            chosen_kb = torch.empty(num_b_cand, device=device, dtype=torch.long)

            for i in range(num_b_cand):
                s = int(b_cand_offsets[i].item())
                e = int(b_cand_offsets[i + 1].item())
                weights = ml_counts_s[s:e].float()
                total_w = weights.sum()
                if total_w <= 0:
                    chosen_bkt[i] = cand_bkt_s[s]
                    chosen_kb[i] = cand_kb_s[s]
                else:
                    probs = weights / total_w
                    idx = int(torch.multinomial(probs, 1).item())
                    chosen_bkt[i] = cand_bkt_s[s + idx]
                    chosen_kb[i] = cand_kb_s[s + idx]

            ml_starts = self.max_list_offsets[chosen_bkt]
            ml_lens = self.max_list_count[chosen_bkt]
            rand_idx = (torch.rand(num_b_cand, device=device) * ml_lens.float()).long()
            rand_idx = rand_idx.clamp(max=ml_lens - 1)
            proposal_a = self.max_list_values[ml_starts + rand_idx]

            num_props = num_b_cand
            rand_prio = torch.rand(num_props, device=device)

            proposal_b = b_with_cand

            min_prio_per_a = torch.full((N,), float('inf'), device=device)
            min_prio_per_a.scatter_reduce_(
                0, proposal_a, rand_prio, reduce="amin", include_self=True
            )

            accepted_mask = rand_prio == min_prio_per_a[proposal_a]

            r_new = proposal_a[accepted_mask]
            b_new = proposal_b[accepted_mask]

            if r_new.numel() > 0:
                was_matched = self.MA[r_new] != -1
                evicted_b = self.MA[r_new[was_matched]].to(torch.long).clone()

                if evicted_b.numel() > 0:
                    self.MB[evicted_b] = -1

                self.MA[r_new] = b_new.to(self.MA.dtype)
                self.MB[b_new] = r_new.to(self.MB.dtype)

                keep_mask = _ensure_bool_buffer(self, "_keep_free_mask", num_free, device)
                keep_mask.fill_(True)
                keep_mask[torch.searchsorted(B_free, b_new)] = False
                still_free = B_free[keep_mask]
                if evicted_b.numel() > 0:
                    F_B_new, _ = torch.sort(torch.cat([still_free, evicted_b]))
                else:
                    F_B_new = still_free
            else:
                F_B_new = B_free

            self.yB[F_B_new] += 1
            if r_new.numel() > 0:
                self.yA[r_new] += 1

            if r_new.numel() > 0:
                changed_a_list = r_new.cpu().tolist()
                for a in changed_a_list:
                    ya_new = int(self.yA[a].item())
                    ya_old = ya_new - 1
                    inv_start = int(self.inv_a_offsets[a].item())
                    inv_end = int(self.inv_a_offsets[a + 1].item())
                    for idx in range(inv_start, inv_end):
                        bkt_shell = int(self.inv_a_bucket_ids[idx].item())
                        center_q = bkt_shell // L
                        k_a = bkt_shell % L
                        max_lv = int(self.max_level_per_center[center_q].item())
                        for k in range(k_a, max_lv + 1):
                            bid = center_q * L + k
                            dm = int(self.d_max[bid].item())
                            if dm < 0:
                                continue
                            if ya_old == dm:
                                self.d_max[bid] = ya_new
                                ml_start = int(self.max_list_offsets[bid].item())
                                self.max_list_values[ml_start] = a
                                self.max_list_count[bid] = 1
                            elif ya_new == dm:
                                ml_start = int(self.max_list_offsets[bid].item())
                                cur_cnt = int(self.max_list_count[bid].item())
                                self.max_list_values[ml_start + cur_cnt] = a
                                self.max_list_count[bid] = cur_cnt + 1

            B_free = F_B_new

        self.cleanup_remaining_points()

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.MB == -1).squeeze(1)
        free_r = torch.nonzero(self.MA == -1).squeeze(1)
        count = min(free_b.numel(), free_r.numel())
        if count > 0:
            self.MB[free_b[:count]] = free_r[:count].to(self.MB.dtype)
            self.MA[free_r[:count]] = free_b[:count].to(self.MA.dtype)
