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

        # ── Structures 5+6: ball_sizes, d_max, max_list (fully vectorized) ────
        num_buckets = K * L

        # ball_sizes[q*L+k] = A-points in shells 0..k of center q
        shell_counts_2d = r_counts_long.view(K, L)
        ball_sizes_2d = torch.cumsum(shell_counts_2d, dim=1)
        ball_sizes = ball_sizes_2d.reshape(num_buckets)
        self.ball_sizes = ball_sizes

        # d_max: 0 for non-empty balls, -1 for empty (y(a)=0 everywhere initially)
        self.d_max = torch.where(
            ball_sizes > 0,
            torch.zeros(num_buckets, device=self.device, dtype=torch.int32),
            torch.full((num_buckets,), -1, device=self.device, dtype=torch.int32)
        )

        # max_list CSR preallocated to ball capacity
        self.max_list_offsets = torch.cat([
            torch.zeros(1, device=self.device, dtype=torch.long),
            torch.cumsum(ball_sizes, 0)
        ])   # (K*L + 1,)
        total_ml_cap = int(ball_sizes.sum().item())
        self.max_list_values = torch.full(
            (total_ml_cap,), -1, device=self.device, dtype=torch.long
        )
        self.max_list_count = torch.zeros(num_buckets, device=self.device, dtype=torch.long)

        # Populate max_list_values incrementally in GPU-sized chunks.
        # This preserves the same final structure as the fully vectorized build,
        # but avoids materializing the full shell-to-ball expansion at once.
        num_shell = self.shell_red_indices.numel()
        if num_shell > 0:
            max_init_writes = 4_000_000
            shell_chunk_size = max(self.batch_size * 64, 16_384)
            shell_start = 0

            while shell_start < num_shell:
                shell_end = min(shell_start + shell_chunk_size, num_shell)

                while True:
                    s_bkt = self.shell_red_expand_bucket_ids[shell_start:shell_end]
                    s_center = s_bkt // L
                    s_ka = s_bkt % L
                    s_cstart = self.shell_red_offsets[s_center * L]
                    s_local = (
                        torch.arange(shell_start, shell_end, device=self.device, dtype=torch.long)
                        - s_cstart
                    )
                    s_maxlv = self.max_level_per_center[s_center]
                    num_balls_each = (s_maxlv - s_ka + 1).clamp(min=0)

                    total_writes = int(num_balls_each.sum().item())
                    chunk_len = shell_end - shell_start
                    if total_writes <= max_init_writes or chunk_len <= 1:
                        break

                    reduced_len = max(1, int(chunk_len * max_init_writes / max(total_writes, 1)))
                    shell_end = shell_start + reduced_len

                if total_writes > 0:
                    cum_nb = torch.cumsum(num_balls_each, 0)
                    seg_nb = cum_nb - num_balls_each
                    gr = _ensure_long_arange(self, '_ml_init_arange', total_writes, self.device)
                    ball_off = gr - torch.repeat_interleave(seg_nb, num_balls_each)

                    exp_center = torch.repeat_interleave(s_center, num_balls_each)
                    exp_ka = torch.repeat_interleave(s_ka, num_balls_each)
                    exp_local = torch.repeat_interleave(s_local, num_balls_each)
                    exp_a_id = torch.repeat_interleave(
                        self.shell_red_indices[shell_start:shell_end], num_balls_each
                    )

                    target_bkt = exp_center * L + exp_ka + ball_off
                    write_pos = self.max_list_offsets[target_bkt] + exp_local
                    self.max_list_values[write_pos] = exp_a_id
                    self.max_list_count.scatter_add_(
                        0,
                        target_bkt,
                        torch.ones(total_writes, device=self.device, dtype=torch.long)
                    )

                shell_start = shell_end

    def solve(self):
        N = self.N
        K = self.num_active_centers
        L = self.max_level_global + 1
        device = self.device
        B_free = torch.arange(N, device=device, dtype=torch.long)
        iteration = 0

        def _print_dual_debug(iteration, free_before, free_after, updated_red):
            if iteration % 100 != 0:
                return
            if free_after.numel() > 0:
                yb_min = int(self.yB[free_after].min().item())
                yb_max = int(self.yB[free_after].max().item())
            else:
                yb_min = -1
                yb_max = -1
            if updated_red.numel() > 0:
                ya_min = int(self.yA[updated_red].min().item())
                ya_max = int(self.yA[updated_red].max().item())
            else:
                ya_min = -1
                ya_max = -1
            print(
                f"[Iter {iteration}] free={free_before} next_free={int(free_after.numel())} "
                f"yB_free_min={yb_min} yB_free_max={yb_max} "
                f"yA_updates={int(updated_red.numel())} yA_new_min={ya_min} yA_new_max={ya_max}",
                flush=True,
            )

        while True:
            num_free = B_free.numel()
            if num_free <= self.epsilon * N:
                break
            if iteration > 50000:
                break
            iteration += 1

            # ── STEP 1: Find zero-slack candidate buckets ─────────────────
            starts_b = self.inv_b_offsets[B_free]
            ends_b = self.inv_b_offsets[B_free + 1]
            lengths_b = ends_b - starts_b
            total_inv = int(lengths_b.sum().item())

            if total_inv == 0:
                self.yB[B_free] += 1
                _print_dual_debug(
                    iteration,
                    num_free,
                    B_free,
                    torch.empty(0, device=device, dtype=torch.long),
                )
                continue

            cum_b = torch.cumsum(lengths_b, 0)
            seg_b = cum_b - lengths_b
            g_range = _ensure_long_arange(self, '_inv_arange', total_inv, device)
            rep_st = torch.repeat_interleave(starts_b, lengths_b)
            off_b = g_range - torch.repeat_interleave(seg_b, lengths_b)
            inv_idx = rep_st + off_b

            active_b = torch.repeat_interleave(B_free, lengths_b)
            active_bkt = self.inv_b_bucket_ids[inv_idx]
            active_kb = self.inv_b_levels[inv_idx]

            target = 2 * active_kb - self.yB[active_b].long()
            dmax_vals = self.d_max[active_bkt].long()
            is_cand = (dmax_vals == target) & (dmax_vals >= 0)

            if not is_cand.any():
                self.yB[B_free] += 1
                _print_dual_debug(
                    iteration,
                    num_free,
                    B_free,
                    torch.empty(0, device=device, dtype=torch.long),
                )
                continue

            cand_b = active_b[is_cand]
            cand_bkt = active_bkt[is_cand]
            cand_kb = active_kb[is_cand]

            # ── STEP 2: Weighted bucket selection — Gumbel-max trick ──────
            ml_counts_f = self.max_list_count[cand_bkt].float()
            has_weight = ml_counts_f > 0
            if not has_weight.any():
                self.yB[B_free] += 1
                _print_dual_debug(
                    iteration,
                    num_free,
                    B_free,
                    torch.empty(0, device=device, dtype=torch.long),
                )
                continue
            if not has_weight.all():
                cand_b = cand_b[has_weight]
                cand_bkt = cand_bkt[has_weight]
                cand_kb = cand_kb[has_weight]
                ml_counts_f = ml_counts_f[has_weight]

            gumbel = -torch.log(
                -torch.log(torch.rand(cand_bkt.numel(), device=device).clamp(min=1e-10))
                + 1e-10
            )
            scores = torch.log(ml_counts_f) + gumbel
            cand_edge_idx = _ensure_long_arange(self, '_cand_edge_arange', cand_bkt.numel(), device)

            best_per_b = torch.full((N,), float('-inf'), device=device)
            best_per_b.scatter_reduce_(0, cand_b, scores, reduce="amax", include_self=True)
            is_best = scores == best_per_b[cand_b]
            best_edge_per_b = torch.full((N,), cand_bkt.numel(), device=device, dtype=torch.long)
            best_edge_per_b.scatter_reduce_(
                0,
                cand_b[is_best],
                cand_edge_idx[is_best],
                reduce="amin",
                include_self=True,
            )
            is_chosen = cand_edge_idx == best_edge_per_b[cand_b]

            b_with_cand = cand_b[is_chosen]
            chosen_bkt = cand_bkt[is_chosen]
            num_b_cand = b_with_cand.numel()

            if num_b_cand == 0:
                self.yB[B_free] += 1
                _print_dual_debug(
                    iteration,
                    num_free,
                    B_free,
                    torch.empty(0, device=device, dtype=torch.long),
                )
                continue

            # ── STEP 3: Proposal — each b draws one a from max_list ───────
            ml_lens_raw = self.max_list_count[chosen_bkt]
            has_entries = ml_lens_raw > 0
            if not has_entries.all():
                chosen_bkt = chosen_bkt[has_entries]
                b_with_cand = b_with_cand[has_entries]
                ml_lens_raw = ml_lens_raw[has_entries]
                num_b_cand = b_with_cand.numel()
                if num_b_cand == 0:
                    self.yB[B_free] += 1
                    _print_dual_debug(
                        iteration,
                        num_free,
                        B_free,
                        torch.empty(0, device=device, dtype=torch.long),
                    )
                    continue
            ml_starts = self.max_list_offsets[chosen_bkt]
            ml_lens = ml_lens_raw.clamp(min=1)
            rand_idx = (torch.rand(num_b_cand, device=device) * ml_lens.float()).long()
            rand_idx = torch.minimum(rand_idx.clamp_min(0), ml_lens - 1)
            proposal_a = self.max_list_values[ml_starts + rand_idx]
            proposal_b = b_with_cand
            # Guard: filter any invalid proposals (should not happen after Change 1+2,
            # but kept as a safety net)
            valid_prop = (proposal_a >= 0) & (proposal_a < self.N)
            if not valid_prop.all():
                proposal_a = proposal_a[valid_prop]
                proposal_b = proposal_b[valid_prop]
                num_b_cand = proposal_a.numel()
            if num_b_cand == 0:
                self.yB[B_free] += 1
                _print_dual_debug(
                    iteration,
                    num_free,
                    B_free,
                    torch.empty(0, device=device, dtype=torch.long),
                )
                continue

            # ── STEP 4: Conflict resolution — each a accepts one proposal ─
            rand_prio = torch.rand(num_b_cand, device=device)
            prop_edge_idx = _ensure_long_arange(self, '_prop_edge_arange', num_b_cand, device)
            min_prio = torch.full((N,), float('inf'), device=device)
            min_prio.scatter_reduce_(0, proposal_a, rand_prio, reduce="amin", include_self=True)
            is_min = rand_prio == min_prio[proposal_a]
            accepted_edge_per_a = torch.full((N,), num_b_cand, device=device, dtype=torch.long)
            accepted_edge_per_a.scatter_reduce_(
                0,
                proposal_a[is_min],
                prop_edge_idx[is_min],
                reduce="amin",
                include_self=True,
            )
            accepted = prop_edge_idx == accepted_edge_per_a[proposal_a]
            r_new = proposal_a[accepted]
            b_new = proposal_b[accepted]

            # ── STEP 5: Matching update + F_B update ──────────────────────
            if r_new.numel() > 0:
                was_matched = self.MA[r_new] != -1
                evicted_b = self.MA[r_new[was_matched]].to(torch.long).clone()
                if evicted_b.numel() > 0:
                    self.MB[evicted_b] = -1
                self.MA[r_new] = b_new.to(self.MA.dtype)
                self.MB[b_new] = r_new.to(self.MB.dtype)

                keep = _ensure_bool_buffer(self, '_keep_free', num_free, device)
                keep.fill_(True)
                keep[torch.searchsorted(B_free, b_new)] = False
                still_free = B_free[keep]
                if evicted_b.numel() > 0:
                    F_B_new, _ = torch.sort(torch.cat([still_free, evicted_b]))
                else:
                    F_B_new = still_free
            else:
                F_B_new = B_free

            # ── STEP 6: Dual update ───────────────────────────────────────
            self.yB[F_B_new] += 1
            if r_new.numel() > 0:
                self.yA[r_new] -= 1
            _print_dual_debug(iteration, num_free, F_B_new, r_new)

            # ── STEP 7: Incremental pre-processing update (vectorized) ────
            if r_new.numel() > 0:
                inv_st = self.inv_a_offsets[r_new]
                inv_en = self.inv_a_offsets[r_new + 1]
                inv_ln = inv_en - inv_st
                total_ia = int(inv_ln.sum().item())

                if total_ia > 0:
                    cum_ia = torch.cumsum(inv_ln, 0)
                    seg_ia = cum_ia - inv_ln
                    gr_ia = _ensure_long_arange(self, '_s7_ia', total_ia, device)
                    rep_ia = torch.repeat_interleave(inv_st, inv_ln)
                    off_ia = gr_ia - torch.repeat_interleave(seg_ia, inv_ln)
                    idx_ia = rep_ia + off_ia

                    rep_a = torch.repeat_interleave(r_new, inv_ln)
                    bkt_sh = self.inv_a_bucket_ids[idx_ia]
                    cq = bkt_sh // L
                    ka = bkt_sh % L

                    maxlv = self.max_level_per_center[cq]
                    nballs = (maxlv - ka + 1).clamp(min=0)
                    total_bup = int(nballs.sum().item())

                    if total_bup > 0:
                        cum_nb = torch.cumsum(nballs, 0)
                        seg_nb = cum_nb - nballs
                        gr_nb = _ensure_long_arange(self, '_s7_nb', total_bup, device)
                        boff = gr_nb - torch.repeat_interleave(seg_nb, nballs)
                        exp_cq = torch.repeat_interleave(cq, nballs)
                        exp_ka = torch.repeat_interleave(ka, nballs)
                        aff_bkts = (exp_cq * L + exp_ka + boff).unique()

                        aff_c = aff_bkts // L
                        aff_k = aff_bkts % L
                        ball_st = self.shell_red_offsets[aff_c * L]
                        ball_en = self.shell_red_offsets[aff_c * L + aff_k + 1]
                        ball_ln = ball_en - ball_st
                        total_ba = int(ball_ln.sum().item())

                        if total_ba > 0:
                            cum_ba = torch.cumsum(ball_ln, 0)
                            seg_ba = cum_ba - ball_ln
                            gr_ba = _ensure_long_arange(self, '_s7_ba', total_ba, device)
                            rep_sba = torch.repeat_interleave(ball_st, ball_ln)
                            off_ba = gr_ba - torch.repeat_interleave(seg_ba, ball_ln)
                            sh_idx = rep_sba + off_ba

                            exp_a_ba = self.shell_red_indices[sh_idx]
                            exp_bk_ba = torch.repeat_interleave(aff_bkts, ball_ln)
                            exp_ya_ba = self.yA[exp_a_ba].long()

                            loc_idx = torch.searchsorted(aff_bkts, exp_bk_ba)

                            # New d_max per affected ball
                            new_dm = torch.zeros(aff_bkts.numel(), device=device, dtype=torch.long)
                            new_dm.scatter_reduce_(0, loc_idx, exp_ya_ba,
                                                   reduce="amax", include_self=True)
                            self.d_max[aff_bkts] = new_dm.to(torch.int32)

                            # New max_list members
                            in_ml = exp_ya_ba == new_dm[loc_idx]
                            ml_a = exp_a_ba[in_ml]
                            ml_bkt = exp_bk_ba[in_ml]

                            perm_ml = torch.argsort(ml_bkt)
                            ml_bkt_s = ml_bkt[perm_ml]
                            ml_a_s = ml_a[perm_ml]

                            loc_ml = torch.searchsorted(aff_bkts, ml_bkt_s)
                            cnt_new = torch.zeros(aff_bkts.numel(), device=device,
                                                  dtype=torch.long)
                            cnt_new.scatter_add_(
                                0, loc_ml,
                                torch.ones(ml_bkt_s.numel(), device=device, dtype=torch.long)
                            )
                            ml_off_l = torch.cat([
                                torch.zeros(1, device=device, dtype=torch.long),
                                torch.cumsum(cnt_new, 0)
                            ])
                            rank_ml = (torch.arange(ml_bkt_s.numel(), device=device,
                                                    dtype=torch.long)
                                       - ml_off_l[loc_ml])
                            wpos = self.max_list_offsets[ml_bkt_s] + rank_ml
                            self.max_list_values[wpos] = ml_a_s
                            self.max_list_count[aff_bkts] = cnt_new

            B_free = F_B_new

        self.cleanup_remaining_points()

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.MB == -1).squeeze(1)
        free_r = torch.nonzero(self.MA == -1).squeeze(1)
        count = min(free_b.numel(), free_r.numel())
        if count > 0:
            self.MB[free_b[:count]] = free_r[:count].to(self.MB.dtype)
            self.MA[free_r[:count]] = free_b[:count].to(self.MA.dtype)
