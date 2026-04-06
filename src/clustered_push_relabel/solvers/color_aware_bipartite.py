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

        # ── Structure 5: d_max over balls, persistent; no global max_list storage ────
        num_buckets = K * L

        shell_counts_2d = r_counts_long.view(K, L)
        ball_sizes_2d = torch.cumsum(shell_counts_2d, dim=1)
        ball_sizes = ball_sizes_2d.reshape(num_buckets)

        self.d_max = torch.where(
            ball_sizes > 0,
            torch.zeros(num_buckets, device=self.device, dtype=torch.int32),
            torch.full((num_buckets,), -1, device=self.device, dtype=torch.int32)
        )

    def _iter_ball_chunks(self, bucket_ids, L, max_ball_entries):
        if bucket_ids.numel() == 0:
            return

        ball_st = self.shell_red_offsets[(bucket_ids // L) * L]
        ball_en = self.shell_red_offsets[(bucket_ids // L) * L + (bucket_ids % L) + 1]
        ball_ln = ball_en - ball_st
        cum_ball_ln = torch.cumsum(ball_ln, 0)

        chunk_start = 0
        num_buckets = bucket_ids.numel()
        while chunk_start < num_buckets:
            base_entries = int(cum_ball_ln[chunk_start - 1].item()) if chunk_start > 0 else 0
            limit_entries = base_entries + max_ball_entries
            chunk_end = int(
                torch.searchsorted(
                    cum_ball_ln,
                    torch.tensor(limit_entries, device=self.device, dtype=torch.long),
                    right=True,
                ).item()
            )
            if chunk_end <= chunk_start:
                chunk_end = chunk_start + 1

            yield (
                chunk_start,
                chunk_end,
                bucket_ids[chunk_start:chunk_end],
                ball_st[chunk_start:chunk_end],
                ball_ln[chunk_start:chunk_end],
            )
            chunk_start = chunk_end

    def _expand_ball_chunk(self, chunk_st, chunk_ln, arange_attr):
        total_chunk_entries = int(chunk_ln.sum().item())
        if total_chunk_entries == 0:
            return (
                torch.empty(0, device=self.device, dtype=torch.long),
                torch.empty(0, device=self.device, dtype=torch.long),
            )

        cum_chunk = torch.cumsum(chunk_ln, 0)
        seg_chunk = cum_chunk - chunk_ln
        gr_chunk = _ensure_long_arange(self, arange_attr, total_chunk_entries, self.device)
        rep_st_chunk = torch.repeat_interleave(chunk_st, chunk_ln)
        off_chunk = gr_chunk - torch.repeat_interleave(seg_chunk, chunk_ln)
        sh_idx = rep_st_chunk + off_chunk

        exp_a_chunk = self.shell_red_indices[sh_idx]
        local_ball_idx = torch.repeat_interleave(
            torch.arange(chunk_ln.numel(), device=self.device, dtype=torch.long),
            chunk_ln,
        )
        return exp_a_chunk, local_ball_idx

    def _compute_bucket_max_counts(self, bucket_ids, L, max_ball_entries=4_000_000):
        counts = torch.zeros(bucket_ids.numel(), device=self.device, dtype=torch.long)
        if bucket_ids.numel() == 0:
            return counts

        for chunk_start, chunk_end, chunk_bkts, chunk_st, chunk_ln in self._iter_ball_chunks(
            bucket_ids, L, max_ball_entries
        ):
            exp_a_chunk, local_ball_idx = self._expand_ball_chunk(chunk_st, chunk_ln, '_tmp_ball_arange')
            if exp_a_chunk.numel() == 0:
                continue

            exp_ya_chunk = self.yA[exp_a_chunk].long()
            dmax_chunk = self.d_max[chunk_bkts].long()
            in_ml = exp_ya_chunk == dmax_chunk[local_ball_idx]

            cnt_chunk = torch.zeros(chunk_bkts.numel(), device=self.device, dtype=torch.long)
            if in_ml.any():
                cnt_chunk.scatter_add_(
                    0,
                    local_ball_idx[in_ml],
                    torch.ones(int(in_ml.sum().item()), device=self.device, dtype=torch.long),
                )
            counts[chunk_start:chunk_end] = cnt_chunk

        return counts

    def _build_temp_max_list(self, bucket_ids, L, max_ball_entries=4_000_000):
        counts = torch.zeros(bucket_ids.numel(), device=self.device, dtype=torch.long)
        values_parts = []

        if bucket_ids.numel() == 0:
            offsets = torch.zeros(1, device=self.device, dtype=torch.long)
            values = torch.empty(0, device=self.device, dtype=torch.long)
            return counts, offsets, values

        for chunk_start, chunk_end, chunk_bkts, chunk_st, chunk_ln in self._iter_ball_chunks(
            bucket_ids, L, max_ball_entries
        ):
            exp_a_chunk, local_ball_idx = self._expand_ball_chunk(chunk_st, chunk_ln, '_tmp_ball_arange')
            if exp_a_chunk.numel() == 0:
                continue

            exp_ya_chunk = self.yA[exp_a_chunk].long()
            dmax_chunk = self.d_max[chunk_bkts].long()
            in_ml = exp_ya_chunk == dmax_chunk[local_ball_idx]

            if in_ml.any():
                ml_local = local_ball_idx[in_ml]
                ml_a = exp_a_chunk[in_ml]
                counts[chunk_start:chunk_end] = torch.bincount(
                    ml_local, minlength=chunk_bkts.numel()
                ).to(device=self.device, dtype=torch.long)
                values_parts.append(ml_a)
            else:
                counts[chunk_start:chunk_end] = 0

        offsets = torch.cat([
            torch.zeros(1, device=self.device, dtype=torch.long),
            torch.cumsum(counts, 0),
        ])
        if values_parts:
            values = torch.cat(values_parts)
        else:
            values = torch.empty(0, device=self.device, dtype=torch.long)
        return counts, offsets, values

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

            # ── STEP 1: Find zero-slack candidate buckets (batched over free B) ─────
            push_batch_size = max(self.batch_size * 8, 2048)
            cand_b_parts = []
            cand_bkt_parts = []
            cand_kb_parts = []

            for batch_start in range(0, num_free, push_batch_size):
                chunk = B_free[batch_start: batch_start + push_batch_size]
                starts_b = self.inv_b_offsets[chunk]
                ends_b = self.inv_b_offsets[chunk + 1]
                lengths_b = ends_b - starts_b
                total_inv = int(lengths_b.sum().item())

                if total_inv == 0:
                    continue

                cum_b = torch.cumsum(lengths_b, 0)
                seg_b = cum_b - lengths_b
                g_range = _ensure_long_arange(self, '_inv_arange', total_inv, device)
                rep_st = torch.repeat_interleave(starts_b, lengths_b)
                off_b = g_range - torch.repeat_interleave(seg_b, lengths_b)
                inv_idx = rep_st + off_b

                active_b = torch.repeat_interleave(chunk, lengths_b)
                active_bkt = self.inv_b_bucket_ids[inv_idx]
                active_kb = self.inv_b_levels[inv_idx]

                target = 2 * active_kb - self.yB[active_b].long()
                dmax_vals = self.d_max[active_bkt].long()
                is_cand = (dmax_vals == target) & (dmax_vals >= 0)

                if is_cand.any():
                    cand_b_parts.append(active_b[is_cand])
                    cand_bkt_parts.append(active_bkt[is_cand])
                    cand_kb_parts.append(active_kb[is_cand])

            if not cand_b_parts:
                self.yB[B_free] += 1
                _print_dual_debug(
                    iteration,
                    num_free,
                    B_free,
                    torch.empty(0, device=device, dtype=torch.long),
                )
                continue

            cand_b = torch.cat(cand_b_parts)
            cand_bkt = torch.cat(cand_bkt_parts)
            cand_kb = torch.cat(cand_kb_parts)

            # ── STEP 2: Weighted bucket selection — Gumbel-max trick ──────
            cand_unique_bkt, cand_inv = torch.unique(cand_bkt, sorted=True, return_inverse=True)
            cand_counts = self._compute_bucket_max_counts(cand_unique_bkt, L)
            ml_counts_f = cand_counts[cand_inv].float()
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
            chosen_unique_bkt, chosen_inv = torch.unique(
                chosen_bkt, sorted=True, return_inverse=True
            )
            ml_counts_raw, ml_offsets, ml_values = self._build_temp_max_list(chosen_unique_bkt, L)
            ml_lens_raw = ml_counts_raw[chosen_inv]
            has_entries = ml_lens_raw > 0
            if not has_entries.all():
                chosen_bkt = chosen_bkt[has_entries]
                b_with_cand = b_with_cand[has_entries]
                chosen_inv = chosen_inv[has_entries]
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

            ml_starts = ml_offsets[chosen_inv]
            ml_lens = ml_lens_raw.clamp(min=1)
            rand_idx = (torch.rand(num_b_cand, device=device) * ml_lens.float()).long()
            rand_idx = torch.minimum(rand_idx.clamp_min(0), ml_lens - 1)
            proposal_a = ml_values[ml_starts + rand_idx]
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
                            max_s7_ball_entries = 4_000_000
                            cum_ball_ln = torch.cumsum(ball_ln, 0)
                            num_aff = aff_bkts.numel()
                            min_long = torch.iinfo(torch.long).min
                            new_dm = torch.full((num_aff,), min_long, device=device, dtype=torch.long)

                            # Pass 1: recompute d_max in bounded chunks of affected balls.
                            chunk_start = 0
                            while chunk_start < num_aff:
                                base_entries = int(cum_ball_ln[chunk_start - 1].item()) if chunk_start > 0 else 0
                                limit_entries = base_entries + max_s7_ball_entries
                                chunk_end = int(
                                    torch.searchsorted(
                                        cum_ball_ln,
                                        torch.tensor(limit_entries, device=device, dtype=torch.long),
                                        right=True,
                                    ).item()
                                )
                                if chunk_end <= chunk_start:
                                    chunk_end = chunk_start + 1

                                chunk_st = ball_st[chunk_start:chunk_end]
                                chunk_ln = ball_ln[chunk_start:chunk_end]
                                total_chunk_entries = int(chunk_ln.sum().item())

                                if total_chunk_entries > 0:
                                    cum_chunk = torch.cumsum(chunk_ln, 0)
                                    seg_chunk = cum_chunk - chunk_ln
                                    gr_chunk = _ensure_long_arange(
                                        self, '_s7_ba', total_chunk_entries, device
                                    )
                                    rep_st_chunk = torch.repeat_interleave(chunk_st, chunk_ln)
                                    off_chunk = gr_chunk - torch.repeat_interleave(seg_chunk, chunk_ln)
                                    sh_idx = rep_st_chunk + off_chunk

                                    exp_a_chunk = self.shell_red_indices[sh_idx]
                                    exp_ya_chunk = self.yA[exp_a_chunk].long()
                                    local_ball_idx = torch.repeat_interleave(
                                        torch.arange(
                                            chunk_end - chunk_start,
                                            device=device,
                                            dtype=torch.long,
                                        ),
                                        chunk_ln,
                                    )
                                    dm_chunk = torch.full(
                                        (chunk_end - chunk_start,),
                                        min_long,
                                        device=device,
                                        dtype=torch.long,
                                    )
                                    dm_chunk.scatter_reduce_(
                                        0,
                                        local_ball_idx,
                                        exp_ya_chunk,
                                        reduce="amax",
                                        include_self=True,
                                    )
                                    new_dm[chunk_start:chunk_end] = dm_chunk

                                chunk_start = chunk_end

                            self.d_max[aff_bkts] = new_dm.to(torch.int32)

            B_free = F_B_new

        self.cleanup_remaining_points()

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.MB == -1).squeeze(1)
        free_r = torch.nonzero(self.MA == -1).squeeze(1)
        count = min(free_b.numel(), free_r.numel())
        if count > 0:
            self.MB[free_b[:count]] = free_r[:count].to(self.MB.dtype)
            self.MA[free_r[:count]] = free_b[:count].to(self.MA.dtype)
