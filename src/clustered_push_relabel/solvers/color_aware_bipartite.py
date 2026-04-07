import torch
import math
import gc
import time
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
        torch.cuda.empty_cache()
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
        entry_bkt = sorted_red_buckets

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

        # ── Structure 5: d_max (per ball, initialized to 0 for non-empty balls) ────
        shell_counts_2d = r_counts_long.view(K, L)
        ball_sizes = torch.cumsum(shell_counts_2d, dim=1).reshape(K * L)
        self.d_max = torch.where(
            ball_sizes > 0,
            torch.zeros(K * L, device=self.device, dtype=torch.int32),
            torch.full((K * L,), -1, device=self.device, dtype=torch.int32),
        )
        del ball_sizes, shell_counts_2d

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

    def solve(self):
        N = self.N
        K = self.num_active_centers
        L = self.max_level_global + 1
        device = self.device
        B_free = torch.arange(N, device=device, dtype=torch.long)
        iteration = 0

        def _sync_if_cuda():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

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

        def _print_phase_timing(iteration, free_before, timings, status):
            if iteration % 100 != 0:
                return
            print(
                f"[Iter {iteration} timing] free={free_before} status={status} "
                f"s1={timings['s1']:.4f}s s2={timings['s2']:.4f}s s3={timings['s3']:.4f}s "
                f"s4={timings['s4']:.4f}s s5={timings['s5']:.4f}s s6={timings['s6']:.4f}s "
                f"s7={timings['s7']:.4f}s total={timings['total']:.4f}s",
                flush=True,
            )

        # Step 7 sub-timing accumulators (accumulated across all phases)
        _s7_t_inv    = 0.0   # INV collection: offsets lookup + repeat_interleave for cq/ka/nballs
        _s7_t_unique = 0.0   # torch.unique to compute aff_bkts
        _s7_t_expand = 0.0   # ball expansion: repeat_interleave for exp_a and b_local
        _s7_t_amax   = 0.0   # scatter_reduce amax for new d_max
        _s7_phases   = 0     # number of phases where r_new.numel() > 0

        while True:
            num_free = B_free.numel()
            if num_free <= self.epsilon * N:
                break
            if iteration > 50000:
                break
            iteration += 1
            timing_enabled = (iteration % 100 == 0)
            timings = {
                's1': 0.0, 's2': 0.0, 's3': 0.0, 's4': 0.0,
                's5': 0.0, 's6': 0.0, 's7': 0.0, 'total': 0.0,
            }
            if timing_enabled:
                _sync_if_cuda()
                iter_t0 = time.perf_counter()

            # ── STEP 1: Find zero-slack candidate buckets (batched over free B) ─────
            if timing_enabled:
                _sync_if_cuda()
                phase_t0 = time.perf_counter()
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
            if timing_enabled:
                _sync_if_cuda()
                timings['s1'] = time.perf_counter() - phase_t0

            if not cand_b_parts:
                self.yB[B_free] += 1
                _print_dual_debug(
                    iteration,
                    num_free,
                    B_free,
                    torch.empty(0, device=device, dtype=torch.long),
                )
                if timing_enabled:
                    _sync_if_cuda()
                    timings['total'] = time.perf_counter() - iter_t0
                    _print_phase_timing(iteration, num_free, timings, "no_candidates")
                continue

            cand_b = torch.cat(cand_b_parts)
            cand_bkt = torch.cat(cand_bkt_parts)
            cand_kb = torch.cat(cand_kb_parts)

            # ── STEP 2: Weighted bucket selection — Gumbel-max trick ──────
            if timing_enabled:
                _sync_if_cuda()
                phase_t0 = time.perf_counter()
            S2_MAX_ENTRIES = 2_000_000

            uniq_cand_bkt, inv_cand_bkt = torch.unique(cand_bkt, return_inverse=True)
            max_counts_unique = torch.zeros(
                uniq_cand_bkt.numel(), device=device, dtype=torch.long
            )

            s2_ball_st = self.shell_red_offsets[(uniq_cand_bkt // L) * L]
            s2_ball_en = self.shell_red_offsets[
                (uniq_cand_bkt // L) * L + (uniq_cand_bkt % L) + 1
            ]
            s2_ball_ln = s2_ball_en - s2_ball_st
            s2_cum_ln = torch.cumsum(s2_ball_ln, 0)

            s2_chunk_start = 0
            while s2_chunk_start < uniq_cand_bkt.numel():
                base_s2 = int(s2_cum_ln[s2_chunk_start - 1].item()) if s2_chunk_start > 0 else 0
                s2_chunk_end = int(
                    torch.searchsorted(
                        s2_cum_ln,
                        torch.tensor(base_s2 + S2_MAX_ENTRIES, device=device, dtype=torch.long),
                        right=True,
                    ).item()
                )
                if s2_chunk_end <= s2_chunk_start:
                    s2_chunk_end = s2_chunk_start + 1

                c_bkt = uniq_cand_bkt[s2_chunk_start:s2_chunk_end]
                c_bst = s2_ball_st[s2_chunk_start:s2_chunk_end]
                c_bln = s2_ball_ln[s2_chunk_start:s2_chunk_end]
                c_size = c_bkt.numel()
                c_total = int(c_bln.sum().item())

                if c_total > 0:
                    cum_c2 = torch.cumsum(c_bln, 0)
                    seg_c2 = cum_c2 - c_bln
                    gr_c2 = _ensure_long_arange(self, '_s2_exp', c_total, device)
                    sh_c2 = (
                        torch.repeat_interleave(c_bst, c_bln)
                        + gr_c2
                        - torch.repeat_interleave(seg_c2, c_bln)
                    )
                    exp_a_c = self.shell_red_indices[sh_c2]
                    bl_c = torch.repeat_interleave(
                        torch.arange(c_size, device=device, dtype=torch.long), c_bln
                    )

                    ya_c = self.yA[exp_a_c].long()
                    dm_c = self.d_max[c_bkt[bl_c]].long()
                    is_mx = ya_c == dm_c

                    max_counts_unique[s2_chunk_start:s2_chunk_end] = torch.bincount(
                        bl_c[is_mx], minlength=c_size
                    )

                    del exp_a_c, bl_c, ya_c, dm_c, is_mx
                    del cum_c2, seg_c2, gr_c2, sh_c2

                del c_bkt, c_bst, c_bln
                s2_chunk_start = s2_chunk_end

            max_counts_f = max_counts_unique[inv_cand_bkt].float()
            del uniq_cand_bkt, inv_cand_bkt, max_counts_unique
            del s2_ball_st, s2_ball_en, s2_ball_ln, s2_cum_ln

            has_weight = max_counts_f > 0
            if not has_weight.any():
                self.yB[B_free] += 1
                _print_dual_debug(
                    iteration,
                    num_free,
                    B_free,
                    torch.empty(0, device=device, dtype=torch.long),
                )
                if timing_enabled:
                    _sync_if_cuda()
                    timings['s2'] = time.perf_counter() - phase_t0
                    timings['total'] = time.perf_counter() - iter_t0
                    _print_phase_timing(iteration, num_free, timings, "no_weight")
                continue
            if not has_weight.all():
                cand_b = cand_b[has_weight]
                cand_bkt = cand_bkt[has_weight]
                cand_kb = cand_kb[has_weight]
                max_counts_f = max_counts_f[has_weight]

            gumbel = -torch.log(
                -torch.log(torch.rand(cand_bkt.numel(), device=device).clamp(min=1e-10))
                + 1e-10
            )
            scores = torch.log(max_counts_f) + gumbel
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
            if timing_enabled:
                _sync_if_cuda()
                timings['s2'] = time.perf_counter() - phase_t0

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
                if timing_enabled:
                    _sync_if_cuda()
                    timings['total'] = time.perf_counter() - iter_t0
                    _print_phase_timing(iteration, num_free, timings, "no_chosen")
                continue

            # ── STEP 3: Proposal — each b draws one a from its current max set ───────
            if timing_enabled:
                _sync_if_cuda()
                phase_t0 = time.perf_counter()
            S3_MAX_ENTRIES = 2_000_000

            proposal_a_parts = []
            proposal_b_parts = []

            # Compute ball boundaries for all chosen buckets
            s3_ball_st = self.shell_red_offsets[(chosen_bkt // L) * L]
            s3_ball_en = self.shell_red_offsets[(chosen_bkt // L) * L + (chosen_bkt % L) + 1]
            s3_ball_ln = s3_ball_en - s3_ball_st
            s3_cum_ln  = torch.cumsum(s3_ball_ln, 0)

            s3_chunk_start = 0
            while s3_chunk_start < num_b_cand:
                base_s3 = int(s3_cum_ln[s3_chunk_start - 1].item()) if s3_chunk_start > 0 else 0
                s3_chunk_end = int(
                    torch.searchsorted(
                        s3_cum_ln,
                        torch.tensor(base_s3 + S3_MAX_ENTRIES, device=device, dtype=torch.long),
                        right=True,
                    ).item()
                )
                if s3_chunk_end <= s3_chunk_start:
                    s3_chunk_end = s3_chunk_start + 1

                c_bkt    = chosen_bkt[s3_chunk_start:s3_chunk_end]
                c_bvals  = b_with_cand[s3_chunk_start:s3_chunk_end]
                c_bst    = s3_ball_st[s3_chunk_start:s3_chunk_end]
                c_bln    = s3_ball_ln[s3_chunk_start:s3_chunk_end]
                c_size   = c_bkt.numel()
                c_total  = int(c_bln.sum().item())

                if c_total > 0:
                    cum_c3  = torch.cumsum(c_bln, 0)
                    seg_c3  = cum_c3 - c_bln
                    gr_c3   = _ensure_long_arange(self, '_s3_exp', c_total, device)
                    sh_c3   = torch.repeat_interleave(c_bst, c_bln) \
                              + gr_c3 - torch.repeat_interleave(seg_c3, c_bln)
                    exp_a_c = self.shell_red_indices[sh_c3]
                    bl_c    = torch.repeat_interleave(
                        torch.arange(c_size, device=device, dtype=torch.long), c_bln
                    )

                    ya_c  = self.yA[exp_a_c].long()
                    dm_c  = self.d_max[c_bkt[bl_c]].long()
                    is_mx = ya_c == dm_c

                    max_a_c  = exp_a_c[is_mx]
                    max_bl_c = bl_c[is_mx]

                    if max_a_c.numel() > 0:
                        # Gumbel-max trick: uniform random sample per ball, fully parallel
                        gumbel_c = -torch.log(
                            -torch.log(torch.rand(max_a_c.numel(), device=device).clamp(min=1e-10))
                            + 1e-10
                        )
                        best_g = torch.full((c_size,), float('-inf'), device=device)
                        best_g.scatter_reduce_(0, max_bl_c, gumbel_c, reduce='amax', include_self=True)
                        is_win = gumbel_c == best_g[max_bl_c]

                        win_idx = torch.full((c_size,), max_a_c.numel(), device=device, dtype=torch.long)
                        eidx_c  = _ensure_long_arange(self, '_s3_eidx', max_a_c.numel(), device)
                        win_idx.scatter_reduce_(
                            0, max_bl_c[is_win], eidx_c[is_win], reduce='amin', include_self=True
                        )
                        valid = win_idx < max_a_c.numel()
                        if valid.any():
                            proposal_a_parts.append(max_a_c[win_idx[valid]])
                            proposal_b_parts.append(c_bvals[valid])

                    del exp_a_c, bl_c, ya_c, dm_c, is_mx, max_a_c, max_bl_c
                    del cum_c3, seg_c3, gr_c3, sh_c3

                del c_bkt, c_bvals, c_bst, c_bln
                s3_chunk_start = s3_chunk_end

            del s3_ball_st, s3_ball_en, s3_ball_ln, s3_cum_ln

            if not proposal_a_parts:
                self.yB[B_free] += 1
                _print_dual_debug(iteration, num_free, B_free,
                                  torch.empty(0, device=device, dtype=torch.long))
                if timing_enabled:
                    _sync_if_cuda()
                    timings['s3'] = time.perf_counter() - phase_t0
                    timings['total'] = time.perf_counter() - iter_t0
                    _print_phase_timing(iteration, num_free, timings, "no_proposals")
                continue

            proposal_a = torch.cat(proposal_a_parts)
            proposal_b = torch.cat(proposal_b_parts)
            num_b_cand = proposal_a.numel()

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
                if timing_enabled:
                    _sync_if_cuda()
                    timings['s3'] = time.perf_counter() - phase_t0
                    timings['total'] = time.perf_counter() - iter_t0
                    _print_phase_timing(iteration, num_free, timings, "no_valid_proposals")
                continue
            if timing_enabled:
                _sync_if_cuda()
                timings['s3'] = time.perf_counter() - phase_t0

            # ── STEP 4: Conflict resolution — each a accepts one proposal ─
            if timing_enabled:
                _sync_if_cuda()
                phase_t0 = time.perf_counter()
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
            if timing_enabled:
                _sync_if_cuda()
                timings['s4'] = time.perf_counter() - phase_t0

            # ── STEP 5: Matching update + F_B update ──────────────────────
            if timing_enabled:
                _sync_if_cuda()
                phase_t0 = time.perf_counter()
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
            if timing_enabled:
                _sync_if_cuda()
                timings['s5'] = time.perf_counter() - phase_t0

            # ── STEP 6: Dual update ───────────────────────────────────────
            if timing_enabled:
                _sync_if_cuda()
                phase_t0 = time.perf_counter()
            self.yB[F_B_new] += 1
            if r_new.numel() > 0:
                self.yA[r_new] -= 1
            if timing_enabled:
                _sync_if_cuda()
                timings['s6'] = time.perf_counter() - phase_t0
            _print_dual_debug(iteration, num_free, F_B_new, r_new)

            # ── STEP 7: Incremental pre-processing update (vectorized) ────
            if timing_enabled:
                _sync_if_cuda()
                phase_t0 = time.perf_counter()
            if r_new.numel() > 0:
                _s7_phases += 1
                S7_CHANGED_CHUNK = 128
                S7_MAX_BALL_ENTRIES = 2_000_000

                for r_chunk_start in range(0, r_new.numel(), S7_CHANGED_CHUNK):
                    r_chunk = r_new[r_chunk_start: r_chunk_start + S7_CHANGED_CHUNK]

                    _sync_if_cuda()
                    _s7_t0 = time.perf_counter()
                    inv_st = self.inv_a_offsets[r_chunk]
                    inv_en = self.inv_a_offsets[r_chunk + 1]
                    inv_ln = inv_en - inv_st
                    total_inv = int(inv_ln.sum().item())
                    if total_inv == 0:
                        _sync_if_cuda()
                        _s7_t_inv += time.perf_counter() - _s7_t0
                        continue

                    cum_inv = torch.cumsum(inv_ln, 0)
                    seg_inv = cum_inv - inv_ln
                    gr_inv = _ensure_long_arange(self, '_s7_ia', total_inv, device)
                    idx_ia = (
                        torch.repeat_interleave(inv_st, inv_ln)
                        + gr_inv
                        - torch.repeat_interleave(seg_inv, inv_ln)
                    )
                    bkt_sh = self.inv_a_bucket_ids[idx_ia]
                    cq = bkt_sh // L
                    ka = bkt_sh % L
                    maxlv = self.max_level_per_center[cq]

                    nballs = (maxlv - ka + 1).clamp(min=0)
                    total_bup = int(nballs.sum().item())
                    if total_bup == 0:
                        del inv_st, inv_en, inv_ln, bkt_sh, cq, ka, maxlv, nballs
                        _sync_if_cuda()
                        _s7_t_inv += time.perf_counter() - _s7_t0
                        continue

                    cum_nb = torch.cumsum(nballs, 0)
                    seg_nb = cum_nb - nballs
                    gr_nb = _ensure_long_arange(self, '_s7_boff', total_bup, device)
                    boff = gr_nb - torch.repeat_interleave(seg_nb, nballs)
                    exp_cq = torch.repeat_interleave(cq, nballs)
                    exp_ka = torch.repeat_interleave(ka, nballs)
                    aff_bkt_input = exp_cq * L + exp_ka + boff

                    del exp_cq, exp_ka, boff, cq, ka, maxlv, nballs, cum_nb, seg_nb
                    del inv_st, inv_en, inv_ln, bkt_sh, gr_inv, idx_ia, gr_nb
                    _sync_if_cuda()
                    _s7_t_inv += time.perf_counter() - _s7_t0

                    _sync_if_cuda()
                    _s7_t0 = time.perf_counter()
                    aff_bkts = torch.unique(aff_bkt_input)
                    _sync_if_cuda()
                    _s7_t_unique += time.perf_counter() - _s7_t0
                    del aff_bkt_input

                    if aff_bkts.numel() == 0:
                        continue

                    ball_st = self.shell_red_offsets[(aff_bkts // L) * L]
                    ball_en = self.shell_red_offsets[(aff_bkts // L) * L + (aff_bkts % L) + 1]
                    ball_ln = ball_en - ball_st
                    cum_ball = torch.cumsum(ball_ln, 0)

                    bkt_chunk_start = 0
                    while bkt_chunk_start < aff_bkts.numel():
                        base = (
                            int(cum_ball[bkt_chunk_start - 1].item())
                            if bkt_chunk_start > 0
                            else 0
                        )
                        bkt_chunk_end = int(
                            torch.searchsorted(
                                cum_ball,
                                torch.tensor(
                                    base + S7_MAX_BALL_ENTRIES,
                                    device=device,
                                    dtype=torch.long,
                                ),
                                right=True,
                            ).item()
                        )
                        if bkt_chunk_end <= bkt_chunk_start:
                            bkt_chunk_end = bkt_chunk_start + 1

                        chunk_bkts = aff_bkts[bkt_chunk_start:bkt_chunk_end]
                        chunk_ball_st = ball_st[bkt_chunk_start:bkt_chunk_end]
                        chunk_ball_ln = ball_ln[bkt_chunk_start:bkt_chunk_end]
                        chunk_size = chunk_bkts.numel()
                        total_chunk = int(chunk_ball_ln.sum().item())

                        if total_chunk > 0:
                            _sync_if_cuda()
                            _s7_t0 = time.perf_counter()
                            cum_cl = torch.cumsum(chunk_ball_ln, 0)
                            seg_cl = cum_cl - chunk_ball_ln
                            gr_cl = _ensure_long_arange(self, '_s7_exp', total_chunk, device)
                            sh_idx = (
                                torch.repeat_interleave(chunk_ball_st, chunk_ball_ln)
                                + gr_cl
                                - torch.repeat_interleave(seg_cl, chunk_ball_ln)
                            )
                            exp_a = self.shell_red_indices[sh_idx]
                            b_local = torch.repeat_interleave(
                                torch.arange(chunk_size, device=device, dtype=torch.long),
                                chunk_ball_ln,
                            )
                            _sync_if_cuda()
                            _s7_t_expand += time.perf_counter() - _s7_t0

                            _sync_if_cuda()
                            _s7_t0 = time.perf_counter()
                            ya_vals = self.yA[exp_a].long()
                            new_dm = torch.full((chunk_size,), -1, device=device, dtype=torch.long)
                            new_dm.scatter_reduce_(
                                0, b_local, ya_vals, reduce='amax', include_self=True
                            )
                            _sync_if_cuda()
                            _s7_t_amax += time.perf_counter() - _s7_t0

                            self.d_max[chunk_bkts] = new_dm.to(torch.int32)

                            del exp_a, b_local, ya_vals, new_dm
                            del sh_idx, cum_cl, seg_cl, gr_cl

                        bkt_chunk_start = bkt_chunk_end

                    del aff_bkts, ball_st, ball_en, ball_ln, cum_ball

            if timing_enabled:
                _sync_if_cuda()
                timings['s7'] = time.perf_counter() - phase_t0
                timings['total'] = time.perf_counter() - iter_t0
                _print_phase_timing(iteration, num_free, timings, "ok")

            B_free = F_B_new

        _sync_if_cuda()
        s7_total = _s7_t_inv + _s7_t_unique + _s7_t_expand + _s7_t_amax
        print(
            f"[Step 7 sub-timing | N={N} phases_with_updates={_s7_phases}]\n"
            f"  inv_collection : {_s7_t_inv:.4f}s\n"
            f"  unique_bkts    : {_s7_t_unique:.4f}s\n"
            f"  ball_expand    : {_s7_t_expand:.4f}s\n"
            f"  scatter_amax   : {_s7_t_amax:.4f}s\n"
            f"  total_accounted: {s7_total:.4f}s",
            flush=True,
        )
        self.cleanup_remaining_points()

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.MB == -1).squeeze(1)
        free_r = torch.nonzero(self.MA == -1).squeeze(1)
        count = min(free_b.numel(), free_r.numel())
        if count > 0:
            self.MB[free_b[:count]] = free_r[:count].to(self.MB.dtype)
            self.MA[free_r[:count]] = free_b[:count].to(self.MA.dtype)
