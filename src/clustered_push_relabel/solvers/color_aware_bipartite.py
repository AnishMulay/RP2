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

        # ── Structures 5+6: ball_sizes, d_max, max_list (chunked persistent build) ────
        num_buckets = K * L

        shell_counts_2d = r_counts_long.view(K, L)
        ball_sizes_2d = torch.cumsum(shell_counts_2d, dim=1)
        ball_sizes = ball_sizes_2d.reshape(num_buckets)
        self.ball_sizes = ball_sizes

        self.d_max = torch.where(
            ball_sizes > 0,
            torch.zeros(num_buckets, device=self.device, dtype=torch.int32),
            torch.full((num_buckets,), -1, device=self.device, dtype=torch.int32)
        )
        self.max_list_offsets = torch.cat([
            torch.zeros(1, device=self.device, dtype=torch.long),
            torch.cumsum(ball_sizes, 0),
        ])
        total_ml_cap = int(ball_sizes.sum().item())
        self.max_list_values = torch.full(
            (total_ml_cap,), -1, device=self.device, dtype=torch.int32
        )
        self.max_list_count = ball_sizes.to(device=self.device, dtype=torch.int32)

        # Populate persistent max_list chunk-by-chunk.
        # Initially yA(a)=0 for all A-points, so max_list(ball) = entire ball.
        max_init_writes = 4_000_000
        for q in range(K):
            center_shell_start = self.shell_red_offsets[q * L]
            center_shell_end = self.shell_red_offsets[(q + 1) * L]
            num_center_shell = int((center_shell_end - center_shell_start).item())
            if num_center_shell == 0:
                continue

            center_levels = self.shell_red_levels[center_shell_start:center_shell_end]
            center_points = self.shell_red_indices[center_shell_start:center_shell_end]
            center_local = _ensure_long_arange(
                self, '_init_center_local', num_center_shell, self.device
            )
            center_max_level = self.max_level_per_center[q]
            num_balls_each = center_max_level - center_levels + 1
            cum_writes = torch.cumsum(num_balls_each, 0)

            shell_chunk_start = 0
            while shell_chunk_start < num_center_shell:
                base_writes = int(cum_writes[shell_chunk_start - 1].item()) if shell_chunk_start > 0 else 0
                limit_writes = base_writes + max_init_writes
                shell_chunk_end = int(
                    torch.searchsorted(
                        cum_writes,
                        torch.tensor(limit_writes, device=self.device, dtype=torch.long),
                        right=True,
                    ).item()
                )
                if shell_chunk_end <= shell_chunk_start:
                    shell_chunk_end = shell_chunk_start + 1

                levels_chunk = center_levels[shell_chunk_start:shell_chunk_end]
                local_chunk = center_local[shell_chunk_start:shell_chunk_end]
                points_chunk = center_points[shell_chunk_start:shell_chunk_end]
                nballs_chunk = num_balls_each[shell_chunk_start:shell_chunk_end]
                total_writes = int(nballs_chunk.sum().item())

                if total_writes > 0:
                    cum_nb = torch.cumsum(nballs_chunk, 0)
                    seg_nb = cum_nb - nballs_chunk
                    gr = _ensure_long_arange(self, '_init_write_arange', total_writes, self.device)
                    ball_off = gr - torch.repeat_interleave(seg_nb, nballs_chunk)

                    exp_levels = torch.repeat_interleave(levels_chunk, nballs_chunk)
                    exp_local = torch.repeat_interleave(local_chunk, nballs_chunk)
                    exp_points = torch.repeat_interleave(points_chunk, nballs_chunk)

                    target_bkt = q * L + exp_levels + ball_off
                    write_pos = self.max_list_offsets[target_bkt] + exp_local
                    self.max_list_values[write_pos] = exp_points.to(self.max_list_values.dtype)

                shell_chunk_start = shell_chunk_end

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

            # ── STEP 3: Proposal — each b draws one a from max_list ───────
            if timing_enabled:
                _sync_if_cuda()
                phase_t0 = time.perf_counter()
            ml_lens_raw = self.max_list_count[chosen_bkt].long()
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
                    if timing_enabled:
                        _sync_if_cuda()
                        timings['s3'] = time.perf_counter() - phase_t0
                        timings['total'] = time.perf_counter() - iter_t0
                        _print_phase_timing(iteration, num_free, timings, "no_entries")
                    continue

            ml_starts = self.max_list_offsets[chosen_bkt]
            ml_lens = ml_lens_raw.clamp(min=1)
            rand_idx = (torch.rand(num_b_cand, device=device) * ml_lens.float()).long()
            rand_idx = torch.minimum(rand_idx.clamp_min(0), ml_lens - 1)
            proposal_a = self.max_list_values[ml_starts + rand_idx].long()
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
                S7_CHANGED_CHUNK = 128
                S7_MAX_BALL_ENTRIES = 2_000_000

                for r_chunk_start in range(0, r_new.numel(), S7_CHANGED_CHUNK):
                    r_chunk = r_new[r_chunk_start: r_chunk_start + S7_CHANGED_CHUNK]

                    inv_st = self.inv_a_offsets[r_chunk]
                    inv_en = self.inv_a_offsets[r_chunk + 1]
                    inv_ln = inv_en - inv_st
                    total_inv = int(inv_ln.sum().item())
                    if total_inv == 0:
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
                        continue

                    cum_nb = torch.cumsum(nballs, 0)
                    seg_nb = cum_nb - nballs
                    gr_nb = _ensure_long_arange(self, '_s7_boff', total_bup, device)
                    boff = gr_nb - torch.repeat_interleave(seg_nb, nballs)
                    exp_cq = torch.repeat_interleave(cq, nballs)
                    exp_ka = torch.repeat_interleave(ka, nballs)
                    aff_bkts = torch.unique(exp_cq * L + exp_ka + boff)

                    del exp_cq, exp_ka, boff, cq, ka, maxlv, nballs, cum_nb, seg_nb
                    del inv_st, inv_en, inv_ln, bkt_sh, gr_inv, idx_ia, gr_nb

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

                            ya_vals = self.yA[exp_a].long()
                            new_dm = torch.full((chunk_size,), -1, device=device, dtype=torch.long)
                            new_dm.scatter_reduce_(
                                0, b_local, ya_vals, reduce='amax', include_self=True
                            )

                            is_max = ya_vals == new_dm[b_local]
                            max_a = exp_a[is_max]
                            max_local = b_local[is_max]

                            new_counts = torch.bincount(max_local, minlength=chunk_size)
                            perm = torch.argsort(max_local, stable=True)
                            s_local = max_local[perm]
                            s_a = max_a[perm]
                            grp_off = torch.cat([
                                torch.zeros(1, device=device, dtype=torch.long),
                                torch.cumsum(new_counts, 0),
                            ])
                            rank = (
                                _ensure_long_arange(self, '_s7_rank', s_a.numel(), device)
                                - grp_off[s_local]
                            )
                            write_pos = self.max_list_offsets[chunk_bkts[s_local]] + rank

                            self.max_list_values[write_pos] = s_a.to(
                                self.max_list_values.dtype
                            )
                            self.max_list_count[chunk_bkts] = new_counts.to(torch.int32)
                            self.d_max[chunk_bkts] = new_dm.to(torch.int32)

                            del exp_a, b_local, ya_vals, new_dm, is_max, max_a, max_local
                            del new_counts, perm, s_local, s_a, grp_off, rank, write_pos
                            del sh_idx, cum_cl, seg_cl, gr_cl

                        bkt_chunk_start = bkt_chunk_end

                    del aff_bkts, ball_st, ball_en, ball_ln, cum_ball

            if timing_enabled:
                _sync_if_cuda()
                timings['s7'] = time.perf_counter() - phase_t0
                timings['total'] = time.perf_counter() - iter_t0
                _print_phase_timing(iteration, num_free, timings, "ok")

            B_free = F_B_new

        self.cleanup_remaining_points()

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.MB == -1).squeeze(1)
        free_r = torch.nonzero(self.MA == -1).squeeze(1)
        count = min(free_b.numel(), free_r.numel())
        if count > 0:
            self.MB[free_b[:count]] = free_r[:count].to(self.MB.dtype)
            self.MA[free_r[:count]] = free_b[:count].to(self.MA.dtype)
