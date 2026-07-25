import torch
import time
import gc
from ..clustering.two_level import FastGPUClustering
from ..clustering.k_level import FastGPUMultiLevelClustering


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


# ---------------------------------------------------------------------------
# Phase-level profiling flag.
# Set to True (from the outside, e.g. the profile runner) to enable
# per-phase timing inside solve().  When False every `if PROFILING:` branch
# is skipped at the Python byte-code level — zero runtime overhead.
# ---------------------------------------------------------------------------
PROFILING = False


class TwoLevelBipartiteSolver:
    """
    Bipartite matching solver using a two-level (flat) cluster decomposition.

    This engine partitions points into a single layer of spatial cells 
    to localize push-relabel operations, significantly reducing peak memory.

    Args:
        P_red (torch.Tensor): Source point set (N, D).
        P_blue (torch.Tensor): Target point set (N, D).
        epsilon (float): Base geometric scale for clustering.
        batch_size (int, optional): Processing chunk size. Defaults to 1024.
        metric (str, optional): Distance metric ("L2" or "L1"). Defaults to "L2".
    """
    def __init__(self, P_red, P_blue, epsilon, batch_size=None, metric="L2", verbose=False, profile_memory=False):
        self.device = P_red.device
        self.N = P_red.shape[0]
        self.epsilon = epsilon
        self.P_red = P_red
        self.P_blue = P_blue
        self.batch_size = 1024 if batch_size is None else batch_size
        self.metric = metric
        self.verbose = verbose
        self.profile_memory = bool(profile_memory)
        self.memory_profile = {}

        if self.verbose:
            print("="*60)
            print(f"[Init] Configuration: N={self.N}, Eps={epsilon}, Batch={self.batch_size}, Metric={metric}, Device={self.device}")

        # 1. Clustering
        if self.verbose:
            print("[Step 1] Running Geometric Clustering...")
        t0 = time.time()
        cluster_engine = FastGPUClustering(
            epsilon, batch_size=self.batch_size, micro_batch_size=32, metric=metric,
            profile_memory=self.profile_memory,
        )
        blue_coo, red_coo, r_mask, b_mask = cluster_engine.run(P_red, P_blue)
        if self.profile_memory:
            self.memory_profile.update(
                {f"clustering.{k}": v for k, v in cluster_engine.memory_profile.items()}
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if self.verbose:
            print(f"         Clustering done in {time.time()-t0:.2f}s")
            print(f"         Raw Blue Shells: {blue_coo[0].numel()}")
            print(f"         Raw Red Shells:  {red_coo[0].numel()}")
        
        # 2. Indexing
        if self.verbose:
            print("[Step 2] Building CSR Index (Group by Center)...")
        t0 = time.time()
        self._build_csr_from_coo_cpu(blue_coo, red_coo, r_mask, b_mask)
        if self.verbose:
            print(f"         Indexing done in {time.time()-t0:.2f}s")
        
        if self.verbose:
            print("   [Debug] Starting GC and State Init...")
        t_setup = time.time()
        del blue_coo, red_coo, cluster_engine
        gc.collect()
        
        # 3. State Init (Integer Scaling)
        self.yA = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.yB = torch.full((self.N,), 1, device=self.device, dtype=torch.int32)
        self.MA = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)
        self.MB = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if self.verbose:
            print(f"   [Debug] GC & State Init finished in {time.time() - t_setup:.4f}s")

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.MB == -1).squeeze(1)
        free_r = torch.nonzero(self.MA == -1).squeeze(1)
        count = min(free_b.numel(), free_r.numel())
        if count > 0:
            self.MB[free_b[:count]] = free_r[:count].to(self.MB.dtype)
            self.MA[free_r[:count]] = free_b[:count].to(self.MA.dtype)
        if self.verbose:
            print(f"[Cleanup] Arbitrarily matched {count} remaining pairs.")

    def calculate_final_stats(self):
        if not self.verbose:
            return
        if self.metric == "L1":
            dists = torch.norm(self.P_blue - self.P_red[self.MB], p=1, dim=1)
            label = "Manhattan"
        else:
            dists = torch.norm(self.P_blue - self.P_red[self.MB], p=2, dim=1)
            label = "Euclidean"
        total_cost = dists.sum()
        avg_cost = total_cost / self.N
        print(f"Total {label} Cost: {total_cost.item():.4f}")
        print(f"Avg {label} Cost: {avg_cost.item():.4f}")

    def _build_csr_from_coo_cpu(self, blue_coo, red_coo, red_mask, blue_mask):
        """
        Builds CSR structures grouped by CENTER ID.
        Stores Levels as attributes.
        """
        b_c, b_l, b_p = blue_coo
        r_c, r_l, r_p = red_coo
        N = self.N

        # Topology + memory analysis before filtering (use raw COO triplets)
        if red_mask.device != r_c.device:
            red_mask = red_mask.cpu()
        if blue_mask.device != b_c.device:
            blue_mask = blue_mask.cpu()

        r_sampled = red_mask[r_c] if r_c.numel() > 0 else torch.empty(0, dtype=torch.bool)
        b_sampled = blue_mask[b_c] if b_c.numel() > 0 else torch.empty(0, dtype=torch.bool)

        def _category_stats(centers, levels):
            edges = int(centers.numel())
            if edges == 0:
                return {
                    "edges": 0,
                    "clusters": 0,
                    "buckets": 0,
                    "avg_cluster": 0.0,
                    "mem_mb": 0.0,
                }
            clusters = int(torch.unique(centers).numel())
            max_level = levels.max().to(torch.long)
            bucket_keys = (centers.to(torch.long) * (max_level + 1)) + levels.to(torch.long)
            buckets = int(torch.unique(bucket_keys).numel())
            avg_cluster = edges / max(clusters, 1)
            mem_mb = edges * 3 * 4 / (1024 ** 2)
            return {
                "edges": edges,
                "clusters": clusters,
                "buckets": buckets,
                "avg_cluster": avg_cluster,
                "mem_mb": mem_mb,
            }

        if self.verbose:
            red_sampled_stats = _category_stats(r_c[r_sampled], r_l[r_sampled])
            red_local_stats = _category_stats(r_c[~r_sampled], r_l[~r_sampled])
            blue_sampled_stats = _category_stats(b_c[b_sampled], b_l[b_sampled])
            blue_local_stats = _category_stats(b_c[~b_sampled], b_l[~b_sampled])

            if r_c.numel() == 0 and b_c.numel() == 0:
                total_stats = {
                    "edges": 0,
                    "clusters": 0,
                    "buckets": 0,
                    "avg_cluster": 0.0,
                    "mem_mb": 0.0,
                }
            else:
                total_centers = torch.cat([r_c, b_c + N])
                total_levels = torch.cat([r_l, b_l])
                total_stats = _category_stats(total_centers, total_levels)

            print("    [Cluster Analysis]")
            print("    Category       | Clusters | Buckets  | Edges        | Avg Clust | Mem (MB)")
            print("    ------------------------------------------------------------------------")
            def _print_row(label, stats):
                print(
                    f"    {label:<14} | {stats['clusters']:>8} | {stats['buckets']:>7} | "
                    f"{stats['edges']:>12,} | {stats['avg_cluster']:>9.1f} | {stats['mem_mb']:>7.1f}"
                )
            _print_row("RED (Sampled)", red_sampled_stats)
            _print_row("RED (Local)", red_local_stats)
            _print_row("BLUE (Sampled)", blue_sampled_stats)
            _print_row("BLUE (Local)", blue_local_stats)
            print("    ------------------------------------------------------------------------")
            _print_row("TOTAL", total_stats)

        # 1. Unify center IDs and merge all triplets on CPU
        b_c_shifted = b_c + N
        all_centers = torch.cat([b_c_shifted, r_c])
        all_levels = torch.cat([b_l, r_l])
        all_points = torch.cat([b_p, r_p])

        # 2. Identify valid centers (contain at least one Red and one Blue point)
        is_red_point = all_points < N
        centers_with_red = torch.unique(all_centers[is_red_point])
        centers_with_blue = torch.unique(all_centers[~is_red_point])
        if centers_with_red.numel() == 0 or centers_with_blue.numel() == 0:
            raise ValueError("No valid clusters found (Intersection Empty).")

        valid_centers = centers_with_red[torch.isin(centers_with_red, centers_with_blue)]
        if valid_centers.numel() == 0:
            raise ValueError("No valid clusters found (Intersection Empty).")

        if self.verbose:
            print(f"         Active Centers: {valid_centers.numel()}")

        # 3. Filter to valid centers only
        mask_valid = torch.isin(all_centers, valid_centers)
        all_centers = all_centers[mask_valid]
        all_levels = all_levels[mask_valid]
        all_points = all_points[mask_valid]
        is_red_point = is_red_point[mask_valid]

        # Map centers to dense IDs 0..K-1 for efficient offsets
        center_map = torch.searchsorted(valid_centers, all_centers)
        self.num_active_centers = int(valid_centers.numel())

        # 4. Build Red CSR (Grouped by Center)
        red_mask = is_red_point
        red_centers = center_map[red_mask]
        red_points = all_points[red_mask]
        red_levels = all_levels[red_mask]

        max_red_level = red_levels.max().to(torch.long)
        r_sort_key = (red_centers.to(torch.long) * (max_red_level + 1)) + red_levels.to(torch.long)
        perm_r = torch.argsort(r_sort_key)
        self.red_indices = red_points[perm_r].to(device=self.device, dtype=torch.long)
        self.red_levels = red_levels[perm_r].to(device=self.device, dtype=torch.int32)
        sorted_r_centers = red_centers[perm_r]

        r_counts = torch.bincount(sorted_r_centers, minlength=self.num_active_centers)
        r_counts_long = r_counts.to(device=self.device, dtype=torch.long)
        self.red_offsets = torch.cat(
            [torch.zeros(1, device=self.device, dtype=torch.long), torch.cumsum(r_counts_long, 0)]
        )

        # Expand Center IDs for scatter ops
        self.red_expand_center_ids = torch.repeat_interleave(
            torch.arange(self.num_active_centers, device=self.device, dtype=torch.long), r_counts_long
        )

        # 5. Build Blue CSR (Inverted: Blue -> [Centers])
        blue_mask = ~is_red_point
        blue_centers = center_map[blue_mask]
        blue_points = all_points[blue_mask] - N  # Rebase Blue IDs to 0..N-1
        blue_levels = all_levels[blue_mask]

        perm_b = torch.argsort(blue_points)
        self.blue_center_indices = blue_centers[perm_b].to(device=self.device, dtype=torch.long)
        self.blue_levels = blue_levels[perm_b].to(device=self.device, dtype=torch.int32)
        sorted_b_pts = blue_points[perm_b]

        b_counts = torch.bincount(sorted_b_pts, minlength=N)
        b_counts_long = b_counts.to(device=self.device, dtype=torch.long)
        self.blue_offsets = torch.cat(
            [torch.zeros(1, device=self.device, dtype=torch.long), torch.cumsum(b_counts_long, 0)]
        )
        self._center_max_yA = torch.empty(self.num_active_centers, device=self.device, dtype=torch.int32)
        self._center_max_L_A = torch.empty(self.num_active_centers, device=self.device, dtype=torch.int32)

        if self.verbose:
            print(f"         Red Entries: {self.red_indices.numel()} (GPU)")
            print(f"         Blue Entries: {self.blue_center_indices.numel()} (GPU)")
            print(f"         Avg Degree: {(self.blue_center_indices.numel() + self.red_indices.numel())/N:.2f}")

    def solve(self):
        if self.verbose:
            print("[Debug] Entering solve() method...")
        # print(f"\n[Step 3] Starting Push-Relabel Loop...")
        iteration = 0
        use_cuda = self.device.type == "cuda"
        DEBUG = False
        DEBUG_EVERY = 50
        DEBUG_SYNC = False
        DEBUG_PIPE = False

        def maybe_sync():
            if use_cuda and DEBUG_SYNC:
                torch.cuda.synchronize()

        def log_mem(stage):
            if not use_cuda:
                return
            alloc = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"    [Mem - {stage}] Alloc: {alloc:.1f}MB | Peak: {peak:.1f}MB")
            torch.cuda.reset_peak_memory_stats()
        
        prev_num_free = None
        stuck_iters = 0
        prev_matched_count = None
        center_max_yA = self._center_max_yA
        center_max_L_A = self._center_max_L_A
        B_free = torch.arange(self.N, device=self.device, dtype=torch.long)

        if PROFILING:
            _prof = {
                'A_maint':     0.0,  # scatter_reduce for center_max_yA / center_max_L_A
                'B_index':     0.0,  # ragged index construction (fast path)
                'B_gather':    0.0,  # CSR lookups for c_ids / b_levels (fast path)
                'B_slack':     0.0,  # slack arithmetic + candidate filter (fast path)
                'C_free_reds': 0.0,  # MA mask filter to get free red points
                'C_sort_rank': 0.0,  # argsort + bincount + cumsum + rank ops
                'C_commit':    0.0,  # final slack check + MB/MA writes
                'D_relabel':   0.0,  # B_free update + yB/yA increments
                'iterations':  0,
            }
            def _psync():
                if use_cuda:
                    torch.cuda.synchronize()

        while True:
            num_free = B_free.numel()
            if num_free <= self.epsilon * self.N:
                # print("[Converged] Free points <= Threshold. Stopping.")
                break

            iteration += 1
            if PROFILING:
                _prof['iterations'] += 1
            if prev_num_free is not None and num_free == prev_num_free:
                stuck_iters += 1
            else:
                stuck_iters = 0
            prev_num_free = num_free
            should_dbg = DEBUG and (iteration % DEBUG_EVERY == 0 or stuck_iters >= 20)
            dbg_total_edges = 0
            dbg_eq0 = 0
            dbg_leq0 = 0
            dbg_min_slack_est = None
            dbg_max_slack_est = None
            zero_deg = 0
            if DEBUG_PIPE:
                dbg_cand_initial = 0
                dbg_after_free_u = 0
                dbg_after_free_v = 0
                dbg_after_conflict = 0
                dbg_after_commit = 0
            # log_mem("Start Iter")

            if self.verbose and iteration % 10 == 0:
                print(f"    [Iter {iteration}] Free: {num_free}")

            # A. Maintenance: Max yA per Center
            maybe_sync()
            if PROFILING: _psync(); _t = time.perf_counter()
            yA_expanded = self.yA[self.red_indices]
            center_max_yA.zero_()
            # We want to check: Slack_Est = 2*L_b - yB - Max_yA
            # So we need max(yA) in the center.
            center_max_yA.scatter_reduce_(
                0,
                self.red_expand_center_ids,
                yA_expanded,
                reduce="amax",
                include_self=False,
            )
            center_max_L_A.zero_()
            center_max_L_A.scatter_reduce_(
                0,
                self.red_expand_center_ids,
                self.red_levels,
                reduce="amax",
                include_self=False,
            )
            maybe_sync()
            if PROFILING: _psync(); _prof['A_maint'] += time.perf_counter() - _t
            # log_mem("After Maint")

            # B. Push (Ragged Gather)
            maybe_sync()
            if PROFILING: _psync(); _t_b_index = time.perf_counter()
            push_batch_size = 5000

            starts_all = self.blue_offsets[B_free]
            ends_all = self.blue_offsets[B_free + 1]
            lengths_all = ends_all - starts_all
            total_edges = int(lengths_all.sum().item())
            if should_dbg:
                dbg_total_edges = total_edges
                zero_deg = int((lengths_all == 0).sum().item())
                if zero_deg > 0:
                    print(f"[DBG it={iteration}] zero-degree free points: {zero_deg}/{num_free}")
            if total_edges == 0:
                maybe_sync()
                self.yB[B_free] += 1
                if should_dbg:
                    if dbg_min_slack_est is None:
                        dbg_min_slack_est = 0
                        dbg_max_slack_est = 0
                    matched_now = self.N - num_free
                    delta_matched = 0 if prev_matched_count is None else matched_now - prev_matched_count
                    print(
                        f"[DBG it={iteration}] free={num_free} stuck={stuck_iters} edges={dbg_total_edges} "
                        f"zeroDeg={zero_deg} eq0={dbg_eq0} leq0={dbg_leq0} "
                        f"minSlack={dbg_min_slack_est} maxSlack={dbg_max_slack_est} "
                        f"matched={matched_now}/{self.N} deltaMatched={delta_matched}"
                    )
                if DEBUG_PIPE and stuck_iters >= 20:
                    print(
                        f"[PIPE it={iteration}] init={dbg_cand_initial} freeU={dbg_after_free_u} "
                        f"freeV={dbg_after_free_v} resolved={dbg_after_conflict} "
                        f"committed={dbg_after_commit}"
                    )
                prev_matched_count = self.N - num_free
                continue

            if num_free <= push_batch_size:
                chunk = B_free
                starts = starts_all
                lengths = lengths_all
                repeat_starts = torch.repeat_interleave(starts, lengths)
                cum_len = torch.cumsum(lengths, 0)
                segment_starts_packed = cum_len - lengths
                global_range = _ensure_long_arange(self, "_edge_arange_buf", total_edges, self.device)
                repeat_packed_starts = torch.repeat_interleave(segment_starts_packed, lengths)
                offsets = global_range - repeat_packed_starts
                active_edge_indices = repeat_starts + offsets
                # log_mem("Mid-Push (Indices)")
                if PROFILING: _psync(); _prof['B_index'] += time.perf_counter() - _t_b_index; _t_b_gather = time.perf_counter()

                # Reconstruct attributes
                active_b_ids = torch.repeat_interleave(chunk, lengths)
                active_c_ids = self.blue_center_indices[active_edge_indices]
                active_b_levels = self.blue_levels[active_edge_indices]
                if PROFILING: _psync(); _prof['B_gather'] += time.perf_counter() - _t_b_gather; _t_b_slack = time.perf_counter()

                # Slack Estimation (Lower Bound)
                # Slack = 2 * max(L_b, L_a) - yB - yA
                # Lower Bound = 2 * L_b - yB - Max_yA
                # If Lower Bound > 0, then Real Slack is definitely > 0 (since L_a >= 0, yA <= Max_yA)
                # So we only check where Lower Bound <= 0
                slacks_est = (
                    2 * active_b_levels
                    - self.yB[active_b_ids]
                    - center_max_yA[active_c_ids]
                )
                active_max_L_A = center_max_L_A[active_c_ids]
                slacks_high = (
                    2 * torch.maximum(active_b_levels, active_max_L_A)
                    - self.yB[active_b_ids]
                    - center_max_yA[active_c_ids]
                )

                if should_dbg:
                    dbg_eq0 += int((slacks_est == 0).sum().item())
                    dbg_leq0 += int((slacks_est <= 0).sum().item())
                    local_min = int(slacks_est.min().item())
                    local_max = int(slacks_est.max().item())
                    if dbg_min_slack_est is None or local_min < dbg_min_slack_est:
                        dbg_min_slack_est = local_min
                    if dbg_max_slack_est is None or local_max > dbg_max_slack_est:
                        dbg_max_slack_est = local_max

                candidates = (slacks_est <= 0) & (slacks_high >= 0)
                if PROFILING: _psync(); _prof['B_slack'] += time.perf_counter() - _t_b_slack
                maybe_sync()
                # log_mem("After Push")

                if candidates.any():
                    win_b = active_b_ids[candidates]
                    win_c = active_c_ids[candidates]
                    win_l_b = active_b_levels[candidates]
                else:
                    win_b = torch.empty(0, device=self.device, dtype=B_free.dtype)
                    win_c = torch.empty(0, device=self.device, dtype=self.blue_center_indices.dtype)
                    win_l_b = torch.empty(0, device=self.device, dtype=self.blue_levels.dtype)
            else:
                all_win_b = []
                all_win_c = []
                all_win_l_b = []

                for i in range(0, num_free, push_batch_size):
                    chunk = B_free[i : i + push_batch_size]
                    starts = self.blue_offsets[chunk]
                    ends = self.blue_offsets[chunk + 1]
                    lengths = ends - starts
                    total_edges = int(lengths.sum().item())
                    if total_edges == 0:
                        continue

                    repeat_starts = torch.repeat_interleave(starts, lengths)
                    cum_len = torch.cumsum(lengths, 0)
                    segment_starts_packed = cum_len - lengths
                    global_range = _ensure_long_arange(self, "_edge_arange_buf", total_edges, self.device)
                    repeat_packed_starts = torch.repeat_interleave(segment_starts_packed, lengths)
                    offsets = global_range - repeat_packed_starts
                    active_edge_indices = repeat_starts + offsets
                    # log_mem("Mid-Push (Indices)")

                    # Reconstruct attributes
                    active_b_ids = torch.repeat_interleave(chunk, lengths)
                    active_c_ids = self.blue_center_indices[active_edge_indices]
                    active_b_levels = self.blue_levels[active_edge_indices]

                    # Slack Estimation (Lower Bound)
                    # Slack = 2 * max(L_b, L_a) - yB - yA
                    # Lower Bound = 2 * L_b - yB - Max_yA
                    # If Lower Bound > 0, then Real Slack is definitely > 0 (since L_a >= 0, yA <= Max_yA)
                    # So we only check where Lower Bound <= 0
                    slacks_est = (
                        2 * active_b_levels 
                        - self.yB[active_b_ids] 
                        - center_max_yA[active_c_ids]
                    )
                    active_max_L_A = center_max_L_A[active_c_ids]
                    slacks_high = (
                        2 * torch.maximum(active_b_levels, active_max_L_A)
                        - self.yB[active_b_ids]
                        - center_max_yA[active_c_ids]
                    )

                    if should_dbg:
                        dbg_eq0 += int((slacks_est == 0).sum().item())
                        dbg_leq0 += int((slacks_est <= 0).sum().item())
                        local_min = int(slacks_est.min().item())
                        local_max = int(slacks_est.max().item())
                        if dbg_min_slack_est is None or local_min < dbg_min_slack_est:
                            dbg_min_slack_est = local_min
                        if dbg_max_slack_est is None or local_max > dbg_max_slack_est:
                            dbg_max_slack_est = local_max

                    candidates = (slacks_est <= 0) & (slacks_high >= 0)
                    maybe_sync()
                    # log_mem("After Push")

                    if candidates.any():
                        all_win_b.append(active_b_ids[candidates])
                        all_win_c.append(active_c_ids[candidates])
                        all_win_l_b.append(active_b_levels[candidates])

                if not all_win_b:
                    win_b = torch.empty(0, device=self.device, dtype=B_free.dtype)
                    win_c = torch.empty(0, device=self.device, dtype=self.blue_center_indices.dtype)
                    win_l_b = torch.empty(0, device=self.device, dtype=self.blue_levels.dtype)
                else:
                    win_b = torch.cat(all_win_b)
                    win_c = torch.cat(all_win_c)
                    win_l_b = torch.cat(all_win_l_b)

            if should_dbg and dbg_min_slack_est is None:
                dbg_min_slack_est = 0
                dbg_max_slack_est = 0
            if DEBUG_PIPE:
                dbg_cand_initial = int(win_b.numel())
                dbg_after_free_u = int(win_b.numel())
                dbg_after_free_v = int(win_b.numel())

            b_match = None
            if win_b.numel() != 0:
                # C. Resolve
                maybe_sync()
                if PROFILING: _psync(); _t_c = time.perf_counter()
                # We have (Blue, Center) pairs that MIGHT have a match.

                free_red_mask = self.MA[self.red_indices] == -1
                # Reds are already sorted by Center then Level.
                red_ids = self.red_indices[free_red_mask]
                red_levels = self.red_levels[free_red_mask]
                red_c_ids = self.red_expand_center_ids[free_red_mask]
                if PROFILING: _psync(); _prof['C_free_reds'] += time.perf_counter() - _t_c; _t_c_sr = time.perf_counter()

                if win_b.numel() != 0 and red_ids.numel() != 0:
                    max_level = torch.maximum(win_l_b.max(), red_levels.max()).to(torch.long)
                    key_scale = max_level + 1

                    b_sort_key = (win_c.to(torch.long) * key_scale) + win_l_b.to(torch.long)
                    b_perm = torch.argsort(b_sort_key)
                    # log_mem("Mid-Resolve (Sort)")
                    b_sorted = win_b[b_perm]
                    c_sorted = win_c[b_perm]
                    l_b_sorted = win_l_b[b_perm]

                    r_sorted = red_ids
                    c_r_sorted = red_c_ids
                    l_r_sorted = red_levels

                    num_centers = self.num_active_centers
                    b_counts = torch.bincount(c_sorted, minlength=num_centers)
                    r_counts = torch.bincount(c_r_sorted, minlength=num_centers)
                    limits = torch.minimum(b_counts, r_counts)

                    b_offsets = _ensure_zero_long_buffer(
                        self, "_resolve_b_offsets_buf", num_centers + 1, self.device
                    )
                    b_offsets[1:] = torch.cumsum(b_counts, 0)
                    r_offsets = _ensure_zero_long_buffer(
                        self, "_resolve_r_offsets_buf", num_centers + 1, self.device
                    )
                    r_offsets[1:] = torch.cumsum(r_counts, 0)

                    b_rank = _ensure_long_arange(
                        self, "_resolve_b_rank_arange_buf", c_sorted.numel(), self.device
                    ) - b_offsets[c_sorted]
                    r_rank = _ensure_long_arange(
                        self, "_resolve_r_rank_arange_buf", c_r_sorted.numel(), self.device
                    ) - r_offsets[c_r_sorted]

                    b_keep = b_rank < limits[c_sorted]
                    r_keep = r_rank < limits[c_r_sorted]
                    if PROFILING: _psync(); _prof['C_sort_rank'] += time.perf_counter() - _t_c_sr; _t_c_commit = time.perf_counter()

                    b_final = b_sorted[b_keep]
                    l_b_final = l_b_sorted[b_keep]
                    r_final = r_sorted[r_keep]
                    l_r_final = l_r_sorted[r_keep]

                    pair_count = min(b_final.numel(), r_final.numel())
                    if pair_count != 0:
                        if b_final.numel() != pair_count:
                            b_final = b_final[:pair_count]
                            l_b_final = l_b_final[:pair_count]
                        if r_final.numel() != pair_count:
                            r_final = r_final[:pair_count]
                            l_r_final = l_r_final[:pair_count]
                        if DEBUG_PIPE:
                            dbg_after_conflict = int(b_final.numel())

                        prev_match_count = None
                        if DEBUG_PIPE:
                            prev_match_count = int((self.MA != -1).sum().item())
                        y_b = self.yB[b_final]
                        y_a = self.yA[r_final]
                        slack = 2 * torch.maximum(l_b_final, l_r_final) - y_b - y_a
                        valid = slack == 0

                        b_match = b_final[valid]
                        r_match = r_final[valid]
                        if b_match.numel() != 0:
                            self.MB[b_match] = r_match.to(self.MB.dtype)
                            self.MA[r_match] = b_match.to(self.MA.dtype)
                        if DEBUG_PIPE:
                            new_count = int((self.MA != -1).sum().item())
                            dbg_after_commit = new_count - prev_match_count
                    if PROFILING: _psync(); _prof['C_commit'] += time.perf_counter() - _t_c_commit
                    del (
                        b_sort_key,
                        b_perm,
                        b_sorted,
                        c_sorted,
                        l_b_sorted,
                        r_sorted,
                        c_r_sorted,
                        l_r_sorted,
                        b_counts,
                        r_counts,
                        limits,
                        b_offsets,
                        r_offsets,
                        b_rank,
                        r_rank,
                        b_keep,
                        r_keep,
                        b_final,
                        r_final,
                        l_b_final,
                        l_r_final,
                    )
                del red_ids, red_levels, red_c_ids, free_red_mask, win_b, win_c, win_l_b
                maybe_sync()
            # log_mem("After Resolve")

            # D. Relabel
            maybe_sync()
            if PROFILING: _psync(); _t_d = time.perf_counter()
            if b_match is not None and b_match.numel() != 0:
                keep_free_mask = _ensure_bool_buffer(
                    self, "_keep_free_mask_buf", num_free, self.device
                )
                keep_free_mask.fill_(True)
                keep_free_mask[torch.searchsorted(B_free, b_match)] = False
                still_free = B_free[keep_free_mask]
            else:
                still_free = B_free
            self.yB[still_free] += 1
            
            matched_r = torch.nonzero(self.MA != -1).squeeze(1)
            self.yA[matched_r] -= 1
            maybe_sync()
            # log_mem("After Relabel")
            B_free = still_free
            if PROFILING: _psync(); _prof['D_relabel'] += time.perf_counter() - _t_d
            matched_now = self.N - still_free.numel()
            if should_dbg:
                delta_matched = 0 if prev_matched_count is None else matched_now - prev_matched_count
                print(
                    f"[DBG it={iteration}] free={num_free} stuck={stuck_iters} edges={dbg_total_edges} "
                    f"zeroDeg={zero_deg} eq0={dbg_eq0} leq0={dbg_leq0} "
                    f"minSlack={dbg_min_slack_est} maxSlack={dbg_max_slack_est} "
                    f"matched={matched_now}/{self.N} deltaMatched={delta_matched}"
                )
            if DEBUG_PIPE and stuck_iters >= 20:
                print(
                    f"[PIPE it={iteration}] init={dbg_cand_initial} freeU={dbg_after_free_u} "
                    f"freeV={dbg_after_free_v} resolved={dbg_after_conflict} "
                    f"committed={dbg_after_commit}"
                )
            prev_matched_count = matched_now

            if iteration > 50000:
                # print("Max Iterations Reached.")
                break

        if PROFILING:
            self._prof = _prof
        self.cleanup_remaining_points()
        if self.verbose:
            print(f"Matched: {(self.MB != -1).sum().item()}/{self.N}")
        self.calculate_final_stats()


class KLevelBipartiteSolver:
    """
    Bipartite matching solver using a K-level hierarchical decomposition.

    This engine builds a deep spatial tree of clusters, enabling extreme 
    scale execution by routing flow through multiple scales of resolution.

    Args:
        P_red (torch.Tensor): Source point set (N, D).
        P_blue (torch.Tensor): Target point set (N, D).
        epsilon (float): Discretization base scale.
        k (int, optional): Number of levels in the hierarchy. Defaults to 4.
        batch_size (int, optional): Processing chunk size. Defaults to 2048.
        metric (str, optional): Distance metric ("L2" or "L1"). Defaults to "L2".
    """
    def __init__(self, P_red, P_blue, epsilon, k=4, batch_size=None, metric="L2", verbose=False, profile_memory=False):
        self.device = P_red.device
        self.N = P_red.shape[0]
        self.epsilon = epsilon
        self.k = k
        self.P_red = P_red
        self.P_blue = P_blue
        self.batch_size = 2048 if batch_size is None else batch_size
        self.metric = metric
        self.verbose = verbose
        self.profile_memory = bool(profile_memory)
        self.memory_profile = {}

        if self.verbose:
            print("="*60)
            print(f"[Init] Config: N={self.N}, Eps={epsilon}, Levels={k}, Batch={self.batch_size}, Metric={metric}, Device={self.device}")

        # 1. Multi-Level Clustering
        if self.verbose:
            print("[Step 1] Running Multi-Level Hierarchical Clustering...")
        t0 = time.time()
        cluster_engine = FastGPUMultiLevelClustering(
            epsilon, k=k, batch_size=self.batch_size, metric=metric,
            profile_memory=self.profile_memory,
        )
        blue_coo, red_coo, levels_red, levels_blue = cluster_engine.run(P_red, P_blue)
        if self.profile_memory:
            self.memory_profile.update(
                {f"clustering.{k_}": v for k_, v in cluster_engine.memory_profile.items()}
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if self.verbose:
            print(f"         Clustering done in {time.time()-t0:.2f}s")
            print(f"         Red Edges:  {red_coo[0].numel()}")
            print(f"         Blue Edges: {blue_coo[0].numel()}")
        
        # 2. Indexing
        if self.verbose:
            print("[Step 2] Building CSR Index (Unified Center Space)...")
        t0 = time.time()
        self._build_csr_from_coo_cpu(blue_coo, red_coo, levels_red, levels_blue)
        if self.verbose:
            print(f"         Indexing done in {time.time()-t0:.2f}s")
        
        # Cleanup
        if self.verbose:
            print("   [Debug] Starting GC and State Init...")
        t_setup = time.time()
        del blue_coo, red_coo, levels_red, levels_blue, cluster_engine
        gc.collect()
        
        # 3. State Init (Integer Scaling)
        self.yA = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.yB = torch.full((self.N,), 1, device=self.device, dtype=torch.int32)
        self.MA = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)
        self.MB = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if self.verbose:
            print(f"   [Debug] GC & State Init finished in {time.time() - t_setup:.4f}s")

    def _build_csr_from_coo_cpu(self, blue_coo, red_coo, levels_red, levels_blue):
        """
        Builds CSR structures.
        Crucial Change: Centers are now disjoint sets from the hierarchy.
        We need to unify 'Red Centers' and 'Blue Centers' into a single index space.
        
        Mappings:
        - Red Centers: Indices 0..N-1 (referenced in red_coo[0])
        - Blue Centers: Indices 0..N-1 (referenced in blue_coo[0])
        
        We treat Blue Centers as IDs N..2N-1 internally here to merge, then filter.
        """
        b_c, b_l, b_p = blue_coo
        r_c, r_l, r_p = red_coo
        N = self.N
        levels_red = levels_red.cpu()
        levels_blue = levels_blue.cpu()

        def _category_stats(centers, levels):
            edges = int(centers.numel())
            if edges == 0:
                return {
                    "edges": 0,
                    "clusters": 0,
                    "buckets": 0,
                    "avg_cluster": 0.0,
                    "mem_mb": 0.0,
                }
            clusters = int(torch.unique(centers).numel())
            max_level = levels.max().to(torch.long)
            bucket_keys = (centers.to(torch.long) * (max_level + 1)) + levels.to(torch.long)
            buckets = int(torch.unique(bucket_keys).numel())
            avg_cluster = edges / max(clusters, 1)
            mem_mb = edges * 12 / (1024 ** 2)
            return {
                "edges": edges,
                "clusters": clusters,
                "buckets": buckets,
                "avg_cluster": avg_cluster,
                "mem_mb": mem_mb,
            }

        if self.verbose:
            total_stats = {
                "edges": 0,
                "clusters": 0,
                "buckets": 0,
                "avg_cluster": 0.0,
                "mem_mb": 0.0,
            }
            if r_c.numel() != 0 or b_c.numel() != 0:
                total_centers = torch.cat([r_c, b_c + N])
                total_levels = torch.cat([r_l, b_l])
                total_stats = _category_stats(total_centers, total_levels)

            print("    [Cluster Analysis]")
            print("    Category       | Clusters | Buckets  | Edges        | Avg Clust | Mem (MB)")
            print("    ------------------------------------------------------------------------")
            def _print_row(label, stats):
                print(
                    f"    {label:<14} | {stats['clusters']:>8} | {stats['buckets']:>7} | "
                    f"{stats['edges']:>12,} | {stats['avg_cluster']:>9.1f} | {stats['mem_mb']:>7.1f}"
                )
            for i in range(self.k - 1, -1, -1):
                red_mask = levels_red[r_c] == i
                blue_mask = levels_blue[b_c] == i
                red_stats = _category_stats(r_c[red_mask], r_l[red_mask])
                blue_stats = _category_stats(b_c[blue_mask], b_l[blue_mask])
                _print_row(f"RED (Lvl {i})", red_stats)
                _print_row(f"BLUE (Lvl {i})", blue_stats)
            print("    ------------------------------------------------------------------------")
            _print_row("TOTAL", total_stats)

        # Shift Blue Center IDs to avoid collision with Red Center IDs
        b_c_shifted = b_c + N
        
        # Concatenate all edges to find Active Centers (those that bridge Red and Blue)
        # We need centers that have at least one Red point and one Blue point.
        
        # Edges coming from Red Centers
        # r_c: Center ID (0..N-1)
        # r_p: Point ID (0..2N-1). 0..N-1 is Red, N..2N-1 is Blue.
        
        # Edges coming from Blue Centers
        # b_c_shifted: Center ID (N..2N-1)
        # b_p: Point ID.
        
        # 1. Global Lists
        all_centers = torch.cat([r_c, b_c_shifted])
        all_levels = torch.cat([r_l, b_l])
        all_points = torch.cat([r_p, b_p])
        
        # 2. Filter Active Centers
        # A center is active if it connects to >=1 Red Node AND >=1 Blue Node.
        is_red_point = all_points < N
        
        # Find centers connected to Red Points
        centers_with_red = torch.unique(all_centers[is_red_point])
        # Find centers connected to Blue Points
        centers_with_blue = torch.unique(all_centers[~is_red_point])
        
        # Intersection
        valid_centers = centers_with_red[torch.isin(centers_with_red, centers_with_blue)]
        
        if valid_centers.numel() == 0:
            raise ValueError("No valid clusters found (Intersection Empty).")
            
        if self.verbose:
            print(f"         Active Centers: {valid_centers.numel()}")
        
        # 3. Filter Edges to Valid Centers Only
        mask_valid = torch.isin(all_centers, valid_centers)
        all_centers = all_centers[mask_valid]
        all_levels = all_levels[mask_valid]
        all_points = all_points[mask_valid]
        is_red_point = is_red_point[mask_valid] # Re-slice mask
        
        # 4. Map Sparse Center IDs to Dense IDs (0..K-1)
        # This is required for CSR offsets
        center_map = torch.searchsorted(valid_centers, all_centers)
        self.num_active_centers = int(valid_centers.numel())
        
        # 5. Build Red CSR (Points 0..N-1)
        # Group by Center
        red_mask = is_red_point
        red_centers = center_map[red_mask]
        red_points = all_points[red_mask] # Already 0..N-1
        red_levels = all_levels[red_mask]
        
        # Sort by (Center, Level)
        max_red_level = red_levels.max().to(torch.long)
        r_sort_key = (red_centers.to(torch.long) * (max_red_level + 1)) + red_levels.to(torch.long)
        perm_r = torch.argsort(r_sort_key)
        
        self.red_indices = red_points[perm_r].to(device=self.device, dtype=torch.long)
        self.red_levels = red_levels[perm_r].to(device=self.device, dtype=torch.int32)
        sorted_r_centers = red_centers[perm_r]

        r_counts = torch.bincount(sorted_r_centers, minlength=self.num_active_centers)
        r_counts_long = r_counts.to(device=self.device, dtype=torch.long)
        self.red_offsets = torch.cat(
            [torch.zeros(1, device=self.device, dtype=torch.long), torch.cumsum(r_counts_long, 0)]
        )
        # Expansion array for scatter operations
        self.red_expand_center_ids = torch.repeat_interleave(
            torch.arange(self.num_active_centers, device=self.device, dtype=torch.long), r_counts_long
        )
        
        # 6. Build Blue CSR (Points N..2N-1 -> Remap to 0..N-1)
        # Inverted Index: We need to lookup Neighbors of Blue Points efficiently.
        # Blue CSR: Rows = Blue Points, Cols = Centers
        blue_mask = ~is_red_point
        blue_centers = center_map[blue_mask]
        blue_points = all_points[blue_mask] - N # Rebase to 0..N-1
        blue_levels = all_levels[blue_mask]
        
        # Sort by Blue Point ID
        perm_b = torch.argsort(blue_points)
        self.blue_center_indices = blue_centers[perm_b].to(device=self.device, dtype=torch.long)
        self.blue_levels = blue_levels[perm_b].to(device=self.device, dtype=torch.int32)
        sorted_b_pts = blue_points[perm_b]

        b_counts = torch.bincount(sorted_b_pts, minlength=N)
        b_counts_long = b_counts.to(device=self.device, dtype=torch.long)
        self.blue_offsets = torch.cat(
            [torch.zeros(1, device=self.device, dtype=torch.long), torch.cumsum(b_counts_long, 0)]
        )
        self._center_max_yA = torch.empty(self.num_active_centers, device=self.device, dtype=torch.int32)
        self._center_max_L_A = torch.empty(self.num_active_centers, device=self.device, dtype=torch.int32)
        
        if self.verbose:
            print(f"         Red CSR Entries: {self.red_indices.numel()}")
            print(f"         Blue CSR Entries: {self.blue_center_indices.numel()}")

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.MB == -1).squeeze(1)
        free_r = torch.nonzero(self.MA == -1).squeeze(1)
        count = min(free_b.numel(), free_r.numel())
        if count > 0:
            self.MB[free_b[:count]] = free_r[:count].to(self.MB.dtype)
            self.MA[free_r[:count]] = free_b[:count].to(self.MA.dtype)
        if self.verbose:
            print(f"[Cleanup] Arbitrarily matched {count} remaining pairs.")

    def calculate_final_stats(self):
        if not self.verbose:
            return
        if self.metric == "L1":
            dists = torch.norm(self.P_blue - self.P_red[self.MB], p=1, dim=1)
            label = "Manhattan"
        else:
            dists = torch.norm(self.P_blue - self.P_red[self.MB], p=2, dim=1)
            label = "Euclidean"
        total_cost = dists.sum()
        avg_cost = total_cost / self.N
        print(f"Total {label} Cost: {total_cost.item():.4f}")
        print(f"Avg {label} Cost: {avg_cost.item():.4f}")

    def solve(self):
        """Public entry point. Wraps _solve_impl() with opt-in, zero-overhead-
        when-disabled peak-memory profiling (see Task 3 / NOTES_FOR_REBUTTAL.md).
        """
        if self.profile_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = self._solve_impl()
        if self.profile_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.memory_profile["solve_overall"] = (
                torch.cuda.max_memory_allocated() / 1024 ** 3
            )
        return result

    def _solve_impl(self):
        if self.verbose:
            print("[Debug] Entering solve() method...")
        # print(f"\n[Step 3] Starting Push-Relabel Loop...")
        iteration = 0
        use_cuda = self.device.type == "cuda"
        DEBUG = False
        DEBUG_EVERY = 50
        DEBUG_SYNC = False
        DEBUG_PIPE = False

        def maybe_sync():
            if use_cuda and DEBUG_SYNC:
                torch.cuda.synchronize()
        
        prev_num_free = None
        stuck_iters = 0
        prev_matched_count = None
        center_max_yA = self._center_max_yA
        center_max_L_A = self._center_max_L_A
        B_free = torch.arange(self.N, device=self.device, dtype=torch.long)

        if PROFILING:
            _prof = {
                'A_maint':     0.0,
                'B_index':     0.0,
                'B_gather':    0.0,
                'B_slack':     0.0,
                'C_free_reds': 0.0,
                'C_sort_rank': 0.0,
                'C_commit':    0.0,
                'D_relabel':   0.0,
                'iterations':  0,
            }
            def _psync():
                if use_cuda:
                    torch.cuda.synchronize()

        while True:
            # Check convergence
            num_free = B_free.numel()
            if num_free <= self.epsilon * self.N:
                # print("[Converged] Free points <= Threshold. Stopping.")
                break

            iteration += 1
            if PROFILING:
                _prof['iterations'] += 1
            if prev_num_free is not None and num_free == prev_num_free:
                stuck_iters += 1
            else:
                stuck_iters = 0
            prev_num_free = num_free
            should_dbg = DEBUG and (iteration % DEBUG_EVERY == 0 or stuck_iters >= 20)
            dbg_total_edges = 0
            dbg_eq0 = 0
            dbg_leq0 = 0
            dbg_min_slack_est = None
            dbg_max_slack_est = None
            zero_deg = 0
            if DEBUG_PIPE:
                dbg_cand_initial = 0
                dbg_after_free_u = 0
                dbg_after_free_v = 0
                dbg_after_conflict = 0
                dbg_after_commit = 0
            if self.verbose and iteration % 10 == 0:
                print(f"    [Iter {iteration}] Free: {num_free}")

            # ---------------------------------------------------------
            # A. Price Refinement (Global Update)
            # ---------------------------------------------------------
            # Calculate max(yA) per cluster
            maybe_sync()
            if PROFILING: _psync(); _t = time.perf_counter()

            yA_expanded = self.yA[self.red_indices]
            center_max_yA.zero_()
            # Scatter max: For each cluster, find highest dual weight of connected Red points
            center_max_yA.scatter_reduce_(
                0,
                self.red_expand_center_ids,
                yA_expanded,
                reduce="amax",
                include_self=False,
            )
            center_max_L_A.zero_()
            center_max_L_A.scatter_reduce_(
                0,
                self.red_expand_center_ids,
                self.red_levels,
                reduce="amax",
                include_self=False,
            )
            if PROFILING: _psync(); _prof['A_maint'] += time.perf_counter() - _t

            # ---------------------------------------------------------
            # B. Push Phase (Ragged Gather)
            # ---------------------------------------------------------
            maybe_sync()
            if PROFILING: _psync(); _t_b_index = time.perf_counter()

            # We process free Blue points in batches to manage memory
            push_batch_size = 5000 

            if should_dbg:
                starts_all = self.blue_offsets[B_free]
                ends_all = self.blue_offsets[B_free + 1]
                lengths_all = ends_all - starts_all
                dbg_total_edges = int(lengths_all.sum().item())
                zero_deg = int((lengths_all == 0).sum().item())
                if zero_deg > 0:
                    print(f"[DBG it={iteration}] zero-degree free points: {zero_deg}/{num_free}")
            else:
                starts_all = self.blue_offsets[B_free]
                ends_all = self.blue_offsets[B_free + 1]
                lengths_all = ends_all - starts_all
            total_edges_all = int(lengths_all.sum().item())

            if total_edges_all == 0:
                self.yB[B_free] += 1
                win_b = torch.empty(0, device=self.device, dtype=torch.long)
                win_c = torch.empty(0, device=self.device, dtype=torch.long)
                win_l_b = torch.empty(0, device=self.device, dtype=torch.long)
            elif num_free <= push_batch_size:
                chunk = B_free
                starts = starts_all
                lengths = lengths_all

                active_b_ids = torch.repeat_interleave(chunk, lengths)
                cum_len = torch.cumsum(lengths, 0)
                segment_starts_packed = cum_len - lengths
                global_range = _ensure_long_arange(self, "_edge_arange_buf", total_edges_all, self.device)
                repeat_packed_starts = torch.repeat_interleave(segment_starts_packed, lengths)
                offsets = global_range - repeat_packed_starts

                repeat_starts = torch.repeat_interleave(starts, lengths)
                active_edge_indices = repeat_starts + offsets
                if PROFILING: _psync(); _prof['B_index'] += time.perf_counter() - _t_b_index; _t_b_gather = time.perf_counter()

                active_c_ids = self.blue_center_indices[active_edge_indices]
                active_b_levels = self.blue_levels[active_edge_indices]
                if PROFILING: _psync(); _prof['B_gather'] += time.perf_counter() - _t_b_gather; _t_b_slack = time.perf_counter()

                slacks_est = (
                    2 * active_b_levels
                    - self.yB[active_b_ids]
                    - center_max_yA[active_c_ids]
                )
                active_max_L_A = center_max_L_A[active_c_ids]
                slacks_high = (
                    2 * torch.maximum(active_b_levels, active_max_L_A)
                    - self.yB[active_b_ids]
                    - center_max_yA[active_c_ids]
                )

                if should_dbg:
                    dbg_eq0 += int((slacks_est == 0).sum().item())
                    dbg_leq0 += int((slacks_est <= 0).sum().item())
                    local_min = int(slacks_est.min().item())
                    local_max = int(slacks_est.max().item())
                    if dbg_min_slack_est is None or local_min < dbg_min_slack_est:
                        dbg_min_slack_est = local_min
                    if dbg_max_slack_est is None or local_max > dbg_max_slack_est:
                        dbg_max_slack_est = local_max

                candidates = (slacks_est <= 0) & (slacks_high >= 0)
                if PROFILING: _psync(); _prof['B_slack'] += time.perf_counter() - _t_b_slack
                if candidates.any():
                    win_b = active_b_ids[candidates]
                    win_c = active_c_ids[candidates]
                    win_l_b = active_b_levels[candidates]
                else:
                    win_b = torch.empty(0, device=self.device, dtype=torch.long)
                    win_c = torch.empty(0, device=self.device, dtype=torch.long)
                    win_l_b = torch.empty(0, device=self.device, dtype=torch.long)
            else:
                all_win_b = []
                all_win_c = []
                all_win_l_b = []

                for i in range(0, num_free, push_batch_size):
                    chunk = B_free[i : i + push_batch_size]
                    
                    # Get CSR ranges for these points
                    starts = self.blue_offsets[chunk]
                    ends = self.blue_offsets[chunk + 1]
                    lengths = ends - starts
                    total_edges = int(lengths.sum().item())
                    
                    if total_edges == 0:
                        self.yB[chunk] += 1 # Relabel disconnected points
                        continue

                    # Vectorized ragged access via repeat_interleave
                    # 1. Expand Point IDs to match edges
                    active_b_ids = torch.repeat_interleave(chunk, lengths)
                    
                    # 2. Compute indices into flat CSR arrays
                    # offset within segment = range(len)
                    cum_len = torch.cumsum(lengths, 0)
                    segment_starts_packed = cum_len - lengths
                    global_range = _ensure_long_arange(self, "_edge_arange_buf", total_edges, self.device)
                    repeat_packed_starts = torch.repeat_interleave(segment_starts_packed, lengths)
                    offsets = global_range - repeat_packed_starts
                    
                    repeat_starts = torch.repeat_interleave(starts, lengths)
                    active_edge_indices = repeat_starts + offsets
                    
                    # 3. Gather Edge Attributes
                    active_c_ids = self.blue_center_indices[active_edge_indices]
                    active_b_levels = self.blue_levels[active_edge_indices]

                    # 4. Slack Check
                    # Condition: Slack <= 0
                    # Slack_Lower_Bound = 2 * Level - yB - Max_yA
                    slacks_est = (
                        2 * active_b_levels 
                        - self.yB[active_b_ids] 
                        - center_max_yA[active_c_ids]
                    )
                    active_max_L_A = center_max_L_A[active_c_ids]
                    slacks_high = (
                        2 * torch.maximum(active_b_levels, active_max_L_A)
                        - self.yB[active_b_ids]
                        - center_max_yA[active_c_ids]
                    )

                    if should_dbg:
                        dbg_eq0 += int((slacks_est == 0).sum().item())
                        dbg_leq0 += int((slacks_est <= 0).sum().item())
                        local_min = int(slacks_est.min().item())
                        local_max = int(slacks_est.max().item())
                        if dbg_min_slack_est is None or local_min < dbg_min_slack_est:
                            dbg_min_slack_est = local_min
                        if dbg_max_slack_est is None or local_max > dbg_max_slack_est:
                            dbg_max_slack_est = local_max

                    candidates = (slacks_est <= 0) & (slacks_high >= 0)
                    
                    if candidates.any():
                        all_win_b.append(active_b_ids[candidates])
                        all_win_c.append(active_c_ids[candidates])
                        all_win_l_b.append(active_b_levels[candidates])
                
                # Combine candidates from all batches
                if not all_win_b:
                    win_b = torch.empty(0, device=self.device, dtype=torch.long)
                    win_c = torch.empty(0, device=self.device, dtype=torch.long)
                    win_l_b = torch.empty(0, device=self.device, dtype=torch.long)
                else:
                    win_b = torch.cat(all_win_b)
                    win_c = torch.cat(all_win_c)
                    win_l_b = torch.cat(all_win_l_b)
            if should_dbg and dbg_min_slack_est is None:
                dbg_min_slack_est = 0
                dbg_max_slack_est = 0
            if DEBUG_PIPE:
                dbg_cand_initial = int(win_b.numel())
                dbg_after_free_u = int(win_b.numel())
                dbg_after_free_v = int(win_b.numel())
            
            # ---------------------------------------------------------
            # C. Resolve Phase (Conflict Resolution)
            # ---------------------------------------------------------
            b_match = None
            if win_b.numel() != 0:
                # Identify free Red points
                if PROFILING: _psync(); _t_c = time.perf_counter()
                free_red_mask = self.MA[self.red_indices] == -1
                red_ids = self.red_indices[free_red_mask]
                red_levels = self.red_levels[free_red_mask]
                red_c_ids = self.red_expand_center_ids[free_red_mask]
                if PROFILING: _psync(); _prof['C_free_reds'] += time.perf_counter() - _t_c; _t_c_sr = time.perf_counter()

                if red_ids.numel() != 0:
                    # Sort candidates by (Center, Level) to match Red's sorting
                    max_level = torch.maximum(win_l_b.max(), red_levels.max()).to(torch.long)
                    key_scale = max_level + 1

                    b_sort_key = (win_c.to(torch.long) * key_scale) + win_l_b.to(torch.long)
                    b_perm = torch.argsort(b_sort_key)

                    b_sorted = win_b[b_perm]
                    c_sorted = win_c[b_perm]
                    l_b_sorted = win_l_b[b_perm]

                    # Match available Reds to candidate Blues per Center
                    # Since both arrays are sorted by Center, we can use counts/offsets
                    num_centers = self.num_active_centers

                    b_counts = torch.bincount(c_sorted, minlength=num_centers)
                    r_counts = torch.bincount(red_c_ids, minlength=num_centers)
                    limits = torch.minimum(b_counts, r_counts) # Min available

                    # Calculate ranks to pair 1-to-1
                    b_offsets_scan = _ensure_zero_long_buffer(
                        self, "_resolve_b_offsets_buf", num_centers + 1, self.device
                    )
                    b_offsets_scan[1:] = torch.cumsum(b_counts, 0)

                    r_offsets_scan = _ensure_zero_long_buffer(
                        self, "_resolve_r_offsets_buf", num_centers + 1, self.device
                    )
                    r_offsets_scan[1:] = torch.cumsum(r_counts, 0)

                    b_rank = _ensure_long_arange(
                        self, "_resolve_b_rank_arange_buf", c_sorted.numel(), self.device
                    ) - b_offsets_scan[c_sorted]
                    r_rank = _ensure_long_arange(
                        self, "_resolve_r_rank_arange_buf", red_c_ids.numel(), self.device
                    ) - r_offsets_scan[red_c_ids]

                    b_keep = b_rank < limits[c_sorted]
                    r_keep = r_rank < limits[red_c_ids]
                    if PROFILING: _psync(); _prof['C_sort_rank'] += time.perf_counter() - _t_c_sr; _t_c_commit = time.perf_counter()

                    b_final = b_sorted[b_keep]
                    l_b_final = l_b_sorted[b_keep]
                    r_final = red_ids[r_keep]
                    l_r_final = red_levels[r_keep]

                    # Exact Slack Check (now that we have pairs)
                    # Slack = 2 * max(L_b, L_a) - yB - yA
                    # Note: We must pair them 1-to-1. The filtering above ensures equal counts per center.
                    # However, strictly speaking, we need to ensure the *ordering* aligns the lowest slack pairs.
                    # Since we sorted by level, we are pairing lowest levels first.

                    pair_count = min(b_final.numel(), r_final.numel())
                    if pair_count > 0:
                        b_final = b_final[:pair_count]
                        r_final = r_final[:pair_count]
                        l_b_final = l_b_final[:pair_count]
                        l_r_final = l_r_final[:pair_count]
                        if DEBUG_PIPE:
                            dbg_after_conflict = int(b_final.numel())

                        prev_match_count = None
                        if DEBUG_PIPE:
                            prev_match_count = int((self.MA != -1).sum().item())
                        y_b = self.yB[b_final]
                        y_a = self.yA[r_final]

                        slack = 2 * torch.maximum(l_b_final, l_r_final) - y_b - y_a
                        valid = slack == 0

                        b_match = b_final[valid]
                        r_match = r_final[valid]

                        if b_match.numel() != 0:
                            self.MB[b_match] = r_match.to(self.MB.dtype)
                            self.MA[r_match] = b_match.to(self.MA.dtype)
                        if DEBUG_PIPE:
                            new_count = int((self.MA != -1).sum().item())
                            dbg_after_commit = new_count - prev_match_count
                    if PROFILING: _psync(); _prof['C_commit'] += time.perf_counter() - _t_c_commit

            # ---------------------------------------------------------
            # D. Relabel Phase
            # ---------------------------------------------------------
            # Increase potential of unmatched Blue points to help them find edges
            if PROFILING: _psync(); _t_d = time.perf_counter()
            if b_match is not None and b_match.numel() != 0:
                keep_free_mask = _ensure_bool_buffer(
                    self, "_keep_free_mask_buf", num_free, self.device
                )
                keep_free_mask.fill_(True)
                keep_free_mask[torch.searchsorted(B_free, b_match)] = False
                still_free = B_free[keep_free_mask]
            else:
                still_free = B_free
            self.yB[still_free] += 1
            
            # Decrease potential of matched Red points to maintain constraints
            matched_r = torch.nonzero(self.MA != -1).squeeze(1)
            self.yA[matched_r] -= 1
            B_free = still_free
            if PROFILING: _psync(); _prof['D_relabel'] += time.perf_counter() - _t_d
            matched_now = self.N - still_free.numel()
            if should_dbg:
                delta_matched = 0 if prev_matched_count is None else matched_now - prev_matched_count
                print(
                    f"[DBG it={iteration}] free={num_free} stuck={stuck_iters} edges={dbg_total_edges} "
                    f"zeroDeg={zero_deg} eq0={dbg_eq0} leq0={dbg_leq0} "
                    f"minSlack={dbg_min_slack_est} maxSlack={dbg_max_slack_est} "
                    f"matched={matched_now}/{self.N} deltaMatched={delta_matched}"
                )
            if DEBUG_PIPE and stuck_iters >= 20:
                print(
                    f"[PIPE it={iteration}] init={dbg_cand_initial} freeU={dbg_after_free_u} "
                    f"freeV={dbg_after_free_v} resolved={dbg_after_conflict} "
                    f"committed={dbg_after_commit}"
                )
            prev_matched_count = matched_now

            if iteration > 50000:
                # print("Max Iterations Reached.")
                break

        if PROFILING:
            self._prof = _prof
        self.cleanup_remaining_points()
        if self.verbose:
            print(f"Matched: {(self.MB != -1).sum().item()}/{self.N}")
        self.calculate_final_stats()


def solve_bipartite_matching(x, y, epsilon, k=4, batch_size=None, metric="L2"):
    """
    Solves min-cost bipartite matching using K-level clustered push-relabel.

    Finds a minimum weight perfect matching between a source and target point
    cloud based on the specified distance metric. For scales k >= 2, routes 
    assignments through a spatial hierarchy.

    Args:
        x (torch.Tensor): Source point cloud of shape (N, D).
        y (torch.Tensor): Target point cloud of shape (N, D).
        epsilon (float): Discretization / stopping threshold parameter.
        k (int, optional): Number of hierarchy levels. Defaults to 4.
        batch_size (int, optional): GPU batch size for clustering. Defaults to None.
        metric (str, optional): Distance metric ("L2" or "L1"). Defaults to "L2".

    Returns:
        dict: A dictionary containing the 'assignment_vector' (torch.Tensor) 
              mapping each target point to a source point.
    """
    if k is None or k < 2:
        solver = TwoLevelBipartiteSolver(x, y, epsilon, batch_size=batch_size, metric=metric)
    else:
        solver = KLevelBipartiteSolver(x, y, epsilon, k=k, batch_size=batch_size, metric=metric)
    solver.solve()
    return {"assignment_vector": solver.MB}
