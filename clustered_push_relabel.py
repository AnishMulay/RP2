import torch
import math
import time
import sys
import gc

# ==========================================
# PART 1: LOW-LEVEL KERNELS (GPU)
# ==========================================

class TiledEuclideanKernel:
    """
    Computes distances ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    without materializing the full N x N matrix.
    """
    def __init__(self, chunk_size=1024):
        self.chunk_size = chunk_size

    def prepare_workspace(self, P):
        return {
            "P": P,
            "P_T": P.t(),
            "P_norms_sq": (P ** 2).sum(dim=1, keepdim=True)
        }

    def compute_dist_tile(self, query_points, workspace):
        P = workspace["P"]
        P_norms_sq = workspace["P_norms_sq"]
        Q_norms_sq = (query_points ** 2).sum(dim=1, keepdim=True).t()
        dists_sq = P_norms_sq + Q_norms_sq
        dists_sq.addmm_(P, query_points.t(), beta=1.0, alpha=-2.0)
        return torch.clamp(dists_sq, min=0.0)

    def compute_squared_dist_tile(self, query_points, workspace):
        return self.compute_dist_tile(query_points, workspace)

class TiledManhattanKernel:
    """
    Computes L1 (Manhattan) distances using torch.cdist.
    """
    def __init__(self, chunk_size=1024):
        self.chunk_size = chunk_size

    def prepare_workspace(self, P):
        return {"P": P}

    def compute_dist_tile(self, query_points, workspace):
        dists = torch.cdist(query_points, workspace["P"], p=1)
        # Match TiledEuclideanKernel's (P, query) layout.
        return dists.t()

# ==========================================
# PART 2: GEOMETRIC CLUSTERING (Voronoi + Shells)
# ==========================================

class FastGPUClustering:
    """
    Optimized Clustering using Voronoi constraints and minimal shells.
    Generates a sparse O(N^1.5) cover instead of a dense one.
    """
    def __init__(self, epsilon, batch_size=1024, micro_batch_size=32, metric="L2"):
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.metric = metric
        if metric == "L1":
            self.kernel = TiledManhattanKernel(chunk_size=batch_size)
        else:
            self.kernel = TiledEuclideanKernel(chunk_size=batch_size)

    def _sample_landmarks(self, n, device):
        # Sample ~sqrt(N) landmarks
        prob = n ** (-0.5)
        mask = torch.rand(n, device=device) < prob
        if not mask.any(): mask[torch.randint(0, n, (1,), device=device)] = True
        return torch.nonzero(mask).squeeze(1), mask

    def _compute_voronoi_bounds(self, targets, landmarks, workspace):
        # D[x] = min d(x, s) for s in landmarks
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
        """
        Builds the cluster cover using the Sampled/Non-Sampled split.
        Returns (Center, Level, Point) triplets on CPU.
        """
        n_centers = centers.shape[0]
        n_targets = targets_ws["P"].shape[0]
        
        chunk_center_ids = []
        chunk_level_ids = []
        chunk_point_ids = []

        # We process centers in batches
        for start_q in range(0, n_centers, self.batch_size):
            end_q = min(start_q + self.batch_size, n_centers)
            q_batch = centers[start_q:end_q]
            curr_bs = q_batch.shape[0]
            
            # Identify which centers in this batch are landmarks
            is_landmark_batch = center_mask_P1[start_q:end_q].view(curr_bs, 1)

            for start_mb in range(0, curr_bs, self.micro_batch_size):
                end_mb = min(start_mb + self.micro_batch_size, curr_bs)
                q_micro = q_batch[start_mb:end_mb]
                micro_is_landmark = is_landmark_batch[start_mb:end_mb]
                
                # 1. Compute Distances (Micro x Targets)
                dists = self.kernel.compute_dist_tile(q_micro, targets_ws).t() # (Micro, Targets)
                
                # 2. Apply Voronoi Constraint
                # If Landmark: Connect to everything (up to some reasonable bound, effectively inf here)
                # If Non-Landmark: Connect only if dist < D_voronoi[point]
                
                # Prepare Voronoi bounds for broadcast: (1, Targets)
                dv_row = D_voronoi.view(1, n_targets)
                
                # Mask Logic:
                # Landmark: True
                # Non-Landmark: dists < dv_row
                
                # Note: dists and dv_row are squared (L2) or linear (L1) distances
                mask = dists < dv_row
                mask = torch.where(micro_is_landmark, torch.tensor(True, device=mask.device), mask)
                
                if not mask.any(): 
                    continue
                
                # 3. Compute Levels for valid edges
                # Level = ceil(dist / epsilon) (dist is L1 or sqrt(L2))
                # We only compute this for the valid mask indices to save ops
                valid_indices = torch.nonzero(mask)
                
                # Extract distances for these pairs
                valid_dists_raw = dists[valid_indices[:, 0], valid_indices[:, 1]]
                if self.metric == "L2":
                    valid_dists = torch.sqrt(valid_dists_raw)
                else:
                    valid_dists = valid_dists_raw
                levels = torch.ceil(valid_dists / self.epsilon).to(torch.long)
                
                # Filter out Level 0 (self-loops with 0 cost can be tricky, set to 1 or keep 0)
                # Usually cost >= 0 is fine.
                
                # Move to CPU
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
        P_all = torch.cat([P_red, P_blue], dim=0)
        workspace = self.kernel.prepare_workspace(P_all)
        
        red_idx, red_mask = self._sample_landmarks(P_red.shape[0], P_red.device)
        blue_idx, blue_mask = self._sample_landmarks(P_blue.shape[0], P_blue.device)
        
        D_red = self._compute_voronoi_bounds(P_all, P_red[red_idx], workspace)
        D_blue = self._compute_voronoi_bounds(P_all, P_blue[blue_idx], workspace)
        
        # Build Shell Covers
        b_c, b_l, b_p = self._build_cover_bulk(P_blue, blue_mask, workspace, D_blue)
        r_c, r_l, r_p = self._build_cover_bulk(P_red, red_mask, workspace, D_red)
        
        return (b_c, b_l, b_p), (r_c, r_l, r_p), red_mask.cpu(), blue_mask.cpu()

# ==========================================
# PART 3: CLUSTERED SOLVER (Dynamic Costs)
# ==========================================

class GPUClusteredSolver:
    def __init__(self, P_red, P_blue, epsilon, metric="L2"):
        self.device = P_red.device
        self.N = P_red.shape[0]
        self.epsilon = epsilon
        self.P_red = P_red
        self.P_blue = P_blue
        self.metric = metric
        
        print("="*60)
        print(f"[Init] Configuration: N={self.N}, Eps={epsilon}, Metric={metric}, Device={self.device}")
        
        # 1. Clustering
        print("[Step 1] Running Geometric Clustering...")
        t0 = time.time()
        cluster_engine = FastGPUClustering(
            epsilon, batch_size=1024, micro_batch_size=32, metric=metric
        )
        blue_coo, red_coo, r_mask, b_mask = cluster_engine.run(P_red, P_blue)
        torch.cuda.synchronize()
        print(f"         Clustering done in {time.time()-t0:.2f}s")
        print(f"         Raw Blue Shells: {blue_coo[0].numel()}")
        print(f"         Raw Red Shells:  {red_coo[0].numel()}")
        
        # 2. Indexing
        print("[Step 2] Building CSR Index (Group by Center)...")
        t0 = time.time()
        self._build_csr_from_coo_cpu(blue_coo, red_coo, r_mask, b_mask)
        print(f"         Indexing done in {time.time()-t0:.2f}s")
        
        del blue_coo, red_coo, cluster_engine
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3. State Init (Integer Scaling)
        self.yA = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.yB = torch.full((self.N,), 1, device=self.device, dtype=torch.int32)
        self.MA = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)
        self.MB = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.MB == -1).squeeze(1)
        free_r = torch.nonzero(self.MA == -1).squeeze(1)
        count = min(free_b.numel(), free_r.numel())
        if count > 0:
            self.MB[free_b[:count]] = free_r[:count].to(self.MB.dtype)
            self.MA[free_r[:count]] = free_b[:count].to(self.MA.dtype)
        print(f"[Cleanup] Arbitrarily matched {count} remaining pairs.")

    def calculate_final_stats(self):
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
            bucket_keys = centers.to(torch.long) * (max_level + 1) + levels.to(torch.long)
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

        print(f"         Active Centers: {valid_centers.numel()}")

        # 3. Filter to valid centers only
        mask_valid = torch.isin(all_centers, valid_centers)
        all_centers = all_centers[mask_valid]
        all_levels = all_levels[mask_valid]
        all_points = all_points[mask_valid]
        is_red_point = is_red_point[mask_valid]

        # Map centers to dense IDs 0..K-1 for efficient offsets
        center_map = torch.searchsorted(valid_centers, all_centers)

        # 4. Build Red CSR (Grouped by Center)
        red_mask = is_red_point
        red_centers = center_map[red_mask]
        red_points = all_points[red_mask]
        red_levels = all_levels[red_mask]

        max_red_level = red_levels.max().to(torch.long)
        r_sort_key = red_centers.to(torch.long) * (max_red_level + 1) + red_levels.to(torch.long)
        perm_r = torch.argsort(r_sort_key)
        self.red_indices = red_points[perm_r].to(device=self.device, dtype=torch.int32)
        self.red_levels = red_levels[perm_r].to(device=self.device, dtype=torch.int32)
        sorted_r_centers = red_centers[perm_r]

        r_counts = torch.bincount(sorted_r_centers, minlength=valid_centers.numel())
        r_counts_i32 = r_counts.to(device=self.device, dtype=torch.int32)
        self.red_offsets = torch.cat(
            [torch.zeros(1, device=self.device, dtype=torch.int32), torch.cumsum(r_counts_i32, 0)]
        )

        # Expand Center IDs for scatter ops
        self.red_expand_center_ids = torch.repeat_interleave(
            torch.arange(valid_centers.numel(), device=self.device, dtype=torch.int32), r_counts_i32
        )

        # 5. Build Blue CSR (Inverted: Blue -> [Centers])
        blue_mask = ~is_red_point
        blue_centers = center_map[blue_mask]
        blue_points = all_points[blue_mask] - N  # Rebase Blue IDs to 0..N-1
        blue_levels = all_levels[blue_mask]

        perm_b = torch.argsort(blue_points)
        self.blue_center_indices = blue_centers[perm_b].to(device=self.device, dtype=torch.int32)
        self.blue_levels = blue_levels[perm_b].to(device=self.device, dtype=torch.int32)
        sorted_b_pts = blue_points[perm_b]

        b_counts = torch.bincount(sorted_b_pts, minlength=N)
        b_counts_i32 = b_counts.to(device=self.device, dtype=torch.int32)
        self.blue_offsets = torch.cat(
            [torch.zeros(1, device=self.device, dtype=torch.int32), torch.cumsum(b_counts_i32, 0)]
        )

        print(f"         Red Entries: {self.red_indices.numel()} (GPU)")
        print(f"         Blue Entries: {self.blue_center_indices.numel()} (GPU)")
        print(f"         Avg Degree: {(self.blue_center_indices.numel() + self.red_indices.numel())/N:.2f}")

    def solve(self):
        # print(f"\n[Step 3] Starting Push-Relabel Loop...")
        iteration = 0
        use_cuda = self.device.type == "cuda"

        def log_mem(stage):
            if not use_cuda:
                return
            alloc = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"    [Mem - {stage}] Alloc: {alloc:.1f}MB | Peak: {peak:.1f}MB")
            torch.cuda.reset_peak_memory_stats()
        
        while True:
            B_free = torch.nonzero(self.MB == -1).squeeze(1)
            num_free = B_free.numel()
            if num_free <= self.epsilon * self.N:
                # print("[Converged] Free points <= Threshold. Stopping.")
                break
            
            iteration += 1
            # log_mem("Start Iter")

            if iteration % 10 == 0:
                # print(f"    [Iter {iteration}] Free: {num_free}")
                pass

            # A. Maintenance: Max yA per Center
            if use_cuda:
                torch.cuda.synchronize()
            yA_expanded = self.yA[self.red_indices]
            center_max_yA = torch.zeros(
                len(self.red_offsets)-1, device=self.device, dtype=torch.int32
            )
            # We want to check: Slack_Est = 2*L_b - yB - Max_yA
            # So we need max(yA) in the center.
            center_max_yA.scatter_reduce_(
                0,
                self.red_expand_center_ids.to(torch.long),
                yA_expanded,
                reduce="amax",
                include_self=False,
            )
            if use_cuda:
                torch.cuda.synchronize()
            # log_mem("After Maint")
            
            # B. Push (Ragged Gather)
            if use_cuda:
                torch.cuda.synchronize()
            push_batch_size = 5000
            all_win_b = []
            all_win_c = []
            all_win_l_b = []

            starts_all = self.blue_offsets[B_free]
            ends_all = self.blue_offsets[B_free + 1]
            lengths_all = ends_all - starts_all
            total_edges = int(lengths_all.sum().item())
            if total_edges == 0:
                if use_cuda:
                    torch.cuda.synchronize()
                self.yB[B_free] += 1
                continue

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
                global_range = torch.arange(total_edges, device=self.device)
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

                candidates = slacks_est == 0
                if use_cuda:
                    torch.cuda.synchronize()
                # log_mem("After Push")

                if candidates.any():
                    all_win_b.append(active_b_ids[candidates])
                    all_win_c.append(active_c_ids[candidates])
                    all_win_l_b.append(active_b_levels[candidates])

                del (
                    repeat_starts,
                    cum_len,
                    segment_starts_packed,
                    global_range,
                    repeat_packed_starts,
                    offsets,
                    active_edge_indices,
                    active_b_ids,
                    active_c_ids,
                    active_b_levels,
                    slacks_est,
                    candidates,
                    starts,
                    ends,
                    lengths,
                    chunk,
                )

            if not all_win_b:
                win_b = torch.empty(0, device=self.device, dtype=B_free.dtype)
                win_c = torch.empty(0, device=self.device, dtype=self.blue_center_indices.dtype)
                win_l_b = torch.empty(0, device=self.device, dtype=self.blue_levels.dtype)
            else:
                win_b = torch.cat(all_win_b)
                win_c = torch.cat(all_win_c)
                win_l_b = torch.cat(all_win_l_b)
            del all_win_b, all_win_c, all_win_l_b

            if win_b.numel() != 0:
                # C. Resolve
                if use_cuda:
                    torch.cuda.synchronize()
                # We have (Blue, Center) pairs that MIGHT have a match.

                free_red_mask = self.MA[self.red_indices] == -1
                # Reds are already sorted by Center then Level.
                red_ids = self.red_indices[free_red_mask]
                red_levels = self.red_levels[free_red_mask]
                red_c_ids = self.red_expand_center_ids[free_red_mask]

                if win_b.numel() != 0 and red_ids.numel() != 0:
                    max_level = torch.maximum(win_l_b.max(), red_levels.max()).to(torch.long)
                    key_scale = max_level + 1

                    b_sort_key = win_c.to(torch.long) * key_scale + win_l_b.to(torch.long)
                    b_perm = torch.argsort(b_sort_key)
                    # log_mem("Mid-Resolve (Sort)")
                    b_sorted = win_b[b_perm]
                    c_sorted = win_c[b_perm]
                    l_b_sorted = win_l_b[b_perm]

                    r_sorted = red_ids
                    c_r_sorted = red_c_ids
                    l_r_sorted = red_levels

                    num_centers = self.red_offsets.numel() - 1
                    b_counts = torch.bincount(c_sorted, minlength=num_centers)
                    r_counts = torch.bincount(c_r_sorted, minlength=num_centers)
                    limits = torch.minimum(b_counts, r_counts)

                    b_offsets = torch.zeros(num_centers + 1, device=self.device, dtype=torch.long)
                    b_offsets[1:] = torch.cumsum(b_counts, 0)
                    r_offsets = torch.zeros(num_centers + 1, device=self.device, dtype=torch.long)
                    r_offsets[1:] = torch.cumsum(r_counts, 0)

                    b_rank = torch.arange(c_sorted.numel(), device=self.device) - b_offsets[c_sorted]
                    r_rank = torch.arange(c_r_sorted.numel(), device=self.device) - r_offsets[c_r_sorted]

                    b_keep = b_rank < limits[c_sorted]
                    r_keep = r_rank < limits[c_r_sorted]

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

                        y_b = self.yB[b_final]
                        y_a = self.yA[r_final]
                        slack = 2 * torch.maximum(l_b_final, l_r_final) - y_b - y_a
                        valid = slack == 0

                        b_match = b_final[valid]
                        r_match = r_final[valid]
                        if b_match.numel() != 0:
                            self.MB[b_match] = r_match.to(self.MB.dtype)
                            self.MA[r_match] = b_match.to(self.MA.dtype)
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
                if use_cuda:
                    torch.cuda.synchronize()
            # log_mem("After Resolve")

            # D. Relabel
            if use_cuda:
                torch.cuda.synchronize()
            still_free = torch.nonzero(self.MB == -1).squeeze(1)
            self.yB[still_free] += 1
            
            matched_r = torch.nonzero(self.MA != -1).squeeze(1)
            self.yA[matched_r] -= 1
            if use_cuda:
                torch.cuda.synchronize()
            # log_mem("After Relabel")
            if use_cuda:
                torch.cuda.empty_cache()
            
            if iteration > 50000:
                # print("Max Iterations Reached.")
                break

        self.cleanup_remaining_points()
        print(f"Matched: {(self.MB != -1).sum().item()}/{self.N}")
        self.calculate_final_stats()

# ==========================================
# PART 4: MAIN ENTRY
# ==========================================
if __name__ == "__main__":
    N = 5000
    DIM = 2
    EPS = 0.1
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    
    torch.manual_seed(42)
    P_red = torch.randn(N, DIM, device=dev)
    P_blue = torch.randn(N, DIM, device=dev)
    
    solver = GPUClusteredSolver(P_red, P_blue, EPS)
    solver.solve()
