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
    Handles efficient batching and tiling.
    """
    def __init__(self, chunk_size=4096):
        self.chunk_size = chunk_size

    def prepare_workspace(self, P):
        return {
            "P": P,
            "P_T": P.t(),
            "P_norms_sq": (P ** 2).sum(dim=1, keepdim=True)
        }

    def compute_squared_dist_tile(self, query_points, workspace):
        P = workspace["P"]
        P_norms_sq = workspace["P_norms_sq"]
        Q_norms_sq = (query_points ** 2).sum(dim=1, keepdim=True).t()
        
        # result = P_norm + Q_norm - 2 P Q^T
        # Shape: (Batch_Size, N)
        dists_sq = P_norms_sq + Q_norms_sq
        dists_sq.addmm_(P, query_points.t(), beta=1.0, alpha=-2.0)
        return torch.clamp(dists_sq, min=0.0)

# ==========================================
# PART 2: MULTI-LEVEL HIERARCHICAL CLUSTERING
# ==========================================

class FastGPUMultiLevelClustering:
    """
    Implements the Multi-Level Hierarchical Clustering (Decomposition) on GPU.
    
    Logic:
    1. Hierarchical Sampling: Partition points into disjoint levels S_0, ..., S_{k-1}.
       S_{k-1} are the 'highest' level landmarks.
    2. Top-Down Sieve: Process levels from k-1 down to 0.
       - A point p connects to a center c in S_i ONLY IF dist(p, c) < dist(p, S_{>i}).
       - This enforces the Voronoi property hierarchically.
    3. Bucketing: Edges are bucketed by ceil(dist / epsilon).
    
    Optimizations:
    - Fused Kernel: Generates edges and updates the 'bound' (nearest higher-level center) in a single pass.
    - Micro-batching: Never materializes full N x N matrices.
    """
    def __init__(self, epsilon, k=4, batch_size=2048):
        self.epsilon = epsilon
        self.k = k
        self.batch_size = batch_size
        self.kernel = TiledEuclideanKernel(chunk_size=batch_size)

    def _sample_disjoint_hierarchy(self, n, device):
        """
        Assigns each point to exactly one level from 0 to k-1.
        Higher levels are rarer.
        """
        # Probability of being promoted to the next level
        prob = n ** (-1.0 / self.k)
        
        # Random scores for every point
        scores = torch.rand(n, device=device)
        
        # We determine the level by thresholds.
        # This effectively implements the "if random > prob: break" logic vectorially.
        # Level k-1 (Highest): score > prob^(k-1) ? No, simpler logic:
        # We want roughly N * prob^i points at level i if they were cumulative.
        # But here we want disjoint. 
        # Logic mirroring decomp.py:
        # Loop i from 0 to k-1. If random > prob, stop -> assign to level i.
        # Else continue.
        # Vectorized: Count how many times we "succeed" in the probability check.
        
        levels = torch.zeros(n, dtype=torch.long, device=device)
        # We perform k-1 trials.
        for _ in range(self.k - 1):
            mask = torch.rand(n, device=device) <= prob
            # If mask is True, we conceptually "advance" to potentially being in a higher level
            levels += mask.long()
            
        # levels is now 0 to k-1.
        # However, purely random might leave top levels empty for small N.
        # Force at least one point into the top level if empty.
        if (levels == self.k - 1).sum() == 0:
            levels[torch.randint(0, n, (1,), device=device)] = self.k - 1
            
        return levels

    def _process_level(self, targets, centers, center_indices, bounds_sq, workspace):
        """
        Fused Operation for a single level S_i:
        1. Compute distances from 'centers' (S_i) to 'targets'.
        2. Identify valid edges: dist_sq < bounds_sq.
        3. Update bounds_sq: new_bound = min(bound, min_dist_to_S_i).
        
        Returns:
            - Tuple of (row_indices, col_indices, costs) for edges found in this level.
            - Updated bounds_sq tensor.
        """
        n_centers = centers.shape[0]
        n_targets = targets.shape[0]
        
        # Accumulate edges for this level
        chunk_center_ids = [] # Global indices of centers
        chunk_point_ids = []  # Indices of targets
        chunk_levels = []     # Discretized costs
        
        # We need to compute the min_dist to S_i to update the bounds for S_{i-1}.
        # Initialize with current bounds (because we only shrink bounds)
        new_bounds_sq = bounds_sq.clone()
        
        # Iterate over S_i in batches
        for start in range(0, n_centers, self.batch_size):
            end = min(start + self.batch_size, n_centers)
            batch_centers = centers[start:end]
            batch_indices = center_indices[start:end]
            
            # 1. Compute Distances: (Batch, N_targets)
            # This is the memory bottleneck, handled by tiling inside kernel or here
            # Since kernel does tiling for us, we get the result. 
            # Note: We used batch_size for kernel init, so it tiles internally if needed.
            dists_sq = self.kernel.compute_squared_dist_tile(batch_centers, workspace).t()
            
            # 2. Update Global Bounds (Reduction)
            # We need the min dist from ANY center in this batch to each point
            batch_min_sq, _ = dists_sq.min(dim=0)
            new_bounds_sq = torch.minimum(new_bounds_sq, batch_min_sq)
            
            # 3. Generate Edges (Sieve)
            # Edge exists if dist_sq < bounds_sq (The OLD bounds, passed in args)
            # Bounds broadcast: (1, N)
            # Mask shape: (Batch, N)
            mask = dists_sq < bounds_sq.unsqueeze(0)
            
            if not mask.any():
                continue
                
            # Materialize sparse indices
            valid_indices = torch.nonzero(mask) # (N_edges, 2) -> [batch_row, target_col]
            
            if valid_indices.numel() > 0:
                rows = valid_indices[:, 0]
                cols = valid_indices[:, 1]
                
                # Get actual distances for valid pairs
                valid_dists_sq = dists_sq[rows, cols]
                valid_dists = torch.sqrt(valid_dists_sq)
                
                # Compute Bucket Level: ceil(dist / epsilon)
                # Ensure level >= 0. For dist=0 (self loop), level=0.
                levels = torch.ceil(valid_dists / self.epsilon).to(torch.long)
                
                # Map batch row to global center index
                global_c = batch_indices[rows]
                
                # Append to lists (move to CPU to save GPU memory)
                chunk_center_ids.append(global_c.cpu())
                chunk_point_ids.append(cols.cpu())
                chunk_levels.append(levels.cpu())
                
            # Explicitly free memory
            del dists_sq, mask, valid_indices
        
        # Consolidate edges
        if chunk_center_ids:
            edges = (torch.cat(chunk_center_ids), torch.cat(chunk_levels), torch.cat(chunk_point_ids))
        else:
            edges = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
            
        return edges, new_bounds_sq

    def run(self, P_red, P_blue):
        """
        Main execution pipeline.
        Returns COO tensors for Red and Blue edges.
        """
        P_all = torch.cat([P_red, P_blue], dim=0)
        n = P_all.shape[0]
        
        # 1. Precompute Norms for fast distance
        workspace = self.kernel.prepare_workspace(P_all)
        
        # 2. Hierarchical Sampling
        # Determine disjoint levels for all points
        levels_red = self._sample_disjoint_hierarchy(P_red.shape[0], P_red.device)
        levels_blue = self._sample_disjoint_hierarchy(P_blue.shape[0], P_blue.device)
        
        # 3. Recursive Sieve (Red Centers)
        # Bounds: distance to nearest center in higher levels. Init to Infinity.
        # We process Red and Blue centers separately to build separate graphs,
        # but they both cover P_all (all points).
        
        def build_cover(centers_source, levels_source):
            all_c, all_l, all_p = [], [], []
            
            # Initial bounds: Infinity
            bounds_sq = torch.full((n,), float('inf'), device=P_all.device)
            
            # Loop from Highest Level (k-1) down to 0
            for i in range(self.k - 1, -1, -1):
                # Identify centers at this level
                mask_i = (levels_source == i)
                if not mask_i.any():
                    continue
                
                # Get center coordinates and their indices
                # Note: indices are local to P_red or P_blue
                idx_i = torch.nonzero(mask_i).squeeze(1)
                pts_i = centers_source[idx_i]
                
                # Process Level
                (c_cpu, l_cpu, p_cpu), bounds_sq = self._process_level(
                    P_all, pts_i, idx_i, bounds_sq, workspace
                )
                
                all_c.append(c_cpu)
                all_l.append(l_cpu)
                all_p.append(p_cpu)
                
            if not all_c:
                return (torch.empty(0, dtype=torch.long),)*3
            return (torch.cat(all_c), torch.cat(all_l), torch.cat(all_p))

        # Build Red Cover (Red Centers covering All Points)
        red_coo = build_cover(P_red, levels_red)
        
        # Build Blue Cover (Blue Centers covering All Points)
        blue_coo = build_cover(P_blue, levels_blue)
        
        return blue_coo, red_coo

# ==========================================
# PART 3: CLUSTERED SOLVER (MULTI-LEVEL ADAPTER)
# ==========================================

class GPUClusteredSolver:
    def __init__(self, P_red, P_blue, epsilon, k=4):
        self.device = P_red.device
        self.N = P_red.shape[0]
        self.epsilon = epsilon
        self.P_red = P_red
        self.P_blue = P_blue
        
        print("="*60)
        print(f"[Init] Config: N={self.N}, Eps={epsilon}, Levels={k}, Device={self.device}")
        
        # 1. Multi-Level Clustering
        print("[Step 1] Running Multi-Level Hierarchical Clustering...")
        t0 = time.time()
        cluster_engine = FastGPUMultiLevelClustering(epsilon, k=k, batch_size=2048)
        blue_coo, red_coo = cluster_engine.run(P_red, P_blue)
        torch.cuda.synchronize()
        print(f"         Clustering done in {time.time()-t0:.2f}s")
        print(f"         Red Edges:  {red_coo[0].numel()}")
        print(f"         Blue Edges: {blue_coo[0].numel()}")
        
        # 2. Indexing
        print("[Step 2] Building CSR Index (Unified Center Space)...")
        t0 = time.time()
        self._build_csr_from_coo_cpu(blue_coo, red_coo)
        print(f"         Indexing done in {time.time()-t0:.2f}s")
        
        # Cleanup
        del blue_coo, red_coo, cluster_engine
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3. State Init (Integer Scaling)
        self.yA = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.yB = torch.full((self.N,), 1, device=self.device, dtype=torch.int32)
        self.MA = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)
        self.MB = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)

    def _build_csr_from_coo_cpu(self, blue_coo, red_coo):
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
        
        # 5. Build Red CSR (Points 0..N-1)
        # Group by Center
        red_mask = is_red_point
        red_centers = center_map[red_mask]
        red_points = all_points[red_mask] # Already 0..N-1
        red_levels = all_levels[red_mask]
        
        # Sort by (Center, Level)
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
        # Expansion array for scatter operations
        self.red_expand_center_ids = torch.repeat_interleave(
            torch.arange(valid_centers.numel(), device=self.device, dtype=torch.int32), r_counts_i32
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
        self.blue_center_indices = blue_centers[perm_b].to(device=self.device, dtype=torch.int32)
        self.blue_levels = blue_levels[perm_b].to(device=self.device, dtype=torch.int32)
        sorted_b_pts = blue_points[perm_b]
        
        b_counts = torch.bincount(sorted_b_pts, minlength=N)
        b_counts_i32 = b_counts.to(device=self.device, dtype=torch.int32)
        self.blue_offsets = torch.cat(
            [torch.zeros(1, device=self.device, dtype=torch.int32), torch.cumsum(b_counts_i32, 0)]
        )
        
        print(f"         Red CSR Entries: {self.red_indices.numel()}")
        print(f"         Blue CSR Entries: {self.blue_center_indices.numel()}")

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.MB == -1).squeeze(1)
        free_r = torch.nonzero(self.MA == -1).squeeze(1)
        count = min(free_b.numel(), free_r.numel())
        if count > 0:
            self.MB[free_b[:count]] = free_r[:count].to(self.MB.dtype)
            self.MA[free_r[:count]] = free_b[:count].to(self.MA.dtype)
        print(f"[Cleanup] Arbitrarily matched {count} remaining pairs.")

    def calculate_final_stats(self):
        dists = torch.norm(self.P_blue - self.P_red[self.MB], p=2, dim=1)
        total_cost = dists.sum()
        avg_cost = total_cost / self.N
        print(f"Total Euclidean Cost: {total_cost.item():.4f}")
        print(f"Avg Euclidean Cost: {avg_cost.item():.4f}")

    def solve(self):
        print(f"\n[Step 3] Starting Push-Relabel Loop...")
        iteration = 0
        use_cuda = self.device.type == "cuda"
        
        while True:
            # Check convergence
            B_free = torch.nonzero(self.MB == -1).squeeze(1)
            num_free = B_free.numel()
            if num_free <= self.epsilon * self.N:
                print("[Converged] Free points <= Threshold. Stopping.")
                break
            
            iteration += 1
            if iteration % 10 == 0:
                print(f"    [Iter {iteration}] Free: {num_free}")

            # ---------------------------------------------------------
            # A. Price Refinement (Global Update)
            # ---------------------------------------------------------
            # Calculate max(yA) per cluster
            if use_cuda: torch.cuda.synchronize()
            
            yA_expanded = self.yA[self.red_indices]
            center_max_yA = torch.zeros(
                len(self.red_offsets)-1, device=self.device, dtype=torch.int32
            )
            # Scatter max: For each cluster, find highest dual weight of connected Red points
            center_max_yA.scatter_reduce_(
                0, self.red_expand_center_ids, yA_expanded, reduce="amax", include_self=False
            )
            
            # ---------------------------------------------------------
            # B. Push Phase (Ragged Gather)
            # ---------------------------------------------------------
            if use_cuda: torch.cuda.synchronize()
            
            # We process free Blue points in batches to manage memory
            push_batch_size = 5000 
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
                global_range = torch.arange(total_edges, device=self.device)
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

                candidates = slacks_est <= 0
                
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
            
            # ---------------------------------------------------------
            # C. Resolve Phase (Conflict Resolution)
            # ---------------------------------------------------------
            if win_b.numel() != 0:
                # Identify free Red points
                free_red_mask = self.MA[self.red_indices] == -1
                red_ids = self.red_indices[free_red_mask]
                red_levels = self.red_levels[free_red_mask]
                red_c_ids = self.red_expand_center_ids[free_red_mask]

                if red_ids.numel() != 0:
                    # Sort candidates by (Center, Level) to match Red's sorting
                    max_level = torch.maximum(win_l_b.max(), red_levels.max()).to(torch.long)
                    key_scale = max_level + 1

                    b_sort_key = win_c.to(torch.long) * key_scale + win_l_b.to(torch.long)
                    b_perm = torch.argsort(b_sort_key)
                    
                    b_sorted = win_b[b_perm]
                    c_sorted = win_c[b_perm]
                    l_b_sorted = win_l_b[b_perm]

                    # Match available Reds to candidate Blues per Center
                    # Since both arrays are sorted by Center, we can use counts/offsets
                    num_centers = self.red_offsets.numel() - 1
                    
                    b_counts = torch.bincount(c_sorted, minlength=num_centers)
                    r_counts = torch.bincount(red_c_ids, minlength=num_centers)
                    limits = torch.minimum(b_counts, r_counts) # Min available

                    # Calculate ranks to pair 1-to-1
                    b_offsets_scan = torch.zeros(num_centers + 1, device=self.device, dtype=torch.long)
                    b_offsets_scan[1:] = torch.cumsum(b_counts, 0)
                    
                    r_offsets_scan = torch.zeros(num_centers + 1, device=self.device, dtype=torch.long)
                    r_offsets_scan[1:] = torch.cumsum(r_counts, 0)

                    b_rank = torch.arange(c_sorted.numel(), device=self.device) - b_offsets_scan[c_sorted]
                    r_rank = torch.arange(red_c_ids.numel(), device=self.device) - r_offsets_scan[red_c_ids]

                    b_keep = b_rank < limits[c_sorted]
                    r_keep = r_rank < limits[red_c_ids]

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

                        y_b = self.yB[b_final]
                        y_a = self.yA[r_final]
                        
                        slack = 2 * torch.maximum(l_b_final, l_r_final) - y_b - y_a
                        valid = slack == 0

                        b_match = b_final[valid]
                        r_match = r_final[valid]
                        
                        if b_match.numel() != 0:
                            self.MB[b_match] = r_match.to(self.MB.dtype)
                            self.MA[r_match] = b_match.to(self.MA.dtype)

            # ---------------------------------------------------------
            # D. Relabel Phase
            # ---------------------------------------------------------
            # Increase potential of unmatched Blue points to help them find edges
            still_free = torch.nonzero(self.MB == -1).squeeze(1)
            self.yB[still_free] += 1
            
            # Decrease potential of matched Red points to maintain constraints
            matched_r = torch.nonzero(self.MA != -1).squeeze(1)
            self.yA[matched_r] -= 1
            
            if iteration > 50000:
                print("Max Iterations Reached.")
                break

        self.cleanup_remaining_points()
        print(f"Matched: {(self.MB != -1).sum().item()}/{self.N}")
        self.calculate_final_stats()

# ==========================================
# PART 4: MAIN ENTRY
# ==========================================
if __name__ == "__main__":
    N = 10000
    DIM = 2
    EPS = 0.1
    LEVELS = 4
    
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
        print("Warning: Running on CPU. This will be slow.")

    print(f"Device: {dev}")
    
    torch.manual_seed(42)
    P_red = torch.randn(N, DIM, device=dev)
    P_blue = torch.randn(N, DIM, device=dev)
    
    solver = GPUClusteredSolver(P_red, P_blue, EPS, k=LEVELS)
    solver.solve()
