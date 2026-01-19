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

    def compute_squared_dist_tile(self, query_points, workspace):
        P = workspace["P"]
        P_norms_sq = workspace["P_norms_sq"]
        Q_norms_sq = (query_points ** 2).sum(dim=1, keepdim=True).t()
        dists_sq = P_norms_sq + Q_norms_sq
        dists_sq.addmm_(P, query_points.t(), beta=1.0, alpha=-2.0)
        return torch.clamp(dists_sq, min=0.0)

# ==========================================
# PART 2: GEOMETRIC CLUSTERING (Voronoi + Shells)
# ==========================================

class FastGPUClustering:
    """
    Optimized Clustering using Voronoi constraints and minimal shells.
    Generates a sparse O(N^1.5) cover instead of a dense one.
    """
    def __init__(self, epsilon, batch_size=1024, micro_batch_size=32):
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
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
        D_y_sq = torch.full((n_targets,), float('inf'), device=targets.device)
        
        for i in range(0, n_landmarks, self.batch_size):
            end = min(i + self.batch_size, n_landmarks)
            batch = landmarks[i:end]
            dists = self.kernel.compute_squared_dist_tile(batch, workspace)
            batch_min, _ = dists.min(dim=1)
            D_y_sq = torch.min(D_y_sq, batch_min)
        return D_y_sq

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
                dists_sq = self.kernel.compute_squared_dist_tile(q_micro, targets_ws).t() # (Micro, Targets)
                
                # 2. Apply Voronoi Constraint
                # If Landmark: Connect to everything (up to some reasonable bound, effectively inf here)
                # If Non-Landmark: Connect only if dist < D_voronoi[point]
                
                # Prepare Voronoi bounds for broadcast: (1, Targets)
                dv_row = D_voronoi.view(1, n_targets)
                
                # Mask Logic:
                # Landmark: True
                # Non-Landmark: dists_sq < dv_row
                
                # Note: d_sq and dv_row are squared distances
                mask = dists_sq < dv_row
                mask = torch.where(micro_is_landmark, torch.tensor(True, device=mask.device), mask)
                
                if not mask.any(): 
                    continue
                
                # 3. Compute Levels for valid edges
                # Level = ceil(sqrt(dist_sq) / epsilon)
                # We only compute this for the valid mask indices to save ops
                valid_indices = torch.nonzero(mask)
                
                # Extract distances for these pairs
                valid_dists_sq = dists_sq[valid_indices[:, 0], valid_indices[:, 1]]
                valid_dists = torch.sqrt(valid_dists_sq)
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
        
        return (b_c, b_l, b_p), (r_c, r_l, r_p)

# ==========================================
# PART 3: CLUSTERED SOLVER (Dynamic Costs)
# ==========================================

class GPUClusteredSolver:
    def __init__(self, P_red, P_blue, epsilon):
        self.device = P_red.device
        self.N = P_red.shape[0]
        self.epsilon = epsilon
        self.P_red = P_red
        self.P_blue = P_blue
        
        print("="*60)
        print(f"[Init] Configuration: N={self.N}, Eps={epsilon}, Device={self.device}")
        
        # 1. Clustering
        print("[Step 1] Running Geometric Clustering...")
        t0 = time.time()
        cluster_engine = FastGPUClustering(epsilon, batch_size=1024, micro_batch_size=32)
        blue_coo, red_coo = cluster_engine.run(P_red, P_blue)
        torch.cuda.synchronize()
        print(f"         Clustering done in {time.time()-t0:.2f}s")
        print(f"         Raw Blue Shells: {blue_coo[0].numel()}")
        print(f"         Raw Red Shells:  {red_coo[0].numel()}")
        
        # 2. Indexing
        print("[Step 2] Building CSR Index (Group by Center)...")
        t0 = time.time()
        self._build_csr_from_coo_cpu(blue_coo, red_coo)
        print(f"         Indexing done in {time.time()-t0:.2f}s")
        
        del blue_coo, red_coo, cluster_engine
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3. State Init (Integer Scaling)
        self.yA = torch.zeros(self.N, device=self.device, dtype=torch.long)
        self.yB = torch.full((self.N,), 1, device=self.device, dtype=torch.long)
        self.MA = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
        self.MB = torch.full((self.N,), -1, device=self.device, dtype=torch.long)

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.MB == -1).squeeze(1)
        free_r = torch.nonzero(self.MA == -1).squeeze(1)
        count = min(free_b.numel(), free_r.numel())
        if count > 0:
            self.MB[free_b[:count]] = free_r[:count]
            self.MA[free_r[:count]] = free_b[:count]
        print(f"[Cleanup] Arbitrarily matched {count} remaining pairs.")

    def calculate_final_stats(self):
        diff = self.P_blue - self.P_red[self.MB]
        dists_sq = (diff ** 2).sum(dim=1)
        total_cost = dists_sq.sum()
        avg_cost = total_cost / self.N
        print(f"[Final Cost] Total Squared Euclidean Cost: {total_cost.item():.6f}")
        print(f"[Final Cost] Average Cost per Point: {avg_cost.item():.6f}")

    def _build_csr_from_coo_cpu(self, blue_coo, red_coo):
        """
        Builds CSR structures grouped by CENTER ID.
        Stores Levels as attributes.
        """
        b_c, b_l, b_p = blue_coo
        r_c, r_l, r_p = red_coo
        N = self.N

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

        perm_r = torch.argsort(red_centers)
        self.red_indices = red_points[perm_r].to(self.device)
        self.red_levels = red_levels[perm_r].to(self.device)
        sorted_r_centers = red_centers[perm_r]

        r_counts = torch.bincount(sorted_r_centers, minlength=valid_centers.numel())
        self.red_offsets = torch.cat([torch.tensor([0]), torch.cumsum(r_counts, 0)]).to(self.device)

        # Expand Center IDs for scatter ops
        self.red_expand_center_ids = torch.repeat_interleave(
            torch.arange(valid_centers.numel(), device=self.device), r_counts.to(self.device)
        )

        # 5. Build Blue CSR (Inverted: Blue -> [Centers])
        blue_mask = ~is_red_point
        blue_centers = center_map[blue_mask]
        blue_points = all_points[blue_mask] - N  # Rebase Blue IDs to 0..N-1
        blue_levels = all_levels[blue_mask]

        perm_b = torch.argsort(blue_points)
        self.blue_center_indices = blue_centers[perm_b].to(self.device)
        self.blue_levels = blue_levels[perm_b].to(self.device)
        sorted_b_pts = blue_points[perm_b]

        b_counts = torch.bincount(sorted_b_pts, minlength=N)
        self.blue_offsets = torch.cat([torch.tensor([0]), torch.cumsum(b_counts, 0)]).to(self.device)

        print(f"         Red Entries: {self.red_indices.numel()} (GPU)")
        print(f"         Blue Entries: {self.blue_center_indices.numel()} (GPU)")
        print(f"         Avg Degree: {(self.blue_center_indices.numel() + self.red_indices.numel())/N:.2f}")

    def solve(self):
        print(f"\n[Step 3] Starting Push-Relabel Loop...")
        start_solve = time.time()
        iteration = 0
        
        while True:
            B_free = torch.nonzero(self.MB == -1).squeeze(1)
            num_free = B_free.numel()
            if num_free <= self.epsilon * self.N:
                print("[Converged] Free points <= Threshold. Stopping.")
                break
            
            iteration += 1
            if iteration % 10 == 0:
                print(f"    [Iter {iteration}] Free: {num_free}")

            # A. Maintenance: Max yA per Center
            yA_expanded = self.yA[self.red_indices]
            center_max_yA = torch.zeros(
                len(self.red_offsets)-1, device=self.device, dtype=torch.long
            )
            # We want to check: Slack_Est = 2*L_b - yB - Max_yA
            # So we need max(yA) in the center.
            center_max_yA.scatter_reduce_(
                0, self.red_expand_center_ids, yA_expanded, reduce="amax", include_self=False
            )
            
            # B. Push (Ragged Gather)
            starts = self.blue_offsets[B_free]
            ends = self.blue_offsets[B_free + 1]
            
            ranges = [torch.arange(s.item(), e.item(), device=self.device) for s, e in zip(starts, ends)]
            if not ranges:
                 self.yB[B_free] += 1
                 continue
            
            active_edge_indices = torch.cat(ranges)
            
            # Reconstruct attributes
            lengths = ends - starts
            active_b_ids = torch.repeat_interleave(B_free, lengths)
            
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
            
            candidates = slacks_est <= 0
            
            if not candidates.any():
                pass
            else:
                # C. Resolve
                # We have (Blue, Center) pairs that MIGHT have a match.
                win_b = active_b_ids[candidates]
                win_c = active_c_ids[candidates]
                win_l_b = active_b_levels[candidates]
                
                # Proposals: Store Center ID
                proposals = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
                proposals[win_b] = win_c
                
                # Also need to store Level_B for the winner to verify in loop?
                # Actually, we can look it up or just re-gather.
                # Let's re-gather in the loop for simplicity (or pass it).
                
                prop_b_active = torch.nonzero(proposals != -1).squeeze(1)
                prop_c_active = proposals[prop_b_active]
                
                perm = torch.argsort(prop_c_active)
                b_sorted = prop_b_active[perm]
                c_sorted = prop_c_active[perm]
                
                u_centers, b_counts = torch.unique_consecutive(c_sorted, return_counts=True)
                b_offsets = torch.cat([torch.tensor([0], device=self.device), b_counts.cumsum(0)])
                
                for i, cid in enumerate(u_centers):
                    cid_val = cid.item()
                    req_blues = b_sorted[b_offsets[i]:b_offsets[i+1]]
                    
                    # We need the L_b for these blues. 
                    # Optimization: Since we don't have L_b map handy here efficiently,
                    # we can iterate Reds and check against global yB.
                    # Real Condition: 2 * max(L_b, L_a) - yB - yA == 0.
                    # This depends on L_b.
                    
                    # Recover L_b:
                    # We know Blue->Center connectivity.
                    # We can use the blue_offsets to find the specific edge index again? Slow.
                    # Fast way: We passed the check `2*L_b - yB - Max_yA <= 0`.
                    # Since yB is uniform for the blue point, L_b must be the one satisfying this.
                    # But we updated proposals blindly.
                    
                    # Let's brute force valid Reds first:
                    r_start = self.red_offsets[cid_val]
                    r_end = self.red_offsets[cid_val+1]
                    reds = self.red_indices[r_start:r_end]
                    red_levels = self.red_levels[r_start:r_end]
                    
                    free_red_mask = (self.MA[reds] == -1)
                    if not free_red_mask.any(): continue
                    
                    cand_reds = reds[free_red_mask]
                    cand_red_levels = red_levels[free_red_mask]
                    cand_yA = self.yA[cand_reds]
                    
                    # Now match against requesting blues
                    # For each blue, we need a red such that Slack == 0.
                    # This requires L_b.
                    # Since we are inside Python loop, let's fetch L_b for these blues.
                    # This is the expensive part if not careful.
                    # But req_blues is small batch.
                    
                    # Hack: The Blue MUST be connected to cid_val.
                    # We can find the edge index in Blue CSR.
                    # Since Blue CSR is sorted by Blue ID (inverted)? No, it's sorted by Blue ID.
                    # blue_offsets gives range.
                    # We search for cid_val in that range.
                    # Since ranges are small (~70), linear scan is OK or simple gather.
                    
                    # Let's assume we match blindly if 2*L_b - yB - Max_yA <= 0 was strong enough?
                    # No, we need exact zero.
                    
                    # Correct matching loop:
                    for b_idx in req_blues:
                        # Find L_b for (b_idx, cid_val)
                        start = self.blue_offsets[b_idx]
                        end = self.blue_offsets[b_idx+1]
                        
                        # Indices in blue arrays
                        range_indices = torch.arange(start, end, device=self.device)
                        centers_in_range = self.blue_center_indices[range_indices]
                        
                        # Find match
                        match_idx = (centers_in_range == cid_val).nonzero()
                        if match_idx.numel() == 0: continue # Should not happen
                        
                        l_b = self.blue_levels[range_indices[match_idx[0]]]
                        y_b = self.yB[b_idx]
                        
                        # Find a compatible Red
                        # Condition: 2 * max(l_b, l_a) - y_b - y_a == 0
                        
                        # Vectorized check against cand_reds
                        # Cost = 2 * torch.maximum(l_b, cand_red_levels)
                        costs = 2 * torch.maximum(l_b, cand_red_levels)
                        slacks = costs - y_b - cand_yA
                        
                        valid_r = (slacks == 0).nonzero()
                        if valid_r.numel() > 0:
                            # Pick first one
                            r_local_idx = valid_r[0].item()
                            r_final = cand_reds[r_local_idx]
                            
                            # Execute Match
                            self.MB[b_idx] = r_final
                            self.MA[r_final] = b_idx
                            
                            # Remove this red from candidates to prevent double matching
                            # (Inefficient slice update, but robust)
                            # Better: keep mask
                            cand_yA[r_local_idx] = -99999 # Invalidate slack

            # D. Relabel
            still_free = torch.nonzero(self.MB == -1).squeeze(1)
            self.yB[still_free] += 1
            
            matched_r = torch.nonzero(self.MA != -1).squeeze(1)
            self.yA[matched_r] -= 1
            
            if iteration > 50000:
                print("Max Iterations Reached.")
                break

        self.cleanup_remaining_points()
        print(f"Algorithm Done. Total Time: {time.time()-start_solve:.2f}s")
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
    P_red = torch.randn(N, DIM, device=dev) + 2.0
    P_blue = torch.randn(N, DIM, device=dev) - 2.0
    
    solver = GPUClusteredSolver(P_red, P_blue, EPS)
    solver.solve()
