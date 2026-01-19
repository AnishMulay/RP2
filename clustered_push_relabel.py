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
        # Precompute norms for the 'Target' set P
        return {
            "P": P,
            "P_T": P.t(),
            "P_norms_sq": (P ** 2).sum(dim=1, keepdim=True)
        }

    def compute_squared_dist_tile(self, query_points, workspace):
        # query_points: (Batch, D)
        # workspace['P']: (N, D)
        P = workspace["P"]
        P_norms_sq = workspace["P_norms_sq"]
        
        Q_norms_sq = (query_points ** 2).sum(dim=1, keepdim=True).t() # (1, Batch)
        
        # dist = P_norm + Q_norm - 2 P @ Q.T
        dists_sq = P_norms_sq + Q_norms_sq
        dists_sq.addmm_(P, query_points.t(), beta=1.0, alpha=-2.0)
        
        return torch.clamp(dists_sq, min=0.0)

# ==========================================
# PART 2: GPU-NATIVE CLUSTERING (With Micro-Batch)
# ==========================================

class FastGPUClustering:
    """
    Optimized Clustering that outputs raw GPU tensors (COO format).
    Includes MICRO-BATCHING to prevent OOM and maximize throughput.
    """
    def __init__(self, epsilon, batch_size=2048, micro_batch_size=32):
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.kernel = TiledEuclideanKernel(chunk_size=batch_size)

    def _sample_landmarks(self, n, device):
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

    def _compute_global_radii(self, P_all, workspace):
        n = P_all.shape[0]
        max_dist_sq = 0.0
        
        # Scan for max distance
        for i in range(0, n, self.batch_size):
            end = min(i + self.batch_size, n)
            batch = P_all[i:end]
            dists = self.kernel.compute_squared_dist_tile(batch, workspace)
            current_max = dists.max().item()
            max_dist_sq = max(max_dist_sq, current_max)
            
        Delta = math.sqrt(max_dist_sq)
        if Delta <= 1e-9: return torch.tensor([0.0], device=P_all.device)
        
        # Powers of (1+eps)
        base = 1.0 + self.epsilon
        t = int(math.ceil(math.log(Delta * 100) / math.log(base))) + 2
        indices = torch.arange(1, t + 1, device=P_all.device, dtype=P_all.dtype)
        radii = torch.pow(base, indices)
        return torch.cat([torch.tensor([0.0], device=P_all.device), radii])

    def _build_cover_bulk(self, centers, center_mask_P1, targets_ws, D_voronoi, radii):
        """
        Generates membership in bulk using MICRO-BATCHING.
        Returns: (center_ids, radius_ids, point_ids) tensors.
        """
        n_centers = centers.shape[0]
        n_targets = targets_ws["P"].shape[0]
        n_radii = radii.shape[0]
        radii_sq = radii ** 2
        
        chunk_center_ids = []
        chunk_radius_ids = []
        chunk_point_ids = []
        
        r_broad = radii_sq.view(1, n_radii, 1) # (1, R, 1)

        # 1. Outer Loop: Batches of Centers
        for start_q in range(0, n_centers, self.batch_size):
            end_q = min(start_q + self.batch_size, n_centers)
            q_batch = centers[start_q:end_q]
            curr_bs = q_batch.shape[0]
            
            # 2. Inner Loop: Micro-batches (Crucial for Speed/Memory)
            for start_mb in range(0, curr_bs, self.micro_batch_size):
                end_mb = min(start_mb + self.micro_batch_size, curr_bs)
                q_micro = q_batch[start_mb:end_mb]
                
                # Distances: (Micro, N_targets) -> (Micro, 1, N_targets)
                d_xq_sq = self.kernel.compute_squared_dist_tile(q_micro, targets_ws).t()
                d_broad = d_xq_sq.unsqueeze(1) 
                
                # Mask: Distance Condition
                mask = d_broad <= r_broad
                
                # Mask: Voronoi Condition
                global_q_idx_start = start_q + start_mb
                global_q_idx_end = start_q + end_mb
                is_landmark = center_mask_P1[global_q_idx_start:global_q_idx_end].view(-1, 1, 1)
                
                if not is_landmark.all():
                    dv_broad = D_voronoi.view(1, 1, n_targets)
                    mask = torch.where(is_landmark, mask, mask & (d_broad < dv_broad))
                
                if not mask.any(): continue
                
                # Extract Triplets (local_center_idx, radius_idx, point_idx)
                indices = torch.nonzero(mask) 
                
                # Convert local center index to global center index
                local_c = indices[:, 0]
                global_c = local_c + global_q_idx_start
                
                chunk_center_ids.append(global_c)
                chunk_radius_ids.append(indices[:, 1])
                chunk_point_ids.append(indices[:, 2])
        
        if not chunk_center_ids:
            return (torch.empty(0, device=centers.device, dtype=torch.long), 
                    torch.empty(0, device=centers.device, dtype=torch.long),
                    torch.empty(0, device=centers.device, dtype=torch.long))

        return (torch.cat(chunk_center_ids), 
                torch.cat(chunk_radius_ids), 
                torch.cat(chunk_point_ids))

    def run(self, P_red, P_blue):
        P_all = torch.cat([P_red, P_blue], dim=0)
        workspace = self.kernel.prepare_workspace(P_all)
        
        red_idx, red_mask = self._sample_landmarks(P_red.shape[0], P_red.device)
        blue_idx, blue_mask = self._sample_landmarks(P_blue.shape[0], P_blue.device)
        
        D_red = self._compute_voronoi_bounds(P_all, P_red[red_idx], workspace)
        D_blue = self._compute_voronoi_bounds(P_all, P_blue[blue_idx], workspace)
        
        radii = self._compute_global_radii(P_all, workspace)
        
        # Build raw COO tensors
        b_c, b_r, b_p = self._build_cover_bulk(P_blue, blue_mask, workspace, D_blue, radii)
        r_c, r_r, r_p = self._build_cover_bulk(P_red, red_mask, workspace, D_red, radii)
        
        return radii, (b_c, b_r, b_p), (r_c, r_r, r_p)

# ==========================================
# PART 3: GPU CLUSTERED SOLVER (Optimized Indexing)
# ==========================================

class GPUClusteredSolver:
    def __init__(self, P_red, P_blue, epsilon):
        self.device = P_red.device
        self.N = P_red.shape[0]
        self.epsilon = epsilon
        
        print("="*60)
        print(f"[Init] Configuration: N={self.N}, Eps={epsilon}, Device={self.device}")
        
        # 1. GPU Clustering (With Micro-Batch)
        print("[Step 1] Running FastGPUClustering...")
        t0 = time.time()
        # Ensure micro_batch_size is passed here
        cluster_engine = FastGPUClustering(epsilon, batch_size=2048, micro_batch_size=64)
        self.radii, blue_coo, red_coo = cluster_engine.run(P_red, P_blue)
        torch.cuda.synchronize()
        print(f"         Clustering done in {time.time()-t0:.2f}s")
        
        # 2. Index Construction (Vectorized Intersection)
        print("[Step 2] Building CSR Index on GPU (Vectorized)...")
        t0 = time.time()
        self._build_csr_from_coo(blue_coo, red_coo)
        torch.cuda.synchronize()
        print(f"         Indexing done in {time.time()-t0:.2f}s")
        
        # Cleanup
        del blue_coo, red_coo, cluster_engine
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3. State Init
        self.yA = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        self.yB = torch.full((self.N,), epsilon, device=self.device, dtype=torch.float32)
        self.MA = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
        self.MB = torch.full((self.N,), -1, device=self.device, dtype=torch.long)

    def _build_csr_from_coo(self, blue_coo, red_coo):
        """
        Converts raw (Center, Radius, Point) triplets into optimized CSR structures.
        Uses fast GPU sorting and set intersection.
        """
        b_c, b_r, b_p = blue_coo
        r_c, r_r, r_p = red_coo
        N = self.N
        n_radii = self.radii.shape[0]
        
        # 1. Generate Global Cluster Keys
        # Blue Keys: (CenterID + N) * MaxRadii + RadiusID
        b_global_keys = (b_c + N) * n_radii + b_r
        # Red Keys: CenterID * MaxRadii + RadiusID
        r_global_keys = r_c * n_radii + r_r
        
        # 2. Flatten Everything
        all_keys = torch.cat([b_global_keys, r_global_keys])
        all_points = torch.cat([b_p, r_p])
        
        # 3. Identify Valid Clusters (Contain BOTH Red and Blue)
        is_red_point = all_points < N
        is_blue_point = all_points >= N
        
        red_member_keys = torch.unique(all_keys[is_red_point])
        blue_member_keys = torch.unique(all_keys[is_blue_point])
        
        # Intersection
        # For large arrays, 'isin' or sort-merge is fast.
        valid_keys = red_member_keys[torch.isin(red_member_keys, blue_member_keys)]
        
        if valid_keys.numel() == 0:
            raise ValueError("No valid clusters found. Check Epsilon.")
            
        print(f"         Found {valid_keys.numel()} valid clusters.")
        
        # 4. Filter and Map to Dense IDs
        # Keep only entries belonging to valid clusters
        is_valid_entry = torch.isin(all_keys, valid_keys)
        final_keys = all_keys[is_valid_entry]
        final_points = all_points[is_valid_entry]
        
        # Map sparse keys to 0..K-1
        final_dense_ids = torch.searchsorted(valid_keys, final_keys)
        
        # 5. Extract Costs
        valid_radii_idx = valid_keys % n_radii
        self.cluster_costs = 2.0 * self.radii[valid_radii_idx]
        
        # 6. Build Red CSR (Forward Index)
        mask_red = final_points < N
        red_dense_ids = final_dense_ids[mask_red]
        red_pt_ids = final_points[mask_red]
        
        perm_r = torch.argsort(red_dense_ids)
        self.red_indices = red_pt_ids[perm_r]
        sorted_red_ids = red_dense_ids[perm_r]
        
        r_counts = torch.bincount(sorted_red_ids, minlength=len(valid_keys))
        self.red_offsets = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(r_counts, 0)])
        
        self.red_expand_cluster_ids = torch.repeat_interleave(
            torch.arange(len(valid_keys), device=self.device), r_counts
        )
        
        # 7. Build Blue Inverted Index
        mask_blue = final_points >= N
        blue_dense_ids = final_dense_ids[mask_blue]
        blue_pt_ids = final_points[mask_blue] - N # Rebase to 0..N-1
        
        perm_b = torch.argsort(blue_pt_ids)
        self.blue_cluster_indices = blue_dense_ids[perm_b]
        sorted_blue_pts = blue_pt_ids[perm_b]
        
        b_counts = torch.bincount(sorted_blue_pts, minlength=N)
        self.blue_offsets = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(b_counts, 0)])
        
        print(f"         Red CSR: {self.red_indices.numel()} entries")
        print(f"         Blue CSR: {self.blue_cluster_indices.numel()} entries")

    def solve(self):
        print(f"\n[Step 3] Starting Push-Relabel Loop...")
        start_solve = time.time()
        
        iteration = 0
        while True:
            # Check Free Supply
            B_free = torch.nonzero(self.MB == -1).squeeze(1)
            if B_free.numel() == 0:
                break
            
            iteration += 1
            if iteration % 10 == 0:
                print(f"    [Iter {iteration}] Free: {B_free.numel()}")

            # A. Maintenance
            yA_expanded = self.yA[self.red_indices]
            cluster_max_yA = torch.zeros(len(self.cluster_costs), device=self.device)
            cluster_max_yA.scatter_reduce_(
                0, self.red_expand_cluster_ids, yA_expanded, reduce="amax", include_self=False
            )
            
            # B. Push (Search)
            starts = self.blue_offsets[B_free]
            ends = self.blue_offsets[B_free + 1]
            counts = ends - starts
            
            if counts.sum() == 0: # Stuck
                 self.yB[B_free] += self.epsilon
                 continue
            
            # Mask for ALL Blue CSR (Vectorized Gather)
            all_b_owners = torch.repeat_interleave(
                 torch.arange(self.N, device=self.device), 
                 self.blue_offsets[1:] - self.blue_offsets[:-1]
            )
            mask_free_csr = torch.isin(all_b_owners, B_free)
            
            active_c_ids = self.blue_cluster_indices[mask_free_csr]
            active_b_ids = all_b_owners[mask_free_csr]
            
            slacks = self.cluster_costs[active_c_ids] - self.yB[active_b_ids] - cluster_max_yA[active_c_ids]
            
            candidates = torch.abs(slacks) < 1e-5
            
            if not candidates.any():
                pass
            else:
                # C. Resolve
                win_b = active_b_ids[candidates]
                win_c = active_c_ids[candidates]
                
                proposals = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
                proposals[win_b] = win_c # Winner takes all (last write wins)
                
                prop_b_active = torch.nonzero(proposals != -1).squeeze(1)
                prop_c_active = proposals[prop_b_active]
                
                perm = torch.argsort(prop_c_active)
                b_sorted = prop_b_active[perm]
                c_sorted = prop_c_active[perm]
                
                u_clusters, b_counts = torch.unique_consecutive(c_sorted, return_counts=True)
                b_offsets = torch.cat([torch.tensor([0], device=self.device), b_counts.cumsum(0)])
                
                # Resolve Loop (Usually small enough for Python, can be kernelized if needed)
                for i, cid in enumerate(u_clusters):
                    cid_val = cid.item()
                    req_blues = b_sorted[b_offsets[i]:b_offsets[i+1]]
                    
                    r_start = self.red_offsets[cid_val]
                    r_end = self.red_offsets[cid_val+1]
                    reds = self.red_indices[r_start:r_end]
                    
                    max_val = cluster_max_yA[cid_val]
                    # Filter: Max Dual AND Free
                    best_reds = reds[(torch.abs(self.yA[reds] - max_val) < 1e-5) & (self.MA[reds] == -1)]
                    
                    k = min(req_blues.numel(), best_reds.numel())
                    if k > 0:
                        mb = req_blues[:k]
                        mr = best_reds[:k]
                        self.MB[mb] = mr
                        self.MA[mr] = mb

            # D. Relabel
            still_free = torch.nonzero(self.MB == -1).squeeze(1)
            self.yB[still_free] += self.epsilon
            
            matched_r = torch.nonzero(self.MA != -1).squeeze(1)
            self.yA[matched_r] -= self.epsilon
            
            if iteration > 50000:
                print("Max Iterations Reached.")
                break
                
        print(f"Algorithm Done. Total Time: {time.time()-start_solve:.2f}s")
        print(f"Matched: {(self.MB != -1).sum().item()}/{self.N}")

# ==========================================
# PART 4: MAIN ENTRY
# ==========================================
if __name__ == "__main__":
    N = 5000
    DIM = 2
    EPS = 0.01
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    
    torch.manual_seed(42)
    P_red = torch.randn(N, DIM, device=dev) + 2.0
    P_blue = torch.randn(N, DIM, device=dev) - 2.0
    
    solver = GPUClusteredSolver(P_red, P_blue, EPS)
    solver.solve()