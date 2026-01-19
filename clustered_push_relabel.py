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
# PART 2: HYBRID CLUSTERING (Compute GPU -> Store CPU)
# ==========================================

class FastGPUClustering:
    """
    Optimized Clustering that computes on GPU but accumulates results on CPU.
    This prevents VRAM OOM when the intermediate cluster cover is large.
    """
    def __init__(self, epsilon, batch_size=1024, micro_batch_size=8):
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
        n_targets = targets.shape[0]
        n_landmarks = landmarks.shape[0]
        D_y_sq = torch.full((n_targets,), float('inf'), device=targets.device)
        
        # Process in chunks to save memory
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
        
        for i in range(0, n, self.batch_size):
            end = min(i + self.batch_size, n)
            batch = P_all[i:end]
            dists = self.kernel.compute_squared_dist_tile(batch, workspace)
            current_max = dists.max().item()
            max_dist_sq = max(max_dist_sq, current_max)
            
        Delta = math.sqrt(max_dist_sq)
        if Delta <= 1e-9: return torch.tensor([0.0], device=P_all.device)
        
        base = 1.0 + self.epsilon
        t = int(math.ceil(math.log(Delta * 100) / math.log(base))) + 2
        indices = torch.arange(1, t + 1, device=P_all.device, dtype=P_all.dtype)
        radii = torch.pow(base, indices)
        return torch.cat([torch.tensor([0.0], device=P_all.device), radii])

    def _build_cover_bulk(self, centers, center_mask_P1, targets_ws, D_voronoi, radii):
        """
        Generates membership. 
        CRITICAL CHANGE: Moves indices to CPU immediately to save GPU memory.
        """
        n_centers = centers.shape[0]
        n_targets = targets_ws["P"].shape[0]
        n_radii = radii.shape[0]
        radii_sq = radii ** 2
        
        # Accumulators on CPU
        chunk_center_ids = []
        chunk_radius_ids = []
        chunk_point_ids = []
        
        r_broad = radii_sq.view(1, n_radii, 1)

        for start_q in range(0, n_centers, self.batch_size):
            end_q = min(start_q + self.batch_size, n_centers)
            q_batch = centers[start_q:end_q]
            curr_bs = q_batch.shape[0]
            
            for start_mb in range(0, curr_bs, self.micro_batch_size):
                end_mb = min(start_mb + self.micro_batch_size, curr_bs)
                q_micro = q_batch[start_mb:end_mb]
                
                # Compute on GPU
                d_xq_sq = self.kernel.compute_squared_dist_tile(q_micro, targets_ws).t()
                d_broad = d_xq_sq.unsqueeze(1) 
                
                mask = d_broad <= r_broad
                
                global_q_idx_start = start_q + start_mb
                global_q_idx_end = start_q + end_mb
                is_landmark = center_mask_P1[global_q_idx_start:global_q_idx_end].view(-1, 1, 1)
                
                if not is_landmark.all():
                    dv_broad = D_voronoi.view(1, 1, n_targets)
                    mask = torch.where(is_landmark, mask, mask & (d_broad < dv_broad))
                
                if not mask.any(): 
                    del d_xq_sq, d_broad, mask
                    continue
                
                # Extract Indices on GPU
                indices = torch.nonzero(mask) 
                
                # MOVE TO CPU IMMEDIATELY
                indices_cpu = indices.cpu()
                
                local_c = indices_cpu[:, 0]
                global_c = local_c + global_q_idx_start
                
                chunk_center_ids.append(global_c)
                chunk_radius_ids.append(indices_cpu[:, 1])
                chunk_point_ids.append(indices_cpu[:, 2])
                
                # Cleanup GPU
                del d_xq_sq, d_broad, mask, indices
        
        if not chunk_center_ids:
            return (torch.empty(0, dtype=torch.long), 
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long))

        # Cat on CPU
        return (torch.cat(chunk_center_ids), 
                torch.cat(chunk_radius_ids), 
                torch.cat(chunk_point_ids))

    def run(self, P_red, P_blue):
        # All inputs on GPU
        P_all = torch.cat([P_red, P_blue], dim=0)
        workspace = self.kernel.prepare_workspace(P_all)
        
        red_idx, red_mask = self._sample_landmarks(P_red.shape[0], P_red.device)
        blue_idx, blue_mask = self._sample_landmarks(P_blue.shape[0], P_blue.device)
        
        D_red = self._compute_voronoi_bounds(P_all, P_red[red_idx], workspace)
        D_blue = self._compute_voronoi_bounds(P_all, P_blue[blue_idx], workspace)
        
        radii = self._compute_global_radii(P_all, workspace)
        
        # Returns CPU Tensors
        b_c, b_r, b_p = self._build_cover_bulk(P_blue, blue_mask, workspace, D_blue, radii)
        r_c, r_r, r_p = self._build_cover_bulk(P_red, red_mask, workspace, D_red, radii)
        
        return radii, (b_c, b_r, b_p), (r_c, r_r, r_p)

# ==========================================
# PART 3: GPU CLUSTERED SOLVER (CPU Staging)
# ==========================================

class GPUClusteredSolver:
    def __init__(self, P_red, P_blue, epsilon):
        self.device = P_red.device
        self.N = P_red.shape[0]
        self.epsilon = epsilon
        
        print("="*60)
        print(f"[Init] Configuration: N={self.N}, Eps={epsilon}, Device={self.device}")
        
        # 1. Clustering (Hybrid)
        print("[Step 1] Running FastGPUClustering (Hybrid Mode)...")
        t0 = time.time()
        # micro_batch_size=8 ensures GPU safety during compute
        cluster_engine = FastGPUClustering(epsilon, batch_size=1024, micro_batch_size=8)
        self.radii, blue_coo, red_coo = cluster_engine.run(P_red, P_blue)
        torch.cuda.synchronize()
        print(f"         Clustering done in {time.time()-t0:.2f}s")
        print(f"         Raw Blue Entries: {blue_coo[0].numel()}")
        print(f"         Raw Red Entries:  {red_coo[0].numel()}")
        
        # 2. Index Construction (CPU Staging)
        print("[Step 2] Building CSR Index (CPU Staging)...")
        t0 = time.time()
        self._build_csr_from_coo_cpu(blue_coo, red_coo)
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

    def _build_csr_from_coo_cpu(self, blue_coo, red_coo):
        """
        Builds the Index on CPU to avoid VRAM OOM, then moves compact results to GPU.
        """
        # Unpack CPU tensors
        b_c, b_r, b_p = blue_coo
        r_c, r_r, r_p = red_coo
        N = self.N
        n_radii = self.radii.shape[0]
        
        # 1. Global Keys (CPU)
        # Blue: (Center + N) * MaxRadii + Radius
        b_global_keys = (b_c + N) * n_radii + b_r
        # Red: Center * MaxRadii + Radius
        r_global_keys = r_c * n_radii + r_r
        
        # 2. Concat on CPU (Safe)
        all_keys = torch.cat([b_global_keys, r_global_keys])
        all_points = torch.cat([b_p, r_p])
        
        # 3. Intersection Logic (CPU)
        is_red_point = all_points < N
        is_blue_point = all_points >= N
        
        red_member_keys = torch.unique(all_keys[is_red_point])
        blue_member_keys = torch.unique(all_keys[is_blue_point])
        
        # Valid = Intersection
        # isin is fast on CPU for 1D sorted/unsorted
        valid_keys = red_member_keys[torch.isin(red_member_keys, blue_member_keys)]
        
        if valid_keys.numel() == 0:
            raise ValueError("No valid clusters found.")
            
        print(f"         Valid Clusters: {valid_keys.numel()}")
        
        # 4. Filter and Dense Map (CPU)
        is_valid_entry = torch.isin(all_keys, valid_keys)
        final_keys = all_keys[is_valid_entry]
        final_points = all_points[is_valid_entry]
        
        # Searchsorted for dense IDs
        final_dense_ids = torch.searchsorted(valid_keys, final_keys)
        
        # 5. Extract Costs (Move to GPU)
        valid_radii_idx = valid_keys % n_radii
        self.cluster_costs = (2.0 * self.radii[valid_radii_idx]).to(self.device)
        
        # 6. Build Red CSR (CPU -> GPU)
        mask_red = final_points < N
        red_dense_ids = final_dense_ids[mask_red]
        red_pt_ids = final_points[mask_red]
        
        perm_r = torch.argsort(red_dense_ids)
        self.red_indices = red_pt_ids[perm_r].to(self.device)
        sorted_red_ids = red_dense_ids[perm_r]
        
        r_counts = torch.bincount(sorted_red_ids, minlength=len(valid_keys))
        self.red_offsets = torch.cat([torch.tensor([0]), torch.cumsum(r_counts, 0)]).to(self.device)
        
        self.red_expand_cluster_ids = torch.repeat_interleave(
            torch.arange(len(valid_keys), device=self.device), r_counts.to(self.device)
        )
        
        # 7. Build Blue CSR (CPU -> GPU)
        mask_blue = final_points >= N
        blue_dense_ids = final_dense_ids[mask_blue]
        blue_pt_ids = final_points[mask_blue] - N 
        
        perm_b = torch.argsort(blue_pt_ids)
        self.blue_cluster_indices = blue_dense_ids[perm_b].to(self.device)
        sorted_blue_pts = blue_pt_ids[perm_b]
        
        b_counts = torch.bincount(sorted_blue_pts, minlength=N)
        self.blue_offsets = torch.cat([torch.tensor([0]), torch.cumsum(b_counts, 0)]).to(self.device)
        
        print(f"         Red CSR Size: {self.red_indices.numel()} (GPU)")
        print(f"         Blue CSR Size: {self.blue_cluster_indices.numel()} (GPU)")


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
            
            if counts.sum() == 0: 
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
            
            if iteration == 1 or iteration % 100 == 0:
                if slacks.numel() > 0:
                    slack_min = slacks.min().item()
                    slack_max = slacks.max().item()
                    slack_mean = slacks.mean().item()
                else:
                    slack_min = float("nan")
                    slack_max = float("nan")
                    slack_mean = float("nan")
                print(f"    [Debug] Slack stats min={slack_min:.6f} max={slack_max:.6f} mean={slack_mean:.6f}")
                
                strict_zero = (torch.abs(slacks) < 1e-5).sum().item()
                one_step = (torch.abs(slacks) < self.epsilon).sum().item()
                print(f"    [Debug] Candidate counts |slack|<1e-5: {strict_zero} |slack|<eps: {one_step}")
                
                k = min(5, active_c_ids.numel())
                if k > 0:
                    c_ids = active_c_ids[:k]
                    b_ids = active_b_ids[:k]
                    costs = self.cluster_costs[c_ids]
                    duals = self.yB[b_ids] + cluster_max_yA[c_ids]
                    gaps = costs - duals
                    print("    [Debug] First active edges (cost, yB+max_yA, slack):")
                    for i in range(k):
                        print(
                            f"      c={c_ids[i].item()} b={b_ids[i].item()} "
                            f"cost={costs[i].item():.6f} sum={duals[i].item():.6f} "
                            f"slack={gaps[i].item():.6f}"
                        )

            candidates = torch.abs(slacks) < 1e-5
            
            if not candidates.any():
                pass
            else:
                # C. Resolve
                win_b = active_b_ids[candidates]
                win_c = active_c_ids[candidates]
                
                proposals = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
                proposals[win_b] = win_c 
                
                prop_b_active = torch.nonzero(proposals != -1).squeeze(1)
                prop_c_active = proposals[prop_b_active]
                
                perm = torch.argsort(prop_c_active)
                b_sorted = prop_b_active[perm]
                c_sorted = prop_c_active[perm]
                
                u_clusters, b_counts = torch.unique_consecutive(c_sorted, return_counts=True)
                b_offsets = torch.cat([torch.tensor([0], device=self.device), b_counts.cumsum(0)])
                
                # Resolve Loop
                for i, cid in enumerate(u_clusters):
                    cid_val = cid.item()
                    req_blues = b_sorted[b_offsets[i]:b_offsets[i+1]]
                    
                    r_start = self.red_offsets[cid_val]
                    r_end = self.red_offsets[cid_val+1]
                    reds = self.red_indices[r_start:r_end]
                    
                    max_val = cluster_max_yA[cid_val]
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
    EPS = 0.1
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    
    torch.manual_seed(42)
    P_red = torch.randn(N, DIM, device=dev) + 2.0
    P_blue = torch.randn(N, DIM, device=dev) - 2.0
    
    solver = GPUClusteredSolver(P_red, P_blue, EPS)
    solver.solve()
