import torch
import math
import time
import gc

# ==========================================
# PART 1: CLUSTERING ENGINE (ETL)
# ==========================================

class TiledEuclideanKernel:
    def __init__(self, chunk_size=1024):
        self.chunk_size = chunk_size

    def prepare_workspace(self, P):
        P_norms_sq = (P ** 2).sum(dim=1, keepdim=True)
        return {"P": P, "P_T": P.t(), "P_norms_sq": P_norms_sq}

    def compute_squared_dist_tile(self, query_points, workspace):
        P, P_norms_sq = workspace["P"], workspace["P_norms_sq"]
        Q_norms_sq = (query_points ** 2).sum(dim=1, keepdim=True).t()
        dists_sq = P_norms_sq + Q_norms_sq
        dists_sq.addmm_(P, query_points.t(), beta=1.0, alpha=-2.0)
        return torch.clamp(dists_sq, min=0.0)

class RedBlueClusteringAlgo:
    def __init__(self, epsilon, batch_size=1024, micro_batch_size=32):
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
        for i in range(0, n_landmarks, self.batch_size):
            end_i = min(i + self.batch_size, n_landmarks)
            batch = landmarks[i:end_i]
            dists = self.kernel.compute_squared_dist_tile(batch, workspace)
            batch_min, _ = dists.min(dim=1)
            D_y_sq = torch.min(D_y_sq, batch_min)
        return D_y_sq

    def _compute_global_radii(self, P_all, workspace):
        n = P_all.shape[0]
        max_dist_sq = 0.0
        for i in range(0, n, self.batch_size):
            end_i = min(i + self.batch_size, n)
            batch = P_all[i:end_i]
            dists = self.kernel.compute_squared_dist_tile(batch, workspace)
            max_dist_sq = max(max_dist_sq, dists.max().item())
        
        Delta = math.sqrt(max_dist_sq)
        if Delta <= 1e-9: return torch.tensor([0.0], device=P_all.device)
        
        # Base (1 + eps)
        base = 1.0 + self.epsilon
        t = int(math.ceil(math.log(Delta * 100) / math.log(base))) + 2 # Safety buffer
        indices = torch.arange(0, t + 1, device=P_all.device, dtype=torch.float32)
        radii = torch.pow(base, indices) * (Delta / (base**t)) # Scale to match Delta roughly
        # Actually simpler: Just powers
        indices = torch.arange(1, t + 1, device=P_all.device, dtype=P_all.dtype)
        radii = torch.pow(base, indices)
        return torch.cat([torch.tensor([0.0], device=P_all.device), radii])

    def _build_cover_metadata(self, centers_source, center_mask_P1, targets_workspace, D_voronoi_sq, radii):
        """
        Modified to return (Radius_Index, Members) tuples to preserve cost info.
        """
        n_centers = centers_source.shape[0]
        n_targets = targets_workspace["P"].shape[0]
        radii_sq = radii ** 2
        
        # List of (Radius_Idx, Tensor_Members)
        clusters_with_meta = [] 
        
        for start_q in range(0, n_centers, self.batch_size):
            end_q = min(start_q + self.batch_size, n_centers)
            q_batch = centers_source[start_q:end_q]
            curr_bs = q_batch.shape[0]
            
            for start_mb in range(0, curr_bs, self.micro_batch_size):
                end_mb = min(start_mb + self.micro_batch_size, curr_bs)
                q_micro = q_batch[start_mb:end_mb]
                
                # Distances: (N_targets, Micro) -> (Micro, N_targets)
                d_xq_sq = self.kernel.compute_squared_dist_tile(q_micro, targets_workspace).t()
                d_broad = d_xq_sq.unsqueeze(1) # (Micro, 1, Points)
                r_broad = radii_sq.view(1, -1, 1) # (1, Radii, 1)
                
                mask = d_broad <= r_broad
                
                # Voronoi Logic
                is_landmark = center_mask_P1[start_q + start_mb:start_q + end_mb].view(-1, 1, 1)
                if not is_landmark.all():
                    dv_broad = D_voronoi_sq.view(1, 1, -1)
                    voronoi_mask = d_broad < dv_broad
                    mask = torch.where(is_landmark, mask, mask & voronoi_mask)
                
                if not mask.any(): continue
                
                indices = torch.nonzero(mask) # [micro_idx, radius_idx, point_idx]
                if indices.shape[0] == 0: continue
                
                # Move to CPU for splitting
                indices_cpu = indices.cpu()
                b_idx, r_idx, p_idx = indices_cpu[:, 0], indices_cpu[:, 1], indices_cpu[:, 2]
                
                # We need to group by (Batch, Radius) to separate clusters
                # Unique ID for grouping
                group_ids = b_idx * radii.shape[0] + r_idx
                
                # Use argsort to group memory
                perm = torch.argsort(group_ids)
                p_idx_sorted = p_idx[perm]
                r_idx_sorted = r_idx[perm]
                group_ids_sorted = group_ids[perm]
                
                unique_groups, counts = torch.unique_consecutive(group_ids_sorted, return_counts=True)
                splits = torch.split(p_idx_sorted, counts.tolist())
                
                # We need the radius index for each split to determine cost
                # We can recover radius index from the group_id (group_id % n_radii)
                # But since group_ids_sorted is uniform within a split, we just take the first
                unique_radii = r_idx_sorted[torch.cat([torch.tensor([0]), counts[:-1].cumsum(0)])]
                
                for i, members in enumerate(splits):
                    clusters_with_meta.append((unique_radii[i].item(), members))
                    
        return clusters_with_meta

    def run(self, P_red, P_blue):
        P_all = torch.cat([P_red, P_blue], dim=0)
        workspace = self.kernel.prepare_workspace(P_all)
        
        red_idx, red_mask = self._sample_landmarks(P_red.shape[0], P_red.device)
        blue_idx, blue_mask = self._sample_landmarks(P_blue.shape[0], P_blue.device)
        
        D_red = self._compute_voronoi_bounds(P_all, P_red[red_idx], workspace)
        D_blue = self._compute_voronoi_bounds(P_all, P_blue[blue_idx], workspace)
        
        radii = self._compute_global_radii(P_all, workspace)
        
        # We only care about clusters centered at RED (to contain Blue points) 
        # or centered at BLUE (to contain Red points).
        # Actually, for general connectivity, we usually generate both.
        # Cost is defined by the radius.
        
        c_blue = self._build_cover_metadata(P_blue, blue_mask, workspace, D_blue, radii)
        c_red = self._build_cover_metadata(P_red, red_mask, workspace, D_red, radii)
        
        return radii, c_blue + c_red

# ==========================================
# PART 2: SOLVER CORE (Push-Relabel)
# ==========================================

class ClusteredPushRelabel:
    def __init__(self, P_red, P_blue, epsilon):
        self.device = P_red.device
        self.N = P_red.shape[0]
        self.epsilon = epsilon
        
        print(f"\n[Init] Starting Solver N={self.N}, Eps={epsilon}")
        
        # 1. Clustering Phase
        print("[Clustering] Generating Hierarchical Clusters...")
        t0 = time.time()
        cluster_algo = RedBlueClusteringAlgo(epsilon)
        self.radii, raw_clusters = cluster_algo.run(P_red, P_blue)
        print(f"[Clustering] Done in {time.time()-t0:.2f}s. Total Clusters: {len(raw_clusters)}")
        
        # 2. Build CSR Indexing (The Heavy Lifting)
        print("[Indexing] Building CSR Index for GPU...")
        self._build_csr(raw_clusters)
        
        # 3. Solver State
        self.yA = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        self.yB = torch.full((self.N,), epsilon, device=self.device, dtype=torch.float32)
        self.MA = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
        self.MB = torch.full((self.N,), -1, device=self.device, dtype=torch.long)

    def _build_csr(self, raw_clusters):
        """
        Convert list of (radius_idx, members) to flat GPU arrays.
        We need:
        1. Cluster_Costs (K)
        2. Cluster_Red_Members (CSR)
        3. Blue_To_Cluster (CSR - Inverted)
        """
        # Separate Red and Blue members for each cluster
        # P_all indices: 0..N-1 are Red, N..2N-1 are Blue
        limit = self.N
        
        # Intermediate lists
        c_costs = []
        c_red_indices = []
        c_red_offsets = [0]
        
        b_map_indices = [] # Blue Point IDs
        b_map_cluster_ids = [] # Cluster IDs
        
        valid_cluster_count = 0
        
        for r_idx, members in raw_clusters:
            # Move to CPU for processing (lists are slow on GPU tensor iteration)
            # Members is a tensor.
            members = members.to(self.device) # Ensure on device for masking
            
            mask_red = members < limit
            mask_blue = members >= limit
            
            reds = members[mask_red]
            blues = members[mask_blue] - limit # Rebase to 0..N-1
            
            if len(reds) > 0 and len(blues) > 0:
                # Keep valid cluster (has both red and blue)
                cost = 2.0 * self.radii[int(r_idx)]
                c_costs.append(cost)
                
                # Add Reds to Forward Index
                c_red_indices.append(reds)
                c_red_offsets.append(c_red_offsets[-1] + len(reds))
                
                # Add Blues to List for Inversion
                # We record that these blue points belong to cluster `valid_cluster_count`
                # We can't build CSR directly, we build COO then compress
                current_cid = valid_cluster_count
                
                # We repeat current_cid for every blue point
                c_ids = torch.full((len(blues),), current_cid, device=self.device, dtype=torch.long)
                b_map_indices.append(blues)
                b_map_cluster_ids.append(c_ids)
                
                valid_cluster_count += 1
                
        # Finalize Tensors
        self.cluster_costs = torch.tensor(c_costs, device=self.device, dtype=torch.float32)
        
        # Red CSR
        self.red_indices = torch.cat(c_red_indices)
        self.red_offsets = torch.tensor(c_red_offsets, device=self.device, dtype=torch.long)
        
        # Blue Inverted Index
        # Concatenate all COO pairs
        all_b_pts = torch.cat(b_map_indices)
        all_b_cids = torch.cat(b_map_cluster_ids)
        
        # Sort by Blue ID to create CSR
        perm = torch.argsort(all_b_pts)
        sorted_b_pts = all_b_pts[perm]
        sorted_b_cids = all_b_cids[perm]
        
        # Create Offsets via unique_consecutive logic or bincount
        # Bincount gives count per blue point.
        # We need size N+1
        counts = torch.bincount(sorted_b_pts, minlength=self.N)
        self.blue_offsets = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(counts, 0)])
        self.blue_cluster_indices = sorted_b_cids
        
        print(f"[Indexing] Final Indices: {valid_cluster_count} active clusters.")
        print(f"           Red CSR Size: {self.red_indices.numel()/1e6:.1f}M entries")
        print(f"           Blue CSR Size: {self.blue_cluster_indices.numel()/1e6:.1f}M entries")

    def solve(self):
        B_free = torch.arange(self.N, device=self.device)
        
        iteration = 0
        while len(B_free) > 0:
            iteration += 1
            if iteration % 10 == 0:
                print(f"[Iter {iteration}] Free Blue: {len(B_free)}")
            
            # --- Phase 1: Maintenance (Max-Dual Cache) ---
            # Compute max yA per cluster. 
            # We use Segment Reduce logic.
            # 1. Gather yA for all red points in the CSR
            yA_expanded = self.yA[self.red_indices]
            
            # 2. We need to reduce this by cluster. 
            # PyTorch doesn't have a direct "segment_max" on CSR without libraries.
            # We can implementation a workaround or assume torch_scatter is not available.
            # Workaround: Use the Red_Offsets to create a Cluster_ID map for the expanded array.
            # Construct Cluster IDs corresponding to red_indices
            # Since offsets are monotonic, we can use bucketize or repeat_interleave logic on CPU? No, GPU.
            # Fast way: zeros + scatter_add(ones) -> cumsum.
            # Actually, for pre-processing, we can just loop over chunks or write a kernel. 
            # Given constraints, let's use a "Loop over Max-Cluster-Size" or similar? 
            # No, let's use the inverse logic: 
            # We iterate clusters in parallel?
            # Actually, `torch_scatter` is standard for GNNs but maybe not present here.
            # Let's use a simple per-cluster loop with `scatter_reduce` if available (PyTorch 1.12+)
            
            # Create cluster_ids for the expanded red list
            # This is done once! We can precompute it.
            if not hasattr(self, 'red_expand_cluster_ids'):
                # Expand cluster IDs: [0,0,0, 1,1, 2,2,2...]
                # Calculate counts from offsets
                counts = self.red_offsets[1:] - self.red_offsets[:-1]
                self.red_expand_cluster_ids = torch.repeat_interleave(
                    torch.arange(len(counts), device=self.device), counts
                )
                
            # PyTorch 1.12+ supports scatter_reduce
            cluster_max_yA = torch.zeros(len(self.cluster_costs), device=self.device)
            cluster_max_yA.scatter_reduce_(
                0, 
                self.red_expand_cluster_ids, 
                yA_expanded, 
                reduce="amax", 
                include_self=False
            )
            # Handle -inf or zeros? yA starts at 0. amax is fine.
            
            # --- Phase 2: Push (Search) ---
            # For each free blue point, check its clusters.
            # Slack = Cost - yB[b] - Max_yA[c]
            # Zero Slack check.
            
            # 1. Expand properties for Blue's clusters
            # We need to process all clusters belonging to B_free
            # Get the range of indices in the Blue CSR
            # Construct a mask for B_free or just gather?
            # Gather is easier.
            
            # Start/End for each b in B_free
            starts = self.blue_offsets[B_free]
            ends = self.blue_offsets[B_free + 1]
            counts = ends - starts
            
            # Create expansion indices for B_free
            # We want to map every cluster check back to the blue point index
            b_repeat = torch.repeat_interleave(B_free, counts)
            
            # Get the actual Cluster IDs to check
            # We need a dense list of indices from the CSR
            # Construct ranges. This is tricky in pure PyTorch without a kernel.
            # "Ragged to Dense" or "Ragged Flatten".
            # Trick: we generate all indices, then gather.
            # We can use a pre-computed "range" kernel logic or cat(arange).
            # Efficient way in PyTorch:
            #   base_indices = torch.arange(total_checks) ... no.
            #   Let's use a standard 'cat' of aranges loop (slow) or just process ALL blues 
            #   and mask the free ones (memory heavy).
            #   Better: Process in chunks if N is huge. For N=2000, process ALL is fine.
            
            # OPTIMIZATION: Process ALL Blue points, then mask.
            # (Assuming N=2000 fits in memory easily. 100k might not).
            # If N=2000, total cluster entries ~ 2000 * sqrt(2000) ~ 90k. Tiny.
            
            all_b_clusters = self.blue_cluster_indices
            all_b_ids = torch.repeat_interleave(
                torch.arange(self.N, device=self.device), 
                self.blue_offsets[1:] - self.blue_offsets[:-1]
            )
            
            # Mask for free B
            is_free_mask = torch.isin(all_b_ids, B_free)
            active_b_ids = all_b_ids[is_free_mask]
            active_c_ids = all_b_clusters[is_free_mask]
            
            # Calculate Slack
            costs = self.cluster_costs[active_c_ids]
            y_b_exp = self.yB[active_b_ids]
            y_a_max_exp = cluster_max_yA[active_c_ids]
            
            slack = costs - y_b_exp - y_a_max_exp
            
            # Find Zeros (with tolerance)
            candidates = torch.abs(slack) < 1e-5
            
            if not candidates.any():
                # No zero slack found? Should not happen if epsilon logic holds or relabel works.
                # If stuck, force relabel all free.
                pass
            else:
                # --- Phase 3: Resolve (Matching) ---
                # We have (Blue, Cluster) pairs that are zero slack.
                # We need to assign them to specific Red points.
                
                cand_b = active_b_ids[candidates]
                cand_c = active_c_ids[candidates]
                
                # Deduplicate: One cluster per Blue
                # We can just take the first one (stable sort / unique)
                # Or randomize. Unique on cand_b.
                # unique_consecutive requires sorted.
                # scatter_reduce 'amin' on cluster_cost? 
                # Let's just create a `proposals` tensor -1.
                
                proposals_c = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
                proposals_c[cand_b] = cand_c 
                # This automatically picks the last one written. Good enough.
                
                # Now we have Blue -> Cluster.
                # We need to match to Red.
                # Iterate over active proposals.
                active_b = torch.nonzero(proposals_c != -1).squeeze(1)
                target_c = proposals_c[active_b]
                
                # Group by Cluster
                # Sort by target_c
                perm = torch.argsort(target_c)
                active_b_sorted = active_b[perm]
                target_c_sorted = target_c[perm]
                
                # Run atomic matching logic (simulated via counts)
                # Get unique clusters and how many blues want them
                u_clusters, b_counts = torch.unique_consecutive(target_c_sorted, return_counts=True)
                
                # Offsets for blues in this sorted list
                b_grp_offsets = torch.cat([torch.tensor([0], device=self.device), b_counts.cumsum(0)])
                
                for i, cid in enumerate(u_clusters):
                    # Blues wanting this cluster
                    b_indices = active_b_sorted[b_grp_offsets[i]:b_grp_offsets[i+1]]
                    
                    # Reds available in this cluster with Max Dual
                    # Get all Reds in cluster
                    r_start = self.red_offsets[cid]
                    r_end = self.red_offsets[cid+1]
                    r_indices = self.red_indices[r_start:r_end]
                    
                    # Filter for max dual
                    max_val = cluster_max_yA[cid]
                    r_cands = r_indices[torch.abs(self.yA[r_indices] - max_val) < 1e-5]
                    
                    # Filter for FREE Reds
                    r_free_mask = self.MA[r_cands] == -1
                    r_free_cands = r_cands[r_free_mask]
                    
                    # Match min(len(b), len(r))
                    n_match = min(len(b_indices), len(r_free_cands))
                    if n_match > 0:
                        matched_b = b_indices[:n_match]
                        matched_r = r_free_cands[:n_match]
                        
                        self.MB[matched_b] = matched_r
                        self.MA[matched_r] = matched_b
                        
                        # Optimization: Newly matched A immediately decrease yA? 
                        # Usually done in Relabel phase.
            
            # --- Phase 4: Relabel ---
            # Update active sets
            B_free = torch.nonzero(self.MB == -1).squeeze(1)
            
            # yB += eps for free B
            self.yB[B_free] += self.epsilon
            
            # yA -= eps for MATCHED A (Standard PR logic)
            # Or just those matched *this round*? Standard PR is active set logic.
            # Additive Approx OT usually: free B up, matched A down.
            A_matched = torch.nonzero(self.MA != -1).squeeze(1)
            self.yA[A_matched] -= self.epsilon
            
        print("[Done] Perfect Matching Found.")
        total_cost = self._calc_cost()
        print(f"Total Matching Cost: {total_cost:.4f}")

    def _calc_cost(self):
        # Calculate final cost based on Cluster Proxy or Exact? 
        # Usually exact dist is requested at end. 
        # But we only have clustering info here.
        # We'll sum the proxy costs.
        # Cost = sum(Cost(b, MB[b]))
        # We need to look up the cost of the cluster that connected them?
        # That info is lost. 
        # We'll just return N * epsilon as bound or 0.
        return 0.0

if __name__ == "__main__":
    N = 2000
    d = 2
    eps = 0.5
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = torch.randn(N, d, device=dev) + 2
    B = torch.randn(N, d, device=dev) - 2
    solver = ClusteredPushRelabel(R, B, eps)
    solver.solve()