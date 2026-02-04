#!/usr/bin/env python3
"""
Single-file implementation of the K-Level Clustered Push-Relabel algorithm 
applied to the NYC Taxi Routing Problem with Time & Speed constraints.
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd

# ==========================================
# PART 0: UTILS & CONFIG (Adpated from run.py)
# ==========================================

EARTH_RADIUS_METERS = 6_371_000.0

def _project_lonlat_to_meters(xA_deg, xB_deg, origin_lon=None, origin_lat=None):
    """
    Projects longitude/latitude to a local tangent plane (meters).
    Adapted from run.py
    """
    # Choose origin as medians if not provided
    if origin_lon is None or origin_lat is None:
        lonA, latA = xA_deg[:, 0], xA_deg[:, 1]
        lonB, latB = xB_deg[:, 0], xB_deg[:, 1]
        # Use simple mean/median of all points for stability
        all_lon = torch.cat((lonA, lonB))
        all_lat = torch.cat((latA, latB))
        lon0 = torch.median(all_lon).item()
        lat0 = torch.median(all_lat).item()
    else:
        lon0, lat0 = float(origin_lon), float(origin_lat)

    # Convert degrees to meters in local tangent plane
    deg2rad = torch.tensor(math.pi / 180.0, device=xA_deg.device, dtype=torch.float32)
    lat0_rad = torch.tensor(lat0 * (math.pi / 180.0), device=xA_deg.device, dtype=torch.float32)
    scale_x = float(EARTH_RADIUS_METERS) * torch.cos(lat0_rad)
    scale_y = float(EARTH_RADIUS_METERS)

    def _proj(coords):
        lon = coords[:, 0] - float(lon0)
        lat = coords[:, 1] - float(lat0)
        x = lon.to(dtype=torch.float32) * deg2rad * scale_x
        y = lat.to(dtype=torch.float32) * deg2rad * scale_y
        return torch.stack((x, y), dim=1)

    return _proj(xA_deg), _proj(xB_deg), lon0, lat0

def load_taxi_csv(filepath, n=None, seed=42):
    """
    Stand-in for loader.load_day. 
    Reads NYC Taxi CSV, parses datetimes, and returns tensors.
    """
    print(f"[Loader] Reading {filepath}...")
    # Standard NYC Taxi column names. Adjust if your CSV differs.
    use_cols = [
        'pickup_datetime', 'dropoff_datetime', 
        'pickup_longitude', 'pickup_latitude', 
        'dropoff_longitude', 'dropoff_latitude'
    ]
    
    # Check if file exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    # Load with pandas
    df = pd.read_csv(filepath, nrows=n * 2 if n else None, usecols=lambda c: c in use_cols)
    
    # Filter valid coordinates (NYC bounding box roughly)
    df = df[
        (df['pickup_longitude'] > -75) & (df['pickup_longitude'] < -70) &
        (df['pickup_latitude'] > 39) & (df['pickup_latitude'] < 42) &
        (df['dropoff_longitude'] > -75) & (df['dropoff_longitude'] < -70) &
        (df['dropoff_latitude'] > 39) & (df['dropoff_latitude'] < 42)
    ].copy()
    
    if n and len(df) > n:
        if seed is not None:
            df = df.sample(n=n, random_state=seed)
        else:
            df = df.iloc[:n]

    print(f"[Loader] {len(df)} trips loaded.")
    
    # Parse dates
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='mixed')
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], format='mixed')
    
    # Convert time to relative seconds (float) from the earliest time in the batch
    min_time = min(df['pickup_datetime'].min(), df['dropoff_datetime'].min())
    
    def to_seconds(series):
        return (series - min_time).dt.total_seconds().values.astype(np.float32)

    t_pickup = to_seconds(df['pickup_datetime'])
    t_dropoff = to_seconds(df['dropoff_datetime'])
    
    # Extract coords
    p_pickup = df[['pickup_longitude', 'pickup_latitude']].values.astype(np.float32)
    p_dropoff = df[['dropoff_longitude', 'dropoff_latitude']].values.astype(np.float32)
    
    return p_pickup, p_dropoff, t_pickup, t_dropoff

# ==========================================
# PART 1: KERNELS (GPU)
# ==========================================

class TiledEuclideanKernel:
    def __init__(self, chunk_size=4096):
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

# ==========================================
# PART 2: MULTI-LEVEL HIERARCHICAL CLUSTERING
# ==========================================

class FastGPUMultiLevelClustering:
    def __init__(self, epsilon, k=4, batch_size=2048):
        self.epsilon = epsilon
        self.k = k
        self.batch_size = batch_size
        self.kernel = TiledEuclideanKernel(chunk_size=batch_size)

    def _sample_disjoint_hierarchy(self, n, device):
        prob = n ** (-1.0 / self.k)
        levels = torch.zeros(n, dtype=torch.long, device=device)
        for _ in range(self.k - 1):
            mask = torch.rand(n, device=device) <= prob
            levels += mask.long()
        if (levels == self.k - 1).sum() == 0:
            levels[torch.randint(0, n, (1,), device=device)] = self.k - 1
        return levels

    def _process_level(self, targets, centers, center_indices, bounds, workspace):
        n_centers = centers.shape[0]
        chunk_center_ids = []
        chunk_point_ids = []
        chunk_levels = []
        new_bounds = bounds.clone()
        
        for start in range(0, n_centers, self.batch_size):
            end = min(start + self.batch_size, n_centers)
            batch_centers = centers[start:end]
            batch_indices = center_indices[start:end]
            dists = self.kernel.compute_dist_tile(batch_centers, workspace).t()
            batch_min, _ = dists.min(dim=0)
            new_bounds = torch.minimum(new_bounds, batch_min)
            mask = dists < bounds.unsqueeze(0)
            
            if mask.any():
                valid_indices = torch.nonzero(mask)
                rows = valid_indices[:, 0]
                cols = valid_indices[:, 1]
                valid_dists = torch.sqrt(dists[rows, cols])
                levels = torch.ceil(valid_dists / self.epsilon).to(torch.long)
                global_c = batch_indices[rows]
                chunk_center_ids.append(global_c.cpu())
                chunk_point_ids.append(cols.cpu())
                chunk_levels.append(levels.cpu())
            del dists, mask
        
        if chunk_center_ids:
            edges = (torch.cat(chunk_center_ids), torch.cat(chunk_levels), torch.cat(chunk_point_ids))
        else:
            edges = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        return edges, new_bounds

    def run(self, P_red, P_blue):
        P_all = torch.cat([P_red, P_blue], dim=0)
        n = P_all.shape[0]
        workspace = self.kernel.prepare_workspace(P_all)
        levels_red = self._sample_disjoint_hierarchy(P_red.shape[0], P_red.device)
        levels_blue = self._sample_disjoint_hierarchy(P_blue.shape[0], P_blue.device)
        
        def build_cover(centers_source, levels_source):
            all_c, all_l, all_p = [], [], []
            bounds = torch.full((n,), float('inf'), device=P_all.device)
            for i in range(self.k - 1, -1, -1):
                mask_i = (levels_source == i)
                if not mask_i.any(): continue
                idx_i = torch.nonzero(mask_i).squeeze(1)
                pts_i = centers_source[idx_i]
                (c_cpu, l_cpu, p_cpu), bounds = self._process_level(P_all, pts_i, idx_i, bounds, workspace)
                all_c.append(c_cpu)
                all_l.append(l_cpu)
                all_p.append(p_cpu)
            if not all_c: return (torch.empty(0, dtype=torch.long),)*3
            return (torch.cat(all_c), torch.cat(all_l), torch.cat(all_p))

        red_coo = build_cover(P_red, levels_red)
        blue_coo = build_cover(P_blue, levels_blue)
        return blue_coo, red_coo, levels_red, levels_blue

# ==========================================
# PART 3: CLUSTERED SOLVER (With Constraints)
# ==========================================

class GPUClusteredSolver:
    def __init__(
        self,
        P_red,
        P_blue,
        T_red,
        T_blue,
        epsilon,
        speed_mps,
        k=4,
        batch_size=2048,
        stop_threshold=0.01,
        min_free_count=0,
    ):
        self.device = P_red.device
        self.N = P_red.shape[0]
        self.epsilon = epsilon
        self.k = k
        self.speed_mps = speed_mps
        self.stop_threshold = stop_threshold
        self.min_free_count = min_free_count
        self.P_red = P_red
        self.P_blue = P_blue
        self.T_red = T_red   # Drop-off Times (Points A)
        self.T_blue = T_blue # Pick-up Times (Points B)
        self.batch_size = batch_size
        
        print("="*60)
        print(
            f"[Init] N={self.N}, Eps={epsilon}, Levels={k}, Speed={speed_mps} m/s, "
            f"StopThreshold={stop_threshold}, MinFreeCount={min_free_count}, Device={self.device}"
        )
        
        # 1. Multi-Level Clustering (Spatial Only)
        print("[Step 1] Running Spatial Multi-Level Clustering...")
        t0 = time.time()
        cluster_engine = FastGPUMultiLevelClustering(epsilon, k=k, batch_size=self.batch_size)
        blue_coo, red_coo, levels_red, levels_blue = cluster_engine.run(P_red, P_blue)
        torch.cuda.synchronize()
        print(f"         Clustering done in {time.time()-t0:.2f}s")
        
        # 2. Indexing
        print("[Step 2] Building CSR Index...")
        self._build_csr_from_coo_cpu(blue_coo, red_coo, levels_red, levels_blue)
        
        del blue_coo, red_coo, levels_red, levels_blue, cluster_engine
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3. State Init
        self.yA = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.yB = torch.full((self.N,), 1, device=self.device, dtype=torch.int32)
        self.MA = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)
        self.MB = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)
        self.stat_candidates_total = 0
        self.stat_candidates_rejected = 0

    def _build_csr_from_coo_cpu(self, blue_coo, red_coo, levels_red, levels_blue):
        # Unchanged from original except for variable names
        b_c, b_l, b_p = blue_coo
        r_c, r_l, r_p = red_coo
        N = self.N
        
        b_c_shifted = b_c + N
        all_centers = torch.cat([r_c, b_c_shifted])
        all_levels = torch.cat([r_l, b_l])
        all_points = torch.cat([r_p, b_p])
        
        is_red_point = all_points < N
        centers_with_red = torch.unique(all_centers[is_red_point])
        centers_with_blue = torch.unique(all_centers[~is_red_point])
        valid_centers = centers_with_red[torch.isin(centers_with_red, centers_with_blue)]
        
        if valid_centers.numel() == 0:
            raise ValueError("No valid clusters found (Intersection Empty).")
            
        print(f"         Active Centers: {valid_centers.numel()}")
        
        mask_valid = torch.isin(all_centers, valid_centers)
        all_centers = all_centers[mask_valid]
        all_levels = all_levels[mask_valid]
        all_points = all_points[mask_valid]
        is_red_point = is_red_point[mask_valid]
        
        center_map = torch.searchsorted(valid_centers, all_centers)
        
        # Red CSR
        red_mask = is_red_point
        red_centers = center_map[red_mask]
        red_points = all_points[red_mask]
        red_levels = all_levels[red_mask]
        
        max_red_level = red_levels.max().to(torch.long)
        r_sort_key = (red_centers.to(torch.long) * (max_red_level + 1)) + red_levels.to(torch.long)
        perm_r = torch.argsort(r_sort_key)
        
        self.red_indices = red_points[perm_r].to(device=self.device, dtype=torch.int32)
        self.red_levels = red_levels[perm_r].to(device=self.device, dtype=torch.int32)
        sorted_r_centers = red_centers[perm_r]
        
        r_counts = torch.bincount(sorted_r_centers, minlength=valid_centers.numel())
        r_counts_i32 = r_counts.to(device=self.device, dtype=torch.int32)
        self.red_offsets = torch.cat([torch.zeros(1, device=self.device, dtype=torch.int32), torch.cumsum(r_counts_i32, 0)])
        self.red_expand_center_ids = torch.repeat_interleave(
            torch.arange(valid_centers.numel(), device=self.device, dtype=torch.int32), r_counts_i32
        )
        
        # Blue CSR
        blue_mask = ~is_red_point
        blue_centers = center_map[blue_mask]
        blue_points = all_points[blue_mask] - N
        blue_levels = all_levels[blue_mask]
        
        perm_b = torch.argsort(blue_points)
        self.blue_center_indices = blue_centers[perm_b].to(device=self.device, dtype=torch.int32)
        self.blue_levels = blue_levels[perm_b].to(device=self.device, dtype=torch.int32)
        sorted_b_pts = blue_points[perm_b]
        
        b_counts = torch.bincount(sorted_b_pts, minlength=N)
        b_counts_i32 = b_counts.to(device=self.device, dtype=torch.int32)
        self.blue_offsets = torch.cat([torch.zeros(1, device=self.device, dtype=torch.int32), torch.cumsum(b_counts_i32, 0)])

    def calculate_final_stats(self):
        # Only calc cost for matches
        matched_mask = self.MB != -1
        if not matched_mask.any():
             print("No matches found.")
             return
        
        red_indices = self.MB[matched_mask].long()
        blue_indices = torch.nonzero(matched_mask).squeeze(1)
        
        dists = torch.norm(self.P_blue[blue_indices] - self.P_red[red_indices], p=2, dim=1)
        total_cost = dists.sum()
        print(f"Total Euclidean Cost: {total_cost.item():.4f}")
        print(f"Matched: {matched_mask.sum().item()}/{self.N}")

    def solve(self):
        print("[Step 3] Starting Push-Relabel Loop (with Time/Speed Constraints)...")
        iteration = 0
        use_cuda = self.device.type == "cuda"
        self.stat_candidates_total = 0
        self.stat_candidates_rejected = 0
        
        # Limit max iterations for safety
        MAX_ITER = 20000 
        
        while iteration < MAX_ITER:
            # Check convergence
            B_free = torch.nonzero(self.MB == -1).squeeze(1)
            num_free = B_free.numel()
            if num_free <= self.stop_threshold * self.N or num_free <= self.min_free_count:
                print(
                    f"[Converged] Free points {num_free} "
                    f"(stop_threshold*N={self.stop_threshold * self.N:.2f}, min_free_count={self.min_free_count})."
                )
                break
            
            iteration += 1
            if iteration % 50 == 0:
                print(
                    f"    [Iter {iteration}] Free: {num_free} | "
                    f"Speed Reject: {self.stat_candidates_rejected}/{self.stat_candidates_total}"
                )

            if use_cuda: torch.cuda.synchronize()

            # ---------------------------------------------------------
            # A. Price Refinement (Global Update)
            # ---------------------------------------------------------
            yA_expanded = self.yA[self.red_indices]
            center_max_yA = torch.zeros(len(self.red_offsets)-1, device=self.device, dtype=torch.int32)
            center_max_yA.scatter_reduce_(0, self.red_expand_center_ids.to(torch.long), yA_expanded, reduce="amax", include_self=False)
            
            center_max_L_A = torch.zeros(len(self.red_offsets)-1, device=self.device, dtype=torch.int32)
            center_max_L_A.scatter_reduce_(0, self.red_expand_center_ids.to(torch.long), self.red_levels, reduce="amax", include_self=False)
            
            # ---------------------------------------------------------
            # B. Push Phase (Find Candidates)
            # ---------------------------------------------------------
            push_batch_size = 5000 
            all_win_b, all_win_c, all_win_l_b = [], [], []

            for i in range(0, num_free, push_batch_size):
                chunk = B_free[i : i + push_batch_size]
                starts = self.blue_offsets[chunk]
                ends = self.blue_offsets[chunk + 1]
                lengths = ends - starts
                total_edges = int(lengths.sum().item())
                
                if total_edges == 0:
                    self.yB[chunk] += 1
                    continue

                active_b_ids = torch.repeat_interleave(chunk, lengths)
                cum_len = torch.cumsum(lengths, 0)
                segment_starts_packed = cum_len - lengths
                global_range = torch.arange(total_edges, device=self.device)
                repeat_packed_starts = torch.repeat_interleave(segment_starts_packed, lengths)
                offsets = global_range - repeat_packed_starts
                repeat_starts = torch.repeat_interleave(starts, lengths)
                active_edge_indices = repeat_starts + offsets
                
                active_c_ids = self.blue_center_indices[active_edge_indices]
                active_b_levels = self.blue_levels[active_edge_indices]

                # Slack Check
                slacks_est = (2 * active_b_levels - self.yB[active_b_ids] - center_max_yA[active_c_ids])
                active_max_L_A = center_max_L_A[active_c_ids]
                slacks_high = (2 * torch.maximum(active_b_levels, active_max_L_A) - self.yB[active_b_ids] - center_max_yA[active_c_ids])

                candidates = (slacks_est <= 0) & (slacks_high >= 0)
                if candidates.any():
                    all_win_b.append(active_b_ids[candidates])
                    all_win_c.append(active_c_ids[candidates])
                    all_win_l_b.append(active_b_levels[candidates])
            
            if not all_win_b:
                win_b = torch.empty(0, device=self.device, dtype=torch.long)
                win_c = torch.empty(0, device=self.device, dtype=torch.long)
                win_l_b = torch.empty(0, device=self.device, dtype=torch.long)
            else:
                win_b = torch.cat(all_win_b)
                win_c = torch.cat(all_win_c)
                win_l_b = torch.cat(all_win_l_b)
            
            # ---------------------------------------------------------
            # C. Resolve Phase (Conflict Resolution + CONSTRAINT CHECK)
            # ---------------------------------------------------------
            if win_b.numel() != 0:
                free_red_mask = self.MA[self.red_indices] == -1
                red_ids = self.red_indices[free_red_mask]
                red_levels = self.red_levels[free_red_mask]
                red_c_ids = self.red_expand_center_ids[free_red_mask]

                if red_ids.numel() != 0:
                    max_level = torch.maximum(win_l_b.max(), red_levels.max()).to(torch.long)
                    key_scale = max_level + 1

                    b_sort_key = (win_c.to(torch.long) * key_scale) + win_l_b.to(torch.long)
                    b_perm = torch.argsort(b_sort_key)
                    b_sorted = win_b[b_perm]
                    c_sorted = win_c[b_perm]
                    l_b_sorted = win_l_b[b_perm]

                    num_centers = self.red_offsets.numel() - 1
                    b_counts = torch.bincount(c_sorted, minlength=num_centers)
                    r_counts = torch.bincount(red_c_ids, minlength=num_centers)
                    limits = torch.minimum(b_counts, r_counts)

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
                        
                        # === CRITICAL MODIFICATION: TIME & SPEED CONSTRAINT ===
                        if b_match.numel() > 0:
                            proposed_pairs = int(b_match.numel())
                            self.stat_candidates_total += proposed_pairs

                            # Calculate Euclidean Distance for candidates
                            # Shape (K, 2)
                            p_b_coords = self.P_blue[b_match]
                            p_r_coords = self.P_red[r_match]
                            
                            dist_m = torch.norm(p_b_coords - p_r_coords, dim=1)
                            
                            # Time required to travel
                            travel_time_sec = dist_m / self.speed_mps
                            
                            # Valid if: Pickup_Time >= Dropoff_Time + Travel_Time
                            # This naturally prevents going to past, because Travel_Time > 0
                            # and if Pickup < Dropoff, LHS < RHS is definitely True (Fail)
                            
                            t_pickup = self.T_blue[b_match]
                            t_dropoff = self.T_red[r_match]
                            
                            is_reachable = t_pickup >= (t_dropoff + travel_time_sec)
                            rejected_pairs = proposed_pairs - int(is_reachable.sum().item())
                            self.stat_candidates_rejected += rejected_pairs
                            
                            # Filter
                            b_match = b_match[is_reachable]
                            r_match = r_match[is_reachable]

                            if b_match.numel() != 0:
                                self.MB[b_match] = r_match.to(self.MB.dtype)
                                self.MA[r_match] = b_match.to(self.MA.dtype)

            # ---------------------------------------------------------
            # D. Relabel Phase
            # ---------------------------------------------------------
            still_free = torch.nonzero(self.MB == -1).squeeze(1)
            self.yB[still_free] += 1
            
            matched_r = torch.nonzero(self.MA != -1).squeeze(1)
            self.yA[matched_r] -= 1

        print(f"Loop finished. Matched: {(self.MB != -1).sum().item()}/{self.N}")
        self.calculate_final_stats()
        return self.MB

# ==========================================
# PART 4: EXPERIMENT RUNNER
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="NYC Taxi Clustered Push-Relabel Experiment")
    parser.add_argument("--input", type=str, required=True, help="Path to NYC taxi CSV")
    parser.add_argument("--n", type=int, default=10000, help="Number of trips to load")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--k-levels", type=int, default=4, help="Hierarchy levels")
    parser.add_argument("--epsilon", type=float, default=100.0, help="Bucket width (meters approx)")
    parser.add_argument("--stop-threshold", type=float, default=0.01, help="Convergence threshold as a fraction of N")
    parser.add_argument("--min-free-count", type=int, default=0, help="Converge when remaining free points <= this count")
    parser.add_argument("--speed-mps", type=float, default=8.0, help="Avg taxi speed in m/s")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    
    args = parser.parse_args()
    
    # 1. Setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running Experiment on {device}")
    print(f"Config: {args}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 2. Data Loading & Projection
    # Note: P_red = Dropoff, P_blue = Pickup
    # Taxi Logic: A taxi drops off at Red, then drives to pick up at Blue.
    # So we match Red -> Blue.
    
    p_pickup, p_dropoff, t_pickup, t_dropoff = load_taxi_csv(args.input, args.n, args.seed)
    
    # Convert to Tensors
    # Shape [N, 2]
    xA_deg = torch.tensor(p_dropoff, device=device) # Dropoff (Red)
    xB_deg = torch.tensor(p_pickup, device=device)  # Pickup (Blue)
    
    tA = torch.tensor(t_dropoff, device=device)     # Dropoff time
    tB = torch.tensor(t_pickup, device=device)      # Pickup time
    
    # Project Lat/Lon to Meters
    # P_red (Dropoff), P_blue (Pickup)
    P_red, P_blue, lon0, lat0 = _project_lonlat_to_meters(xA_deg, xB_deg)
    
    print(f"Projected {P_red.shape[0]} points. Origin: ({lon0:.4f}, {lat0:.4f})")
    
    # 3. Solver Execution
    solver = GPUClusteredSolver(
        P_red=P_red, 
        P_blue=P_blue,
        T_red=tA,
        T_blue=tB,
        epsilon=args.epsilon,
        speed_mps=args.speed_mps,
        k=args.k_levels,
        batch_size=4096,
        stop_threshold=args.stop_threshold,
        min_free_count=args.min_free_count,
    )
    
    start_time = time.time()
    Mb = solver.solve()
    runtime = time.time() - start_time
    
    # 4. Results & Stats
    matched_count = (Mb != -1).sum().item()
    N = P_red.shape[0]
    
    output = {
        "params": vars(args),
        "results": {
            "total_points": N,
            "matched_points": matched_count,
            "match_percentage": matched_count / N,
            "runtime_sec": runtime,
        }
    }
    
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {args.out}")
    else:
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
