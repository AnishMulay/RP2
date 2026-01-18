import torch
import math
import time
import gc

# ==========================================
# Component 1: Tiled Euclidean Kernel (Shared)
# ==========================================

class TiledEuclideanKernel:
    """
    Handles distance computations in a memory-efficient, tiled manner.
    Uses expansion: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>.
    """
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size

    def prepare_workspace(self, P: torch.Tensor) -> dict:
        """
        Precomputes norms and transpose for the target set P.
        Args:
            P: Tensor of shape (N, D) - The "Targets"
        """
        # Precompute squared norms: ||x||^2
        P_norms_sq = (P ** 2).sum(dim=1, keepdim=True)
        # Transpose for matrix multiplication: (D, N)
        P_T = P.t()
        
        return {
            "P": P,
            "P_T": P_T,
            "P_norms_sq": P_norms_sq
        }

    def compute_squared_dist_tile(self, query_points: torch.Tensor, workspace: dict) -> torch.Tensor:
        """
        Computes squared distances between ALL workspace points and the query_points.
        Args:
            query_points: Tensor (Batch, D) - The "Centers"
            workspace: Prepared data for the "Targets"
        Returns:
            Tensor (N_targets, Batch_centers)
        """
        P = workspace["P"]
        P_norms_sq = workspace["P_norms_sq"]
        
        # Query norms: (Batch, 1) -> (1, Batch)
        Q_norms_sq = (query_points ** 2).sum(dim=1, keepdim=True).t()
        
        # dist^2 = P_norm + Q_norm - 2 P @ Q.T
        # Init with broadcast sum
        dists_sq = P_norms_sq + Q_norms_sq
        
        # Addmm: dists_sq += -2.0 * (P @ Q.T)
        dists_sq.addmm_(P, query_points.t(), beta=1.0, alpha=-2.0)
        
        return torch.clamp(dists_sq, min=0.0)


# ==========================================
# Component 2: Red-Blue Clustering Algo
# ==========================================

class RedBlueClusteringAlgo:
    """
    Implements Parallel Red-Blue Clustering.
    Generates two independent covers: one centered at Blue points, one at Red points.
    Both covers cluster the entire dataset (Union of Red and Blue).
    """

    def __init__(self, epsilon: float, batch_size: int = 1024):
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.kernel = TiledEuclideanKernel(chunk_size=batch_size)

    def _sample_landmarks(self, n_points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Selects landmarks with probability 1/sqrt(n).
        Returns: (Indices, BooleanMask)
        """
        prob = n_points ** (-0.5)
        mask = torch.rand(n_points, device=device) < prob
        
        # Safety: Ensure at least one landmark
        if not mask.any():
             mask[torch.randint(0, n_points, (1,), device=device)] = True
             
        indices = torch.nonzero(mask).squeeze(1)
        return indices, mask

    def _compute_voronoi_bounds(self, targets: torch.Tensor, landmarks: torch.Tensor, 
                                workspace: dict) -> torch.Tensor:
        """
        Computes D(x) = min d(x, s) for all x in targets, s in landmarks.
        Uses tiling to handle large landmark sets safely.
        """
        n_targets = targets.shape[0]
        n_landmarks = landmarks.shape[0]
        device = targets.device
        
        D_y_sq = torch.full((n_targets,), float('inf'), device=device)
        
        # Stream landmarks in batches
        for i in range(0, n_landmarks, self.batch_size):
            end_i = min(i + self.batch_size, n_landmarks)
            landmark_batch = landmarks[i:end_i]
            
            # Compute dists: (N_targets, Batch_Size)
            dists_batch = self.kernel.compute_squared_dist_tile(landmark_batch, workspace)
            
            # Update min
            batch_min, _ = dists_batch.min(dim=1)
            D_y_sq = torch.min(D_y_sq, batch_min)
            
            del dists_batch
            
        return D_y_sq

    def _compute_global_radii(self, P_all: torch.Tensor, workspace: dict) -> torch.Tensor:
        """
        Computes radii scales based on max distance in the full dataset.
        """
        n_points = P_all.shape[0]
        device = P_all.device
        max_dist_sq = 0.0
        
        # Tiled reduction for max distance
        for i in range(0, n_points, self.batch_size):
            end_i = min(i + self.batch_size, n_points)
            batch_points = P_all[i:end_i]
            
            dists = self.kernel.compute_squared_dist_tile(batch_points, workspace)
            current_max = dists.max().item()
            if current_max > max_dist_sq:
                max_dist_sq = current_max
            del dists
            
        Delta = math.sqrt(max_dist_sq)
        if Delta <= 1e-9:
             return torch.tensor([0.0], device=device)

        base = 1.0 + self.epsilon
        t = int(math.ceil(1.0 / self.epsilon))
        
        indices = torch.arange(1, t + 1, device=device, dtype=P_all.dtype)
        radii = torch.pow(base, indices)
        radii = torch.cat([torch.tensor([0.0], device=device, dtype=P_all.dtype), radii])
        
        return radii

    def _build_cover(self, 
                     centers_source: torch.Tensor, 
                     center_mask_P1: torch.Tensor,
                     targets_workspace: dict, 
                     D_voronoi_sq: torch.Tensor, 
                     radii: torch.Tensor) -> list[torch.Tensor]:
        """
        Generic function to build clusters (Red or Blue).
        OPTIMIZED: Vectorized over radii to remove Python loop latency.
        """
        n_centers = centers_source.shape[0]
        n_targets = targets_workspace["P"].shape[0]
        n_radii = radii.shape[0]
        radii_sq = radii ** 2
        
        clusters = []
        micro_batch_size = 32
        r_broad = radii_sq.view(1, n_radii, 1)
        
        # Iterate over CENTERS in batches
        for start_q in range(0, n_centers, self.batch_size):
            end_q = min(start_q + self.batch_size, n_centers)
            q_batch = centers_source[start_q:end_q]
            current_batch_size = q_batch.shape[0]
            
            for start_mb in range(0, current_batch_size, micro_batch_size):
                end_mb = min(start_mb + micro_batch_size, current_batch_size)
                q_micro = q_batch[start_mb:end_mb]
                current_micro_size = q_micro.shape[0]

                # 2. Compute Distances: (N_all, Micro_Batch_Size)
                # Transpose to (Micro_Batch_Size, N_all) for easier broadcasting below
                d_xq_sq = self.kernel.compute_squared_dist_tile(q_micro, targets_workspace).t()
                
                # 3. Vectorize: Prepare for (Micro, Radii, Points) broadcasting
                # Distances: (Micro, 1, Points)
                d_broad = d_xq_sq.unsqueeze(1)
                
                # 4. Distance Condition (Broadcasted): (Micro, Radii, Points)
                mask = d_broad <= r_broad
                
                # 5. Voronoi Condition
                # Is the center a landmark? (Micro,) -> (Micro, 1, 1)
                is_landmark_batch = center_mask_P1[start_q + start_mb:start_q + end_mb].view(
                    current_micro_size, 1, 1
                )
                
                if not is_landmark_batch.all():
                    # D_voronoi: (Points,) -> (1, 1, Points)
                    dv_broad = D_voronoi_sq.view(1, 1, n_targets)
                    # Check: d < D[x]
                    voronoi_mask = d_broad < dv_broad
                    
                    # Logic: Landmark ? Distance : (Distance & Voronoi)
                    mask = torch.where(is_landmark_batch, mask, mask & voronoi_mask)
                
                # 6. Bulk Extraction
                # We want to get all indices (b, r, n) where mask is True.
                # nonzero() returns tensor of shape (M, 3) -> [batch_idx, radius_idx, point_idx]
                if not mask.any():
                    del mask, d_xq_sq, d_broad
                    continue
                    
                all_indices = torch.nonzero(mask)
                
                # Move to CPU ONCE per micro-batch (Critical Optimization)
                all_indices_cpu = all_indices.cpu()
                
                # 7. Fast Splitting on CPU
                # We need to separate this massive list into individual clusters.
                # We group by (batch_idx, radius_idx).
                # Create a unique flat ID for every (micro, radius) pair.
                b_idx = all_indices_cpu[:, 0]
                r_idx = all_indices_cpu[:, 1]
                point_idx = all_indices_cpu[:, 2]
                
                flat_group_ids = b_idx * n_radii + r_idx
                
                # Calculate size of every group using bincount
                # minlength ensures we account for all potential groups in this micro-batch
                total_groups = current_micro_size * n_radii
                counts = torch.bincount(flat_group_ids, minlength=total_groups)
                
                # Split the values (point indices) into a tuple of tensors
                grouped_clusters = torch.split(point_idx, counts.tolist())
                
                # Append non-empty clusters to the main list
                # (Matches original behavior: we only stored clusters that had members)
                for c in grouped_clusters:
                    if c.numel() > 0:
                        clusters.append(c)

                # Cleanup GPU memory
                del mask, d_xq_sq, d_broad, all_indices

        return clusters

    def run(self, P_red: torch.Tensor, P_blue: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """
        Main Execution Method.
        Args:
            P_red: (N_red, D)
            P_blue: (N_blue, D)
        Returns:
            Dict containing 'blue_clusters' and 'red_clusters'.
            Values are lists of PyTorch tensors (indices into the combined P_all).
        """
        # 0. Setup and Precision
        if not P_red.is_floating_point(): P_red = P_red.to(torch.float32)
        if not P_blue.is_floating_point(): P_blue = P_blue.to(torch.float32)
        
        device = P_red.device
        
        with torch.no_grad():
            # 1. Prepare Combined Dataset (Targets)
            # Concatenate to enable clustering over the union
            # Indices [0...N_red-1] are Red points
            # Indices [N_red...N_total-1] are Blue points
            P_all = torch.cat([P_red, P_blue], dim=0)
            workspace = self.kernel.prepare_workspace(P_all)
            
            # 2. Sampling (Independent)
            red_indices, red_mask = self._sample_landmarks(P_red.shape[0], device)
            blue_indices, blue_mask = self._sample_landmarks(P_blue.shape[0], device)
            
            # 3. Voronoi Bounds (Independent)
            # D_red[x] = min dist from x to any RED landmark
            D_red_sq = self._compute_voronoi_bounds(P_all, P_red[red_indices], workspace)
            
            # D_blue[x] = min dist from x to any BLUE landmark
            D_blue_sq = self._compute_voronoi_bounds(P_all, P_blue[blue_indices], workspace)
            
            # 4. Radii (Global)
            radii = self._compute_global_radii(P_all, workspace)
            
            # 5. Build Covers
            blue_clusters = self._build_cover(
                centers_source=P_blue,
                center_mask_P1=blue_mask,
                targets_workspace=workspace,
                D_voronoi_sq=D_blue_sq,
                radii=radii
            )
            
            red_clusters = self._build_cover(
                centers_source=P_red,
                center_mask_P1=red_mask,
                targets_workspace=workspace,
                D_voronoi_sq=D_red_sq,
                radii=radii
            )
            
            return {
                "blue_clusters": blue_clusters,
                "red_clusters": red_clusters
            }

# ==========================================
# Experiment Runner (Main)
# ==========================================

def run_experiment(n_red, n_blue, dimensions):
    print(f"\n--- Running Experiment: Red={n_red}, Blue={n_blue}, Dim={dimensions} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # Generate Synthetic Data
    torch.manual_seed(42)
    
    # Create Red points (shifted slightly positive)
    P_red = torch.randn(n_red, dimensions, device=device) + 1.0
    
    # Create Blue points (shifted slightly negative)
    P_blue = torch.randn(n_blue, dimensions, device=device) - 1.0
    
    # Initialize Algo
    # epsilon=1.0 is standard approximation factor
    # batch_size=512 is safe for most GPUs
    algo = RedBlueClusteringAlgo(epsilon=1.0, batch_size=512)
    
    # Run Timing
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    # Returns lists of tensors
    results = algo.run(P_red, P_blue)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        end_time = time.time()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak GPU Mem: {peak_mem:.2f} MB")
    else:
        end_time = time.time()

    # Output Stats
    print(f"Execution Time: {end_time - start_time:.4f}s")
    print(f"Blue Clusters Found: {len(results['blue_clusters'])}")
    print(f"Red Clusters Found:  {len(results['red_clusters'])}")
    
    # Verify Tensor Output
    if len(results['blue_clusters']) > 0:
        first_cluster = results['blue_clusters'][0]
        print(f"Sample Output Type: {type(first_cluster)}") # Should be <class 'torch.Tensor'>
        print(f"Sample Cluster Size: {first_cluster.shape[0]} points")
    
    return results

if __name__ == "__main__":
    # ==========================================
    # USER PARAMETERS - CHANGE THESE
    # ==========================================
    N_RED = 2000
    N_BLUE = 2000
    DIMENSIONS = 2  # Set to 64 for high-dim, 2 for vis
    
    # Run
    output_pipes = run_experiment(N_RED, N_BLUE, DIMENSIONS)
    
    # 'output_pipes' now contains your tensors ready for the push-reliable alg
