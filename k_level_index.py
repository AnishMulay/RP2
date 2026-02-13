import torch

class KLevelVectorIndex:
    """
    All-points clustering index with leveled multi-membership:
    Membership: dist(y, p) <= threshold[y]
    Level: ceil(dist(y, p) / epsilon)
    """
    def __init__(self, epsilon=1.0, k=2, batch_size=1024):
        # k is conceptually 2 here (Landmarks + Points)
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        self.epsilon = float(epsilon)
        self.batch_size = batch_size
        self.dataset = None
        self.landmark_indices = None
        self.thresholds = None

        # Forward CSR: center -> points
        self.crow_indices = None
        self.col_indices = None

        # Inverted CSR: point -> centers (sorted by level)
        self.point_crow_indices = None
        self.point_col_indices = None
        self.point_levels = None

    def build_index(self, X):
        """
        Builds dual CSR indices with batched distance blocks to avoid NxN materialization.
        """
        N = X.shape[0]
        print(f"[*] Building leveled all-points index on {N} vectors...")
        self.dataset = X

        # 1. Sample landmarks (S)
        num_landmarks = int(N**0.5)
        self.landmark_indices = torch.randperm(N, device=X.device)[:num_landmarks]
        landmarks = X[self.landmark_indices]
        
        print(f"[*] Sampled {num_landmarks} Landmarks (S).")

        # 2. Pre-calculate thresholds D_S[y] = min_dist(y, S)
        D_S = torch.empty(N, device=X.device)
        
        for i in range(0, N, self.batch_size):
            end = min(i + self.batch_size, N)
            batch_X = X[i:end]
            dists = torch.cdist(batch_X, landmarks)
            min_dists, _ = dists.min(dim=1)
            D_S[i:end] = min_dists
        self.thresholds = D_S
        
        print("[*] Thresholds (dist to S) calculated.")

        # 3. Generate sparse triplets (center_id, point_id, level) in CPU memory
        triplet_centers = []
        triplet_points = []
        triplet_levels = []
        eps = torch.tensor(self.epsilon, device=X.device, dtype=X.dtype)

        print("[*] Constructing membership triplets...")
        for c_start in range(0, N, self.batch_size):
            c_end = min(c_start + self.batch_size, N)
            centers = X[c_start:c_end]

            for p_start in range(0, N, self.batch_size):
                p_end = min(p_start + self.batch_size, N)
                points = X[p_start:p_end]
                point_thresholds = D_S[p_start:p_end]

                dists = torch.cdist(centers, points)  # (Bc, Bp)
                mask = dists <= point_thresholds.unsqueeze(0)
                nz = torch.nonzero(mask, as_tuple=False)

                if nz.numel() == 0:
                    continue

                local_center_idx = nz[:, 0]
                local_point_idx = nz[:, 1]
                valid_dists = dists[local_center_idx, local_point_idx]
                levels = torch.ceil(valid_dists / eps).to(torch.int32)

                global_centers = (local_center_idx + c_start).to(torch.int64).cpu()
                global_points = (local_point_idx + p_start).to(torch.int64).cpu()
                levels_cpu = levels.to(torch.int32).cpu()

                triplet_centers.append(global_centers)
                triplet_points.append(global_points)
                triplet_levels.append(levels_cpu)

            if (c_start // self.batch_size) % 10 == 0:
                print(f"    Processed {c_start}/{N} centers...")

        if not triplet_centers:
            # Defensive fallback: empty graph
            self.crow_indices = torch.zeros(N + 1, dtype=torch.long, device=X.device)
            self.col_indices = torch.empty(0, dtype=torch.long, device=X.device)
            self.point_crow_indices = torch.zeros(N + 1, dtype=torch.long, device=X.device)
            self.point_col_indices = torch.empty(0, dtype=torch.long, device=X.device)
            self.point_levels = torch.empty(0, dtype=torch.int32, device=X.device)
            print("[*] Index built with no memberships.")
            return

        centers_all = torch.cat(triplet_centers, dim=0)  # CPU int64
        points_all = torch.cat(triplet_points, dim=0)    # CPU int64
        levels_all = torch.cat(triplet_levels, dim=0)    # CPU int32

        # 4A. Forward CSR (center -> points)
        fwd_key = centers_all * N + points_all
        fwd_order = torch.argsort(fwd_key)
        centers_fwd = centers_all[fwd_order]
        points_fwd = points_all[fwd_order]

        center_counts = torch.bincount(centers_fwd, minlength=N)
        crow_cpu = torch.empty(N + 1, dtype=torch.long)
        crow_cpu[0] = 0
        crow_cpu[1:] = torch.cumsum(center_counts, dim=0)

        self.crow_indices = crow_cpu.to(X.device)
        self.col_indices = points_fwd.to(dtype=torch.long, device=X.device)

        # 4B. Inverted CSR (point -> centers), sorted by (point, level, center)
        max_level = int(levels_all.max().item())
        inv_key = ((points_all * (max_level + 1)) + levels_all.to(torch.int64)) * N + centers_all
        inv_order = torch.argsort(inv_key)
        points_inv = points_all[inv_order]
        centers_inv = centers_all[inv_order]
        levels_inv = levels_all[inv_order]

        point_counts = torch.bincount(points_inv, minlength=N)
        point_crow_cpu = torch.empty(N + 1, dtype=torch.long)
        point_crow_cpu[0] = 0
        point_crow_cpu[1:] = torch.cumsum(point_counts, dim=0)

        self.point_crow_indices = point_crow_cpu.to(X.device)
        self.point_col_indices = centers_inv.to(dtype=torch.long, device=X.device)
        self.point_levels = levels_inv.to(device=X.device)

        avg_cluster = self.crow_indices[-1].item() / max(1, N)
        print(f"[*] Index built. Avg Cluster Size: {avg_cluster:.1f}")
