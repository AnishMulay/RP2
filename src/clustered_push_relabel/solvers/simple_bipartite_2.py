import gc
import time
from typing import Tuple

import torch

from ..clustering.simple import SimpleClustering


def _ensure_long_arange(owner, attr_name, size, device):
    buf = getattr(owner, attr_name, None)
    if buf is None or buf.numel() < size:
        buf = torch.arange(size, device=device, dtype=torch.long)
        setattr(owner, attr_name, buf)
    return buf[:size]


def _ensure_bool_buffer(owner, attr_name, size, device):
    buf = getattr(owner, attr_name, None)
    if buf is None or buf.numel() < size:
        buf = torch.empty(size, device=device, dtype=torch.bool)
        setattr(owner, attr_name, buf)
    return buf[:size]


class SimpleGPUSolver2:
    """
    Correctness-first clustered push-relabel style solver.

    This solver uses the same clustering outputs as the original SimpleGPUSolver,
    but it does NOT maintain an explicit V matrix. Set 1 admissibility is
    computed directly from DR_int and the dual weights on the fly.

    Important design choices:
    - Set 2 (adjacency-list direct edges) is explicit.
    - Set 1 (sampled-center edges) is computed on the fly.
    - For each sampled center s, only free blues b with:
          nearest_s[b] = s
      and maximum y_B[b] among that group
      are allowed to participate in Set 1 during that phase.
    - Weighted probability logic between Set 1 and Set 2 is preserved.
    - Dual update order is:
          1) update matching
          2) decrement y_A on accepted reds
          3) compute delta on remaining free blues
          4) increment y_B on remaining free blues
    """

    def __init__(
        self,
        A,
        B,
        epsilon,
        batch_size=None,
        tile_size=None,
        verbose=False,
        max_iters=50000,
        diameter: float = 1.0,
        sample_factor: float = 1.0,
    ):
        if A.device != B.device:
            raise ValueError("A and B must be on the same device")
        if A.device.type != "cuda":
            raise ValueError("SimpleGPUSolver2 requires CUDA tensors")
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be rank-2 tensors")
        if A.shape != B.shape:
            raise ValueError("A and B must have the same shape (N, d)")
        if not A.is_floating_point() or not B.is_floating_point():
            raise TypeError("A and B must be floating-point tensors")
        if A.shape[0] == 0:
            raise ValueError("A and B must be non-empty")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.device = A.device
        self.N = A.shape[0]
        self.epsilon = float(epsilon)
        self.epsilon_int = int(round(self.epsilon * self.N))
        self.diameter = float(diameter)
        self.sample_factor = float(sample_factor)
        self.verbose = bool(verbose)
        self.max_iters = int(max_iters)

        if tile_size is None:
            tile_size = 2048 if batch_size is None else batch_size
        self.batch_size = int(tile_size)

        self.P_red = A
        self.P_blue = B

        if self.verbose:
            print(
                "=" * 60
                + f"\n[Init Simple2] N={self.N}, epsilon={self.epsilon}, "
                + f"tile={self.batch_size}, device={self.device}"
            )

        t0 = time.time()
        cluster_engine = SimpleClustering(
            epsilon=self.epsilon,
            tile_size=self.batch_size,
            sample_factor=self.sample_factor,
        )
        clustering = cluster_engine.run(A, B)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if self.verbose:
            print(f"[Init Simple2] clustering done in {time.time() - t0:.2f}s")

        # Clustering outputs used directly.
        self.DR_int = clustering["DR_int"]                  # (S, N)
        self.d_min_b_int = clustering["d_min_b_int"]        # (N,)
        self.nearest_s = clustering["nearest_s"]            # (N,)
        self.adj_ptr = clustering["adj_ptr"]                # (N+1,)
        self.adj_col = clustering["adj_col"]                # (M,)
        self.adj_dist_int = clustering["adj_dist_int"]      # (M,)

        del clustering, cluster_engine
        gc.collect()

        self.num_samples = self.DR_int.shape[0]

        # Duals and matching
        self.y_A = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.y_B = torch.zeros(self.N, device=self.device, dtype=torch.int32)

        self.match_A = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
        self.match_B = torch.full((self.N,), -1, device=self.device, dtype=torch.long)

        # Cleanup bookkeeping
        self.cleanup_blues = torch.empty(0, device=self.device, dtype=torch.long)

        # Provenance for phase-matched edges
        self.phase_match_is_set1 = torch.zeros(
            self.N, device=self.device, dtype=torch.bool
        )

        # Compatibility aliases used by other experiment code
        self.yA = self.y_A
        self.yB = self.y_B
        self.MA = self.match_A
        self.MB = self.match_B
        self.iterations = 0

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self):
        device = self.device
        B_free = torch.arange(self.N, device=device, dtype=torch.long)
        iteration = 0

        while True:
            num_free = B_free.numel()
            if num_free <= self.epsilon_int:
                break
            if iteration >= self.max_iters:
                break
            iteration += 1

            # Determine which free blues are eligible for Set 1 in this phase:
            # for each sampled center, only free blues mapped to that center
            # with maximum y_B are allowed to do Set 1 search.
            set1_active = self._compute_set1_active_mask(B_free)

            # Find one candidate from Set 1 and Set 2 for each free blue,
            # together with the candidate counts used in weighted selection.
            set1_counts, set1_choice = self._find_set1_candidates(B_free, set1_active)
            set2_counts, set2_choice = self._find_set2_candidates(B_free)

            total_counts = set1_counts + set2_counts
            has_any = total_counts > 0

            if not has_any.any().item():
                delta = self._compute_delta(B_free, set1_active)
                if delta <= 0:
                    raise RuntimeError(
                        "SimpleGPUSolver2: no proposals and no positive delta."
                    )
                self.y_B[B_free] += delta
                continue

            # Weighted random choice between Set 1 and Set 2
            p1 = torch.zeros(num_free, device=device, dtype=torch.float32)
            valid_weight = total_counts > 0
            p1[valid_weight] = (
                set1_counts[valid_weight].float() / total_counts[valid_weight].float()
            )

            rand_pick = torch.rand(num_free, device=device)
            choose_set1 = (
                has_any
                & (set1_counts > 0)
                & ((set2_counts == 0) | (rand_pick < p1))
            )
            choose_set2 = has_any & (set2_counts > 0) & (~choose_set1)

            proposal_mask = choose_set1 | choose_set2
            proposal_b = B_free[proposal_mask]

            proposal_a = torch.empty(
                proposal_b.numel(), device=device, dtype=torch.long
            )
            proposal_is_set1 = torch.empty(
                proposal_b.numel(), device=device, dtype=torch.bool
            )

            if proposal_b.numel() == 0:
                delta = self._compute_delta(B_free, set1_active)
                if delta <= 0:
                    raise RuntimeError(
                        "SimpleGPUSolver2: no proposal vertices and no positive delta."
                    )
                self.y_B[B_free] += delta
                continue

            pos = torch.nonzero(proposal_mask, as_tuple=True)[0]
            set1_pos = pos[choose_set1[proposal_mask]]
            set2_pos = pos[choose_set2[proposal_mask]]

            if set1_pos.numel() > 0:
                proposal_a[choose_set1[proposal_mask]] = set1_choice[set1_pos]
                proposal_is_set1[choose_set1[proposal_mask]] = True

            if set2_pos.numel() > 0:
                proposal_a[choose_set2[proposal_mask]] = set2_choice[set2_pos]
                proposal_is_set1[choose_set2[proposal_mask]] = False

            # Resolve conflicts: each red accepts at most one blue
            accepted_prop_idx = self._resolve_conflicts(proposal_a)
            if accepted_prop_idx.numel() == 0:
                delta = self._compute_delta(B_free, set1_active)
                if delta <= 0:
                    raise RuntimeError(
                        "SimpleGPUSolver2: no accepted proposals and no positive delta."
                    )
                self.y_B[B_free] += delta
                continue

            r_new = proposal_a[accepted_prop_idx]
            b_new = proposal_b[accepted_prop_idx]
            accepted_is_set1 = proposal_is_set1[accepted_prop_idx]

            # Update matching; evicted blues return to the free set.
            F_B_new = self._update_matching(B_free, r_new, b_new)
            self.phase_match_is_set1[b_new] = accepted_is_set1

            # Lower red duals on accepted reds for this phase.
            self.y_A[r_new] -= 1

            # Compute delta for the remaining free blues using updated y_A.
            if F_B_new.numel() > 0:
                set1_active_new = self._compute_set1_active_mask(F_B_new)
                delta = self._compute_delta(F_B_new, set1_active_new)
                if delta <= 0:
                    raise RuntimeError(
                        "SimpleGPUSolver2: non-positive delta after matching update."
                    )
                self.y_B[F_B_new] += delta

            B_free = F_B_new

        self.iterations = iteration
        self.cleanup_remaining_points()
        return self.match_B

    # ------------------------------------------------------------------
    # Set 1 / Set 2 candidate search
    # ------------------------------------------------------------------

    def _compute_set1_active_mask(self, B_free):
        """
        For each sampled center s, mark exactly those free blues mapped to s
        whose y_B value is maximal among that group.
        """
        num_free = B_free.numel()
        if num_free == 0:
            return torch.empty(0, device=self.device, dtype=torch.bool)

        free_s = self.nearest_s[B_free]
        free_yB = self.y_B[B_free].to(torch.long)

        group_max = torch.full(
            (self.num_samples,),
            torch.iinfo(torch.int64).min,
            device=self.device,
            dtype=torch.long,
        )
        group_max.scatter_reduce_(
            0,
            free_s,
            free_yB,
            reduce="amax",
            include_self=True,
        )
        return free_yB == group_max[free_s]

    def _find_set1_candidates(self, B_free, set1_active):
        """
        For each free blue b that is active for Set 1 in this phase:
        - scan all reds a
        - admissibility test:
              y_B[b] + y_A[a] == d_min_b_int[b] + DR_int[s_b, a] + 1

        Returns:
        - counts: (num_free,) long
        - choice: (num_free,) long, sampled admissible red or -1
        """
        num_free = B_free.numel()
        counts = torch.zeros(num_free, device=self.device, dtype=torch.long)
        choice = torch.full(num_free, -1, device=self.device, dtype=torch.long)

        active_pos = torch.nonzero(set1_active, as_tuple=True)[0]
        for pos in active_pos.tolist():
            b = B_free[pos]
            s = self.nearest_s[b]
            lhs = self.y_B[b].to(torch.long) + self.y_A.to(torch.long)
            rhs = (
                self.d_min_b_int[b].to(torch.long)
                + self.DR_int[s].to(torch.long)
                + 1
            )
            admissible = lhs == rhs
            cand = torch.nonzero(admissible, as_tuple=True)[0]
            if cand.numel() == 0:
                continue

            counts[pos] = cand.numel()
            pick = torch.randint(cand.numel(), (1,), device=self.device).item()
            choice[pos] = cand[pick]

        return counts, choice

    def _find_set2_candidates(self, B_free):
        """
        For each free blue b:
        - scan adjacency list Adj(b)
        - admissibility test:
              y_B[b] + y_A[a] == adj_dist_int(b,a) + 1

        Returns:
        - counts: (num_free,) long
        - choice: (num_free,) long, sampled admissible red or -1
        """
        num_free = B_free.numel()
        counts = torch.zeros(num_free, device=self.device, dtype=torch.long)
        choice = torch.full(num_free, -1, device=self.device, dtype=torch.long)

        for pos in range(num_free):
            b = B_free[pos]
            start = int(self.adj_ptr[b].item())
            end = int(self.adj_ptr[b + 1].item())
            if start == end:
                continue

            edge_idx = torch.arange(start, end, device=self.device, dtype=torch.long)
            a_idx = self.adj_col[edge_idx]
            lhs = self.y_B[b].to(torch.long) + self.y_A[a_idx].to(torch.long)
            rhs = self.adj_dist_int[edge_idx].to(torch.long) + 1
            admissible = lhs == rhs
            cand_edges = torch.nonzero(admissible, as_tuple=True)[0]
            if cand_edges.numel() == 0:
                continue

            counts[pos] = cand_edges.numel()
            pick = torch.randint(cand_edges.numel(), (1,), device=self.device).item()
            choice[pos] = a_idx[cand_edges[pick]]

        return counts, choice

    # ------------------------------------------------------------------
    # Conflict resolution / matching update
    # ------------------------------------------------------------------

    def _resolve_conflicts(self, proposal_a):
        """
        proposal_a: (num_props,) red indices

        Returns accepted proposal indices into proposal_a / proposal_b arrays.
        """
        num_props = proposal_a.numel()
        if num_props == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        prop_idx = _ensure_long_arange(
            self, "_simple2_prop_arange", num_props, self.device
        )
        rand_prio = torch.rand(num_props, device=self.device)

        min_prio = torch.full((self.N,), float("inf"), device=self.device)
        min_prio.scatter_reduce_(
            0, proposal_a, rand_prio, reduce="amin", include_self=True
        )

        is_min = rand_prio == min_prio[proposal_a]

        accepted_idx = torch.full(
            (self.N,), num_props, device=self.device, dtype=torch.long
        )
        accepted_idx.scatter_reduce_(
            0,
            proposal_a[is_min],
            prop_idx[is_min],
            reduce="amin",
            include_self=True,
        )

        accepted_mask = prop_idx == accepted_idx[proposal_a]
        return prop_idx[accepted_mask]

    def _update_matching(self, B_free, r_new, b_new):
        """
        Apply newly accepted matches.
        If a red was already matched, evict its previous blue.
        Return the new free-blue set.
        """
        was_matched = self.match_A[r_new] != -1
        evicted_b = self.match_A[r_new[was_matched]].clone()
        if evicted_b.numel() > 0:
            self.match_B[evicted_b] = -1
            self.phase_match_is_set1[evicted_b] = False

        self.match_A[r_new] = b_new
        self.match_B[b_new] = r_new

        keep_free = _ensure_bool_buffer(
            self, "_simple2_keep_free_mask", B_free.numel(), self.device
        )
        keep_free.fill_(True)
        keep_free[torch.searchsorted(B_free, b_new)] = False
        still_free = B_free[keep_free]

        if evicted_b.numel() == 0:
            return still_free

        F_B_new, _ = torch.sort(torch.cat([still_free, evicted_b]))
        return F_B_new

    # ------------------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------------------

    def _compute_delta(self, B_free, set1_active):
        """
        Compute the global minimum positive slack over the currently relevant edges.

        Set 2 contributes for all free blues.
        Set 1 contributes only for free blues active in the current Set 1 groups.
        """
        num_free = B_free.numel()
        if num_free == 0:
            return 1

        sentinel = torch.iinfo(torch.int64).max // 4
        min_slack_per_blue = torch.full(
            (num_free,), sentinel, device=self.device, dtype=torch.long
        )

        for pos in range(num_free):
            b = B_free[pos]
            yb = self.y_B[b].to(torch.long)

            # Set 2 min slack over adjacency list
            start = int(self.adj_ptr[b].item())
            end = int(self.adj_ptr[b + 1].item())
            min_slack2 = sentinel
            if start != end:
                edge_idx = torch.arange(start, end, device=self.device, dtype=torch.long)
                a_idx = self.adj_col[edge_idx]
                slack2 = (
                    self.adj_dist_int[edge_idx].to(torch.long)
                    + 1
                    - yb
                    - self.y_A[a_idx].to(torch.long)
                )
                min_slack2 = int(slack2.min().item())

            # Set 1 min slack over all reds, but only for set1-active blues
            min_slack1 = sentinel
            if set1_active[pos].item():
                s = self.nearest_s[b]
                slack1 = (
                    self.d_min_b_int[b].to(torch.long)
                    + self.DR_int[s].to(torch.long)
                    + 1
                    - yb
                    - self.y_A.to(torch.long)
                )
                min_slack1 = int(slack1.min().item())

            min_slack_per_blue[pos] = min(min_slack1, min_slack2)

        positive_mask = min_slack_per_blue > 0
        if positive_mask.any().item():
            delta = int(min_slack_per_blue[positive_mask].min().item())
            return max(delta, 1)

        # If there is no positive slack, return 0 so the caller can fail loudly.
        return 0

    # ------------------------------------------------------------------
    # Cleanup / stats / verification
    # ------------------------------------------------------------------

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.match_B == -1, as_tuple=True)[0]
        free_a = torch.nonzero(self.match_A == -1, as_tuple=True)[0]
        count = min(free_b.numel(), free_a.numel())
        if count > 0:
            self.match_B[free_b[:count]] = free_a[:count]
            self.match_A[free_a[:count]] = free_b[:count]
            self.cleanup_blues = free_b[:count].clone()

    def calculate_final_stats(self):
        dists = torch.norm(self.P_blue - self.P_red[self.match_B], p=2, dim=1)
        dists = dists * self.diameter
        total_cost = dists.sum()
        avg_cost = total_cost / self.N
        print(f"Total Euclidean Cost: {total_cost.item():.4f}")
        print(f"Avg Euclidean Cost: {avg_cost.item():.4f}")

    def verify_solution(self):
        """
        Full O(N^2) verification using the same direct/triangle proxy formulas
        this solver uses conceptually.

        Feasibility for all pairs:
            y_B[b] + y_A[a] <= proxy(b,a) + 1

        Admissibility for matched edges:
            - if phase-matched via Set 1:  use triangle proxy
            - if phase-matched via Set 2:  use direct proxy when edge exists in adjacency,
                                          otherwise fall back conservatively to triangle proxy
            - cleanup edges are excluded from matched-admissibility counts
        """
        device = self.device
        N = self.N

        # Build direct-edge lookup from adjacency
        if self.adj_col.numel() > 0:
            all_b_for_adj = torch.repeat_interleave(
                torch.arange(N, device=device, dtype=torch.long),
                self.adj_ptr[1:] - self.adj_ptr[:-1],
            )
            adj_keys = all_b_for_adj * N + self.adj_col
            sort_idx = torch.argsort(adj_keys)
            sorted_keys = adj_keys[sort_idx]
            sorted_dists = self.adj_dist_int[sort_idx].to(torch.long)
        else:
            sorted_keys = torch.empty(0, device=device, dtype=torch.long)
            sorted_dists = torch.empty(0, device=device, dtype=torch.long)

        def lookup_direct_proxy(b_idx, a_idx):
            proxy = torch.full(
                (b_idx.numel(),),
                -1,
                device=device,
                dtype=torch.long,
            )
            if sorted_keys.numel() == 0 or b_idx.numel() == 0:
                return proxy

            keys = b_idx.to(torch.long) * N + a_idx.to(torch.long)
            pos = torch.searchsorted(sorted_keys, keys)
            in_bounds = pos < sorted_keys.numel()
            if in_bounds.any():
                hit = in_bounds.clone()
                hit[in_bounds] = sorted_keys[pos[in_bounds]] == keys[in_bounds]
                if hit.any():
                    proxy[hit] = sorted_dists[pos[hit]]
            return proxy

        # ---------- Full-pair feasibility using the piecewise direct/triangle rule
        # Default triangle proxy for all pairs
        yB_all = self.y_B.to(torch.long).unsqueeze(1)         # (N,1)
        yA_all = self.y_A.to(torch.long).unsqueeze(0)         # (1,N)
        s_all = self.nearest_s                                # (N,)
        tri_proxy = (
            self.d_min_b_int.to(torch.long).unsqueeze(1)
            + self.DR_int[s_all].to(torch.long)
        )                                                    # (N,N)

        piecewise_proxy = tri_proxy.clone()

        if self.adj_col.numel() > 0:
            b_idx = torch.repeat_interleave(
                torch.arange(N, device=device, dtype=torch.long),
                self.adj_ptr[1:] - self.adj_ptr[:-1],
            )
            a_idx = self.adj_col
            piecewise_proxy[b_idx, a_idx] = self.adj_dist_int.to(torch.long)

        feas_slack = piecewise_proxy + 1 - yB_all - yA_all
        feas_violations = int((feas_slack < 0).sum().item())
        feas_worst_excess = int((-feas_slack).clamp_min(0).max().item())

        # ---------- Matched-edge admissibility using actual phase provenance
        matched_b = torch.nonzero(self.match_B != -1, as_tuple=True)[0]
        matched_a = self.match_B[matched_b]

        is_cleanup = torch.zeros(N, device=device, dtype=torch.bool)
        if self.cleanup_blues.numel() > 0:
            is_cleanup[self.cleanup_blues] = True

        phase_mask = ~is_cleanup[matched_b]
        phase_b = matched_b[phase_mask]
        phase_a = matched_a[phase_mask]

        if phase_b.numel() > 0:
            tri_phase = (
                self.d_min_b_int[phase_b].to(torch.long)
                + self.DR_int[self.nearest_s[phase_b], phase_a].to(torch.long)
            )
            dir_phase = lookup_direct_proxy(phase_b, phase_a)

            matched_proxy = tri_phase.clone()
            set2_mask = ~self.phase_match_is_set1[phase_b]
            if set2_mask.any():
                # For phase edges marked as Set 2, use direct proxy when present.
                valid_dir = dir_phase[set2_mask] >= 0
                if valid_dir.any():
                    tmp = matched_proxy[set2_mask]
                    tmp[valid_dir] = dir_phase[set2_mask][valid_dir]
                    matched_proxy[set2_mask] = tmp

            matched_diff = (
                matched_proxy
                - self.y_B[phase_b].to(torch.long)
                - self.y_A[phase_a].to(torch.long)
            )
            phase_violations = int((matched_diff != 0).sum().item())
            phase_worst_abs_diff = int(matched_diff.abs().max().item())
        else:
            phase_violations = 0
            phase_worst_abs_diff = 0

        return {
            "feas_total": int(N * N),
            "feas_violations": feas_violations,
            "feas_worst_excess": feas_worst_excess,
            "phase_total": int(phase_b.numel()),
            "phase_violations": phase_violations,
            "phase_worst_abs_diff": phase_worst_abs_diff,
        }