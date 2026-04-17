import gc
import time

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


class SimpleGPUSolver:
    """
    Epsilon-approximate GPU bipartite matcher over the SimpleClustering graph.

    Red points are A/right-side points and blue points are B/left-side points.
    All hot-path admissibility checks use integer multiples of epsilon.
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
        set1_pair_batch=64,
        diameter: float = 1.0,
        sample_factor: float = 1.0,
    ):
        if A.device != B.device:
            raise ValueError("A and B must be on the same device")
        if A.device.type != "cuda":
            raise ValueError("SimpleGPUSolver requires CUDA tensors")
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
        self.verbose = verbose
        self.max_iters = int(max_iters)
        self.set1_pair_batch = int(set1_pair_batch)
        if self.set1_pair_batch <= 0:
            raise ValueError("set1_pair_batch must be positive")

        if tile_size is None:
            tile_size = 2048 if batch_size is None else batch_size
        self.batch_size = int(tile_size)

        self.P_red = A
        self.P_blue = B

        if self.verbose:
            print(
                "=" * 60
                + f"\n[Init Simple] N={self.N}, epsilon={self.epsilon}, "
                + f"tile={self.batch_size}, device={self.device}"
            )

        t0 = time.time()
        cluster_engine = SimpleClustering(
            epsilon=self.epsilon,
            tile_size=self.batch_size,
            sample_factor=sample_factor,
        )
        clustering = cluster_engine.run(A, B)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if self.verbose:
            print(f"[Init Simple] clustering done in {time.time() - t0:.2f}s")

        self.d_min_b_int = clustering["d_min_b_int"]
        self.nearest_s = clustering["nearest_s"]
        self.adj_ptr = clustering["adj_ptr"]
        self.adj_col = clustering["adj_col"]
        self.adj_dist_int = clustering["adj_dist_int"]

        self.V = clustering["DR_int"].clone()
        self.V.neg_()

        del clustering, cluster_engine
        gc.collect()

        self.y_A = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.y_B = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.match_A = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
        self.match_B = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
        self.cleanup_blues = torch.empty(0, device=self.device, dtype=torch.long)

        # Existing experiment code uses MA/MB and yA/yB naming.
        self.yA = self.y_A
        self.yB = self.y_B
        self.MA = self.match_A
        self.MB = self.match_B
        self.iterations = 0

    def solve(self):
        N = self.N
        device = self.device
        B_free = torch.arange(N, device=device, dtype=torch.long)
        iteration = 0

        def _print_progress(iteration, free_before, free_after, status):
            if self.verbose and iteration % 100 == 0:
                print(
                    f"[Simple iter {iteration}] free_before={free_before} "
                    f"free_after={free_after} status={status}",
                    flush=True,
                )

        while True:
            num_free = B_free.numel()
            if num_free <= self.epsilon_int:
                break
            if iteration >= self.max_iters:
                break
            iteration += 1

            pair_inverse, set1_counts, set1_offsets, set1_values = (
                self._set1_groups(B_free)
            )
            unique_pairs, delta_pair_inverse = self._set1_delta_groups(B_free)
            set1_count_per_blue = set1_counts[pair_inverse]
            set2_count_per_blue = self._set2_counts(B_free)

            total_count = (set1_count_per_blue + set2_count_per_blue).float()
            has_any = total_count > 0
            p1 = torch.where(
                has_any,
                set1_count_per_blue.float() / total_count,
                torch.zeros_like(total_count),
            )
            rand_pick = torch.rand(num_free, device=device)
            choose_set1 = (
                has_any
                & (set1_count_per_blue > 0)
                & ((set2_count_per_blue == 0) | (rand_pick < p1))
            )
            choose_set2 = has_any & (set2_count_per_blue > 0) & (~choose_set1)

            proposal_a_parts = []
            proposal_b_parts = []

            b1, a1 = self._sample_set1_choices(
                B_free,
                pair_inverse,
                set1_counts,
                set1_offsets,
                set1_values,
                choose_set1,
            )
            if a1.numel() != 0:
                proposal_a_parts.append(a1)
                proposal_b_parts.append(b1)

            b2, a2 = self._set2_sample(B_free, choose_set2)
            if a2.numel() != 0:
                proposal_a_parts.append(a2)
                proposal_b_parts.append(b2)

            if not proposal_a_parts:
                delta = self._compute_delta(
                    B_free, unique_pairs, delta_pair_inverse
                )
                self.y_B[B_free] += delta
                _print_progress(iteration, num_free, num_free, "no_proposals")
                continue

            proposal_a = torch.cat(proposal_a_parts)
            proposal_b = torch.cat(proposal_b_parts)

            r_new, b_new = self._resolve_conflicts(proposal_a, proposal_b)

            if r_new.numel() == 0:
                delta = self._compute_delta(
                    B_free, unique_pairs, delta_pair_inverse
                )
                self.y_B[B_free] += delta
                _print_progress(iteration, num_free, num_free, "no_accepts")
                continue

            F_B_new = self._update_matching(B_free, r_new, b_new)

            unique_pairs_new, pair_inverse_new = self._set1_delta_groups(F_B_new)
            delta = self._compute_delta(
                F_B_new, unique_pairs_new, pair_inverse_new
            )
            self.y_B[F_B_new] += delta
            self.y_A[r_new] -= 1
            self.V[:, r_new] -= 1

            _print_progress(iteration, num_free, F_B_new.numel(), "ok")
            B_free = F_B_new

        self.iterations = iteration
        self.cleanup_remaining_points()
        if self.verbose:
            print(f"[Simple] Matched: {(self.match_B != -1).sum().item()}/{self.N}")
            self.calculate_final_stats()
            self.verify_solution()
        return self.match_B

    def _set1_groups(self, B_free):
        free_s = self.nearest_s[B_free]
        free_t = 1 - self.y_B[B_free] - self.d_min_b_int[B_free]

        order = torch.argsort(free_s)
        sorted_pairs = torch.stack(
            (free_s[order], free_t[order].to(torch.long)),
            dim=1,
        )
        unique_pairs, inverse_sorted = torch.unique(
            sorted_pairs, dim=0, return_inverse=True
        )

        pair_inverse = torch.empty_like(inverse_sorted)
        pair_inverse[order] = inverse_sorted

        num_pairs = unique_pairs.shape[0]
        set1_counts = torch.empty(num_pairs, device=self.device, dtype=torch.long)
        set1_value_parts = []
        for start in range(0, unique_pairs.shape[0], self.set1_pair_batch):
            end = min(start + self.set1_pair_batch, unique_pairs.shape[0])
            s = unique_pairs[start:end, 0].to(torch.long)
            t = unique_pairs[start:end, 1].to(torch.int32)
            matches = self.V[s] == t.unsqueeze(1)
            set1_counts[start:end] = matches.sum(dim=1)

            _, a_idx = matches.nonzero(as_tuple=True)
            if a_idx.numel() != 0:
                set1_value_parts.append(a_idx)

        set1_offsets = torch.empty(num_pairs + 1, device=self.device, dtype=torch.long)
        set1_offsets[0] = 0
        set1_offsets[1:] = torch.cumsum(set1_counts, dim=0)

        if set1_value_parts:
            set1_values = torch.cat(set1_value_parts)
        else:
            set1_values = torch.empty(0, device=self.device, dtype=torch.long)

        return pair_inverse, set1_counts, set1_offsets, set1_values

    def _sample_set1_choices(
        self,
        B_free,
        pair_inverse,
        set1_counts,
        set1_offsets,
        set1_values,
        choose_set1,
    ):
        selected_b = B_free[choose_set1]
        if selected_b.numel() == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty

        selected_pair = pair_inverse[choose_set1]
        selected_counts = set1_counts[selected_pair]

        rand_idx = (
            torch.rand(selected_b.numel(), device=self.device)
            * selected_counts.float()
        ).to(torch.long)
        rand_idx.clamp_(max=selected_counts - 1)
        value_idx = set1_offsets[selected_pair] + rand_idx
        return selected_b, set1_values[value_idx]

    def _set1_delta_groups(self, B_free):
        free_s = self.nearest_s[B_free]
        free_t = 1 - self.y_B[B_free] - self.d_min_b_int[B_free]

        order = torch.argsort(free_s)
        sorted_pairs = torch.stack(
            (free_s[order], free_t[order].to(torch.long)),
            dim=1,
        )
        unique_pairs, inverse_sorted = torch.unique(
            sorted_pairs, dim=0, return_inverse=True
        )

        pair_inverse = torch.empty_like(inverse_sorted)
        pair_inverse[order] = inverse_sorted
        return unique_pairs, pair_inverse

    def _compute_delta(self, B_free, unique_pairs, pair_inverse):
        num_free = B_free.numel()
        if num_free == 0:
            return 1

        pair_s = unique_pairs[:, 0].to(torch.long)
        v_pair_row_max = self.V[pair_s].max(dim=1).values.to(torch.long)
        target1 = (
            self.d_min_b_int[B_free] + 1 - self.y_B[B_free]
        ).to(torch.long)
        min_slack1_per_blue = target1 - v_pair_row_max[pair_inverse]

        sentinel = torch.iinfo(torch.int64).max // 4
        min_adj_term = torch.full(
            (num_free,), sentinel, device=self.device, dtype=torch.long
        )

        starts = self.adj_ptr[B_free]
        ends = self.adj_ptr[B_free + 1]
        lengths = ends - starts
        total_edges = int(lengths.sum().item())
        if total_edges != 0:
            edge_range = _ensure_long_arange(
                self, "_delta_set2_edge_arange", total_edges, self.device
            )
            free_pos = _ensure_long_arange(
                self, "_delta_set2_free_pos", num_free, self.device
            )
            cum_len = torch.cumsum(lengths, dim=0)
            packed_starts = cum_len - lengths

            active_free_pos = torch.repeat_interleave(free_pos, lengths)
            active_edge_idx = (
                torch.repeat_interleave(starts, lengths)
                + edge_range
                - torch.repeat_interleave(packed_starts, lengths)
            )

            active_a = self.adj_col[active_edge_idx]
            adj_term = (
                self.adj_dist_int[active_edge_idx].to(torch.long)
                - self.y_A[active_a].to(torch.long)
            )
            min_adj_term.scatter_reduce_(
                0, active_free_pos, adj_term, reduce="amin", include_self=True
            )

        min_slack2_per_blue = (
            1 - self.y_B[B_free].to(torch.long) + min_adj_term
        )
        min_slack_per_blue = torch.minimum(
            min_slack1_per_blue, min_slack2_per_blue
        )

        positive_mask = min_slack_per_blue > 0
        if positive_mask.any().item():
            delta = int(min_slack_per_blue[positive_mask].min().item())
            return max(delta, 1)
        return 1

    def _set2_counts(self, B_free):
        num_free = B_free.numel()
        set2_count_per_blue = torch.zeros(
            num_free, device=self.device, dtype=torch.long
        )

        starts = self.adj_ptr[B_free]
        ends = self.adj_ptr[B_free + 1]
        lengths = ends - starts
        total_edges = int(lengths.sum().item())
        if total_edges == 0:
            return set2_count_per_blue

        edge_range = _ensure_long_arange(
            self, "_set2_edge_arange", total_edges, self.device
        )
        free_pos = _ensure_long_arange(self, "_set2_free_pos", num_free, self.device)
        cum_len = torch.cumsum(lengths, dim=0)
        packed_starts = cum_len - lengths

        active_free_pos = torch.repeat_interleave(free_pos, lengths)
        active_b = B_free[active_free_pos]
        active_edge_idx = (
            torch.repeat_interleave(starts, lengths)
            + edge_range
            - torch.repeat_interleave(packed_starts, lengths)
        )

        active_a = self.adj_col[active_edge_idx]
        target_y_a = self.adj_dist_int[active_edge_idx] + 1 - self.y_B[active_b]
        is_candidate = self.y_A[active_a] == target_y_a
        cand_free_pos = active_free_pos[is_candidate]

        if cand_free_pos.numel() != 0:
            set2_count_per_blue.scatter_add_(
                0, cand_free_pos, torch.ones_like(cand_free_pos)
            )
        return set2_count_per_blue

    def _set2_sample(self, B_free, choose_set2):
        selected_b = B_free[choose_set2]
        num_selected = selected_b.numel()
        if num_selected == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty

        starts = self.adj_ptr[selected_b]
        ends = self.adj_ptr[selected_b + 1]
        lengths = ends - starts
        total_edges = int(lengths.sum().item())
        if total_edges == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty

        edge_range = _ensure_long_arange(
            self, "_set2_sample_edge_arange", total_edges, self.device
        )
        selected_pos = _ensure_long_arange(
            self, "_set2_sample_pos", num_selected, self.device
        )
        cum_len = torch.cumsum(lengths, dim=0)
        packed_starts = cum_len - lengths

        active_selected_pos = torch.repeat_interleave(selected_pos, lengths)
        active_b = selected_b[active_selected_pos]
        active_edge_idx = (
            torch.repeat_interleave(starts, lengths)
            + edge_range
            - torch.repeat_interleave(packed_starts, lengths)
        )

        active_a = self.adj_col[active_edge_idx]
        target_y_a = self.adj_dist_int[active_edge_idx] + 1 - self.y_B[active_b]
        is_candidate = self.y_A[active_a] == target_y_a
        cand_selected_pos = active_selected_pos[is_candidate]
        cand_a = active_a[is_candidate]
        cand_count = cand_a.numel()
        if cand_count == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty

        cand_idx = _ensure_long_arange(
            self, "_set2_sample_cand_arange", cand_count, self.device
        )

        rand_prio = torch.rand(cand_count, device=self.device)
        min_prio = torch.full((num_selected,), float("inf"), device=self.device)
        min_prio.scatter_reduce_(
            0, cand_selected_pos, rand_prio, reduce="amin", include_self=True
        )

        is_min = rand_prio == min_prio[cand_selected_pos]
        winner_idx = torch.full(
            (num_selected,), cand_count, device=self.device, dtype=torch.long
        )
        winner_idx.scatter_reduce_(
            0,
            cand_selected_pos[is_min],
            cand_idx[is_min],
            reduce="amin",
            include_self=True,
        )

        valid = winner_idx < cand_count
        return selected_b[valid], cand_a[winner_idx[valid]]

    def _resolve_conflicts(self, proposal_a, proposal_b):
        num_props = proposal_a.numel()
        prop_idx = _ensure_long_arange(self, "_proposal_arange", num_props, self.device)
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

        accepted = prop_idx == accepted_idx[proposal_a]
        return proposal_a[accepted], proposal_b[accepted]

    def _update_matching(self, B_free, r_new, b_new):
        was_matched = self.match_A[r_new] != -1
        evicted_b = self.match_A[r_new[was_matched]].clone()
        if evicted_b.numel() != 0:
            self.match_B[evicted_b] = -1

        self.match_A[r_new] = b_new
        self.match_B[b_new] = r_new

        keep_free = _ensure_bool_buffer(
            self, "_keep_free_mask", B_free.numel(), self.device
        )
        keep_free.fill_(True)
        keep_free[torch.searchsorted(B_free, b_new)] = False
        still_free = B_free[keep_free]

        if evicted_b.numel() == 0:
            return still_free
        F_B_new, _ = torch.sort(torch.cat([still_free, evicted_b]))
        return F_B_new

    def cleanup_remaining_points(self):
        free_b = torch.nonzero(self.match_B == -1).squeeze(1)
        free_a = torch.nonzero(self.match_A == -1).squeeze(1)
        count = min(free_b.numel(), free_a.numel())
        if count > 0:
            self.match_B[free_b[:count]] = free_a[:count]
            self.cleanup_blues = free_b[:count].clone()
            self.match_A[free_a[:count]] = free_b[:count]

    def calculate_final_stats(self):
        dists = torch.norm(self.P_blue - self.P_red[self.match_B], p=2, dim=1)
        dists = dists * self.diameter
        total_cost = dists.sum()
        avg_cost = total_cost / self.N
        print(f"Total Euclidean Cost: {total_cost.item():.4f}")
        print(f"Avg Euclidean Cost: {avg_cost.item():.4f}")

    def verify_solution(self):
        """
        Post-solve correctness checks based on epsilon-complementary slackness.

        Three checks are run:

        Check 1 — Global dual feasibility (all CSR / Set 2 edges):
            For every blue b and every a in b's adjacency list:
                slack = adj_dist_int[entry] + 1 - y_B[b] - y_A[a]  >=  0
            A negative slack means the feasibility invariant was violated — a
            fundamental bug. Reports count and worst-case value.

        Check 2 — Tightness of phase-matched edges:
            For every matched (b, a) pair that was NOT cleanup-matched:
                slack = cost_int(b, a) - y_B[b] - y_A[a]  ==  0
            At the moment of matching, admissibility gave:
                y_B[b] + y_A[a] == cost(b, a) + 1
            Then y_A[a] was immediately decremented by 1, giving slack exactly 0.
            y_B[b] never increments again (b left F_B) and y_A[a] never decrements
            again (a is matched). So any non-zero slack here is a bug.
            Reports count of violations and the slack distribution.

        Check 3 — Cleanup edges are feasible but not tight:
            For cleanup-matched (b, a) pairs, slack >= 0 must hold but tightness
            is not expected. Reports slack distribution separately.
        """
        device = self.device
        N = self.N

        print("\n[Verify] Running solution verification...")

        # ── Check 1: Global dual feasibility over all CSR edges ──────────────
        num_free = N  # check all blues, not just free ones
        all_b = torch.arange(N, device=device, dtype=torch.long)
        starts = self.adj_ptr[all_b]
        ends   = self.adj_ptr[all_b + 1]
        lengths = ends - starts
        total_edges = int(lengths.sum().item())
        n_violations = 0
        worst = None

        if total_edges > 0:
            edge_range   = torch.arange(total_edges, device=device, dtype=torch.long)
            cum_len      = torch.cumsum(lengths, dim=0)
            packed_starts = cum_len - lengths
            active_b_pos = torch.repeat_interleave(all_b, lengths)
            active_edge_idx = (
                torch.repeat_interleave(starts, lengths)
                + edge_range
                - torch.repeat_interleave(packed_starts, lengths)
            )
            active_a   = self.adj_col[active_edge_idx]
            slack_set2 = (
                self.adj_dist_int[active_edge_idx].to(torch.long)
                + 1
                - self.y_B[active_b_pos].to(torch.long)
                - self.y_A[active_a].to(torch.long)
            )
            n_violations = int((slack_set2 < 0).sum().item())
            worst        = int(slack_set2.min().item())
            print(f"[Verify] Check 1 — Feasibility violations: {n_violations} "
                  f"(worst slack = {worst})")
        else:
            print("[Verify] Check 1 — No CSR edges to check.")

        # ── Check 2 & 3: Slack of matched edges ──────────────────────────────
        matched_b_mask = self.match_B != -1
        matched_b      = torch.nonzero(matched_b_mask, as_tuple=True)[0]
        matched_a      = self.match_B[matched_b]

        if matched_b.numel() == 0:
            print("[Verify] No matched edges to check.")
            return {
                "check1_violations": n_violations,
                "check1_worst_slack": worst,
                "check2_total": 0,
                "check2_violations": 0,
                "check2_min": None,
                "check2_max": None,
                "check3_total": 0,
                "check3_violations": 0,
                "check3_min": None,
                "check3_max": None,
            }

        # ── Determine proxy cost for each matched edge ────────────────────────
        # Step 1: encode all CSR entries as b*N + a for O(M) membership lookup
        N = self.N
        all_b_for_adj = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            self.adj_ptr[1:] - self.adj_ptr[:-1],
        )
        adj_keys = all_b_for_adj * N + self.adj_col  # (M,) unique keys

        # Step 2: encode matched pairs the same way
        matched_keys = matched_b * N + matched_a  # (num_matched,)

        # Step 3: membership test — which matched pairs are in the adj list
        in_adj = torch.isin(matched_keys, adj_keys)  # (num_matched,) bool

        # Step 4: for in-adj pairs, retrieve adj_dist_int via sorted binary search
        proxy_cost = torch.zeros(matched_b.numel(), device=device, dtype=torch.long)

        if in_adj.any():
            sort_idx     = torch.argsort(adj_keys)
            sorted_keys  = adj_keys[sort_idx]
            sorted_dists = self.adj_dist_int[sort_idx].to(torch.long)
            pos = torch.searchsorted(sorted_keys, matched_keys[in_adj])
            proxy_cost[in_adj] = sorted_dists[pos]

        # Step 5: for non-adj pairs, use triangle proxy through nearest sampled center
        #   proxy = d_min_b_int[b] + DR_int[nearest_s[b]][a]
        #   DR_int[s][a] = y_A[a] - V[s][a]   (V definition: V[s][a] = y_A[a] - DR_int[s][a])
        if (~in_adj).any():
            na_b = matched_b[~in_adj]
            na_a = matched_a[~in_adj]
            s_na = self.nearest_s[na_b]
            dr_int_na = (
                self.y_A[na_a].to(torch.long)
                - self.V[s_na, na_a].to(torch.long)
            )
            proxy_cost[~in_adj] = (
                self.d_min_b_int[na_b].to(torch.long) + dr_int_na
            )

        # Step 6: compute slack against proxy cost
        #   For phase-matched edges, tightness was established at match time:
        #     y_B[b] + y_A[a] == proxy_cost(b,a) + 1
        #   Then y_A[a] -= 1 immediately, giving slack = proxy_cost - y_B - y_A = 0
        slack_matched = (
            proxy_cost
            - self.y_B[matched_b].to(torch.long)
            - self.y_A[matched_a].to(torch.long)
        )

        # Separate phase-matched from cleanup-matched
        is_cleanup = torch.zeros(N, dtype=torch.bool, device=device)
        if self.cleanup_blues.numel() > 0:
            is_cleanup[self.cleanup_blues] = True
        matched_is_cleanup = is_cleanup[matched_b]

        phase_slack   = slack_matched[~matched_is_cleanup]
        cleanup_slack = slack_matched[ matched_is_cleanup]

        # Check 2: phase-matched edges must have slack == 0
        if phase_slack.numel() > 0:
            n_nonzero = int((phase_slack != 0).sum().item())
            print(f"[Verify] Check 2 — Phase-matched edges: {phase_slack.numel()} total, "
                  f"{n_nonzero} with slack != 0 (must be 0). "
                  f"Min={int(phase_slack.min().item())} "
                  f"Max={int(phase_slack.max().item())}")
        else:
            print("[Verify] Check 2 — No phase-matched edges.")

        # Check 3: cleanup edges, feasibility only
        if cleanup_slack.numel() > 0:
            n_neg = int((cleanup_slack < 0).sum().item())
            print(f"[Verify] Check 3 — Cleanup-matched edges: {cleanup_slack.numel()} total, "
                  f"{n_neg} with slack < 0 (feasibility violations). "
                  f"Min={int(cleanup_slack.min().item())} "
                  f"Max={int(cleanup_slack.max().item())}")
        else:
            print("[Verify] Check 3 — No cleanup-matched edges.")

        return {
            "check1_violations": n_violations,
            "check1_worst_slack": worst,
            "check2_total": phase_slack.numel() if phase_slack.numel() > 0 else 0,
            "check2_violations": n_nonzero if phase_slack.numel() > 0 else 0,
            "check2_min": int(phase_slack.min().item()) if phase_slack.numel() > 0 else None,
            "check2_max": int(phase_slack.max().item()) if phase_slack.numel() > 0 else None,
            "check3_total": cleanup_slack.numel() if cleanup_slack.numel() > 0 else 0,
            "check3_violations": n_neg if cleanup_slack.numel() > 0 else 0,
            "check3_min": int(cleanup_slack.min().item()) if cleanup_slack.numel() > 0 else None,
            "check3_max": int(cleanup_slack.max().item()) if cleanup_slack.numel() > 0 else None,
        }
