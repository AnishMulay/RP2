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
        clustering_class=None,
        precomputed_clustering=None,
    ):
        if clustering_class is None:
            from ..clustering.simple import SimpleClustering as clustering_class
        self._clustering_class = clustering_class

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        if precomputed_clustering is not None:
            self.N = int(precomputed_clustering["adj_ptr"].shape[0]) - 1
            self.device = precomputed_clustering["adj_ptr"].device
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

            if self.verbose:
                print(
                    "=" * 60
                    + f"\n[Init Simple] N={self.N}, epsilon={self.epsilon}, "
                    + f"tile={self.batch_size}, device={self.device}"
                )

            if (
                A is not None
                and B is not None
                and A.device == self.device
                and B.device == self.device
                and A.ndim == 2
                and B.ndim == 2
                and A.shape == B.shape
                and A.shape[0] == self.N
                and A.is_floating_point()
                and B.is_floating_point()
            ):
                self.P_red = A
                self.P_blue = B
            elif self.verbose:
                self.P_red = torch.zeros(
                    self.N, 1, device=self.device, dtype=torch.float32
                )
                self.P_blue = torch.zeros(
                    self.N, 1, device=self.device, dtype=torch.float32
                )

            clustering = precomputed_clustering
        else:
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
            cluster_engine = self._clustering_class(
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

        del clustering
        if precomputed_clustering is None:
            del cluster_engine
        gc.collect()

        self.y_A = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.y_B = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.match_A = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
        self.match_B = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
        self.cleanup_blues = torch.empty(0, device=self.device, dtype=torch.long)
        self.phase_match_is_set1 = torch.zeros(
            self.N, device=self.device, dtype=torch.bool
        )

        # Existing experiment code uses MA/MB and yA/yB naming.
        self.yA = self.y_A
        self.yB = self.y_B
        self.MA = self.match_A
        self.MB = self.match_B
        self.iterations = 0
        self.debug_audit = False
        self.debug_stop_on_first_violation = True
        # self.debug_audit = True
        self._debug_bad_checkpoint_seen = False
        self._debug_last_B_free = None
        self._debug_last_r_new = None
        self._debug_last_b_new = None
        self._debug_last_F_B_new = None
        self._debug_last_evicted_b = None
        self._debug_last_iteration = None
        self._debug_last_choose_set1_by_b = None
        self._debug_last_choose_set2_by_b = None
        self._debug_last_set1_count_by_b = None
        self._debug_last_set2_count_by_b = None
        self._debug_last_set1_eligible_by_b = None
        self._debug_last_proposal_a = None
        self._debug_last_proposal_b = None
        self._debug_last_proposal_is_set1 = None
        self._debug_last_proposal_keys = None
        self._debug_last_proposed_any_by_b = None
        self._debug_last_proposed_a_by_b = None
        self._debug_last_proposed_is_set1_by_b = None
        self._debug_last_accepted_by_b = None
        self._debug_last_lost_conflict_by_b = None
        self._debug_last_y_A_at_proposal = None
        self._debug_last_y_B_at_proposal = None
        self._debug_last_V_at_proposal = None

    def solve(self):
        N = self.N
        device = self.device
        B_free = torch.arange(N, device=device, dtype=torch.long)
        total_set2_admissible_edges = 0
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
            iteration += 1
            if self.debug_audit:
                self.audit_full_feasibility(
                    "phase_start",
                    iteration,
                    require_matched_equality=True,
                )
                self._snapshot_phase_context(B_free=B_free, reset_phase=True)

            pair_inverse, set1_counts, set1_offsets, set1_values = (
                self._set1_groups(B_free)
            )
            set1_count_per_blue = set1_counts[pair_inverse]
            set2_count_per_blue = self._set2_counts(B_free)
            if self.debug_audit:
                set1_eligible = self._set1_eligible_mask(B_free)
            total_set2_admissible_edges += int(set2_count_per_blue.sum().item())

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
                if self.debug_audit:
                    empty_long = torch.empty(0, device=device, dtype=torch.long)
                    empty_bool = torch.empty(0, device=device, dtype=torch.bool)
                    self._snapshot_proposal_context(
                        iteration,
                        B_free,
                        choose_set1,
                        choose_set2,
                        set1_count_per_blue,
                        set2_count_per_blue,
                        set1_eligible,
                        empty_long,
                        empty_long,
                        empty_bool,
                    )
                    self.audit_full_feasibility(
                        "after_proposal_generation_before_conflict_resolution",
                        iteration,
                        require_matched_equality=True,
                    )
                    self._snapshot_phase_context(F_B_new=B_free)
                    self.audit_full_feasibility(
                        "after_matching_update_before_dual_updates",
                        iteration,
                        require_matched_equality=True,
                    )
                unique_delta_pairs, delta_pair_inverse = self._set1_delta_groups(B_free)
                delta = self._compute_delta(
                    B_free, unique_delta_pairs, delta_pair_inverse
                )
                self.y_B[B_free] += delta
                if self.debug_audit:
                    self.audit_full_feasibility(
                        "after_dual_updates",
                        iteration,
                        require_matched_equality=True,
                    )
                _print_progress(iteration, num_free, num_free, "no_proposals")
                continue

            proposal_a = torch.cat(proposal_a_parts)
            proposal_b = torch.cat(proposal_b_parts)

            proposal_is_set1_parts = []

            if a1.numel() != 0:
                proposal_is_set1_parts.append(
                    torch.ones(a1.numel(), device=device, dtype=torch.bool)
                )

            if a2.numel() != 0:
                proposal_is_set1_parts.append(
                    torch.zeros(a2.numel(), device=device, dtype=torch.bool)
                )

            proposal_is_set1 = torch.cat(proposal_is_set1_parts)

            if self.debug_audit:
                self._snapshot_proposal_context(
                    iteration,
                    B_free,
                    choose_set1,
                    choose_set2,
                    set1_count_per_blue,
                    set2_count_per_blue,
                    set1_eligible,
                    proposal_a,
                    proposal_b,
                    proposal_is_set1,
                )
                self.audit_full_feasibility(
                    "after_proposal_generation_before_conflict_resolution",
                    iteration,
                    require_matched_equality=True,
                )

            r_new, b_new = self._resolve_conflicts(proposal_a, proposal_b)
            if self.debug_audit:
                self._snapshot_phase_context(r_new=r_new, b_new=b_new)
                self._snapshot_conflict_context(r_new, b_new)
                self.audit_full_feasibility(
                    "after_conflict_resolution_before_matching_update",
                    iteration,
                    require_matched_equality=True,
                )

            if r_new.numel() == 0:
                if self.debug_audit:
                    self._snapshot_phase_context(F_B_new=B_free)
                    self.audit_full_feasibility(
                        "after_matching_update_before_dual_updates",
                        iteration,
                        require_matched_equality=True,
                    )
                unique_delta_pairs, delta_pair_inverse = self._set1_delta_groups(B_free)
                delta = self._compute_delta(
                    B_free, unique_delta_pairs, delta_pair_inverse
                )
                self.y_B[B_free] += delta
                if self.debug_audit:
                    self.audit_full_feasibility(
                        "after_dual_updates",
                        iteration,
                        require_matched_equality=True,
                    )
                _print_progress(iteration, num_free, num_free, "no_accepts")
                continue

            accepted_keys = b_new.to(torch.long) * self.N + r_new.to(torch.long)
            proposal_keys = proposal_b.to(torch.long) * self.N + proposal_a.to(torch.long)

            sort_prop_idx = torch.argsort(proposal_keys)
            sorted_prop_keys = proposal_keys[sort_prop_idx]

            accepted_pos = torch.searchsorted(sorted_prop_keys, accepted_keys)
            accepted_prop_idx = sort_prop_idx[accepted_pos]

            accepted_is_set1 = proposal_is_set1[accepted_prop_idx]
            self.phase_match_is_set1[b_new] = accepted_is_set1

            F_B_new = self._update_matching(B_free, r_new, b_new)
            if self.debug_audit:
                self._snapshot_phase_context(F_B_new=F_B_new)
                self.audit_full_feasibility(
                    "after_matching_update_before_dual_updates",
                    iteration,
                    require_matched_equality=False,
                )

            unique_delta_pairs, delta_pair_inverse = self._set1_delta_groups(F_B_new)
            delta = self._compute_delta(
                F_B_new, unique_delta_pairs, delta_pair_inverse
            )
            self.y_B[F_B_new] += delta
            self.y_A[r_new] -= 1
            self.V[:, r_new] -= 1
            if self.debug_audit:
                self.audit_full_feasibility(
                    "after_dual_updates",
                    iteration,
                    require_matched_equality=True,
                )

            _print_progress(iteration, num_free, F_B_new.numel(), "ok")
            B_free = F_B_new

        # print(f"[Diag] Total Set 2 admissible edges found across all phases: {total_set2_admissible_edges}")
        self.iterations = iteration
        verify_results = None
        if self.debug_audit:
            verify_results = self.verify_solution()
        self._last_verify = verify_results
        self.cleanup_remaining_points()
        if self.verbose:
            print(f"[Simple] Matched: {(self.match_B != -1).sum().item()}/{self.N}")
            self.calculate_final_stats()
        return self.match_B

    def _set1_eligible_mask(self, B_free):
        num_free = B_free.numel()
        if num_free == 0:
            return torch.empty(0, device=self.device, dtype=torch.bool)

        free_s = self.nearest_s[B_free]
        free_score = (
            self.y_B[B_free].to(torch.long)
            - self.d_min_b_int[B_free].to(torch.long)
        )
        max_score_by_s = torch.full(
            (self.V.shape[0],),
            torch.iinfo(torch.int64).min,
            device=self.device,
            dtype=torch.long,
        )
        max_score_by_s.scatter_reduce_(
            0,
            free_s,
            free_score,
            reduce="amax",
            include_self=True,
        )
        return free_score == max_score_by_s[free_s]

    def _set1_groups(self, B_free):
        num_free = B_free.numel()
        eligible = self._set1_eligible_mask(B_free)
        eligible_pos = torch.nonzero(eligible, as_tuple=True)[0]

        pair_inverse = torch.zeros(num_free, device=self.device, dtype=torch.long)
        if eligible_pos.numel() == 0:
            set1_counts = torch.zeros(1, device=self.device, dtype=torch.long)
            set1_offsets = torch.zeros(2, device=self.device, dtype=torch.long)
            set1_values = torch.empty(0, device=self.device, dtype=torch.long)
            return pair_inverse, set1_counts, set1_offsets, set1_values

        B_eligible = B_free[eligible_pos]
        free_s = self.nearest_s[B_eligible]
        free_t = self.d_min_b_int[B_eligible] + 1 - self.y_B[B_eligible]

        order = torch.argsort(free_s)
        sorted_pairs = torch.stack(
            (free_s[order], free_t[order].to(torch.long)),
            dim=1,
        )
        unique_pairs, inverse_sorted = torch.unique(
            sorted_pairs, dim=0, return_inverse=True
        )

        eligible_pair_inverse = torch.empty_like(inverse_sorted)
        eligible_pair_inverse[order] = inverse_sorted

        num_pairs = unique_pairs.shape[0]
        pair_inverse.fill_(num_pairs)
        pair_inverse[eligible_pos] = eligible_pair_inverse

        set1_counts = torch.zeros(num_pairs + 1, device=self.device, dtype=torch.long)
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

        set1_offsets = torch.empty(num_pairs + 2, device=self.device, dtype=torch.long)
        set1_offsets[0] = 0
        set1_offsets[1:] = torch.cumsum(set1_counts, dim=0)

        if set1_value_parts:
            set1_values = torch.cat(set1_value_parts)
        else:
            set1_values = torch.empty(0, device=self.device, dtype=torch.long)

        return pair_inverse, set1_counts, set1_offsets, set1_values

    def _set1_delta_groups(self, B_free):
        eligible = self._set1_eligible_mask(B_free)
        B_eligible = B_free[eligible]
        if B_eligible.numel() == 0:
            unique_pairs = torch.empty(0, 2, device=self.device, dtype=torch.long)
            pair_inverse = torch.zeros(B_free.numel(), device=self.device, dtype=torch.long)
            return unique_pairs, pair_inverse

        free_s = self.nearest_s[B_eligible]
        free_t = self.d_min_b_int[B_eligible] + 1 - self.y_B[B_eligible]

        order = torch.argsort(free_s)
        sorted_pairs = torch.stack(
            (free_s[order], free_t[order].to(torch.long)),
            dim=1,
        )
        unique_pairs, inverse_sorted = torch.unique(
            sorted_pairs, dim=0, return_inverse=True
        )

        eligible_pair_inverse = torch.empty_like(inverse_sorted)
        eligible_pair_inverse[order] = inverse_sorted

        pair_inverse = torch.zeros(B_free.numel(), device=self.device, dtype=torch.long)
        pair_inverse[eligible] = eligible_pair_inverse
        return unique_pairs, pair_inverse

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

    def _compute_delta(self, B_free, unique_pairs, pair_inverse):
        num_free = B_free.numel()
        if num_free == 0:
            return 0

        sentinel = torch.iinfo(torch.int64).max // 4
        target1 = (
            self.d_min_b_int[B_free] + 1 - self.y_B[B_free]
        ).to(torch.long)
        min_slack1_per_blue = torch.full(
            (num_free,), sentinel, device=self.device, dtype=torch.long
        )
        if unique_pairs.numel() != 0:
            set1_eligible = self._set1_eligible_mask(B_free)
            if set1_eligible.any().item():
                pair_s = unique_pairs[:, 0].to(torch.long)
                num_pairs = pair_s.numel()
                v_pair_row_max = torch.empty(num_pairs, device=self.device, dtype=torch.long)
                for start in range(0, num_pairs, self.set1_pair_batch):
                    end = min(start + self.set1_pair_batch, num_pairs)
                    v_pair_row_max[start:end] = (
                        self.V[pair_s[start:end]].max(dim=1).values.to(torch.long)
                    )
                min_slack1_per_blue[set1_eligible] = (
                    target1[set1_eligible]
                    - v_pair_row_max[pair_inverse[set1_eligible]]
                )

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

        min_slack2_per_blue = torch.full(
            (num_free,), sentinel, device=self.device, dtype=torch.long
        )
        has_set2_edges = min_adj_term != sentinel
        if has_set2_edges.any().item():
            min_slack2_per_blue[has_set2_edges] = (
                1
                - self.y_B[B_free[has_set2_edges]].to(torch.long)
                + min_adj_term[has_set2_edges]
            )
        min_slack_per_blue = torch.minimum(
            min_slack1_per_blue, min_slack2_per_blue
        )

        valid_mask = min_slack_per_blue != sentinel
        if valid_mask.any().item():
            delta = int(min_slack_per_blue[valid_mask].min().item())
            return max(delta, 0)
        return 0

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
        self._debug_last_evicted_b = evicted_b.detach().clone()
        if evicted_b.numel() != 0:
            self.match_B[evicted_b] = -1
            self.phase_match_is_set1[evicted_b] = False

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

    def _clone_debug_indices(self, values):
        if values is None:
            return None
        return values.detach().to(device=self.device, dtype=torch.long).clone()

    def _snapshot_phase_context(
        self,
        B_free=None,
        r_new=None,
        b_new=None,
        F_B_new=None,
        evicted_b=None,
        reset_phase=False,
    ):
        if reset_phase:
            self._debug_last_r_new = None
            self._debug_last_b_new = None
            self._debug_last_F_B_new = None
            self._debug_last_evicted_b = None
        if B_free is not None:
            self._debug_last_B_free = self._clone_debug_indices(B_free)
        if r_new is not None:
            self._debug_last_r_new = self._clone_debug_indices(r_new)
        if b_new is not None:
            self._debug_last_b_new = self._clone_debug_indices(b_new)
        if F_B_new is not None:
            self._debug_last_F_B_new = self._clone_debug_indices(F_B_new)
        if evicted_b is not None:
            self._debug_last_evicted_b = self._clone_debug_indices(evicted_b)

    def _snapshot_proposal_context(
        self,
        iteration,
        B_free,
        choose_set1,
        choose_set2,
        set1_count_per_blue,
        set2_count_per_blue,
        set1_eligible,
        proposal_a,
        proposal_b,
        proposal_is_set1,
    ):
        N = self.N
        device = self.device

        self._debug_last_iteration = int(iteration)
        self._debug_last_y_A_at_proposal = self.y_A.detach().clone()
        self._debug_last_y_B_at_proposal = self.y_B.detach().clone()
        self._debug_last_V_at_proposal = self.V.detach().clone()

        choose_set1_by_b = torch.zeros(N, device=device, dtype=torch.bool)
        choose_set2_by_b = torch.zeros(N, device=device, dtype=torch.bool)
        set1_count_by_b = torch.full((N,), -1, device=device, dtype=torch.long)
        set2_count_by_b = torch.full((N,), -1, device=device, dtype=torch.long)
        set1_eligible_by_b = torch.zeros(N, device=device, dtype=torch.bool)

        choose_set1_by_b[B_free] = choose_set1
        choose_set2_by_b[B_free] = choose_set2
        set1_count_by_b[B_free] = set1_count_per_blue.to(torch.long)
        set2_count_by_b[B_free] = set2_count_per_blue.to(torch.long)
        set1_eligible_by_b[B_free] = set1_eligible

        proposed_any_by_b = torch.zeros(N, device=device, dtype=torch.bool)
        proposed_a_by_b = torch.full((N,), -1, device=device, dtype=torch.long)
        proposed_is_set1_by_b = torch.zeros(N, device=device, dtype=torch.bool)

        if proposal_b.numel() != 0:
            proposal_b_long = proposal_b.to(torch.long)
            proposed_any_by_b[proposal_b_long] = True
            proposed_a_by_b[proposal_b_long] = proposal_a.to(torch.long)
            proposed_is_set1_by_b[proposal_b_long] = proposal_is_set1

        self._debug_last_choose_set1_by_b = choose_set1_by_b
        self._debug_last_choose_set2_by_b = choose_set2_by_b
        self._debug_last_set1_count_by_b = set1_count_by_b
        self._debug_last_set2_count_by_b = set2_count_by_b
        self._debug_last_set1_eligible_by_b = set1_eligible_by_b
        self._debug_last_proposal_a = proposal_a.detach().to(torch.long).clone()
        self._debug_last_proposal_b = proposal_b.detach().to(torch.long).clone()
        self._debug_last_proposal_is_set1 = proposal_is_set1.detach().clone()
        self._debug_last_proposal_keys = (
            self._debug_last_proposal_b * N + self._debug_last_proposal_a
        )
        self._debug_last_proposed_any_by_b = proposed_any_by_b
        self._debug_last_proposed_a_by_b = proposed_a_by_b
        self._debug_last_proposed_is_set1_by_b = proposed_is_set1_by_b
        self._debug_last_accepted_by_b = torch.zeros(N, device=device, dtype=torch.bool)
        self._debug_last_lost_conflict_by_b = torch.zeros(
            N, device=device, dtype=torch.bool
        )

    def _snapshot_conflict_context(self, r_new, b_new):
        if self._debug_last_proposed_any_by_b is None:
            return
        accepted_by_b = torch.zeros(self.N, device=self.device, dtype=torch.bool)
        if b_new.numel() != 0:
            accepted_by_b[b_new.to(torch.long)] = True
        self._debug_last_accepted_by_b = accepted_by_b
        self._debug_last_lost_conflict_by_b = (
            self._debug_last_proposed_any_by_b & ~accepted_by_b
        )

    def _debug_membership_mask(self, values):
        if values is None:
            return None
        mask = torch.zeros(self.N, device=self.device, dtype=torch.bool)
        if values.numel() != 0:
            mask[values.to(torch.long)] = True
        return mask

    def _debug_set_size(self, values):
        return "N/A" if values is None else int(values.numel())

    def _debug_mask_value(self, mask, idx):
        if mask is None:
            return "N/A"
        return bool(mask[idx].item())

    def _debug_node_intersection_count(self, node_mask, context_mask):
        if context_mask is None:
            return "N/A"
        return int((node_mask & context_mask).sum().item())

    def _debug_edge_touch_count(self, edge_nodes, context_mask):
        if context_mask is None:
            return "N/A"
        return int(context_mask[edge_nodes].sum().item())

    def _debug_top_count_indices(self, counts, limit=10):
        counts_cpu = counts.detach().cpu()
        indices = [i for i, count in enumerate(counts_cpu.tolist()) if count > 0]
        indices.sort(key=lambda i: (-int(counts_cpu[i].item()), i))
        return indices[:limit]

    def _debug_bool_at(self, values, idx):
        if values is None:
            return "N/A"
        return bool(values[idx].item())

    def _debug_int_at(self, values, idx):
        if values is None:
            return "N/A"
        return int(values[idx].item())

    def _debug_adj_info(self, b, a):
        start = int(self.adj_ptr[b].item())
        end = int(self.adj_ptr[b + 1].item())
        if end <= start:
            return False, None
        adj_a = self.adj_col[start:end]
        matches = torch.nonzero(adj_a == a, as_tuple=True)[0]
        if matches.numel() == 0:
            return False, None
        edge_idx = start + int(matches[0].item())
        return True, int(self.adj_dist_int[edge_idx].item())

    def _proposal_time_context_available(self):
        return (
            self._debug_last_y_A_at_proposal is not None
            and self._debug_last_y_B_at_proposal is not None
            and self._debug_last_V_at_proposal is not None
            and self._debug_last_proposed_any_by_b is not None
        )

    def _proposal_time_set2_candidates(self, b):
        if not self._proposal_time_context_available():
            return None, []

        y_A_snap = self._debug_last_y_A_at_proposal.to(torch.long)
        y_B_b = int(self._debug_last_y_B_at_proposal[b].item())
        start = int(self.adj_ptr[b].item())
        end = int(self.adj_ptr[b + 1].item())
        if end <= start:
            return 0, []

        adj_a = self.adj_col[start:end]
        adj_dist = self.adj_dist_int[start:end].to(torch.long)
        is_candidate = y_A_snap[adj_a] == (adj_dist + 1 - y_B_b)
        cand_a = adj_a[is_candidate]
        return int(cand_a.numel()), [int(x.item()) for x in cand_a[:10]]

    def _proposal_time_set1_candidates(self, b):
        if not self._proposal_time_context_available():
            return None, []
        last_B_free_mask = self._debug_membership_mask(self._debug_last_B_free)
        if last_B_free_mask is None:
            return None, []
        if not bool(last_B_free_mask[b].item()):
            return 0, []
        if self._debug_last_set1_eligible_by_b is None:
            return None, []
        if not bool(self._debug_last_set1_eligible_by_b[b].item()):
            return 0, []

        y_B_b = int(self._debug_last_y_B_at_proposal[b].item())
        s = int(self.nearest_s[b].item())
        target = int(self.d_min_b_int[b].item()) + 1 - y_B_b
        row = self._debug_last_V_at_proposal[s].to(torch.long)
        cand_a = torch.nonzero(row == target, as_tuple=True)[0]
        return int(cand_a.numel()), [int(x.item()) for x in cand_a[:10]]

    def _proposal_time_edge_set2_admissible(self, b, a):
        if not self._proposal_time_context_available():
            return "N/A"
        in_adj, adj_dist = self._debug_adj_info(b, a)
        if not in_adj:
            return False
        y_A_a = int(self._debug_last_y_A_at_proposal[a].item())
        y_B_b = int(self._debug_last_y_B_at_proposal[b].item())
        return y_A_a == adj_dist + 1 - y_B_b

    def _proposal_time_edge_set1_admissible(self, b, a):
        if not self._proposal_time_context_available():
            return "N/A"
        if self._debug_last_set1_eligible_by_b is None:
            return "N/A"
        if self._debug_last_B_free is None:
            return "N/A"
        in_B_free = bool(self._debug_membership_mask(self._debug_last_B_free)[b].item())
        if not in_B_free or not bool(self._debug_last_set1_eligible_by_b[b].item()):
            return False

        y_B_b = int(self._debug_last_y_B_at_proposal[b].item())
        s = int(self.nearest_s[b].item())
        target = int(self.d_min_b_int[b].item()) + 1 - y_B_b
        return int(self._debug_last_V_at_proposal[s, a].item()) == target

    def _debug_small_candidate_list_text(self, count, candidates):
        if count is None:
            return "N/A"
        suffix = "" if count <= len(candidates) else " ..."
        return f"{candidates}{suffix}"

    def _print_sampled_feasibility_edges(
        self,
        violation_b,
        violation_a,
        proxy,
        triangle_proxy,
        direct_costs,
        in_adj_mask,
        feas_slack,
        y_B_long,
        y_A_long,
    ):
        sample_count = min(10, int(violation_b.numel()))
        print("\n[Audit] Sample violating feasibility edges")
        for sample_idx in range(sample_count):
            b = int(violation_b[sample_idx].item())
            a = int(violation_a[sample_idx].item())
            matched_red = int(self.match_B[b].item())
            b_is_free = matched_red == -1
            current_matched_edge = matched_red == a
            phase_set1 = (
                "N/A"
                if b_is_free
                else str(bool(self.phase_match_is_set1[b].item()))
            )
            in_adj = bool(in_adj_mask[b, a].item())
            direct_proxy = int(direct_costs[b, a].item()) if in_adj else "N/A"
            triangle_value = int(triangle_proxy[b, a].item())
            chosen_proxy = int(proxy[b, a].item())
            y_b_value = int(y_B_long[b].item())
            y_a_value = int(y_A_long[a].item())
            slack_value = int(feas_slack[b, a].item())
            matched_diff = (
                int(proxy[b, a].item() - y_b_value - y_a_value)
                if current_matched_edge
                else "N/A"
            )
            matched_red_text = "N/A" if b_is_free else matched_red
            b_state = "free" if b_is_free else "matched"
            nearest_s = int(self.nearest_s[b].item())
            d_min = int(self.d_min_b_int[b].item())

            print(f"  edge[{sample_idx}] b={b} a={a}")
            print(
                f"    in_adj={in_adj} b_state={b_state} "
                f"matched_red={matched_red_text} "
                f"current_matched_edge={current_matched_edge} "
                f"phase_match_is_set1={phase_set1}"
            )
            print(
                f"    y_B={y_b_value} y_A={y_a_value} "
                f"direct_proxy={direct_proxy} triangle_proxy={triangle_value} "
                f"chosen_proxy={chosen_proxy}"
            )
            print(
                f"    feas_slack={slack_value} matched_diff={matched_diff} "
                f"nearest_s={nearest_s} d_min_b_int={d_min}"
            )

    def _classify_violating_edge_backtrace(
        self,
        context_available,
        was_b_in_last_B_free,
        did_propose,
        proposed_this_edge,
        proposal_accepted,
        proposal_lost_conflict,
        edge_set1_admissible,
        edge_set2_admissible,
        set1_count,
        set2_count,
    ):
        if not context_available:
            return "cannot determine from retained phase context"
        if was_b_in_last_B_free is not True:
            return "cannot determine from retained phase context"
        edge_was_admissible = (
            edge_set1_admissible is True or edge_set2_admissible is True
        )
        if did_propose is not True:
            if edge_was_admissible:
                return "zero-slack edge existed but blue made no proposal"
            if set1_count == 0 and set2_count == 0:
                return "blue had no admissible edges at proposal time"
            return "blue had admissible edges but made no proposal"
        if proposed_this_edge:
            if proposal_accepted is True:
                return "blue proposed this edge and it was accepted"
            if proposal_lost_conflict is True:
                return "blue proposed this edge and lost conflict"
            return "blue proposed this edge; conflict result unknown"
        return "blue proposed a different edge"

    def _print_violating_edge_backtrace(
        self,
        sample_idx,
        b,
        a,
        proxy,
        feas_slack,
        y_B_long,
        y_A_long,
    ):
        context_available = self._proposal_time_context_available()
        last_B_free_mask = self._debug_membership_mask(self._debug_last_B_free)
        last_F_B_new_mask = self._debug_membership_mask(self._debug_last_F_B_new)
        was_b_in_last_B_free = self._debug_mask_value(last_B_free_mask, b)
        was_b_in_last_F_B_new = self._debug_mask_value(last_F_B_new_mask, b)

        did_propose = self._debug_bool_at(self._debug_last_proposed_any_by_b, b)
        choose_set1 = self._debug_bool_at(self._debug_last_choose_set1_by_b, b)
        choose_set2 = self._debug_bool_at(self._debug_last_choose_set2_by_b, b)
        proposed_red = self._debug_int_at(self._debug_last_proposed_a_by_b, b)
        proposed_is_set1 = self._debug_bool_at(
            self._debug_last_proposed_is_set1_by_b, b
        )
        proposal_kind = "N/A"
        if did_propose is True:
            proposal_kind = "Set1" if proposed_is_set1 else "Set2"
        proposal_accepted = self._debug_bool_at(self._debug_last_accepted_by_b, b)
        proposal_lost_conflict = self._debug_bool_at(
            self._debug_last_lost_conflict_by_b, b
        )

        in_adj, _ = self._debug_adj_info(b, a)
        edge_set2_admissible = self._proposal_time_edge_set2_admissible(b, a)
        edge_set1_admissible = self._proposal_time_edge_set1_admissible(b, a)
        proposed_this_edge = did_propose is True and proposed_red == a
        proposed_this_accepted = proposal_accepted if proposed_this_edge else "N/A"

        set1_count, set1_candidates = self._proposal_time_set1_candidates(b)
        set2_count, set2_candidates = self._proposal_time_set2_candidates(b)
        set1_eligible = self._debug_bool_at(self._debug_last_set1_eligible_by_b, b)
        set1_contains_edge = edge_set1_admissible is True
        set2_contains_edge = edge_set2_admissible is True

        not_proposed_but_admissible = (
            not proposed_this_edge
            and (edge_set1_admissible is True or edge_set2_admissible is True)
        )
        classification = self._classify_violating_edge_backtrace(
            context_available,
            was_b_in_last_B_free,
            did_propose,
            proposed_this_edge,
            proposal_accepted,
            proposal_lost_conflict,
            edge_set1_admissible,
            edge_set2_admissible,
            set1_count,
            set2_count,
        )

        current_y_B = int(y_B_long[b].item())
        current_y_A = int(y_A_long[a].item())
        current_proxy = int(proxy[b, a].item())
        current_slack = int(feas_slack[b, a].item())

        proposal_y_B = self._debug_int_at(self._debug_last_y_B_at_proposal, b)
        proposal_y_A = self._debug_int_at(self._debug_last_y_A_at_proposal, a)

        print(f"\n[Audit] Backtrace for violating edge[{sample_idx}] (b={b}, a={a})")
        print(
            "  current_bad_state: "
            f"y_B={current_y_B} y_A={current_y_A} "
            f"chosen_proxy={current_proxy} feas_slack={current_slack}"
        )
        print(
            "  proposal_time_state: "
            f"retained_iteration={self._debug_last_iteration} "
            f"proposal_y_B={proposal_y_B} proposal_y_A={proposal_y_A}"
        )
        print(
            "  blue_proposal_status: "
            f"in_last_B_free={was_b_in_last_B_free} "
            f"in_last_F_B_new={was_b_in_last_F_B_new} "
            f"choose_set1={choose_set1} choose_set2={choose_set2} "
            f"did_propose={did_propose} proposed_red={proposed_red} "
            f"proposal_kind={proposal_kind} accepted={proposal_accepted} "
            f"lost_conflict={proposal_lost_conflict}"
        )
        print(
            "  specific_edge_status: "
            f"in_adj={in_adj} "
            f"set2_admissible_at_proposal={edge_set2_admissible} "
            f"set1_admissible_at_proposal={edge_set1_admissible} "
            f"actually_proposed={proposed_this_edge} "
            f"proposed_edge_accepted={proposed_this_accepted} "
            f"not_proposed_but_admissible={not_proposed_but_admissible}"
        )
        print(
            "  candidate_summary_for_b: "
            f"set1_eligible={set1_eligible} set1_count={set1_count} "
            f"set1_candidates_cap10={self._debug_small_candidate_list_text(set1_count, set1_candidates)} "
            f"set1_contains_edge={set1_contains_edge}"
        )
        print(
            "  candidate_summary_for_b: "
            f"set2_count={set2_count} "
            f"set2_candidates_cap10={self._debug_small_candidate_list_text(set2_count, set2_candidates)} "
            f"set2_contains_edge={set2_contains_edge}"
        )
        print(f"  classification={classification}")

    def _print_sampled_edge_backtraces(
        self,
        violation_b,
        violation_a,
        proxy,
        feas_slack,
        y_B_long,
        y_A_long,
    ):
        sample_count = min(10, int(violation_b.numel()))
        print("\n[Audit] Proposal-time backtrace for sampled violating edges")
        for sample_idx in range(sample_count):
            b = int(violation_b[sample_idx].item())
            a = int(violation_a[sample_idx].item())
            self._print_violating_edge_backtrace(
                sample_idx,
                b,
                a,
                proxy,
                feas_slack,
                y_B_long,
                y_A_long,
            )

    def _print_feasibility_concentration(
        self,
        violation_b,
        violation_a,
        y_B_long,
        y_A_long,
        last_B_free_mask,
        last_b_new_mask,
        last_F_B_new_mask,
        last_evicted_b_mask,
        last_r_new_mask,
    ):
        N = self.N
        blue_counts = torch.bincount(violation_b, minlength=N)
        red_counts = torch.bincount(violation_a, minlength=N)
        distinct_blue_count = int((blue_counts > 0).sum().item())
        distinct_red_count = int((red_counts > 0).sum().item())

        print("\n[Audit] Concentration summary")
        print(f"  distinct_violating_blues={distinct_blue_count}")
        print(f"  distinct_violating_reds={distinct_red_count}")

        print("  top_violating_blues:")
        for b in self._debug_top_count_indices(blue_counts, limit=10):
            count = int(blue_counts[b].item())
            matched_red = int(self.match_B[b].item())
            b_state = "free" if matched_red == -1 else "matched"
            matched_red_text = "N/A" if matched_red == -1 else matched_red
            adj_size = int((self.adj_ptr[b + 1] - self.adj_ptr[b]).item())
            print(
                f"    b={b} count={count} y_B={int(y_B_long[b].item())} "
                f"state={b_state} matched_red={matched_red_text} "
                f"nearest_s={int(self.nearest_s[b].item())} "
                f"d_min_b_int={int(self.d_min_b_int[b].item())} "
                f"adj_size={adj_size} "
                f"in_last_B_free={self._debug_mask_value(last_B_free_mask, b)} "
                f"in_last_F_B_new={self._debug_mask_value(last_F_B_new_mask, b)} "
                f"was_accepted_this_phase={self._debug_mask_value(last_b_new_mask, b)} "
                f"was_evicted_this_phase={self._debug_mask_value(last_evicted_b_mask, b)}"
            )

        print("  top_violating_reds:")
        for a in self._debug_top_count_indices(red_counts, limit=10):
            count = int(red_counts[a].item())
            matched_blue = int(self.match_A[a].item())
            a_state = "free" if matched_blue == -1 else "matched"
            matched_blue_text = "N/A" if matched_blue == -1 else matched_blue
            print(
                f"    a={a} count={count} y_A={int(y_A_long[a].item())} "
                f"state={a_state} matched_blue={matched_blue_text} "
                f"in_last_r_new={self._debug_mask_value(last_r_new_mask, a)}"
            )

    def _print_phase_relation(
        self,
        violation_b,
        violation_a,
        last_B_free_mask,
        last_b_new_mask,
        last_F_B_new_mask,
        last_evicted_b_mask,
        last_r_new_mask,
    ):
        N = self.N
        violating_blue_mask = torch.zeros(N, device=self.device, dtype=torch.bool)
        violating_red_mask = torch.zeros(N, device=self.device, dtype=torch.bool)
        violating_blue_mask[violation_b] = True
        violating_red_mask[violation_a] = True

        print("\n[Audit] Relation to most recent phase update")
        print(
            "  latest_set_sizes: "
            f"B_free={self._debug_set_size(self._debug_last_B_free)} "
            f"r_new={self._debug_set_size(self._debug_last_r_new)} "
            f"b_new={self._debug_set_size(self._debug_last_b_new)} "
            f"F_B_new={self._debug_set_size(self._debug_last_F_B_new)} "
            f"evicted_b={self._debug_set_size(self._debug_last_evicted_b)}"
        )
        print(
            "  distinct_violating_blues_in_last_B_free="
            f"{self._debug_node_intersection_count(violating_blue_mask, last_B_free_mask)}"
        )
        print(
            "  distinct_violating_blues_in_last_F_B_new="
            f"{self._debug_node_intersection_count(violating_blue_mask, last_F_B_new_mask)}"
        )
        print(
            "  distinct_violating_reds_in_last_r_new="
            f"{self._debug_node_intersection_count(violating_red_mask, last_r_new_mask)}"
        )
        print(
            "  violating_edges_touch_blue_in_last_F_B_new="
            f"{self._debug_edge_touch_count(violation_b, last_F_B_new_mask)}"
        )
        print(
            "  violating_edges_touch_red_in_last_r_new="
            f"{self._debug_edge_touch_count(violation_a, last_r_new_mask)}"
        )

    def _print_first_bad_feasibility_report(
        self,
        label,
        iteration,
        first_bad_checkpoint,
        num_feas_violations,
        min_feas_slack,
        num_match_violations,
        violation_b,
        violation_a,
        proxy,
        triangle_proxy,
        direct_costs,
        in_adj_mask,
        feas_slack,
        y_B_long,
        y_A_long,
    ):
        last_B_free_mask = self._debug_membership_mask(self._debug_last_B_free)
        last_r_new_mask = self._debug_membership_mask(self._debug_last_r_new)
        last_b_new_mask = self._debug_membership_mask(self._debug_last_b_new)
        last_F_B_new_mask = self._debug_membership_mask(self._debug_last_F_B_new)
        last_evicted_b_mask = self._debug_membership_mask(self._debug_last_evicted_b)

        print("\n" + "=" * 80)
        print("[Audit] FIRST BAD STATE: feasibility violation detected")
        print(
            f"  iteration={iteration} checkpoint={label} "
            f"feasibility_violating_edges={num_feas_violations} "
            f"min_feasibility_slack={min_feas_slack} "
            f"admissibility_violating_matched_edges={num_match_violations} "
            f"first_detected_bad_checkpoint={first_bad_checkpoint}"
        )

        self._print_sampled_feasibility_edges(
            violation_b,
            violation_a,
            proxy,
            triangle_proxy,
            direct_costs,
            in_adj_mask,
            feas_slack,
            y_B_long,
            y_A_long,
        )
        self._print_sampled_edge_backtraces(
            violation_b,
            violation_a,
            proxy,
            feas_slack,
            y_B_long,
            y_A_long,
        )
        self._print_feasibility_concentration(
            violation_b,
            violation_a,
            y_B_long,
            y_A_long,
            last_B_free_mask,
            last_b_new_mask,
            last_F_B_new_mask,
            last_evicted_b_mask,
            last_r_new_mask,
        )
        self._print_phase_relation(
            violation_b,
            violation_a,
            last_B_free_mask,
            last_b_new_mask,
            last_F_B_new_mask,
            last_evicted_b_mask,
            last_r_new_mask,
        )
        print("=" * 80)

    def _print_matched_equality_violation(
        self,
        label,
        iteration,
        proxy,
        y_B_long,
        y_A_long,
        worst_matched_pair,
    ):
        b, a = worst_matched_pair
        diff = int(proxy[b, a].item() - y_B_long[b].item() - y_A_long[a].item())
        print("\n" + "=" * 80)
        print("[Audit] MATCHED ADMISSIBILITY VIOLATION")
        print(
            f"  iteration={iteration} checkpoint={label} "
            f"b={b} a={a} matched_diff={diff} "
            f"y_B={int(y_B_long[b].item())} y_A={int(y_A_long[a].item())} "
            f"proxy={int(proxy[b, a].item())} "
            f"phase_match_is_set1={bool(self.phase_match_is_set1[b].item())}"
        )
        print("=" * 80)

    def audit_full_feasibility(self, label, iteration, require_matched_equality=False):
        device = self.device
        N = self.N
        total_pairs = N * N

        in_adj_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        direct_costs = torch.zeros(N, N, dtype=torch.long, device=device)
        if self.adj_col.numel() > 0:
            all_b_for_adj = torch.repeat_interleave(
                torch.arange(N, device=device, dtype=torch.long),
                self.adj_ptr[1:] - self.adj_ptr[:-1],
            )
            in_adj_mask[all_b_for_adj, self.adj_col] = True
            direct_costs[all_b_for_adj, self.adj_col] = self.adj_dist_int.to(torch.long)

        y_A_long = self.y_A.to(torch.long)
        y_B_long = self.y_B.to(torch.long)
        V_rows = self.V[self.nearest_s].to(torch.long)
        triangle_proxy = (
            self.d_min_b_int.to(torch.long).unsqueeze(1)
            + y_A_long.unsqueeze(0)
            - V_rows
        )
        proxy = torch.where(in_adj_mask, direct_costs, triangle_proxy)
        feas_slack = proxy + 1 - y_B_long.unsqueeze(1) - y_A_long.unsqueeze(0)

        min_feas_slack = int(feas_slack.min().item())
        feasibility_violating = feas_slack < 0
        num_feas_violations = int(feasibility_violating.sum().item())
        worst_feas_flat = int(torch.argmin(feas_slack).item())
        worst_feas_b = worst_feas_flat // N
        worst_feas_a = worst_feas_flat % N
        worst_feas_pair = (worst_feas_b, worst_feas_a)

        matched_b = torch.nonzero(self.match_B != -1, as_tuple=True)[0]
        if matched_b.numel() > 0:
            matched_a = self.match_B[matched_b]
            matched_proxy = proxy[matched_b, matched_a]
            matched_diff = (
                matched_proxy
                - y_B_long[matched_b]
                - y_A_long[matched_a]
            )
            num_match_violations = int((matched_diff != 0).sum().item())
            matched_abs_diff = matched_diff.abs()
            worst_match_idx = int(torch.argmax(matched_abs_diff).item())
            worst_abs_matched_diff = int(matched_abs_diff[worst_match_idx].item())
            worst_matched_pair = (
                int(matched_b[worst_match_idx].item()),
                int(matched_a[worst_match_idx].item()),
            )
        else:
            matched_diff = torch.empty(0, device=device, dtype=torch.long)
            num_match_violations = 0
            worst_abs_matched_diff = 0
            worst_matched_pair = None

        if num_feas_violations > 0:
            if not self._debug_bad_checkpoint_seen:
                violation_b, violation_a = torch.nonzero(
                    feasibility_violating, as_tuple=True
                )
                first_bad_checkpoint = not self._debug_bad_checkpoint_seen
                self._debug_bad_checkpoint_seen = True
                self._print_first_bad_feasibility_report(
                    label,
                    iteration,
                    first_bad_checkpoint,
                    num_feas_violations,
                    min_feas_slack,
                    num_match_violations,
                    violation_b,
                    violation_a,
                    proxy,
                    triangle_proxy,
                    direct_costs,
                    in_adj_mask,
                    feas_slack,
                    y_B_long,
                    y_A_long,
                )
            if self.debug_stop_on_first_violation:
                raise RuntimeError(
                    f"Feasibility violation at iteration {iteration}, step {label}"
                )

        if (
            require_matched_equality
            and num_match_violations > 0
            and self.debug_stop_on_first_violation
        ):
            self._print_matched_equality_violation(
                label,
                iteration,
                proxy,
                y_B_long,
                y_A_long,
                worst_matched_pair,
            )
            raise RuntimeError(
                f"Matched equality violation at iteration {iteration}, step {label}"
            )

        return {
            "total_pairs": total_pairs,
            "min_feas_slack": min_feas_slack,
            "num_feas_violations": num_feas_violations,
            "worst_feas_pair": worst_feas_pair,
            "num_match_violations": num_match_violations,
            "worst_matched_pair": worst_matched_pair,
            "worst_abs_matched_diff": worst_abs_matched_diff,
        }

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
        Full N² feasibility and admissibility check. Run before cleanup.

        For EVERY pair (b, a) — matched or not:
            y_B[b] + y_A[a]  <=  proxy_cost(b, a) + 1      [feasibility]

        For EVERY phase-matched pair (b, a):
            y_B[b] + y_A[a]  ==  proxy_cost(b, a)           [admissibility]

        Proxy cost:
            a in adj(b)  ->  adj_dist_int[entry]
            otherwise    ->  d_min_b_int[b] + DR_int[nearest_s[b]][a]
                             where DR_int[s][a] = y_A[a] - V[s][a]

        Additionally, for every phase-matched pair that violates admissibility
        and was matched via Set 1: check whether there is currently a zero-slack
        edge in b's adjacency list that was missed.
        """
        device = self.device
        N = self.N
        TILE = 256

        # Build dense N x N adjacency mask and direct cost matrix
        in_adj_mask  = torch.zeros(N, N, dtype=torch.bool,  device=device)
        direct_costs = torch.zeros(N, N, dtype=torch.long,  device=device)
        if self.adj_col.numel() > 0:
            all_b_for_adj = torch.repeat_interleave(
                torch.arange(N, device=device, dtype=torch.long),
                self.adj_ptr[1:] - self.adj_ptr[:-1],
            )
            in_adj_mask[all_b_for_adj, self.adj_col] = True
            direct_costs[all_b_for_adj, self.adj_col] = self.adj_dist_int.to(torch.long)

        # Phase-matched blues
        is_cleanup = torch.zeros(N, dtype=torch.bool, device=device)
        if self.cleanup_blues.numel() > 0:
            is_cleanup[self.cleanup_blues] = True
        matched_b_all = torch.nonzero(self.match_B != -1, as_tuple=True)[0]
        phase_matched_b = matched_b_all[~is_cleanup[matched_b_all]]
        is_phase_matched = torch.zeros(N, dtype=torch.bool, device=device)
        is_phase_matched[phase_matched_b] = True

        y_A_long = self.y_A.to(torch.long)
        y_B_long = self.y_B.to(torch.long)

        total_feasibility_violations   = 0
        total_admissibility_violations = 0
        worst_feasibility_excess       = 0
        worst_admissibility_diff       = 0

        # Collect violating matched pairs for Diagnostic 3
        viol_matched_b_list = []
        viol_matched_a_list = []
        viol_is_set1_list   = []

        for b_start in range(0, N, TILE):
            b_end = min(b_start + TILE, N)
            b_idx = torch.arange(b_start, b_end, device=device, dtype=torch.long)
            tb    = b_end - b_start

            # Triangle proxy for all (b, a) in this tile
            s_tile    = self.nearest_s[b_idx]
            V_rows    = self.V[s_tile]
            tri_proxy = (
                self.d_min_b_int[b_idx].to(torch.long).unsqueeze(1)
                + y_A_long.unsqueeze(0)
                - V_rows.to(torch.long)
            )  # (tb, N)

            # Override with direct cost where a is in adj(b)
            proxy = torch.where(in_adj_mask[b_idx], direct_costs[b_idx], tri_proxy)  # (tb, N)

            lhs = y_B_long[b_idx].unsqueeze(1) + y_A_long.unsqueeze(0)  # (tb, N)

            # --- Diagnostic 1: feasibility over all pairs ---
            excess = lhs - proxy - 1
            feas_viols = int((excess > 0).sum().item())
            total_feasibility_violations += feas_viols
            tile_worst = int(excess.max().item())
            if tile_worst > worst_feasibility_excess:
                worst_feasibility_excess = tile_worst

            # --- Diagnostic 2: admissibility for phase-matched pairs ---
            phase_tile = is_phase_matched[b_idx]
            if phase_tile.any():
                local_idx    = phase_tile.nonzero(as_tuple=True)[0]
                global_b     = b_idx[local_idx]
                matched_a    = self.match_B[global_b]
                lhs_matched  = lhs[local_idx, matched_a]
                prx_matched  = proxy[local_idx, matched_a]
                diff         = lhs_matched - prx_matched
                admis_viols  = int((diff != 0).sum().item())
                total_admissibility_violations += admis_viols
                tile_worst_a = int(diff.abs().max().item()) if diff.numel() > 0 else 0
                if tile_worst_a > worst_admissibility_diff:
                    worst_admissibility_diff = tile_worst_a

                # Collect violating pairs for Diagnostic 3
                viol_mask = diff != 0
                if viol_mask.any():
                    viol_matched_b_list.append(global_b[viol_mask])
                    viol_matched_a_list.append(matched_a[viol_mask])
                    viol_is_set1_list.append(self.phase_match_is_set1[global_b[viol_mask]])

        print(f"\n[Verify] Full N² check over all {N}x{N} = {N*N} pairs:")
        print(
            f"  All pairs     — feasibility  (lhs <= proxy+1): "
            f"{total_feasibility_violations} violations  "
            f"(worst excess = {worst_feasibility_excess})"
        )
        print(
            f"  Matched pairs — admissibility (lhs == proxy):  "
            f"{total_admissibility_violations} violations  "
            f"(worst |diff| = {worst_admissibility_diff})"
        )

        # --- Diagnostic 3: for admissibility-violating matched pairs ---
        # Check: was it matched via Set 1, and is there a missed zero-slack adj edge?
        if viol_matched_b_list:
            viol_b   = torch.cat(viol_matched_b_list)
            viol_a   = torch.cat(viol_matched_a_list)
            viol_s1  = torch.cat(viol_is_set1_list)

            set1_viol_b = viol_b[viol_s1]
            set1_viol_a = viol_a[viol_s1]

            missed_adj_count = 0
            if set1_viol_b.numel() > 0:
                for i in range(set1_viol_b.numel()):
                    b = set1_viol_b[i]
                    start = int(self.adj_ptr[b].item())
                    end   = int(self.adj_ptr[b + 1].item())
                    if end > start:
                        adj_a    = self.adj_col[start:end]
                        adj_dist = self.adj_dist_int[start:end].to(torch.long)
                        lhs_b    = y_B_long[b] + y_A_long[adj_a]
                        if (lhs_b == adj_dist).any():
                            missed_adj_count += 1

            print(
                f"\n[Verify] Of {total_admissibility_violations} admissibility violations:"
            )
            print(
                f"  Matched via Set 1:                        {int(viol_s1.sum().item())}"
            )
            print(
                f"  Matched via Set 2:                        {int((~viol_s1).sum().item())}"
            )
            print(
                f"  Set-1-matched with missed zero-slack adj edge: {missed_adj_count}"
            )
        else:
            print("\n[Verify] No admissibility violations — no Diagnostic 3 needed.")

        return {
            "feasibility_violations":   total_feasibility_violations,
            "feasibility_worst_excess": worst_feasibility_excess,
            "admissibility_violations": total_admissibility_violations,
            "admissibility_worst_diff": worst_admissibility_diff,
        }
