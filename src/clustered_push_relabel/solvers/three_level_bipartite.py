import gc
import time

import torch

from ..clustering.simple_three_level import ThreeLevelClustering
from .simple_bipartite import (
    SimpleGPUSolver,
    _ensure_bool_buffer,
    _ensure_long_arange,
)

_HIST_SIZE_LIMIT = 50_000_000


class ThreeLevelGPUSolver(SimpleGPUSolver):
    """
    Epsilon-approximate GPU bipartite matcher over the ThreeLevelClustering graph.

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
            clustering_class = ThreeLevelClustering
        self._clustering_class = clustering_class

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        if precomputed_clustering is not None:
            self.N = int(precomputed_clustering["adj_B_ptr"].shape[0]) - 1
            self.device = precomputed_clustering["adj_B_ptr"].device
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
                    + f"\n[Init ThreeLevel] N={self.N}, epsilon={self.epsilon}, "
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
                raise ValueError("ThreeLevelGPUSolver requires CUDA tensors")
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
                    + f"\n[Init ThreeLevel] N={self.N}, epsilon={self.epsilon}, "
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
                print(f"[Init ThreeLevel] clustering done in {time.time() - t0:.2f}s")

        self.V = clustering["DR_int"].clone()
        self.V.neg_()

        self.nearest_s2 = clustering["nearest_s2"]
        self.d_min_b_A2_int = clustering["d_min_b_A2_int"]
        self.nearest_s1 = clustering["nearest_s1"]
        self.d_min_b_A1_int = clustering["d_min_b_A1_int"]
        self.adj_A1_ptr = clustering["adj_A1_ptr"]
        self.S1 = int(clustering["sampled_idx_A1"].shape[0])
        self.S2 = int(clustering["sampled_idx_A2"].shape[0])
        self.adj_A1_col = clustering["adj_A1_col"]
        self.adj_A1_dist_int = clustering["adj_A1_dist_int"]
        self.adj_B_ptr = clustering["adj_B_ptr"]
        self.adj_B_col = clustering["adj_B_col"]
        self.adj_B_dist_int = clustering["adj_B_dist_int"]
        self._precompute_A1_structure()

        # Set-1 logic is still the A2-matrix fallback; retain these aliases only
        # for inherited debugging helpers that are specific to the matrix set.
        self.nearest_s = self.nearest_s2
        self.d_min_b_int = self.d_min_b_A2_int

        del clustering
        if precomputed_clustering is None:
            del cluster_engine
        gc.collect()

        self.y_A = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.y_B = torch.zeros(self.N, device=self.device, dtype=torch.int32)
        self.match_A = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
        self.match_B = torch.full((self.N,), -1, device=self.device, dtype=torch.long)
        self.cleanup_blues = torch.empty(0, device=self.device, dtype=torch.long)
        self.phase_match_set = torch.zeros(
            self.N, device=self.device, dtype=torch.int32
        )
        self.phase_match_is_set1 = torch.zeros(
            self.N, device=self.device, dtype=torch.bool
        )

        self.yA = self.y_A
        self.yB = self.y_B
        self.MA = self.match_A
        self.MB = self.match_B
        self.iterations = 0
        # Keep full audits on initially for three-level bring-up. Turn this off
        # for large production runs once correctness is confirmed.
        self.debug_audit = True
        self.debug_stop_on_first_violation = True
        self._debug_bad_checkpoint_seen = False
        self._debug_last_B_free = None
        self._debug_last_r_new = None
        self._debug_last_b_new = None
        self._debug_last_F_B_new = None
        self._debug_last_evicted_b = None
        self._debug_last_iteration = None

    def solve(self):
        N = self.N
        device = self.device
        B_free = torch.arange(N, device=device, dtype=torch.long)
        iteration = 0

        def _print_progress(iteration, free_before, free_after, status):
            if self.verbose and iteration % 100 == 0:
                print(
                    f"[ThreeLevel iter {iteration}] free_before={free_before} "
                    f"free_after={free_after} status={status}",
                    flush=True,
                )

        while True:
            num_free = B_free.numel()
            if num_free <= self.epsilon_int:
                break
            iteration += 1
            if self.max_iters > 0 and iteration > self.max_iters:
                raise RuntimeError(
                    f"ThreeLevelGPUSolver exceeded max_iters={self.max_iters}"
                )

            if self.debug_audit:
                self.audit_full_feasibility(
                    "phase_start",
                    iteration,
                    require_matched_equality=True,
                )
                self._snapshot_phase_context(B_free=B_free, reset_phase=True)
                self._debug_last_iteration = int(iteration)

            pair_inverse, set1_counts, set1_offsets, set1_values = (
                self._set1_groups(B_free)
            )
            set1_count_per_blue = set1_counts[pair_inverse]
            set2_count_per_blue = self._set2_counts(B_free)
            set3_count_per_blue = self._set3_counts(B_free)

            total_count = (
                set1_count_per_blue + set2_count_per_blue + set3_count_per_blue
            ).float()
            has_any = total_count > 0
            p1 = torch.where(
                has_any,
                set1_count_per_blue.float() / total_count,
                torch.zeros_like(total_count),
            )
            p2 = torch.where(
                has_any,
                set2_count_per_blue.float() / total_count,
                torch.zeros_like(total_count),
            )
            rand_pick = torch.rand(num_free, device=device)
            choose_set1 = has_any & (set1_count_per_blue > 0) & (rand_pick < p1)
            choose_set2 = (
                has_any
                & (set2_count_per_blue > 0)
                & (~choose_set1)
                & (rand_pick < (p1 + p2))
            )
            choose_set3 = (
                has_any
                & (set3_count_per_blue > 0)
                & (~choose_set1)
                & (~choose_set2)
            )

            proposal_a_parts = []
            proposal_b_parts = []
            proposal_set_parts = []

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
                proposal_set_parts.append(
                    torch.full((a1.numel(),), 1, device=device, dtype=torch.int32)
                )

            b2, a2 = self._set2_sample(B_free, choose_set2)
            if a2.numel() != 0:
                proposal_a_parts.append(a2)
                proposal_b_parts.append(b2)
                proposal_set_parts.append(
                    torch.full((a2.numel(),), 2, device=device, dtype=torch.int32)
                )

            b3, a3 = self._set3_sample(B_free, choose_set3)
            if a3.numel() != 0:
                proposal_a_parts.append(a3)
                proposal_b_parts.append(b3)
                proposal_set_parts.append(
                    torch.full((a3.numel(),), 3, device=device, dtype=torch.int32)
                )

            if not proposal_a_parts:
                if self.debug_audit:
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
            proposal_set = torch.cat(proposal_set_parts)

            if self.debug_audit:
                self.audit_full_feasibility(
                    "after_proposal_generation_before_conflict_resolution",
                    iteration,
                    require_matched_equality=True,
                )

            r_new, b_new = self._resolve_conflicts(proposal_a, proposal_b)
            if self.debug_audit:
                self._snapshot_phase_context(r_new=r_new, b_new=b_new)
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
            accepted_set = proposal_set[accepted_prop_idx]

            self.phase_match_set[b_new] = accepted_set
            self.phase_match_is_set1[b_new] = accepted_set == 1

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

        self.iterations = iteration
        verify_results = None
        if self.debug_audit:
            verify_results = self.verify_solution()
        self._last_verify = verify_results
        self.cleanup_remaining_points()
        if self.verbose:
            print(f"[ThreeLevel] Matched: {(self.match_B != -1).sum().item()}/{self.N}")
            self.calculate_final_stats()
        return self.match_B

    def _set1_eligible_mask(self, B_free):
        num_free = B_free.numel()
        if num_free == 0:
            return torch.empty(0, device=self.device, dtype=torch.bool)

        free_s = self.nearest_s2[B_free]
        free_score = (
            self.y_B[B_free].to(torch.long)
            - self.d_min_b_A2_int[B_free].to(torch.long)
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
        free_s = self.nearest_s2[B_eligible]
        free_t = self.d_min_b_A2_int[B_eligible] + 1 - self.y_B[B_eligible]

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

        free_s = self.nearest_s2[B_eligible]
        free_t = self.d_min_b_A2_int[B_eligible] + 1 - self.y_B[B_eligible]

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

    def _expand_csr_rows(self, row_ids, ptr, edge_attr_name, pos_attr_name):
        num_rows = row_ids.numel()
        empty = torch.empty(0, device=self.device, dtype=torch.long)
        if num_rows == 0:
            return empty, empty

        starts = ptr[row_ids]
        ends = ptr[row_ids + 1]
        lengths = ends - starts
        total_edges = int(lengths.sum().item())
        if total_edges == 0:
            return empty, empty

        edge_range = _ensure_long_arange(
            self, edge_attr_name, total_edges, self.device
        )
        row_pos = _ensure_long_arange(self, pos_attr_name, num_rows, self.device)
        cum_len = torch.cumsum(lengths, dim=0)
        packed_starts = cum_len - lengths

        active_row_pos = torch.repeat_interleave(row_pos, lengths)
        active_edge_idx = (
            torch.repeat_interleave(starts, lengths)
            + edge_range
            - torch.repeat_interleave(packed_starts, lengths)
        )
        return active_row_pos, active_edge_idx

    def _precompute_A1_structure(self):
        """
        Precompute the owner A1 center of every flat Adj_A1 CSR entry.
        """
        S1 = int(self.adj_A1_ptr.shape[0]) - 1
        lengths = self.adj_A1_ptr[1:] - self.adj_A1_ptr[:-1]
        self.center_of_A1_entry = torch.repeat_interleave(
            torch.arange(S1, device=self.device, dtype=torch.long),
            lengths,
        )

    def _set2_build_histogram_sort_fallback(self, B_query, rhs, S1, MA1):
        """
        Sort-based fallback when the bounded-range histogram would be too large.
        """
        NQ = B_query.numel()

        lhs_all = (
            self.y_A[self.adj_A1_col].to(torch.long)
            - self.adj_A1_dist_int.to(torch.long)
        )
        centers_all = self.center_of_A1_entry

        global_min = int(min(lhs_all.min().item(), rhs.min().item()))
        global_max = int(max(lhs_all.max().item(), rhs.max().item()))
        V_stride = global_max - global_min + 1
        key_entries_all = centers_all * V_stride + (lhs_all - global_min)
        key_blues = self.nearest_s1[B_query] * V_stride + (rhs - global_min)

        hist_size_full = S1 * V_stride
        if hist_size_full > _HIST_SIZE_LIMIT:
            sorted_entry_keys, sorted_order = torch.sort(key_entries_all)
            left = torch.searchsorted(sorted_entry_keys, key_blues, right=False)
            right = torch.searchsorted(sorted_entry_keys, key_blues, right=True)
            counts_nq = (right - left).clamp_(min=0)
            return {
                "hist": None,
                "filtered_idx": sorted_order,
                "key_entries": sorted_entry_keys,
                "query_key": key_blues,
                "rhs_min": global_min,
                "V_range": V_stride,
                "_sort_fallback": True,
                "_counts_nq": counts_nq,
                "_sorted_red": self.adj_A1_col[sorted_order],
                "_left": left,
            }

        hist = torch.zeros(hist_size_full, dtype=torch.long, device=self.device)
        hist.scatter_add_(
            0,
            key_entries_all,
            torch.ones(MA1, dtype=torch.long, device=self.device),
        )
        return {
            "hist": hist,
            "filtered_idx": torch.arange(MA1, device=self.device),
            "key_entries": key_entries_all,
            "query_key": key_blues,
            "rhs_min": global_min,
            "V_range": V_stride,
        }

    def _set2_build_histogram(self, B_query):
        """
        Build the (center, lhs-value) histogram restricted to B_query's rhs range.
        """
        MA1 = int(self.adj_A1_col.numel())
        NQ = int(B_query.numel())
        if NQ == 0 or MA1 == 0:
            return None

        rhs = (
            self.d_min_b_A1_int[B_query].to(torch.long)
            + 1
            - self.y_B[B_query].to(torch.long)
        )

        rhs_min = int(rhs.min().item())
        rhs_max = int(rhs.max().item())
        V_range = rhs_max - rhs_min + 1
        S1 = int(self.adj_A1_ptr.shape[0]) - 1

        hist_size = S1 * V_range
        if hist_size > _HIST_SIZE_LIMIT:
            return self._set2_build_histogram_sort_fallback(B_query, rhs, S1, MA1)

        lhs_all = (
            self.y_A[self.adj_A1_col].to(torch.long)
            - self.adj_A1_dist_int.to(torch.long)
        )
        in_range = (lhs_all >= rhs_min) & (lhs_all <= rhs_max)
        filtered_idx = in_range.nonzero(as_tuple=True)[0]
        if filtered_idx.numel() == 0:
            return None

        lhs_filtered = lhs_all[filtered_idx]
        centers_filtered = self.center_of_A1_entry[filtered_idx]
        key_entries = centers_filtered * V_range + (lhs_filtered - rhs_min)

        hist = torch.zeros(hist_size, dtype=torch.long, device=self.device)
        hist.scatter_add_(
            0,
            key_entries,
            torch.ones(filtered_idx.numel(), dtype=torch.long, device=self.device),
        )

        query_key = self.nearest_s1[B_query] * V_range + (rhs - rhs_min)
        return {
            "hist": hist,
            "filtered_idx": filtered_idx,
            "key_entries": key_entries,
            "query_key": query_key,
            "rhs_min": rhs_min,
            "V_range": V_range,
        }

    def _expand_set2_A1_entries(self, blues, edge_attr_name, pos_attr_name):
        num_blues = blues.numel()
        empty = torch.empty(0, device=self.device, dtype=torch.long)
        if num_blues == 0:
            return empty, empty, empty

        blues_s1 = self.nearest_s1[blues]
        sort_order = torch.argsort(blues_s1)
        sorted_blues = blues[sort_order]
        sorted_s1 = blues_s1[sort_order]

        active_sorted_pos, active_edge_idx = self._expand_csr_rows(
            sorted_s1,
            self.adj_A1_ptr,
            edge_attr_name,
            pos_attr_name,
        )
        if active_edge_idx.numel() == 0:
            return empty, empty, empty

        active_orig_pos = sort_order[active_sorted_pos]
        active_b = sorted_blues[active_sorted_pos]
        return active_orig_pos, active_b, active_edge_idx

    def _set2_counts(self, B_free):
        num_free = B_free.numel()
        zero = torch.zeros(num_free, dtype=torch.long, device=self.device)
        if num_free == 0:
            return zero

        info = self._set2_build_histogram(B_free)
        if info is None:
            return zero
        if info.get("_sort_fallback"):
            return info["_counts_nq"]
        return info["hist"][info["query_key"]]

    def _set2_sample_sort(self, selected_b, info):
        empty = torch.empty(0, device=self.device, dtype=torch.long)
        counts_nq = info["_counts_nq"]
        sorted_red = info["_sorted_red"]
        left = info["_left"]

        has_cand = counts_nq > 0
        if not has_cand.any():
            return empty, empty

        sel_valid = selected_b[has_cand]
        count_valid = counts_nq[has_cand]
        left_valid = left[has_cand]

        rand_rank = (
            torch.rand(sel_valid.numel(), device=self.device) * count_valid.float()
        ).long().clamp_(max=count_valid - 1)
        sampled_a = sorted_red[left_valid + rand_rank]
        return sel_valid, sampled_a

    def _set2_sample(self, B_free, choose_set2):
        selected_b = B_free[choose_set2]
        empty = torch.empty(0, device=self.device, dtype=torch.long)
        if selected_b.numel() == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty

        info = self._set2_build_histogram(selected_b)
        if info is None:
            return empty, empty
        if info.get("_sort_fallback"):
            return self._set2_sample_sort(selected_b, info)

        hist = info["hist"]
        filtered_idx = info["filtered_idx"]
        key_entries = info["key_entries"]
        query_key = info["query_key"]

        count_per_blue = hist[query_key]
        has_cand = count_per_blue > 0
        if not has_cand.any():
            return empty, empty

        sel_valid = selected_b[has_cand]
        qkey_valid = query_key[has_cand]
        count_valid = count_per_blue[has_cand]

        sort_order = torch.argsort(key_entries, stable=True)
        sorted_keys = key_entries[sort_order]
        sorted_red = self.adj_A1_col[filtered_idx[sort_order]]

        group_start = torch.searchsorted(sorted_keys, qkey_valid)
        rand_rank = (
            torch.rand(sel_valid.numel(), device=self.device) * count_valid.float()
        ).long().clamp_(max=count_valid - 1)
        sampled_a = sorted_red[group_start + rand_rank]
        return sel_valid, sampled_a

    def _set3_counts(self, B_free):
        num_free = B_free.numel()
        set3_count_per_blue = torch.zeros(
            num_free, device=self.device, dtype=torch.long
        )

        active_free_pos, active_edge_idx = self._expand_csr_rows(
            B_free,
            self.adj_B_ptr,
            "_set3_B_edge_arange",
            "_set3_B_free_pos",
        )
        if active_edge_idx.numel() == 0:
            return set3_count_per_blue

        active_b = B_free[active_free_pos]
        active_a = self.adj_B_col[active_edge_idx]
        target_y_a = self.adj_B_dist_int[active_edge_idx] + 1 - self.y_B[active_b]
        is_candidate = self.y_A[active_a] == target_y_a
        cand_free_pos = active_free_pos[is_candidate]

        if cand_free_pos.numel() != 0:
            set3_count_per_blue.scatter_add_(
                0, cand_free_pos, torch.ones_like(cand_free_pos)
            )
        return set3_count_per_blue

    def _set3_sample(self, B_free, choose_set3):
        selected_b = B_free[choose_set3]
        num_selected = selected_b.numel()
        if num_selected == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty

        active_selected_pos, active_edge_idx = self._expand_csr_rows(
            selected_b,
            self.adj_B_ptr,
            "_set3_B_sample_edge_arange",
            "_set3_B_sample_pos",
        )
        if active_edge_idx.numel() == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty

        active_b = selected_b[active_selected_pos]
        active_a = self.adj_B_col[active_edge_idx]
        target_y_a = self.adj_B_dist_int[active_edge_idx] + 1 - self.y_B[active_b]
        is_candidate = self.y_A[active_a] == target_y_a
        cand_selected_pos = active_selected_pos[is_candidate]
        cand_a = active_a[is_candidate]
        cand_count = cand_a.numel()
        if cand_count == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty

        cand_idx = _ensure_long_arange(
            self, "_set3_B_sample_cand_arange", cand_count, self.device
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

    def _compute_delta(self, B_free, unique_pairs, pair_inverse):
        num_free = B_free.numel()
        if num_free == 0:
            return 0

        sentinel = torch.iinfo(torch.int64).max // 4
        device = self.device
        target2 = (
            self.d_min_b_A2_int[B_free] + 1 - self.y_B[B_free]
        ).to(torch.long)
        min_slack1_per_blue = torch.full(
            (num_free,), sentinel, device=device, dtype=torch.long
        )
        if unique_pairs.numel() != 0:
            set1_eligible = self._set1_eligible_mask(B_free)
            if set1_eligible.any().item():
                pair_s = unique_pairs[:, 0].to(torch.long)
                num_pairs = pair_s.numel()
                v_pair_row_max = torch.empty(
                    num_pairs, device=device, dtype=torch.long
                )
                for start in range(0, num_pairs, self.set1_pair_batch):
                    end = min(start + self.set1_pair_batch, num_pairs)
                    v_pair_row_max[start:end] = (
                        self.V[pair_s[start:end]].max(dim=1).values.to(torch.long)
                    )
                min_slack1_per_blue[set1_eligible] = (
                    target2[set1_eligible]
                    - v_pair_row_max[pair_inverse[set1_eligible]]
                )

        min_slack2_per_blue = torch.full(
            (num_free,), sentinel, device=device, dtype=torch.long
        )
        MA1 = int(self.adj_A1_col.numel())
        if MA1 > 0:
            S1 = int(self.adj_A1_ptr.shape[0]) - 1
            entry_term = (
                self.adj_A1_dist_int.to(torch.long)
                - self.y_A[self.adj_A1_col].to(torch.long)
            )
            min_entry_term = torch.full(
                (S1,), sentinel, device=device, dtype=torch.long
            )
            min_entry_term.scatter_reduce_(
                0,
                self.center_of_A1_entry,
                entry_term,
                reduce="amin",
                include_self=True,
            )

            rhs_b = (
                self.d_min_b_A1_int[B_free] + 1 - self.y_B[B_free]
            ).to(torch.long)
            nearest_min = min_entry_term[self.nearest_s1[B_free]]
            has_edges = nearest_min != sentinel
            if has_edges.any():
                min_slack2_per_blue[has_edges] = (
                    rhs_b[has_edges] + nearest_min[has_edges]
                )

        min_slack3_per_blue = torch.full(
            (num_free,), sentinel, device=device, dtype=torch.long
        )
        starts = self.adj_B_ptr[B_free]
        ends = self.adj_B_ptr[B_free + 1]
        lengths = ends - starts
        total_B = int(lengths.sum().item())
        if total_B > 0:
            edge_range = _ensure_long_arange(
                self, "_delta_B_edge_arange", total_B, device
            )
            free_pos = _ensure_long_arange(
                self, "_delta_B_free_pos", num_free, device
            )
            cum_len = torch.cumsum(lengths, dim=0)
            packed_starts = cum_len - lengths

            active_free_pos = torch.repeat_interleave(free_pos[:num_free], lengths)
            active_edge_idx = (
                torch.repeat_interleave(starts, lengths)
                + edge_range[:total_B]
                - torch.repeat_interleave(packed_starts, lengths)
            )
            active_a = self.adj_B_col[active_edge_idx]
            adj_B_term = (
                self.adj_B_dist_int[active_edge_idx].to(torch.long)
                - self.y_A[active_a].to(torch.long)
            )
            min_adj_B = torch.full(
                (num_free,), sentinel, device=device, dtype=torch.long
            )
            min_adj_B.scatter_reduce_(
                0,
                active_free_pos,
                adj_B_term,
                reduce="amin",
                include_self=True,
            )
            has_B = min_adj_B != sentinel
            if has_B.any():
                min_slack3_per_blue[has_B] = (
                    1 - self.y_B[B_free[has_B]].to(torch.long) + min_adj_B[has_B]
                )

        min_slack_per_blue = torch.minimum(
            min_slack1_per_blue,
            torch.minimum(min_slack2_per_blue, min_slack3_per_blue),
        )
        valid_mask = (min_slack_per_blue != sentinel) & (min_slack_per_blue > 0)
        if valid_mask.any().item():
            return int(min_slack_per_blue[valid_mask].min().item())
        return 0

    def _update_matching(self, B_free, r_new, b_new):
        was_matched = self.match_A[r_new] != -1
        evicted_b = self.match_A[r_new[was_matched]].clone()
        self._debug_last_evicted_b = evicted_b.detach().clone()
        if evicted_b.numel() != 0:
            self.match_B[evicted_b] = -1
            self.phase_match_set[evicted_b] = 0
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

    def _build_dense_proxy_structures(self):
        device = self.device
        N = self.N

        in_adj_B_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        direct_B_costs = torch.zeros(N, N, dtype=torch.long, device=device)
        if self.adj_B_col.numel() > 0:
            all_b_for_adj_B = torch.repeat_interleave(
                torch.arange(N, device=device, dtype=torch.long),
                self.adj_B_ptr[1:] - self.adj_B_ptr[:-1],
            )
            in_adj_B_mask[all_b_for_adj_B, self.adj_B_col] = True
            direct_B_costs[all_b_for_adj_B, self.adj_B_col] = self.adj_B_dist_int.to(
                torch.long
            )

        in_adj_A1_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        mid_A1_costs = torch.zeros(N, N, dtype=torch.long, device=device)
        all_b = torch.arange(N, device=device, dtype=torch.long)
        _, active_b, active_edge_idx = self._expand_set2_A1_entries(
            all_b,
            "_verify_set2_A1_edge_arange",
            "_verify_set2_A1_sorted_pos",
        )
        if active_edge_idx.numel() > 0:
            active_a = self.adj_A1_col[active_edge_idx]
            in_adj_A1_mask[active_b, active_a] = True
            mid_A1_costs[active_b, active_a] = (
                self.d_min_b_A1_int[active_b].to(torch.long)
                + self.adj_A1_dist_int[active_edge_idx].to(torch.long)
            )

        y_A_long = self.y_A.to(torch.long)
        y_B_long = self.y_B.to(torch.long)
        V_rows = self.V[self.nearest_s2].to(torch.long)
        fallback_A2_proxy = (
            self.d_min_b_A2_int.to(torch.long).unsqueeze(1)
            + y_A_long.unsqueeze(0)
            - V_rows
        )
        proxy = torch.where(
            in_adj_B_mask,
            direct_B_costs,
            torch.where(in_adj_A1_mask, mid_A1_costs, fallback_A2_proxy),
        )
        return (
            in_adj_B_mask,
            direct_B_costs,
            in_adj_A1_mask,
            mid_A1_costs,
            fallback_A2_proxy,
            proxy,
            y_A_long,
            y_B_long,
        )

    def audit_full_feasibility(self, label, iteration, require_matched_equality=False):
        (
            in_adj_B_mask,
            direct_B_costs,
            in_adj_A1_mask,
            mid_A1_costs,
            fallback_A2_proxy,
            proxy,
            y_A_long,
            y_B_long,
        ) = self._build_dense_proxy_structures()

        feas_slack = proxy + 1 - y_B_long.unsqueeze(1) - y_A_long.unsqueeze(0)
        min_feas_slack = int(feas_slack.min().item())
        feasibility_violating = feas_slack < 0
        num_feas_violations = int(feasibility_violating.sum().item())
        worst_feas_flat = int(torch.argmin(feas_slack).item())
        worst_feas_pair = (worst_feas_flat // self.N, worst_feas_flat % self.N)

        matched_b = torch.nonzero(self.match_B != -1, as_tuple=True)[0]
        if matched_b.numel() > 0:
            matched_a = self.match_B[matched_b]
            matched_proxy = proxy[matched_b, matched_a]
            matched_diff = matched_proxy - y_B_long[matched_b] - y_A_long[matched_a]
            num_match_violations = int((matched_diff != 0).sum().item())
            matched_abs_diff = matched_diff.abs()
            worst_match_idx = int(torch.argmax(matched_abs_diff).item())
            worst_abs_matched_diff = int(matched_abs_diff[worst_match_idx].item())
            worst_matched_pair = (
                int(matched_b[worst_match_idx].item()),
                int(matched_a[worst_match_idx].item()),
            )
        else:
            num_match_violations = 0
            worst_abs_matched_diff = 0
            worst_matched_pair = None

        if num_feas_violations > 0:
            b, a = worst_feas_pair
            print("\n" + "=" * 80)
            print("[Audit] THREE-LEVEL FEASIBILITY VIOLATION")
            print(
                f"  iteration={iteration} checkpoint={label} "
                f"violating_edges={num_feas_violations} min_slack={min_feas_slack}"
            )
            print(
                f"  worst_pair=(b={b}, a={a}) y_B={int(y_B_long[b].item())} "
                f"y_A={int(y_A_long[a].item())} proxy={int(proxy[b, a].item())}"
            )
            print(
                f"  direct={bool(in_adj_B_mask[b, a].item())} "
                f"via_A1={bool(in_adj_A1_mask[b, a].item())} "
                f"direct_cost={int(direct_B_costs[b, a].item())} "
                f"via_A1_cost={int(mid_A1_costs[b, a].item())} "
                f"via_A2_cost={int(fallback_A2_proxy[b, a].item())}"
            )
            print("=" * 80)
            if self.debug_stop_on_first_violation:
                raise RuntimeError(
                    f"Feasibility violation at iteration {iteration}, step {label}"
                )

        if (
            require_matched_equality
            and num_match_violations > 0
            and self.debug_stop_on_first_violation
        ):
            b, a = worst_matched_pair
            print("\n" + "=" * 80)
            print("[Audit] THREE-LEVEL MATCHED ADMISSIBILITY VIOLATION")
            print(
                f"  iteration={iteration} checkpoint={label} "
                f"b={b} a={a} matched_diff="
                f"{int(proxy[b, a].item() - y_B_long[b].item() - y_A_long[a].item())}"
            )
            print(
                f"  phase_match_set={int(self.phase_match_set[b].item())} "
                f"proxy={int(proxy[b, a].item())} "
                f"y_B={int(y_B_long[b].item())} y_A={int(y_A_long[a].item())}"
            )
            print("=" * 80)
            raise RuntimeError(
                f"Matched equality violation at iteration {iteration}, step {label}"
            )

        return {
            "total_pairs": self.N * self.N,
            "min_feas_slack": min_feas_slack,
            "num_feas_violations": num_feas_violations,
            "worst_feas_pair": worst_feas_pair,
            "num_match_violations": num_match_violations,
            "worst_matched_pair": worst_matched_pair,
            "worst_abs_matched_diff": worst_abs_matched_diff,
        }

    def verify_solution(self):
        """
        Full N² feasibility and admissibility check. Run before cleanup.

        For EVERY pair (b, a) — matched or not:
            y_B[b] + y_A[a] <= proxy_cost(b, a) + 1      [feasibility]

        For EVERY phase-matched pair (b, a):
            y_B[b] + y_A[a] == proxy_cost(b, a)           [admissibility]

        Proxy cost:
            a in Adj_B(b)                      -> adj_B_dist_int
            a in Adj_A1(nearest_s1[b])         -> d_min_b_A1_int[b] + adj_A1_dist_int
            otherwise                          -> d_min_b_A2_int[b] + DR_int
                                                  where DR_int = y_A - V
        """
        device = self.device
        N = self.N

        (
            in_adj_B_mask,
            direct_B_costs,
            in_adj_A1_mask,
            mid_A1_costs,
            fallback_A2_proxy,
            proxy,
            y_A_long,
            y_B_long,
        ) = self._build_dense_proxy_structures()

        is_cleanup = torch.zeros(N, dtype=torch.bool, device=device)
        if self.cleanup_blues.numel() > 0:
            is_cleanup[self.cleanup_blues] = True
        matched_b_all = torch.nonzero(self.match_B != -1, as_tuple=True)[0]
        phase_matched_b = matched_b_all[~is_cleanup[matched_b_all]]
        is_phase_matched = torch.zeros(N, dtype=torch.bool, device=device)
        is_phase_matched[phase_matched_b] = True

        lhs_all = y_B_long.unsqueeze(1) + y_A_long.unsqueeze(0)
        excess = lhs_all - proxy - 1
        total_feasibility_violations = int((excess > 0).sum().item())
        worst_feasibility_excess = int(excess.max().item())

        viol_matched_b = torch.empty(0, device=device, dtype=torch.long)
        viol_matched_a = torch.empty(0, device=device, dtype=torch.long)
        viol_match_set = torch.empty(0, device=device, dtype=torch.int32)
        total_admissibility_violations = 0
        worst_admissibility_diff = 0

        if phase_matched_b.numel() > 0:
            matched_a = self.match_B[phase_matched_b]
            lhs_matched = lhs_all[phase_matched_b, matched_a]
            proxy_matched = proxy[phase_matched_b, matched_a]
            diff = lhs_matched - proxy_matched
            total_admissibility_violations = int((diff != 0).sum().item())
            if diff.numel() > 0:
                worst_admissibility_diff = int(diff.abs().max().item())
            viol_mask = diff != 0
            if viol_mask.any():
                viol_matched_b = phase_matched_b[viol_mask]
                viol_matched_a = matched_a[viol_mask]
                viol_match_set = self.phase_match_set[viol_matched_b]

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

        if viol_matched_b.numel() > 0:
            direct_zero_slack = lhs_all == direct_B_costs
            mid_only_mask = in_adj_A1_mask & (~in_adj_B_mask)
            mid_zero_slack = (lhs_all == mid_A1_costs) & mid_only_mask

            set1_mask = viol_match_set == 1
            set2_mask = viol_match_set == 2
            set3_mask = viol_match_set == 3

            set1_viol_b = viol_matched_b[set1_mask]
            set2_viol_b = viol_matched_b[set2_mask]

            set1_with_lower_level_zero_slack = 0
            if set1_viol_b.numel() > 0:
                lower_level_zero_slack = direct_zero_slack[set1_viol_b] | mid_zero_slack[
                    set1_viol_b
                ]
                set1_with_lower_level_zero_slack = int(
                    lower_level_zero_slack.any(dim=1).sum().item()
                )

            set2_with_direct_zero_slack = 0
            if set2_viol_b.numel() > 0:
                set2_with_direct_zero_slack = int(
                    direct_zero_slack[set2_viol_b].any(dim=1).sum().item()
                )

            print(
                f"\n[Verify] Of {total_admissibility_violations} admissibility violations:"
            )
            print(
                f"  Matched via Set 1 (A2 matrix):             {int(set1_mask.sum().item())}"
            )
            print(
                f"  Matched via Set 2 (A1 adjacency):          {int(set2_mask.sum().item())}"
            )
            print(
                f"  Matched via Set 3 (direct adjacency):      {int(set3_mask.sum().item())}"
            )
            print(
                "  Set-1-matched with missed zero-slack lower-level edge: "
                f"{set1_with_lower_level_zero_slack}"
            )
            print(
                "  Set-2-matched with missed zero-slack direct edge:      "
                f"{set2_with_direct_zero_slack}"
            )
        else:
            print("\n[Verify] No admissibility violations — no Diagnostic 3 needed.")

        return {
            "feasibility_violations": total_feasibility_violations,
            "feasibility_worst_excess": worst_feasibility_excess,
            "admissibility_violations": total_admissibility_violations,
            "admissibility_worst_diff": worst_admissibility_diff,
        }
