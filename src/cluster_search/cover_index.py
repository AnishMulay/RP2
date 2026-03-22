"""Cover index utilities for shell-based candidate lookup."""

from __future__ import annotations


class CoverIndex:
    """Indexes a clustering cover for shell and candidate lookup.

    The input COO tuple is expected to come directly from
    ``clustered_push_relabel.clustering.k_level.k_level_cluster`` and contains
    three parallel one-dimensional tensors or tensor-like objects:
    ``(center_ids, level_ids, point_ids)``.

    Args:
        coo: Tuple of parallel center, level, and point ID arrays.

    Raises:
        ValueError: If ``coo`` does not contain exactly three arrays or if the
            arrays are not the same length.
    """

    def __init__(self, coo: object) -> None:
        center_ids, level_ids, point_ids = self._normalize_coo(coo)

        self._point_to_shells: dict[int, list[tuple[int, int]]] = {}
        self._shell_to_members: dict[tuple[int, int], list[int]] = {}

        for center_id, level_id, point_id in zip(center_ids, level_ids, point_ids):
            shell = (center_id, level_id)
            self._point_to_shells.setdefault(point_id, []).append(shell)
            self._shell_to_members.setdefault(shell, []).append(point_id)

    @staticmethod
    def _normalize_coo(coo: object) -> tuple[list[int], list[int], list[int]]:
        try:
            raw_center_ids, raw_level_ids, raw_point_ids = coo  # type: ignore[misc]
        except (TypeError, ValueError) as exc:
            raise ValueError("coo must be a tuple of (center_ids, level_ids, point_ids).") from exc

        center_ids = CoverIndex._to_int_list(raw_center_ids)
        level_ids = CoverIndex._to_int_list(raw_level_ids)
        point_ids = CoverIndex._to_int_list(raw_point_ids)

        if not (len(center_ids) == len(level_ids) == len(point_ids)):
            raise ValueError("coo arrays must be parallel 1D sequences of equal length.")

        return center_ids, level_ids, point_ids

    @staticmethod
    def _to_int_list(values: object) -> list[int]:
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [int(value) for value in values]  # type: ignore[arg-type]

    def get_shells(self, point_id: int) -> list[tuple[int, int]]:
        """Returns the shells that contain a point.

        Args:
            point_id: Point identifier to look up.

        Returns:
            A list of ``(center_id, level_id)`` pairs for the point. Returns an
            empty list when the point is not indexed.
        """

        return list(self._point_to_shells.get(point_id, ()))

    def get_candidates(self, point_id: int) -> list[int]:
        """Returns the deduplicated union of members across a point's shells.

        Args:
            point_id: Point identifier whose shell members should be collected.

        Returns:
            A deduplicated list of member point IDs across all shells containing
            ``point_id``. The input point is not filtered out.
        """

        candidates: dict[int, None] = {}
        for shell in self._point_to_shells.get(point_id, ()):
            for member_id in self._shell_to_members.get(shell, ()):
                candidates.setdefault(member_id, None)
        return list(candidates)

    def num_shells(self) -> int:
        """Returns the number of unique shells in the index.

        Returns:
            The count of unique ``(center_id, level_id)`` shell keys.
        """

        return len(self._shell_to_members)

    def __repr__(self) -> str:
        shell_count = len(self._shell_to_members)
        avg_members_per_shell = (
            sum(len(members) for members in self._shell_to_members.values()) / shell_count
            if shell_count
            else 0.0
        )
        return (
            f"CoverIndex(num_points={len(self._point_to_shells)}, "
            f"num_shells={shell_count}, "
            f"avg_members_per_shell={avg_members_per_shell:.1f})"
        )
