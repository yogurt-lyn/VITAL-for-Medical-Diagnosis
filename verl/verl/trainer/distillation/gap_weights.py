"""Positive teacher-minus-student gap weights for OPD rollouts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def positive_gap_rollout_weights(
    teacher: Sequence[float],
    student: Sequence[float],
    group_ids: Sequence[object],
    normalize: bool = True,
) -> np.ndarray:
    """Weight each rollout by max(teacher - student, 0).

    When ``normalize`` is true, divide positive gaps by the mean positive gap
    among rollouts that share the same ``group_id`` (same prompt). Teacher-not-better
    rollouts stay at weight 0 and are excluded from the mean.
    """
    teacher_arr = np.asarray(teacher, dtype=np.float64)
    student_arr = np.asarray(student, dtype=np.float64)
    if teacher_arr.shape != student_arr.shape or teacher_arr.ndim != 1:
        raise ValueError(
            f"expected 1D matching teacher/student scores, got {teacher_arr.shape} and {student_arr.shape}"
        )
    if len(group_ids) != teacher_arr.shape[0]:
        raise ValueError(
            f"group_ids length {len(group_ids)} does not match scores {teacher_arr.shape[0]}"
        )
    positive_gap = np.maximum(teacher_arr - student_arr, 0.0)
    weights = positive_gap.copy()
    if not normalize:
        return weights.astype(np.float32)

    buckets: dict[object, list[int]] = {}
    for row, group_id in enumerate(group_ids):
        buckets.setdefault(group_id, []).append(row)
    for indices in buckets.values():
        idx = np.asarray(indices, dtype=np.int64)
        group_gap = positive_gap[idx]
        active = group_gap > 0
        if np.any(active):
            active_mean = max(float(group_gap[active].mean()), 1e-6)
            weights[idx] = np.where(active, group_gap / active_mean, 0.0)
        else:
            weights[idx] = 0.0
    return weights.astype(np.float32)


def select_positive_gap_indices(
    teacher: Sequence[float],
    student: Sequence[float],
    group_ids: Sequence[object],
    keep_k: int,
) -> np.ndarray:
    """DAPO-style keep: per prompt, keep up to ``keep_k`` positive-gap rollouts.

    Valid means ``teacher - student > 0``. Larger gaps are kept first. Groups with
    no positive gap are dropped. If a group has fewer than ``keep_k`` valid
    rollouts, all valid ones are kept (zeros are never used as padding).
    Returns original indices in increasing order.
    """
    if keep_k <= 0:
        raise ValueError(f"keep_k must be positive, got {keep_k}")
    teacher_arr = np.asarray(teacher, dtype=np.float64)
    student_arr = np.asarray(student, dtype=np.float64)
    if teacher_arr.shape != student_arr.shape or teacher_arr.ndim != 1:
        raise ValueError(
            f"expected 1D matching teacher/student scores, got {teacher_arr.shape} and {student_arr.shape}"
        )
    if len(group_ids) != teacher_arr.shape[0]:
        raise ValueError(
            f"group_ids length {len(group_ids)} does not match scores {teacher_arr.shape[0]}"
        )
    positive_gap = np.maximum(teacher_arr - student_arr, 0.0)
    buckets: dict[object, list[int]] = {}
    for row, group_id in enumerate(group_ids):
        buckets.setdefault(group_id, []).append(row)
    keep: list[int] = []
    for indices in buckets.values():
        valid = [idx for idx in indices if positive_gap[idx] > 0.0]
        if not valid:
            continue
        valid.sort(key=lambda idx: (-float(positive_gap[idx]), idx))
        keep.extend(valid[:keep_k])
    keep.sort()
    return np.asarray(keep, dtype=np.int64)


GT_SAT_SINGLE_METRICS = ("rouge_l", "chexpert_f1")
GT_SAT_MIXED_METRICS = ("r05c05", "eq", "equal")
GT_SAT_METRICS = GT_SAT_SINGLE_METRICS + GT_SAT_MIXED_METRICS


def mixed_gspo_student_scores(
    rouge_l: Sequence[float],
    ce_f1: Sequence[float],
    w_rougel: float = 0.5,
    w_chexpert: float = 0.5,
) -> np.ndarray:
    """Same mix as the 8B GSPO teacher: 0.5 ROUGE-L + 0.5 CheXpert-F1 vs GT."""
    rouge_arr = np.asarray(rouge_l, dtype=np.float64)
    ce_arr = np.asarray(ce_f1, dtype=np.float64)
    if rouge_arr.shape != ce_arr.shape or rouge_arr.ndim != 1:
        raise ValueError(
            f"expected 1D matching rouge_l/ce_f1, got {rouge_arr.shape} and {ce_arr.shape}"
        )
    total = float(w_rougel) + float(w_chexpert)
    if total <= 0:
        raise ValueError(f"mixed GSPO weights must sum > 0, got {w_rougel} + {w_chexpert}")
    mixed = (float(w_rougel) * rouge_arr + float(w_chexpert) * ce_arr) / total
    return np.clip(mixed, 0.0, 1.0)


def gt_saturation_rollout_weights(
    student: Sequence[float],
    group_ids: Sequence[object],
    normalize: bool = True,
    teacher: Sequence[float] | None = None,
    use_teacher_factor: bool = False,
) -> np.ndarray:
    """Weight each rollout by how unsaturated the student is vs GT.

    ``w_i = clip(1 - m_S, 0, 1)``. Teacher quality is **not** a hard gate: there is
    no ReLU on ``(m_T - m_S)`` and teacher-worse-than-student is not zeroed.

    When ``use_teacher_factor`` is true, multiply by ``clip(m_T, 0, 1)`` as a soft
    teacher-quality factor (default off). When ``normalize`` is true, divide by the
    mean weight over **all** rollouts that share the same ``group_id`` (including
    near-zero / already-perfect students).
    """
    student_arr = np.asarray(student, dtype=np.float64)
    if student_arr.ndim != 1:
        raise ValueError(f"expected 1D student scores, got {student_arr.shape}")
    if len(group_ids) != student_arr.shape[0]:
        raise ValueError(
            f"group_ids length {len(group_ids)} does not match scores {student_arr.shape[0]}"
        )
    weights = np.clip(1.0 - student_arr, 0.0, 1.0)
    if use_teacher_factor:
        if teacher is None:
            raise ValueError("use_teacher_factor=True requires teacher scores")
        teacher_arr = np.asarray(teacher, dtype=np.float64)
        if teacher_arr.shape != student_arr.shape:
            raise ValueError(
                f"expected matching teacher/student scores, got {teacher_arr.shape} and {student_arr.shape}"
            )
        weights = weights * np.clip(teacher_arr, 0.0, 1.0)
    if not normalize:
        return weights.astype(np.float32)

    buckets: dict[object, list[int]] = {}
    for row, group_id in enumerate(group_ids):
        buckets.setdefault(group_id, []).append(row)
    for indices in buckets.values():
        idx = np.asarray(indices, dtype=np.int64)
        group_w = weights[idx]
        mean_w = float(group_w.mean())
        if mean_w > 0.0:
            weights[idx] = group_w / max(mean_w, 1e-6)
        else:
            weights[idx] = 0.0
    return weights.astype(np.float32)
