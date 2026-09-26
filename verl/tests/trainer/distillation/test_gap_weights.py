from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_GAP_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3] / "verl" / "trainer" / "distillation" / "gap_weights.py"
)
_SPEC = importlib.util.spec_from_file_location("chexpert_gap_weights", _GAP_WEIGHTS_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
positive_gap_rollout_weights = _MOD.positive_gap_rollout_weights


select_positive_gap_indices = _MOD.select_positive_gap_indices


def test_chexpert_gap_mean_normalizes_within_prompt_and_zeros_student_better():
    teacher = [0.8, 0.8, 0.8, 0.8]
    student = [0.2, 0.5, 0.9, 0.8]
    weights = positive_gap_rollout_weights(teacher, student, group_ids=[7, 7, 7, 7], normalize=True)
    expected_gaps = np.array([0.6, 0.3, 0.0, 0.0], dtype=np.float64)
    active_mean = expected_gaps[:2].mean()
    np.testing.assert_allclose(weights, [0.6 / active_mean, 0.3 / active_mean, 0.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(float(weights[weights > 0].mean()), 1.0, rtol=1e-6)


def test_chexpert_gap_does_not_mix_prompts():
    teacher = [1.0, 1.0, 0.4, 0.4]
    student = [0.0, 0.5, 0.0, 0.2]
    weights = positive_gap_rollout_weights(teacher, student, group_ids=["a", "a", "b", "b"], normalize=True)
    np.testing.assert_allclose(weights, [1.0 / 0.75, 0.5 / 0.75, 0.4 / 0.3, 0.2 / 0.3], rtol=1e-6)


def test_select_positive_gap_keeps_top_k_and_drops_nonpositive():
    teacher = [0.8] * 8
    student = [0.1, 0.2, 0.9, 0.3, 0.4, 0.85, 0.5, 0.0]
    # gaps: 0.7, 0.6, 0, 0.5, 0.4, 0, 0.3, 0.8  -> top4: idx 7,0,1,3
    keep = select_positive_gap_indices(teacher, student, group_ids=[1] * 8, keep_k=4)
    np.testing.assert_array_equal(keep, [0, 1, 3, 7])


def test_select_positive_gap_keeps_all_when_fewer_than_k():
    teacher = [0.5, 0.5, 0.5, 0.5]
    student = [0.1, 0.6, 0.2, 0.7]
    keep = select_positive_gap_indices(teacher, student, group_ids=[9, 9, 9, 9], keep_k=4)
    np.testing.assert_array_equal(keep, [0, 2])


def test_select_positive_gap_does_not_mix_prompts():
    teacher = [1.0] * 4
    student = [0.0, 0.2, 0.0, 0.9]
    keep = select_positive_gap_indices(teacher, student, group_ids=["a", "a", "b", "b"], keep_k=1)
    np.testing.assert_array_equal(keep, [0, 2])


gt_saturation_rollout_weights = _MOD.gt_saturation_rollout_weights
mixed_gspo_student_scores = _MOD.mixed_gspo_student_scores


def test_mixed_gspo_is_equal_rouge_and_chexpert():
    mixed = mixed_gspo_student_scores([0.2, 1.0], [0.8, 0.0])
    np.testing.assert_allclose(mixed, [0.5, 0.5], rtol=1e-6)


def test_gt_sat_mixed_metric_weights_bad_rollouts_more():
    rouge = [0.8, 0.2, 0.5, 0.1]
    ce = [0.8, 0.2, 0.5, 0.1]
    student = mixed_gspo_student_scores(rouge, ce)
    weights = gt_saturation_rollout_weights(student, group_ids=[1, 1, 1, 1], normalize=True)
    # Worse rollouts (idx 1,3) must get larger teacher-OPD weight than good ones (idx 0).
    assert float(weights[3]) > float(weights[0])
    assert float(weights[1]) > float(weights[0])
    np.testing.assert_allclose(float(np.mean(weights)), 1.0, rtol=1e-6)


def test_gt_sat_is_one_minus_student_and_mean_normalizes_all_rollouts():
    student = [0.2, 0.5, 0.9, 1.0]
    weights = gt_saturation_rollout_weights(student, group_ids=[7, 7, 7, 7], normalize=True)
    raw = np.array([0.8, 0.5, 0.1, 0.0], dtype=np.float64)
    expected = raw / raw.mean()
    np.testing.assert_allclose(weights, expected, rtol=1e-6)
    np.testing.assert_allclose(float(np.mean(weights)), 1.0, rtol=1e-6)


def test_gt_sat_does_not_relu_teacher_gap():
    student = [0.9, 0.1]
    teacher = [0.2, 0.05]  # teacher worse than student on both
    weights = gt_saturation_rollout_weights(
        student, group_ids=[1, 1], normalize=False, teacher=teacher, use_teacher_factor=False
    )
    np.testing.assert_allclose(weights, [0.1, 0.9], rtol=1e-6)


def test_gt_sat_teacher_factor_is_optional_soft_multiplier():
    student = [0.0, 0.0]
    teacher = [0.5, 1.0]
    off = gt_saturation_rollout_weights(
        student, group_ids=[1, 1], normalize=False, teacher=teacher, use_teacher_factor=False
    )
    on = gt_saturation_rollout_weights(
        student, group_ids=[1, 1], normalize=False, teacher=teacher, use_teacher_factor=True
    )
    np.testing.assert_allclose(off, [1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(on, [0.5, 1.0], rtol=1e-6)


def test_gt_sat_does_not_mix_prompts():
    student = [0.0, 0.5, 0.0, 0.8]
    weights = gt_saturation_rollout_weights(student, group_ids=["a", "a", "b", "b"], normalize=True)
    raw_a = np.array([1.0, 0.5])
    raw_b = np.array([1.0, 0.2])
    expected = np.concatenate([raw_a / raw_a.mean(), raw_b / raw_b.mean()])
    np.testing.assert_allclose(weights, expected, rtol=1e-6)
