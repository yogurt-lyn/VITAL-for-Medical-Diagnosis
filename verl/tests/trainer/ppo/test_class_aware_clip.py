import json

import pytest
import torch

from verl.trainer.ppo.class_aware_clip import build_class_aware_tensors, class_aware_parameters, mid_rarity_peak_parameters
from verl.trainer.ppo.core_algos import compute_policy_loss_reward_decoupled
from verl.utils.reward_score.neocxr import _tail_balanced_fbeta_reward, _tail_recall_reward
from verl.workers.config import ActorConfig, PolicyLossConfig


FREQUENCIES = {
    "pneumonia": 1708,
    "normal": 668,
    "nrds": 502,
    "ttn": 358,
    "bpd": 170,
    "pneumothorax": 41,
    "pleural_effusion": 8,
    "atelectasis": 7,
}


def test_tail_recall_reward_only_credits_recalled_reference_labels():
    assert _tail_recall_reward({"atelectasis"}, {"atelectasis"}) == pytest.approx(1.0)
    assert _tail_recall_reward(set(), {"atelectasis"}) == pytest.approx(0.0)
    assert _tail_recall_reward({"pneumonia"}, {"pneumonia"}) == pytest.approx(0.0)


def test_tail_balanced_reward_preserves_head_and_penalizes_false_positives():
    assert _tail_balanced_fbeta_reward({"pneumonia"}, {"pneumonia"}) == pytest.approx(0.0)
    clean_hit = _tail_balanced_fbeta_reward({"atelectasis"}, {"atelectasis"})
    noisy_hit = _tail_balanced_fbeta_reward({"atelectasis", "pneumonia"}, {"atelectasis"})
    miss = _tail_balanced_fbeta_reward(set(), {"atelectasis"})
    assert clean_hit == pytest.approx(1.0)
    assert 0.0 < noisy_hit < clean_hit
    assert miss == pytest.approx(0.0)


def _config(enabled: bool, positive_only: bool = False, separate_branch: bool = False, **policy_kwargs) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=8,
        use_dynamic_bsz=True,
        clip_ratio_low=3e-4,
        clip_ratio_high=4e-4,
        clip_ratio_c=10.0,
        policy_loss=PolicyLossConfig(
            loss_mode="reward_decoupled",
            class_aware_clip_enabled=enabled,
            class_aware_weight_positive_only=positive_only,
            class_aware_frequency_json=json.dumps(FREQUENCIES),
            class_aware_separate_branch_enabled=separate_branch,
            class_aware_loss_coef=0.1,
            **policy_kwargs,
        ),
    )


def test_head_and_tail_class_parameters():
    rarity = torch.tensor([0.0, 1.0])
    _, weight, positive_clip = class_aware_parameters(
        rarity, alpha=1.0, epsilon_base=3e-4, epsilon_max=4e-4, gamma=1.0, saturation_tau=2.0
    )
    assert weight[0].item() == pytest.approx(1.0)
    assert positive_clip[0].item() == pytest.approx(3e-4)
    assert weight[1] > weight[0]
    assert positive_clip[1] > positive_clip[0]
    # The lower clip is configured independently and remains 1 - epsilon_base.
    assert 1.0 - 3e-4 == pytest.approx(0.9997)


def test_mid_rarity_peak_clip_prefers_learnable_middle_classes():
    rarity = torch.tensor([0.0, 0.55, 1.0])
    z, weight, positive_clip = mid_rarity_peak_parameters(
        rarity,
        alpha=1.0,
        epsilon_base=4e-4,
        epsilon_max=8e-4,
        mu=0.55,
        sigma=0.25,
    )
    assert torch.all(weight == 1.0)
    assert z[1] > z[0]
    assert z[1] > z[2]
    assert positive_clip[1].item() == pytest.approx(8e-4)
    assert positive_clip[1] > positive_clip[0]
    assert positive_clip[1] > positive_clip[2]


def test_multi_label_defaults_to_max_rarity():
    mask = torch.ones(2, 3)
    infos = [
        {"disease_label": "Neonatal Pneumonia"},
        {"disease_label": "Neonatal Pneumonia, Atelectasis"},
    ]
    rarity, weight, positive_clip, tier = build_class_aware_tensors(infos, mask, _config(True).policy_loss)
    assert torch.all(rarity[0] == 0)
    assert torch.all(rarity[1] == 1)
    assert torch.all(weight[1] > weight[0])
    assert torch.all(positive_clip[1] > positive_clip[0])
    assert torch.all(tier[0] == 0)
    assert torch.all(tier[1] == 2)


def test_disabled_path_strictly_matches_original_and_enabled_backward():
    old_log_prob = torch.zeros(3, 4)
    response_mask = torch.ones(3, 4, dtype=torch.bool)
    sequence_advantages = torch.tensor([[0.2] * 4, [-0.1] * 4, [0.3] * 4])
    token_advantages = torch.tensor([[0.4] * 4, [-0.2] * 4, [0.1] * 4])
    advantages = sequence_advantages + token_advantages

    log_prob_original = torch.tensor(
        [[0.001, 0.0, -0.001, 0.002], [0.0, -0.002, 0.001, 0.0], [0.001, 0.001, 0.0, -0.001]],
        requires_grad=True,
    )
    original_loss, _ = compute_policy_loss_reward_decoupled(
        old_log_prob,
        log_prob_original,
        advantages,
        response_mask,
        config=_config(False),
        sequence_advantages=sequence_advantages,
        token_advantages=token_advantages,
    )
    supplied_but_disabled_loss, _ = compute_policy_loss_reward_decoupled(
        old_log_prob,
        log_prob_original,
        advantages,
        response_mask,
        config=_config(False),
        sequence_advantages=sequence_advantages,
        token_advantages=token_advantages,
        class_weight=torch.full_like(token_advantages, 2.0),
        class_positive_clip=torch.full_like(token_advantages, 9e-4),
        class_tier=torch.full_like(token_advantages, 2.0),
    )
    assert torch.equal(original_loss, supplied_but_disabled_loss)

    log_prob_enabled = log_prob_original.detach().clone().requires_grad_(True)
    class_weight = torch.tensor([[1.0] * 4, [1.5] * 4, [2.0] * 4])
    positive_clip = torch.tensor([[3e-4] * 4, [3.5e-4] * 4, [4e-4] * 4])
    tier = torch.tensor([[0.0] * 4, [1.0] * 4, [2.0] * 4])
    enabled_loss, metrics = compute_policy_loss_reward_decoupled(
        old_log_prob,
        log_prob_enabled,
        advantages,
        response_mask,
        config=_config(True),
        sequence_advantages=sequence_advantages,
        token_advantages=token_advantages,
        class_weight=class_weight,
        class_positive_clip=positive_clip,
        class_tier=tier,
    )
    enabled_loss.backward()
    assert log_prob_enabled.grad is not None
    assert torch.isfinite(log_prob_enabled.grad).all()
    assert "actor/class_aware_clipfrac_head" in metrics
    assert "actor/class_aware_clipfrac_middle" in metrics
    assert "actor/class_aware_clipfrac_tail" in metrics


def test_positive_only_weighting_keeps_negative_advantage_scale():
    old_log_prob = torch.zeros(1, 2)
    log_prob = torch.tensor([[0.001, -0.001]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    token_advantage = torch.tensor([[1.0, -1.0]])
    weight = torch.full_like(token_advantage, 2.0)
    loss, metrics = compute_policy_loss_reward_decoupled(
        old_log_prob,
        log_prob,
        token_advantage,
        mask,
        config=_config(True, positive_only=True),
        sequence_advantages=torch.zeros_like(token_advantage),
        token_advantages=token_advantage,
        class_weight=weight,
        class_positive_clip=torch.full_like(token_advantage, 4e-4),
        class_tier=torch.zeros_like(token_advantage),
    )
    loss.backward()
    assert torch.isfinite(log_prob.grad).all()
    assert metrics["actor/class_aware_weighted_adv_abs_mean"] == pytest.approx(1.5)


def test_separate_class_branch_backward_and_zero_mean_advantage():
    old_log_prob = torch.zeros(2, 3)
    log_prob = torch.tensor([[0.001, 0.0, -0.001], [-0.001, 0.0, 0.001]], requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.bool)
    class_advantage = torch.tensor([[1.25] * 3, [-1.25] * 3])
    zeros = torch.zeros_like(class_advantage)
    loss, metrics = compute_policy_loss_reward_decoupled(
        old_log_prob,
        log_prob,
        zeros,
        mask,
        config=_config(True, separate_branch=True),
        sequence_advantages=zeros,
        token_advantages=zeros,
        class_advantages=class_advantage,
        class_weight=torch.full_like(zeros, 1.25),
        class_positive_clip=torch.full_like(zeros, 6e-4),
        class_tier=torch.full_like(zeros, 2.0),
    )
    loss.backward()
    assert class_advantage.mean().item() == pytest.approx(0.0)
    assert torch.isfinite(log_prob.grad).all()
    assert metrics["actor/class_aware_branch_loss_coef"] == pytest.approx(0.1)


def test_separate_branch_weights_only_positive_post_normalized_advantage():
    old_log_prob = torch.zeros(1, 2)
    log_prob = torch.zeros(1, 2, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    class_advantage = torch.tensor([[1.0, -1.0]])
    zeros = torch.zeros_like(class_advantage)
    loss, metrics = compute_policy_loss_reward_decoupled(
        old_log_prob,
        log_prob,
        zeros,
        mask,
        config=_config(True, positive_only=True, separate_branch=True),
        sequence_advantages=zeros,
        token_advantages=zeros,
        class_advantages=class_advantage,
        class_weight=torch.full_like(zeros, 1.2),
        class_positive_clip=torch.full_like(zeros, 4.5e-4),
        class_tier=torch.full_like(zeros, 2.0),
    )
    loss.backward()
    assert torch.isfinite(log_prob.grad).all()
    assert metrics["actor/class_aware_weighted_adv_abs_mean"] == pytest.approx(1.1)


def test_cac_rarity_tp_clip_is_asymmetric_and_preserves_wrong_token_clip():
    config = _config(
        True,
        enable_class_aware_clip=True,
        class_aware_clip_mode="rarity_tp_token",
        base_clip_epsilon=4e-4,
        max_clip_epsilon=8e-4,
        rarity_clip_lambda=1.0,
    )

    old_log_prob = torch.zeros(1, 3)
    # 1.0005 is above the base upper bound, but below the tail TP upper bound.
    log_prob = torch.tensor([[5e-4, 5e-4, -5e-4]], requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)
    token_advantage = torch.tensor([[1.0, 1.0, -1.0]])
    positive_clip = torch.tensor([[8e-4, 4e-4, 8e-4]])
    class_id = torch.tensor([[7, 0, 7]])
    tp_mask = torch.tensor([[True, False, True]])
    fp_mask = torch.tensor([[False, True, False]])
    event_counts = torch.zeros(1, 8, 3)
    event_counts[0, 7] = torch.tensor([1.0, 1.0, 0.0])
    event_counts[0, 0] = torch.tensor([0.0, 0.0, 1.0])

    loss, metrics = compute_policy_loss_reward_decoupled(
        old_log_prob,
        log_prob,
        token_advantage,
        mask,
        config=config,
        sequence_advantages=torch.zeros_like(token_advantage),
        token_advantages=token_advantage,
        class_weight=torch.tensor([[2.0, 1.0, 2.0]]),
        class_positive_clip=positive_clip,
        class_tier=torch.tensor([[2.0, 0.0, 2.0]]),
        class_category_id=class_id,
        class_tp_mask=tp_mask,
        class_fp_mask=fp_mask,
        class_event_counts=event_counts,
    )
    loss.backward()
    assert torch.isfinite(log_prob.grad).all()
    assert metrics["actor/cac/atelectasis/epsilon_plus_mean"] == pytest.approx(6e-4)
    assert metrics["actor/cac/pneumonia/epsilon_plus_mean"] == pytest.approx(4e-4)
    assert metrics["actor/cac/atelectasis/tp_count"] == pytest.approx(1.0)
    assert metrics["actor/cac/atelectasis/fn_count"] == pytest.approx(1.0)
    assert metrics["actor/cac/pneumonia/fp_count"] == pytest.approx(1.0)
