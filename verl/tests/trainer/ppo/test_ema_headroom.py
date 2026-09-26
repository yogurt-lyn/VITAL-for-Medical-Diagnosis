"""Tests for EMA reward-headroom loss scales."""

from __future__ import annotations

import torch

from verl.trainer.ppo.ema_headroom import (
    apply_ema_headroom_loss_scales,
    apply_ema_headroom_weights,
    compute_ema_headroom_scales,
    reset_ema_state,
)


def test_low_reward_branch_gets_higher_alpha():
    reset_ema_state()
    # Initialize EMA near these means.
    out = compute_ema_headroom_scales(0.9, 0.2, beta=0.0)  # beta=0 → M=Rbar
    assert out["alpha_tok"] > out["alpha_seq"]
    # α sums to H/(H+ε) ≈ 1; 2α sums to ≈ 2 (ε keeps a tiny deficit)
    assert abs(out["alpha_seq"] + out["alpha_tok"] - 1.0) < 1e-5
    assert abs(out["scale_seq"] + out["scale_tok"] - 2.0) < 1e-5


def test_apply_scales_advantages():
    reset_ema_state()

    class _Batch:
        non_tensor_batch = {"uid": ["a", "a"]}
        batch = {
            "response_mask": torch.ones(2, 3),
            "sequence_advantages": torch.ones(2, 3),
            "token_advantages": torch.ones(2, 3),
        }

    reward = {"rouge_l": [0.9, 0.8], "cls_f1": [0.2, 0.1]}
    cfg = {"ema_headroom_enabled": True, "ema_headroom_beta": 0.0}
    batch = _Batch()
    apply_ema_headroom_weights(batch, reward, cfg)
    apply_ema_headroom_loss_scales(batch, cfg)
    assert batch.batch["token_advantages"][0, 0] > batch.batch["sequence_advantages"][0, 0]
    assert abs(
        float(batch.batch["sequence_advantages"][0, 0] + batch.batch["token_advantages"][0, 0])
        - 2.0
    ) < 1e-5
