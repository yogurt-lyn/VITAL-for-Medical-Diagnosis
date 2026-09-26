"""Compatibility shim: Focal prototype replaced by EMA-headroom loss weights."""

from verl.trainer.ppo.ema_headroom import (  # noqa: F401
    apply_ema_headroom_loss_scales as apply_focal_loss_branch_scales,
    apply_ema_headroom_weights as apply_focal_reward_reweight,
    ema_headroom_enabled as focal_reward_enabled,
)
