"""Shim: load recovered module bytecode, then restore extended PolicyLossConfig.

The git-restored actor.pyc only has upstream PolicyLossConfig fields. actor.yaml and
training overrides also pass reward-decoupled / CAC / EMA / adaptclip / localization
keys, so Hydra InstantiationException unless PolicyLossConfig accepts them.
"""
from __future__ import annotations

import marshal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_backup = str(Path(__file__).resolve().parents[2] / "_bytecode_backup" / "actor.pyc")
with open(_backup, "rb") as _f:
    _data = _f.read()
_code = marshal.loads(_data[16:])
del _backup, _f, _data, marshal, Path
exec(_code, globals())

# Rebuild PolicyLossConfig with all fields used by actor.yaml + training scripts.
# Keep RolloutCorrectionConfig from the loaded bytecode / trainer.config.
from verl.base_config import BaseConfig  # noqa: E402
from verl.trainer.config import RolloutCorrectionConfig  # noqa: E402


@dataclass
class PolicyLossConfig(BaseConfig):
    """Policy loss config including GDPO reward-decoupled / CAC / EMA / adaptclip fields."""

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)

    # Reward-decoupled GDPO (sequence / token / format branches)
    reward_decoupled_sequence_keys: list[Any] = field(
        default_factory=lambda: ["img_part", "dis_part", "tpl_pen"]
    )
    reward_decoupled_token_keys: list[Any] = field(default_factory=lambda: ["gated_cls_fmt"])
    reward_decoupled_format_token_keys: list[Any] = field(default_factory=list)
    reward_decoupled_sequence_loss_coef: float = 1.0
    reward_decoupled_token_loss_coef: float = 1.0
    reward_decoupled_format_token_loss_coef: float = 1.0
    reward_decoupled_sequence_clip_ratio_low: Optional[float] = None
    reward_decoupled_sequence_clip_ratio_high: Optional[float] = None
    reward_decoupled_token_clip_ratio_low: Optional[float] = None
    reward_decoupled_token_clip_ratio_high: Optional[float] = None

    # Class-aware clip (CAC)
    class_aware_clip_enabled: bool = False
    class_aware_frequency_json: str = "{}"
    class_aware_alpha: float = 1.0
    class_aware_weight_positive_only: bool = False
    class_aware_epsilon_base: float = 0.0003
    class_aware_epsilon_max: float = 0.0004
    class_aware_gamma: float = 1.0
    class_aware_saturation_tau: float = 2.0
    class_aware_multi_label_aggregation: str = "max"
    class_aware_separate_branch_enabled: bool = False
    class_aware_reward_keys: list[Any] = field(default_factory=lambda: ["tail_recall_reward"])
    class_aware_clip_mode: str = ""
    class_aware_bound_mode: str = ""
    class_aware_mid_mu: float = 0.55
    class_aware_mid_sigma: float = 0.25
    class_aware_apply_weight: bool = True
    class_aware_loss_coef: float = 0.1
    enable_class_aware_clip: Optional[bool] = None
    class_frequency_file: str = ""
    base_clip_epsilon: float = 0.0004
    rarity_clip_lambda: float = 1.0
    max_clip_epsilon: float = 0.0008

    # Diagnosis-field localization
    localize_neocxr_diagnosis_token_advantage: bool = False
    localize_chexbert_token_advantage: bool = False
    diag_localization_debug_interval: int = 0
    diag_background_weight: float = 0.0

    # EMA branch loss normalization
    ema_branch_loss_norm_enabled: bool = False
    ema_branch_loss_norm_beta: float = 0.9
    ema_branch_loss_norm_eps: float = 1e-8

    # EMA reward headroom → dynamic 2α seq/tok loss scales
    ema_headroom_enabled: bool = False
    ema_headroom_beta: float = 0.9
    ema_headroom_eps: float = 1e-6
    # Legacy alias (Focal prototype); still accepted by ema_headroom_enabled().
    focal_reward_enabled: bool = False

    # Adaptive branch clip (+ head-only rarity gate)
    adaptive_branch_clip_enabled: bool = False
    adaptive_branch_clip_ema_beta: float = 0.9
    adaptive_branch_clip_deadzone: float = 0.1
    adaptive_branch_clip_alpha: float = 0.5
    adaptive_branch_clip_min_scale: float = 0.5
    adaptive_branch_clip_head_only: bool = False
    adaptive_branch_clip_rarity_threshold: float = 0.5


# Hydra / package re-exports resolve PolicyLossConfig from this module.
try:
    import verl.workers.config as _cfg_pkg

    _cfg_pkg.PolicyLossConfig = PolicyLossConfig
except Exception:
    pass
