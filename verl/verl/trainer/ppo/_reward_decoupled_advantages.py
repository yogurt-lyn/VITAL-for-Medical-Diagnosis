"""Source rewrite of reward-decoupled advantage assembly with explicit localization.

The recovered ray_trainer bytecode builds ``diag_key_token_mask`` but forgets to pass
it into ``build_localized_diag_advantage``. This module provides the corrected
implementation used to override the bytecode function after shim load.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from verl.trainer.ppo import core_algos


def add_reward_decoupled_advantages(
    batch,
    reward_extra_infos_dict: dict,
    policy_loss_config,
    norm_adv_by_std_in_grpo: bool,
    tokenizer,
    *,
    _normalize_component_keys,
    _component_rewards_to_tensor,
):
    """Assemble sequence/token/format advantages; optionally localize token advantages."""
    from verl.trainer.ppo.ema_headroom import apply_ema_headroom_weights

    # EMA reward headroom → dynamic 2α branch loss scales (before GRPO adv).
    headroom_metrics = apply_ema_headroom_weights(
        batch, reward_extra_infos_dict, policy_loss_config
    )
    rare_metrics = {}
    try:
        from verl.trainer.ppo.neocxr_rare_metrics import rare_diagnosis_batch_metrics

        rare_metrics = rare_diagnosis_batch_metrics(
            reward_extra_infos_dict, policy_loss_config
        )
    except Exception:
        rare_metrics = {}
    merged_adv_metrics = {**headroom_metrics, **rare_metrics}
    if merged_adv_metrics:
        meta = getattr(batch, "meta_info", None)
        if isinstance(meta, dict):
            metrics = meta.setdefault("metrics", {})
            if isinstance(metrics, dict):
                metrics.update(merged_adv_metrics)

    sequence_keys = _normalize_component_keys(
        policy_loss_config.get("reward_decoupled_sequence_keys", [])
    )
    token_keys = _normalize_component_keys(policy_loss_config.get("reward_decoupled_token_keys", []))
    format_token_keys = _normalize_component_keys(
        policy_loss_config.get("reward_decoupled_format_token_keys", [])
    )

    sequence_rewards = _component_rewards_to_tensor(batch, reward_extra_infos_dict, sequence_keys)
    token_rewards = _component_rewards_to_tensor(batch, reward_extra_infos_dict, token_keys)
    format_token_rewards = _component_rewards_to_tensor(
        batch, reward_extra_infos_dict, format_token_keys
    )

    sequence_advantages, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=sequence_rewards,
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    token_advantages, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    format_token_advantages, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=format_token_rewards,
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )

    batch.batch["sequence_advantages"] = sequence_advantages

    localize_chex = bool(policy_loss_config.get("localize_chexbert_token_advantage", False))
    localize_neocxr = bool(policy_loss_config.get("localize_neocxr_diagnosis_token_advantage", False))
    adapt_head_only = bool(policy_loss_config.get("adaptive_branch_clip_head_only", False))
    response_mask = batch.batch["response_mask"]
    bg_weight = float(policy_loss_config.get("diag_background_weight", 0.05))
    debug_interval = int(policy_loss_config.get("diag_localization_debug_interval", 0))

    if localize_chex and "chex_f1_part" in token_keys:
        from verl.trainer.ppo.diagnostic_localization import (
            build_diagnostic_token_mask,
            build_diagnostic_token_mask_and_rarity,
            build_localized_diag_advantage,
        )

        rarity_by_label = None
        if adapt_head_only:
            from verl.trainer.ppo.class_aware_clip import class_frequency_records

            _, _, rarity_by_label = class_frequency_records(policy_loss_config)

        if rarity_by_label is not None:
            diag_key_token_mask, adapt_token_rarity = build_diagnostic_token_mask_and_rarity(
                response_ids=batch.batch["responses"],
                completion_mask=response_mask,
                tokenizer=tokenizer,
                predicted_labels=reward_extra_infos_dict.get("pred_chexbert_statuses"),
                rarity_by_label=rarity_by_label,
                debug_interval=debug_interval,
            )
            batch.batch["class_rarity"] = adapt_token_rarity
        else:
            diag_key_token_mask = build_diagnostic_token_mask(
                response_ids=batch.batch["responses"],
                completion_mask=response_mask,
                tokenizer=tokenizer,
                predicted_labels=reward_extra_infos_dict.get("pred_chexbert_statuses"),
                debug_interval=debug_interval,
            )

        lengths = response_mask.sum(dim=-1).clamp(min=1)
        report_diag_advantage = (token_advantages * response_mask).sum(dim=-1) / lengths
        token_advantages = build_localized_diag_advantage(
            report_advantage=report_diag_advantage,
            key_token_mask=diag_key_token_mask,
            completion_mask=response_mask,
            background_weight=bg_weight,
        )
        batch.batch["diag_key_token_mask"] = diag_key_token_mask

    elif localize_neocxr and (
        "cls_reward" in token_keys
        or "gated_cls_reward" in token_keys
        or "gated_cls_fmt" in token_keys
    ):
        from verl.trainer.ppo.diagnostic_localization import (
            build_localized_diag_advantage,
            build_neocxr_diagnosis_field_token_mask,
        )

        diag_key_token_mask = build_neocxr_diagnosis_field_token_mask(
            response_ids=batch.batch["responses"],
            completion_mask=response_mask,
            tokenizer=tokenizer,
            debug_interval=debug_interval,
        )
        lengths = response_mask.sum(dim=-1).clamp(min=1)
        report_diag_advantage = (token_advantages * response_mask).sum(dim=-1) / lengths
        token_advantages = build_localized_diag_advantage(
            report_advantage=report_diag_advantage,
            key_token_mask=diag_key_token_mask,
            completion_mask=response_mask,
            background_weight=bg_weight,
        )
        batch.batch["diag_key_token_mask"] = diag_key_token_mask

        if adapt_head_only:
            from verl.trainer.ppo.class_aware_clip import _LABEL_ALIASES, class_frequency_records

            _, _, rarity_by_label = class_frequency_records(policy_loss_config)
            sample_rarities: list[float] = []
            for info in batch.non_tensor_batch["extra_info"]:
                labels = (info or {}).get("disease_label")
                if isinstance(labels, str):
                    raw = [labels]
                elif isinstance(labels, (list, tuple, set)):
                    raw = list(labels)
                else:
                    raw = []
                canon = [
                    _LABEL_ALIASES.get(str(label).strip().lower(), str(label).strip().lower())
                    for label in raw
                ]
                rarities = [
                    float(rarity_by_label[label]) for label in canon if label in rarity_by_label
                ]
                sample_rarities.append(max(rarities) if rarities else 0.0)

            rarity = torch.tensor(
                sample_rarities, dtype=torch.float32, device=response_mask.device
            )
            adapt_token_rarity = (
                rarity.unsqueeze(-1).expand_as(response_mask).to(dtype=torch.float32)
            )
            adapt_token_rarity = torch.where(
                diag_key_token_mask.bool(),
                adapt_token_rarity,
                torch.zeros_like(adapt_token_rarity),
            )
            batch.batch["class_rarity"] = adapt_token_rarity

    batch.batch["token_advantages"] = token_advantages
    batch.batch["format_token_advantages"] = format_token_advantages

    # Apply 2α scales to seq/tok advantages (after localization).
    from verl.trainer.ppo.ema_headroom import apply_ema_headroom_loss_scales

    loss_headroom_metrics = apply_ema_headroom_loss_scales(batch, policy_loss_config)
    if loss_headroom_metrics:
        meta = getattr(batch, "meta_info", None)
        if isinstance(meta, dict):
            metrics = meta.setdefault("metrics", {})
            if isinstance(metrics, dict):
                metrics.update(loss_headroom_metrics)

    from verl.trainer.ppo.class_aware_clip import class_aware_enabled

    if not class_aware_enabled(policy_loss_config):
        return

    from verl.trainer.ppo.class_aware_clip import (
        build_class_aware_tensors,
        build_rarity_tp_token_tensors,
    )

    if policy_loss_config.get("class_aware_clip_mode", "") == "rarity_tp_token":
        cac_tensors = build_rarity_tp_token_tensors(
            batch.non_tensor_batch["extra_info"],
            batch.batch["response_mask"],
            batch.batch["responses"],
            tokenizer,
            policy_loss_config,
            reward_extra_infos_dict,
        )
        for key, value in cac_tensors.items():
            batch.batch[key] = value
    else:
        rarity, weight, positive_clip, tier = build_class_aware_tensors(
            batch.non_tensor_batch["extra_info"],
            batch.batch["response_mask"],
            policy_loss_config,
            reward_extra_infos_dict,
        )
        batch.batch["class_rarity"] = rarity
        batch.batch["class_weight"] = weight
        batch.batch["class_positive_clip"] = positive_clip
        batch.batch["class_tier"] = tier

    if not policy_loss_config.get("class_aware_separate_branch_enabled", False):
        return

    class_keys = _normalize_component_keys(
        policy_loss_config.get("class_aware_reward_keys", [])
    )
    class_rewards = _component_rewards_to_tensor(batch, reward_extra_infos_dict, class_keys)
    class_advantages, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=class_rewards,
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )

    if policy_loss_config.get("class_aware_apply_weight", True):
        response_mask = batch.batch["response_mask"]
        lengths = response_mask.sum(dim=-1).clamp(min=1)
        scalar_advantage = (class_advantages * response_mask).sum(dim=-1) / lengths
        weighted_scalar = scalar_advantage * weight[:, 0]
        centered_scalar = torch.empty_like(weighted_scalar)
        uids = batch.non_tensor_batch["uid"]
        for uid in np.unique(uids):
            group_mask = torch.as_tensor(uids == uid, device=weighted_scalar.device)
            centered_scalar[group_mask] = weighted_scalar[group_mask] - weighted_scalar[
                group_mask
            ].mean()
        class_advantages = centered_scalar.unsqueeze(-1) * response_mask

    batch.batch["class_advantages"] = class_advantages
