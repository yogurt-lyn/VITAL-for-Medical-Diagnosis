"""Shim: load recovered module bytecode, then override broken localization calls.

The bytecode body still provides RayPPOTrainer and most helpers. Only
``_add_reward_decoupled_advantages`` is replaced with a source implementation that
explicitly passes ``key_token_mask`` into ``build_localized_diag_advantage``.
"""
from __future__ import annotations

import marshal
from pathlib import Path

_backup = str(Path(__file__).resolve().parents[2] / "_bytecode_backup" / "ray_trainer.pyc")
with open(_backup, "rb") as _f:
    _data = _f.read()
_code = marshal.loads(_data[16:])
del _backup, _f, _data, marshal, Path
exec(_code, globals())

# Explicit source override (fixes missing key_token_mask in recovered bytecode).
from verl.trainer.ppo._reward_decoupled_advantages import (  # noqa: E402
    add_reward_decoupled_advantages as _add_reward_decoupled_advantages_src,
)

_normalize_component_keys_bc = _normalize_component_keys  # noqa: F821
_component_rewards_to_tensor_bc = _component_rewards_to_tensor  # noqa: F821


def _add_reward_decoupled_advantages(
    batch,
                reward_extra_infos_dict,
                policy_loss_config,
    norm_adv_by_std_in_grpo,
        tokenizer,
):
    return _add_reward_decoupled_advantages_src(
                            batch,
        reward_extra_infos_dict,
        policy_loss_config,
        norm_adv_by_std_in_grpo,
        tokenizer,
        _normalize_component_keys=_normalize_component_keys_bc,
        _component_rewards_to_tensor=_component_rewards_to_tensor_bc,
    )
