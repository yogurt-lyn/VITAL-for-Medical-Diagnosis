"""EMA reward headroom → dynamic dual-branch loss weights.

Per step t, for b ∈ {seq, tok}:

  R̄_b^t = mean_batch(R_{i,b}^t)          # raw reward in [0,1]
  M_b^t  = β M_b^{t-1} + (1-β) R̄_b^t     # EMA of reward level
  H_b^t  = 1 - M_b^t                      # remaining headroom
  α_b^t  = H_b^t / (H_seq^t + H_tok^t + ε)

Policy loss:

  L = 2 α_seq L_seq + 2 α_tok L_tok       # keeps Σ coef = 2

Applied by scaling sequence/token advantages by ``2 α_b`` (same effect when
loss is linear in each branch advantage).

Enable: NEOCXR_EMA_HEADROOM=1 or policy_loss.ema_headroom_enabled=true
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

# Process-local EMA state (advantage assembly runs on trainer process).
_M_SEQ: float | None = None
_M_TOK: float | None = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def ema_headroom_enabled(policy_loss_config: Any | None = None) -> bool:
    if policy_loss_config is not None:
        val = policy_loss_config.get("ema_headroom_enabled", None)
        if val is None:
            # Legacy Hydra key from the Focal prototype.
            val = policy_loss_config.get("focal_reward_enabled", None)
        if val is not None:
            return bool(val)
    return (
        _env_bool("NEOCXR_EMA_HEADROOM", False)
        or _env_bool("EMA_HEADROOM", False)
        or _env_bool("NEOCXR_FOCAL_REWARD", False)
    )


def reset_ema_state() -> None:
    global _M_SEQ, _M_TOK
    _M_SEQ = None
    _M_TOK = None


def compute_ema_headroom_scales(
    r_seq_mean: float,
    r_tok_mean: float,
    *,
    beta: float = 0.9,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Update EMA and return ``2α`` loss scales plus diagnostics."""
    global _M_SEQ, _M_TOK

    r_seq = float(np.clip(r_seq_mean, 0.0, 1.0))
    r_tok = float(np.clip(r_tok_mean, 0.0, 1.0))
    beta = float(np.clip(beta, 0.0, 0.999999))
    eps = max(float(eps), 1e-12)

    if _M_SEQ is None:
        _M_SEQ = r_seq
    else:
        _M_SEQ = beta * _M_SEQ + (1.0 - beta) * r_seq
    if _M_TOK is None:
        _M_TOK = r_tok
    else:
        _M_TOK = beta * _M_TOK + (1.0 - beta) * r_tok

    h_seq = 1.0 - float(_M_SEQ)
    h_tok = 1.0 - float(_M_TOK)
    denom = h_seq + h_tok + eps
    alpha_seq = h_seq / denom
    alpha_tok = h_tok / denom
    return {
        "m_seq": float(_M_SEQ),
        "m_tok": float(_M_TOK),
        "h_seq": h_seq,
        "h_tok": h_tok,
        "alpha_seq": alpha_seq,
        "alpha_tok": alpha_tok,
        "scale_seq": 2.0 * alpha_seq,
        "scale_tok": 2.0 * alpha_tok,
        "r_seq_mean": r_seq,
        "r_tok_mean": r_tok,
    }


def _col_mean(extra: dict, key: str, *fallbacks: str) -> float:
    vals = extra.get(key)
    for fb in fallbacks:
        if vals is not None:
            break
        vals = extra.get(fb)
    if vals is None:
        raise KeyError(f"ema_headroom missing reward key '{key}'")
    arr = np.asarray([float(x) for x in vals], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.clip(arr, 0.0, 1.0).mean())


def apply_ema_headroom_weights(
    batch,
    reward_extra_infos_dict: dict,
    policy_loss_config: Any | None = None,
) -> dict[str, float]:
    """Compute batch-level ``2α`` scales from EMA headroom; stash on batch."""
    if not ema_headroom_enabled(policy_loss_config):
        return {}

    cfg = policy_loss_config or {}
    beta = float(cfg.get("ema_headroom_beta", _env_float("NEOCXR_EMA_HEADROOM_BETA", 0.9)))
    eps = float(cfg.get("ema_headroom_eps", _env_float("NEOCXR_EMA_HEADROOM_EPS", 1e-6)))

    r_seq = _col_mean(reward_extra_infos_dict, "rouge_l")
    r_tok = _col_mean(reward_extra_infos_dict, "cls_f1", "class_aware_cls_f1")
    out = compute_ema_headroom_scales(r_seq, r_tok, beta=beta, eps=eps)

    n = len(batch.non_tensor_batch["uid"])
    scale_seq = np.full(n, out["scale_seq"], dtype=np.float32)
    scale_tok = np.full(n, out["scale_tok"], dtype=np.float32)

    reward_extra_infos_dict["ema_headroom_scale_seq"] = [float(out["scale_seq"])] * n
    reward_extra_infos_dict["ema_headroom_scale_tok"] = [float(out["scale_tok"])] * n
    reward_extra_infos_dict["ema_headroom_alpha_seq"] = [float(out["alpha_seq"])] * n
    reward_extra_infos_dict["ema_headroom_alpha_tok"] = [float(out["alpha_tok"])] * n
    reward_extra_infos_dict["ema_headroom_M_seq"] = [float(out["m_seq"])] * n
    reward_extra_infos_dict["ema_headroom_M_tok"] = [float(out["m_tok"])] * n

    # Keep old focal_* keys so existing log scrapers still see scales.
    reward_extra_infos_dict["focal_loss_scale_seq"] = list(map(float, scale_seq))
    reward_extra_infos_dict["focal_loss_scale_tok"] = list(map(float, scale_tok))

    if getattr(batch, "batch", None) is not None:
        batch.batch["focal_loss_scale_seq"] = torch.as_tensor(scale_seq)
        batch.batch["focal_loss_scale_tok"] = torch.as_tensor(scale_tok)
        batch.batch["ema_headroom_scale_seq"] = torch.as_tensor(scale_seq)
        batch.batch["ema_headroom_scale_tok"] = torch.as_tensor(scale_tok)

    return {
        "ema_headroom/M_seq": out["m_seq"],
        "ema_headroom/M_tok": out["m_tok"],
        "ema_headroom/H_seq": out["h_seq"],
        "ema_headroom/H_tok": out["h_tok"],
        "ema_headroom/alpha_seq": out["alpha_seq"],
        "ema_headroom/alpha_tok": out["alpha_tok"],
        "ema_headroom/scale_seq": out["scale_seq"],
        "ema_headroom/scale_tok": out["scale_tok"],
        "ema_headroom/Rbar_seq": out["r_seq_mean"],
        "ema_headroom/Rbar_tok": out["r_tok_mean"],
        "ema_headroom/beta": beta,
    }


def apply_ema_headroom_loss_scales(
    batch,
    policy_loss_config: Any | None = None,
) -> dict[str, float]:
    """Scale seq/tok advantages by ``2α`` (dynamic branch loss coefs)."""
    if not ema_headroom_enabled(policy_loss_config):
        return {}
    key_seq = "ema_headroom_scale_seq" if "ema_headroom_scale_seq" in batch.batch else "focal_loss_scale_seq"
    key_tok = "ema_headroom_scale_tok" if "ema_headroom_scale_tok" in batch.batch else "focal_loss_scale_tok"
    if key_seq not in batch.batch:
        return {}

    device = batch.batch["response_mask"].device
    dtype = batch.batch["sequence_advantages"].dtype
    scale_seq = batch.batch[key_seq].to(device=device, dtype=dtype).unsqueeze(-1)
    batch.batch["sequence_advantages"] = batch.batch["sequence_advantages"] * scale_seq
    metrics = {"ema_headroom/applied_scale_seq": float(scale_seq.mean().item())}

    if "token_advantages" in batch.batch and key_tok in batch.batch:
        scale_tok = batch.batch[key_tok].to(device=device, dtype=dtype).unsqueeze(-1)
        batch.batch["token_advantages"] = batch.batch["token_advantages"] * scale_tok
        metrics["ema_headroom/applied_scale_tok"] = float(scale_tok.mean().item())
    return metrics


# --- Compatibility shims (old focal_* call sites) ---------------------------------

def apply_focal_reward_reweight(batch, reward_extra_infos_dict, policy_loss_config=None):
    return apply_ema_headroom_weights(batch, reward_extra_infos_dict, policy_loss_config)


def apply_focal_loss_branch_scales(batch, policy_loss_config=None):
    return apply_ema_headroom_loss_scales(batch, policy_loss_config)
