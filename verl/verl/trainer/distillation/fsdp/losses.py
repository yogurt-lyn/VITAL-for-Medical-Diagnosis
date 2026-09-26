# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def _amid_cfg_float(loss_config: DistillationLossConfig, name: str, env_key: str, default: float) -> float:
    val = getattr(loss_config, name, None)
    if val is None:
        raw = os.environ.get(env_key)
        val = float(raw) if raw is not None else default
    return float(val)


def _amid_cfg_str(loss_config: DistillationLossConfig, name: str, env_key: str, default: str) -> str:
    val = getattr(loss_config, name, None)
    if val is None:
        val = os.environ.get(env_key, default)
    return str(val)


def amid_divergence_topk(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    *,
    alpha: float = 0.5,
    lam: float = 0.5,
    div_name: str = "ab",
    div_order: str = "pr",
    ab_alpha: float = 0.2,
    ab_beta: float = 0.7,
) -> torch.Tensor:
    """AMiD on the teacher top-k support (renormalized). Returns (...,) token losses.

    Mirrors aailab-kaist/AMiD ``amid()`` with paper defaults:
    ``div_name=ab``, ``div_order=pr``, ``alpha=0.5``, ``lam=0.5``.
    """
    # Restrict both distributions to the shared top-k support.
    log_p = F.log_softmax(teacher_topk_log_probs.float(), dim=-1)
    log_q = F.log_softmax(student_topk_log_probs.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()

    if lam <= 0.0:
        log_r = log_q
        r = q
    elif lam >= 1.0:
        log_r = log_p
        r = p
    elif alpha >= 1.0:
        logr_unnorm = lam * log_p + (1.0 - lam) * log_q
        log_r = F.log_softmax(logr_unnorm, dim=-1)
        r = log_r.exp()
    else:
        t1 = math.log(lam) + 0.5 * (1.0 - alpha) * log_p
        t2 = math.log(1.0 - lam) + 0.5 * (1.0 - alpha) * log_q
        logr_unnorm = (2.0 / (1.0 - alpha)) * torch.logaddexp(t1, t2)
        log_r = F.log_softmax(logr_unnorm, dim=-1)
        r = log_r.exp()

    div_name = div_name.lower()
    div_order = div_order.lower()
    if div_name == "fkl":
        if div_order == "pr":
            prod = p * (log_p - log_r)
        elif div_order == "qr":
            prod = q * (log_q - log_r)
        elif div_order == "rp":
            prod = r * (log_r - log_p)
        elif div_order == "rq":
            prod = r * (log_r - log_q)
        else:
            raise ValueError(f"Unsupported amid div_order={div_order!r} for fkl")
        return prod.sum(dim=-1)

    if div_name == "ab":
        apb = ab_alpha + ab_beta
        if div_order == "pr":
            a_log, b_log = log_p, log_r
        elif div_order == "qr":
            a_log, b_log = log_q, log_r
        elif div_order == "rp":
            a_log, b_log = log_r, log_p
        elif div_order == "rq":
            a_log, b_log = log_r, log_q
        else:
            raise ValueError(f"Unsupported amid div_order={div_order!r} for ab")
        term1 = torch.exp(torch.logsumexp(ab_alpha * a_log + ab_beta * b_log, dim=-1))
        term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * a_log, dim=-1))
        term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * b_log, dim=-1))
        divergence = -(term1 - term2 - term3) / (ab_alpha * ab_beta)
        return torch.where(torch.isfinite(divergence), divergence, torch.zeros_like(divergence))

    raise ValueError(f"Unsupported amid div_name={div_name!r}")


def jsd_divergence_topk(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Jensen–Shannon divergence on the shared top-k support (CVPD latent-transfer term).

    Matches mbzuai-oryx/CVPD ``_jsd`` / ``loss_pos = D_JS(π_+ ‖ π_s)``, with
    π_+ approximated by the external teacher top-k distribution.
    """
    eps = 1e-10
    log_p = F.log_softmax(teacher_topk_log_probs.float(), dim=-1)
    log_q = F.log_softmax(student_topk_log_probs.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    midpoint = 0.5 * (p + q)
    log_mid = midpoint.add(eps).log()
    return 0.5 * (
        (p * (log_p - log_mid)).sum(dim=-1) + (q * (log_q - log_mid)).sum(dim=-1)
    )


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute top-k distillation loss (forward KL, AMiD, or CVPD-JSD).

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise distillation on the teacher top-k support
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)

    loss_mode = str(getattr(loss_config, "loss_mode", "forward_kl_topk")).lower()
    if loss_mode in ("amid", "amid_topk", "amid_ab"):
        distillation_losses = amid_divergence_topk(
            student_topk_log_probs,
            teacher_topk_log_probs,
            alpha=_amid_cfg_float(loss_config, "amid_alpha", "AMID_ALPHA", 0.5),
            lam=_amid_cfg_float(loss_config, "amid_lam", "AMID_LAM", 0.5),
            div_name=_amid_cfg_str(loss_config, "amid_div_name", "AMID_DIV_NAME", "ab"),
            div_order=_amid_cfg_str(loss_config, "amid_div_order", "AMID_DIV_ORDER", "pr"),
            ab_alpha=_amid_cfg_float(loss_config, "amid_ab_alpha", "AMID_AB_ALPHA", 0.2),
            ab_beta=_amid_cfg_float(loss_config, "amid_ab_beta", "AMID_AB_BETA", 0.7),
        )
    elif loss_mode in ("cvpd", "cvpd_jsd", "jsd", "jsd_topk"):
        # CVPD paper latent-transfer term only (no crop/ghost ranking / discovery).
        distillation_losses = jsd_divergence_topk(student_topk_log_probs, teacher_topk_log_probs)
    else:
        distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    return {"distillation_losses": distillation_losses, "student_mass": student_mass, "teacher_mass": teacher_mass}
