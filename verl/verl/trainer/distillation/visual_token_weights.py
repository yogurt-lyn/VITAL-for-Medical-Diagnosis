"""Teacher-visual token gates for OPD (no hard top-k by default).

Includes Med-OPD / VAOPD-style MEA reweighting (gate=vaopd):
  m_t = w(k) * g_t
  - w(k): softmax over z-scored mean-VA within contiguous rollout groups
  - g_t: top-p_v tokens get λ/|V|, rest (1-λ)/|L|
"""

from __future__ import annotations

import math
import os

import torch


def compute_teacher_visual_token_weights(
    sensitivity: torch.Tensor,
    response_mask: torch.Tensor,
    gate: str = "sigmoid",
    top_fraction: float = 0.3,
    sigmoid_tau: float = 0.5,
    sigmoid_bias: float = 0.0,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map |Δlogp| sensitivity to per-token OPD weights.

    ``gate``:
      - ``sigmoid``: z-score within the response, then
        ``σ((z - bias) / τ)``. Low-sensitivity tokens are decayed, not zeroed.
      - ``dense``: raw ``s_t`` (same as former ``top_fraction=1``).
      - ``topk``: hard mask on the top fraction (legacy vis-only).
      - ``vaopd`` / ``medopd``: Med-OPD grouped MEA weights ``m_t = w(k)·g_t``
        (env: VAOPD_PV, VAOPD_LAMBDA, VAOPD_TAU, VAOPD_GROUP_SIZE).

    When ``normalize`` is true (except vaopd), divide by the mean weight over
    response tokens so ``E[w]=1`` on each sequence.
    """
    if sensitivity.shape != response_mask.shape:
        raise ValueError(
            f"sensitivity {tuple(sensitivity.shape)} != mask {tuple(response_mask.shape)}"
        )
    gate_name = str(gate or "sigmoid").lower()
    mask_f = response_mask.to(dtype=sensitivity.dtype)
    mask_bool = response_mask.bool()

    if gate_name in ("sigmoid", "sig", "soft"):
        tau = float(sigmoid_tau)
        if tau <= 0.0:
            raise ValueError(f"sigmoid_tau must be > 0, got {tau}")
        token_count = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (sensitivity * mask_f).sum(dim=-1, keepdim=True) / token_count
        var = (((sensitivity - mean) ** 2) * mask_f).sum(dim=-1, keepdim=True) / token_count
        std = var.sqrt().clamp_min(1e-6)
        z = (sensitivity - mean) / std
        raw = torch.sigmoid((z - float(sigmoid_bias)) / tau) * mask_f
        gate_mask = raw
    elif gate_name in ("dense", "linear", "softlinear"):
        raw = sensitivity * mask_f
        gate_mask = mask_f
    elif gate_name in ("topk", "top_k", "hard"):
        fraction = float(top_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"top_fraction must be in (0, 1], got {fraction}")
        if fraction >= 1.0:
            raw = sensitivity * mask_f
            gate_mask = mask_f
        else:
            gate_mask = torch.zeros_like(sensitivity)
            for row in range(sensitivity.shape[0]):
                valid_indices = mask_bool[row].nonzero(as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    continue
                selected_count = max(1, math.ceil(valid_indices.numel() * fraction))
                row_sensitivity = sensitivity[row, valid_indices]
                selected_local = torch.topk(
                    row_sensitivity, k=selected_count, largest=True, sorted=False
                ).indices
                gate_mask[row, valid_indices[selected_local]] = 1.0
            raw = gate_mask
    elif gate_name in ("vaopd", "medopd", "mea", "va"):
        # Med-OPD / VAOPD (Liu et al.): m_t = w(k) * g_t
        pv = float(os.getenv("VAOPD_PV", "0.2"))
        lam = float(os.getenv("VAOPD_LAMBDA", "0.5"))
        tau = float(os.getenv("VAOPD_TAU", "1.0"))
        group_size = int(os.getenv("VAOPD_GROUP_SIZE", os.getenv("ROLLOUT_N", "4")))
        if not 0.0 < pv <= 1.0:
            raise ValueError(f"VAOPD_PV must be in (0, 1], got {pv}")
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"VAOPD_LAMBDA must be in [0, 1], got {lam}")
        if tau <= 0.0:
            raise ValueError(f"VAOPD_TAU must be > 0, got {tau}")
        if group_size < 1:
            raise ValueError(f"VAOPD_GROUP_SIZE must be >= 1, got {group_size}")

        bsz, _seqlen = sensitivity.shape
        # Token group weight g_t (sums to 1 per rollout)
        g = torch.zeros_like(sensitivity)
        gate_mask = torch.zeros_like(sensitivity)
        for row in range(bsz):
            valid_indices = mask_bool[row].nonzero(as_tuple=False).squeeze(-1)
            n_valid = int(valid_indices.numel())
            if n_valid == 0:
                continue
            n_high = max(1, int(math.ceil(n_valid * pv)))
            n_high = min(n_high, n_valid)
            row_va = sensitivity[row, valid_indices]
            top_local = torch.topk(row_va, k=n_high, largest=True, sorted=False).indices
            high = torch.zeros(n_valid, dtype=torch.bool, device=sensitivity.device)
            high[top_local] = True
            low = ~high
            n_low = int(low.sum().item())
            g_row = torch.zeros(n_valid, dtype=sensitivity.dtype, device=sensitivity.device)
            g_row[high] = float(lam) / float(n_high)
            if n_low > 0:
                g_row[low] = float(1.0 - lam) / float(n_low)
            else:
                # all tokens in high group: put remaining mass on high
                g_row[high] = 1.0 / float(n_high)
            g[row, valid_indices] = g_row
            gate_mask[row, valid_indices[high]] = 1.0

        # Trajectory weight w(k): z-score mean-VA within contiguous groups, softmax
        tok_count = mask_f.sum(dim=-1).clamp_min(1.0)
        avg_va = (sensitivity * mask_f).sum(dim=-1) / tok_count
        w = torch.ones(bsz, dtype=sensitivity.dtype, device=sensitivity.device)
        for start in range(0, bsz, group_size):
            end = min(start + group_size, bsz)
            grp = avg_va[start:end]
            if grp.numel() == 1:
                w[start:end] = 1.0
                continue
            mean = grp.mean()
            std = grp.std(unbiased=False).clamp_min(1e-6)
            z = (grp - mean) / std
            w[start:end] = torch.softmax(z / tau, dim=0)

        raw = w.unsqueeze(-1) * g
        # Paper m_t already normalized within prompt/rollout; skip E[w]=1 rescale.
        weights = raw
        return weights, gate_mask
    else:
        raise ValueError(
            f"teacher_visual_token_gate must be sigmoid, dense, topk, or vaopd, got {gate!r}"
        )

    weights = raw
    if normalize:
        token_count = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_w = (weights * mask_f).sum(dim=-1, keepdim=True) / token_count
        weights = weights / mean_w.clamp_min(1e-6)
        weights = weights * mask_f
    return weights, gate_mask
