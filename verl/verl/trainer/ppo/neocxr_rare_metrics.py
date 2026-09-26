"""Batch diagnostics for NeoCXR rare-GT vs diagnosis accuracy (logged into trainer metrics)."""

from __future__ import annotations

from typing import Any

import numpy as np

from verl.trainer.ppo.class_aware_clip import _LABEL_ALIASES, class_frequency_records


_RARE_THRESHOLD = 0.4  # bpd≈0.42 and rarer in class_frequencies.json


def _canon_labels(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        parts = [str(x) for x in value]
    else:
        parts = str(value).replace("|", ",").split(",")
    out = set()
    for part in parts:
        key = _LABEL_ALIASES.get(part.strip().lower(), part.strip().lower())
        if key:
            out.add(key)
    return out


def rare_diagnosis_batch_metrics(
    reward_extra_infos_dict: dict,
    policy_loss_config: Any,
    *,
    rare_threshold: float = _RARE_THRESHOLD,
) -> dict[str, float]:
    """Summarise rare-GT coverage and inaccurate-diagnosis rate for the batch."""
    try:
        _, _, rarity_by_label = class_frequency_records(policy_loss_config)
    except Exception:
        return {}

    n = None
    for key in ("cls_f1", "ref_label_list", "pred_label_list", "score"):
        vals = reward_extra_infos_dict.get(key)
        if vals is not None:
            n = len(vals)
            break
    if not n:
        return {}

    cls_f1 = reward_extra_infos_dict.get("cls_f1")
    ref_lists = reward_extra_infos_dict.get("ref_label_list")
    pred_lists = reward_extra_infos_dict.get("pred_label_list")
    # disease_label from extra_info is more reliable for GT when present
    # (ref_label_list is encoded string); fall back to ref_label_list.

    rare_flags = []
    inaccurate_rare = []
    f1_rare = []
    f1_head = []
    ratio_proxy_wrong_rare = 0  # count only

    for i in range(n):
        ref = set()
        if ref_lists is not None:
            ref |= _canon_labels(ref_lists[i])
        pred = _canon_labels(pred_lists[i]) if pred_lists is not None else set()
        f1 = float(cls_f1[i]) if cls_f1 is not None else float("nan")
        rare_score = max((rarity_by_label.get(lab, 0.0) for lab in ref), default=0.0)
        is_rare = rare_score >= rare_threshold
        rare_flags.append(is_rare)
        wrong = (not np.isnan(f1) and f1 < 0.999) or (pred != ref and ref)
        if is_rare:
            inaccurate_rare.append(1.0 if wrong else 0.0)
            if not np.isnan(f1):
                f1_rare.append(f1)
            if wrong:
                ratio_proxy_wrong_rare += 1
        else:
            if not np.isnan(f1):
                f1_head.append(f1)

    rare_flags_a = np.asarray(rare_flags, dtype=np.float64)
    out = {
        "neocxr/rare_gt_frac": float(rare_flags_a.mean()) if len(rare_flags_a) else 0.0,
        "neocxr/rare_gt_count": float(rare_flags_a.sum()),
        "neocxr/batch_size": float(n),
    }
    if inaccurate_rare:
        out["neocxr/rare_inaccurate_frac"] = float(np.mean(inaccurate_rare))
        out["neocxr/rare_inaccurate_count"] = float(np.sum(inaccurate_rare))
    else:
        out["neocxr/rare_inaccurate_frac"] = 0.0
        out["neocxr/rare_inaccurate_count"] = 0.0
    if f1_rare:
        out["neocxr/cls_f1_rare_mean"] = float(np.mean(f1_rare))
    if f1_head:
        out["neocxr/cls_f1_head_mean"] = float(np.mean(f1_head))
    return out
