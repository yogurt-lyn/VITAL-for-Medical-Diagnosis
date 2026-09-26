"""MIMIC-CXR reward: load recovered bytecode, then optionally apply head-FP penalty.

Base scorer (bytecode) returns ROUGE-1/L + CheXbert CE-F1 parts.
When MIMICCXR_HEAD_FP_PEN=1, subtract a light frequency-weighted penalty for
false-positive predictions on high-frequency CheXbert classes from chex_f1_part
(token branch), mirroring NeoCXR's NEOCXR_HEAD_FP_PEN at small weight.
"""

from __future__ import annotations

import marshal
import os
from pathlib import Path

_backup = str(Path(__file__).resolve().parents[2] / "_bytecode_backup" / "mimiccxr.pyc")
with open(_backup, "rb") as _f:
    _data = _f.read()
    _code = marshal.loads(_data[16:])
del _backup, _f, _data, marshal, Path
exec(_code, globals())

_ORIG_COMPUTE_SCORE = compute_score  # noqa: F821 — from bytecode


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


# Default: most frequent positive labels in rl_3k train counts (excl. rare tails).
_DEFAULT_HEAD_FP_CLASSES = (
    "lung_opacity,pleural_effusion,support_devices,cardiomegaly,atelectasis"
)


def _norm_label(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _head_fp_classes() -> set[str]:
    raw = os.environ.get("MIMICCXR_HEAD_FP_CLASSES", _DEFAULT_HEAD_FP_CLASSES)
    return {_norm_label(p) for p in raw.split(",") if p.strip()}


def _labels_from_positive_tuple(pos) -> set[str]:
    conditions = list(globals().get("_CHEXBERT_CONDITIONS") or [])
    out: set[str] = set()
    if not conditions or pos is None:
        return out
    for name, flag in zip(conditions, pos):
        if flag:
            out.add(_norm_label(name))
    return out


def _labels_from_extra(extra: dict, key: str) -> set[str]:
    raw = extra.get(key)
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple)):
        # may be names or 0/1 flags aligned to CONDITIONS
        if raw and all(isinstance(x, (int, float, bool)) for x in raw):
            return _labels_from_positive_tuple(raw)
        return {_norm_label(x) for x in raw if x}
    if isinstance(raw, str):
        return {_norm_label(p) for p in raw.split(",") if p.strip()}
    return set()


def _head_fp_penalty(pred_labels: set[str], ref_labels: set[str]) -> tuple[float, float, int]:
    """Return (penalty, head_fp_rate, n_head_fps)."""
    if not _env_bool("MIMICCXR_HEAD_FP_PEN", False):
        return 0.0, 0.0, 0

    head = _head_fp_classes()
    if not head:
        return 0.0, 0.0, 0

    head_fps = (pred_labels - ref_labels) & head
    n = len(head_fps)
    if n <= 0:
        return 0.0, 0.0, 0

    # Frequency prior from train counts when available; else uniform within head.
    counts = {
        "lung_opacity": 906.0,
        "pleural_effusion": 845.0,
        "support_devices": 749.0,
        "cardiomegaly": 657.0,
        "atelectasis": 578.0,
        "edema": 554.0,
        "pneumonia": 304.0,
        "consolidation": 161.0,
        "lung_lesion": 116.0,
        "enlarged_cardiomediastinum": 98.0,
        "fracture": 87.0,
        "pneumothorax": 76.0,
        "pleural_other": 54.0,
    }
    max_count = max(counts.get(lbl, 1.0) for lbl in head_fps) or 1.0
    mass = sum(counts.get(lbl, max_count * 0.5) / max_count for lbl in head_fps)
    head_fp_rate = min(1.0, mass / max(1.0, float(len(head))))
    weight = max(0.0, _env_float("MIMICCXR_HEAD_FP_PEN_WEIGHT", 0.05))
    return weight * head_fp_rate, head_fp_rate, n


def compute_score(solution_str, ground_truth, extra_info=None, **kwargs):
    res = _ORIG_COMPUTE_SCORE(solution_str, ground_truth, extra_info=extra_info, **kwargs)

    # Normalize to dict (bytecode returns dict for decoupled training).
    if not isinstance(res, dict):
        if isinstance(res, (list, tuple)) and res:
            score0 = float(res[0])
            info = res[1] if len(res) > 1 and isinstance(res[1], dict) else {}
            res = {"score": score0, **info}
        else:
            return res

    if not _env_bool("MIMICCXR_HEAD_FP_PEN", False):
        res.setdefault("head_fp_pen", 0.0)
        res.setdefault("head_fp_rate", 0.0)
        res.setdefault("n_head_fps", 0.0)
        return res

    pred_labels = _labels_from_extra(res, "pred_chexbert_labels")
    ref_labels = _labels_from_extra(res, "ref_chexbert_labels")
    if not pred_labels or not ref_labels:
        # Fall back to re-labeling from texts (uses cached CheXbert).
        try:
            pred_text = globals()["_strip_thinking"](str(solution_str)).strip()
            ref_text = str(ground_truth).strip()
            pred_pos = globals()["_chexbert_positive_tuple"](pred_text)
            ref_pos = globals()["_chexbert_positive_tuple"](ref_text)
            if not pred_labels:
                pred_labels = _labels_from_positive_tuple(pred_pos)
            if not ref_labels:
                ref_labels = _labels_from_positive_tuple(ref_pos)
        except Exception:
            pass

    pen, rate, n_fps = _head_fp_penalty(pred_labels, ref_labels)
    chex = float(res.get("chex_f1_part", 0.0))
    chex = max(0.0, chex - pen)
    res["chex_f1_part"] = chex
    # Keep total score consistent with parts when present.
    if "score" in res:
        # Prefer recomposing from known parts when available.
        r1 = float(res.get("rouge1_part", 0.0))
        rl = float(res.get("rouge_l_part", res.get("rougel_part", 0.0)))
        if "rouge1_part" in res or "rouge_l_part" in res or "rougel_part" in res:
            res["score"] = max(0.0, min(1.0, r1 + rl + chex))
        else:
            res["score"] = max(0.0, float(res["score"]) - pen)

    res["head_fp_pen"] = float(pen)
    res["head_fp_rate"] = float(rate)
    res["n_head_fps"] = float(n_fps)
    res["reward_version"] = "mimiccxr_v1_rougel_chexbert_headfp"
    return res
