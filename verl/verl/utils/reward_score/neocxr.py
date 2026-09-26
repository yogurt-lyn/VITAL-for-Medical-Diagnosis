# NeoCXR GRPO reward — composite score (v2.9, equal-weight simple):
#   0.50 * ROUGE-L(full report)  +  0.50 * diagnosis_cls_fbeta  -  tpl_pen
#   optional: subtract diagnosis-field repeat penalty (duplicate disease mentions)
#
# v2.9.2: diagnosis repeat penalty (NEOCXR_DIAG_REPEAT_PEN) — discourage
# repeating the same disease phrase in Disease diagnosis; optional head-weighted
# cost so common labels hurt more when spammed (not via clip head-only).
# v2.9.1: re-enable template penalty on eq profile (generic imaging phrase collapse).
# v2.9: equal weights; drop gate / format from the score. Classification
# is plain precision-favoring F-beta (default 0.5). The RL trainer applies the
# cls advantage only on Disease-diagnosis field tokens via a response mask.
# v2.8: full-report ROUGE-L; no format.
# v2.6: diagnosis label-set F-beta (default beta=0.5).
#
# ground_truth format (from preprocess_neocxr_verl.py):
#   "Imaging conclusion: <text> Disease diagnosis: <text>"
#   extra_info["disease_label"] = "Neonatal Pneumonia, NRDS"  (comma-separated, optional)

from __future__ import annotations

import math
import os
import re
from collections import Counter

_IMG = "imaging conclusion:"
_DIS = "disease diagnosis:"

# Canonical disease names → normalised key (lowercase, no punctuation)
_DISEASE_ALIASES: dict[str, str] = {
    "neonatal pneumonia": "pneumonia",
    "neonatal respiratory distress syndrome": "nrds",
    "nrds": "nrds",
    "neonatal transient tachypnea": "ttn",
    "ttn": "ttn",
    "transient tachypnea of the newborn": "ttn",
    "bronchopulmonary dysplasia": "bpd",
    "bpd": "bpd",
    "pneumothorax": "pneumothorax",
    "pleural effusion": "pleural_effusion",
    "atelectasis": "atelectasis",
    "no obvious abnormalities": "normal",
    "no obvious abnormality": "normal",
    "normal": "normal",
}

# Generic imaging phrases that SFT/GRPO collapse onto (from badcase + v2.1/v2.2 preds).
_GENERIC_IMAGING_PHRASES: tuple[str, ...] = (
    "increased texture on both lungs",
    "increased texture in both lungs",
    "texture on both lungs becomes thicker",
    "texture on both lungs is slightly increased",
    "texture in both lungs is slightly increased",
    "texture in both lungs",
    "bilateral lung texture is increased",
    "both lungs were better than the previous radiograph",
    "both lungs exudate, similar to the previous radiograph",
    "morphology of intestinal lumen is irregular",
    "there were no obvious abnormalities in the abdomen",
    "no obvious abnormalities in the abdomen",
    "no obvious abnormality in the abdomen",
)


def _strip_thinking(s: str) -> str:
    s = s or ""
    bt = chr(96)
    s = re.sub(bt * 3 + r"thinking\s*[\s\S]*?" + bt * 3, "", s, flags=re.IGNORECASE)
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<redacted_reasoning>[\s\S]*?</redacted_reasoning>", "", s, flags=re.IGNORECASE)
    return s


def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_neocxr_fields(text: str) -> tuple[str | None, str | None]:
    low = text.lower()
    i1 = low.find(_IMG)
    i2 = low.find(_DIS)
    if i1 < 0 or i2 < 0 or i2 <= i1:
        return None, None
    img = text[i1 + len(_IMG) : i2].strip()
    dis = text[i2 + len(_DIS) :].strip()
    return img, dis


# ── ROUGE-L ──────────────────────────────────────────────────────────────────

def _lcs_length(a: list, b: list) -> int:
    """Length of LCS via DP (O(n*m), fine for short medical sentences)."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])
        prev = curr
    return prev[n]


def _rouge_l(pred: str, ref: str) -> float:
    pt = _normalize(pred).split()
    rt = _normalize(ref).split()
    if not rt:
        return 1.0 if not pt else 0.0
    if not pt:
        return 0.0
    lcs = _lcs_length(pt, rt)
    precision = lcs / len(pt)
    recall = lcs / len(rt)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _bleu1(pred: str, ref: str) -> float:
    pt = _normalize(pred).split()
    rt = _normalize(ref).split()
    if not rt:
        return 1.0 if not pt else 0.0
    if not pt:
        return 0.0
    pred_counts = Counter(pt)
    ref_counts = Counter(rt)
    overlap = sum(min(count, ref_counts[token]) for token, count in pred_counts.items())
    precision = overlap / max(len(pt), 1)
    bp = 1.0 if len(pt) > len(rt) else math.exp(1.0 - len(rt) / max(len(pt), 1))
    return max(0.0, min(1.0, bp * precision))


# ── Diagnosis classification F1 ──────────────────────────────────────────────

_CLASS_COUNTS: dict[str, int] = {
    # Counts from data/neocxr_verl/train_grpo.parquet.
    "pneumonia": 1708,
    "normal": 668,
    "nrds": 502,
    "ttn": 358,
    "bpd": 170,
    "pneumothorax": 41,
    "pleural_effusion": 8,
    "atelectasis": 7,
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _class_weight(label: str) -> float:
    """Smoothed inverse-frequency weight, clipped for reward stability."""
    max_count = max(_CLASS_COUNTS.values())
    count = max(1, _CLASS_COUNTS.get(label, max_count))
    raw = math.sqrt(max_count / count)
    cap = _env_float("NEOCXR_CLASS_WEIGHT_MAX", 4.0)
    return max(1.0, min(cap, raw))

def _extract_disease_labels(text: str) -> set[str]:
    """Extract normalised disease keys from free text."""
    low = _normalize(text)
    found: set[str] = set()
    for phrase, key in _DISEASE_ALIASES.items():
        if phrase in low:
            found.add(key)
    return found


def _disease_mention_counts(text: str) -> Counter:
    """Count disease mentions in diagnosis text (greedy longest-phrase, non-overlap).

    Label-set F-beta ignores multiplicity; this counter is for repeat penalty only.

    Same-key spans separated only by an abbreviation gloss are merged into one
    mention, e.g. ``Neonatal Respiratory Distress Syndrome (NRDS)`` → nrds×1.
    True list repeats (``Neonatal Pneumonia, Neonatal Pneumonia``) still count as 2.
    """
    # Keep parentheses so we can detect "full name (abbr)" glosses. Other
    # punctuation (commas, etc.) still collapses to spaces like _normalize.
    s = (text or "").lower().strip()
    s = re.sub(r"[^\w\s\(\)\[\]]", " ", s)
    low = re.sub(r"\s+", " ", s).strip()
    if not low:
        return Counter()
    # Longest phrase first so "neonatal pneumonia" wins over "pneumonia".
    phrases = sorted(_DISEASE_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True)
    occupied = [False] * len(low)
    spans: list[tuple[int, int, str]] = []
    for phrase, key in phrases:
        start = 0
        while True:
            idx = low.find(phrase, start)
            if idx < 0:
                break
            end = idx + len(phrase)
            if not any(occupied[idx:end]):
                for i in range(idx, end):
                    occupied[i] = True
                spans.append((idx, end, key))
            start = idx + 1

    spans.sort(key=lambda sp: sp[0])
    merged: list[tuple[int, int, str]] = []
    for start, end, key in spans:
        if merged and merged[-1][2] == key:
            prev_start, prev_end, _ = merged[-1]
            gap = low[prev_end:start]
            # Abbreviation gloss only: gap must include a parenthesis.
            if gap and re.fullmatch(r"[\s\(\)\[\]]*", gap) and any(ch in gap for ch in "()[]"):
                merged[-1] = (prev_start, end, key)
                continue
        merged.append((start, end, key))

    counts: Counter = Counter()
    for _, _, key in merged:
        counts[key] += 1
    return counts


def _diagnosis_repeat_penalty(pred_dis_text: str) -> tuple[float, float, int, int]:
    """Penalty for repeating the same disease key in the diagnosis field.

    Returns (penalty, repeat_rate, n_extra_mentions, n_total_mentions).
    Enabled by NEOCXR_DIAG_REPEAT_PEN=1.
    Weight NEOCXR_DIAG_REPEAT_PEN_WEIGHT (default 0.15) scales the rate into
    a score subtracted from cls_reward.
    If NEOCXR_DIAG_REPEAT_HEAD_WEIGHT>0, extra mentions of head classes cost more.
    """
    if not _env_bool("NEOCXR_DIAG_REPEAT_PEN", False):
        return 0.0, 0.0, 0, 0

    counts = _disease_mention_counts(pred_dis_text)
    total = int(sum(counts.values()))
    if total <= 1:
        return 0.0, 0.0, 0, total

    head_w = max(0.0, _env_float("NEOCXR_DIAG_REPEAT_HEAD_WEIGHT", 0.5))
    extra_mass = 0.0
    n_extra = 0
    for label, count in counts.items():
        excess = max(0, int(count) - 1)
        if excess <= 0:
            continue
        n_extra += excess
        # Base cost 1; optional headness uplift so spamming common Dx hurts more.
        cost = 1.0 + head_w * _frequency_headness(label)
        extra_mass += excess * cost

    # Normalize by total mentions so rate ∈ [0, ~1+head_w].
    denom = float(total) * (1.0 + head_w)
    repeat_rate = min(1.0, extra_mass / denom) if denom > 0 else 0.0
    weight = max(0.0, _env_float("NEOCXR_DIAG_REPEAT_PEN_WEIGHT", 0.15))
    return weight * repeat_rate, repeat_rate, n_extra, total


def _head_fp_classes() -> set[str]:
    """High-frequency diagnosis keys whose false positives we want to discourage.

    Default targets the classes the F1 policy tends to over-emit for recall hacking.
    Override with NEOCXR_HEAD_FP_CLASSES=pneumonia,nrds,normal
    """
    raw = os.environ.get("NEOCXR_HEAD_FP_CLASSES", "pneumonia,nrds,normal")
    keys = {part.strip().lower() for part in raw.split(",") if part.strip()}
    return {k for k in keys if k in _CLASS_COUNTS}


def _head_fp_penalty(pred_labels: set[str], ref_labels: set[str]) -> tuple[float, float, int]:
    """Penalise false-positive predictions of high-frequency diseases.

    This targets the hacking pattern of always emitting pneumonia/NRDS to inflate
    recall. Unlike diag-repeat (same string twice), this fires on label-set FPs.

    Returns (penalty, head_fp_rate, n_head_fps).
    Enabled by NEOCXR_HEAD_FP_PEN=1.

    When NEOCXR_HEAD_FP_ONLY_OVERREPORT=1 (default False for back-compat), the
    penalty fires only if |pred| > |ref| (cardinality over-report). Single-label
    swaps / under-prediction are left to F1 alone so correct head-class TPs are
    not discouraged by the same extra term that punishes surplus head tags.
    """
    if not _env_bool("NEOCXR_HEAD_FP_PEN", False):
        return 0.0, 0.0, 0

    if _env_bool("NEOCXR_HEAD_FP_ONLY_OVERREPORT", False):
        if len(pred_labels) <= len(ref_labels):
            return 0.0, 0.0, 0

    head = _head_fp_classes()
    if not head:
        return 0.0, 0.0, 0

    head_fps = (pred_labels - ref_labels) & head
    n_head_fps = len(head_fps)
    if n_head_fps <= 0:
        return 0.0, 0.0, 0

    # Frequency-scaled cost: pneumonia FP costs more than nrds FP.
    max_count = max(_CLASS_COUNTS[label] for label in head)
    mass = sum(_CLASS_COUNTS[label] / max_count for label in head_fps)
    # Normalize by |head| so rate is roughly in [0, 1] when all head FPs fire.
    head_fp_rate = min(1.0, mass / max(1.0, float(len(head))))
    weight = max(0.0, _env_float("NEOCXR_HEAD_FP_PEN_WEIGHT", 0.20))
    return weight * head_fp_rate, head_fp_rate, n_head_fps


def _reference_labels(ref_text: str, label_str: str | None = None) -> set[str]:
    if label_str:
        ref_labels: set[str] = set()
        for part in str(label_str).split(","):
            key = _extract_disease_labels(part.strip())
            ref_labels |= key
    else:
        ref_labels = _extract_disease_labels(ref_text)
    return ref_labels


def _cls_f_beta() -> float:
    """F-beta used by the diagnosis classification reward.

    beta < 1 favors precision (penalizes over-prediction / FP more than misses);
    beta = 1 recovers ordinary F1. Default 0.5 because symmetric F1 was observed
    to inflate recall at the expense of precision on NeoCXR.
    """
    return max(0.0, _env_float("NEOCXR_CLS_F_BETA", 0.5))


def _fbeta(precision: float, recall: float, beta: float | None = None) -> float:
    """Compute F-beta from precision/recall; beta=0 reduces to precision."""
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta = _cls_f_beta() if beta is None else max(0.0, float(beta))
    if beta == 0.0:
        return precision
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


def _f1_from_labels(pred_labels: set[str], ref_labels: set[str]) -> float:
    """Unweighted sample-level label-set F-beta (default F0.5)."""
    if not ref_labels:
        return 1.0 if not pred_labels else 0.0

    tp = len(pred_labels & ref_labels)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_labels) if pred_labels else 0.0
    recall = tp / len(ref_labels)
    return _fbeta(precision, recall)


def _weighted_f1_from_labels(pred_labels: set[str], ref_labels: set[str]) -> float:
    """Class-aware F-beta: rare-label hits carry more reward and rare FPs cost more."""
    if not ref_labels:
        return 1.0 if not pred_labels else 0.0

    tp_labels = pred_labels & ref_labels
    tp_weight = sum(_class_weight(label) for label in tp_labels)
    if tp_weight <= 0:
        return 0.0

    pred_weight = sum(_class_weight(label) for label in pred_labels) if pred_labels else 0.0
    ref_weight = sum(_class_weight(label) for label in ref_labels)
    precision = tp_weight / pred_weight if pred_weight > 0 else 0.0
    recall = tp_weight / ref_weight if ref_weight > 0 else 0.0
    return _fbeta(precision, recall)


def _avg_ref_class_weight(ref_labels: set[str]) -> float:
    if not ref_labels:
        return 1.0
    return sum(_class_weight(label) for label in ref_labels) / len(ref_labels)


def _normalized_rarity(label: str) -> float:
    max_count = max(_CLASS_COUNTS.values())
    min_count = min(_CLASS_COUNTS.values())
    count = _CLASS_COUNTS.get(label, max_count)
    return math.log(max_count / count) / math.log(max_count / min_count)


def _frequency_rarity(label: str) -> float:
    """Rarity relative to a uniform class distribution."""
    total = sum(_CLASS_COUNTS.values())
    num_classes = max(1, len(_CLASS_COUNTS))
    uniform = 1.0 / num_classes
    freq = _CLASS_COUNTS.get(label, max(_CLASS_COUNTS.values())) / total
    return max(0.0, (uniform - freq) / uniform)


def _frequency_headness(label: str) -> float:
    """Headness relative to a uniform class distribution."""
    total = sum(_CLASS_COUNTS.values())
    num_classes = max(1, len(_CLASS_COUNTS))
    uniform = 1.0 / num_classes
    freq = _CLASS_COUNTS.get(label, max(_CLASS_COUNTS.values())) / total
    return max(0.0, (freq - uniform) / (1.0 - uniform))


def _mean_centered_class_scores(label: str) -> tuple[float, float]:
    """Return minority rarity and majority headness from count-minus-mean.

    Counts above the class-count mean are majority classes and only receive a
    false-positive penalty score. Counts below the mean are minority classes and
    only receive a true-positive recall score. Both sides are normalized by the
    largest deviation from the mean, so no dataset-specific threshold is needed.
    """
    counts = list(_CLASS_COUNTS.values())
    mean_count = sum(counts) / max(1, len(counts))
    count = float(_CLASS_COUNTS.get(label, max(counts)))
    head_den = max(max(counts) - mean_count, 1.0)
    rare_den = max(mean_count - min(counts), 1.0)
    rarity = max(0.0, (mean_count - count) / rare_den)
    headness = max(0.0, (count - mean_count) / head_den)
    return min(1.0, rarity), min(1.0, headness)


def _class_count_reward_scale() -> float:
    num_classes = max(2, len(_CLASS_COUNTS))
    return _W_CLS / math.sqrt(num_classes - 1)


def _encode_label_set(labels: set[str]) -> str:
    """Stable string encoding for reward extra infos; validation skips strings."""
    return ",".join(sorted(labels))


def _mean(values: list[float], denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return sum(values) / denominator


def _cac_reward(pred_labels: set[str], ref_labels: set[str]) -> float:
    """Class-aware signed signal: rare TP up, head FP and rare FN down."""
    if not ref_labels:
        return -_mean([_frequency_headness(label) for label in pred_labels], len(pred_labels) + 1)

    true_positive = pred_labels & ref_labels
    false_positive = pred_labels - ref_labels
    false_negative = ref_labels - pred_labels

    rare_tp = _mean([_frequency_rarity(label) for label in true_positive], len(ref_labels))
    head_fp = _mean([_frequency_headness(label) for label in false_positive], len(false_positive) + 1)
    rare_fn = _mean([_frequency_rarity(label) for label in false_negative], len(false_negative) + 1)
    return max(-1.0, min(1.0, rare_tp - head_fp - rare_fn))


_CAC_SUPPRESS_FP_CLASSES: set[str] = {"pneumonia", "nrds", "normal"}
_CAC_BOOST_TP_CLASSES: set[str] = {
    "normal",
    "ttn",
    "bpd",
    "pneumothorax",
    "pleural_effusion",
    "atelectasis",
}


def _cac_clip_scores(pred_labels: set[str], ref_labels: set[str]) -> tuple[float, float]:
    """Clip-only CAC signals: suppress majority FP, encourage difficult/tail TP."""
    fp_labels = (pred_labels - ref_labels) & _CAC_SUPPRESS_FP_CLASSES
    tp_labels = (pred_labels & ref_labels) & _CAC_BOOST_TP_CLASSES

    head_fp_score = max((_frequency_headness(label) for label in fp_labels), default=0.0)
    tail_tp_score = max((_frequency_rarity(label) for label in tp_labels), default=0.0)
    if "normal" in tp_labels:
        # Normal is frequent but difficult: true positives deserve a small clip boost.
        tail_tp_score = max(tail_tp_score, 0.25)
    return head_fp_score, tail_tp_score


def _tail_recall_reward(pred_labels: set[str], ref_labels: set[str]) -> float:
    """Recall-only signal whose contribution grows continuously with class rarity."""
    rarity = {label: _normalized_rarity(label) for label in ref_labels}
    denominator = sum(rarity.values())
    if denominator <= 0:
        return 0.0
    return sum(weight for label, weight in rarity.items() if label in pred_labels) / denominator


def _tail_balanced_fbeta_reward(pred_labels: set[str], ref_labels: set[str]) -> float:
    """Smooth tail-only classification reward with recall preference and false-positive control."""
    if not ref_labels:
        return 0.0

    ref_rarities = {label: _normalized_rarity(label) for label in ref_labels}
    if max(ref_rarities.values(), default=0.0) <= 0.0:
        # Head-only prompts follow the original GDPO-KL objective exactly.
        return 0.0

    rarity_strength = max(0.0, min(1.0, _env_float("NEOCXR_TAIL_RARITY_STRENGTH", 0.5)))

    def label_weight(label: str) -> float:
        return 1.0 + rarity_strength * _normalized_rarity(label)

    true_positive = sum(label_weight(label) for label in pred_labels & ref_labels)
    if true_positive <= 0.0:
        return 0.0
    false_positive = sum(label_weight(label) for label in pred_labels - ref_labels)
    reference_mass = sum(label_weight(label) for label in ref_labels)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / reference_mass

    beta = max(1.0, min(3.0, _env_float("NEOCXR_TAIL_F_BETA", 1.5)))
    beta_sq = beta * beta
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def _cls_scores(pred_text: str, ref_text: str, label_str: str | None = None) -> dict[str, float | set[str]]:
    pred_labels = _extract_disease_labels(pred_text)
    ref_labels = _reference_labels(ref_text, label_str)
    if not ref_labels:
        precision = 1.0 if not pred_labels else 0.0
        recall = 1.0 if not pred_labels else 0.0
    else:
        tp = len(pred_labels & ref_labels)
        precision = (tp / len(pred_labels)) if pred_labels else 0.0
        recall = tp / len(ref_labels)
    cls_f1 = _f1_from_labels(pred_labels, ref_labels)
    weighted_cls_f1 = _weighted_f1_from_labels(pred_labels, ref_labels)
    ref_class_weight = _avg_ref_class_weight(ref_labels)
    alpha = max(0.0, min(1.0, _env_float("NEOCXR_CLASS_AWARE_ALPHA", 0.5)))
    class_aware_cls_f1 = (1.0 - alpha) * cls_f1 + alpha * weighted_cls_f1
    return {
        "pred_labels": pred_labels,
        "ref_labels": ref_labels,
        "cls_precision": precision,
        "cls_recall": recall,
        "cls_f_beta": _cls_f_beta(),
        "cls_f1": cls_f1,
        "weighted_cls_f1": weighted_cls_f1,
        "class_aware_cls_f1": class_aware_cls_f1,
        "ref_class_weight": ref_class_weight,
    }


# ── Format reward ─────────────────────────────────────────────────────────────

def _format_reward(pred: str) -> float:
    """1.0 if both section headers present in correct order; 0.5 if only one; 0.0 otherwise."""
    low = pred.lower()
    has_img = _IMG in low
    has_dis = _DIS in low
    if has_img and has_dis:
        return 1.0 if low.find(_IMG) < low.find(_DIS) else 0.5
    if has_img or has_dis:
        return 0.5
    return 0.0


def _template_penalty(img_text: str | None) -> float:
    """Penalise collapsed generic imaging templates."""
    if not img_text:
        return 0.0
    low = _normalize(img_text)
    hits = sum(1 for phrase in _GENERIC_IMAGING_PHRASES if phrase in low)
    if hits == 0:
        return 0.0
    return min(_TPL_PEN_MAX, _TPL_PEN_PER_HIT * hits)


def _diagnosis_gate(cls_f1: float) -> float:
    """Scale down reward when diagnosis labels are wrong or incomplete."""
    cls_f1 = max(0.0, min(1.0, cls_f1))
    return 0.10 + 0.90 * cls_f1


# ── Composite score ───────────────────────────────────────────────────────────

REWARD_VERSION = "v2.9.1-equal-rouge-cls-fbeta-tpl"
REWARD_VERSION_NO_TPL = "v2.9-equal-rouge-cls-fbeta"
REWARD_VERSION_REPEAT_PEN = "v2.9.2-equal-rouge-cls-fbeta-reppen"
REWARD_VERSION_HEAD_FP_PEN = "v2.9.3-equal-rouge-cls-fbeta-headfp"
REWARD_VERSION_HEAD_FP_OVERREPORT = "v2.9.4-equal-rouge-cls-fbeta-headfp-overreport"
CLASS_AWARE_REWARD_VERSION = "v2.9-equal-class-aware-fbeta"
RARE_FN_FP_REWARD_VERSION = "v2.9-equal-rare-fn-fp-fbeta"
OOD_REWARD_VERSION = "v3.0-ood-gated-rare-fbeta"

_W_ROUGE = 0.50  # equal weight with diagnosis F-beta (profile=eq)
_W_CLS = 0.50
_W_IMG = 0.0  # legacy field-wise keys kept at 0 for old scripts / logging
_W_DIS_TEXT = 0.0
_W_FMT = 0.0
_W_BLEU1 = 0.0

_TPL_PEN_MAX = 0.15
_TPL_PEN_PER_HIT = 0.10
_GATE_FORMULA = "disabled in v2.9 (cls_reward = w_cls * fbeta); tpl_pen subtracted when enabled"
_OOD_GATE_FORMULA = "gate=0.10+0.90*cls_fbeta; tpl_pen subtracted; rare adj added when enabled"


def _reward_profile() -> str:
    return os.environ.get("NEOCXR_REWARD_PROFILE", "eq").strip().lower()


def _use_gate() -> bool:
    profile = _reward_profile()
    if profile == "ood":
        return True
    # eq defaults off so 0.5/0.5 jobs stay unchanged. NEOCXR_USE_GATE=1
    # re-enables the diagnosis-confidence mixer inside the scalar sum.
    return _env_bool("NEOCXR_USE_GATE", False)


def _use_tpl_pen() -> bool:
    profile = _reward_profile()
    if profile == "ood":
        return True
    if profile == "eq":
        return _env_bool("NEOCXR_USE_TPL_PEN", True)
    return _env_bool("NEOCXR_USE_TPL_PEN", False)


def _reward_weights() -> tuple[float, float, float, float, float, float]:
    """Return (w_rouge, w_img, w_dis, w_cls, w_fmt, w_bleu1)."""
    profile = _reward_profile()
    if profile == "ood":
        return (
            _env_float("NEOCXR_W_ROUGE", 0.30),
            _env_float("NEOCXR_W_IMG", 0.15),
            _env_float("NEOCXR_W_DIS", 0.10),
            _env_float("NEOCXR_W_CLS", 0.40),
            _env_float("NEOCXR_W_FMT", 0.05),
            0.0,
        )
    if profile == "eq":
        # Allow per-run overrides (e.g. 0.6/0.4) without changing defaults.
        return (
            _env_float("NEOCXR_W_ROUGE", _W_ROUGE),
            _W_IMG,
            _W_DIS_TEXT,
            _env_float("NEOCXR_W_CLS", _W_CLS),
            _W_FMT,
            _W_BLEU1,
        )
    return (
        _env_float("NEOCXR_W_ROUGE", _W_ROUGE),
        _env_float("NEOCXR_W_IMG", _W_IMG),
        _env_float("NEOCXR_W_DIS", _W_DIS_TEXT),
        _env_float("NEOCXR_W_CLS", _W_CLS),
        _env_float("NEOCXR_W_FMT", _W_FMT),
        _env_float("NEOCXR_W_BLEU1", _W_BLEU1),
    )
_CLASS_AWARE_FORMULA = (
    "class_aware_cls_fbeta=(1-alpha)*cls_fbeta+alpha*weighted_cls_fbeta; "
    "cls_boost=1+boost_beta*(avg_ref_class_weight-1); "
    "fbeta uses NEOCXR_CLS_F_BETA (default 0.5)"
)


def _rare_fn_fp_components(pred_labels: set[str], ref_labels: set[str]) -> tuple[float, float, float, float, float]:
    """Tail recall reward with explicit rare-FN and head/rare-FP suppression.

    The magnitudes are intentionally small because this is an auxiliary
    clinical signal on top of the original GDPO-KL/GAC objective.
    """
    if os.environ.get("NEOCXR_RARE_REWARD_MODE", "").strip().lower() == "class_count_mean":
        reward_scale = _class_count_reward_scale()
        true_positive = pred_labels & ref_labels
        false_positive = pred_labels - ref_labels

        ref_rarity = {label: _mean_centered_class_scores(label)[0] for label in ref_labels}
        ref_tail_mass = sum(value for value in ref_rarity.values() if value > 0.0)
        if ref_tail_mass > 0.0:
            rare_recall_rate = sum(ref_rarity.get(label, 0.0) for label in true_positive) / ref_tail_mass
        else:
            rare_recall_rate = 0.0

        fp_head_mass = sum(_mean_centered_class_scores(label)[1] for label in false_positive)
        fp_rate = min(1.0, fp_head_mass)

        rare_recall_reward = reward_scale * rare_recall_rate
        rare_fn_pen = 0.0
        head_rare_fp_pen = reward_scale * fp_rate
        return rare_recall_reward, rare_fn_pen, head_rare_fp_pen, rare_recall_rate, fp_rate

    tail_threshold = max(0.0, min(1.0, _env_float("NEOCXR_TAIL_RARITY_THRESHOLD", 0.40)))
    recall_weight = max(0.0, _env_float("NEOCXR_RARE_RECALL_WEIGHT", 0.12))
    fn_pen_weight = max(0.0, _env_float("NEOCXR_RARE_FN_PEN_WEIGHT", 0.08))
    fp_pen_weight = max(0.0, _env_float("NEOCXR_HEAD_RARE_FP_PEN_WEIGHT", 0.08))

    tail_ref = {label for label in ref_labels if _normalized_rarity(label) >= tail_threshold}
    tail_mass = sum(_normalized_rarity(label) for label in tail_ref)
    if tail_mass > 0:
        tail_tp_mass = sum(_normalized_rarity(label) for label in tail_ref & pred_labels)
        tail_fn_mass = sum(_normalized_rarity(label) for label in tail_ref - pred_labels)
        rare_recall_rate = tail_tp_mass / tail_mass
        rare_fn_rate = tail_fn_mass / tail_mass
    else:
        rare_recall_rate = 0.0
        rare_fn_rate = 0.0

    fp_labels = pred_labels - ref_labels
    head_fp_mass = sum(_frequency_headness(label) for label in fp_labels)
    rare_fp_mass = sum(
        _normalized_rarity(label)
        for label in fp_labels
        if _normalized_rarity(label) >= tail_threshold
    )
    fp_rate = min(1.0, head_fp_mass + rare_fp_mass)

    rare_recall_reward = recall_weight * rare_recall_rate
    rare_fn_pen = fn_pen_weight * rare_fn_rate
    head_rare_fp_pen = fp_pen_weight * fp_rate
    return rare_recall_reward, rare_fn_pen, head_rare_fp_pen, rare_recall_rate, fp_rate


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth or "")

    sol = _strip_thinking(solution_str or "")

    p_img, p_dis = _parse_neocxr_fields(sol)
    g_img, g_dis = _parse_neocxr_fields(ground_truth)

    # Diagnostic field-wise ROUGE (logged only; not used in the score).
    if p_img is not None and g_img is not None:
        rouge_img = _rouge_l(p_img, g_img)
    else:
        rouge_img = _rouge_l(sol, ground_truth) * 0.5

    if p_dis is not None and g_dis is not None:
        rouge_dis = _rouge_l(p_dis, g_dis)
    else:
        rouge_dis = _rouge_l(sol, ground_truth) * 0.5

    rouge_l = _rouge_l(sol, ground_truth)

    disease_label = (extra_info or {}).get("disease_label", None)
    pred_dis_text = p_dis if p_dis is not None else sol
    ref_dis_text = g_dis if g_dis is not None else ground_truth
    cls_scores = _cls_scores(pred_dis_text, ref_dis_text, disease_label)
    cls_f1 = float(cls_scores["cls_f1"])
    tail_recall_reward = _tail_recall_reward(cls_scores["pred_labels"], cls_scores["ref_labels"])
    tail_balanced_fbeta_reward = _tail_balanced_fbeta_reward(
        cls_scores["pred_labels"], cls_scores["ref_labels"]
    )
    cac_reward = _cac_reward(cls_scores["pred_labels"], cls_scores["ref_labels"])
    (
        rare_recall_reward,
        rare_fn_pen,
        head_rare_fp_pen,
        rare_recall_rate,
        head_rare_fp_rate,
    ) = _rare_fn_fp_components(cls_scores["pred_labels"], cls_scores["ref_labels"])
    head_fp_clip_score, tail_tp_clip_score = _cac_clip_scores(
        cls_scores["pred_labels"], cls_scores["ref_labels"]
    )

    fmt = _format_reward(sol)
    profile = _reward_profile()
    class_aware_enabled = _env_bool(
        "NEOCXR_CLASS_AWARE_REWARD",
        profile == "ood",
    )
    rare_fn_fp_enabled = _env_bool(
        "NEOCXR_RARE_FN_FP_REWARD",
        profile == "ood",
    )
    class_aware_cls_f1 = float(cls_scores["class_aware_cls_f1"]) if class_aware_enabled else cls_f1
    ref_class_weight = float(cls_scores["ref_class_weight"])
    boost_beta = max(0.0, _env_float("NEOCXR_CLASS_BOOST_BETA", 0.35))
    cls_boost = 1.0 + boost_beta * max(0.0, ref_class_weight - 1.0) if class_aware_enabled else 1.0
    cls_boost = min(_env_float("NEOCXR_CLASS_BOOST_MAX", 2.0), cls_boost)
    gate = _diagnosis_gate(cls_f1) if _use_gate() else 1.0
    tpl_pen = _template_penalty(p_img) if _use_tpl_pen() else 0.0

    w_rouge, w_img, w_dis, w_cls, w_fmt, w_bleu1 = _reward_weights()
    rouge_l_part = w_rouge * rouge_l
    img_part = w_img * rouge_img
    dis_part = w_dis * rouge_dis
    bleu1 = 0.0
    bleu1_part = w_bleu1 * bleu1
    diag_repeat_pen, diag_repeat_rate, diag_repeat_extra, diag_mention_total = (
        _diagnosis_repeat_penalty(pred_dis_text)
    )
    head_fp_pen, head_fp_rate, n_head_fps = _head_fp_penalty(
        cls_scores["pred_labels"], cls_scores["ref_labels"]
    )
    cls_reward = max(
        0.0,
        w_cls * cls_boost * class_aware_cls_f1 - diag_repeat_pen - head_fp_pen,
    )
    gated_cls_reward = gate * cls_reward
    fmt_part = w_fmt * fmt
    gated_format_reward = gate * fmt_part
    gated_cls_fmt = gated_cls_reward + gated_format_reward
    rare_adj = 0.0
    if rare_fn_fp_enabled:
        rare_adj = rare_recall_reward - rare_fn_pen - head_rare_fp_pen
    score = max(
        0.0,
        min(
            1.0,
            rouge_l_part + img_part + dis_part + gated_cls_fmt + rare_adj - tpl_pen,
        ),
    )

    if profile == "ood":
        reward_version = OOD_REWARD_VERSION
    elif rare_fn_fp_enabled:
        reward_version = RARE_FN_FP_REWARD_VERSION
    elif class_aware_enabled:
        reward_version = CLASS_AWARE_REWARD_VERSION
    elif head_fp_pen > 0.0 or _env_bool("NEOCXR_HEAD_FP_PEN", False):
        if _env_bool("NEOCXR_HEAD_FP_ONLY_OVERREPORT", False):
            reward_version = REWARD_VERSION_HEAD_FP_OVERREPORT
        else:
            reward_version = REWARD_VERSION_HEAD_FP_PEN
    elif diag_repeat_pen > 0.0 or _env_bool("NEOCXR_DIAG_REPEAT_PEN", False):
        reward_version = REWARD_VERSION_REPEAT_PEN
    elif _use_tpl_pen():
        reward_version = REWARD_VERSION
    else:
        reward_version = REWARD_VERSION_NO_TPL

    return {
        "score": score,
        "reward_version": reward_version,
        "reward_profile": profile,
        "rouge_l": rouge_l,
        "rouge_l_part": rouge_l_part,
        "rouge_img": rouge_img,
        "rouge_dis": rouge_dis,
        "bleu1": bleu1,
        "bleu1_part": bleu1_part,
        "cls_f1": cls_f1,
        "cls_precision": float(cls_scores["cls_precision"]),
        "cls_recall": float(cls_scores["cls_recall"]),
        "cls_f_beta": float(cls_scores["cls_f_beta"]),
        "weighted_cls_f1": float(cls_scores["weighted_cls_f1"]),
        "class_aware_cls_f1": class_aware_cls_f1,
        "ref_class_weight": ref_class_weight,
        "cls_boost": cls_boost,
        "fmt": fmt,
        "gate": gate,
        "tpl_pen": tpl_pen,
        "img_part": img_part,
        "dis_part": dis_part,
        "cls_reward": cls_reward,
        "diag_repeat_pen": diag_repeat_pen,
        "diag_repeat_rate": diag_repeat_rate,
        "diag_repeat_extra": float(diag_repeat_extra),
        "diag_mention_total": float(diag_mention_total),
        "head_fp_pen": head_fp_pen,
        "head_fp_rate": head_fp_rate,
        "n_head_fps": float(n_head_fps),
        "gated_cls_reward": gated_cls_reward,
        "gated_format_reward": gated_format_reward,
        "gated_cls_fmt": gated_cls_fmt,
        "tail_recall_reward": tail_recall_reward,
        "tail_balanced_fbeta_reward": tail_balanced_fbeta_reward,
        "cac_reward": cac_reward,
        "rare_recall_reward": rare_recall_reward,
        "rare_fn_pen": rare_fn_pen,
        "head_rare_fp_pen": head_rare_fp_pen,
        "rare_recall_rate": rare_recall_rate,
        "head_rare_fp_rate": head_rare_fp_rate,
        "head_fp_clip_score": head_fp_clip_score,
        "tail_tp_clip_score": tail_tp_clip_score,
        # CAC consumes these explicit outcome labels. They intentionally do not
        # depend on a rollout advantage or any policy-gradient quantity.
        "pred_label_list": _encode_label_set(cls_scores["pred_labels"]),
        "ref_label_list": _encode_label_set(cls_scores["ref_labels"]),
        "tp_label_list": _encode_label_set(cls_scores["pred_labels"] & cls_scores["ref_labels"]),
        "fp_label_list": _encode_label_set(cls_scores["pred_labels"] - cls_scores["ref_labels"]),
        "fn_label_list": _encode_label_set(cls_scores["ref_labels"] - cls_scores["pred_labels"]),
    }
