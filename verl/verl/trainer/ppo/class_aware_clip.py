import json
import math
import re
from pathlib import Path
from typing import Any

import torch


_LABEL_ALIASES = {
    "no obvious abnormalities": "normal",
    "normal": "normal",
    "neonatal pneumonia": "pneumonia",
    "pneumonia": "pneumonia",
    "neonatal respiratory distress syndrome (nrds)": "nrds",
    "nrds": "nrds",
    "neonatal transient tachypnea (ttn)": "ttn",
    "ttn": "ttn",
    "bronchopulmonary dysplasia (bpd)": "bpd",
    "bpd": "bpd",
    "pneumothorax": "pneumothorax",
    "pleural effusion": "pleural_effusion",
    "atelectasis": "atelectasis",
}


def class_aware_enabled(config: Any) -> bool:
    """Honor the new CAC switch while keeping older experiment configs valid."""
    enabled = config.get("enable_class_aware_clip", None)
    return bool(config.get("class_aware_clip_enabled", False) if enabled is None else enabled)


def _frequency_payload(config: Any) -> dict[str, Any]:
    frequency_file = str(config.get("class_frequency_file", "") or "")
    if frequency_file:
        payload = json.loads(Path(frequency_file).read_text(encoding="utf-8"))
        if "counts" not in payload:
            raise ValueError("class_frequency_file must contain a 'counts' object")
        return payload
    frequencies = json.loads(config.get("class_aware_frequency_json", "{}"))
    return {"counts": frequencies}


def class_frequency_records(config: Any) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Load counts and log-minmax rarity scores from the configurable frequency source."""
    payload = _frequency_payload(config)
    counts = {
        _LABEL_ALIASES.get(str(label).strip().lower(), str(label).strip().lower()): float(count)
        for label, count in payload["counts"].items()
    }
    if not counts or min(counts.values()) <= 0:
        raise ValueError("class frequencies must be positive and non-empty")
    labels = list(counts)
    provided_rarity = payload.get("rarity", {})
    if provided_rarity:
        rarity = {
            label: float(provided_rarity.get(label, provided_rarity.get(str(label), 0.0)))
            for label in labels
        }
    else:
        max_count, min_count = max(counts.values()), min(counts.values())
        denominator = math.log(max_count / min_count) if max_count > min_count else 1.0
        rarity = {label: math.log(max_count / count) / denominator for label, count in counts.items()}
    return labels, counts, {label: max(0.0, min(1.0, value)) for label, value in rarity.items()}


def class_aware_stat_field_names(config: Any) -> list[str]:
    labels, _, _ = class_frequency_records(config)
    return [f"class_event_counts_{label}" for label in labels]


def _label_list(values: Any, index: int) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        return set(_canonical_labels(values if index == 0 else ""))
    if index >= len(values):
        return set()
    return set(_canonical_labels(values[index]))


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def build_rarity_tp_token_tensors(
    extra_infos: Any,
    response_mask: torch.Tensor,
    response_ids: torch.Tensor,
    tokenizer: Any,
    config: Any,
    reward_extra_infos: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Build CAC tensors for disease-name tokens only.

    A token is enhanced only when it belongs to a generated diagnosis label that
    is a ground-truth positive (TP). FP/FN are recorded independently for
    diagnostics and retain the base clip. This deliberately avoids using the
    sign of the advantage as a correctness proxy.
    """
    labels, _, rarity_by_label = class_frequency_records(config)
    label_to_id = {label: index for index, label in enumerate(labels)}
    shape = response_mask.shape
    device = response_mask.device
    base_epsilon = float(config.get("base_clip_epsilon", config.get("class_aware_epsilon_base", 4e-4)))
    max_epsilon = float(config.get("max_clip_epsilon", config.get("class_aware_epsilon_max", base_epsilon)))
    clip_lambda = float(config.get("rarity_clip_lambda", config.get("class_aware_alpha", 1.0)))
    advantage_alpha = float(config.get("class_aware_alpha", 1.0))

    rarity = torch.zeros(shape, dtype=torch.float32, device=device)
    weight = torch.ones(shape, dtype=torch.float32, device=device)
    positive_clip = torch.full(shape, base_epsilon, dtype=torch.float32, device=device)
    tier = torch.zeros(shape, dtype=torch.float32, device=device)
    category_id = torch.full(shape, -1, dtype=torch.long, device=device)
    tp_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    fp_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    # Per sample and class: [TP, FN, FP]. Counts are placed at the first valid
    # response token so they survive the standard padded-tensor worker path.
    event_counts = torch.zeros((shape[0], len(labels), 3), dtype=torch.float32, device=device)

    pred_values = reward_extra_infos.get("pred_label_list")
    ref_values = reward_extra_infos.get("ref_label_list")
    tp_values = reward_extra_infos.get("tp_label_list")
    fp_values = reward_extra_infos.get("fp_label_list")
    fn_values = reward_extra_infos.get("fn_label_list")

    alias_items = sorted(_LABEL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)
    for row in range(shape[0]):
        valid_length = int(response_mask[row].sum().item())
        if valid_length <= 0:
            continue
        pred_labels = _label_list(pred_values, row)
        ref_labels = _label_list(ref_values, row)
        tp_labels = _label_list(tp_values, row) or (pred_labels & ref_labels)
        fp_labels = _label_list(fp_values, row) or (pred_labels - ref_labels)
        fn_labels = _label_list(fn_values, row) or (ref_labels - pred_labels)
        for label, stat_index in ((tp_labels, 0), (fn_labels, 1), (fp_labels, 2)):
            for disease in label:
                if disease in label_to_id:
                    event_counts[row, label_to_id[disease], stat_index] += 1

        ids = response_ids[row, :valid_length].detach().cpu().tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        diagnosis_start = text.lower().find("disease diagnosis:")
        if diagnosis_start < 0:
            continue
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            encoded_ids = list(encoded["input_ids"])
            offsets = list(encoded["offset_mapping"])
        except (TypeError, KeyError, AttributeError):
            continue
        response_start = _find_subsequence(ids, encoded_ids)
        if response_start < 0:
            # Refuse an uncertain token alignment rather than expanding clip on
            # arbitrary tokens. Qwen tokenizers use the exact-match path.
            continue
        spans: list[tuple[int, int, str]] = []
        diagnosis_text = text[diagnosis_start + len("disease diagnosis:") :]
        offset_base = diagnosis_start + len("disease diagnosis:")
        for phrase, disease in alias_items:
            if disease not in label_to_id:
                continue
            for match in re.finditer(rf"(?<!\\w){re.escape(phrase)}(?!\\w)", diagnosis_text, flags=re.IGNORECASE):
                spans.append((offset_base + match.start(), offset_base + match.end(), disease))
        for token_index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            response_index = response_start + token_index
            if response_index >= valid_length:
                break
            matched = [disease for span_start, span_end, disease in spans if start < span_end and end > span_start]
            if not matched:
                continue
            disease = max(matched, key=lambda item: rarity_by_label[item])
            disease_id = label_to_id[disease]
            q = rarity_by_label[disease]
            category_id[row, response_index] = disease_id
            rarity[row, response_index] = q
            tier[row, response_index] = 0 if q < 1 / 3 else (1 if q < 2 / 3 else 2)
            if disease in tp_labels:
                tp_mask[row, response_index] = True
                weight[row, response_index] = 1.0 + advantage_alpha * q
                positive_clip[row, response_index] = min(max_epsilon, max(base_epsilon, base_epsilon * (1.0 + clip_lambda * q)))
            elif disease in fp_labels:
                fp_mask[row, response_index] = True

    return {
        "class_rarity": rarity,
        "class_weight": weight,
        "class_positive_clip": positive_clip,
        "class_tier": tier,
        "class_category_id": category_id,
        "class_tp_mask": tp_mask,
        "class_fp_mask": fp_mask,
        "class_event_counts": event_counts,
    }


def _canonical_labels(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        parts = [str(item) for item in value]
    else:
        parts = re.split(r"[,;|\n]+", str(value or ""))
    return [_LABEL_ALIASES.get(part.strip().lower(), part.strip().lower()) for part in parts if part.strip()]


def class_aware_parameters(
    rarity: torch.Tensor,
    *,
    alpha: float,
    epsilon_base: float,
    epsilon_max: float,
    gamma: float,
    saturation_tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map normalized rarity to saturation, advantage weight, and upper clip."""
    rarity = rarity.clamp(0.0, 1.0)
    if saturation_tau > 0:
        denominator = 1.0 - math.exp(-saturation_tau)
        z = (1.0 - torch.exp(-saturation_tau * rarity)) / denominator
    else:
        z = rarity
    weight = 1.0 + alpha * z
    positive_clip = epsilon_base + (epsilon_max - epsilon_base) * z.pow(gamma)
    return z, weight, positive_clip


def mid_rarity_peak_parameters(
    rarity: torch.Tensor,
    *,
    alpha: float,
    epsilon_base: float,
    epsilon_max: float,
    mu: float,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map class rarity to a mid-rarity peak for upper clip expansion."""
    rarity = rarity.clamp(0.0, 1.0)
    sigma = max(float(sigma), 1e-6)
    z = torch.exp(-((rarity - float(mu)) ** 2) / (2.0 * sigma * sigma))
    z = z.clamp(0.0, 1.0)
    weight = torch.ones_like(rarity)
    positive_clip = (epsilon_base * (1.0 + float(alpha) * z)).clamp(epsilon_base, epsilon_max)
    return z, weight, positive_clip


def _class_count_delta(num_classes: int) -> float:
    """Derive the clip expansion range from the label-space size.

    A larger label space gives a smaller per-class clip expansion. For NeoCXR
    (K=8), this yields 1/sqrt(7)=0.378, so epsilon_0=0.20 maps to
    approximately [0.124, 0.276], close to the DAPO-style [0.20, 0.28] upper
    range without hard-coding dataset-specific bounds.
    """
    return 1.0 / math.sqrt(max(float(num_classes - 1), 1.0))


def build_class_aware_tensors(
    extra_infos: Any,
    response_mask: torch.Tensor,
    config: Any,
    reward_extra_infos: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-token rarity/weight/clip/tier tensors from sample labels."""
    frequency_map = json.loads(config.get("class_aware_frequency_json", "{}"))
    frequencies = {_LABEL_ALIASES.get(k.strip().lower(), k.strip().lower()): float(v) for k, v in frequency_map.items()}
    if not frequencies:
        raise ValueError("class_aware_frequency_json must contain training-set class frequencies")

    max_count = max(frequencies.values())
    min_count = min(frequencies.values())
    log_range = math.log(max_count / min_count) if max_count > min_count else 1.0
    aggregation = config.get("class_aware_multi_label_aggregation", "max")
    sample_rarities = []
    for info in extra_infos:
        labels = _canonical_labels((info or {}).get("disease_label"))
        rarities = [math.log(max_count / frequencies[label]) / log_range for label in labels if label in frequencies]
        if not rarities:
            rarity = 0.0
        elif aggregation == "mean":
            rarity = sum(rarities) / len(rarities)
        elif aggregation == "max":
            rarity = max(rarities)
        else:
            raise ValueError(f"Unsupported class-aware multi-label aggregation: {aggregation}")
        sample_rarities.append(rarity)

    alpha = float(config.get("class_aware_alpha", 1.0))
    epsilon_base = float(config.get("class_aware_epsilon_base", 3e-4))
    epsilon_max = float(config.get("class_aware_epsilon_max", epsilon_base * (1.0 + alpha)))
    epsilon_min = float(config.get("class_aware_epsilon_min", epsilon_base * max(0.0, 1.0 - alpha)))
    bound_mode = str(config.get("class_aware_bound_mode", "") or "")
    class_count_delta = _class_count_delta(len(frequencies))
    if bound_mode == "class_count":
        epsilon_min = epsilon_base * (1.0 - class_count_delta)
        epsilon_max = epsilon_base * (1.0 + class_count_delta)
    mode = str(config.get("class_aware_clip_mode", "") or "")
    signal_key = str(config.get("class_aware_signal_key", "") or "")
    rarity = torch.tensor(sample_rarities, dtype=torch.float32, device=response_mask.device)
    if mode == "head_fp_tail_tp" and reward_extra_infos is not None:
        head_values = reward_extra_infos.get("head_fp_clip_score", [0.0] * len(sample_rarities))
        tail_values = reward_extra_infos.get("tail_tp_clip_score", [0.0] * len(sample_rarities))
        head_fp = torch.tensor([float(value) for value in head_values], dtype=torch.float32, device=response_mask.device)
        tail_tp = torch.tensor([float(value) for value in tail_values], dtype=torch.float32, device=response_mask.device)
        head_fp = head_fp.clamp(0.0, 1.0)
        tail_tp = tail_tp.clamp(0.0, 1.0)
        z = tail_tp
        weight = torch.ones_like(rarity)
        scale_delta = class_count_delta if bound_mode == "class_count" else alpha
        positive_clip = (epsilon_base * (1.0 - scale_delta * head_fp + scale_delta * tail_tp)).clamp(
            epsilon_min,
            epsilon_max,
        )
    elif mode == "rare_recall_headfp" and reward_extra_infos is not None:
        recall_values = reward_extra_infos.get("rare_recall_rate", [0.0] * len(sample_rarities))
        fp_values = reward_extra_infos.get("head_rare_fp_rate", [0.0] * len(sample_rarities))
        rare_recall = torch.tensor([float(value) for value in recall_values], dtype=torch.float32, device=response_mask.device)
        head_fp = torch.tensor([float(value) for value in fp_values], dtype=torch.float32, device=response_mask.device)
        rare_recall = rare_recall.clamp(0.0, 1.0)
        head_fp = head_fp.clamp(0.0, 1.0)
        signal = rare_recall - head_fp
        z = rare_recall
        weight = torch.ones_like(rarity)
        scale_delta = class_count_delta if bound_mode == "class_count" else alpha
        positive_clip = (epsilon_base * (1.0 + scale_delta * signal)).clamp(
            epsilon_min,
            epsilon_max,
        )
    elif signal_key and reward_extra_infos is not None and signal_key in reward_extra_infos:
        signal = torch.tensor(
            [float(value) for value in reward_extra_infos[signal_key]],
            dtype=torch.float32,
            device=response_mask.device,
        ).clamp(-1.0, 1.0)
        scale = (1.0 + alpha * signal).clamp(1.0 - alpha, 1.0 + alpha)
        z = ((signal + 1.0) * 0.5).clamp(0.0, 1.0)
        weight = scale
        positive_clip = (epsilon_base * scale).clamp(epsilon_base * (1.0 - alpha), epsilon_max)
    else:
        if bound_mode == "mid_rarity_peak":
            z, weight, positive_clip = mid_rarity_peak_parameters(
                rarity,
                alpha=alpha,
                epsilon_base=epsilon_base,
                epsilon_max=epsilon_max,
                mu=float(config.get("class_aware_mid_mu", 0.55)),
                sigma=float(config.get("class_aware_mid_sigma", 0.25)),
            )
        else:
            z, weight, positive_clip = class_aware_parameters(
                rarity,
                alpha=alpha,
                epsilon_base=epsilon_base,
                epsilon_max=epsilon_max,
                gamma=float(config.get("class_aware_gamma", 1.0)),
                saturation_tau=float(config.get("class_aware_saturation_tau", 2.0)),
            )
    # 0=head, 1=middle, 2=tail. Tiers are used only for diagnostics.
    tier = torch.where(z < 1 / 3, 0, torch.where(z < 2 / 3, 1, 2)).to(torch.float32)

    def expand(values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(-1).expand(response_mask.shape)

    return expand(z), expand(weight), expand(positive_clip), expand(tier)
