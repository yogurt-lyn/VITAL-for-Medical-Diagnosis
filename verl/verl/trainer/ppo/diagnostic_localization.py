"""CheXbert-guided localization for report-level diagnosis advantages.

The CheXbert F1 reward remains a report-level scalar.  This module only maps
the generated positive CheXbert labels back to medically relevant response
tokens so the existing token-level PPO branch can weight its advantage.
"""

from __future__ import annotations

import re
from typing import Any

import torch

DISEASE_TERMS: dict[str, tuple[str, ...]] = {
    "enlarged_cardiomediastinum": (
        "enlarged cardiomediastinum",
        "widened mediastinum",
        "mediastinal enlargement",
    ),
    "cardiomegaly": (
        "cardiomegaly",
        "enlarged cardiac silhouette",
        "cardiac enlargement",
    ),
    "lung_opacity": (
        "lung opacity",
        "airspace opacity",
        "pulmonary opacity",
        "infiltrate",
        "infiltration",
    ),
    "lung_lesion": (
        "lung lesion",
        "pulmonary nodule",
        "pulmonary mass",
        "lung mass",
    ),
    "edema": (
        "pulmonary edema",
        "edema",
        "vascular congestion",
        "interstitial congestion",
    ),
    "consolidation": (
        "airspace consolidation",
        "consolidation",
    ),
    "pneumonia": (
        "pneumonia",
        "infectious infiltrate",
    ),
    "atelectasis": (
        "atelectasis",
        "atelectatic",
        "volume loss",
    ),
    "pneumothorax": (
        "pneumothorax",
        "pleural air",
    ),
    "pleural_effusion": (
        "pleural effusion",
        "pleural fluid",
        "effusion",
    ),
    "pleural_other": (
        "pleural thickening",
        "pleural plaque",
    ),
    "fracture": (
        "fracture",
        "fractured",
    ),
    "support_devices": (
        "endotracheal tube",
        "enteric tube",
        "nasogastric tube",
        "chest tube",
        "central venous catheter",
        "picc line",
    ),
}

_MODIFIER_TERMS: tuple[str, ...] = (
    "no evidence of",
    "negative for",
    "cannot exclude",
    "may represent",
    "concerning for",
    "without",
    "absent",
    "possible",
    "possibly",
    "suspected",
    "mild",
    "moderate",
    "severe",
    "small",
    "large",
    "trace",
    "minimal",
    "improved",
    "worsened",
    "stable",
    "increased",
    "decreased",
    "left",
    "right",
    "bilateral",
)

_SENTENCE_RE = re.compile(r"[^.!?;\n]+[.!?;\n]?", re.MULTILINE)
_WORD_RE = re.compile(r"\S+")
_DEBUG_CALL_COUNT = 0


def _normalise_labels(labels: Any) -> set[str]:
    values: set[str] = set()
    if labels is None:
        return values
    if isinstance(labels, (str, bytes)):
        item = labels.decode() if isinstance(labels, bytes) else labels
        if item:
            values.add(str(item).strip().lower().replace(" ", "_"))
        return values
    try:
        iterator = list(labels)
    except TypeError:
        return values
    for value in iterator:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values |= _normalise_labels(value)
            continue
        text = str(value).strip().lower().replace(" ", "_")
        if text:
            values.add(text)
    return values


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    first = needle[0]
    for start in range(0, len(haystack) - len(needle) + 1):
        if haystack[start] != first:
            continue
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def _modifier_expanded_span(sentence: str, term_start: int, term_end: int) -> tuple[int, int]:
    words = list(_WORD_RE.finditer(sentence))
    if not words:
        return term_start, term_end
    start = term_start
    end = term_end
    # Expand left over nearby modifier phrases.
    for match in reversed(words):
        if match.end() > term_start:
            continue
        preceding = sentence[match.start() : term_start].strip().lower()
        if not preceding:
            start = match.start()
            continue
        matched_modifier = False
        for modifier in _MODIFIER_TERMS:
            if preceding.endswith(modifier) or preceding == modifier:
                start = match.start()
                matched_modifier = True
                break
        if not matched_modifier:
            break
    # Expand right over trailing modifiers / adjectives.
    for match in words:
        if match.start() < term_end:
            continue
        following = sentence[term_end : match.end()].strip().lower()
        matched_modifier = False
        for modifier in _MODIFIER_TERMS:
            if following.startswith(modifier) or following == modifier:
                end = match.end()
                matched_modifier = True
                break
        if not matched_modifier:
            break
    return start, end


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def spans_to_token_mask(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    mask = [0] * len(offsets)
    for index, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        for span_start, span_end in spans:
            if tok_start < span_end and tok_end > span_start:
                mask[index] = 1
                break
    return mask


def extract_diagnostic_spans_by_label(
    text: str,
    extracted_labels: list[str] | tuple[str, ...] | set[str],
    disease_terms: dict[str, tuple[str, ...]] = DISEASE_TERMS,
) -> dict[str, list[tuple[int, int]]]:
    """Locate disease mentions grouped by CheXbert label (merged per label)."""
    if not text:
        return {}
    labels = _normalise_labels(extracted_labels)
    by_label: dict[str, list[tuple[int, int]]] = {}
    for sentence_match in _SENTENCE_RE.finditer(text):
        sentence = sentence_match.group()
        for label in labels:
            for term in disease_terms.get(label, ()):
                for match in re.finditer(rf"(?<!\w){re.escape(term)}(?!\w)", sentence, flags=re.IGNORECASE):
                    start, end = _modifier_expanded_span(sentence, match.start(), match.end())
                    by_label.setdefault(label, []).append(
                        (sentence_match.start() + start, sentence_match.start() + end)
                    )
    return {label: _merge_spans(spans) for label, spans in by_label.items()}


def extract_diagnostic_spans(
    text: str,
    extracted_labels: list[str] | tuple[str, ...] | set[str],
    disease_terms: dict[str, tuple[str, ...]] = DISEASE_TERMS,
) -> list[tuple[int, int]]:
    """Locate generated CheXbert-positive disease mentions with modifiers.

    Each span is contained in one sentence (period, semicolon, or newline
    delimited). Labels are only used to select terms; no per-disease reward is
    calculated.
    """
    by_label = extract_diagnostic_spans_by_label(text, extracted_labels, disease_terms=disease_terms)
    merged: list[tuple[int, int]] = []
    for spans in by_label.values():
        merged.extend(spans)
    return _merge_spans(merged)


def build_localized_diag_advantage(
    report_advantage: torch.Tensor,
    key_token_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    background_weight: float = 0.05,
) -> torch.Tensor:
    """Broadcast a report-level advantage onto diagnostic tokens."""
    if report_advantage.ndim < completion_mask.ndim:
        while report_advantage.ndim < completion_mask.ndim:
            report_advantage = report_advantage.unsqueeze(-1)
        report_advantage = report_advantage.expand_as(completion_mask)
    weights = torch.where(
        key_token_mask.bool(),
        torch.ones_like(report_advantage),
        torch.full_like(report_advantage, float(background_weight)),
    )
    weights = weights * completion_mask
    return report_advantage * weights


def build_diagnostic_token_mask(
    response_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    tokenizer: Any,
    predicted_labels: Any,
    *,
    debug_interval: int = 0,
) -> torch.Tensor:
    """Build a [batch, response_length] CheXbert-keyword mask on the CPU-side driver."""
    mask, _ = build_diagnostic_token_mask_and_rarity(
        response_ids,
        completion_mask,
        tokenizer,
        predicted_labels,
        rarity_by_label=None,
        debug_interval=debug_interval,
    )
    return mask


def build_diagnostic_token_mask_and_rarity(
    response_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    tokenizer: Any,
    predicted_labels: Any,
    *,
    rarity_by_label: dict[str, float] | None,
    debug_interval: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build diagnostic mask and optional per-token max rarity over covering labels.

    When a token is covered by multiple disease spans, rarity is the *max* of
    those labels so rare-disease mentions keep exploration (are not treated as
    head-only tokens).
    """
    global _DEBUG_CALL_COUNT
    _DEBUG_CALL_COUNT += 1
    device = response_ids.device
    mask = torch.zeros_like(completion_mask, dtype=torch.bool, device=device)
    rarity = torch.zeros_like(completion_mask, dtype=torch.float32, device=device)
    should_debug = debug_interval > 0 and _DEBUG_CALL_COUNT % debug_interval == 0
    rarity_map = rarity_by_label or {}

    for row in range(response_ids.shape[0]):
        valid_length = int(completion_mask[row].sum().item())
        if valid_length <= 0:
            continue
        ids = response_ids[row, :valid_length].detach().cpu().tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        labels = predicted_labels[row] if predicted_labels is not None and row < len(predicted_labels) else ()
        by_label = extract_diagnostic_spans_by_label(text, labels)
        if not by_label:
            continue
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            encoded_ids = list(encoded["input_ids"])
            offsets = list(encoded["offset_mapping"])
        except (TypeError, KeyError, AttributeError):
            continue
        response_start = _find_subsequence(ids, encoded_ids)
        if response_start < 0:
            continue
        for label, spans in by_label.items():
            label_rarity = float(rarity_map.get(label, 0.0))
            local_mask = spans_to_token_mask(offsets, spans)
            for index, matched in enumerate(local_mask):
                response_index = response_start + index
                if response_index >= valid_length:
                    break
                if not matched:
                    continue
                mask[row, response_index] = True
                if rarity_map:
                    rarity[row, response_index] = max(float(rarity[row, response_index].item()), label_rarity)
        if should_debug and row == 0:
            spans = extract_diagnostic_spans(text, labels)
            localized = [text[start:end] for start, end in spans]
            token_strings = tokenizer.convert_ids_to_tokens(ids)
            selected = [token for idx, token in enumerate(token_strings) if mask[row, idx]]
            print(
                "[diag-localization] Generated report:\n"
                f"{text}\nLocalized diagnostic spans: {localized}\n"
                f"Localized tokens: {' | '.join(selected)}",
                flush=True,
            )
    return mask & completion_mask.bool(), rarity


def extract_neocxr_diagnosis_field_span(text: str) -> list[tuple[int, int]]:
    """Character span covering NeoCXR Disease-diagnosis field content (not the header)."""
    if not text:
        return []
    low = text.lower()
    marker = "disease diagnosis:"
    start = low.find(marker)
    if start < 0:
        return []
    content_start = start + len(marker)
    while content_start < len(text) and text[content_start].isspace():
        content_start += 1
    if content_start >= len(text):
        return []
    return [(content_start, len(text))]


def build_neocxr_diagnosis_field_token_mask(
    response_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    tokenizer: Any,
    *,
    debug_interval: int = 0,
) -> torch.Tensor:
    """Mask response tokens that fall inside the Disease diagnosis field."""
    global _DEBUG_CALL_COUNT
    _DEBUG_CALL_COUNT += 1
    device = response_ids.device
    mask = torch.zeros_like(completion_mask, dtype=torch.bool, device=device)
    should_debug = debug_interval > 0 and _DEBUG_CALL_COUNT % debug_interval == 0

    for row in range(response_ids.shape[0]):
        valid_length = int(completion_mask[row].sum().item())
        if valid_length <= 0:
            continue
        ids = response_ids[row, :valid_length].detach().cpu().tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        spans = extract_neocxr_diagnosis_field_span(text)
        if not spans:
            continue
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            encoded_ids = list(encoded["input_ids"])
            offsets = list(encoded["offset_mapping"])
        except (TypeError, KeyError, AttributeError):
            continue
        response_start = _find_subsequence(ids, encoded_ids)
        if response_start < 0:
            continue
        local_mask = spans_to_token_mask(offsets, spans)
        for index, matched in enumerate(local_mask):
            response_index = response_start + index
            if response_index >= valid_length:
                break
            if matched:
                mask[row, response_index] = True
        if should_debug and row == 0:
            localized = [text[start:end] for start, end in spans]
            token_strings = tokenizer.convert_ids_to_tokens(ids)
            selected = [token for idx, token in enumerate(token_strings) if mask[row, idx]]
            print(
                "[neocxr-diag-field] Generated report:\n"
                f"{text}\nLocalized diagnosis field: {localized}\n"
                f"Localized tokens: {' | '.join(selected)}",
                flush=True,
            )
    return mask & completion_mask.bool()
