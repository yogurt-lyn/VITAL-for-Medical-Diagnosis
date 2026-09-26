#!/usr/bin/env python3
"""MIMIC-CXR frontal chat JSON -> VERL parquet with test-distribution sampling.

Samples 4000 training examples to match the CheXpert positive-label distribution
of the test split, then splits the selected examples into 1000 SFT and 3000 RL
examples with the same target distribution.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

# Disease diagnosis field: exclude non-disease CheXpert classes
# (Support Devices = tubes/lines/hardware, not a pathology).
DISEASE_DIAGNOSIS_CONDITIONS = [c for c in CONDITIONS if c != "Support Devices"]

IMAGE_MAX_PIXELS: int | None = None


def _is_pos(value: Any) -> bool:
    try:
        return float(value) == 1.0 and not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def positive_labels(item: dict[str, Any]) -> tuple[str, ...]:
    labels = item.get("chexpert_labels") or {}
    if not isinstance(labels, dict):
        labels = {}
    out = [name for name in CONDITIONS if _is_pos(labels.get(name))]
    return tuple(out or ["No Finding"])


def disease_diagnosis_labels(item: dict[str, Any]) -> tuple[str, ...]:
    """Positive CheXpert labels suitable for Disease diagnosis (excludes Support Devices)."""
    labels = item.get("chexpert_labels") or {}
    if not isinstance(labels, dict):
        labels = {}
    out = [name for name in DISEASE_DIAGNOSIS_CONDITIONS if _is_pos(labels.get(name))]
    return tuple(out or ["No Finding"])


def format_disease_diagnosis(item: dict[str, Any]) -> str:
    """CheXpert-extractor disease labels as a Disease diagnosis field value."""
    return ", ".join(disease_diagnosis_labels(item))


def append_disease_diagnosis(findings: str, item: dict[str, Any]) -> str:
    """Append NeoCXR-compatible Disease diagnosis field to findings text.

    Example:
      <findings>
      Disease diagnosis: Edema, Fracture, Lung Opacity
    """
    findings = (findings or "").rstrip()
    diag = format_disease_diagnosis(item)
    if not findings:
        return f"Disease diagnosis: {diag}"
    # Avoid double-append if already present.
    if "disease diagnosis:" in findings.lower():
        return findings
    return f"{findings}\nDisease diagnosis: {diag}"


def label_counter(items: list[dict[str, Any]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for item in items:
        c.update(positive_labels(item))
    return c


def label_signature(item: dict[str, Any]) -> str:
    return "|".join(positive_labels(item))


def signature_counter(items: list[dict[str, Any]]) -> Counter[str]:
    return Counter(label_signature(item) for item in items)


def scaled_counts(dist: Counter[str], n: int) -> dict[str, int]:
    total = sum(dist.values())
    if total <= 0:
        raise ValueError("Cannot scale an empty distribution.")
    raw = {k: n * v / total for k, v in dist.items()}
    targets = {k: int(math.floor(v)) for k, v in raw.items()}
    remain = n - sum(targets.values())
    for k, _ in sorted(raw.items(), key=lambda kv: kv[1] - math.floor(kv[1]), reverse=True)[:remain]:
        targets[k] += 1
    return targets


def stratified_signature_sample(
    items: list[dict[str, Any]],
    n: int,
    targets: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_sig: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_sig[label_signature(item)].append(item)
    for bucket in by_sig.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    fill_items = list(items)
    rng.shuffle(fill_items)
    fill_cursor = 0

    for sig, target in sorted(targets.items(), key=lambda kv: kv[1]):
        bucket = by_sig.get(sig, [])
        take = min(target, len(bucket))
        for item in bucket[:take]:
            if id(item) in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(id(item))

    while len(selected) < n:
        while fill_cursor < len(fill_items):
            item = fill_items[fill_cursor]
            fill_cursor += 1
            if id(item) in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(id(item))
            break
        else:
            break

    if len(selected) < n:
        raise ValueError(f"Only selected {len(selected)} samples, requested {n}.")
    return selected[:n]


def find_assistant_text(item: dict[str, Any]) -> str:
    for msg in item.get("conversations") or []:
        if msg.get("from") == "gpt":
            text = (msg.get("value") or "").strip()
            if text:
                return text
    return ""


def find_user_text(item: dict[str, Any]) -> str:
    for msg in item.get("conversations") or []:
        if msg.get("from") == "human":
            text = (msg.get("value") or "").strip()
            if text:
                return text
    return "<image>\nDescribe the findings of the chest x-ray."


def _local_image_candidates(image_root: Path, rel: str) -> list[Path]:
    rel_path = Path(rel)
    candidates = [image_root / rel_path]
    parts = rel_path.parts
    if parts and parts[0] == "mimic":
        candidates.append(image_root / "files" / Path(*parts[1:]))
        candidates.append(image_root / Path(*parts[1:]))
    candidates.append(image_root / "files" / rel_path)
    return candidates


def collect_images(image_root: Path, image_field: str) -> tuple[list[dict[str, Any]], bool]:
    rels = [p.strip() for p in (image_field or "").split(",") if p.strip()][:1]
    out: list[dict[str, Any]] = []
    for rel in rels:
        p = next((candidate for candidate in _local_image_candidates(image_root, rel) if candidate.is_file()), None)
        if p is None:
            return [], False
        image_dict = {"type": "image", "image": p.resolve().as_uri()}
        if IMAGE_MAX_PIXELS is not None:
            image_dict["max_pixels"] = IMAGE_MAX_PIXELS
        out.append(image_dict)
    return out, bool(out)


def row_common(
    item: dict[str, Any],
    image_root: Path,
    index: int,
) -> tuple[dict[str, Any], str] | None:
    findings = find_assistant_text(item)
    prompt = find_user_text(item)
    if not findings:
        return None
    target = append_disease_diagnosis(findings, item)
    images, ok = collect_images(image_root, item.get("image") or "")
    if not ok:
        return None
    pos = positive_labels(item)
    diag_labels = disease_diagnosis_labels(item)
    extra_info = {
        "index": index,
        "id": item.get("id"),
        "image": item.get("image"),
        "view": item.get("view"),
        "orientation": item.get("orientation"),
        "positive_labels": ",".join(pos),
        "disease_diagnosis": ", ".join(diag_labels),
        "chexpert_labels": item.get("chexpert_labels") or {},
        "findings": findings,
    }
    return {"prompt": prompt, "target": target, "images": images, "extra_info": extra_info}, target


def build_sft_rows(
    items: list[dict[str, Any]],
    image_root: Path,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for j, item in enumerate(items):
        common = row_common(item, image_root, offset + j)
        if common is None:
            skipped += 1
            continue
        r, target = common
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": target},
                ],
                "images": r["images"],
            }
        )
    return rows, skipped


def build_rl_rows(
    items: list[dict[str, Any]],
    image_root: Path,
    offset: int,
    split: str,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for j, item in enumerate(items):
        common = row_common(item, image_root, offset + j)
        if common is None:
            skipped += 1
            continue
        r, target = common
        extra = dict(r["extra_info"])
        extra["split"] = split
        rows.append(
            {
                "data_source": "mimiccxr",
                "prompt": [{"role": "user", "content": r["prompt"]}],
                "images": r["images"],
                "reward_model": {"style": "rule", "ground_truth": target},
                "extra_info": extra,
            }
        )
    return rows, skipped


def top_up_sft_rows(
    rows: list[dict[str, Any]],
    train_raw: list[dict[str, Any]],
    selected_ids: set[int],
    image_root: Path,
    target_n: int,
    seed: int,
) -> int:
    rng = random.Random(seed)
    candidates = list(train_raw)
    rng.shuffle(candidates)
    added = 0
    for item in candidates:
        if len(rows) >= target_n:
            break
        if id(item) in selected_ids:
            continue
        common = row_common(item, image_root, 9_000_000 + added)
        if common is None:
            continue
        r, target = common
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": target},
                ],
                "images": r["images"],
            }
        )
        selected_ids.add(id(item))
        added += 1
    return added


def top_up_rl_rows(
    rows: list[dict[str, Any]],
    train_raw: list[dict[str, Any]],
    selected_ids: set[int],
    image_root: Path,
    target_n: int,
    seed: int,
) -> int:
    rng = random.Random(seed)
    candidates = list(train_raw)
    rng.shuffle(candidates)
    added = 0
    for item in candidates:
        if len(rows) >= target_n:
            break
        if id(item) in selected_ids:
            continue
        common = row_common(item, image_root, 9_500_000 + added)
        if common is None:
            continue
        r, target = common
        extra = dict(r["extra_info"])
        extra["split"] = "train_topup"
        rows.append(
            {
                "data_source": "mimiccxr",
                "prompt": [{"role": "user", "content": r["prompt"]}],
                "images": r["images"],
                "reward_model": {"style": "rule", "ground_truth": target},
                "extra_info": extra,
            }
        )
        selected_ids.add(id(item))
        added += 1
    return added


def main() -> None:
    p = argparse.ArgumentParser()
    base = Path("/path/to/MedEvalKit/utils/MIMIC-CXRfrontal/MIMICCXR")
    p.add_argument("--train_json", type=Path, default=base / "chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json")
    p.add_argument("--val_json", type=Path, default=base / "chat_dev_MIMIC_CXR_all_gpt4extract_rulebased_v1.json")
    p.add_argument("--test_json", type=Path, default=base / "chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json")
    p.add_argument(
        "--image_root",
        type=Path,
        default=Path("/path/to/mimic-cxr-jpg/2.0.0"),
        help="Local MIMIC-CXR-JPG root containing the files/ directory.",
    )
    p.add_argument("--output_dir", type=Path, default=Path("data/mimiccxr_verl_4k"))
    p.add_argument("--total_samples", type=int, default=4000)
    p.add_argument("--sft_samples", type=int, default=1000)
    p.add_argument(
        "--image_max_pixels",
        type=int,
        default=0,
        help="Optional max_pixels attached to each image dict for qwen_vl_utils smart_resize. Use 0 for Qwen default, matching NeoCXR.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    global IMAGE_MAX_PIXELS
    IMAGE_MAX_PIXELS = args.image_max_pixels if args.image_max_pixels > 0 else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_raw = json.load(open(args.train_json, encoding="utf-8"))
    val_raw = json.load(open(args.val_json, encoding="utf-8"))
    test_raw = json.load(open(args.test_json, encoding="utf-8"))

    total_signature_targets = scaled_counts(signature_counter(test_raw), args.total_samples)
    selected = stratified_signature_sample(train_raw, args.total_samples, total_signature_targets, args.seed)
    sft_signature_targets = scaled_counts(signature_counter(test_raw), args.sft_samples)
    sft_items = stratified_signature_sample(selected, args.sft_samples, sft_signature_targets, args.seed + 1)
    sft_ids = {id(item) for item in sft_items}
    rl_items = [item for item in selected if id(item) not in sft_ids]

    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("Please install pandas and pyarrow in the active environment.") from e

    sft_rows, sft_skipped = build_sft_rows(sft_items, args.image_root, 0)
    rl_rows, rl_skipped = build_rl_rows(rl_items, args.image_root, 1_000_000, "train")
    used_ids = {id(item) for item in selected}
    sft_topup = top_up_sft_rows(sft_rows, train_raw, used_ids, args.image_root, args.sft_samples, args.seed + 101)
    rl_topup = top_up_rl_rows(
        rl_rows,
        train_raw,
        used_ids,
        args.image_root,
        args.total_samples - args.sft_samples,
        args.seed + 202,
    )
    val_rows, val_skipped = build_rl_rows(val_raw, args.image_root, 2_000_000, "val")
    test_rows, test_skipped = build_rl_rows(test_raw, args.image_root, 3_000_000, "test")

    pd.DataFrame(sft_rows).to_parquet(args.output_dir / "train_sft.parquet", index=False)
    pd.DataFrame(rl_rows).to_parquet(args.output_dir / "train_grpo.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(args.output_dir / "val_grpo.parquet", index=False)
    pd.DataFrame(test_rows).to_parquet(args.output_dir / "test_grpo.parquet", index=False)

    stats = {
        "train_json": str(args.train_json),
        "val_json": str(args.val_json),
        "test_json": str(args.test_json),
        "image_root": str(args.image_root),
        "image_max_pixels": IMAGE_MAX_PIXELS,
        "seed": args.seed,
        "total_train_json": len(train_raw),
        "target_distribution_source": "test split CheXpert positive labels",
        "test_label_counts": dict(label_counter(test_raw)),
        "test_signature_counts": dict(signature_counter(test_raw)),
        "target_4k_signature_counts": total_signature_targets,
        "selected_4k_label_counts": dict(label_counter(selected)),
        "selected_4k_signature_counts": dict(signature_counter(selected)),
        "target_1k_signature_counts": sft_signature_targets,
        "sft_1k_label_counts": dict(label_counter(sft_items)),
        "sft_1k_signature_counts": dict(signature_counter(sft_items)),
        "rl_3k_label_counts": dict(label_counter(rl_items)),
        "rl_3k_signature_counts": dict(signature_counter(rl_items)),
        "sft_rows": len(sft_rows),
        "rl_train_rows": len(rl_rows),
        "sft_topup_rows": sft_topup,
        "rl_topup_rows": rl_topup,
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "sft_skipped_missing_image_or_text": sft_skipped,
        "rl_skipped_missing_image_or_text": rl_skipped,
        "val_skipped_missing_image_or_text": val_skipped,
        "test_skipped_missing_image_or_text": test_skipped,
    }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
