# Copyright 2025 Meituan Ltd. and/or its affiliates
"""Teacher patch–token visual similarity for GT-consistent OPD.

Loads only the 8B teacher vision tower (incl. merger) and `embed_tokens` on CPU.
This is an explicit fallback: the teacher vLLM engine (gpu_memory_utilization
~0.75) cannot host a second full 8B replica or return hidden states.
"""

from __future__ import annotations

import gc
import hashlib
import io
import logging
import os
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import ray
import torch
import torch.nn.functional as F
from PIL import Image

from verl.utils.transformers_compat import unpack_visual_output

logger = logging.getLogger(__name__)

_PATCH_CACHE_MAX = 64


def _pil_from_payload(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            return Image.fromarray(array, mode="L").convert("RGB")
        if array.shape[-1] == 4:
            return Image.fromarray(array, mode="RGBA").convert("RGB")
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"unsupported image payload type {type(image).__name__}")


def _image_cache_key(images: list[Image.Image]) -> str:
    digest = hashlib.sha1()
    for image in images:
        digest.update(str(image.size).encode("utf-8"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def _locate_visual(model):
    if hasattr(model, "visual") and model.visual is not None:
        return model.visual
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "visual"):
        return inner.visual
    raise AttributeError("Qwen3-VL teacher has no visual module")


class TeacherVisualSimExtractor:
    """Frozen teacher vision+merger+embed sidecar (CPU, eval, no grad)."""

    def __init__(self, teacher_path: str, device: str = "cpu"):
        self.teacher_path = teacher_path
        self.device = torch.device(device)
        torch.set_num_threads(int(os.environ.get("GT_VISUAL_SIM_THREADS", "4")))

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        logger.info(
            "[gt_visual_sim] loading teacher visual+embed from %s on %s "
            "(not a full 8B replica; language layers are discarded)",
            teacher_path,
            self.device,
        )
        processor = AutoProcessor.from_pretrained(teacher_path, trust_remote_code=True)
        self.image_processor = getattr(processor, "image_processor", processor)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            teacher_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        visual = _locate_visual(model)
        embed = model.get_input_embeddings()
        embed_weight = embed.weight.detach().to(device=self.device, dtype=torch.float32).contiguous()
        visual = visual.to(self.device).eval()
        for parameter in visual.parameters():
            parameter.requires_grad_(False)

        # Drop the language model so we do not keep an 8B replica next to vLLM.
        if hasattr(model, "lm_head"):
            model.lm_head = None
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "language_model"):
            inner.language_model = None
        del model, embed
        gc.collect()

        self.visual = visual
        self.embed_weight = embed_weight
        self._patch_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        logger.info(
            "[gt_visual_sim] ready: embed_weight=%s visual_device=%s",
            tuple(self.embed_weight.shape),
            self.device,
        )

    def _patches_for_images(self, images: list[Image.Image]) -> torch.Tensor:
        key = _image_cache_key(images)
        cached = self._patch_cache.get(key)
        if cached is not None:
            self._patch_cache.move_to_end(key)
            return cached

        processed = self.image_processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=self.device)
        grid_thw = processed.get("image_grid_thw")
        if grid_thw is not None:
            grid_thw = grid_thw.to(device=self.device)
        visual_dtype = next(self.visual.parameters()).dtype
        pixel_values = pixel_values.to(dtype=visual_dtype)
        with torch.inference_mode():
            visual_out = self.visual(pixel_values, grid_thw=grid_thw)
        patches, _deepstack = unpack_visual_output(visual_out)
        if patches is None:
            raise RuntimeError("teacher visual forward returned no pooler/merger output")
        patches = patches.detach().to(device="cpu", dtype=torch.float32)
        if patches.ndim == 3:
            patches = patches.reshape(-1, patches.shape[-1])
        self._patch_cache[key] = patches
        while len(self._patch_cache) > _PATCH_CACHE_MAX:
            self._patch_cache.popitem(last=False)
        return patches

    def token_max_patch_cosine(
        self, images: list[Any], token_ids: list[int]
    ) -> list[float]:
        """Per-token score = max_p cosine(normalize(patch_p), normalize(embed(token)))."""
        if not token_ids:
            return []
        pil_images = [_pil_from_payload(image) for image in images]
        if not pil_images:
            raise ValueError("gt_visual_sim requires at least one image")
        patches = self._patches_for_images(pil_images)
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        token_tensor = token_tensor.clamp(min=0, max=self.embed_weight.shape[0] - 1)
        token_emb = F.embedding(token_tensor, self.embed_weight)
        patch_n = F.normalize(patches, dim=-1)
        token_n = F.normalize(token_emb, dim=-1)
        # token_sim = normalize(patch) @ normalize(text_token_emb).T ; max over patches
        scores = torch.matmul(token_n, patch_n.transpose(0, 1)).max(dim=-1).values
        return scores.tolist()


@ray.remote(num_cpus=4, num_gpus=0, max_concurrency=8)
class TeacherVisualSimSidecar:
    """Single shared CPU sidecar so 8 agent-loop workers do not each load vision."""

    def __init__(self, teacher_path: str):
        self.extractor = TeacherVisualSimExtractor(teacher_path, device="cpu")

    def ready(self) -> str:
        return f"gt_visual_sim:{self.extractor.teacher_path}"

    def compute_scores(self, images: list[Any], token_ids: list[int]) -> list[float]:
        return self.extractor.token_max_patch_cosine(images, token_ids)


def get_or_create_sidecar(teacher_path: str, actor_name: str = "teacher_gt_visual_sim"):
    try:
        return ray.get_actor(actor_name)
    except ValueError:
        sidecar = TeacherVisualSimSidecar.options(
            name=actor_name, get_if_exists=True, lifetime=None
        ).remote(teacher_path)
        ray.get(sidecar.ready.remote())
        logger.info("[gt_visual_sim] sidecar ready: %s", teacher_path)
        return sidecar
