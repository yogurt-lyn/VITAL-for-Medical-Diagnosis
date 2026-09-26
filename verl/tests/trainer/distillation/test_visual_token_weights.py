from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_PATH = (
    Path(__file__).resolve().parents[3]
    / "verl"
    / "trainer"
    / "distillation"
    / "visual_token_weights.py"
)
_SPEC = importlib.util.spec_from_file_location("visual_token_weights", _PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
compute_teacher_visual_token_weights = _MOD.compute_teacher_visual_token_weights


def test_sigmoid_gate_decays_low_sensitivity_without_zeroing():
    sensitivity = torch.tensor([[0.0, 0.1, 1.0, 2.0]], dtype=torch.float32)
    mask = torch.ones_like(sensitivity)
    weights, raw = compute_teacher_visual_token_weights(
        sensitivity, mask, gate="sigmoid", sigmoid_tau=0.5, sigmoid_bias=0.0, normalize=True
    )
    assert float(raw[0, 0]) > 0.0
    assert float(raw[0, 0]) < float(raw[0, 3])
    assert float(weights[0, 0]) < float(weights[0, 3])
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)


def test_sigmoid_does_not_use_hard_topk():
    sensitivity = torch.linspace(0.0, 1.0, 10).unsqueeze(0)
    mask = torch.ones_like(sensitivity)
    weights, _ = compute_teacher_visual_token_weights(
        sensitivity, mask, gate="sigmoid", sigmoid_tau=0.5, normalize=True
    )
    assert int((weights[0] > 0).sum()) == 10
