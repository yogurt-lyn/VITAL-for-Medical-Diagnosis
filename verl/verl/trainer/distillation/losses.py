"""Load OPD distillation losses from compiled bytecode (source was removed)."""
from __future__ import annotations

import marshal
from pathlib import Path

_here = Path(__file__).resolve()
_pyc = next(
    (_root / "_bytecode_backup" / "distillation_losses.pyc")
    for _root in _here.parents
    if (_root / "_bytecode_backup" / "distillation_losses.pyc").is_file()
)
with open(_pyc, "rb") as _f:
    _data = _f.read()
_code = marshal.loads(_data[16:])
del _here, _pyc, _f, _data, marshal, Path
exec(_code, globals())

# Extra top-k distillation modes; math lives in fsdp/losses.py.
# Reuses the forward_kl_topk unpacker registered under each new name.
try:
    register_distillation_loss(
        DistillationLossSettings(names=["amid", "amid_topk", "amid_ab"], use_topk=True)
    )(compute_forward_kl_topk)
except Exception as _amid_exc:  # pragma: no cover
    import warnings

    warnings.warn(f"Failed to register AMiD distillation loss mode: {_amid_exc}", stacklevel=1)

try:
    # CVPD (BMVC 2026): D_JS(π_+ ‖ π_s) latent-transfer term on teacher top-k.
    # Full CVPD also needs crop/ghost discovery; this is the loss-only 8B→2B baseline.
    register_distillation_loss(
        DistillationLossSettings(names=["cvpd", "cvpd_jsd", "jsd", "jsd_topk"], use_topk=True)
    )(compute_forward_kl_topk)
except Exception as _cvpd_exc:  # pragma: no cover
    import warnings

    warnings.warn(f"Failed to register CVPD/JSD distillation loss mode: {_cvpd_exc}", stacklevel=1)


def compute_topk_loss(config, distillation_config, data, student_logits, data_format):
    """Same as bytecode version, but treat ``fsdp2`` like ``fsdp`` (needed for amid/cvpd)."""
    strategy = getattr(config, "strategy", None)
    if strategy in ("fsdp", "fsdp2", "veomni"):
        from verl.trainer.distillation.fsdp import losses as fsdp_losses

        distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
    elif strategy == "megatron":
        from verl.trainer.distillation.megatron import losses as megatron_losses

        distillation_loss_fn = megatron_losses.compute_forward_kl_topk
    else:
        raise NotImplementedError(f"Unsupported strategy: config.strategy={strategy!r}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )
    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for k={k!r}."
    return outputs
