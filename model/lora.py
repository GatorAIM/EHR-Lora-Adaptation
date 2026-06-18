"""
Low-rank adaptation (LoRA) helpers for the EHR transformer.

LoRA wraps a frozen base `nn.Linear` as
    y = base(x) + (alpha / r) * B(A(x))
where A is rank x in_features (Kaiming init) and B is out_features x rank
(zero init, so the wrapper is identity at step 0). Only A and B are
trainable; the base weight stays frozen.

Inside this codebase LoRA is applied only to selected suffix-named
submodules (self_attn.W_Q/K/V/W_output, pos_ffn.fc1/fc2) inside the last
`n` of the transformer block stack — all earlier blocks stay frozen.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank residual."""

    def __init__(self, base: nn.Linear, *, r: int, alpha: int, dropout: float):
        super().__init__()
        if r <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not (0.0 <= float(dropout) <= 1.0):
            raise ValueError("dropout must be in [0, 1]")
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(float(dropout))

        in_f, out_f = int(base.in_features), int(base.out_features)
        self.lora_A = nn.Linear(in_f, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Mirror device / dtype of the wrapped Linear.
        ref = self.base.weight
        self.lora_A.to(device=ref.device, dtype=ref.dtype)
        self.lora_B.to(device=ref.device, dtype=ref.dtype)

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, r: int, alpha: int, dropout: float) -> "LoRALinear":
        return cls(linear, r=r, alpha=alpha, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _set_module_attr(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Re-bind `root.<dotted_name>` to `new_module` by walking the path."""
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inject_lora(
    model: nn.Module,
    *,
    target_suffixes: Sequence[str],
    layer_idxs: Optional[Sequence[int]] = None,
    r: int,
    alpha: int,
    dropout: float,
) -> List[str]:
    """
    Replace targeted nn.Linear modules with LoRALinear wrappers in place.

    target_suffixes are matched against the fully-qualified module name
    (e.g. "self_attn.W_Q", "pos_ffn.fc1"). If `layer_idxs` is given,
    only blocks whose name contains "transformer_blocks.{idx}." are
    eligible. Returns the list of replaced module names.
    """
    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if layer_idxs is not None and not any(
            f"transformer_blocks.{int(i)}." in name for i in layer_idxs
        ):
            continue
        if not any(name.endswith(sfx) for sfx in target_suffixes):
            continue
        wrapper = LoRALinear.from_linear(module, r=r, alpha=alpha, dropout=dropout)
        _set_module_attr(model, name, wrapper)
        replaced.append(name)
    return replaced


def freeze_base_params(model: nn.Module) -> None:
    """Set requires_grad=False on every parameter currently in the model."""
    for p in model.parameters():
        p.requires_grad = False


def mark_lora_trainable(model: nn.Module) -> None:
    """Re-enable requires_grad on LoRA A / B matrices only."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_A.parameters():
                p.requires_grad = True
            for p in m.lora_B.parameters():
                p.requires_grad = True


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA A / B tensors, namespaced by the wrapper path."""
    out: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            out[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            out[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return out


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Copy A / B tensors into matching LoRA wrappers; ignore other keys."""
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        a_key, b_key = f"{name}.lora_A.weight", f"{name}.lora_B.weight"
        if a_key in state:
            module.lora_A.weight.data.copy_(state[a_key].to(module.lora_A.weight.device))
        if b_key in state:
            module.lora_B.weight.data.copy_(state[b_key].to(module.lora_B.weight.device))
