"""
Cross-site (external) evaluation.

A model that was finetuned on a *source* site (or on the combined "ALL"
cohort) is scored on a *target* site's test split without any further
training on the target side. This is the regime that surfaces negative
transfer and calibration drift.

Non-LoRA checkpoints (freeze-all / tune-last-N / full finetune) only
need the architecture and state_dict; LoRA checkpoints additionally
need the adapter configuration to rebuild the same LoRA shell before
loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from model.ehr_backbone import FinetuneModel
from model.lora import inject_lora
from eval.metrics import evaluate_all, write_metrics_sidecar


def load_adapter_config(sidecar_path: str) -> Dict:
    """
    Read the source-side `.metrics.json` and return the adapter block.
    The block describes either a LoRA configuration (targets / rank /
    alpha / last_n_layers / dropout) or a simple strategy name for
    non-LoRA baselines.
    """
    return json.loads(Path(sidecar_path).read_text(encoding="utf-8")).get("adapter", {})


def rebuild_model_for_inference(
    config,
    block_cls,
    base_state,
    *,
    adapter_cfg: Dict,
    device,
) -> FinetuneModel:
    """
    Allocate `FinetuneModel`, partial-load the pretrained backbone, and
    — for LoRA runs only — re-inject the same LoRA shell described in
    the source-side adapter block.
    """
    model = FinetuneModel(config, block_cls)
    model.load_weight(base_state)
    model = model.to(device)

    if adapter_cfg.get("kind", "").lower() == "lora":
        n_blocks = len(model.transformer.transformer_blocks)
        last_n = int(adapter_cfg.get("last_n_layers", n_blocks))
        layer_idxs = list(range(n_blocks - last_n, n_blocks))
        replaced = inject_lora(
            model,
            target_suffixes=list(adapter_cfg["targets"]),
            layer_idxs=layer_idxs,
            r=int(adapter_cfg["r"]),
            alpha=int(adapter_cfg["alpha"]),
            dropout=float(adapter_cfg.get("dropout", 0.0)),
        )
        if not replaced:
            raise RuntimeError(
                "LoRA injection produced zero matches; adapter config likely "
                "does not match the architecture."
            )
    return model


def load_source_checkpoint(model, source_ckpt: str) -> None:
    """Strict state_dict load so any architecture mismatch errors loudly."""
    state = torch.load(source_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)


def external_evaluate(
    *,
    config,
    block_cls,
    base_state,
    source_ckpt: str,
    source_sidecar: str,
    target_loader,
    out_marker: str,
    device,
    ece_bins: int,
    extras: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    End-to-end cross-site evaluation for one (source, target) pair.

    Steps:
      1. Read the adapter config from the source sidecar.
      2. Rebuild the architecture and inject LoRA if applicable.
      3. Strict-load the source checkpoint state.
      4. Run `evaluate_all` on the target test loader.
      5. Write a target-side sidecar at `out_marker.metrics.json` so
         downstream aggregators recognise it as an external evaluation.
    """
    adapter_cfg = load_adapter_config(source_sidecar)
    model = rebuild_model_for_inference(
        config, block_cls, base_state, adapter_cfg=adapter_cfg, device=device
    )
    load_source_checkpoint(model, source_ckpt)
    metrics = evaluate_all(model, target_loader, device, ece_bins=int(ece_bins))
    payload_extras = {"source_checkpoint": str(source_ckpt),
                      "source_sidecar": str(source_sidecar)}
    if extras:
        payload_extras.update(extras)
    write_metrics_sidecar(
        out_marker,
        metrics,
        adapter_config=adapter_cfg,
        extras=payload_extras,
    )
    return metrics
