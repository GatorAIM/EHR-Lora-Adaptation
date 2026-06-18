# EHR-Lora-Adaptation

Reference implementation outlines for the paper
**"Parameter-Efficient Adaptation of Pretrained EHR Transformer Models across Clinical Prediction Tasks and Health Systems."**

This repository contains *idea-level* sketches of the components used in the
paper. Each module describes what the function does and how it is wired
into the overall pipeline, without committing to a particular runtime
configuration. The intent is to expose the design — not a turn-key script.

## Layout

```
./model      transformer backbone + LoRA adapter sketches
./data       raw EHR ingestion, tokenization, train/test split
./train      MLM pretrain and downstream finetune routines
./eval       cross-site evaluation and calibration metrics
./analysis   aggregation and figure-ready tables
```

## Pipeline at a glance

1. **Pretrain** an EHR transformer with a masked-LM objective on a
   combined multi-site corpus (`./data`, `./train`).
2. **Adapt** the pretrained backbone for each downstream task using one
   of: Freeze-all-but-head, Tune-last-N, full finetune, or LoRA with a
   sweep over `(rank, alpha, last_n_layers, target_modules)` (`./train`).
3. **Evaluate** in two regimes (`./eval`):
   - *Internal:* same-site test split, discrimination metrics
     (AUROC / AUPRC / F1 / Accuracy).
   - *External / cross-site:* take the source-trained checkpoint and
     score it on a target-site test split without any target-site
     adaptation.
4. **Calibrate** every checkpoint by computing Brier score and Expected
   Calibration Error on the appropriate test set (`./eval/metrics.py`).
5. **Aggregate** per-run metrics into figure-ready tables and compute
   paired contrasts that surface negative transfer and calibration drift
   (`./analysis`).

## Data assumption

The code expects PCORnet-style site tables (DX, PX, LAB, MED admin,
vitals, cohort labels). Raw data paths are not committed; modules take
them as parameters.

