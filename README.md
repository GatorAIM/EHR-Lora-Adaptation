# EHR LoRA Adaptation

Reference implementation for **Parameter-Efficient Adaptation of Pretrained EHR
Transformer Models across Clinical Prediction Tasks and Health Systems**.

This repository documents the model components, preprocessing contracts,
adaptation strategies, validation-based selection, and evaluation procedures
used in the study. Patient-level EHR data and fitted checkpoints are not
distributed because the multisite data are governed by institutional data-use
agreements.

## Study Design

The study uses four source academic medical centers and one external target
institution. Model pretraining and downstream development use encounters from
2009-2020, while 2021 encounters form held-out temporal test sets.

The downstream tasks use common, outcome-independent prediction landmarks:

| Task | Model input | Outcome window |
|---|---|---|
| AKI onset | Hospital days 0-1 | Qualifying onset on days 2-8 |
| Electrolyte abnormalities | Hospital days 0-1 | Qualifying abnormal result on days 2-8 |
| In-hospital mortality | Hospital days 0-1 | Death from day 2 through discharge |
| AKI early reversal | Through the AKI-onset day | Reversal within seven days after AKI onset |
| AKI recovery | Through the AKI-onset day | Recovery based on the last available SCr before discharge |

The same prediction landmark is applied to cases and non-cases within each
task. Outcome status is not used to determine the observation window.

## Input Contract

Each encounter is represented by aligned columns:

- `SUBJECT_ID`: patient identifier.
- `HADM_ID`: encounter identifier.
- `SOURCE_SITE`: institution identifier for site-specific evaluation.
- `Events`: ordered event-token strings.
- `Type`: clinical-domain identifiers aligned with `Events`.
- `Time`: hospital-day offsets aligned with `Events`.
- One binary label column for each downstream outcome.

Modeled domains include demographics, diagnoses, procedures, inpatient
administered medications, numerical laboratory measurements, and categorical
laboratory results. The fixed study vocabulary contains 10,777 tokens,
including four special tokens, and sequences are limited to 1,024 tokens.

Missing longitudinal events remain absent from the sequence; no statistical
imputation is performed. Unknown demographic categories may be represented by
explicit categorical tokens, while unusable laboratory values are omitted from
numerical tokenization. Preprocessing specifications are derived from unlabeled
internal pretraining data without external-site data or outcome labels and are
then fixed for all downstream analyses.

## Repository Layout

```text
model/      EHR Transformer backbone and LoRA injection
data/       sequence assembly, temporal splitting, and landmark truncation
train/      masked-event pretraining and downstream adaptation strategies
eval/       discrimination, calibration, and checkpoint evaluation
analysis/   repeated-run aggregation and configuration comparison
```

## Environment

The reference modules require Python 3.10 or later, PyTorch, pandas, NumPy,
scikit-learn, and SciPy. Create the provided environment from the repository
root:

```bash
conda env create -f environment.yml
conda activate ehr-lora
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

## Reproduction Workflow

1. Harmonize locally authorized PCORnet-style tables and construct aligned
   encounter sequences.
2. Apply the common prediction landmark before creating downstream datasets.
3. Separate development and temporal-test encounters by admission year, then
   create patient-disjoint training and validation partitions.
4. Load the fixed pretrained backbone and configure Freeze-all, last 2-layer
   fine-tuning, LoRA, or full fine-tuning.
5. Calculate the positive-class loss weight from the training partition only.
6. Use validation AUPRC for early stopping, checkpoint selection, model
   selection, and LoRA configuration selection.
7. Reload the selected checkpoint and evaluate the held-out test set once.
8. Aggregate repeated runs and perform paired comparisons using matched runs.

For external-site adaptation, each strategy is fitted using the target
institution's development cohort and evaluated on its fixed temporal test set.
The task-specific LoRA configuration is selected using mean internal validation
AUPRC and is fixed before the external-site experiments.

For the limited-data analysis, nested, outcome-stratified subject samples are
drawn from each task-specific external development cohort at 1%, 5%, and 10%
data availability. Each sampled cohort is divided into patient-disjoint
training and validation partitions; the external test cohort remains fixed.

## LoRA Configuration Search

The configuration search varies:

- Target modules: `qv`, `qkvo`, and `qkvo_ffn`.
- Adaptation depth: the last 1, 10, or 20 Transformer layers.
- Rank: 8, 16, or 32, with alpha set to twice the rank.

Configurations are selected by mean validation AUPRC across repeated runs.
Test AUROC and AUPRC are not used for configuration or checkpoint selection.

## Evaluation

AUPRC is the primary discrimination metric, with outcome prevalence reported
as its no-skill reference. AUROC is secondary. Additional evaluation includes
Brier score, Brier skill score, calibration intercept and slope, calibration
curves, validation-selected classification thresholds, and decision-curve
analysis.

Logistic recalibration parameters and classification thresholds are estimated
from validation predictions and then applied to test predictions. Reported
operating points include maximum validation F1, maximum validation Youden
index, and approximately 80% validation sensitivity. Paired method comparisons
use two-sided Wilcoxon signed-rank tests with Holm correction.

External adaptation gains are calculated on the same external test set as the
performance of an active adaptation strategy minus the performance of
Freeze-all within each matched run.

## Data and Model Availability

Raw and encounter-level EHR data cannot be redistributed. The code operates on
locally authorized extracts satisfying the input contract above. The public
repository provides the environment specification, model and adaptation
components, and preprocessing and evaluation procedures needed to reproduce the
analysis with authorized data.
