# CV Lab — Multi-Agent Alzheimer's Detection System

> A hierarchical ensemble of specialist AI agents that diagnoses Alzheimer's disease from structural MRI scans and clinical data — and knows when it's unsure.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Why Multi-Agent?](#2-why-multi-agent)
3. [System Architecture](#3-system-architecture)
4. [The Models (What Each Agent Learns)](#4-the-models-what-each-agent-learns)
5. [How Confidence Works — Monte Carlo Dropout](#5-how-confidence-works--monte-carlo-dropout)
6. [How Arbitration Works](#6-how-arbitration-works)
7. [The Dataset](#7-the-dataset)
8. [Project File Structure](#8-project-file-structure)
9. [Configuration System](#9-configuration-system)
10. [How to Run](#10-how-to-run)
11. [Training Phases Explained](#11-training-phases-explained)
12. [Reading the Output](#12-reading-the-output)

---

## 1. What This Project Does

This system classifies patients into three groups based on their brain state:

| Label | Name | Meaning |
|---|---|---|
| `CN` | Cognitively Normal | No signs of Alzheimer's |
| `MCI` | Mild Cognitive Impairment | Early warning stage — may progress to AD |
| `AD` | Alzheimer's Disease | Confirmed neurodegeneration |

**Inputs it reads:**
- A 3D structural MRI (sMRI) scan of the patient's brain
- Clinical data: MMSE score, CDR rating, age, sex, medications, etc.
- Cognitive test scores: memory, executive function, ADAS-Cog, etc.

**Output it produces:**
```
DIAGNOSIS       : AD
Confidence      : 78.3%
Probabilities   : CN=8.2%  MCI=13.5%  AD=78.3%
sMRI weight     : 64.7%
Features weight : 35.3%
Disagreement    : YES -- clinician review advised
Top brain regions   : Hippocampus_L, ParaHippocampal_R, Amygdala_L
Top clinical factors: MMSE, CDR, ADAS_Cog_11
```

---

## 2. Why Multi-Agent?

Most existing Alzheimer's detection models — including the CE-AH paper this builds on — work like a black box: one model, one answer, no explanation of why or how confident it is.

The problem is that **real medical diagnosis doesn't work that way.** A doctor:
- Looks at multiple sources of evidence
- Consults colleagues with different specialisations
- Flags disagreements for review rather than silently committing to a wrong answer
- Communicates uncertainty explicitly

This system is designed to behave the same way. The key properties:

| Behaviour | How it's achieved |
|---|---|
| Multiple expert opinions | 4 specialist assistant agents |
| Domain separation | Two independent branches (MRI vs. clinical) |
| Honest uncertainty | Monte Carlo Dropout — 30 stochastic passes per assistant |
| Disagreement detection | Cosine similarity between agent predictions |
| Clinician escalation | `disagreement_flag` raised when teams conflict |
| Explainability | Named brain regions and clinical factors in every output |

---

## 3. System Architecture

```
INPUT
  ├── sMRI Scan  [B, 1, 91, 109, 91]
  └── Clinical + Cognitive features  [B, 12]  [B, 8]
        |
        v
┌──────────────────────────────────────────────────────────────┐
│  TWO PARALLEL BRANCHES                                       │
│                                                              │
│  ┌────────────────────────┐   ┌────────────────────────────┐ │
│  │   sMRI BRANCH          │   │   FEATURES BRANCH          │ │
│  │                        │   │                            │ │
│  │  [CE-AH Assistant]     │   │  [Clinical Assistant]      │ │
│  │   3-level Transformer  │   │   MLP + ResNet blocks      │ │
│  │   on 3D brain patches  │   │   12 clinical features     │ │
│  │        |               │   │        |                   │ │
│  │  [GNN Assistant]       │   │  [Cognitive Assistant]     │ │
│  │   Graph Attention Net  │   │   MLP + ResNet blocks      │ │
│  │   90-region brain      │   │   8 cognitive test scores  │ │
│  │        |               │   │        |                   │ │
│  │  [Intra-Arbitrator]    │   │  [Intra-Arbitrator]        │ │
│  │   cosine agreement     │   │   cosine agreement         │ │
│  │   weighted consensus   │   │   weighted consensus       │ │
│  └──────────┬─────────────┘   └──────────┬─────────────────┘ │
│             |  BranchResult              |  BranchResult      │
└─────────────|──────────────────────────--|────────────────────┘
              └──────────────┬────────────┘
                             v
                   ┌─────────────────────┐
                   │   CHIEF PHYSICIAN   │
                   │                     │
                   │  trust = confidence │
                   │       × agreement   │
                   │                     │
                   │  weighted fusion    │
                   │  + conflict detect  │
                   └──────────┬──────────┘
                              v
                        FinalResult
```

### The Four Specialists

| Agent | File | What it sees | What it detects |
|---|---|---|---|
| **CE-AH** | `models/ceah/ceah_model.py` | 3D MRI volume | WHERE structures are abnormal (patch → region → whole brain) |
| **GNN** | `models/gnn/gnn_model.py` | Brain as a graph of 90 regions | HOW regions relate to each other (co-atrophy patterns) |
| **Clinical** | `models/tabular/clinical_model.py` | MMSE, CDR, age, etc. | Clinical risk profile |
| **Cognitive** | `models/tabular/cognitive_model.py` | Memory, ADAS-Cog, etc. | Neuropsychological decline pattern |

---

## 4. The Models (What Each Agent Learns)

### CE-AH — Contrast-Enhanced Attention Hierarchical

The flagship model. Processes MRI in three hierarchical levels:

```
3D MRI [B, 1, 91, 109, 91]
    |
    v
LEVEL 1 — Patch Encoder
    Conv3D splits volume into 8x8x8 patches  -> [B, N_patches, 128]
    Transformer blocks (depth=2, heads=4)
    Positional encoding
    |
    v
LEVEL 2 — Region Encoder
    Assigns each patch to its nearest AAL brain region
    Mean-pools patches into 90 region embeddings -> [B, 90, 256]
    Transformer blocks with anatomical positional encoding
    |
    v
LEVEL 3 — Whole-Brain Encoder
    Prepends a [CLS] token (like BERT)
    Transformer attends over all 90 regions -> [B, 91, 512]
    CLS token output = brain summary vector -> [B, 512]
    |
    v
Classification Head -> logits [B, 3]
Feature vector  [B, 512]  <- used by arbitration layer
```

**Why this hierarchy?**
- Level 1 catches fine local texture — small lesions, subtle grey matter changes
- Level 2 captures region-specific atrophy — "hippocampus is 30% smaller than normal"
- Level 3 integrates everything — bilateral asymmetries, multi-lobe patterns

---

### GNN — Brain Graph Network

Treats the brain as a network, not a collection of independent regions. Uses the AAL atlas (90 anatomical regions as nodes) and connects nearby regions with edges.

```
3D MRI [B, 1, 91, 109, 91]
    |
    v
BrainGraphBuilder
    Per-voxel nearest-region assignment (pre-computed at init)
    Extract 4 statistics per region:  mean, std, max, min intensity
    -> raw node features [B, 90, 4]
    Linear projection -> [B, 90, 64]
    |
    v
Graph Attention Network (3 GAT layers)
    Each region attends over its anatomical neighbours
    Multi-head attention (4 heads) learns edge weights
    -> [B, 90, 128]
    |
    v
Global mean pool across all regions -> [B, 256]  (graph embedding)
    |
    v
Classification Head -> logits [B, 3]
```

**Why GNN catches things CE-AH misses:**
CE-AH looks at each region somewhat independently. GNN explicitly models *which regions are losing their normal relationship with each other*. In early Alzheimer's, the hippocampus–entorhinal cortex connection breaks down *before* either region shows visible atrophy. GNN catches that.

---

### Clinical Model

A residual MLP for tabular clinical features:

```
12 clinical features (MMSE, CDR, Age, Sex, Education, ...)
    |
    v
Linear(12 -> 64) + BatchNorm + GELU + Dropout
    |
    v
ResidualBlock(64 -> 64)     <- skip connections for stable training
Linear(64 -> 128) + BN + GELU
ResidualBlock(128 -> 128)
Linear(128 -> 64)
    |
    v
Classification Head -> logits [B, 3]
Feature vector [B, 64]  <- for arbitration
```

---

### Cognitive Model

Same architecture as Clinical, but for 8 neuropsychological test scores:
- Memory composite, Executive function, Language, Visuospatial
- Attention, Processing speed, ADAS-Cog 11, Digit span

---

## 5. How Confidence Works — Monte Carlo Dropout

Standard neural networks are **overconfident**: a model can say "99% AD" and be completely wrong.

We use **Monte Carlo Dropout** to measure *actual* uncertainty:

```
Normal inference:   dropout OFF  →  one deterministic prediction
MC Dropout:         dropout ON   →  30 different stochastic predictions

Run 1:  softmax → [0.05, 0.10, 0.85]
Run 2:  softmax → [0.08, 0.15, 0.77]
Run 3:  softmax → [0.20, 0.25, 0.55]  ← dropout disabled a key path
Run 4:  softmax → [0.06, 0.12, 0.82]
...
Run 30: softmax → [0.07, 0.11, 0.82]

mean_probs = [0.09, 0.16, 0.75]   <- final prediction
std_probs  = [0.06, 0.07, 0.12]   <- uncertainty per class

confidence = 1 - mean(std_probs)  <- high = stable = confident
```

**Why this matters:**
- If 29/30 runs agree on AD, confidence is high (~0.95)
- If 15/30 runs say MCI and 15 say AD, confidence is low (~0.55)
- Low confidence automatically propagates up to raise the `disagreement_flag`

Implemented in [`utils/mc_dropout.py`](utils/mc_dropout.py).

---

## 6. How Arbitration Works

### Intra-Branch Arbitration (within each team)

After both assistants in a branch run, the `IntraArbitrator` reconciles them:

```python
# Step 1: Agreement matrix — cosine similarity between each pair's predictions
agreement_matrix[CE-AH, GNN] = cosine_sim(ceah_probs, gnn_probs)

# Example: both say AD strongly
CE-AH: [0.05, 0.10, 0.85]
GNN:   [0.08, 0.12, 0.80]
agreement = 0.998  <- very high

# Example: they disagree
CE-AH: [0.05, 0.10, 0.85]  <- says AD
GNN:   [0.60, 0.30, 0.10]  <- says CN
agreement = 0.12   <- very low

# Step 2: Peer weights — assistant gets more weight if it agreed with others
weight_i = mean(agreement with all j != i)

# Step 3: Minority report detection
# If an assistant has LOW weight but HIGH confidence -> specialist signal
# (not suppressed — passed up to Chief Physician as a flag)

# Step 4: Weighted consensus
branch_probs = sum(weight_i * probs_i)
```

Implemented in [`agents/arbitration/intra_arbitrator.py`](agents/arbitration/intra_arbitrator.py).

---

### Meta-Arbitration (Chief Physician)

The Chief Physician fuses the two branch verdicts:

```python
# Step 1: Trust score per branch
trust_smri     = smri_confidence     * smri_intra_agreement
trust_features = features_confidence * features_intra_agreement

# Example:
# sMRI team: confidence=0.82, intra_agreement=0.75 -> trust=0.615
# Features:  confidence=0.61, intra_agreement=0.55 -> trust=0.336

# Step 2: Normalize to weights
w_smri     = 0.615 / (0.615 + 0.336) = 0.647
w_features = 0.336 / (0.615 + 0.336) = 0.353

# Step 3: Detect conflict
smri says:     AD  (argmax of smri_probs)
features says: MCI (argmax of features_probs)
-> disagreement_flag = True

# Step 4: Amplify weight gap on conflict (lean harder into trusted team)
# Power sharpening: w_i^(1/k) then renormalise (k=2.0 by default)
w_smri     -> 0.700  (advantage amplified)
w_features -> 0.300

# Step 5: Final probabilities
final_probs = 0.700 * smri_probs + 0.300 * features_probs
diagnosis   = argmax(final_probs)
```

Implemented in [`agents/arbitration/chief_physician.py`](agents/arbitration/chief_physician.py).

---

## 7. The Dataset

### Option A — Built-in Synthetic Mode (default, no files needed)

When `cfg.data.synthetic = true`, the `AlzheimerDataset` generates random tensors in memory with class-specific signals:

```python
# AD patients: hippocampal region intensity reduced
mri[i, 0, 35:55, 45:65, 35:55] -= 0.5   # AD
mri[i, 0, 35:55, 45:65, 35:55] -= 0.2   # MCI
# CN: no modification
```

Good for: smoke tests, architecture validation, quick iteration.

---

### Option B — Generated ADNI-Style Dataset (recommended for training)

Run `data/generate_dataset.py` to create a full file-based dataset:

```
data/raw/
  metadata.csv          subject_id, label (0/1/2), split (train/val/test)
  clinical.csv          12 clinical features, indexed by subject_id
  cognitive.csv         8 cognitive features, indexed by subject_id
  mri/
    ADNI_0000.nii.gz    91x109x91 NIfTI structural MRI
    ADNI_0001.nii.gz
    ...
```

#### Clinical features (12 total)

| Feature | CN typical | MCI typical | AD typical | Notes |
|---|---|---|---|---|
| `MMSE` | 29.0 | 26.5 | 18.0 | Max 30, lower = worse cognition |
| `CDR` | 0.0 | 0.5 | 1.5 | 0=normal, 0.5=questionable, 1+=dementia |
| `Age` | 72 | 75 | 78 | Years |
| `Sex` | — | — | — | Binary (0/1) |
| `Education_years` | 15 | 13.5 | 12 | Years of formal education |
| `Handedness` | 0.9 | 0.9 | 0.9 | 1=right-handed |
| `BMI` | 26 | 26.5 | 27 | Body mass index |
| `Systolic_BP` | 128 | 132 | 136 | mmHg |
| `Family_history` | 0.3 | 0.4 | 0.5 | Binary — first-degree relative with AD |
| `Depression_history` | 0.15 | 0.25 | 0.40 | Binary |
| `Num_medications` | 2 | 4 | 6 | Count |
| `Comorbidity_index` | 1 | 2 | 3.5 | Charlson comorbidity index |

#### Cognitive features (8 total)

| Feature | CN typical | MCI typical | AD typical | Notes |
|---|---|---|---|---|
| `Memory_composite` | 0.8 | 0.4 | -0.6 | Z-score |
| `Executive_function` | 0.7 | 0.3 | -0.4 | Z-score |
| `Language` | 0.6 | 0.2 | -0.5 | Z-score |
| `Visuospatial` | 0.5 | 0.1 | -0.6 | Z-score |
| `Attention` | 0.7 | 0.3 | -0.3 | Z-score |
| `Processing_speed` | 0.6 | 0.2 | -0.5 | Z-score |
| `ADAS_Cog_11` | 8 | 18 | 35 | Lower = better (0–70 scale) |
| `Digit_span` | 10 | 8 | 5.5 | Working memory test (0–20) |

#### MRI signals per class

Each synthetic MRI is a 91×109×91 volume with:
- Gaussian background simulating brain tissue
- Ellipsoidal brain mask (rough skull-stripping)
- Class-specific atrophy patterns:

```
CN:  Hippocampus has higher intensity (healthy grey matter density)

MCI: Mild hippocampal signal reduction
     Slight entorhinal cortex thinning

AD:  Strong hippocampal atrophy (bilateral)
     Entorhinal cortex signal loss
     Ventricle expansion (CSF filling space left by atrophied tissue)
```

Anatomical voxel regions used:
```python
hippo_L  = [35:52, 45:62, 28:45]    # left hippocampus
hippo_R  = [35:52, 45:62, 46:63]    # right hippocampus
entorh   = [38:50, 48:62, 32:58]    # entorhinal cortex
ventr    = [38:53, 52:70, 38:53]    # lateral ventricles
```

---

### Option C — Real ADNI/OASIS-3 Data

Set `cfg.data.synthetic = false` and `cfg.data.root = /path/to/your/data`. Expects:

```
your_data_root/
  metadata.csv           columns: subject_id, label, split
  clinical.csv           indexed by subject_id, 12 columns
  cognitive.csv          indexed by subject_id, 8 columns
  mri/<subject_id>.nii.gz
```

The pipeline calls `nibabel.load()` on each NIfTI file, normalises intensity to zero-mean unit-variance, and crops/pads to 91×109×91 (MNI152 standard space).

---

## 8. Project File Structure

```
CV Lab/
│
├── ARCHITECTURE.md          Original design document
├── README.md                This file
├── requirements.txt         Python dependencies
│
├── run.bat                  Interactive launcher (double-click to start)
├── smoke_test.py            End-to-end pipeline test (no data needed)
├── train.py                 Three-phase training script
├── pipeline.py              AlzheimerPipeline — main inference entry point
│
├── configs/
│   ├── config.yaml          Master config (data, paths, general)
│   ├── model/
│   │   ├── ceah/ceah.yaml   CE-AH hyperparameters
│   │   ├── gnn/gnn.yaml     GNN hyperparameters
│   │   ├── clinical/        Clinical MLP hyperparameters
│   │   └── cognitive/       Cognitive MLP hyperparameters
│   └── training/
│       └── default.yaml     Training hyperparameters (LR, epochs, etc.)
│
├── agents/
│   ├── base_assistant.py    Abstract base — defines predict() + MC Dropout
│   ├── smri_branch/
│   │   ├── ceah_assistant.py     Wraps CEAHModel
│   │   ├── gnn_assistant.py      Wraps GNNModel + BrainGraphBuilder
│   │   └── smri_supervisor.py    Runs CE-AH + GNN, calls IntraArbitrator
│   ├── features_branch/
│   │   ├── clinical_assistant.py     Wraps ClinicalModel
│   │   ├── cognitive_assistant.py    Wraps CognitiveModel
│   │   └── features_supervisor.py    Runs Clinical + Cognitive, arbitrates
│   └── arbitration/
│       ├── intra_arbitrator.py   Cosine-similarity weighted consensus
│       └── chief_physician.py    Trust-weighted meta-fusion, FinalResult
│
├── models/
│   ├── ceah/
│   │   ├── ceah_model.py         Assembles the three-level CE-AH pipeline
│   │   ├── patch_encoder.py      Level 1: Conv3D + Transformer on patches
│   │   ├── region_encoder.py     Level 2: Pool patches to 90 AAL regions
│   │   └── whole_brain_encoder.py Level 3: CLS-token global Transformer
│   ├── gnn/
│   │   ├── brain_graph.py        BrainGraphBuilder (voxel -> node features)
│   │   └── gnn_model.py          Multi-layer GAT (Graph Attention Network)
│   ├── heads/
│   │   └── classification_head.py Shared Linear -> BN -> GELU -> Linear head
│   └── tabular/
│       ├── clinical_model.py     Residual MLP for clinical features
│       └── cognitive_model.py    Residual MLP for cognitive scores
│
├── data/
│   ├── dataset.py            AlzheimerDataset + DataLoader factory
│   ├── generate_dataset.py   Synthetic ADNI-style dataset generator
│   └── raw/                  Generated or real data lives here
│       ├── metadata.csv
│       ├── clinical.csv
│       ├── cognitive.csv
│       └── mri/
│
└── utils/
    ├── brain_atlas.py        AAL atlas: 90 region names, centroids, edges
    └── mc_dropout.py         Monte Carlo Dropout inference helper
```

---

## 9. Configuration System

The project uses **Hydra** for configuration. All settings live in YAML files and can be overridden from the command line without editing files.

### How configs compose

```
configs/config.yaml          (root config)
    imports model/ceah/ceah.yaml   -> merged at cfg.model.ceah.*
    imports model/gnn/gnn.yaml     -> merged at cfg.model.gnn.*
    imports model/clinical/        -> merged at cfg.model.clinical.*
    imports model/cognitive/       -> merged at cfg.model.cognitive.*
    imports training/default.yaml  -> merged at cfg.training.*
```

### Key settings

```yaml
# configs/config.yaml
data:
  synthetic: true           # false = load from data/raw/
  num_synthetic_samples: 200
  batch_size: 4
  mri_shape: [91, 109, 91]

# configs/training/default.yaml
lr: 1e-4
epochs: 100
patience: 15
train_smri_branch: true
train_features_branch: true
joint_finetune: true        # Phase 3 joint fine-tuning
```

### Overriding from CLI (no YAML editing needed)

```bash
# Change any setting on the fly
python train.py training.epochs=50
python train.py data.synthetic=false data.root=./data/raw
python train.py data.batch_size=8 training.lr=5e-5
```

---

## 10. How to Run

### Prerequisites

```bash
pip install torch>=2.0.0 torchvision
pip install hydra-core omegaconf
pip install scikit-learn numpy pandas nibabel tqdm scipy matplotlib
```

> GPU is not required — everything runs on CPU (slower but functional).

---

### Quick Start — Double-click `run.bat`

```
============================================================
  CV Lab | Multi-Agent Alzheimer's Detector
============================================================

  [1]  Install dependencies
  [2]  Run smoke test          (verify pipeline, no data needed)
  [3]  Generate synthetic dataset
  [4]  Train   (synthetic data, quick -- no files needed)
  [5]  Train   (generated dataset, full file-based)
  [6]  Inspect generated dataset
  [7]  Quick-train demo        (5 epochs, synthetic, fast)
  [8]  Exit
```

---

### Step by step (manual)

#### Step 1 — Verify the pipeline builds correctly
```bash
python smoke_test.py
# Expected: "ALL CHECKS PASSED -- pipeline is healthy!"
```

#### Step 2 — Generate a dataset (300 subjects, ~2 min)
```bash
python data/generate_dataset.py --n 300 --seed 42

# Options:
#   --n 500         more subjects
#   --no-mri        skip NIfTI files (CSVs only, much faster)
#   --cn 0.4        fraction of CN patients
#   --mci 0.3       fraction of MCI patients
#   --ad 0.3        fraction of AD patients
#   --out ./data/raw  output directory
```

#### Step 3 — Train with the generated dataset
```bash
python train.py \
  data.synthetic=false \
  data.root=./data/raw \
  data.num_workers=0 \
  training.epochs=100
```

#### Step 4 — Or train with synthetic data (fastest)
```bash
python train.py data.synthetic=true data.num_synthetic_samples=500
```

#### Step 5 — Quick 5-epoch demo to verify training works
```bash
python train.py \
  data.synthetic=true \
  data.num_synthetic_samples=80 \
  data.batch_size=4 \
  data.num_workers=0 \
  training.epochs=5 \
  training.patience=5 \
  training.joint_finetune=false
```

---

## 11. Training Phases Explained

Training runs in three sequential phases. Each is controlled by flags in `configs/training/default.yaml`.

### Phase 1 — sMRI Branch

```
1a. Train CE-AH assistant independently
      Loss: CrossEntropy(CE-AH logits, labels) + label_smoothing
      Optimiser: AdamW, cosine LR schedule with linear warmup
      Checkpoint: checkpoints/ceah_best.pt   (best val macro-F1)

1b. Train GNN assistant independently
      Loss: CrossEntropy(GNN logits, labels)
      Checkpoint: checkpoints/gnn_best.pt

1c. Joint sMRI fine-tune (3 epochs)
      Loss: 0.5 * CE(ceah_logits) + 0.5 * CE(gnn_logits)
      Both assistants update simultaneously
      Checkpoint: checkpoints/smri_joint_best.pt
```

### Phase 2 — Features Branch

```
2a. Train Clinical assistant independently
2b. Train Cognitive assistant independently
2c. Joint features fine-tune (3 epochs)
```

### Phase 3 — Full Joint Fine-Tune (optional)

All four assistants trained simultaneously:
```
Loss = (CE(ceah) + CE(gnn) + CE(clinical) + CE(cognitive)) / 4

Flags: joint_finetune: true
       joint_finetune_epochs: 20
       joint_finetune_lr: 1e-5   (reduced LR for fine-tuning)
```

### Early Stopping

Each assistant training uses patience-based early stopping on val macro-F1. If F1 hasn't improved for `patience` epochs, training stops and the best checkpoint is restored.

### Final checkpoint

After all phases, the full pipeline is saved:
```
checkpoints/pipeline_final.pt
  ├── smri_branch
  │     ├── ceah  (state_dict)
  │     └── gnn   (state_dict)
  └── features_branch
        ├── clinical  (state_dict)
        └── cognitive (state_dict)
```

---

## 12. Reading the Output

### FinalResult fields

```python
result = pipeline.predict(batch)

result.diagnosis          # "CN" | "MCI" | "AD"
result.probabilities      # {"CN": 0.08, "MCI": 0.14, "AD": 0.78}
result.confidence         # 0.78  (probability of top class)
result.modality_weights   # {"sMRI": 0.65, "Features": 0.35}
result.disagreement_flag  # True/False — clinician review advisory
result.explanation        # {
                          #   "top_brain_regions":    ["Hippocampus_L", ...],
                          #   "top_clinical_factors": ["MMSE", "CDR", ...],
                          #   "minority_signals":     [...]   # if any
                          # }
```

### Interpreting modality weights

| Scenario | What it means |
|---|---|
| `sMRI=0.65, Features=0.35` | MRI evidence is stronger AND the two sMRI assistants agreed more |
| `sMRI=0.35, Features=0.65` | Clinical/cognitive data is providing clearer signal |
| Weights near 50/50 | Both teams equally trusted — or both uncertain |

### Interpreting the disagreement flag

`disagreement_flag = True` is raised when:
- The sMRI team and features team predicted different top classes

This does **not** mean the answer is wrong. It means the system detected a genuinely ambiguous case and is recommending a clinician review rather than committing silently.

### Interpreting minority signals

A minority signal appears when an assistant:
- Has **low peer weight** (its peers disagreed with it)
- But has **high confidence** (it's sure about its own prediction)

This is the system saying: *"One specialist strongly disagrees with everyone else — this might be worth looking into."*

---

## Brain Atlas Reference

The system uses the **AAL (Automated Anatomical Labeling)** atlas with 90 brain regions. The most AD-relevant ones the system watches:

| Region | Relevance |
|---|---|
| `Hippocampus_L / R` | First to atrophy in AD — key diagnostic marker |
| `ParaHippocampal_L / R` | Surrounds hippocampus, affected early |
| `Amygdala_L / R` | Emotion/memory hub, shrinks in AD |
| `Temporal_Inf_L / R` | Language and semantic memory |
| `Cingulum_Post_L / R` | Default mode network hub |
| `Precuneus_L / R` | Episodic memory, imagery |

The GNN uses all 90 regions as graph nodes and connects them by anatomical proximity (< 40mm between centroids in MNI space).

---

*Built on PyTorch 2.0+, Hydra 1.3, and the CE-AH paper backbone.*
#   c v - p r o j e c t  
 