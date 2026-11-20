# How We're Building a Smarter Alzheimer's Detector
### A plain-English walkthrough of our multi-agent architecture

---

## The Problem with Existing Models

There's a research paper called **CE-AH** — Contrast-Enhanced Attention Hierarchical — that does a pretty good job detecting Alzheimer's from brain MRI scans. It's smart, it's been benchmarked, and it works.

But it has one fundamental issue that bothered us: **it gives you one answer, from one model, with no way to question it.**

That's not how real medical diagnosis works. A good doctor doesn't look at one thing and immediately declare a verdict. They gather multiple opinions, look at the same patient from different angles, compare what they find, and only then commit to a conclusion — with appropriate uncertainty.

So our question became: what if we built a system that worked more like that?

---

## The Big Idea

Instead of one model that sees everything and spits out a label, we have a **team of specialist agents** that each look at the patient from a different angle, produce a detailed report of what they found, and then *debate* their findings before arriving at a final answer.

And crucially — the system also knows *when it's unsure.* If the agents strongly disagree, it flags the case for human review instead of silently committing to a wrong answer.

Here's what the whole thing looks like:

```
INPUT
  ├── sMRI Scan (3D structural MRI)
  └── Other Features (clinical data, cognitive test scores, demographics)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  TWO PARALLEL TEAMS OF SPECIALISTS                              │
│                                                                 │
│  ┌──────────────────────────┐  ┌───────────────────────────┐   │
│  │   sMRI TEAM              │  │   CLINICAL FEATURES TEAM  │   │
│  │                          │  │                           │   │
│  │  ┌──────────┐            │  │  ┌──────────┐             │   │
│  │  │ CE-AH    │            │  │  │ Clinical │             │   │
│  │  │ Assistant│            │  │  │ Assistant│             │   │
│  │  └────┬─────┘            │  │  └────┬─────┘             │   │
│  │       │  [their report]  │  │       │  [their report]   │   │
│  │  ┌────┴─────┐            │  │  ┌────┴─────┐             │   │
│  │  │  GNN     │            │  │  │ Cognitive│             │   │
│  │  │ Assistant│            │  │  │ Assistant│             │   │
│  │  └────┬─────┘            │  │  └────┬─────┘             │   │
│  │       │  [their report]  │  │       │  [their report]   │   │
│  │       ▼                  │  │       ▼                   │   │
│  │  ┌──────────────────┐    │  │  ┌──────────────────┐     │   │
│  │  │  TEAM SUPERVISOR │    │  │  │  TEAM SUPERVISOR │     │   │
│  │  │  compares reports│    │  │  │  compares reports│     │   │
│  │  │  assigns weights │    │  │  │  assigns weights │     │   │
│  │  └────────┬─────────┘    │  │  └────────┬─────────┘     │   │
│  │     team verdict         │  │     team verdict           │   │
│  └───────────┬──────────────┘  └───────────┬───────────────┘   │
└──────────────┼─────────────────────────────┼───────────────────┘
               │      [Feature Vectors]      │
               │   (CE-AH, GNN, Clin, Cogn)  │
               │              │              │
               │              ▼              │
               │    ┌───────────────────┐    │
               │    │  CROSS-ATTENTION  │    │
               │    │  FEATURE FUSION   │    │
               │    │  (inter-modality  │    │
               │    │   token queries)  │    │
               │    └─────────┬─────────┘    │
               │              │              │
               ▼              ▼              ▼
          [sMRI verdict] [Cross-Attn] [clinical verdict]
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                ┌───────────────────────────┐
                │      CHIEF PHYSICIAN      │
                │  combines dynamic peer    │
                │  consensus with neural    │
                │  cross-attention logits   │
                └─────────────┬─────────────┘
                              │
                              ▼
                       FINAL DIAGNOSIS
                (CN / MCI / AD + calibrated confidence
                 + 4x4 attention matrix & explanation)
```

---

## The Two Teams

We split the input into two fundamentally different types of information, and each type gets its own dedicated team.

**The sMRI Team** looks at the actual 3D brain scan. They're the radiologists.

**The Clinical Features Team** looks at everything else — cognitive test scores, memory assessments, clinical history. They're the neuropsychologists and GPs.

Why separate teams? Because these two types of information are so different in nature that mixing them in one model usually makes both worse. Better to let each team become a true specialist in their domain, and reconcile them at the end.

---

## Who's in the sMRI Team?

We thought hard about this. At one point we had five assistants planned — frequency domain analysis, surface morphometry, longitudinal tracking, contrastive encoders, and more. Then we started questioning each one honestly.

**Frequency domain?** Dropped. CE-AH already processes MRI hierarchically from fine patches up to whole-brain level — it already captures local texture implicitly. Frequency domain would just be a different mathematical representation of the same information.

**Surface morphometry?** Dropped. Requires running FreeSurfer — a preprocessing tool that takes 6 to 8 hours per scan. That's an enormous operational cost for marginal benefit over what CE-AH already captures volumetrically.

**Longitudinal LSTM?** Dropped. Only works if you have multiple MRI scans of the same patient over time. Most datasets, including ADNI, don't guarantee that for every patient. Building a dependency on data that might not exist is a bad design decision.

**Contrastive encoder?** Dropped as a standalone agent. Contrastive learning is a *training strategy*, not an inference tool. We'll use it to pretrain the CE-AH backbone and make its features stronger — but having it run in parallel at inference time adds cost for no real benefit.

After all that pruning, we're left with two assistants — and they're genuinely the right two:

**Assistant 1 — CE-AH Core**
This is the base model from the research paper. It looks at the brain scan in a hierarchy: tiny local patches first, then anatomical regions, then the whole brain. It's asking: *"Which structures look abnormal? Where exactly is the damage?"*

**Assistant 2 — GNN (Brain Graph Network)**
This one takes a completely different approach. Instead of treating brain regions independently, it builds a graph where each region is a node and the structural relationships between regions are edges. Then it runs a graph neural network over that structure. It's asking: *"Which regions have lost their normal relationship with each other?"*

These two are genuinely orthogonal. One looks at individual regions in isolation. The other looks at the connectivity patterns between regions. When a patient has early Alzheimer's, the hippocampus might not yet show dramatic visible atrophy — but its connectivity with the entorhinal cortex may already be breaking down. That's what the GNN catches that CE-AH misses.

When they agree, we're confident. When they disagree, something genuinely interesting is happening in this patient's brain.

---

## What Each Assistant Actually Produces

This is important to understand. These assistants don't just output *"probably AD."* They output a **detailed report** of everything they found:

```python
# Example output from the CE-AH assistant after examining a scan
{
    "attribute_vector": {
        "hippocampus_atrophy":    0.82,   # significant shrinkage detected
        "entorhinal_cortex_loss": 0.71,   # notable degradation
        "ventricle_expansion":    0.65,   # ventricles expanding = brain shrinking
    },
    "class_probs":  [0.09, 0.16, 0.75],  # 75% confident this is AD
    "confidence":   0.88,                 # how sure is CE-AH about its own answer
}
```

The `attribute_vector` is the *evidence* — named, measurable observations about what the model found. This is what the team supervisor later uses to compare what CE-AH and GNN actually saw. If CE-AH says hippocampus atrophy is 0.82 and GNN essentially agrees, that agreement carries weight. If they wildly differ on the same patient, that tells you something.

The `class_probs` is the vote. The `confidence` is how firmly each assistant stands behind its vote.

---

## How Confidence Actually Works

You might think confidence is just `max(class_probs)` — if the model says 85% AD, then confidence is 0.85. Simple.

The problem is that neural networks are notoriously overconfident. A model can scream "definitely AD!" with 95% probability and be completely wrong. The number is not calibrated.

We use something better called **Monte Carlo Dropout** (MC Dropout). Here's the intuition:

During training, dropout randomly switches off neurons to prevent the model from becoming too reliant on any single pathway. Usually we turn dropout off at inference time. We don't — we leave it on, and run the same scan through the model 30 times.

Each run, a slightly different random set of neurons is disabled. If the prediction is genuinely robust, it barely changes across runs. If the model is actually uncertain, the prediction wobbles considerably.

```
Run 1:  [0.05, 0.10, 0.85]   → AD
Run 2:  [0.08, 0.15, 0.77]   → AD
Run 3:  [0.20, 0.25, 0.55]   → AD but weaker (dropout hit something critical)
Run 4:  [0.06, 0.12, 0.82]   → AD
...

Mean:    [0.09, 0.16, 0.75]   ← actual prediction we report
Std dev: [0.06, 0.07, 0.12]   ← how much it wobbled

confidence = 1 - std_dev      ← stable = confident, unstable = uncertain
```

This all happens inside the assistant itself. No extra agent, no extra complexity — just 30 forward passes that give us an honest measure of uncertainty.

---

## How the Team Supervisor Works (Intra-Branch Arbitration)

Once both assistants in the sMRI team have submitted their reports, the **Team Supervisor** takes over. Its job is to produce a single trusted team verdict from two potentially different opinions.

**First, it measures how much they agreed.**

It computes cosine similarity between CE-AH's probability output and GNN's probability output. Cosine similarity is essentially asking: "Are these two vectors pointing in the same direction?" If both strongly say AD, they point the same way. If one says AD and the other says CN, they point in opposite directions.

```python
# Both saying similar things → high agreement
CE_AH_probs = [0.05, 0.10, 0.85]
GNN_probs   = [0.08, 0.12, 0.80]
agreement   = 0.998   ✓

# Completely opposite → low agreement
CE_AH_probs = [0.05, 0.10, 0.85]
GNN_probs   = [0.60, 0.30, 0.10]
agreement   = 0.12    ← something unusual is going on
```

**Then it computes weights.**

Each assistant's weight in the final team verdict is determined by how much it agreed with the other assistant on average. An assistant that aligned with its colleague gets more say. An outlier gets less say.

**But here's the important exception — the Minority Report.**

What if one assistant strongly disagrees with everyone else, but does so with *very high confidence*? That's not a noisy outlier — that might be a specialist catching something subtle that the others completely missed. We don't suppress it. We flag it as a **minority report** and pass it all the way up to the Chief Physician, who will factor it in separately.

**Finally, it produces the weighted team verdict.**

```python
team_verdict = weighted_average(CE_AH_probs, GNN_probs, by_weights)
# Plus: how confident the team overall is, how much they agreed, any minority reports
```

---

## How the Chief Physician Works (Meta-Arbitration Fusion)

This is the top of the hierarchy. The Chief Physician receives one verdict from the sMRI team and one from the clinical features team, and has to reconcile them into a final answer.

Here's what that looks like in practice:

```python
sMRI_team = {
    "class_probs":     [0.09, 0.16, 0.75],   # sMRI team says: AD
    "confidence":       0.82,                  # they're fairly sure
    "intra_agreement":  0.75,                  # the two sMRI assistants mostly agreed
}

clinical_team = {
    "class_probs":     [0.20, 0.50, 0.30],   # clinical team says: MCI  ← conflict!
    "confidence":       0.61,                  # less sure
    "intra_agreement":  0.55,                  # their assistants were less aligned too
}
```

**Step 1 — How much should we trust each team?**

Trust isn't set manually. It's computed from what actually happened inside each team:

```
trust_sMRI     = 0.82 × 0.75 = 0.615
trust_clinical = 0.61 × 0.55 = 0.336
```

If a team's own assistants barely agreed with each other, its trust score drops automatically. A team that wasn't internally coherent shouldn't carry full weight in the final decision.

**Step 2 — Normalize to weights**

```
sMRI team gets:     64.7% of the final say
Clinical team gets: 35.3% of the final say
```

**Step 3 — Check for conflict**

The sMRI team says AD. The clinical team says MCI. These are different. The system flags this conflict and amplifies the weight difference rather than naively averaging — we lean harder into the team we trust more.

**Step 4 — Cross-Attention Multimodal Feature Fusion**

Instead of purely relying on late-stage probability averaging, the pipeline embeds a **Neural Cross-Attention Fusion module** (`models/fusion/cross_attention.py`):
1. Projects the specialist attribute representations (CE-AH: 512, GNN: 256, Clinical: 64, Cognitive: 64) into a shared 128-dimensional latent space.
2. Applies learnable modality-type embeddings to identify each agent token.
3. Multi-Head Cross/Self Attention allows agents to directly query one another across modalities:
   - CE-AH queries Cognitive to verify whether hippocampal atrophy aligns with memory decline.
   - GNN queries Clinical to verify whether structural network disruption correlates with CDR severity.
4. Generates a 4×4 inter-modality cross-attention explainability matrix alongside deep fused classification probabilities.

**Step 5 — Final Blended Verdict**

The Chief Physician blends calibrated Bayesian arbitration with the deep Cross-Attention representations (`0.5 * rule_probs + 0.5 * fusion_probs`), deferring gracefully to tabular specialists if an MRI scan is unavailable.

```
CN  = 0.647×0.09 + 0.353×0.20 = 0.13
MCI = 0.647×0.16 + 0.353×0.50 = 0.28
AD  = 0.647×0.75 + 0.353×0.30 = 0.59

→ Blended with Neural Cross-Attention Logits
→ Diagnosis: AD (64% confidence)
→ Disagreement flag: raised — a clinician should review this case
```

The final output is not just a label. It's a full report:

```python
FinalResult = {
    "diagnosis":         "AD",
    "probabilities":     {"CN": 0.11, "MCI": 0.25, "AD": 0.64},
    "confidence":        0.642,
    "modality_weights":  {"sMRI": 0.647, "clinical": 0.353},
    "disagreement_flag": True,    # ← teams didn't fully agree
    "explanation": {
        "top_brain_regions":        ["hippocampus", "entorhinal cortex"],
        "top_clinical_factors":     ["MMSE decline", "CDR score"],
        "cross_attention_weights":  [[... 4x4 matrix ...]],
        "minority_signals":         []
    }
}
```

---

## What Actually Lives Inside Each Assistant

A question that comes up naturally: *if the assistant outputs attributes, who computes the class probabilities and confidence? Do we need another mini-agent inside each assistant?*

No. Each assistant is one model with two output heads — they're trained together and run together:

```
Raw Input (e.g., 3D brain scan)
    │
    ▼
┌─────────────────────┐
│  BACKBONE           │  ← The heavy-lifting model (CE-AH Transformer or GNN)
│                     │    This is where features are actually learned
└─────────┬───────────┘
          │
          ▼
    attribute_vector     ← Named, interpretable observations about the scan
          │
          ▼
┌─────────────────────┐
│  CLASSIFICATION     │  ← A small linear layer + softmax
│  HEAD               │    Converts attributes → class probabilities
└─────────┬───────────┘
          │
          ▼
    class_probs          ← [CN, MCI, AD] vote
    confidence           ← stability across 30 MC Dropout passes
```

The backbone and classification head train together, end to end. During inference, the MC Dropout runs happen automatically. It's one assistant, two outputs, no internal agents needed.

---

## The Doctor Analogy

If all of the above feels abstract, here's the whole system mapped to something familiar:

| What the system does | What a medical team does |
|---|---|
| CE-AH Assistant | Radiologist reading the brain scan |
| GNN Assistant | Neurologist examining brain network patterns |
| Clinical Assistant | General practitioner reviewing patient history |
| Cognitive Assistant | Neuropsychologist reviewing test battery results |
| Team Supervisor (Intra-Branch Arbitration) | Department head reconciling two specialists who slightly disagreed |
| Chief Physician (Fusion Agent) | Senior consultant combining radiology and clinical findings |
| High confidence | Doctor says "I'm certain about this" |
| Low intra-agreement | Two doctors in the same room reached different conclusions |
| Minority Report | Junior doctor quietly noting something unusual the seniors overlooked |
| Disagreement flag raised | "Let's get a second opinion before we proceed" |

This isn't just a metaphor — it shaped every design decision. The system is designed to *behave* like a careful, self-aware medical team, not like a black box that silently outputs a label.

---

## Growing the System Later

One thing we care about is not painting ourselves into a corner. Everything here is built to expand cleanly:

- Want to add **PET scans** as a third modality? Create a new team with its own assistants and supervisor. Plug it into the Chief Physician. Nothing else changes.
- Want to add a **new sMRI assistant**? Register it in the sMRI team. The supervisor automatically recomputes weights.
- Want to make the arbitration **learned** instead of heuristic? Replace the cosine-similarity weight computation with a trained attention module. The assistants don't change at all.
- Want to add **genetic data** (APOE-ε4 status)? New assistant in the clinical team. Done.

The point is that each layer only knows about its immediate neighbours. Assistants don't know about the Chief Physician. The Chief Physician doesn't know about individual assistants. That separation is what makes the system modular.

---

## What We're Building With

| What | Tool |
|---|---|
| Core deep learning | PyTorch |
| CE-AH backbone | Custom Transformer (hierarchical attention) |
| GNN backbone | PyTorch Geometric |
| Brain scan preprocessing | NiBabel, SimpleITK |
| Clinical tabular models | XGBoost, Scikit-learn |
| Uncertainty estimation | MC Dropout (native PyTorch) |
| Explaining decisions | Captum (Grad-CAM, Integrated Gradients), SHAP |
| Tracking experiments | Weights & Biases |
| Managing configs | Hydra + YAML |
| Data | ADNI / OASIS-3 |
