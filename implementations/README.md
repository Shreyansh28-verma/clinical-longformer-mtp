# Clinical Longformer — Implementations

This directory contains paper-grounded improvements to the baseline **TeacherLongformer** reranker for patient-to-trial matching. Each sub-directory is a self-contained implementation that can be trained independently on the server.

---

## Baseline Results (for comparison)

| Dataset | MAP | NDCG@10 | P@10 | Recall@20 |
|---|---|---|---|---|
| 2021 WholeQ | 0.0882 | 0.4307 | 0.3467 | 0.1009 |
| 2022 WholeQ | 0.1112 | 0.4463 | 0.3960 | 0.1267 |

---

## Implementations

### 01 — Three-Class Training (`01_three_class_training/`)
**Paper**: TREC 2021 & 2022 Clinical Trials Track Overviews  
**Key Idea**: Replace binary (relevant/irrelevant) training with three-class (eligible/excluded/not-relevant) to teach the model to distinguish truly eligible trials from excluded ones.

**Data**: Merges `train_1196_deepseek_clean.jsonl` with TREC qrels to produce 3-class labels. Expands training from ~2K to ~35K labeled pairs.

**Training**:
```bash
cd /path/to/dataset
python implementations/01_three_class_training/prepare_three_class_data.py
python implementations/01_three_class_training/train_three_class.py
python implementations/01_three_class_training/inference_three_class.py
```
Outputs to: `output/predictions_three_class/`

---

### 02 — Structural Global Attention (`02_structural_attention/`)
**Paper**: Longformer: The Long-Document Transformer (Beltagy et al., 2020)  
**Key Idea**: Add global attention to section-header tokens in clinical trial text (e.g. "Inclusion Criteria", "Exclusion Criteria") so these critical tokens attend to the full document context.

**Training**:
```bash
python implementations/02_structural_attention/train_structural.py
python implementations/02_structural_attention/inference_structural.py
```
Outputs to: `output/predictions_structural/`

---

### 03 — GPL Pseudo-Labeling (`03_gpl_pseudo_labeling/`)
**Paper**: GPL: Generative Pseudo Labeling (Wang et al., NAACL 2022)  
**Key Idea**: Use the trained baseline TeacherReranker as a "teacher" to generate soft pseudo-labels for thousands of unlabeled (patient, trial) pairs. Train a new model on combined real + pseudo-labeled data using MarginMSE loss.

**Training**:
```bash
# Step 1: generate pseudo-labels using baseline checkpoint
python implementations/03_gpl_pseudo_labeling/generate_pseudo_labels.py

# Step 2: train on real + pseudo-labeled data
python implementations/03_gpl_pseudo_labeling/train_with_pseudo_labels.py
```
Outputs to: `output/predictions_gpl/`

---

### 04 — Score Fusion Ensemble (`04_score_fusion/`)
**Paper**: TREC 2022 top systems (RRF ensembling)  
**Key Idea**: Combine multiple TREC run files using Reciprocal Rank Fusion (RRF). No retraining needed — just run on existing ranked output files.

**Usage**:
```bash
python implementations/04_score_fusion/fuse_scores.py
```
Outputs to: `output/predictions_fused/`

---

## Evaluation

After any inference step, update `evaluate_run.py` to include the new run files, then:
```bash
python evaluate_run.py
```

---

## File Tree
```
implementations/
├── README.md
├── 01_three_class_training/
│   ├── prepare_three_class_data.py   # merges qrels into 3-class JSONL
│   ├── train_three_class.py          # trains TeacherReranker3Class
│   └── inference_three_class.py      # runs inference with 3-class model
├── 02_structural_attention/
│   ├── train_structural.py           # trains with structural global attention
│   └── inference_structural.py       # runs inference with structural attention
├── 03_gpl_pseudo_labeling/
│   ├── generate_pseudo_labels.py     # generates soft labels from teacher
│   └── train_with_pseudo_labels.py   # trains with MarginMSE on real+pseudo
└── 04_score_fusion/
    └── fuse_scores.py                # RRF / linear fusion of run files
```
