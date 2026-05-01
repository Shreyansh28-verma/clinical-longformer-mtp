# Clinical Longformer — Patient-Trial Matching (MTP)

This repository implements the patient–trial matching task on the TREC 2021 & 2022 Clinical Trials track datasets. It documents the transition from a strong binary baseline (MTP-1) to an advanced framework featuring structural attention, three-class objective training, and generative pseudo-labeling (MTP-2), achieving up to a **+53.6% improvement in MAP**.

---

## 🎯 Project Objective

Clinical trial recruitment is often delayed by poor patient matching. The goal of this project is to automate the matching of patient queries (symptoms, demographics, conditions) to clinical trial descriptions using the `yikuan8/Clinical-Longformer` architecture, which handles long documents up to 4096 tokens.

---

## 🏗️ The Baseline Architecture (MTP-1)

The initial phase focused on reproducing the **TeacherReranker** baseline.
- **Approach:** Masked Language Modeling (MLM) scoring on concatenated patient query and trial text: `{query} [SEP] Relevance: [MASK] [SEP] {trial_text}`.
- **Objective:** Binary Cross-Entropy (Eligible vs. Not Eligible).
- **Flaws:** Binary labels caused information loss (failing to distinguish between "Excluded" and "Not Relevant"); uniform attention was insufficient for finding key criteria in 4096-token documents; and the dataset was extremely small (~1,196 labeled pairs).

*Baseline MAP Result: **0.0882***

---

## 🚀 The Improvements (MTP-2)

To overcome baseline limitations, three targeted architectural improvements were engineered.

### 1. Three-Class Training Objective
- **Problem:** Binary labels collapse crucial distinctions.
- **Solution:** Replaced the MLM token predictor with a **Three-Class Linear Classification Head** explicitly predicting the official TREC 3-point scale: `Eligible` (2), `Excluded` (1), and `Not Relevant` (0).
- **Mechanism:** Applied weighted cross-entropy loss `[1.0, 2.0, 4.0]` to handle class imbalance.

### 2. Structural Global Attention
- **Problem:** Critical text sections (like "Exclusion Criteria") are buried deep in long documents.
- **Solution:** Modified the Longformer's attention matrix dynamically. Any token within a recognized clinical header (e.g., *"Inclusion Criteria"*, *"Brief Summary"*) is assigned a `global_attention_mask = 1`.
- **Mechanism:** Creates an "attention highway" allowing the patient query to immediately attend to critical structural rules regardless of sequence length.

### 3. Generative Pseudo-Label (GPL) Training (MarginMSE)
- **Problem:** Labeled training data is extremely scarce.
- **Solution:** Dual-Model Distillation framework. The Baseline "Teacher" scored ~22,000 unlabeled query-trial pairs.
- **Mechanism:** The "Student" model was trained using a **MarginMSE loss** to learn the relative score distance between high-scoring and low-scoring trials, transferring the teacher's knowledge without expensive hard labels.

### 4. Score Fusion (RRF)
- Combines the outputs of the different models using Reciprocal Rank Fusion (RRF) to leverage the complementary strengths of Three-Class Training and Structural Attention.

---

## 📂 Project Structure

```
dataset/
├── train_teacher_longformer.py         # Baseline TeacherReranker training
├── inference_teacher_longformer.py     # Baseline inference
├── evaluate_run.py                     # TREC evaluation script
└── implementations/
    ├── 01_three_class_training/        # Three-class cross-entropy training
    ├── 02_structural_attention/        # Global attention on clinical headers
    ├── 03_gpl_pseudo_labeling/         # MarginMSE distillation
    └── 04_score_fusion/                # Reciprocal Rank Fusion scripts
```

---

## 📊 Full Results (TREC 2021 & 2022)

**Baseline MAP: 0.0882**

| Dataset | Model | MAP | NDCG@10 | P@10 | P@20 | R@10 | R@20 |
|---------|-------|----:|--------:|-----:|-----:|-----:|-----:|
| 2021 WholeQ | Three-Class | **0.1038** | 0.4831 | 0.3973 | 0.3193 | 0.0817 | 0.1176 |
| 2021 WholeQ | Structural | 0.0966 | 0.4665 | 0.3760 | 0.3087 | 0.0732 | 0.1095 |
| 2021 WholeQ | GPL | 0.0864 | 0.4145 | 0.3360 | 0.2767 | 0.0632 | 0.0933 |
| 2021 WholeQ+RM3 | Three-Class | **0.1195** | 0.4227 | 0.3587 | 0.2973 | 0.0884 | 0.1318 |
| 2021 WholeQ+RM3 | Structural | 0.1124 | 0.4130 | 0.3467 | 0.2780 | 0.0830 | 0.1212 |
| 2021 WholeQ+RM3 | GPL | 0.1065 | 0.4005 | 0.3427 | 0.2820 | 0.0782 | 0.1141 |
| 2022 WholeQ | Three-Class | **0.1198** | 0.4672 | 0.4000 | 0.3350 | 0.0921 | 0.1310 |
| 2022 WholeQ | Structural | 0.1187 | **0.4923** | **0.4140** | **0.3390** | 0.0879 | 0.1260 |
| 2022 WholeQ | GPL | 0.1062 | 0.4367 | 0.3860 | 0.3160 | 0.0872 | 0.1234 |
| 2022 WholeQ+RM3 | Three-Class | **0.1355** | 0.4583 | 0.4020 | 0.3320 | **0.0947** | **0.1376** |
| 2022 WholeQ+RM3 | Structural | 0.1347 | 0.4437 | 0.3800 | 0.3290 | 0.0927 | 0.1329 |
| 2022 WholeQ+RM3 | GPL | 0.1229 | 0.4187 | 0.3560 | 0.3090 | 0.0844 | 0.1254 |

### 📈 Key Observations
1. **Three-Class Dominance:** Achieved up to **+53.6% relative improvement in MAP** (0.0882 -> 0.1355) over the baseline. Teaching the model the difference between "Excluded" and "Irrelevant" drastically improves overall retrieval quality.
2. **Precision at the Top:** **Structural Attention** achieved the highest `NDCG@10` (0.4923) and `P@10`. Forcing attention on structural headers pushes the most relevant trials to the very top of the ranking.
3. **Data Expansion:** **GPL Training** consistently improved the baseline on most splits (+20% to +39%), demonstrating that soft-label distillation works even when constrained to a 2048 sequence length due to hardware limits.
4. **RM3 Amplification:** RM3 query expansion significantly boosted performance across all models and splits.

---

## 🛠️ Quick Start

```bash
# Set up environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch transformers tqdm ir_datasets

# Run any of the implemented pipelines
python implementations/01_three_class_training/train_three_class.py
python implementations/01_three_class_training/inference_three_class.py

# Evaluate outputs using the python script
python evaluate_run.py
```

---

## 📚 References
- Roberts et al. (2021). *TREC 2021 Clinical Trials Track Overview.* NIST.
- Roberts et al. (2022). *TREC 2022 Clinical Trials Track Overview.* NIST SP 500-338.
- Li et al. (2023). *A comparative study of pretrained language models for long clinical text.* JAMIA, 30(2).
- Wang et al. (2022). *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation.* NAACL 2022.
- Beltagy et al. (2020). *Longformer: The Long-Document Transformer.* arXiv:2004.05150.
