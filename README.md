# Clinical Longformer — Patient-Trial Matching (MTP)

This repository implements improvements to the **Clinical Longformer** model for the patient–trial matching task, evaluated on the TREC 2021 & 2022 Clinical Trials track datasets.

## 📂 Project Structure

```
dataset/
├── train_teacher_longformer.py         # Baseline TeacherReranker training
├── inference_teacher_longformer.py     # Baseline inference
├── evaluate_run.py                     # TREC evaluation (MAP, NDCG@10, P@10/20, R@10/20)
└── implementations/
    ├── 01_three_class_training/
    │   ├── prepare_three_class_data.py # Merge qrels → 3-class labels
    │   ├── train_three_class.py        # Three-class cross-entropy training
    │   └── inference_three_class.py    # Generates TREC-style run files
    └── 02_structural_attention/
        ├── train_structural.py         # Global attention on clinical section headers
        └── inference_structural.py     # Generates TREC-style run files
```

## 🔬 Implemented Methods

### 1. Three-Class Training
Instead of binary relevant / not-relevant, the model is trained to distinguish three classes aligned with the official TREC grading scale:
- **Eligible** (grade 2) — patient meets inclusion criteria
- **Excluded** (grade 1) — patient is actively excluded
- **Not Relevant** (grade 0)

A weighted cross-entropy loss handles class imbalance. At inference time, the score used for ranking is the logit for the "eligible" class.

### 2. Structural Global Attention
The Longformer's global attention is extended to key structural section headers in the clinical trial text (e.g., *"Inclusion Criteria"*, *"Exclusion Criteria"*, *"Brief Summary"*). This allows the model to directly attend to the most decision-relevant sections across the full 4096-token context.

## 🚀 Quick Start

```bash
# Set up environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch transformers tqdm ir_datasets

# Three-Class Training
python implementations/01_three_class_training/prepare_three_class_data.py
python implementations/01_three_class_training/train_three_class.py
python implementations/01_three_class_training/inference_three_class.py

# Structural Attention
python implementations/02_structural_attention/train_structural.py
python implementations/02_structural_attention/inference_structural.py

# Evaluate
python evaluate_run.py
```

## 📊 Results (TREC 2021 & 2022)

Baseline MAP: **0.0882** (binary TeacherLongformer).

| Dataset | Model | MAP | NDCG@10 | P@10 | P@20 | R@10 | R@20 |
|---------|-------|----:|--------:|-----:|-----:|-----:|-----:|
| 2021 WholeQ | Three-Class | **0.1038** | 0.4831 | 0.3973 | 0.3193 | 0.0817 | 0.1176 |
| 2021 WholeQ+RM3 | Three-Class | **0.1195** | 0.4227 | 0.3587 | 0.2973 | 0.0884 | 0.1318 |
| 2022 WholeQ | Three-Class | **0.1198** | 0.4672 | 0.4000 | 0.3350 | 0.0921 | 0.1310 |
| 2022 WholeQ+RM3 | Three-Class | **0.1355** | 0.4583 | 0.4020 | 0.3320 | 0.0947 | 0.1376 |
| 2021 WholeQ | Structural | 0.0966 | 0.4665 | 0.3760 | 0.3087 | 0.0732 | 0.1095 |
| 2021 WholeQ+RM3 | Structural | 0.1124 | 0.4130 | 0.3467 | 0.2780 | 0.0830 | 0.1212 |
| 2022 WholeQ | Structural | 0.1187 | 0.4923 | 0.4140 | 0.3390 | 0.0879 | 0.1260 |
| 2022 WholeQ+RM3 | Structural | 0.1347 | 0.4437 | 0.3800 | 0.3290 | 0.0927 | 0.1329 |

> **Key takeaway:** Three-Class training achieves up to **+53% MAP improvement** over the baseline (2022 WholeQ+RM3: 0.0882 → 0.1355). Structural Attention also consistently improves over baseline across all four datasets.

## 📚 References

- Roberts et al. (2021). *TREC 2021 Clinical Trials Track Overview.* NIST.
- Roberts et al. (2022). *TREC 2022 Clinical Trials Track Overview.* NIST SP 500-338.
- Li et al. (2023). *A comparative study of pretrained language models for long clinical text.* JAMIA, 30(2).
- Wang et al. (2022). *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation.* NAACL 2022.
- Pradeep et al. (2022). *Zero-shot Ranking for Clinical Trial Matching via Neural Query Synthesis.* UWaterloo.
