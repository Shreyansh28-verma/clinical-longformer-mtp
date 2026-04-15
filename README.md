# Clinical Longformer — Patient-Trial Matching

Improvements to the Clinical Longformer model for patient–trial matching, evaluated on the TREC 2021 & 2022 Clinical Trials datasets.

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
    │   └── inference_three_class.py    # Generates three_class_run.txt files
    ├── 02_structural_attention/
    │   ├── train_structural.py         # Global attention on section headers
    │   └── inference_structural.py     # Generates structural_run.txt files
    ├── 03_gpl_pseudo_labeling/
    │   ├── generate_pseudo_labels.py   # Teacher scoring → soft pseudo labels
    │   └── train_with_pseudo_labels.py # MarginMSE training
    └── 04_score_fusion/
        └── fuse_scores.py              # Reciprocal Rank Fusion ensemble
```

## 🔬 Implementations

| # | Method | Key Idea | Reference |
|---|--------|----------|-----------|
| 1 | **Three-Class Training** | Distinguish eligible / excluded / not-relevant at [MASK] | Roberts et al. (2021, 2022) |
| 2 | **Structural Global Attention** | Attend globally to section headers ("Inclusion Criteria", etc.) | Li et al. (2023) |
| 3 | **GPL Pseudo-Labeling** | Expand training data with soft teacher labels (MarginMSE) | Wang et al. (2022) |
| 4 | **Score Fusion (RRF)** | Ensemble multiple reranker outputs | — |

## 🚀 Quick Start

### Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch transformers tqdm ir_datasets ir_measures
```

### Running the Pipeline

```bash
# 1. Prepare three-class data
python implementations/01_three_class_training/prepare_three_class_data.py

# 2. Train & infer (three-class model)
python implementations/01_three_class_training/train_three_class.py
python implementations/01_three_class_training/inference_three_class.py

# 3. Train & infer (structural model)
python implementations/02_structural_attention/train_structural.py
python implementations/02_structural_attention/inference_structural.py

# 4. Evaluate
python evaluate_run.py
```

## 📊 Results

Evaluated on **TREC 2021 Clinical Trials** (2021 qrels from `ir_datasets`, 2022 qrels from local `qrels2022.txt`).

| Model | MAP | NDCG@10 | P@10 | P@20 |
|-------|-----|---------|------|------|
| Baseline (TeacherLongformer) | 0.0882 | — | — | — |
| Three-Class | *see evaluation_results.json* | | | |
| Structural Attention | *see evaluation_results.json* | | | |

## 📚 References

- Roberts et al. (2021). *TREC 2021 Clinical Trials Track Overview.* NIST.
- Roberts et al. (2022). *TREC 2022 Clinical Trials Track Overview.* NIST SP 500-338.
- Li et al. (2023). *A comparative study of pretrained language models for long clinical text.* JAMIA, 30(2). DOI: 10.1093/jamia/ocac225.
- Wang et al. (2022). *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation.* NAACL 2022.
- Pradeep et al. (2022). *Zero-shot Ranking for Clinical Trial Matching via Neural Query Synthesis.* UWaterloo.
