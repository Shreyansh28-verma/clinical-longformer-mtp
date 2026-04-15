# prepare_three_class_data.py
# ---------------------------------------------------------------
# Merges train_1196_deepseek_clean.jsonl with TREC 2021 & 2022
# qrels to produce a 3-class labeled dataset:
#   label 0 = not-relevant (qrel grade 0)
#   label 1 = excluded     (qrel grade 1)
#   label 2 = eligible     (qrel grade 2)
#
# Steps:
#   1. Load all TREC qrels (2021 via ir_datasets, 2022 via file)
#   2. Load train_1196 JSONL (has reasoning + binary relevance)
#   3. Load WholeQ inference JSONL files (all 4 configs)
#   4. For every (topic_id, trial_id) pair in any of these files:
#      - if a qrel judgment exists → use the qrel grade (0/1/2)
#      - else if binary label exists → map Relevant→2, Non-Relevant→0
#   5. Save to three_class_train.jsonl
# ---------------------------------------------------------------

import os
import sys
import json
import argparse
import ir_datasets
from collections import defaultdict

# ---- PATHS (relative to dataset/ working directory) ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # dataset/
TRAIN_JSONL      = os.path.join(BASE_DIR, "train_1196_deepseek_clean.jsonl")
QRELS_2022_FILE  = os.path.join(BASE_DIR, "qrels2022.txt")
WHOLEQ_FILES     = [
    os.path.join(BASE_DIR, "WholeQ_RETRIEVAL_T2021_deepseek_clean.jsonl"),
    os.path.join(BASE_DIR, "WholeQ_RM3_RETRIEVAL_T2021_deepseek_clean.jsonl"),
    os.path.join(BASE_DIR, "WholeQ_RETRIEVAL_T2022_deepseek_clean.jsonl"),
    os.path.join(BASE_DIR, "WholeQ_RM3_RETRIEVAL_T2022_deepseek_clean.jsonl"),
]
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # implementations/01_three_class_training/


def load_qrels_ir_datasets(dataset_id: str) -> dict:
    """Load TREC qrels via ir_datasets → {(topic_id, trial_id): grade}"""
    print(f"  Loading qrels from ir_datasets: {dataset_id}")
    dataset = ir_datasets.load(dataset_id)
    qrels = {}
    for q in dataset.qrels_iter():
        qrels[(str(q.query_id), str(q.doc_id))] = int(q.relevance)
    print(f"    -> {len(qrels):,} judgments loaded")
    return qrels


def load_qrels_file(path: str) -> dict:
    """Load TREC qrels file → {(topic_id, trial_id): grade}"""
    print(f"  Loading qrels from file: {os.path.basename(path)}")
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, did, rel = parts[:4]
                qrels[(qid, did)] = int(rel)
    print(f"    -> {len(qrels):,} judgments loaded")
    return qrels


def load_jsonl_records(path: str) -> list:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def binary_to_grade(relevance_str: str) -> int:
    """Map binary label string to grade: Relevant→2, anything else→0"""
    return 2 if str(relevance_str).strip().lower() == "relevant" else 0


def main(args):
    print("=" * 60)
    print("  Prepare Three-Class Training Data")
    print("=" * 60)

    # ---- 1. Load all qrels ----
    print("\n[Step 1] Loading TREC qrels...")
    qrels = {}
    try:
        qrels.update(load_qrels_ir_datasets("clinicaltrials/2021/trec-ct-2021"))
    except Exception as e:
        print(f"  Warning: Could not load 2021 qrels from ir_datasets: {e}")
    qrels.update(load_qrels_file(QRELS_2022_FILE))
    print(f"  Total combined qrel judgments: {len(qrels):,}")

    # ---- 2. Load all source records (train + WholeQ files) ----
    print("\n[Step 2] Loading source JSONL records...")
    all_records: dict = {}  # key: (topic_id, trial_id) → record dict

    # Load training data (has reasoning)
    train_records = load_jsonl_records(TRAIN_JSONL)
    for r in train_records:
        key = (str(r["topic_id"]), str(r["trial_id"]))
        all_records[key] = {
            "topic_id": str(r["topic_id"]),
            "trial_id": str(r["trial_id"]),
            "reasoning": r.get("reasoning", ""),
            "binary_relevance": r.get("relevance", "Non-Relevant"),
        }
    print(f"  Loaded {len(train_records):,} records from train_1196_deepseek_clean.jsonl")

    # Load WholeQ inference data
    for path in WHOLEQ_FILES:
        if not os.path.exists(path):
            print(f"  Skipping missing file: {os.path.basename(path)}")
            continue
        records = load_jsonl_records(path)
        for r in records:
            key = (str(r["topic_id"]), str(r["trial_id"]))
            if key not in all_records:
                all_records[key] = {
                    "topic_id": str(r["topic_id"]),
                    "trial_id": str(r["trial_id"]),
                    "reasoning": r.get("reasoning", ""),
                    "binary_relevance": r.get("relevance", "Non-Relevant"),
                }
        print(f"  Loaded {len(records):,} records from {os.path.basename(path)}")

    print(f"  Total unique (topic, trial) pairs: {len(all_records):,}")

    # ---- 3. Assign labels ----
    print("\n[Step 3] Assigning 3-class labels...")
    output_records = []
    qrel_sourced = 0
    binary_sourced = 0
    skipped = 0

    for key, rec in all_records.items():
        if args.limit and len(output_records) >= args.limit:
            break

        if key in qrels:
            label = qrels[key]   # 0, 1, or 2 from official TREC judgment
            qrel_sourced += 1
        else:
            label = binary_to_grade(rec["binary_relevance"])  # 0 or 2 (no grade-1 from binary)
            binary_sourced += 1

        output_records.append({
            "topic_id": rec["topic_id"],
            "trial_id": rec["trial_id"],
            "reasoning": rec["reasoning"],
            "label": label,   # 0=not-relevant, 1=excluded, 2=eligible
        })

    # ---- 4. Save output ----
    suffix = "_debug" if args.limit else ""
    out_file = os.path.join(OUTPUT_DIR, f"three_class_train{suffix}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")

    # ---- Summary ----
    from collections import Counter
    label_dist = Counter(r["label"] for r in output_records)
    print(f"\n  Label distribution:")
    print(f"    0 (not-relevant): {label_dist[0]:,}")
    print(f"    1 (excluded):     {label_dist[1]:,}")
    print(f"    2 (eligible):     {label_dist[2]:,}")
    print(f"  Source: {qrel_sourced:,} from qrels, {binary_sourced:,} from binary labels")
    print(f"\n  Saved {len(output_records):,} samples to: {out_file}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare three-class training data")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit output records (for debugging)")
    args = parser.parse_args()
    main(args)
