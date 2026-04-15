# ==========================================================
# evaluate_run.py
# ----------------------------------------------------------
# Evaluates TeacherLongformer TREC-style run files against
# official TREC 2021 & 2022 Clinical Trials qrels.
# Uses ir_datasets to fetch qrels. Metrics (MAP, NDCG@10,
# P@10, Recall@100) are computed manually for Py3.8 compat.
# ==========================================================
from __future__ import annotations
import math
import json
import os
import ir_datasets

# ====================== CONFIG ===========================
PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "predictions_teacher_reasoning")

RUNS = {
    "2021_wholeq": {
        "run_file": os.path.join(PREDICTIONS_DIR, "2021_wholeq_teacher_run.txt"),
        "dataset_id": "clinicaltrials/2021/trec-ct-2021",
    },
    "2021_wholeq_rm3": {
        "run_file": os.path.join(PREDICTIONS_DIR, "2021_wholeq_rm3_teacher_run.txt"),
        "dataset_id": "clinicaltrials/2021/trec-ct-2021",
    },
    "2022_wholeq": {
        "run_file": os.path.join(PREDICTIONS_DIR, "2022_wholeq_teacher_run.txt"),
        "qrels_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "qrels2022.txt"),
    },
    "2022_wholeq_rm3": {
        "run_file": os.path.join(PREDICTIONS_DIR, "2022_wholeq_rm3_teacher_run.txt"),
        "qrels_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "qrels2022.txt"),
    },
}

# TREC CT relevance: 0=not-relevant, 1=excluded (treated as not-relevant), 2=eligible (relevant)
RELEVANT_THRESHOLD = 2  # Only "eligible" trials count as relevant


# ====================== METRIC FUNCTIONS ==================

def precision_at_k(ranked_docs, qrels_for_query, k=10):
    """Precision@K: fraction of top-k docs that are relevant."""
    top_k = ranked_docs[:k]
    relevant = sum(1 for doc_id, _ in top_k if qrels_for_query.get(doc_id, 0) >= RELEVANT_THRESHOLD)
    return relevant / k


def recall_at_k(ranked_docs, qrels_for_query, k=100):
    """Recall@K: fraction of all relevant docs found in top-k."""
    total_relevant = sum(1 for rel in qrels_for_query.values() if rel >= RELEVANT_THRESHOLD)
    if total_relevant == 0:
        return 0.0
    top_k = ranked_docs[:k]
    found = sum(1 for doc_id, _ in top_k if qrels_for_query.get(doc_id, 0) >= RELEVANT_THRESHOLD)
    return found / total_relevant


def average_precision(ranked_docs, qrels_for_query):
    """Average Precision (AP) for a single query."""
    total_relevant = sum(1 for rel in qrels_for_query.values() if rel >= RELEVANT_THRESHOLD)
    if total_relevant == 0:
        return 0.0
    cum_relevant = 0
    cum_precision = 0.0
    for rank, (doc_id, _) in enumerate(ranked_docs, 1):
        if qrels_for_query.get(doc_id, 0) >= RELEVANT_THRESHOLD:
            cum_relevant += 1
            cum_precision += cum_relevant / rank
    return cum_precision / total_relevant


def dcg_at_k(ranked_docs, qrels_for_query, k=10):
    """DCG@K using the formula: sum( (2^rel - 1) / log2(i+1) ) for i=1..k"""
    dcg = 0.0
    for i, (doc_id, _) in enumerate(ranked_docs[:k]):
        rel = qrels_for_query.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because i is 0-indexed, log2(1+1)=1
    return dcg


def ndcg_at_k(ranked_docs, qrels_for_query, k=10):
    """NDCG@K: DCG@K / ideal DCG@K."""
    actual_dcg = dcg_at_k(ranked_docs, qrels_for_query, k)
    # Ideal: sort all judged docs by relevance descending
    ideal_docs = sorted(qrels_for_query.items(), key=lambda x: x[1], reverse=True)
    ideal_dcg = dcg_at_k([(did, None) for did, _ in ideal_docs], qrels_for_query, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


# ====================== DATA LOADERS =====================

def load_qrels_from_ir_datasets(dataset_id):
    """Load qrels from ir_datasets: {qid: {docid: rel}}"""
    dataset = ir_datasets.load(dataset_id)
    qrels = {}
    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        did = str(qrel.doc_id)
        rel = int(qrel.relevance)
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = rel
    return qrels


def load_qrels_from_file(qrels_path):
    """Load qrels from a standard TREC qrels file: {qid: {docid: rel}}"""
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, rel = parts[:4]
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = int(rel)
    return qrels


def parse_trec_run(run_file):
    """Parse TREC run file: {qid: [(docid, score), ...]} sorted by score desc."""
    run = {}
    with open(run_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, doc_id, rank, score, _ = parts[:6]
            if qid not in run:
                run[qid] = []
            run[qid].append((doc_id, float(score)))
    # Make sure sorted by score descending
    for qid in run:
        run[qid].sort(key=lambda x: x[1], reverse=True)
    return run


# ====================== MAIN =============================

def main():
    all_results = {}
    qrels_cache = {}

    for run_name, config in RUNS.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {run_name}")
        print(f"{'='*60}")

        run_file = config["run_file"]

        if not os.path.exists(run_file):
            print(f"  Warning: Run file not found: {run_file}")
            continue

        # Load qrels (cached)
        dataset_id = config.get("dataset_id")
        qrels_file = config.get("qrels_file")
        cache_key = dataset_id or qrels_file
        if cache_key not in qrels_cache:
            if qrels_file:
                print(f"  Loading qrels from file: {os.path.basename(qrels_file)}")
                qrels_cache[cache_key] = load_qrels_from_file(qrels_file)
            else:
                print(f"  Loading qrels from: {dataset_id}")
                qrels_cache[cache_key] = load_qrels_from_ir_datasets(dataset_id)
        qrels = qrels_cache[cache_key]
        total_judgments = sum(len(v) for v in qrels.values())
        print(f"  Loaded {total_judgments} qrel judgments across {len(qrels)} topics")

        # Parse run
        run = parse_trec_run(run_file)
        total_entries = sum(len(v) for v in run.values())
        print(f"  Loaded {total_entries} run entries across {len(run)} topics")

        # Only evaluate on topics that have qrels
        eval_topics = [qid for qid in run if qid in qrels]
        print(f"  Evaluating on {len(eval_topics)} overlapping topics")

        # Compute per-query metrics
        maps, ndcgs = [], []
        p10s, p20s = [], []
        r10s, r20s = [], []
        for qid in eval_topics:
            ranked = run[qid]
            q_qrels = qrels[qid]
            maps.append(average_precision(ranked, q_qrels))
            ndcgs.append(ndcg_at_k(ranked, q_qrels, k=10))
            p10s.append(precision_at_k(ranked, q_qrels, k=10))
            p20s.append(precision_at_k(ranked, q_qrels, k=20))
            r10s.append(recall_at_k(ranked, q_qrels, k=10))
            r20s.append(recall_at_k(ranked, q_qrels, k=20))

        n = len(eval_topics) if eval_topics else 1
        results = {
            "MAP": round(sum(maps) / n, 4),
            "NDCG@10": round(sum(ndcgs) / n, 4),
            "P@10": round(sum(p10s) / n, 4),
            "P@20": round(sum(p20s) / n, 4),
            "Recall@10": round(sum(r10s) / n, 4),
            "Recall@20": round(sum(r20s) / n, 4),
        }

        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

        all_results[run_name] = results

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Results saved to: {output_path}")

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'Dataset':<25} {'MAP':>8} {'NDCG@10':>10} {'P@10':>8} {'P@20':>8} {'R@10':>8} {'R@20':>8}")
    print(f"{'-'*100}")
    for rn, m in all_results.items():
        print(f"{rn:<25} {m['MAP']:>8.4f} {m['NDCG@10']:>10.4f} {m['P@10']:>8.4f} {m['P@20']:>8.4f} {m['Recall@10']:>8.4f} {m['Recall@20']:>8.4f}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
