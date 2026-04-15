# fuse_scores.py
# ---------------------------------------------------------------
# Combines multiple TREC run files into a single ensembled run
# using Reciprocal Rank Fusion (RRF).
#
# RRF score(d) = Σ (1 / (k + rank_i(d)))
#
# Usage:
#   python fuse_scores.py \
#       --runs run1.txt run2.txt ... \
#       --output fused_run.txt \
#       --k 60
# ---------------------------------------------------------------

import os
import argparse
from collections import defaultdict

# Default runs if none provided
DEFAULT_RUNS = [
    "output/predictions_teacher_reasoning/2021_wholeq_teacher_run.txt",
    "output/predictions_three_class/2021_wholeq_three_class_run.txt"
]

def load_trec_run(filepath):
    """Loads a TREC run file. Returns dict: {topic_id: [(doc_id, score, rank)]}"""
    run_data = defaultdict(list)
    print(f"Loading {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"  Warning: File not found: {filepath}")
        return run_data

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                topic_id, _, doc_id, rank, score, run_name = parts[:6]
                run_data[topic_id].append({
                    "doc_id": doc_id,
                    "rank": int(rank),
                    "score": float(score)
                })
                
    # Ensure ranked correctly internally
    for topic_id in run_data:
        run_data[topic_id].sort(key=lambda x: x["rank"])
        
    return run_data


def reciprocal_rank_fusion(runs, k=60):
    """
    Computes RRF across multiple runs.
    runs: list of run dicts from load_trec_run
    Returns a unified run dict: {topic_id: [(doc_id, rrf_score)]}
    """
    fused = defaultdict(lambda: defaultdict(float))
    
    # Collect all topics that exist in any run
    all_topics = set()
    for run in runs:
        all_topics.update(run.keys())
        
    for topic_id in all_topics:
        for run in runs:
            if topic_id not in run: continue
            
            # Add RRF score for each doc in this run
            for item in run[topic_id]:
                doc_id = item["doc_id"]
                rank = item["rank"]
                
                # RRF formula
                rrf_score = 1.0 / (k + rank)
                fused[topic_id][doc_id] += rrf_score
                
    # Sort and format output
    final_fused = defaultdict(list)
    for topic_id, doc_scores in fused.items():
        # Sort by RRF score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        final_fused[topic_id] = sorted_docs
        
    return final_fused


def write_trec_run(fused_run, output_path, run_name="Fused_RRF"):
    """Writes the fused run dict to a TREC-formatted file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Sort topics numerically
    sorted_topics = sorted(fused_run.keys(), key=lambda x: int(x))
    
    with open(output_path, "w", encoding="utf-8") as f:
        for topic_id in sorted_topics:
            for rank, (doc_id, score) in enumerate(fused_run[topic_id], start=1):
                f.write(f"{topic_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")
                
    print(f"✅ Saved fused run to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fuse TREC runs using RRF")
    parser.add_argument("--runs", nargs="+", help="Paths to TREC run files to fuse", default=[])
    parser.add_argument("--output", default="output/predictions_fused/fused_run.txt", help="Output file path")
    parser.add_argument("--k", type=int, default=60, help="RRF k parameter (default 60)")
    
    args = parser.parse_args()
    
    run_paths = args.runs if args.runs else DEFAULT_RUNS
    run_paths = [os.path.abspath(p) for p in run_paths]
    
    print("========== SCORE FUSION (RRF) ==========")
    print(f"RRF parameter k = {args.k}")
    
    loaded_runs = []
    for path in run_paths:
        r = load_trec_run(path)
        if r: loaded_runs.append(r)
        
    if not loaded_runs:
        print("❌ No valid runs loaded. Exiting.")
        return
        
    print(f"\nFusing {len(loaded_runs)} runs...")
    fused = reciprocal_rank_fusion(loaded_runs, k=args.k)
    
    write_trec_run(fused, args.output)


if __name__ == "__main__":
    main()
