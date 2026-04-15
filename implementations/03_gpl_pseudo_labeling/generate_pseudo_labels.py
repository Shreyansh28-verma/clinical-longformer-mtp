# generate_pseudo_labels.py
# ---------------------------------------------------------------
# Generates soft pseudo-labels using the trained baseline
# TeacherReranker. This expands our training data from ~2.3K
# labeled pairs to ~22K (topic, trial) pairs extracted from
# the WHOLEQ inference files.
# ---------------------------------------------------------------

import os
import sys

# Allow importing the model class from the original project directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Import baseline model
from train_teacher_longformer import TeacherReranker, load_trials

# ====================== CONFIG ===========================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Baseline trained model (from previous conversation)
MODEL_PATH  = os.path.join(BASE_DIR, "models_new", "Teacher_ClinicalLongformer_1196", "alpha0.2", "best_teacher_alpha0.2.pt")
MODEL_NAME  = "yikuan8/Clinical-Longformer"

MAX_LENGTH  = 4096
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_FILE    = os.path.join(OUTPUT_DIR, "pseudo_labels.jsonl")

# We will generate pseudo-labels for all pairs in these files
WHOLEQ_FILES = [
    os.path.join(BASE_DIR, "WholeQ_RETRIEVAL_T2021_deepseek_clean.jsonl"),
    os.path.join(BASE_DIR, "WholeQ_RM3_RETRIEVAL_T2021_deepseek_clean.jsonl"),
    os.path.join(BASE_DIR, "WholeQ_RETRIEVAL_T2022_deepseek_clean.jsonl"),
    os.path.join(BASE_DIR, "WholeQ_RM3_RETRIEVAL_T2022_deepseek_clean.jsonl"),
]
QUERIES_TSV_FILES = [
    os.path.join(BASE_DIR, "ct_2021_queries.tsv"),
    os.path.join(BASE_DIR, "ct_2022_queries.tsv"),
    os.path.join(BASE_DIR, "synthetic_gold_queries.tsv")
]
TRIALS_JSONL = os.path.join(BASE_DIR, "concatenated_trials.jsonl")


def load_all_queries():
    queries = {}
    for tsv_path in QUERIES_TSV_FILES:
        if not os.path.exists(tsv_path): continue
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                topic_id, text = line.strip().split("\t", 1)
                queries[topic_id] = text
    return queries


def load_all_candidate_pairs():
    """Extracts all unique (topic_id, trial_id, reasoning) tuples from inference files"""
    candidates = {}
    for file_path in WHOLEQ_FILES:
        if not os.path.exists(file_path): continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                key = (str(obj["topic_id"]), str(obj["trial_id"]))
                if key not in candidates:
                    candidates[key] = obj.get("reasoning", "")
    return candidates


def main():
    print("========== GPL: GENERATE PSEUDO-LABELS ==========")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Baseline Teacher model not found at {MODEL_PATH}")
        return

    queries = load_all_queries()
    trials = load_trials(TRIALS_JSONL)
    candidates = load_all_candidate_pairs()
    
    print(f"Loaded {len(queries)} queries, {len(trials):,} trials.")
    print(f"Total candidate pairs to score: {len(candidates):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TeacherReranker(tokenizer).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    pseudo_labeled_data = []

    with torch.no_grad():
        for (topic_id, trial_id), reasoning in tqdm(candidates.items(), desc="Generating labels"):
            if topic_id not in queries or trial_id not in trials:
                continue

            query = queries[topic_id]
            trial_text = trials[trial_id]
            q_prompt = f"{query} {tokenizer.sep_token} Relevance: {tokenizer.mask_token}"
            second_text = f"{trial_text} {tokenizer.sep_token} Reasoning: {reasoning}" if reasoning else trial_text

            enc = tokenizer(
                q_prompt, second_text,
                truncation="only_second", padding="max_length", max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)

            input_ids = enc["input_ids"]
            mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if mask_positions.numel() == 0: continue
            mask_idx = mask_positions[0].item()

            score_logit = model(
                enc["input_ids"], enc["attention_mask"], 
                torch.tensor([mask_idx], device=DEVICE).long()
            ).item()
            
            # Map logit to [0,1] score
            score = torch.sigmoid(torch.tensor(score_logit)).item()
            
            pseudo_labeled_data.append({
                "topic_id": topic_id,
                "trial_id": trial_id,
                "reasoning": reasoning,
                "score": score
            })

    # Save to JSONL
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for item in pseudo_labeled_data:
            f.write(json.dumps(item) + "\n")

    print(f"\n✅ Saved {len(pseudo_labeled_data):,} pseudo-labeled pairs to {OUT_FILE}")


if __name__ == "__main__":
    main()
