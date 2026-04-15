# inference_three_class.py
# ---------------------------------------------------------------
# Inference script for the Three-Class TeacherReranker.
# Generates TREC-style ranked results for the 4 WHOLEQ datasets.
# ---------------------------------------------------------------

import os
import sys
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
os.environ["TRANSFORMERS_TIE_WORD_EMBEDDINGS"] = "false"

# Allow importing the model class from train_three_class.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_three_class import TeacherReranker3Class, load_trials

# ====================== CONFIG ===========================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH  = os.path.join(BASE_DIR, "models_new", "ThreeClass_ClinicalLongformer", "best_three_class.pt")
MODEL_NAME  = "yikuan8/Clinical-Longformer"

MAX_LENGTH  = 4096
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR  = os.path.join(BASE_DIR, "output", "predictions_three_class")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = {
    "2021_wholeq": {
        "queries": os.path.join(BASE_DIR, "ct_2021_queries.tsv"),
        "reasonings": os.path.join(BASE_DIR, "WholeQ_RETRIEVAL_T2021_deepseek_clean.jsonl"),
    },
    "2021_wholeq_rm3": {
        "queries": os.path.join(BASE_DIR, "ct_2021_queries.tsv"),
        "reasonings": os.path.join(BASE_DIR, "WholeQ_RM3_RETRIEVAL_T2021_deepseek_clean.jsonl"),
    },
    "2022_wholeq": {
        "queries": os.path.join(BASE_DIR, "ct_2022_queries.tsv"),
        "reasonings": os.path.join(BASE_DIR, "WholeQ_RETRIEVAL_T2022_deepseek_clean.jsonl"),
    },
    "2022_wholeq_rm3": {
        "queries": os.path.join(BASE_DIR, "ct_2022_queries.tsv"),
        "reasonings": os.path.join(BASE_DIR, "WholeQ_RM3_RETRIEVAL_T2022_deepseek_clean.jsonl"),
    }
}
TRIALS_JSONL = os.path.join(BASE_DIR, "concatenated_trials.jsonl")


def load_queries(tsv_path):
    queries = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            topic_id, text = line.strip().split("\t", 1)
            queries[topic_id] = text
    return queries


def run_inference(dataset_name, queries_file, reasonings_file, trials, model, tokenizer):
    print(f"\n🚀 Inference: {dataset_name}")
    out_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_three_class_run.txt")
    
    queries = load_queries(queries_file)
    with open(reasonings_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    trec_entries = []
    
    with torch.no_grad():
        for obj in tqdm(data, desc=f"Evaluating {dataset_name}"):
            topic_id  = str(obj["topic_id"])
            trial_id  = str(obj["trial_id"])
            reasoning = obj.get("reasoning", "")
            trial_text = trials.get(trial_id, "")

            if topic_id not in queries:
                continue

            query = queries[topic_id]
            q_prompt = f"{query} {tokenizer.sep_token} Relevance: {tokenizer.mask_token}"
            second_text = f"{trial_text} {tokenizer.sep_token} Reasoning: {reasoning}" if reasoning else trial_text

            enc = tokenizer(
                q_prompt, second_text,
                truncation="only_second", padding="max_length", max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)

            input_ids = enc["input_ids"]
            mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if mask_positions.numel() == 0:
                continue
            mask_idx = mask_positions[0].item()

            score_logit = model(
                enc["input_ids"], enc["attention_mask"], 
                torch.tensor([mask_idx], device=DEVICE).long()
            ).item()
            
            # Use sigmoid to map to (0, 1) range for TREC formatted scores
            score = torch.sigmoid(torch.tensor(score_logit)).item()
            trec_entries.append((topic_id, trial_id, score))

    # Sort numerically by topic_id, then desc by score
    unique_topics = sorted(set(t[0] for t in trec_entries), key=lambda x: int(x))

    with open(out_file, "w", encoding="utf-8") as f:
        for topic_id in unique_topics:
            topic_entries = [(t[1], t[2]) for t in trec_entries if t[0] == topic_id]
            topic_entries.sort(key=lambda x: x[1], reverse=True)
            for rank, (trial_id, score) in enumerate(topic_entries, start=1):
                f.write(f"{topic_id} Q0 {trial_id} {rank} {score:.6f} ThreeClassLongformer\n")

    print(f"✅ Saved to: {os.path.basename(out_file)}")


def main():
    print("========== THREE-CLASS INFERENCE ==========")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Run train_three_class.py first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TeacherReranker3Class(tokenizer).to(DEVICE)
    
    print(f"Loading weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    trials = load_trials(TRIALS_JSONL)
    print(f"Loaded {len(trials):,} trials.")

    for dataset_name, files in DATASETS.items():
        run_inference(dataset_name, files["queries"], files["reasonings"], trials, model, tokenizer)


if __name__ == "__main__":
    main()
