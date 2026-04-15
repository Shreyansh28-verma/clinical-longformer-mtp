# inference_structural.py
# ---------------------------------------------------------------
# Inference script for the Structural Attention TeacherReranker.
# Applies the same global attention masks to the test data.
# ---------------------------------------------------------------

import os
import sys
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
os.environ["TRANSFORMERS_TIE_WORD_EMBEDDINGS"] = "false"

# Allow importing the model class from train_structural.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_structural import TeacherRerankerStructural, STRUCTURAL_PHRASES, load_trials

# ====================== CONFIG ===========================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH  = os.path.join(BASE_DIR, "models_new", "Structural_ClinicalLongformer", "best_structural.pt")
MODEL_NAME  = "yikuan8/Clinical-Longformer"

MAX_LENGTH  = 4096
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR  = os.path.join(BASE_DIR, "output", "predictions_structural")
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


def build_global_mask(input_ids, tokenizer, structural_sequences):
    """Rebuilds the global attention mask exactly as during training"""
    global_attention_mask = torch.zeros_like(input_ids)
    
    global_attention_mask[input_ids == tokenizer.cls_token_id] = 1
    global_attention_mask[input_ids == tokenizer.sep_token_id] = 1
    global_attention_mask[input_ids == tokenizer.mask_token_id] = 1
    
    input_list = input_ids.tolist()
    for seq in structural_sequences:
        seq_len = len(seq)
        if seq_len == 0: continue
        
        for i in range(len(input_list) - seq_len + 1):
            if input_list[i : i+seq_len] == seq:
                global_attention_mask[i : i+seq_len] = 1
                
    return global_attention_mask


def run_inference(dataset_name, queries_file, reasonings_file, trials, model, tokenizer, structural_sequences):
    print(f"\n🚀 Inference: {dataset_name}")
    out_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_structural_run.txt")
    
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
            attention_mask = enc["attention_mask"]
            
            # Apply identical global mask builder
            global_mask = build_global_mask(input_ids.squeeze(0), tokenizer, structural_sequences).unsqueeze(0).to(DEVICE)
            
            mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if mask_positions.numel() == 0:
                continue
            mask_idx = mask_positions[0].item()

            score_logit = model(
                input_ids, attention_mask, global_mask,
                torch.tensor([mask_idx], device=DEVICE).long()
            ).item()
            
            score = torch.sigmoid(torch.tensor(score_logit)).item()
            trec_entries.append((topic_id, trial_id, score))

    unique_topics = sorted(set(t[0] for t in trec_entries), key=lambda x: int(x))

    with open(out_file, "w", encoding="utf-8") as f:
        for topic_id in unique_topics:
            topic_entries = [(t[1], t[2]) for t in trec_entries if t[0] == topic_id]
            topic_entries.sort(key=lambda x: x[1], reverse=True)
            for rank, (trial_id, score) in enumerate(topic_entries, start=1):
                f.write(f"{topic_id} Q0 {trial_id} {rank} {score:.6f} StructuralLongformer\n")

    print(f"✅ Saved to: {os.path.basename(out_file)}")


def main():
    print("========== STRUCTURAL ATTENTION INFERENCE ==========")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Run train_structural.py first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Pre-tokenize the structural phrases
    structural_sequences = []
    for phrase in STRUCTURAL_PHRASES:
        ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        if ids:
            structural_sequences.append(ids)
            
    model = TeacherRerankerStructural(tokenizer).to(DEVICE)
    
    print(f"Loading weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    trials = load_trials(TRIALS_JSONL)
    print(f"Loaded {len(trials):,} trials.")

    for dataset_name, files in DATASETS.items():
        run_inference(dataset_name, files["queries"], files["reasonings"], trials, model, tokenizer, structural_sequences)


if __name__ == "__main__":
    main()
