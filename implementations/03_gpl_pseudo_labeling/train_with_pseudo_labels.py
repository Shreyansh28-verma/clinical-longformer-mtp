# train_with_pseudo_labels.py
# ---------------------------------------------------------------
# Trains the baseline TeacherReranker using a combination of
# hard labels (train_1196) and soft pseudo-labels (via MarginMSE).
#
# MarginMSE loss: MSE( (logit_p - logit_n), (score_p - score_n) )
# The model learns to match the score margins output by the Teacher.
# ---------------------------------------------------------------

import os
import sys

# Allow importing from parent directory (dataset/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import json
import csv
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_scheduler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.amp import autocast
import random

# Import baseline model
from train_teacher_longformer import TeacherReranker, load_trials

# ====================== CONFIG ======================
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIS_DIR      = os.path.dirname(os.path.abspath(__file__))

QUERIES_TSV_FILES = [
    os.path.join(BASE_DIR, "ct_2021_queries.tsv"),
    os.path.join(BASE_DIR, "ct_2022_queries.tsv"),
    os.path.join(BASE_DIR, "synthetic_gold_queries.tsv")
]
PSEUDO_JSONL  = os.path.join(THIS_DIR, "pseudo_labels.jsonl")      # produced by generate script
TRAIN_JSONL   = os.path.join(BASE_DIR, "train_1196_deepseek_clean.jsonl")
TRIALS_JSONL  = os.path.join(BASE_DIR, "concatenated_trials.jsonl")
MODEL_NAME    = "yikuan8/Clinical-Longformer"

MAX_LENGTH    = 4096
BATCH_SIZE    = 2      # Reduced because MarginMSE needs pairs (effectively 4 per batch)
EPOCHS        = 10
BASE_LR       = 3e-5
CLS_LR        = 6e-5
PATIENCE      = 5
SAVE_DIR      = os.path.join(BASE_DIR, "models_new", "GPL_ClinicalLongformer")


def log_message(msg: str, log_file: str):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


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


def build_margin_pairs(pseudo_jsonl, train_jsonl):
    """
    Builds positive-negative pairs for MarginMSE training.
    Groups records by topic_id, then separates into Positives (score > 0.8)
    and Negatives (score < 0.2). Creates pairs.
    """
    topic_scores = {}
    
    # 1. Load pseudo-labels
    if os.path.exists(pseudo_jsonl):
        with open(pseudo_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                tid = str(obj["topic_id"])
                if tid not in topic_scores: topic_scores[tid] = {"pos": [], "neg": []}
                
                score = float(obj["score"])
                item = (str(obj["trial_id"]), obj.get("reasoning", ""), score)
                if score >= 0.8:
                    topic_scores[tid]["pos"].append(item)
                elif score <= 0.2:
                    topic_scores[tid]["neg"].append(item)

    # 2. Add real labels as hard 1.0 / 0.0 scores
    if os.path.exists(train_jsonl):
         with open(train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                tid = str(obj["topic_id"])
                if tid not in topic_scores: topic_scores[tid] = {"pos": [], "neg": []}
                
                label = 1.0 if str(obj.get("relevance", "")).lower() == "relevant" else 0.0
                item = (str(obj["trial_id"]), obj.get("reasoning", ""), label)
                
                if label == 1.0:
                    topic_scores[tid]["pos"].append(item)
                else:
                    topic_scores[tid]["neg"].append(item)

    # 3. Create pairs
    margin_pairs = []
    for tid, groups in topic_scores.items():
        positives = groups["pos"]
        negatives = groups["neg"]
        
        # Max pairs per topic to prevent imbalance
        MAX_PAIRS_PER_TOPIC = 200
        pair_count = 0
        
        # Randomly pair each positive with a negative
        for p in positives:
            if not negatives: break
            # Choose a random hard negative for this positive
            n = random.choice(negatives)
            
            p_trial, p_reasoning, p_score = p
            n_trial, n_reasoning, n_score = n
            
            margin_pairs.append({
                "topic_id": tid,
                "pos_trial": p_trial, "pos_reasoning": p_reasoning, "pos_score": p_score,
                "neg_trial": n_trial, "neg_reasoning": n_reasoning, "neg_score": n_score,
                "margin": p_score - n_score
            })
            
            pair_count += 1
            if pair_count >= MAX_PAIRS_PER_TOPIC: break

    return margin_pairs


class MarginTrialDataset(Dataset):
    def __init__(self, pairs: list, queries: dict, trials: dict, tokenizer):
        # Filter pairs where we have both trials and the query
        self.pairs = [
            p for p in pairs 
            if p["topic_id"] in queries 
               and p["pos_trial"] in trials 
               and p["neg_trial"] in trials
        ]
        self.queries = queries
        self.trials = trials
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def _prepare_input(self, topic_id, trial_id, reasoning):
        query = self.queries[topic_id]
        trial_text = self.trials[trial_id]

        query_with_prompt = f"{query} {self.tokenizer.sep_token} Relevance: {self.tokenizer.mask_token}"
        second_text = f"{trial_text} {self.tokenizer.sep_token} Reasoning: {reasoning}" if reasoning else trial_text

        tokenized = self.tokenizer(
            query_with_prompt, second_text,
            max_length=MAX_LENGTH, padding="max_length", truncation="only_second", return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"].squeeze(0)
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        mask_idx = mask_positions[0].item() if mask_positions.numel() > 0 else 0
        
        return input_ids, tokenized["attention_mask"].squeeze(0), mask_idx

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        pos_input_ids, pos_attn_mask, pos_mask_idx = self._prepare_input(pair["topic_id"], pair["pos_trial"], pair["pos_reasoning"])
        neg_input_ids, neg_attn_mask, neg_mask_idx = self._prepare_input(pair["topic_id"], pair["neg_trial"], pair["neg_reasoning"])

        return {
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_attn_mask,
            "pos_mask_idx": torch.tensor(pos_mask_idx, dtype=torch.long),
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_attn_mask,
            "neg_mask_idx": torch.tensor(neg_mask_idx, dtype=torch.long),
            "margin": torch.tensor(pair["margin"], dtype=torch.float)
        }


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_file = os.path.join(SAVE_DIR, "training_log.txt")

    log_message("\n========== GPL PSEUDO-LABEL TRAINING (MarginMSE) ==========", log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(PSEUDO_JSONL):
        log_message(f"❌ Pseudo-labels missing: {PSEUDO_JSONL}", log_file)
        return

    queries = load_all_queries()
    trials = load_trials(TRIALS_JSONL)
    
    margin_pairs = build_margin_pairs(PSEUDO_JSONL, TRAIN_JSONL)
    log_message(f"Built {len(margin_pairs):,} margin pairs for training.", log_file)

    train_data, val_data = train_test_split(margin_pairs, test_size=0.1, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = MAX_LENGTH
    if hasattr(tokenizer, "add_prefix_space"): tokenizer.add_prefix_space = True

    train_ds = MarginTrialDataset(train_data, queries, trials, tokenizer)
    val_ds   = MarginTrialDataset(val_data, queries, trials, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = TeacherReranker(tokenizer).to(device)
    model.backbone.gradient_checkpointing_enable()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Optimizer
    backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
    lm_head_params = list(backbone.lm_head.parameters())
    lm_head_ids    = {id(p) for p in lm_head_params}
    encoder_params = [p for p in backbone.longformer.parameters() if id(p) not in lm_head_ids]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params,  "lr": BASE_LR},
        {"params": lm_head_params,  "lr": CLS_LR},
    ])
    num_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler("cosine", optimizer=optimizer,
                              num_warmup_steps=max(1, int(0.05 * num_steps)),
                              num_training_steps=num_steps)

    mse_loss_fn = nn.MSELoss()
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            optimizer.zero_grad()
            
            # Forward pass positives
            pos_logits = model(
                batch["pos_input_ids"].to(device, non_blocking=True),
                batch["pos_attention_mask"].to(device, non_blocking=True),
                batch["pos_mask_idx"].to(device, non_blocking=True)
            )
            
            # Forward pass negatives
            neg_logits = model(
                batch["neg_input_ids"].to(device, non_blocking=True),
                batch["neg_attention_mask"].to(device, non_blocking=True),
                batch["neg_mask_idx"].to(device, non_blocking=True)
            )
            
            margin_target = batch["margin"].to(device, non_blocking=True)
            margin_pred = pos_logits - neg_logits
            
            loss = mse_loss_fn(margin_pred, margin_target)

            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_train_loss += loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

        avg_train_loss = total_train_loss / len(train_loader)
        log_message(f"Epoch {epoch+1} | Train Loss (MSE): {avg_train_loss:.4f}", log_file)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                pos_logits = model(
                    batch["pos_input_ids"].to(device),
                    batch["pos_attention_mask"].to(device),
                    batch["pos_mask_idx"].to(device)
                )
                neg_logits = model(
                    batch["neg_input_ids"].to(device),
                    batch["neg_attention_mask"].to(device),
                    batch["neg_mask_idx"].to(device)
                )
                margin_target = batch["margin"].to(device)
                margin_pred = pos_logits - neg_logits
                loss = mse_loss_fn(margin_pred, margin_target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        log_message(f"Epoch {epoch+1} | Val Loss (MSE): {avg_val_loss:.4f}", log_file)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            ckpt = os.path.join(SAVE_DIR, "best_gpl_margin.pt")
            torch.save(to_save, ckpt)
            log_message(f"  ✅ Saved best model → {ckpt}", log_file)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log_message(f"  ⏹️ Early stopping at epoch {epoch+1}", log_file)
                break

    log_message(f"\nTraining complete. Best Val MSE: {best_val_loss:.4f}", log_file)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
