# train_structural.py
# ---------------------------------------------------------------
# Structural Global Attention Training for TeacherReranker.
# Modifies the Longformer baseline to assign global attention to
# specific structural section headers in the clinical trial text:
#   - "Inclusion Criteria"
#   - "Exclusion Criteria"
#   - "Brief Summary"
#   - "Condition"
# This helps the model bridge the gap between patient query and
# distant criteria sections in very long documents.
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

# ====================== CONFIG ======================
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUERIES_TSV   = os.path.join(BASE_DIR, "synthetic_gold_queries.tsv")
REASONINGS_JSONL = os.path.join(BASE_DIR, "train_1196_deepseek_clean.jsonl")
TRIALS_JSONL  = os.path.join(BASE_DIR, "concatenated_trials.jsonl")
MODEL_NAME    = "yikuan8/Clinical-Longformer"

MAX_LENGTH    = 4096
BATCH_SIZE    = 4
EPOCHS        = 10
BASE_LR       = 3e-5
CLS_LR        = 6e-5
PATIENCE      = 5
SAVE_DIR      = os.path.join(BASE_DIR, "models_new", "Structural_ClinicalLongformer")

# Key structural phrases we want global attention on
STRUCTURAL_PHRASES = [
    "inclusion criteria",
    "exclusion criteria",
    "brief summary",
    "condition,"  # (many trials list 'Condition, intervention' in concatenation)
]

# ====================== LOGGING ======================
def log_message(msg: str, log_file: str):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ====================== DATA LOADERS ======================
def load_queries(tsv_file: str) -> dict:
    queries = {}
    with open(tsv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) == 2:
                topic_id, query_text = row
                queries[topic_id] = query_text.strip()
    return queries


def load_reasonings(jsonl_file: str) -> list:
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            label = 1 if str(obj.get("relevance", "")).lower() == "relevant" else 0
            data.append((obj["topic_id"], obj["trial_id"], obj.get("reasoning", ""), label))
    return data


def load_trials(jsonl_file: str) -> dict:
    trials = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            trials[obj["id"]] = obj.get("concatenated_text", "")
    return trials


# ====================== DATASET ======================
class StructuralTrialDataset(Dataset):
    def __init__(self, data: list, queries: dict, trials: dict, tokenizer):
        self.data = data
        self.queries = queries
        self.trials = trials
        self.tokenizer = tokenizer

        # Pre-tokenize the structural phrases to find their token ID sequences
        self.structural_sequences = []
        for phrase in STRUCTURAL_PHRASES:
            # We use add_special_tokens=False. 
            # Note: phrase might start with space depending on tokenizer config
            ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
            if ids:
                self.structural_sequences.append(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        topic_id, trial_id, reasoning, label = self.data[idx]
        query = self.queries[topic_id]
        trial_text = self.trials[trial_id]

        query_with_prompt = f"{query} {self.tokenizer.sep_token} Relevance: {self.tokenizer.mask_token}"
        second_text = f"{trial_text} {self.tokenizer.sep_token} Reasoning: {reasoning}" if reasoning else trial_text

        tokenized = self.tokenizer(
            query_with_prompt,
            second_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation="only_second",
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        # Build standard global attention mask for Longformer
        # By default, [CLS] is globally attended (id 0)
        global_attention_mask = torch.zeros_like(input_ids)
        
        # 1. Global attention on special tokens ([CLS], [SEP], [MASK])
        global_attention_mask[input_ids == self.tokenizer.cls_token_id] = 1
        global_attention_mask[input_ids == self.tokenizer.sep_token_id] = 1
        global_attention_mask[input_ids == self.tokenizer.mask_token_id] = 1
        
        # 2. Global attention on structural headers
        # We manually slide over to find matches for the phrase sequences
        input_list = input_ids.tolist()
        for seq in self.structural_sequences:
            seq_len = len(seq)
            if seq_len == 0: continue
            
            for i in range(len(input_list) - seq_len + 1):
                if input_list[i : i+seq_len] == seq:
                    # Mark all tokens in the matched phrase for global attention
                    global_attention_mask[i : i+seq_len] = 1

        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        mask_idx = mask_positions[0].item() if mask_positions.numel() > 0 else 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "labels": torch.tensor(label, dtype=torch.float),
            "mask_idx": torch.tensor(mask_idx, dtype=torch.long),
        }


# ====================== MODEL ======================
class TeacherRerankerStructural(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
        self.tokenizer = tokenizer

        relevant_ids = tokenizer(" relevant", add_special_tokens=False)["input_ids"]
        irrelevant_ids = tokenizer(" irrelevant", add_special_tokens=False)["input_ids"]

        self.relevant_token_id = relevant_ids[0]
        self.irrelevant_token_id = irrelevant_ids[0]

    def forward(self, input_ids, attention_mask, global_attention_mask, mask_idx, mlm_labels=None):
        if mlm_labels is not None:
             # Pass global_attention_mask to Longformer
            outputs = self.backbone(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                labels=mlm_labels
            )
            loss = outputs.loss
            logits = outputs.logits
        else:
            outputs = self.backbone(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            logits = outputs.logits
            loss = None

        batch_indices = torch.arange(logits.size(0), device=logits.device)
        mask_logits = logits[batch_indices, mask_idx, :]
        relevant_logit = mask_logits[:, self.relevant_token_id]
        irrelevant_logit = mask_logits[:, self.irrelevant_token_id]
        ranking_score = relevant_logit - irrelevant_logit

        if loss is not None:
            return loss, ranking_score
        return ranking_score


# ====================== TRAINING ======================
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_file = os.path.join(SAVE_DIR, "training_log.txt")

    log_message("\n========== STRUCTURAL ATTENTION TEACHER ==========", log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    queries = load_queries(QUERIES_TSV)
    all_data = load_reasonings(REASONINGS_JSONL)
    trials = load_trials(TRIALS_JSONL)
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = MAX_LENGTH
    if hasattr(tokenizer, "add_prefix_space"):
        tokenizer.add_prefix_space = True

    train_ds = StructuralTrialDataset(train_data, queries, trials, tokenizer)
    val_ds   = StructuralTrialDataset(val_data, queries, trials, tokenizer)

    # Check how many global attention tokens we actually get
    sample = train_ds[0]
    global_count = sample["global_attention_mask"].sum().item()
    log_message(f"Sample global attention tokens: {global_count} (out of {MAX_LENGTH})", log_file)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              num_workers=4, pin_memory=True)

    model = TeacherRerankerStructural(tokenizer).to(device)
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

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            global_mask    = batch["global_attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)
            mask_idx       = batch["mask_idx"].to(device, non_blocking=True)

            model_obj = model.module if isinstance(model, nn.DataParallel) else model
            mlm_labels = torch.full_like(input_ids, fill_value=-100)
            for i in range(input_ids.size(0)):
                mlm_labels[i, mask_idx[i]] = model_obj.relevant_token_id if labels[i] == 1 else model_obj.irrelevant_token_id

            try:
                with autocast("cuda", dtype=torch.bfloat16):
                    mlm_loss, _ = model(
                        input_ids, attention_mask, global_mask, mask_idx, mlm_labels=mlm_labels
                    )
                    mlm_loss = mlm_loss.mean()

                mlm_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_train_loss += mlm_loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

        avg_train_loss = total_train_loss / len(train_loader)
        log_message(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}", log_file)

        # Validation
        model.eval()
        total_val_loss = 0.0
        all_scores, all_labels = [], []

        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                global_mask    = batch["global_attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                mask_idx       = batch["mask_idx"].to(device)

                model_obj = model.module if isinstance(model, nn.DataParallel) else model
                mlm_labels = torch.full_like(input_ids, fill_value=-100)
                for i in range(input_ids.size(0)):
                    mlm_labels[i, mask_idx[i]] = model_obj.relevant_token_id if labels[i] == 1 else model_obj.irrelevant_token_id

                mlm_loss, rank_score = model(
                    input_ids, attention_mask, global_mask, mask_idx, mlm_labels=mlm_labels
                )
                total_val_loss += mlm_loss.mean().item()
                all_scores.extend(rank_score.cpu().float().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_val_loss = total_val_loss / len(val_loader)
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) == 2 else float("nan")
        except Exception:
            val_auc = float("nan")

        log_message(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | AUC: {val_auc:.4f}", log_file)

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            ckpt = os.path.join(SAVE_DIR, "best_structural.pt")
            torch.save(to_save, ckpt)
            log_message(f"  ✅ Saved best model → {ckpt}", log_file)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log_message(f"  ⏹️ Early stopping at epoch {epoch+1}", log_file)
                break

    log_message(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}", log_file)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
