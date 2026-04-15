#train_teacher_longformer.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import json
import csv
from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.amp import autocast

# ================= CONFIG =================
QUERIES_TSV = "synthetic_gold_queries.tsv"
REASONINGS_JSONL = "train_1196_deepseek_clean.jsonl"
TRIALS_JSONL = "concatenated_trials.jsonl"
MODEL_NAME = "yikuan8/Clinical-Longformer"

MAX_LENGTH = 4096
BATCH_SIZE = 4
EPOCHS = 10
BASE_LR = 3e-5
CLS_LR = 6e-5
PATIENCE = 5
#ALPHA_VALUES = [0.1, 0.2, 0.3]
ALPHA_VALUES = [0.2]  # ✅ multiple alpha runs


# ================= Logging Utility =================
def log_message(msg, log_file):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ================= Data Loaders =================
def load_queries(tsv_file):
    queries = {}
    with open(tsv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) == 2:
                topic_id, query_text = row
                queries[topic_id] = query_text.strip()
    return queries


def load_reasonings(jsonl_file):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            label = 1 if str(obj.get("relevance", "")).lower() == "relevant" else 0
            data.append((obj["topic_id"], obj["trial_id"], obj.get("reasoning", ""), label))
    return data


def load_trials(jsonl_file):
    trials = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            trials[obj["id"]] = obj.get("concatenated_text", "")
    return trials


# ================= Dataset =================
class TrialDataset(Dataset):
    def __init__(self, data, queries, trials, tokenizer):
        self.data = data
        self.queries = queries
        self.trials = trials
        self.tokenizer = tokenizer
        if tokenizer.mask_token is None:
            raise ValueError("Tokenizer must provide a mask token for masked language modeling.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        topic_id, trial_id, reasoning, label = self.data[idx]
        query = self.queries[topic_id]
        trial_text = self.trials[trial_id]

        query_with_prompt = f"{query} {self.tokenizer.sep_token} Relevance: {self.tokenizer.mask_token}"
        secondary_segments = [trial_text]
        if reasoning:
            secondary_segments.append(f"{self.tokenizer.sep_token} Reasoning: {reasoning}")
        second_text = " ".join(secondary_segments)

        tokenized = self.tokenizer(
            query_with_prompt,
            second_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation="only_second",
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if mask_positions.numel() != 1:
            raise ValueError("Exactly one mask token is required in each input instance.")
        mask_idx = mask_positions.item()

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
            "mask_idx": torch.tensor(mask_idx, dtype=torch.long),
            "topic_id": topic_id,
            "trial_id": trial_id
        }


# ================= Model =================
class TeacherReranker(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
        self.tokenizer = tokenizer

        relevant_ids = tokenizer(" relevant", add_special_tokens=False)["input_ids"]
        irrelevant_ids = tokenizer(" irrelevant", add_special_tokens=False)["input_ids"]

        if len(relevant_ids) != 1 or len(irrelevant_ids) != 1:
            raise ValueError("Tokenizer must encode 'relevant' and 'irrelevant' as single tokens.")

        self.relevant_token_id = relevant_ids[0]
        self.irrelevant_token_id = irrelevant_ids[0]

    def forward(self, input_ids, attention_mask, mask_idx, mlm_labels=None):
        if mlm_labels is not None:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=mlm_labels)
            loss = outputs.loss
            logits = outputs.logits
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            mask_logits = logits[batch_indices, mask_idx, :]
            relevant_logit = mask_logits[:, self.relevant_token_id]
            irrelevant_logit = mask_logits[:, self.irrelevant_token_id]
            return loss, relevant_logit - irrelevant_logit
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            mask_logits = logits[batch_indices, mask_idx, :]
            relevant_logit = mask_logits[:, self.relevant_token_id]
            irrelevant_logit = mask_logits[:, self.irrelevant_token_id]
            return relevant_logit - irrelevant_logit





# ================= Training Function =================
def train_teacher(alpha):
    save_dir = f"models_new/Teacher_ClinicalLongformer_1196/alpha{alpha}"
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"training_log_alpha{alpha}.txt")

    log_message(f"\n==================== TRAINING TEACHER (α={alpha}) ====================", log_file)
    log_message(f"Using GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}", log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    queries = load_queries(QUERIES_TSV)
    reasonings_data = load_reasonings(REASONINGS_JSONL)
    trials = load_trials(TRIALS_JSONL)
    train_data, val_data = train_test_split(reasonings_data, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = MAX_LENGTH
    if hasattr(tokenizer, "add_prefix_space"):
        tokenizer.add_prefix_space = True

    train_loader = DataLoader(TrialDataset(train_data, queries, trials, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(TrialDataset(val_data, queries, trials, tokenizer),
                            batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # ---- Model ----
    model = TeacherReranker(tokenizer).to(device)
    model.backbone.gradient_checkpointing_enable()
    log_message("Gradient checkpointing enabled for MLM backbone.", log_file)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log_message(f"Using DataParallel with {torch.cuda.device_count()} GPUs.", log_file)

    # ---- Optimizer & Scheduler ----
    backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
    decoder_weight = backbone.lm_head.decoder.weight
    lm_head_params = [
        param for name, param in backbone.lm_head.named_parameters()
        if name != "decoder.weight"
    ]
    lm_head_param_ids = {id(p) for p in lm_head_params}
    lm_head_param_ids.add(id(decoder_weight))
    param_longformer = [p for p in backbone.longformer.parameters() if id(p) not in lm_head_param_ids]
    param_classifier = lm_head_params
    optimizer = torch.optim.AdamW([
        {"params": param_longformer, "lr": BASE_LR},
        {"params": param_classifier, "lr": CLS_LR}
    ])
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=max(1, int(0.05 * num_training_steps)),
        num_training_steps=num_training_steps
    )

    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    patience_counter = 0

    # ---- Epoch loop ----
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"[α={alpha}] Epoch {epoch+1} [Train]"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask_idx = batch["mask_idx"].to(device, non_blocking=True)
            topic_ids = batch["topic_id"]

            try:
                model_obj = model.module if isinstance(model, nn.DataParallel) else model
                rel_id = model_obj.relevant_token_id
                irrel_id = model_obj.irrelevant_token_id

                mlm_labels = torch.full_like(input_ids, fill_value=-100)
                for i in range(input_ids.size(0)):
                    mlm_labels[i, mask_idx[i]] = rel_id if labels[i] == 1 else irrel_id

                with autocast("cuda", dtype=torch.bfloat16):
                    loss_batch, logits = model(input_ids, attention_mask, mask_idx, mlm_labels=mlm_labels)
                    loss = loss_batch.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        avg_train_loss = total_train_loss / len(train_loader)
        log_message(f"[α={alpha}] Epoch {epoch+1} Train: Loss={avg_train_loss:.4f}", log_file)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0
        all_logits, all_labels = [], []

        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            for batch in tqdm(val_loader, desc=f"[α={alpha}] Epoch {epoch+1} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                mask_idx = batch["mask_idx"].to(device)
                topic_ids = batch["topic_id"]

                model_obj = model.module if isinstance(model, nn.DataParallel) else model
                rel_id = model_obj.relevant_token_id
                irrel_id = model_obj.irrelevant_token_id
                
                mlm_labels = torch.full_like(input_ids, fill_value=-100)
                for i in range(input_ids.size(0)):
                    mlm_labels[i, mask_idx[i]] = rel_id if labels[i] == 1 else irrel_id

                loss_batch, logits = model(input_ids, attention_mask, mask_idx, mlm_labels=mlm_labels)
                loss = loss_batch.mean()

                total_val_loss += loss.item()
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = total_val_loss / len(val_loader)
        all_logits = torch.cat(all_logits).float().numpy()
        all_labels = torch.cat(all_labels).float().numpy()

        try:
            val_auc = roc_auc_score(all_labels, all_logits) if len(set(all_labels.tolist())) == 2 else float("nan")
        except Exception:
            val_auc = float("nan")

        log_message(f"[α={alpha}] Epoch {epoch+1} Val: Loss={avg_val_loss:.4f} | AUC={val_auc:.4f}", log_file)

        # ---- Checkpoint ----
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(to_save, os.path.join(save_dir, f"best_teacher_alpha{alpha}.pt"))
            log_message(f"✅ Saved new best teacher model (α={alpha})", log_file)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log_message(f"⏹️ Early stopping triggered for α={alpha}", log_file)
                break

    log_message(f"Training complete for α={alpha}. Best Val Loss: {best_val_loss:.4f}", log_file)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    for alpha in ALPHA_VALUES:
        train_teacher(alpha)
        torch.cuda.empty_cache()
