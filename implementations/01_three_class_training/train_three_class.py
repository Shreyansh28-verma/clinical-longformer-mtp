# train_three_class.py
# ---------------------------------------------------------------
# Three-Class TeacherReranker training.
# Extends the baseline binary (relevant/irrelevant) MLM prompt to
# a three-class objective:
#   label 0 → [MASK] should predict "irrelevant"
#   label 1 → [MASK] should predict "excluded"
#   label 2 → [MASK] should predict "eligible"
#
# Ranking score (for TREC output):
#   score = sigmoid(logit("eligible") − logit("irrelevant"))
#
# Key improvements over baseline:
#   1. Three-class MLM objective (teaches model to separate excluded)
#   2. Weighted cross-entropy: eligible (2.5×) > excluded (1.5×) > not-relevant (1.0×)
#   3. Training from the larger 3-class dataset (~35K samples from qrels)
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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.amp import autocast

# ====================== CONFIG ======================
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIS_DIR      = os.path.dirname(os.path.abspath(__file__))

QUERIES_TSV   = os.path.join(BASE_DIR, "synthetic_gold_queries.tsv")   # synthetics (fallback)
CT_2021_TSV   = os.path.join(BASE_DIR, "ct_2021_queries.tsv")
CT_2022_TSV   = os.path.join(BASE_DIR, "ct_2022_queries.tsv")
TRAIN_JSONL   = os.path.join(THIS_DIR, "three_class_train.jsonl")      # produced by prepare script
TRIALS_JSONL  = os.path.join(BASE_DIR, "concatenated_trials.jsonl")
MODEL_NAME    = "yikuan8/Clinical-Longformer"

MAX_LENGTH    = 4096
BATCH_SIZE    = 4
EPOCHS        = 10
BASE_LR       = 3e-5
CLS_LR        = 6e-5
PATIENCE      = 5
SAVE_DIR      = os.path.join(BASE_DIR, "models_new", "ThreeClass_ClinicalLongformer")

# Class weights: eligible gets highest weight (hardest to distinguish)
CLASS_WEIGHTS = {0: 1.0, 1: 2.0, 2: 3.0}


# ====================== LOGGING ======================
def log_message(msg: str, log_file: str):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ====================== DATA LOADERS ======================
def load_all_queries() -> dict:
    """Load all queries from multiple query TSV files, keyed by topic_id."""
    queries = {}
    for tsv_path in [CT_2021_TSV, CT_2022_TSV, QUERIES_TSV]:
        if not os.path.exists(tsv_path):
            continue
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) == 2:
                    topic_id, text = row
                    queries[topic_id.strip()] = text.strip()
    return queries


def load_three_class_data(jsonl_file: str) -> list:
    """Load 3-class labeled data: [(topic_id, trial_id, reasoning, label)]"""
    data = []
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(
            f"Three-class training data not found: {jsonl_file}\n"
            f"Run prepare_three_class_data.py first."
        )
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = int(obj.get("label", 0))  # 0, 1, or 2
            data.append((
                str(obj["topic_id"]),
                str(obj["trial_id"]),
                obj.get("reasoning", ""),
                label,
            ))
    return data


def load_trials(jsonl_file: str) -> dict:
    trials = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            trials[obj["id"]] = obj.get("concatenated_text", "")
    return trials


# ====================== DATASET ======================
class ThreeClassTrialDataset(Dataset):
    def __init__(self, data: list, queries: dict, trials: dict, tokenizer):
        # Filter to only records whose topic_id and trial_id we have
        self.data = [
            (tid, trid, reason, lbl)
            for tid, trid, reason, lbl in data
            if tid in queries and trid in trials
        ]
        self.queries = queries
        self.trials = trials
        self.tokenizer = tokenizer
        if tokenizer.mask_token is None:
            raise ValueError("Tokenizer must provide a mask token.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        topic_id, trial_id, reasoning, label = self.data[idx]
        query = self.queries[topic_id]
        trial_text = self.trials[trial_id]

        query_with_prompt = f"{query} {self.tokenizer.sep_token} Relevance: {self.tokenizer.mask_token}"
        second_text = (
            f"{trial_text} {self.tokenizer.sep_token} Reasoning: {reasoning}"
            if reasoning else trial_text
        )

        tokenized = self.tokenizer(
            query_with_prompt,
            second_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation="only_second",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if mask_positions.numel() != 1:
            raise ValueError(f"Expected exactly one [MASK] token, got {mask_positions.numel()}")
        mask_idx = mask_positions.item()

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "mask_idx": torch.tensor(mask_idx, dtype=torch.long),
        }


# ====================== MODEL ======================
class TeacherReranker3Class(nn.Module):
    """Clinical-Longformer with three-class MLM prompt scoring.

    Prompt format: [Query] [SEP] Relevance: [MASK] [SEP] [Trial] [SEP] Reasoning: [...]
    The [MASK] position predicts one of:
        "eligible"   → label 2
        "excluded"   → label 1
        "irrelevant" → label 0

    Ranking score = sigmoid(logit("eligible") - logit("irrelevant"))
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
        self.tokenizer = tokenizer

        # Verify all three tokens encode as single tokens
        eligible_ids   = tokenizer(" eligible",   add_special_tokens=False)["input_ids"]
        excluded_ids   = tokenizer(" excluded",   add_special_tokens=False)["input_ids"]
        irrelevant_ids = tokenizer(" irrelevant", add_special_tokens=False)["input_ids"]

        if len(eligible_ids) != 1:
            raise ValueError(f"'eligible' tokenizes to {len(eligible_ids)} tokens, need 1")
        if len(excluded_ids) != 1:
            raise ValueError(f"'excluded' tokenizes to {len(excluded_ids)} tokens, need 1")
        if len(irrelevant_ids) != 1:
            raise ValueError(f"'irrelevant' tokenizes to {len(irrelevant_ids)} tokens, need 1")

        self.eligible_token_id   = eligible_ids[0]
        self.excluded_token_id   = excluded_ids[0]
        self.irrelevant_token_id = irrelevant_ids[0]

        # Map label → target token id
        self.label_to_token = {
            0: self.irrelevant_token_id,
            1: self.excluded_token_id,
            2: self.eligible_token_id,
        }

    def forward(self, input_ids, attention_mask, mask_idx, mlm_labels=None):
        if mlm_labels is not None:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=mlm_labels,
            )
            loss = outputs.loss
            logits = outputs.logits
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = None

        batch_indices = torch.arange(logits.size(0), device=logits.device)
        mask_logits = logits[batch_indices, mask_idx, :]

        eligible_logit   = mask_logits[:, self.eligible_token_id]
        irrelevant_logit = mask_logits[:, self.irrelevant_token_id]
        ranking_score    = eligible_logit - irrelevant_logit  # used for TREC ranking

        if loss is not None:
            return loss, ranking_score
        return ranking_score


# ====================== TRAINING ======================
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_file = os.path.join(SAVE_DIR, "training_log.txt")

    log_message("\n========== THREE-CLASS TEACHER TRAINING ==========", log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Device: {device}", log_file)

    # ---- Data ----
    log_message("\n[Data] Loading queries and trials...", log_file)
    queries = load_all_queries()
    log_message(f"  Queries loaded: {len(queries):,}", log_file)

    log_message("[Data] Loading 3-class training records...", log_file)
    all_data = load_three_class_data(TRAIN_JSONL)
    log_message(f"  Records: {len(all_data):,}", log_file)

    from collections import Counter
    label_dist = Counter(lbl for _, _, _, lbl in all_data)
    log_message(f"  Label dist: {dict(label_dist)}", log_file)

    log_message("[Data] Loading trial corpus (this may take a few minutes)...", log_file)
    trials = load_trials(TRIALS_JSONL)
    log_message(f"  Trials loaded: {len(trials):,}", log_file)

    train_data, val_data = train_test_split(all_data, test_size=0.15, random_state=42)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = MAX_LENGTH
    if hasattr(tokenizer, "add_prefix_space"):
        tokenizer.add_prefix_space = True

    train_ds = ThreeClassTrialDataset(train_data, queries, trials, tokenizer)
    val_ds   = ThreeClassTrialDataset(val_data,   queries, trials, tokenizer)
    log_message(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}", log_file)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              num_workers=4, pin_memory=True)

    # ---- Model ----
    model = TeacherReranker3Class(tokenizer).to(device)
    model.backbone.gradient_checkpointing_enable()
    log_message("Gradient checkpointing enabled.", log_file)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log_message(f"DataParallel: {torch.cuda.device_count()} GPUs", log_file)

    # ---- Class-weighted loss ----
    weight_tensor = torch.tensor(
        [CLASS_WEIGHTS[0], CLASS_WEIGHTS[1], CLASS_WEIGHTS[2]],
        dtype=torch.float, device=device
    )
    ce_loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    # ---- Optimizer ----
    backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
    lm_head_params = list(backbone.lm_head.parameters())
    lm_head_ids    = {id(p) for p in lm_head_params}
    encoder_params = [p for p in backbone.longformer.parameters()
                      if id(p) not in lm_head_ids]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params,  "lr": BASE_LR},
        {"params": lm_head_params,  "lr": CLS_LR},
    ])
    num_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=max(1, int(0.05 * num_steps)),
        num_training_steps=num_steps,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    # ---- Epoch Loop ----
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)    # (B,) long 0/1/2
            mask_idx       = batch["mask_idx"].to(device, non_blocking=True)

            model_obj = model.module if isinstance(model, nn.DataParallel) else model

            # Build MLM label tensor (mlm_labels at [MASK] position, -100 elsewhere)
            mlm_labels = torch.full_like(input_ids, fill_value=-100)
            for i in range(input_ids.size(0)):
                mlm_labels[i, mask_idx[i]] = model_obj.label_to_token[labels[i].item()]

            try:
                with autocast("cuda", dtype=torch.bfloat16):
                    # backbone's built-in MLM loss (standard CE over vocab)
                    mlm_loss, ranking_score = model(
                        input_ids, attention_mask, mask_idx, mlm_labels=mlm_labels
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

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        log_message(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}", log_file)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0.0
        all_scores, all_labels_bin = [], []

        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                mask_idx       = batch["mask_idx"].to(device)

                model_obj = model.module if isinstance(model, nn.DataParallel) else model
                mlm_labels = torch.full_like(input_ids, fill_value=-100)
                for i in range(input_ids.size(0)):
                    mlm_labels[i, mask_idx[i]] = model_obj.label_to_token[labels[i].item()]

                mlm_loss, ranking_score = model(
                    input_ids, attention_mask, mask_idx, mlm_labels=mlm_labels
                )
                total_val_loss += mlm_loss.mean().item()

                scores_cpu = ranking_score.cpu().float()
                # Binary AUC: eligible (2) vs non-eligible (0 or 1)
                binary_labels = (labels.cpu() == 2).float()
                all_scores.extend(scores_cpu.tolist())
                all_labels_bin.extend(binary_labels.tolist())

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(all_labels_bin, all_scores) \
                if len(set(all_labels_bin)) == 2 else float("nan")
        except Exception:
            val_auc = float("nan")

        log_message(
            f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | AUC: {val_auc:.4f}",
            log_file
        )

        # ---- Checkpoint ----
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            ckpt = os.path.join(SAVE_DIR, "best_three_class.pt")
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
