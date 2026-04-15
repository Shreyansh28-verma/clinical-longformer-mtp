import re

# 1. Update train_teacher_longformer.py
with open("c:/Users/shrey/Downloads/dataset/dataset/train_teacher_longformer.py", "r", encoding="utf-8") as f:
    text = f.read()

# Update Forward function
old_forward = """    def forward(self, input_ids, attention_mask, mask_idx):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        mask_logits = logits[batch_indices, mask_idx, :]
        relevant_logit = mask_logits[:, self.relevant_token_id]
        irrelevant_logit = mask_logits[:, self.irrelevant_token_id]
        return relevant_logit - irrelevant_logit"""

new_forward = """    def forward(self, input_ids, attention_mask, mask_idx, mlm_labels=None):
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
            return relevant_logit - irrelevant_logit"""

text = text.replace(old_forward, new_forward)

# Remove pairwise ranking function
text = re.sub(r'# =+ Pairwise Ranking =+.*?return torch\.stack\(losses\)\.mean\(\)', '', text, flags=re.DOTALL)


# Update Training Loop
old_train = """    for epoch in range(EPOCHS):
        model.train()
        total_train_loss, total_bce, total_rank, batches_with_rank = 0, 0, 0, 0

        for batch in tqdm(train_loader, desc=f"[α={alpha}] Epoch {epoch+1} [Train]"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask_idx = batch["mask_idx"].to(device, non_blocking=True)
            topic_ids = batch["topic_id"]

            try:
                with autocast("cuda"):
                    logits = model(input_ids, attention_mask, mask_idx)
                    bce_loss = criterion(logits, labels)
                    rank_loss = pairwise_ranking_loss_from_batch(logits, labels, topic_ids, device)
                    loss = bce_loss + alpha * rank_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                total_train_loss += loss.item()
                total_bce += bce_loss.item()
                total_rank += rank_loss.item()
                batches_with_rank += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        avg_train_loss = total_train_loss / len(train_loader)
        avg_bce = total_bce / len(train_loader)
        avg_rank = total_rank / max(1, batches_with_rank)
        log_message(f"[α={alpha}] Epoch {epoch+1} Train: Loss={avg_train_loss:.4f} | BCE={avg_bce:.4f} | Rank={avg_rank:.6f}", log_file)"""

new_train = """    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"[α={alpha}] Epoch {epoch+1} [Train]"):
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

                with autocast("cuda"):
                    loss_batch, logits = model(input_ids, attention_mask, mask_idx, mlm_labels=mlm_labels)
                    loss = loss_batch.mean()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                total_train_loss += loss.item()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        avg_train_loss = total_train_loss / len(train_loader)
        log_message(f"[α={alpha}] Epoch {epoch+1} Train: Loss={avg_train_loss:.4f}", log_file)"""

text = text.replace(old_train, new_train)


# Update Validation Loop
old_val = """        # ---- Validation ----
        model.eval()
        total_val_loss, total_val_rank, val_batches_with_rank = 0, 0, 0
        all_logits, all_labels = [], []

        with torch.no_grad(), autocast("cuda"):
            for batch in tqdm(val_loader, desc=f"[α={alpha}] Epoch {epoch+1} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                mask_idx = batch["mask_idx"].to(device)
                topic_ids = batch["topic_id"]

                logits = model(input_ids, attention_mask, mask_idx)
                bce_loss = criterion(logits, labels)
                rank_loss = pairwise_ranking_loss_from_batch(logits, labels, topic_ids, device)
                loss = bce_loss + alpha * rank_loss

                total_val_loss += loss.item()
                total_val_rank += rank_loss.item()
                val_batches_with_rank += 1
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_rank = total_val_rank / max(1, val_batches_with_rank)
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()

        try:
            val_auc = roc_auc_score(all_labels, all_logits) if len(set(all_labels.tolist())) == 2 else float("nan")
        except Exception:
            val_auc = float("nan")

        log_message(f"[α={alpha}] Epoch {epoch+1} Val: Loss={avg_val_loss:.4f} | Rank={avg_val_rank:.6f} | AUC={val_auc:.4f}", log_file)"""

new_val = """        # ---- Validation ----
        model.eval()
        total_val_loss = 0
        all_logits, all_labels = [], []

        with torch.no_grad(), autocast("cuda"):
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
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()

        try:
            val_auc = roc_auc_score(all_labels, all_logits) if len(set(all_labels.tolist())) == 2 else float("nan")
        except Exception:
            val_auc = float("nan")

        log_message(f"[α={alpha}] Epoch {epoch+1} Val: Loss={avg_val_loss:.4f} | AUC={val_auc:.4f}", log_file)"""

text = text.replace(old_val, new_val)

with open("c:/Users/shrey/Downloads/dataset/dataset/train_teacher_longformer.py", "w", encoding="utf-8") as f:
    f.write(text)


# 2. Update inference_teacher_longformer.py
with open("c:/Users/shrey/Downloads/dataset/dataset/inference_teacher_longformer.py", "r", encoding="utf-8") as f:
    text_inf = f.read()

# Fix TeacherReranker initialization missing tokenizer
old_init = "model = TeacherReranker().to(DEVICE)"
new_init = "model = TeacherReranker(tokenizer).to(DEVICE)"
text_inf = text_inf.replace(old_init, new_init)

# Fix prompt formatting and mask_idx extraction
old_inference_loop = """            second_text = f"{trial_text} {tokenizer.sep_token} Reasoning: {reasoning}" if reasoning else trial_text

            enc = tokenizer(
                queries[topic_id],
                second_text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)

            logit = model(enc["input_ids"], enc["attention_mask"]).item()"""

new_inference_loop = """            query = queries[topic_id]
            query_with_prompt = f"{query} {tokenizer.sep_token} Relevance: {tokenizer.mask_token}"
            second_text = f"{trial_text} {tokenizer.sep_token} Reasoning: {reasoning}" if reasoning else trial_text

            enc = tokenizer(
                query_with_prompt,
                second_text,
                truncation="only_second",
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)

            input_ids = enc["input_ids"]
            mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if mask_positions.numel() == 0:
                print(f"Warning: No mask token found for topic {topic_id}")
                continue
            mask_idx = mask_positions[0].item()

            with torch.no_grad():
                logit = model(enc["input_ids"], enc["attention_mask"], torch.tensor([mask_idx], device=DEVICE).long()).item()"""

text_inf = text_inf.replace(old_inference_loop, new_inference_loop)

# Fix truncation warning from "truncation=True" dropping "only_second"
old_enc = """truncation=True,"""
# It's already in the replacement above!

with open("c:/Users/shrey/Downloads/dataset/dataset/inference_teacher_longformer.py", "w", encoding="utf-8") as f:
    f.write(text_inf)
