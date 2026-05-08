"""
train.py — shared training loop for fine-tuning classification models.
"""

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    device,
    epochs=3,
    lr=2e-5,
    warmup_ratio=0.1,
):
    """
    Fine-tune a HuggingFace model for sequence classification.

    Args:
        model:              HuggingFace model with a classification head
        train_dataloader:   DataLoader for training data
        val_dataloader:     DataLoader for validation data
        device:             torch device
        epochs:             number of training epochs
        lr:                 learning rate
        warmup_ratio:       fraction of steps used for LR warmup

    Returns:
        history: list of dicts with train_loss and val_accuracy per epoch
    """
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [val]"):
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()
                         if k in ("input_ids", "attention_mask", "token_type_ids")}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100

        print(f"Epoch {epoch + 1}: loss={avg_train_loss:.4f}, val_acc={val_accuracy:.2f}%")

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_accuracy": round(val_accuracy, 2),
        })

    return history
