"""
evaluate.py — evaluation utilities for classification tasks.
"""

import torch
import numpy as np
import evaluate as hf_evaluate
from tqdm import tqdm


def evaluate_model(model, dataloader, device):
    """
    Run model on a dataloader and return accuracy + predictions.

    Returns:
        dict with 'accuracy', 'predictions', 'labels'
    """
    model.eval()
    model.to(device)

    metric = hf_evaluate.load("accuracy")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ("input_ids", "attention_mask", "token_type_ids")}

            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    result = metric.compute(predictions=all_preds, references=all_labels)

    return {
        "accuracy": round(result["accuracy"] * 100, 2),
        "predictions": all_preds,
        "labels": all_labels,
    }
