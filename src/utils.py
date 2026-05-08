"""
utils.py — shared helpers for timing, model size, and result saving.
"""

import time
import json
import os
import torch
import numpy as np
from pathlib import Path


# ─── Model Size ──────────────────────────────────────────────────────────────

def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model):
    """Estimate model size in MB based on parameter count (float32)."""
    total, _ = count_parameters(model)
    return round(total * 4 / (1024 ** 2), 2)  # 4 bytes per float32


def print_model_info(model, name="Model"):
    total, trainable = count_parameters(model)
    size = model_size_mb(model)
    print(f"{name}")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Estimated size:       {size} MB")


# ─── Inference Speed ─────────────────────────────────────────────────────────

def measure_inference_speed(model, dataloader, device, n_batches=50):
    """
    Measure inference speed in samples/second.
    Runs n_batches batches and returns the average.
    """
    model.eval()
    model.to(device)

    times = []
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ("input_ids", "attention_mask", "token_type_ids")}
            batch_size = batch["input_ids"].shape[0]

            start = time.perf_counter()
            _ = model(**batch)
            end = time.perf_counter()

            times.append(end - start)
            total_samples += batch_size

    avg_time_per_batch = np.mean(times)
    avg_batch_size = total_samples / len(times)
    samples_per_sec = avg_batch_size / avg_time_per_batch

    return {
        "avg_time_per_batch_ms": round(avg_time_per_batch * 1000, 2),
        "samples_per_second": round(samples_per_sec, 2),
        "n_batches_measured": len(times),
    }


# ─── Results Saving ──────────────────────────────────────────────────────────

def save_results(results: dict, filename: str, results_dir: str = "../results"):
    """Save a results dict as JSON."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def load_results(filename: str, results_dir: str = "../results"):
    """Load a results JSON file."""
    path = os.path.join(results_dir, filename)
    with open(path) as f:
        return json.load(f)
