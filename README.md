# DistilBERT: Reproduction & Extension

> Reproduction and extension of [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019)
>
> TU Wien — Recommender Systems / ML course project, 2025

---

## Paper Summary

DistilBERT is a compressed version of BERT-base that retains 97% of its language understanding performance while being 40% smaller and 60% faster. It achieves this through **knowledge distillation** during pre-training, using a triple loss: soft cross-entropy (distillation), cosine embedding (hidden state alignment), and masked language modeling (MLM).

---

## Team

| Name | Branch | Role |
|------|--------|------|
| Person A | `setup`, `presentation` | Repo setup, slides, architecture explainer |
| Person B | `reproduction/bert-baseline` | BERT-base fine-tuning + benchmarks |
| Person C | `reproduction/distilbert` | DistilBERT fine-tuning + benchmarks |
| Person D | `extension/task-distillation` | Task-specific distillation extension |

---

## Repository Structure

```
distilbert-reproduction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── download_sst2.py          # script to download/cache SST-2
├── notebooks/
│   ├── 01_bert_baseline.ipynb    # BERT-base fine-tuning on SST-2
│   ├── 02_distilbert.ipynb       # DistilBERT fine-tuning on SST-2
│   ├── 03_benchmarks.ipynb       # Speed, size, accuracy comparison
│   └── 04_extension.ipynb        # Task-specific distillation
├── src/
│   ├── train.py                  # Shared training loop
│   ├── evaluate.py               # Evaluation utilities
│   └── utils.py                  # Misc helpers (timing, model size)
└── results/
    ├── bert_baseline.json
    ├── distilbert.json
    ├── benchmarks.json
    └── extension.json
```

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, reviewed code only. No direct pushes. |
| `setup` | Repo scaffolding, README, requirements |
| `reproduction/bert-baseline` | BERT-base fine-tuning notebook |
| `reproduction/distilbert` | DistilBERT fine-tuning notebook |
| `reproduction/benchmarks` | Speed + size + accuracy measurements |
| `extension/task-distillation` | Extension: distill fine-tuned BERT into smaller student |
| `presentation` | Generated figures and plots for slides |

All branches are merged into `main` via pull request with at least one reviewer.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Running on Google Colab

Each notebook in `notebooks/` is self-contained and designed to run top-to-bottom on a free Colab T4 GPU. Open any notebook, go to **Runtime → Run all**.

Make sure to enable GPU: **Runtime → Change runtime type → T4 GPU**

---

## Reproduction

We reproduce the fine-tuning results from the paper using the released pre-trained weights (`bert-base-uncased` and `distilbert-base-uncased` from HuggingFace). Full pre-training is not reproduced due to compute constraints — this is consistent with the assignment guidelines.

### Task: SST-2 (Sentiment Classification)

| Model | Accuracy | Parameters | Model Size | Inference Speed |
|-------|----------|------------|------------|-----------------|
| BERT-base | TBD | 110M | ~440MB | TBD samples/sec |
| DistilBERT | TBD | 66M | ~265MB | TBD samples/sec |
| Paper (BERT-base) | 93.5% | 110M | — | 1x |
| Paper (DistilBERT) | 91.3% | 66M | 40% smaller | 1.6x faster |

*Results to be filled in after running notebooks.*

---

## Extension

### Task-Specific Distillation

The original paper performs distillation during **pre-training**. Our extension asks: *is distillation during fine-tuning alone sufficient and practical?*

We fine-tune BERT-base on SST-2 (teacher), then train a 3-layer student model using the teacher's soft logits as supervision — no pre-training distillation involved.

**Comparison:**

| Model | Accuracy | Parameters | Training Time |
|-------|----------|------------|---------------|
| BERT-base (teacher) | TBD | 110M | TBD |
| DistilBERT (pre-train distilled) | TBD | 66M | TBD |
| Task-distilled student (ours) | TBD | ~33M | TBD |

*Results to be filled in after running notebooks.*

---

## Results Summary

To be completed. All raw results are saved as JSON files in `results/`.

---

## References

```
@article{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}
```
