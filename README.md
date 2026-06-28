# DistilBERT: Reproduction & Extension

> Reproduction and extension of [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019)
>
> TU Wien — NLP course project, 2025

---

## Paper Summary

DistilBERT is a compressed version of BERT-base that retains 97% of its language understanding performance while being 40% smaller and 60% faster. It achieves this through **knowledge distillation** during pre-training, using a triple loss: soft cross-entropy (distillation), cosine embedding (hidden state alignment), and masked language modeling (MLM).

---

## Team

| Name | Task |
|------|------|
| Christine | SST-2 + GLUE (Table 1) |
| Eman | IMDb sentiment + Layer Ablation Extension |
| Djem & Sneha | SQuAD v1.1 QA |

---

## Repository Structure

```
distilbert-reproduction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── download_sst2.py           # script to download/cache datasets
├── notebooks/
│   ├── pipeline_sst2_glue.ipynb   # Christine — SST-2 & GLUE tasks
│   ├── pipeline_imdb.ipynb        # Eman — IMDb sentiment
│   ├── pipeline_squad.ipynb       # Djem & Sneha — SQuAD v1.1
│   ├── comparison.ipynb           # Final comparison table & plots (run last)
│   └── extension.ipynb            # Extension — Layer Ablation Study on DistilBERT
├── src/
│   ├── train.py                   # Shared training loop
│   ├── evaluate.py                # Evaluation utilities
│   └── utils.py                   # Helpers (model size, speed, saving results)
└── results/                       # All experiment outputs saved here as JSON
```

---

## Code Structure (`src/`)

The `src/` folder contains shared helper code used by all notebooks. Instead of writing the same logic in every notebook, we put it here once and import it — this also ensures every model is trained and evaluated under identical conditions.

### `train.py`
The training loop used to fine-tune models. Handles the optimizer, learning rate scheduler, and validation after each epoch.

### `evaluate.py`
Runs a trained model on the validation set and returns the accuracy. Used by all notebooks so every model is evaluated the same way.

### `utils.py`
A collection of helper functions:
- **`count_parameters`** — counts how many parameters a model has
- **`model_size_mb`** — estimates the model's size in megabytes
- **`measure_inference_speed`** — measures how many samples per second the model processes
- **`save_results` / `load_results`** — saves and loads experiment results as JSON files

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, reviewed code only. No direct pushes. |
| `develop` | All ongoing work goes here |

Work on `develop`, open a Pull Request to merge into `main` when done.

---

## How to Run

### Requirements

```bash
pip install -r requirements.txt
```

### On Google Colab / JupyterHub

Each notebook is self-contained and runs top-to-bottom on a GPU environment.

1. Open the notebook
2. Enable GPU if on Colab: **Runtime → Change runtime type → T4 GPU**
3. Click **Runtime → Run all**

### Order of execution

```
pipeline_sst2_glue.ipynb  ──┐
pipeline_imdb.ipynb        ──┼──► results/  ──► comparison.ipynb
pipeline_squad.ipynb       ──┘

extension.ipynb  ──► results/  (layer ablation results saved separately)
```

Run the three pipeline notebooks first (in any order), then run `comparison.ipynb` to generate the final tables and plots. The extension notebook is independent and can be run separately.

---

## Reproduction Results

We reproduce fine-tuning results using released pre-trained weights from HuggingFace. Full pre-training is not reproduced due to compute constraints.

### Accuracy

| Model | IMDb | SQuAD F1 |
|-------|------|----------|
| BERT-base (paper) | 93.46% | 88.5 |
| DistilBERT (paper) | 92.82% | 85.8 |
| BERT-base (ours) | **94.16%** | TBD |
| DistilBERT (ours) | **93.00%** | TBD |

> DistilBERT retains **98.8%** of BERT accuracy on IMDb in our experiments.

### Model Size & Speed

| Model | Parameters | Size | Speed |
|-------|------------|------|-------|
| BERT-base | 110M | ~418MB | 1x |
| DistilBERT | 66M | ~255MB | ~2x faster |

---

## Extension: Layer Ablation Study

The original DistilBERT uses 6 transformer layers. We investigate whether further compression is possible by reducing to 4 or 3 layers, and measure the accuracy/speed/size trade-off on IMDb.

### Results

| Layers | Accuracy | Parameters | Size | Speed |
|--------|----------|------------|------|-------|
| 6 (DistilBERT) | 93.21% | 66.96M | 255MB | 5040 samples/s |
| 4 | 91.99% | 52.78M | 201MB | 6734 samples/s |
| 3 | 91.01% | 45.69M | 174MB | 8228 samples/s |

**Key finding:** Reducing from 6 to 4 layers saves ~21% parameters and runs ~34% faster, while losing only ~1.2% accuracy. The 3-layer model achieves ~63% more speed than the original but drops ~2.2% in accuracy.

See `notebooks/extension.ipynb` for full implementation and plots.

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
