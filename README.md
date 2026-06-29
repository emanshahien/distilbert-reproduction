# DistilBERT: Reproduction & Extension

> Reproduction and extension of [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019)
>
> TU Wien · 192.039 Deep Learning for NLP (VU 4,0) 2026S

---

## Paper Summary

DistilBERT is a compressed version of BERT-base that retains 97% of its language understanding performance while being 40% smaller and 60% faster. It achieves this through **knowledge distillation** during pre-training, using a triple loss: soft cross-entropy (distillation), cosine embedding (hidden state alignment), and masked language modeling (MLM).

---

## Team

| Name | Task |
|------|------|
| Eman | Repo setup + IMDb sentiment |
| Christine | SST-2 + full GLUE benchmark |
| Djem | SQuAD v1.1 QA |
| Sneha | Extension — Layer Ablation Study |

---

## Repository Structure

```
distilbert-reproduction/
├── README.md
├── requirements.txt
├── .gitignore
│
├── pipeline_glue_finetune_tasks/      # Christine — GLUE benchmark
│   ├── notebooks/
│   │   ├── 01-setup.ipynb
│   │   ├── 03-finetune-glue.ipynb
│   │   └── 04-evaluate.ipynb
│   ├── distilbert_repro/
│   │   ├── utils.py
│   │   └── glue_benchmark.py
│   ├── datasets/
│   ├── glue_official_submission/
│   ├── requirements.txt
│   └── README.md
│
├── notebooks/
│   ├── pipeline_imdb.ipynb            # Eman — IMDb sentiment
│   ├── pipeline_squad.ipynb           # Djem — SQuAD v1.1
│   ├── comparison.ipynb               # Final comparison table & plots
│   └── extension.ipynb                # Sneha — Layer Ablation Study
│
├── src/
│   ├── train.py                       # Shared training loop
│   ├── evaluate.py                    # Evaluation utilities
│   └── utils.py                       # Helpers (model size, speed, saving results)
│
└── results/                           # All experiment outputs saved here as JSON
```

---

## Code Structure (`src/`)

The `src/` folder contains shared helper code used by the IMDb, SQuAD, and extension notebooks.

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

> Christine's GLUE pipeline uses its own utilities in `pipeline_glue_finetune_tasks/distilbert_repro/`. See the README inside that folder for details.

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

For the GLUE pipeline, see `pipeline_glue_finetune_tasks/README.md` for its own setup instructions.

### On Google Colab / JupyterHub

Each notebook is self-contained and runs top-to-bottom on a GPU environment.

1. Open the notebook
2. Enable GPU if on Colab: **Runtime → Change runtime type → T4 GPU**
3. Click **Runtime → Run all**

### Order of execution

```
pipeline_glue_finetune_tasks/  ──────────────────────────────────► results/
pipeline_imdb.ipynb            ──┐
pipeline_squad.ipynb           ──┼──► results/  ──► comparison.ipynb
                                 ┘
extension.ipynb  ──► results/  (layer ablation results saved separately)
```

---

## Reproduction Results

We reproduce fine-tuning results using released pre-trained weights from HuggingFace. Full pre-training is not reproduced due to compute constraints.

### GLUE Benchmark (Christine)

| Model | Score Avg. | CoLa | MNLI*) | MRPC | QNLI | QQP | RTE | SST-2 | STS-B*) | WNLI |
|-------|------------|------|--------|------|------|-----|-----|-------|---------|------|
|BERT-base (paper) | 79.5 | 56.3 | 86.7 | 88.6 | 91.8 | 89.6 | 69.3 | 92.7 | 89.0 | 53.5 | 
|BERT-base (ours)| 79.4 | 57.6 | 84.6 | 86.5 | 91.4 | 91.0 | 66.8 | 92.3 |  89.0 | 55.0 | 
|DistilBERT (paper) | 77.0 | 51.3 | 82.2 | 87.5 | 89.2 | 88.5 | 59.9 | 91.3 | 86.9 | 56.3 |
|DistilBERT (ours)| 76.8 | 51.1 | 82.2 | 85.1 | 88.6 | 90.2 | 59.6 | 91.0  | 87.0 | 56.3 |  
|DistilBERT (GLUE*)| 75.5 | 51.1 | 82.5 | 82.7 | 88.3 | 88.2 | 56.6 | 92.4 | 82.6 | 65.1 |  

(GLUE*) GLUE Scores from submission, median over 5 submissions with the 5 trained models (different seeds). 

> Results are the median of 5 runs with different seeds. See `pipeline_glue_finetune_tasks/README.md` for full methodology and limitations (Due to our assumptions on MNLI and STS-B datasets the these numbers and also the overall average may not be comparable).


### IMDb Sentiment (Eman)

| Model | IMDb Accuracy |
|-------|---------------|
| BERT-base (paper) | 93.46% |
| DistilBERT (paper) | 92.82% |
| BERT-base (ours) | **94.16%** |
| DistilBERT (ours) | **93.00%** |

> DistilBERT retains **98.8%** of BERT accuracy on IMDb in our experiments.

### SQuAD v1.1 (Djem)

| Model | Paper EM / F1 | Ours EM / F1 |
|-------|---------------|--------------|
| BERT-base | 81.2 / 88.5 | **81.00 / 88.39** |
| DistilBERT | 77.7 / 85.8 | **77.14 / 85.30** |

### Model Size & Speed

| Model | Parameters | Size | Speed |
|-------|------------|------|-------|
| BERT-base | 110M | ~418MB | 1x |
| DistilBERT | 66M | ~255MB | ~2x faster |

---

## Extension: Layer Ablation Study (Sneha)

The original DistilBERT uses 6 transformer layers. We investigate whether further compression is possible by reducing to 4 or 3 layers, measuring the accuracy/speed/size trade-off on IMDb.

### Results

| Layers | Accuracy | Parameters | Size | Inference Speed |
|--------|----------|------------|------|-----------------|
| 6 (DistilBERT) | 93.21% | 66.96M | 255MB | 5040 samples/s |
| 4 | 91.99% | 52.78M | 201MB | 6734 samples/s |
| 3 | 91.01% | 45.69M | 174MB | 8228 samples/s |

**Key finding:** Reducing from 6 to 4 layers saves ~21% parameters and runs ~34% faster, with only ~1.2% accuracy loss. The 3-layer model achieves ~63% more speed than the original but drops ~2.2% in accuracy.

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
