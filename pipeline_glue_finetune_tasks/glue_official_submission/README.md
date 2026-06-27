# GLUE Official Submission

This folder contains a notebook workflow to generate official GLUE submission files from a fine-tuned model.

## Contents

- `01-generate-glue-submission.ipynb`: Main notebook to create submission TSV files and a ZIP package.
- `configs/submission_config.json`: User-editable config (defaults to SST-2).
- `requirements.txt`: Python dependencies.
- `outputs/`: Generated submission artifacts.

## Quick Start

1. Create/activate your Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Open and run `01-generate-glue-submission.ipynb`.
4. Edit `configs/submission_config.json`:
   - `model_path`: path to your trained checkpoint
   - `tasks_to_submit`: e.g. `["sst2"]`
   - For multi-task submission, prefer `task_model_paths` with one model path per task.
5. Run all notebook cells.

The notebook writes task-specific TSV files and a final ZIP file under `outputs/`.

## Notes

- Official GLUE test scores are returned by the GLUE server after submission; they cannot be computed locally:
   The Files where submitted manually on the GLUE site. Only 2 submission in 24 hours are accepted - so this took some days. 

- Development notes: SST-2 was used for prototyping, finally all GLUE task where calculated to be ready for GLUE submission.


## Full GLUE Setup

- `tasks_to_submit` can include all 9 GLUE tasks.
- `task_model_paths` should point each task to its own fine-tuned checkpoint/run directory. This was changed manually on the TU cluster - where the trained models are stored.  
- If tokenizer files are missing in a checkpoint folder, set `tokenizer_path` or per-task `task_tokenizer_paths`.
