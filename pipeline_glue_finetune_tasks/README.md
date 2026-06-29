# Reproduce DistilBERT GLUE Benchmark

This part of the reproduction project implements the fine-tuning on the GLUE tasks for DistilBERT and BERT as described in "DistilBERT, a distilled version of BERT" (Sanh et al., 2019).

## Author
Christine Hubinger

## Key Points

- **Multiple seeds**: Median of 5 runs with different seeds
- **All GLUE tasks**: SST-2 was used for prototying, finally all tasks implemented 
- **Models**: DistilBERT, BERT-base
- **Results**: metrics saved as JSON Files

## Structure

```
datasets/                             # GLUE Benchmark specific data
  AX.tsv                              # Diagnostic dataset from GLUE

notebooks/
  01-setup.ipynb                      # Environment setup and dependency checks
  03-finetune-glue.ipynb              # GLUE fine-tuning with multiple seeds
  04-evaluate.ipynb                   # Results aggregation and analysis

distilbert_repro/
  utils.py                            # Common utilities (seed setting, paths)
  glue_benchmark.py                   # GLUE-specific classes and utilities

pipeline_glue_finetune_tasks/         # Pipeline for official GLUE Submission 
  configs/
    submission_config.json            # configuration file
  outputs/                            # Folder for submission-ready files
  01-generate-glue-submission.ipynb   # use official GLUE test-set on trained models
  README.md                           # additional information for submission task

requirements.txt                      # Python dependencies
README.md                             # This file
```

## Final Results 

Compare the scores for each dataset with the reproduced scores. 

| Model   | Score Avg. | CoLa | MNLI*) | MRPC | QNLI | QQP  | RTE  | SST-2 | STS-B*) | WNLI |
|---------|------|------|------|------|------|------|------|-------|-------|------|
|Bert-base (paper) | 79.5 | 56.3 | 86.7 | 88.6 | 91.8 | 89.6 | 69.3 | 92.7 | 89.0 | 53.5 | 
|Bert-base (ours)| 79.4 | 57.6 | 84.6 | 86.5 | 91.4 | 91.0 | 66.8 | 92.3 |  89.0 | 55.0 | 
|DistilBERT (paper) | 77.0 | 51.3 | 82.2 | 87.5 | 89.2 | 88.5 | 59.9 | 91.3 | 86.9 | 56.3 |
|DistilBERT (ours)| 76.8 | 51.1 | 82.2 | 85.1 | 88.6 | 90.2 | 59.6 | 91.0  | 87.0 | 56.3 |  
|DistilBERT (GLUE*)| 75.5 | 51.1 | 82.5 | 82.7 | 88.3 | 88.2 | 56.6 | 92.4 | 82.6 | 65.1 |  


(GLUE*) GLUE Scores from submission, median over 5 submissions with the 5 trained models (difference was only the seed). 

As it was not described clearly in the paper, how they made the GLUE submissions and how they aggregated their results, we just took all 5 with different seeds trained models and submitted them to GLUE to get the official test scores. The values here are a simple average.  

*) Limitations:
Due to our assumptions on MNLI and STS-B datasets the these numbers and also the overall average may not be comparable.



## Task Description                                

There are different scores used for the datasets following **GLUE: A MULTI-TASK BENCHMARK AND ANALYSIS PLATFORM FOR NATURAL LANGUAGE UNDERSTANDING (Wang et al.)** https://arxiv.org/pdf/1804.07461:

| Task  | Description                                      | Task Type                     | Input Format                         | Metric                |
|-------|--------------------------------------------------|-------------------------------|--------------------------------------|-----------------------|
| CoLA  | Linguistic acceptability judgment                | Binary classification         | Single sentence                      | Matthews corr. (MCC)  |
| MNLI  | Multi-genre natural language inference           | 3-way classification (NLI)    | Sentence pair (premise, hypothesis)  | Accuracy              |
| MRPC  | Paraphrase detection of sentence pairs           | Binary classification         | Sentence pair (sentence1, sentence2) | Accuracy              |
| QNLI  | Question-answering reformulated as NLI           | Binary classification (NLI)   | Question + sentence pair             | Accuracy              |
| QQP   | Duplicate question detection (Quora)             | Binary classification         | Question pair                        | Accuracy              |
| RTE   | Recognizing textual entailment                   | Binary classification (NLI)   | Sentence pair (premise, hypothesis)  | Accuracy              |
| SST-2 | Sentiment analysis (movie reviews)               | Binary classification         | Single sentence                      | Accuracy              |
| STS-B | Semantic textual similarity (score 0–5)          | Regression                    | Sentence pair                        | Pearson               |
| WNLI  | Winograd schema coreference resolution (NLI)     | Binary classification         | Sentence pair (with coreference)     | Accuracy              |


Limitations:
The exact configuration of the fine-tuning was not documented, so we used some default hyperparameters:

- Number Training epochs: 3
- Batch size: 16
- Learning rate: 2e-5 (0.00002)

STS-B: GLUE reports a Pearson and Spearman correlation. In the paper, only one number shows up. It's not clearly defined, how this is calculated. We assumed that only Pearson Correlation was used. 

MNLI: GLUE reports two results for this dataset: for "matched" and "unmatched". It was not clearly defined in the paper which one was used, or how to aggregate this. We only validated during training with the "matched" validations split. 


## How to run

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run Fine-Tuning for GLUE Benchmark

The workflow for fine-tuning consist of several notebooks:

#### Notebook ```01-setup.ipynb```: Setup

This notebook installs all necessary packages and dependencies. 

#### Notebook ```03-finetune-glue.ipynb```: Fine-tune on GLUE Tasks
This notebook trains DistilBERT and BERT on GLUE tasks.

Edit the **Configuration**! The provided notebook runs all taks and both models - this will take hours (or better: days). 

**Run**:
Runnig the notebook for all datasets and models needs several hours. 
For a quick test 
- reduce the dataset, for example only TASKS_TO_RUN = "sst2" (quite small dataset), 
- set NUM_EPOCHS = 2
- TRAIN_BATCH_SIZE = 32  # Larger for speed.    

#### Notebook ```04-evaluate.ipynb```: Evaluate and Analyse Results
This notebook aggregates results, computes statistics, and compares with paper results.

**Features**:
- Loads all fine-tuning results
- Computes median, mean, std, min, max across seeds
- Creates summary tables and model comparisons
- Validates against published results (paper reference)
- Exports consolidated JSON

## Sourcecode

### The GLUE_Benchmark.py Module

Core classes for GLUE benchmarking:

#### TaskConfig

Configuration for each GLUE task is set in GLUE_TASK_CONFIG. The TaskConfig class describes the individual datasets, how they can be trained and measured. 


#### GLUEDataPreprocessor
Tokenizes datasets for fine-tuning: Uses the tokenizer of the pre-trained model for each dataset. The class decides based on the description of the task how to preprocess the data: Eg. SST-2 has only one sentence, STS-B has sentence pairs to consider.   

The preprocessor imports the training data from the dataset and fixes column name issues to make automation easier. 


#### GLUEBenchmarkRunner
Main training orchestration. The functions run_benchmark and fine_tune_task bring all together and run the training following the defined rules and calculates the metrics.



### Advanced Features

#### Multi-Seed Statistical Analysis
Results are automatically aggregated:
- **Median**: Primary metric (as in paper)
- **Mean**: Average across seeds
- **Std**: Standard deviation
- **Min/Max**: Range

#### Reproducibility
- All random seeds (PyTorch, NumPy, Python) are set
- Deterministic evaluation
- Full hyperparameter logging
- Results saved to JSON

#### Dataset Download Issues
Datasets are cached after first download in `~/.cache/huggingface/datasets/`


## Results Format

Results are saved to `outputs/glue_benchmark/`:

This GitHub repository only contains the summary JSON files. There is just not enough space to save all trained models (9 x 5) in this repository. 

```
outputs/glue_benchmark/
  distilbert-base-uncased/
    sst2/
      seed_19/
      seed_42/
      ...
    benchmark_results.json
  bert-base-uncased/
    ...
    benchmark_results.json
  results_distilbert-base-uncased.json
  results_bert-base-uncased.json
  consolidated_results.json          # All results + statistics
```

## Future Work

**Distillation**: Instead of using pretrained models, this approach could also be applied. However, it was not done in this project due to the required computational resources and time.

## References

- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- GLUE Benchmark: https://gluebenchmark.com/
- GLUE: A MULTI-TASK BENCHMARK AND ANALYSIS PLATFORM FOR NATURAL LANGUAGE UNDERSTANDING (Dataset description): https://arxiv.org/pdf/1804.07461)
- DistilBERT Paper: https://arxiv.org/abs/1910.01108