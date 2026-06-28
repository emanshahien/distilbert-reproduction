"""
GLUE Benchmark utilities for DistilBERT reproduction.

This module provides helpers for fine-tuning and evaluating on GLUE tasks.
Supports 9 GLUE datasets with configurable seed management.
"""

from __future__ import annotations

import inspect
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import evaluate


# GLUE task configurations
GLUE_TASK_CONFIG = {
    "sst2": {
        "dataset_name": "sst2",
        "num_labels": 2,
        "metric_name": "accuracy",
        "text_column": "sentence",
        "sentence_a_column": None,
        "sentence_b_column": None,
    },
    "mrpc": {
        "dataset_name": "mrpc",
        "num_labels": 2,
        "metric_name": "accuracy",
        "text_column": None,
        "sentence_a_column": "sentence1",
        "sentence_b_column": "sentence2",
    },
    "qqp": {
        "dataset_name": "qqp",
        "num_labels": 2,
        "metric_name": "accuracy",
        "text_column": None,
        "sentence_a_column": "question1",
        "sentence_b_column": "question2",
    },
    "stsb": {
        "dataset_name": "stsb",
        "num_labels": 1,
        "metric_name": "pearsonr",
        "text_column": None,
        "sentence_a_column": "sentence1",
        "sentence_b_column": "sentence2",
        "is_regression": True,
    },
    "cola": {
        "dataset_name": "cola",
        "num_labels": 2,
        "metric_name": "matthews_correlation",
        "text_column": "sentence",
        "sentence_a_column": None,
        "sentence_b_column": None,
    },
    "mnli": {
        "dataset_name": "mnli",
        "num_labels": 3,
        "metric_name": "accuracy",
        "text_column": None,
        "sentence_a_column": "premise",
        "sentence_b_column": "hypothesis",
    },
    "qnli": {
        "dataset_name": "qnli",
        "num_labels": 2,
        "metric_name": "accuracy",
        "text_column": None,
        "sentence_a_column": "question",
        "sentence_b_column": "sentence",
    },
    "rte": {
        "dataset_name": "rte",
        "num_labels": 2,
        "metric_name": "accuracy",
        "text_column": None,
        "sentence_a_column": "sentence1",
        "sentence_b_column": "sentence2",
    },
    "wnli": {
        "dataset_name": "wnli",
        "num_labels": 2,
        "metric_name": "accuracy",
        "text_column": None,
        "sentence_a_column": "sentence1",
        "sentence_b_column": "sentence2",
    },
}


@dataclass
class TaskConfig:
    """Configuration for a GLUE task."""
    task_name: str
    dataset_name: str
    num_labels: int
    metric_name: str
    text_column: Optional[str]
    sentence_a_column: Optional[str]
    sentence_b_column: Optional[str]
    is_regression: bool = False

    @classmethod
    def from_task(cls, task_name: str) -> TaskConfig:
        """Create TaskConfig from task name."""
        if task_name not in GLUE_TASK_CONFIG:
            raise ValueError(f"Unknown GLUE task: {task_name}")
        cfg = GLUE_TASK_CONFIG[task_name]
        return cls(
            task_name=task_name,
            **{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__},
        )


class GLUEDataPreprocessor:
    """Preprocesses GLUE datasets for fine-tuning."""

    def __init__(self, task_name: str, tokenizer, max_length: int = 512):
        """Initialize preprocessor.

        Args:
            task_name: Name of GLUE task
            tokenizer: HF tokenizer instance
            max_length: Max sequence length after tokenization
        """
        self.task_config = TaskConfig.from_task(task_name)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess_single_sentence(self, examples):
        """Tokenize single-sentence task."""
        column = self.task_config.text_column
        return self.tokenizer(
            examples[column],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

    def preprocess_sentence_pair(self, examples):
        """Tokenize sentence-pair task."""
        col_a = self.task_config.sentence_a_column
        col_b = self.task_config.sentence_b_column
        return self.tokenizer(
            examples[col_a],
            examples[col_b],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

    def preprocess(self, dataset: DatasetDict) -> DatasetDict:
        """Apply tokenization to dataset.

        Args:
            dataset: HF DatasetDict with 'train', 'validation', 'test' splits

        Returns:
            Tokenized DatasetDict with 'labels' column renamed
        """
        if self.task_config.text_column:
            preprocess_fn = self.preprocess_single_sentence
        else:
            preprocess_fn = self.preprocess_sentence_pair

        # Keep label columns; remove only raw text/source columns.
        train_columns = dataset["train"].column_names
        columns_to_remove = [c for c in train_columns if c not in {"label", "label_ids"}]

        # Tokenize all splits
        encoded = dataset.map(
            preprocess_fn,
            batched=True,
            remove_columns=columns_to_remove,
        )

        # HF Trainer expects "labels"; normalize common label column names.
        sample_columns = encoded["train"].column_names
        if "label" in sample_columns and "labels" not in sample_columns:
            encoded = encoded.rename_column("label", "labels")
        elif "label_ids" in sample_columns and "labels" not in sample_columns:
            encoded = encoded.rename_column("label_ids", "labels")

        return encoded


class GLUEBenchmarkRunner:
    """Runs fine-tuning and evaluation for GLUE benchmarks."""

    def __init__(
        self,
        model_name: str,
        output_dir: Path = Path("./outputs/glue_results"),
        seeds: list[int] = None,
    ):
        """Initialize runner.

        Args:
            model_name: HF model identifier (e.g., 'distilbert-base-uncased')
            output_dir: Directory to save results
            seeds: List of random seeds for multiple runs
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = seeds or [42, 1337, 2022, 3142, 9999]
        self.results = defaultdict(list)

    def fine_tune_task(
        self,
        task_name: str,
        seed: int = 42,
        train_batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        eval_strategy: str = "epoch",
    ) -> dict:
        """Fine-tune model on a single GLUE task with a single seed.

        Args:
            task_name: GLUE task name
            seed: Random seed
            train_batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            eval_strategy: Evaluation strategy ('epoch', 'steps', 'no')

        Returns:
            Dictionary with final metrics
        """
        from distilbert_repro.utils import set_seed

        set_seed(seed)

        task_config = TaskConfig.from_task(task_name)

        # Load dataset
        dataset = load_dataset("glue", task_config.dataset_name)

        # Handle dataset structure (mnli has matched/mismatched: only use matched here)
        if task_name == "mnli":
            eval_dataset = dataset["validation_matched"]
        else:
            eval_dataset = dataset["validation"]

        # Preprocess
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        preprocessor = GLUEDataPreprocessor(task_name, tokenizer)
        encoded_dataset = preprocessor.preprocess(dataset)

        if task_name == "mnli":
            encoded_eval = encoded_dataset["validation_matched"]
        else:
            encoded_eval = encoded_dataset["validation"]

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=task_config.num_labels,
        )

        # Setup training
        task_output_dir = self.output_dir / task_name / f"seed_{seed}"
        task_output_dir.mkdir(parents=True, exist_ok=True)

        training_kwargs = {
            "output_dir": str(task_output_dir),
            "per_device_train_batch_size": train_batch_size,
            "per_device_eval_batch_size": 32,
            "num_train_epochs": num_epochs,
            "learning_rate": learning_rate,
            "save_strategy": eval_strategy,
            "load_best_model_at_end": True,
            "metric_for_best_model": task_config.metric_name,
            "logging_steps": 50,
            "seed": seed,
        }

        # transformers renamed this argument in newer versions.
        arg_names = inspect.signature(TrainingArguments.__init__).parameters
        if "eval_strategy" in arg_names:
            training_kwargs["eval_strategy"] = eval_strategy
        else:
            training_kwargs["evaluation_strategy"] = eval_strategy

        training_args = TrainingArguments(**training_kwargs)

        # Create trainer
        data_collator = DataCollatorWithPadding(tokenizer)

        def compute_metrics(eval_pred):
            """Compute task-specific metrics."""
            predictions, labels = eval_pred

            if task_config.metric_name == "accuracy":
                predictions = np.argmax(predictions, axis=1)
                metric = evaluate.load("accuracy")
                return metric.compute(predictions=predictions, references=labels)

            elif task_config.metric_name == "f1":
                predictions = np.argmax(predictions, axis=1)
                metric = evaluate.load("f1", config_name="binary")
                return metric.compute(predictions=predictions, references=labels, average="binary")

            elif task_config.metric_name == "pearsonr":
                predictions = predictions.squeeze()
                metric = evaluate.load("pearsonr")
                return metric.compute(predictions=predictions, references=labels)

            elif task_config.metric_name == "matthews_correlation":
                predictions = np.argmax(predictions, axis=1)
                metric = evaluate.load("matthews_correlation")
                return metric.compute(predictions=predictions, references=labels)

            else:
                raise ValueError(f"Unknown metric: {task_config.metric_name}")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_eval,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train
        trainer.train()

        # Get best metrics with compatibility across transformers versions.
        if hasattr(trainer.state, "best_metrics") and trainer.state.best_metrics is not None:
            best_metrics = trainer.state.best_metrics
        else:
            best_metrics = dict(trainer.evaluate(eval_dataset=encoded_eval))
            if hasattr(trainer.state, "best_metric") and trainer.state.best_metric is not None:
                best_metrics["best_metric"] = trainer.state.best_metric
            if hasattr(trainer.state, "best_model_checkpoint") and trainer.state.best_model_checkpoint is not None:
                best_metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint

        return best_metrics

    def run_benchmark(
        self,
        task_names: list[str] = None,
        train_batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
    ) -> dict:
        """Run full benchmark with multiple seeds.

        Args:
            task_names: List of GLUE task names. Defaults to ['sst2']
            train_batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Dictionary with results for all tasks and seeds
        """
        task_names = task_names or ["sst2"]
        all_results = {}

        for task_name in task_names:
            print(f"\n{'='*60}")
            print(f"Running benchmark for {task_name.upper()}")
            print(f"{'='*60}")

            task_results = []
            for seed in self.seeds:
                print(f"\n  Seed: {seed}")
                metrics = self.fine_tune_task(
                    task_name,
                    seed=seed,
                    train_batch_size=train_batch_size,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                )
                task_results.append(metrics)
                print(f"    Metrics: {metrics}")

            all_results[task_name] = task_results

        # Save results
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\nResults saved to {results_path}")
        return all_results

    @staticmethod
    def aggregate_results(
        results: dict,
        metric_name: str = "eval_accuracy",
    ) -> dict:
        """Compute median and std of results across seeds.

        Args:
            results: Dictionary with results from run_benchmark
            metric_name: Name of metric to aggregate

        Returns:
            Dictionary with median and std for each task
        """
        aggregated = {}
        for task_name, task_results in results.items():
            scores = [r.get(metric_name, float("nan")) for r in task_results]
            scores = [s for s in scores if not np.isnan(s)]

            if scores:
                aggregated[task_name] = {
                    "median": float(np.median(scores)),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "scores": scores,
                }

        return aggregated
