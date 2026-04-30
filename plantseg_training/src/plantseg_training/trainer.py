from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
from typing import Any

import mlflow
import torch
from transformers import Trainer, TrainingArguments, set_seed

from plantseg_training.data import make_datasets, segmentation_collate_fn
from plantseg_training.metrics import compute_binary_segmentation_metrics
from plantseg_training.models import build_model


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(_flatten_dict(value, new_key))
        else:
            items[new_key] = value
    return items


def _training_arguments(training_cfg: dict[str, Any]) -> TrainingArguments:
    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": training_cfg.get("num_train_epochs", 20),
        "per_device_train_batch_size": training_cfg.get("per_device_train_batch_size", 8),
        "per_device_eval_batch_size": training_cfg.get("per_device_eval_batch_size", 8),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps", 1),
        "learning_rate": training_cfg.get("learning_rate", 1e-4),
        "weight_decay": training_cfg.get("weight_decay", 1e-4),
        "warmup_ratio": training_cfg.get("warmup_ratio", 0.0),
        "logging_steps": training_cfg.get("logging_steps", 25),
        "save_strategy": training_cfg.get("save_strategy", "epoch"),
        "save_total_limit": training_cfg.get("save_total_limit", 3),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "eval_dice"),
        "greater_is_better": training_cfg.get("greater_is_better", True),
        "load_best_model_at_end": True,
        "remove_unused_columns": False,
        "report_to": ["mlflow"],
        "dataloader_num_workers": training_cfg.get("num_workers", 0),
        "dataloader_pin_memory": training_cfg.get("pin_memory", True),
        "fp16": bool(training_cfg.get("fp16", torch.cuda.is_available())),
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = training_cfg.get("eval_strategy", "epoch")
    else:
        kwargs["evaluation_strategy"] = training_cfg.get("eval_strategy", "epoch")
    return TrainingArguments(**kwargs)


def run_training(config: dict[str, Any], config_path: Path | None = None) -> None:
    set_seed(int(config.get("seed", 42)))

    training_cfg = config["training"]
    if bool(training_cfg.get("require_cuda", True)) and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by this experiment config, but torch.cuda.is_available() is False. "
            "Activate F:\\PyTorch_GPU\\torch_gpu\\Scripts\\Activate.ps1 and verify the GPU build of PyTorch."
        )

    dataset_cfg = config["dataset"]
    dataset_cfg["num_workers"] = dataset_cfg.get("num_workers", 0)
    dataset_cfg["pin_memory"] = dataset_cfg.get("pin_memory", True)

    mlflow_tracking_uri = training_cfg.get("mlflow_tracking_uri", "plantseg_training/outputs/mlruns")
    mlflow.set_tracking_uri(str(Path(mlflow_tracking_uri)))
    mlflow.set_experiment(config.get("experiment_name", "plantseg-segmentation"))

    os.environ["MLFLOW_TRACKING_URI"] = str(Path(mlflow_tracking_uri))
    os.environ["MLFLOW_EXPERIMENT_NAME"] = config.get("experiment_name", "plantseg-segmentation")
    os.environ["MLFLOW_RUN_NAME"] = config.get("run_name", config["model"]["name"])

    train_dataset, val_dataset, test_dataset = make_datasets(dataset_cfg)
    model = build_model(config["model"])
    threshold = float(training_cfg.get("threshold", 0.5))

    args_cfg = dict(training_cfg)
    args_cfg["num_workers"] = dataset_cfg.get("num_workers", 0)
    args_cfg["pin_memory"] = dataset_cfg.get("pin_memory", True)
    training_args = _training_arguments(args_cfg)

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        return compute_binary_segmentation_metrics(eval_pred, threshold=threshold)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=segmentation_collate_fn,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name=config.get("run_name", config["model"]["name"]), nested=False):
        mlflow.log_params(_flatten_dict(config))
        mlflow.log_param("train_samples", len(train_dataset))
        mlflow.log_param("val_samples", len(val_dataset))
        mlflow.log_param("test_samples", len(test_dataset))
        mlflow.log_param("cuda_available", torch.cuda.is_available())
        mlflow.log_param("cuda_device_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
        if config_path is not None:
            mlflow.log_artifact(str(config_path), artifact_path="config")

        trainer.train()
        val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

        mlflow.log_metrics({f"final_{k}": float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({f"final_{k}": float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))})

        final_dir = Path(training_cfg["output_dir"]) / "final_model"
        trainer.save_model(str(final_dir))
        mlflow.log_artifacts(str(final_dir), artifact_path="model")

        metrics_path = Path(training_cfg["output_dir"]) / "final_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
