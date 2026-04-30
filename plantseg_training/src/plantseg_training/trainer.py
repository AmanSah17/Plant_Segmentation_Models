from __future__ import annotations

import json
import inspect
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from plantseg_training.data import make_datasets, segmentation_collate_fn
from plantseg_training.metrics import compute_binary_segmentation_metrics, compute_multiclass_segmentation_metrics
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


def _log_params_safely(params: dict[str, Any]) -> None:
    active_run = mlflow.active_run()
    if active_run is None:
        return
    client = mlflow.tracking.MlflowClient()
    existing = client.get_run(active_run.info.run_id).data.params
    for key, value in params.items():
        value_str = str(value)
        if len(value_str) > 500:
            value_str = f"<{type(value).__name__} length={len(value)}>"
        if key not in existing:
            mlflow.log_param(key, value_str)
        elif existing[key] != value_str:
            mlflow.log_param(f"{key}.override", value_str)


def _resolve_run_name(config: dict[str, Any]) -> str:
    base_name = config.get("run_name") or config.get("model", {}).get("name", "plantseg-run")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base_name}-{stamp}"


def _auto_pos_weight(train_dataset: Any, max_pos_weight: float = 25.0) -> tuple[float, float]:
    mask_ratio = train_dataset.frame["Mask ratio"].astype(float).clip(lower=1e-6, upper=1.0)
    positive_ratio = float(mask_ratio.mean())
    pos_weight = (1.0 - positive_ratio) / positive_ratio
    return float(min(pos_weight, max_pos_weight)), positive_ratio


def _load_class_weights(path: str | None, num_classes: int) -> list[float] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    weights = payload.get("class_weights", payload)
    if len(weights) != num_classes:
        raise ValueError(f"Expected {num_classes} class weights in {path}, found {len(weights)}")
    return [float(weight) for weight in weights]


def _summarize_loss_cfg(loss_cfg: dict[str, Any]) -> dict[str, Any]:
    summary = dict(loss_cfg)
    class_weights = summary.get("class_weights")
    if isinstance(class_weights, list):
        summary["class_weights"] = {
            "count": len(class_weights),
            "min": round(float(min(class_weights)), 6),
            "max": round(float(max(class_weights)), 6),
            "mean": round(float(sum(class_weights) / len(class_weights)), 6),
        }
    return summary


class PlantSegMonitoringCallback(TrainerCallback):
    def __init__(self, history_path: Path) -> None:
        self.history_path = history_path
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_path.exists():
            self.history_path.write_text("step,epoch,key,value\n", encoding="utf-8")

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: dict[str, float] | None = None, **_: Any) -> None:
        if not logs:
            return
        numeric_logs = {key: float(value) for key, value in logs.items() if isinstance(value, (int, float))}
        if numeric_logs:
            mlflow.log_metrics(numeric_logs, step=int(state.global_step))
            with self.history_path.open("a", encoding="utf-8") as f:
                for key, value in numeric_logs.items():
                    f.write(f"{state.global_step},{state.epoch or 0.0:.6f},{key},{value}\n")

        if "loss" in numeric_logs:
            print(
                f"[train] epoch={state.epoch or 0.0:.2f}/{args.num_train_epochs} "
                f"step={state.global_step}/{state.max_steps} loss={numeric_logs['loss']:.6f}",
                flush=True,
            )

    def on_epoch_end(self, args: TrainingArguments, state: Any, control: Any, **_: Any) -> None:
        print(f"[epoch] completed {state.epoch or 0.0:.2f}/{args.num_train_epochs}", flush=True)


class PlantSegTrainer(Trainer):
    def __init__(self, *args: Any, class_balanced_sampling: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_balanced_sampling = class_balanced_sampling

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if not self.class_balanced_sampling:
            return super().get_train_dataloader()

        class_counts = self.train_dataset.frame["Disease"].value_counts()
        sample_weights = self.train_dataset.frame["Disease"].map(lambda label: 1.0 / class_counts[label]).to_numpy()
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


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
        "max_steps": training_cfg.get("max_steps", -1),
        "max_grad_norm": training_cfg.get("max_grad_norm", 1.0),
        "logging_steps": training_cfg.get("logging_steps", 25),
        "eval_accumulation_steps": training_cfg.get("eval_accumulation_steps"),
        "save_strategy": training_cfg.get("save_strategy", "epoch"),
        "save_total_limit": training_cfg.get("save_total_limit", 3),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "eval_dice"),
        "greater_is_better": training_cfg.get("greater_is_better", True),
        "load_best_model_at_end": True,
        "remove_unused_columns": False,
        "report_to": training_cfg.get("report_to", ["tensorboard"]),
        "logging_strategy": "steps",
        "disable_tqdm": bool(training_cfg.get("disable_tqdm", False)),
        "dataloader_num_workers": training_cfg.get("num_workers", 0),
        "dataloader_pin_memory": training_cfg.get("pin_memory", True),
        "fp16": bool(training_cfg.get("fp16", False)),
        "run_name": training_cfg.get("run_name"),
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = training_cfg.get("eval_strategy", "epoch")
    else:
        kwargs["evaluation_strategy"] = training_cfg.get("eval_strategy", "epoch")
    return TrainingArguments(**kwargs)


def run_training(config: dict[str, Any], config_path: Path | None = None) -> None:
    set_seed(int(config.get("seed", 42)))
    torch.backends.cudnn.benchmark = bool(config.get("training", {}).get("cudnn_benchmark", True))

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

    train_dataset, val_dataset, test_dataset = make_datasets(dataset_cfg)
    model_cfg = dict(config["model"])
    loss_cfg = dict(model_cfg.get("loss", {}))
    task = dataset_cfg.get("task", "binary_segmentation")
    if task == "binary_segmentation" and loss_cfg.get("pos_weight", "auto") == "auto":
        pos_weight, positive_ratio = _auto_pos_weight(
            train_dataset,
            max_pos_weight=float(loss_cfg.get("max_pos_weight", 25.0)),
        )
        loss_cfg["pos_weight"] = pos_weight
        loss_cfg["train_positive_ratio"] = positive_ratio
    if task == "multiclass_segmentation":
        num_classes = int(model_cfg["num_classes"])
        if loss_cfg.get("class_weights", "auto") == "auto":
            weights_path = dataset_cfg.get("class_weights_path")
            loss_cfg["class_weights"] = _load_class_weights(weights_path, num_classes)
            if loss_cfg["class_weights"] is None:
                raise ValueError("multiclass_segmentation with class_weights='auto' requires dataset.class_weights_path")
    model_cfg["loss"] = loss_cfg
    config["model"] = model_cfg

    model = build_model(config["model"])
    threshold = float(training_cfg.get("threshold", 0.5))

    run_name = _resolve_run_name(config)
    if bool(training_cfg.get("create_run_subdir", True)):
        base_output_dir = Path(training_cfg["output_dir"])
        training_cfg["output_dir"] = str(base_output_dir / "runs" / run_name)

    args_cfg = dict(training_cfg)
    args_cfg["num_workers"] = dataset_cfg.get("num_workers", 0)
    args_cfg["pin_memory"] = dataset_cfg.get("pin_memory", True)
    args_cfg["run_name"] = run_name
    training_args = _training_arguments(args_cfg)
    steps_per_epoch = (len(train_dataset) + training_args.per_device_train_batch_size - 1) // training_args.per_device_train_batch_size
    total_batch_steps = steps_per_epoch * int(float(training_args.num_train_epochs))
    print(
        f"Training plan: {training_args.num_train_epochs} epochs, "
        f"~{steps_per_epoch} dataloader batches/epoch, ~{total_batch_steps} total batch iterations.",
        flush=True,
    )
    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    optimizer_steps_per_epoch = (steps_per_epoch + training_args.gradient_accumulation_steps - 1) // training_args.gradient_accumulation_steps
    print(
        f"Effective batch size: {effective_batch_size} "
        f"({training_args.per_device_train_batch_size} batch x {training_args.gradient_accumulation_steps} accumulation). "
        f"Optimizer steps/epoch ~= {optimizer_steps_per_epoch}.",
        flush=True,
    )
    print(
        f"Loss summary: {_summarize_loss_cfg(model_cfg['loss'])}",
        flush=True,
    )
    print(
        f"CUDA check: available={torch.cuda.is_available()}, "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}, "
        f"torch_cuda={torch.version.cuda}, python={sys.executable}",
        flush=True,
    )
    print(f"Run output_dir: {training_cfg['output_dir']}", flush=True)
    print(f"MLflow tracking_uri: {mlflow_tracking_uri}", flush=True)

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        if task == "multiclass_segmentation":
            return compute_multiclass_segmentation_metrics(eval_pred, num_classes=int(model_cfg["num_classes"]))
        return compute_binary_segmentation_metrics(eval_pred, threshold=threshold)

    def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, tuple):
            logits = logits[0]
        if task == "multiclass_segmentation":
            return torch.argmax(logits, dim=1)
        return logits

    history_path = Path(training_cfg["output_dir"]) / "logs" / "training_history.csv"
    trainer = PlantSegTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=segmentation_collate_fn,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[PlantSegMonitoringCallback(history_path)],
        class_balanced_sampling=bool(training_cfg.get("class_balanced_sampling", True)),
    )

    with mlflow.start_run(run_name=args_cfg["run_name"], nested=False):
        _log_params_safely(_flatten_dict(config))
        _log_params_safely(
            {
                "run_name": args_cfg["run_name"],
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "steps_per_epoch": steps_per_epoch,
                "planned_total_batch_steps": total_batch_steps,
                "planned_optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        )
        if config_path is not None:
            mlflow.log_artifact(str(config_path), artifact_path="config")
        class_counts_path = Path(training_cfg["output_dir"]) / "logs" / "class_distribution.csv"
        class_counts_path.parent.mkdir(parents=True, exist_ok=True)
        train_dataset.frame["Disease"].value_counts().rename_axis("disease").reset_index(name="count").to_csv(
            class_counts_path,
            index=False,
        )
        mlflow.log_artifact(str(class_counts_path), artifact_path="data")

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
        mlflow.log_artifact(str(history_path), artifact_path="metrics")
