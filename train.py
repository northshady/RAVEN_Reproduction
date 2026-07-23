#!/usr/bin/env python3
"""Train RAVEN on label-aligned RADIal raw ADC."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from raven_repro.config import load_config, save_config
from raven_repro.data import (
    RADIalADCDataset,
    RangeAngleEncoder,
    SplitBundle,
    create_dataloaders,
    create_split_bundle,
)
from raven_repro.models import RAVENADC
from raven_repro.runtime import resolve_device, seed_everything
from raven_repro.training import build_segmentation_loss, evaluate_epoch, train_epoch
from raven_repro.training.checkpoint import load_checkpoint, save_checkpoint


def build_optimizer(model, config: dict):
    name = config["optimizer"]["name"].lower()
    if name != "adam":
        raise ValueError(f"Unsupported optimizer: {name!r}")
    return torch.optim.Adam(model.parameters(), lr=float(config["optimizer"]["lr"]))


def build_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    name = scheduler_config["type"].lower()
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config["step_size"]),
            gamma=float(scheduler_config["gamma"]),
        )
    raise ValueError(f"Unsupported scheduler: {name!r}")


def write_history(history: list[dict], run_dir: Path) -> None:
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    if history:
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


def epoch_checkpoint_name(epoch: int, val_metrics: dict, include_metrics: bool) -> str:
    """Build a sortable checkpoint name containing the validation summary."""
    name = f"epoch_{epoch + 1:03d}_valloss_{float(val_metrics['loss']):.4f}"
    if include_metrics:
        name += (
            f"_P_{float(val_metrics['precision']):.4f}"
            f"_R_{float(val_metrics['recall']):.4f}"
            f"_F1_{float(val_metrics['f1']):.4f}"
            f"_mIoU_{float(val_metrics['miou']):.4f}"
            f"_Bal_{float(val_metrics['balanced_score']):.4f}"
        )
    return name + ".pth"


def checkpoint_state(
    *, model, optimizer, scheduler, scaler, epoch, history, best, config, split
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "history": history,
        "best": best,
        "config": {key: value for key, value in config.items() if key != "config_path"},
        "split": split.as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_root:
        config["paths"]["dataset_root"] = str(Path(args.dataset_root).expanduser().resolve())
    if args.output_dir:
        config["paths"]["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["training"].get("cudnn_benchmark", True))

    encoder = RangeAngleEncoder(
        config["dataset"]["geometry"], config["dataset"]["statistics"]
    )
    dataset = RADIalADCDataset(
        config["paths"]["dataset_root"],
        encoder.encode,
        include_difficult=bool(config["dataset"].get("include_difficult", True)),
    )

    resume_state = load_checkpoint(args.resume) if args.resume else None
    if resume_state:
        split = SplitBundle(**resume_state["split"])
        run_dir = Path(args.resume).expanduser().resolve().parent
    else:
        split = create_split_bundle(dataset, config["split"], seed)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        run_dir = Path(config["paths"]["output_dir"]) / f"{config['experiment_name']}__{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        save_config(config, run_dir / "config.json")
        (run_dir / "split.json").write_text(
            json.dumps(split.as_dict(), indent=2) + "\n", encoding="utf-8"
        )

    loaders = create_dataloaders(dataset, split, config["loader"], seed)
    model = RAVENADC(config["model"]).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    amp = bool(config["training"].get("amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    segmentation_criterion = build_segmentation_loss(config["loss"]["segmentation"])

    history: list[dict] = []
    best = {"val_loss": float("inf"), "f1": -1.0, "miou": -1.0, "balanced": -1.0}
    start_epoch = 0
    if resume_state:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        scaler.load_state_dict(resume_state.get("scaler_state_dict", {}))
        history = resume_state["history"]
        best.update(resume_state.get("best", {}))
        start_epoch = int(resume_state["epoch"]) + 1

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}; AMP: {amp}; parameters: {parameter_count:,}")
    print(
        f"Split: train={len(split.train)}, val={len(split.val)}, test={len(split.test)}, "
        f"test_aliases_val={split.test_aliases_val}"
    )
    print("Input: unscaled raw ADC counts; DC removal: false")

    writer = SummaryWriter(run_dir / "tensorboard")
    epochs = int(config["training"]["epochs"])
    metrics_start = int(config["training"].get("metrics_start_epoch", 0))
    save_every = int(config["training"].get("save_every", 1))
    gradient_clip = config["training"].get("gradient_clip")

    for epoch in range(start_epoch, epochs):
        train_metrics = train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            segmentation_criterion,
            config["loss"],
            device,
            amp=amp,
            gradient_clip=gradient_clip,
            epoch=epoch,
        )
        compute_metrics = epoch >= metrics_start
        val_metrics = evaluate_epoch(
            model,
            loaders["val"],
            encoder,
            segmentation_criterion,
            config["loss"],
            config["evaluation"],
            device,
            amp=amp,
            compute_metrics=compute_metrics,
            description=f"val {epoch:03d}",
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        write_history(history, run_dir)
        for key, value in row.items():
            if key != "epoch" and value is not None:
                writer.add_scalar(key, value, epoch)

        improved_val_loss = float(val_metrics["loss"]) < best["val_loss"]
        if improved_val_loss:
            best["val_loss"] = float(val_metrics["loss"])
        improved = {}
        if compute_metrics:
            selections = {
                "f1": ("best_f1.pth", float(val_metrics["f1"])),
                "miou": ("best_miou.pth", float(val_metrics["miou"])),
                "balanced": ("best_balanced.pth", float(val_metrics["balanced_score"])),
            }
            for key, (filename, value) in selections.items():
                if value > best[key]:
                    best[key] = value
                    improved[key] = filename

        state = checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            history=history,
            best=best.copy(),
            config=config,
            split=split,
        )
        save_checkpoint(state, run_dir / "last.pth")
        if save_every > 0 and (epoch + 1) % save_every == 0:
            save_checkpoint(
                state,
                run_dir / epoch_checkpoint_name(epoch, val_metrics, compute_metrics),
            )
        if improved_val_loss:
            save_checkpoint(state, run_dir / "best_val_loss.pth")
        for filename in improved.values():
            save_checkpoint(state, run_dir / filename)

        metric_text = (
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
            f"F1={val_metrics['f1']:.4f} mIoU={val_metrics['miou']:.4f} "
            f"balanced={val_metrics['balanced_score']:.4f} "
            f"TP={val_metrics['tp']} FP={val_metrics['fp']} FN={val_metrics['fn']}"
            if compute_metrics
            else "metrics=deferred"
        )
        print(
            f"Epoch {epoch + 1:03d}/{epochs}: train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_cls={val_metrics['classification_loss']:.4f} "
            f"val_reg={val_metrics['regression_loss']:.4f} "
            f"val_seg={val_metrics['segmentation_loss']:.4f} {metric_text} "
            f"lr={optimizer.param_groups[0]['lr']:.8f}"
        )
    writer.close()


if __name__ == "__main__":
    main()
