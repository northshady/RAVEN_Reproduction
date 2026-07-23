#!/usr/bin/env python3
"""Generate README figures from a completed RAVEN run and full evaluations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PAPER_FULL_FRAME_MAP = 0.95


def public_evaluation(result: dict) -> dict:
    """Remove machine-specific paths before writing a repository artifact."""
    cleaned = deepcopy(result)
    if "checkpoint" in cleaned:
        cleaned["checkpoint"] = Path(cleaned["checkpoint"]).name
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="assets/results")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(run_dir / "history.csv")
    train = json.loads((run_dir / "raven_full_train.json").read_text())
    test = json.loads((run_dir / "raven_full_val.json").read_text())
    epochs = history["epoch"] + 1

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    axis.plot(epochs, history["train_loss"], label="Train loss", linewidth=2.0)
    axis.plot(
        epochs,
        history["val_loss"],
        label="Held-out test loss (validation split)",
        linewidth=2.0,
    )
    axis.axvline(
        test["checkpoint_epoch"],
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label=f"Reported checkpoint (epoch {test['checkpoint_epoch']})",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted multitask loss (log scale)")
    axis.set_title("RAVEN random 80/20 split: training and held-out loss")
    axis.legend(frameon=True)
    axis.grid(True, which="both", alpha=0.25)
    figure.savefig(output_dir / "random80-20-loss.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    labels = ["Train", "Held-out test"]
    values = [train["mAP"], test["mAP"]]
    bars = axis.bar(labels, values, width=0.55, color=["#4C78A8", "#F58518"])
    axis.axhline(
        PAPER_FULL_FRAME_MAP,
        color="#C44E52",
        linestyle="--",
        linewidth=1.8,
        label=f"Paper full-frame claim ({PAPER_FULL_FRAME_MAP:.2f})",
    )
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("mAP")
    axis.set_title("RAVEN full-evaluation mAP at selected checkpoint")
    axis.legend(loc="lower left", frameon=True)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "random80-20-map.png", dpi=180)
    plt.close(figure)

    summary = {
        "run_name": run_dir.name,
        "epochs": int(len(history)),
        "final_train_loss": float(history.iloc[-1]["train_loss"]),
        "final_test_loss": float(history.iloc[-1]["val_loss"]),
        "selected_checkpoint_epoch": int(test["checkpoint_epoch"]),
        "train_full_evaluation": public_evaluation(train),
        "test_full_evaluation": public_evaluation(test),
    }
    (output_dir / "random80-20-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote report figures and summary to {output_dir}")


if __name__ == "__main__":
    main()
