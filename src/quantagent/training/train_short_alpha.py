from __future__ import annotations

import argparse
import random
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError("Training requires: pip install -e .[training]") from exc

from quantagent.data.dataset import EquityWindowDataset, WindowSpec
from quantagent.data.io import read_frame
from quantagent.models.alpha_transformer import AlphaTransformer
from quantagent.training.losses import alpha_multi_task_loss
from quantagent.training.metrics import information_coefficient_summary, rank_ic_by_date
from quantagent.training.walk_forward import TimeSplit, split_by_date


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the short-horizon alpha model.")
    parser.add_argument("--config", default="configs/training/short_alpha.yaml")
    args = parser.parse_args()
    config = _load_yaml(args.config)
    _seed_everything(config["training"].get("seed", 42))

    frame = read_frame(config["data"]["input_path"])
    split = TimeSplit(
        train_start=date.fromisoformat(config["training"]["train_start"]),
        train_end=date.fromisoformat(config["training"]["train_end"]),
        valid_start=date.fromisoformat(config["training"]["valid_start"]),
        valid_end=date.fromisoformat(config["training"]["valid_end"]),
    )
    train_frame, valid_frame = split_by_date(frame, split, config["data"]["date_column"])

    spec = WindowSpec(
        lookback_days=int(config["features"]["lookback_days"]),
        feature_columns=tuple(config["features"]["columns"]),
        label_columns=tuple(config["labels"]["columns"]),
        date_column=config["data"]["date_column"],
        symbol_column=config["data"]["symbol_column"],
    )
    train_dataset = EquityWindowDataset(train_frame, spec)
    valid_dataset = EquityWindowDataset(valid_frame, spec)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    model = AlphaTransformer(
        num_features=len(spec.feature_columns),
        d_model=int(model_config["d_model"]),
        nhead=int(model_config["nhead"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        output_dim=len(spec.label_columns),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = GradScaler("cuda", enabled=bool(config["training"].get("amp", True)) and device.type == "cuda")

    checkpoint_dir = Path(config["data"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_valid = float("inf")
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, scaler, device, train=True)
        valid_loss = _run_epoch(model, valid_loader, optimizer, scaler, device, train=False)
        rank_summary = _evaluate_rank_ic(model, valid_loader, device, spec)
        print(
            "epoch="
            f"{epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            f"rank_ic={rank_summary['rank_ic_mean']:.6f} icir={rank_summary['icir']:.6f}"
        )
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "valid_loss": valid_loss,
                    "epoch": epoch,
                },
                checkpoint_dir / "best.pt",
            )


def _run_epoch(model, loader, optimizer, scaler, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast(device_type=device.type, enabled=scaler.is_enabled() and device.type == "cuda"):
                pred = model(features)
                loss = alpha_multi_task_loss(pred, labels)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = features.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def _evaluate_rank_ic(model, loader, device, spec: WindowSpec) -> dict[str, float]:
    model.eval()
    target_index = _target_index(spec.label_columns)
    records = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].cpu().numpy()
            pred = model(features).detach().cpu().numpy()
            for symbol, trade_date, pred_row, label_row in zip(
                batch["symbol"],
                batch["trade_date"],
                pred,
                labels,
                strict=False,
            ):
                records.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "prediction": float(pred_row[target_index]),
                        "target": float(label_row[target_index]),
                    }
                )
    if not records:
        return {"rank_ic_mean": np.nan, "rank_ic_std": np.nan, "icir": np.nan}
    rank_ic = rank_ic_by_date(pd.DataFrame(records), "prediction", "target")
    return information_coefficient_summary(rank_ic)


def _target_index(label_columns: tuple[str, ...]) -> int:
    if "future_5d_excess_return" in label_columns:
        return label_columns.index("future_5d_excess_return")
    return 0


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
