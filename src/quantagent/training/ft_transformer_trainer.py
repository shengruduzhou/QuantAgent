"""Governed FT-Transformer trainer facade.

The legacy implementation is retained byte-for-byte in
``ft_transformer_trainer_impl.py`` so the training path can be reviewed against
its previous semantics.  This facade changes one production-critical property:
checkpoint selection and early stopping now use the same pointwise + per-date
listwise objective family that training uses.

Existing model artifacts receive no trust from this code change.  The current
production model remains governed by ``configs/live_model_trust.json`` and must
be retrained/revalidated before any new evidence can be claimed.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.training import ft_transformer_trainer_impl as _impl


OBJECTIVE_SEMANTICS_VERSION = "ft_transformer_objective_v2_per_date_listwise_validation"

FTTransformerTrainerConfig = _impl.FTTransformerTrainerConfig
FTTransformerArtifacts = _impl.FTTransformerArtifacts
FTTransformerPredictionResult = _impl.FTTransformerPredictionResult
predict_ft_transformer_artifact = _impl.predict_ft_transformer_artifact

# Preserve private helper imports used by existing regression tests/research.
_auto_feature_columns = _impl._auto_feature_columns
_prepare = _impl._prepare
_resolve_device = _impl._resolve_device
_torch_device_report = _impl._torch_device_report
_softmax_listwise_loss = _impl._softmax_listwise_loss
_batched_loss = _impl._batched_loss
_batched_predict_tensor = _impl._batched_predict_tensor


def _numpy_huber_loss(predictions: np.ndarray, targets: np.ndarray, *, delta: float) -> float:
    """PyTorch-compatible mean Huber loss for validation diagnostics."""
    pred = np.asarray(predictions, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    if pred.shape != tgt.shape or pred.size == 0:
        raise ValueError("validation predictions/targets must be non-empty and shape-aligned")
    if not np.isfinite(pred).all() or not np.isfinite(tgt).all():
        return float("nan")
    d = float(delta)
    if not np.isfinite(d) or d <= 0:
        raise ValueError("huber delta must be finite and > 0")
    error = np.abs(pred - tgt)
    values = np.where(error <= d, 0.5 * error * error, d * (error - 0.5 * d))
    return float(values.mean())


def _numpy_listwise_portfolio_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    temperature: float,
) -> float:
    """Numerically stable no-grad equivalent of the training listwise loss."""
    pred = np.asarray(predictions, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[:, None]
        tgt = tgt[:, None]
    if pred.shape != tgt.shape or pred.shape[0] < 2:
        raise ValueError("listwise validation requires >=2 aligned rows")
    if not np.isfinite(pred).all() or not np.isfinite(tgt).all():
        return float("nan")
    temp = max(float(temperature), 1e-6)
    logits = pred / temp
    logits = logits - np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(logits)
    weights = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
    portfolio_return = np.sum(weights * tgt, axis=0)
    return float(-np.mean(portfolio_return))


def validation_objective_from_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    trade_dates: pd.Series | np.ndarray,
    *,
    huber_delta: float,
    rank_loss_weight: float,
    rank_loss_temperature: float,
) -> dict[str, float]:
    """Compute validation objective without mixing separate trade dates.

    Huber is averaged across all validation rows/horizons.  The listwise term is
    computed independently for every trade date with at least two names and is
    then averaged across dates, matching the date-grouped training semantics.
    Missing/invalid dates fail closed whenever rank loss is enabled.
    """
    pred = np.asarray(predictions)
    tgt = np.asarray(targets)
    if pred.shape != tgt.shape or pred.ndim not in (1, 2) or pred.shape[0] == 0:
        raise ValueError("validation predictions/targets must be non-empty and shape-aligned")
    huber = _numpy_huber_loss(pred, tgt, delta=huber_delta)
    weight = float(rank_loss_weight)
    if weight <= 0:
        return {
            "huber": float(huber),
            "rank": 0.0,
            "composite": float(huber),
            "rank_groups": 0.0,
        }

    dates = pd.Series(trade_dates).reset_index(drop=True)
    if len(dates) != pred.shape[0]:
        raise ValueError("validation trade_dates must align one-for-one with predictions")
    dates = pd.to_datetime(dates, errors="coerce")
    if dates.isna().any():
        raise ValueError("rank-loss validation requires valid trade_date for every row")

    rank_losses: list[float] = []
    for _, positions in dates.groupby(dates).groups.items():
        idx = np.asarray(list(positions), dtype=np.int64)
        if idx.size < 2:
            continue
        rank_losses.append(
            _numpy_listwise_portfolio_loss(
                pred[idx],
                tgt[idx],
                temperature=rank_loss_temperature,
            )
        )
    if not rank_losses:
        raise ValueError("rank-loss validation requires at least one trade date with >=2 names")
    rank = float(np.mean(rank_losses))
    composite = float(huber + weight * rank)
    return {
        "huber": float(huber),
        "rank": rank,
        "composite": composite,
        "rank_groups": float(len(rank_losses)),
    }


def _validation_objective(
    model: object,
    features: "torch.Tensor",
    targets: "torch.Tensor",
    trade_dates: pd.Series | None,
    loss_fn: object,
    *,
    batch_size: int,
    huber_delta: float,
    rank_loss_weight: float,
    rank_loss_temperature: float,
) -> dict[str, float]:
    """Evaluate the governed validation objective with bounded prediction memory."""
    if rank_loss_weight <= 0:
        # Exact legacy path when the listwise objective is disabled.
        huber = _impl._batched_loss(
            model,
            features,
            targets,
            loss_fn,
            batch_size=batch_size,
        )
        return {"huber": huber, "rank": 0.0, "composite": huber, "rank_groups": 0.0}
    if trade_dates is None:
        raise ValueError("rank-loss validation requires validation trade dates")
    predictions = _impl._batched_predict_tensor(
        model,
        features,
        batch_size=batch_size,
    ).numpy()
    target_values = targets.detach().cpu().numpy()
    return validation_objective_from_predictions(
        predictions,
        target_values,
        trade_dates,
        huber_delta=huber_delta,
        rank_loss_weight=rank_loss_weight,
        rank_loss_temperature=rank_loss_temperature,
    )


class FTTransformerTrainer(_impl.FTTransformerTrainer):
    """FT-Transformer trainer with objective-aligned validation/checkpointing."""

    def _fit_torch(self, dataset: pd.DataFrame, validation_dataset: pd.DataFrame | None) -> FTTransformerArtifacts:
        import torch
        from torch import nn

        from quantagent.models.ft_transformer import FTTransformer, FTTransformerConfig

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        feature_columns = list(self.config.feature_columns) or _impl._auto_feature_columns(dataset)
        horizons = [h for h in self.config.horizons if f"forward_return_{h}d" in dataset.columns]
        if not horizons:
            raise ValueError("dataset has no forward_return_*d columns matching configured horizons")

        if validation_dataset is None or validation_dataset.empty:
            dataset = dataset.copy()
            dataset["trade_date"] = pd.to_datetime(dataset["trade_date"], errors="coerce")
            dataset = dataset.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
            unique = dataset["trade_date"].dropna().unique()
            if len(unique) >= 5:
                cutoff = unique[int(0.8 * len(unique))]
                train_frame = dataset[dataset["trade_date"] < cutoff]
                validation_dataset = dataset[dataset["trade_date"] >= cutoff]
                dataset = train_frame

        train_x, train_y, train_dates = _impl._prepare(dataset, feature_columns, horizons)
        val_x, val_y, val_dates = (
            _impl._prepare(validation_dataset, feature_columns, horizons)
            if validation_dataset is not None and not validation_dataset.empty
            else (None, None, None)
        )
        if val_x is not None and self.config.rank_loss_weight > 0:
            if val_dates is None or len(val_dates) != len(val_x):
                raise ValueError("rank-loss validation requires aligned validation trade dates")
            parsed_val_dates = pd.to_datetime(val_dates, errors="coerce")
            if parsed_val_dates.isna().any():
                raise ValueError("rank-loss validation requires valid trade_date for every row")
            if int(parsed_val_dates.value_counts().max()) < 2:
                raise ValueError("rank-loss validation requires at least one trade date with >=2 names")

        device = _impl._resolve_device(self.config.device, require_gpu=self.config.require_gpu)
        device_report = _impl._torch_device_report(device)
        means = np.nan_to_num(train_x.mean(axis=0))
        scales = np.nan_to_num(train_x.std(axis=0)) + 1e-9
        train_tensor = torch.tensor((train_x - means) / scales, dtype=torch.float32, device=device)
        target_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)
        if val_x is not None:
            val_tensor = torch.tensor((val_x - means) / scales, dtype=torch.float32, device=device)
            val_target = torch.tensor(val_y, dtype=torch.float32, device=device)
        else:
            val_tensor = None
            val_target = None

        config = FTTransformerConfig(
            num_features=train_tensor.shape[1],
            num_horizons=len(horizons),
            d_token=self.config.d_token,
            n_blocks=self.config.n_blocks,
            n_heads=self.config.n_heads,
            attention_dropout=self.config.attention_dropout,
            ffn_dropout=self.config.ffn_dropout,
            use_missing_mask=self.config.use_missing_mask,
        )
        model = FTTransformer(config).to(device)
        if self.config.resume_checkpoint:
            state = torch.load(self.config.resume_checkpoint, map_location=device)
            model.load_state_dict(state["model"])
        if self.config.activation_checkpointing and hasattr(model, "blocks"):
            from torch.utils.checkpoint import checkpoint as _ckpt

            class _CheckpointedBlock(nn.Module):
                def __init__(self, block: nn.Module) -> None:
                    super().__init__()
                    self.block = block

                def forward(self, x):
                    if self.training and torch.is_grad_enabled():
                        return _ckpt(self.block, x, use_reentrant=False)
                    return self.block(x)

            model.blocks = nn.ModuleList([_CheckpointedBlock(b) for b in model.blocks])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        huber = nn.HuberLoss(delta=self.config.huber_delta)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.config.use_amp and device == "cuda"))
        date_codes = (
            torch.tensor(pd.Categorical(train_dates).codes, dtype=torch.long, device=device)
            if train_dates is not None
            else None
        )

        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        patience = 0
        history: list[dict[str, float]] = []
        dates_per_step = max(1, int(self.config.dates_per_step))
        micro_batch = self.config.train_micro_batch

        for epoch in range(self.config.max_epochs):
            model.train()
            if date_codes is not None:
                unique_dates = torch.unique(date_codes)
                date_order = unique_dates[torch.randperm(unique_dates.shape[0])]
            else:
                date_order = torch.tensor([0], device=device)
            epoch_loss = 0.0
            total_rows = 0
            finite_steps = 0
            nonfinite_steps = 0
            chunks = torch.split(date_order, dates_per_step)

            for chunk in chunks:
                if date_codes is not None:
                    mask = torch.isin(date_codes, chunk)
                else:
                    mask = torch.ones(train_tensor.shape[0], dtype=torch.bool, device=device)
                if int(mask.sum()) < 2:
                    continue
                xb = train_tensor[mask]
                yb = target_tensor[mask]
                chunk_codes = date_codes[mask] if date_codes is not None else None
                optimizer.zero_grad()
                step_loss_value = 0.0
                step_rows = 0

                if micro_batch and int(xb.shape[0]) > int(micro_batch) and chunk_codes is not None:
                    for d in chunk:
                        sub_mask = chunk_codes == d
                        if int(sub_mask.sum()) < 2:
                            continue
                        xs = xb[sub_mask]
                        ys = yb[sub_mask]
                        with torch.cuda.amp.autocast(enabled=(self.config.use_amp and device == "cuda")):
                            preds_s = model(xs)
                            loss_s = huber(preds_s, ys)
                            if self.config.rank_loss_weight > 0 and xs.shape[0] >= 2:
                                rank_loss_s = _impl._softmax_listwise_loss(
                                    preds_s,
                                    ys,
                                    temperature=self.config.rank_loss_temperature,
                                )
                                loss_s = loss_s + self.config.rank_loss_weight * rank_loss_s
                            loss_s = loss_s / max(1, len(chunk))
                        if not torch.isfinite(loss_s.detach()):
                            nonfinite_steps += 1
                            continue
                        scaler.scale(loss_s).backward()
                        step_loss_value += float(loss_s.detach().cpu()) * int(xs.shape[0])
                        step_rows += int(xs.shape[0])
                    if step_rows == 0:
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    scaler.unscale_(optimizer)
                    if self.config.gradient_clip_norm and float(self.config.gradient_clip_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config.gradient_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                    finite_steps += 1
                else:
                    with torch.cuda.amp.autocast(enabled=(self.config.use_amp and device == "cuda")):
                        preds = model(xb)
                        loss = huber(preds, yb)
                        if self.config.rank_loss_weight > 0 and xb.shape[0] >= 2:
                            if chunk_codes is not None:
                                rank_loss_acc = torch.zeros((), device=preds.device, dtype=preds.dtype)
                                n_groups = 0
                                for d in chunk:
                                    m = chunk_codes == d
                                    if int(m.sum()) >= 2:
                                        rank_loss_acc = rank_loss_acc + _impl._softmax_listwise_loss(
                                            preds[m],
                                            yb[m],
                                            temperature=self.config.rank_loss_temperature,
                                        )
                                        n_groups += 1
                                if n_groups > 0:
                                    rank_loss = rank_loss_acc / float(n_groups)
                                    loss = loss + self.config.rank_loss_weight * rank_loss
                            else:
                                rank_loss = _impl._softmax_listwise_loss(
                                    preds,
                                    yb,
                                    temperature=self.config.rank_loss_temperature,
                                )
                                loss = loss + self.config.rank_loss_weight * rank_loss
                    if not torch.isfinite(loss.detach()):
                        optimizer.zero_grad(set_to_none=True)
                        nonfinite_steps += 1
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if self.config.gradient_clip_norm and float(self.config.gradient_clip_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config.gradient_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                    step_loss_value = float(loss.detach().cpu()) * int(xb.shape[0])
                    step_rows = int(xb.shape[0])
                    finite_steps += 1

                epoch_loss += step_loss_value
                total_rows += step_rows

            epoch_loss /= max(1, total_rows)
            entry: dict[str, float] = {
                "epoch": float(epoch),
                "loss": float(epoch_loss),
                "finite_steps": float(finite_steps),
                "nonfinite_steps": float(nonfinite_steps),
            }
            if finite_steps == 0:
                entry["diverged"] = 1.0
                history.append(entry)
                break

            if val_tensor is not None and val_target is not None:
                model.eval()
                with torch.no_grad():
                    val_parts = _validation_objective(
                        model,
                        val_tensor,
                        val_target,
                        val_dates,
                        huber,
                        batch_size=self.config.batch_size,
                        huber_delta=self.config.huber_delta,
                        rank_loss_weight=self.config.rank_loss_weight,
                        rank_loss_temperature=self.config.rank_loss_temperature,
                    )
                val_loss = float(val_parts["composite"])
                if not np.isfinite(val_loss):
                    raise RuntimeError(
                        f"FT-Transformer non-finite validation objective at epoch={epoch}; "
                        "training stopped before saving a polluted checkpoint"
                    )
                entry["val_loss"] = val_loss
                entry["val_huber_loss"] = float(val_parts["huber"])
                entry["val_rank_loss"] = float(val_parts["rank"])
                entry["val_composite_loss"] = val_loss
                entry["val_rank_groups"] = float(val_parts["rank_groups"])
                improved = val_loss < best_val - 1e-6
                if improved:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if improved:
                    flag = "improved"
                elif epoch_loss < val_loss:
                    flag = "OVERFIT"
                else:
                    flag = "plateau"
                print(
                    f"[ft] epoch {epoch:>3d} train={epoch_loss:.6f} "
                    f"val_composite={val_loss:.6f} val_huber={val_parts['huber']:.6f} "
                    f"val_rank={val_parts['rank']:.6f} best={best_val:.6f} "
                    f"patience={patience}/{self.config.early_stopping_patience} "
                    f"nonfinite_steps={nonfinite_steps} [{flag}]",
                    flush=True,
                )
                if (not improved) and patience >= self.config.early_stopping_patience:
                    print(
                        f"[ft] EARLY STOP at epoch {epoch}: composite validation objective "
                        f"plateaued {patience} epochs; restoring best checkpoint "
                        f"(val_composite={best_val:.6f})",
                        flush=True,
                    )
                    history.append(entry)
                    break
            history.append(entry)

        if best_state is not None:
            model.load_state_dict(best_state)
        if self.config.activation_checkpointing and hasattr(model, "blocks"):
            model.blocks = nn.ModuleList([m.block if hasattr(m, "block") else m for m in model.blocks])

        if torch.cuda.is_available() and self.config.log_gpu_memory and str(device).startswith("cuda"):
            try:
                device_report["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                device_report["peak_gpu_memory_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
            except Exception:
                pass

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "ft_transformer.pt"
        config_path = output_dir / "ft_transformer_config.json"
        schema_path = output_dir / "ft_transformer_feature_schema.json"
        metrics_path = output_dir / "ft_transformer_metrics.json"
        objective = {
            "semantics_version": OBJECTIVE_SEMANTICS_VERSION,
            "pointwise_loss": "huber",
            "huber_delta": float(self.config.huber_delta),
            "rank_loss": "per_trade_date_softmax_listwise_portfolio_return",
            "rank_loss_weight": float(self.config.rank_loss_weight),
            "rank_loss_temperature": float(self.config.rank_loss_temperature),
            "checkpoint_selection": "minimum_validation_composite_objective",
        }
        torch.save(
            {
                "model": model.state_dict(),
                "feature_columns": feature_columns,
                "horizons": horizons,
                "feature_means": means.tolist(),
                "feature_scales": scales.tolist(),
                "config": config.__dict__,
                "objective_semantics": objective,
            },
            checkpoint_path,
        )
        config_path.write_text(
            json.dumps(asdict(self.config), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        schema_path.write_text(
            json.dumps(
                {
                    "feature_columns": feature_columns,
                    "horizons": horizons,
                    "backend": "torch",
                    "architecture": "ft_transformer",
                    "version": "v7",
                    "objective_semantics_version": OBJECTIVE_SEMANTICS_VERSION,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        metrics_path.write_text(
            json.dumps(
                {
                    "training_history": history,
                    "backend": "torch",
                    "device": device,
                    "cuda_available": device_report["cuda_available"],
                    "gpu_name": device_report["gpu_name"],
                    "horizons": horizons,
                    "objective_semantics": objective,
                    "peak_gpu_memory_mb": device_report.get("peak_gpu_memory_mb"),
                    "peak_gpu_memory_reserved_mb": device_report.get("peak_gpu_memory_reserved_mb"),
                    "dates_per_step": int(self.config.dates_per_step),
                    "d_token": int(self.config.d_token),
                    "n_blocks": int(self.config.n_blocks),
                    "n_heads": int(self.config.n_heads),
                    "batch_size": int(self.config.batch_size),
                    "max_epochs": int(self.config.max_epochs),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return FTTransformerArtifacts(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            schema_path=schema_path,
            metrics_path=metrics_path,
            backend="torch",
            device=device,
            cuda_available=bool(device_report["cuda_available"]),
            gpu_name=device_report["gpu_name"],
            horizons=horizons,
            feature_columns=feature_columns,
            training_history=history,
        )


__all__ = [
    "OBJECTIVE_SEMANTICS_VERSION",
    "FTTransformerTrainer",
    "FTTransformerTrainerConfig",
    "FTTransformerArtifacts",
    "FTTransformerPredictionResult",
    "predict_ft_transformer_artifact",
    "validation_objective_from_predictions",
]
