"""Standalone FT-Transformer trainer for multi-horizon alpha tasks.

The default :class:`V7DeepAlphaTrainer` is intentionally tied to a small
multi-horizon MLP so its checkpoint format stays trivially serialisable
(numpy arrays inside JSON). For research workflows that genuinely want a
larger tabular architecture, :class:`FTTransformerTrainer` here uses the
:class:`quantagent.models.ft_transformer.FTTransformer` model with a
proper PyTorch state-dict checkpoint, mixed-precision when CUDA is
available, and time-aware validation splits.

Live trading is never enabled by this module — it only produces a
``predictions.parquet`` frame plus checkpoint, and downstream
``build-target-weights-v7`` / backtest steps still apply all
A-share trading constraints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.cuda_runtime import (
    configure_cuda_environment,
    cuda_runtime_probe,
    format_cuda_diagnostic,
)

configure_cuda_environment()


def _default_ft_output_dir() -> str:
    return str(quant_paths().models / "v7_alpha" / "ft_transformer")


@dataclass(frozen=True)
class FTTransformerTrainerConfig:
    horizons: tuple[int, ...] = (1, 5, 20, 60, 120, 126)
    d_token: int = 128
    n_blocks: int = 5
    n_heads: int = 8
    attention_dropout: float = 0.10
    ffn_dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8192
    max_epochs: int = 60
    early_stopping_patience: int = 10
    huber_delta: float = 1.0
    rank_loss_weight: float = 0.5
    rank_loss_temperature: float = 0.5  # softmax temperature for top-K listwise loss
    dates_per_step: int = 8
    train_micro_batch: int | None = None
    gradient_clip_norm: float = 1.0
    log_gpu_memory: bool = True
    device: str = "auto"
    require_gpu: bool = False
    seed: int = 1729
    feature_columns: tuple[str, ...] = ()
    use_missing_mask: bool = True
    use_amp: bool = True
    output_dir: str = field(default_factory=_default_ft_output_dir)
    resume_checkpoint: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class FTTransformerArtifacts:
    checkpoint_path: Path
    config_path: Path
    schema_path: Path
    metrics_path: Path
    backend: str
    device: str
    cuda_available: bool
    gpu_name: str | None
    horizons: list[int]
    feature_columns: list[str]
    training_history: list[dict[str, float]]


@dataclass(frozen=True)
class FTTransformerPredictionResult:
    predictions: pd.DataFrame
    horizons: tuple[int, ...]
    feature_columns: tuple[str, ...]
    artifact_dir: str


class FTTransformerTrainer:
    """Date-aware FT-Transformer trainer with checkpoint resume."""

    def __init__(self, config: FTTransformerTrainerConfig | None = None) -> None:
        self.config = config or FTTransformerTrainerConfig()

    def fit_and_save(
        self,
        dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame | None = None,
    ) -> FTTransformerArtifacts:
        """Fit the model and persist all artefacts under ``output_dir``."""
        if dataset is None or dataset.empty:
            raise ValueError("FT-Transformer trainer requires a non-empty dataset")
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "FTTransformerTrainer requires PyTorch — install quantagent[training]"
            ) from exc
        return self._fit_torch(dataset, validation_dataset)

    # ------------------------------------------------------------------
    def _fit_torch(  # pragma: no cover - torch path exercised manually
        self,
        dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame | None,
    ) -> FTTransformerArtifacts:
        import torch
        from torch import nn

        from quantagent.models.ft_transformer import FTTransformer, FTTransformerConfig

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        feature_columns = list(self.config.feature_columns) or _auto_feature_columns(dataset)
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

        train_x, train_y, train_dates = _prepare(dataset, feature_columns, horizons)
        val_x, val_y, _ = (
            _prepare(validation_dataset, feature_columns, horizons)
            if validation_dataset is not None and not validation_dataset.empty
            else (None, None, None)
        )

        device = _resolve_device(self.config.device, require_gpu=self.config.require_gpu)
        device_report = _torch_device_report(device)
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
                # Optional micro-batch split to control activation memory for
                # very wide cross-sections without losing per-date rank loss.
                if micro_batch and int(xb.shape[0]) > int(micro_batch) and chunk_codes is not None:
                    sub_dates = chunk
                else:
                    sub_dates = [None]
                optimizer.zero_grad()
                step_loss_value = 0.0
                step_rows = 0
                if micro_batch and int(xb.shape[0]) > int(micro_batch) and chunk_codes is not None:
                    # split chunk by date groups, each forward/backward over one sub-batch
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
                                rank_loss_s = _softmax_listwise_loss(
                                    preds_s, ys, temperature=self.config.rank_loss_temperature
                                )
                                loss_s = loss_s + self.config.rank_loss_weight * rank_loss_s
                            loss_s = loss_s / max(1, len(chunk))
                        if not torch.isfinite(loss_s.detach()):
                            # Graceful skip of this sub-batch (see normal path).
                            nonfinite_steps += 1
                            continue
                        scaler.scale(loss_s).backward()
                        step_loss_value += float(loss_s.detach().cpu()) * int(xs.shape[0])
                        step_rows += int(xs.shape[0])
                    if step_rows == 0:
                        # whole chunk was non-finite — drop grads, skip the step
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
                                        rank_loss_acc = rank_loss_acc + _softmax_listwise_loss(
                                            preds[m], yb[m], temperature=self.config.rank_loss_temperature
                                        )
                                        n_groups += 1
                                if n_groups > 0:
                                    rank_loss = rank_loss_acc / float(n_groups)
                                    loss = loss + self.config.rank_loss_weight * rank_loss
                            else:
                                rank_loss = _softmax_listwise_loss(
                                    preds, yb, temperature=self.config.rank_loss_temperature
                                )
                                loss = loss + self.config.rank_loss_weight * rank_loss
                    if not torch.isfinite(loss.detach()):
                        # Graceful skip: an AMP fp16 overflow on one step must
                        # not throw away an otherwise-healthy multi-hour run.
                        # Drop this step's grads and move on; the per-epoch
                        # guard below stops training only if an ENTIRE epoch
                        # produced no finite step.
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
            entry = {"epoch": epoch, "loss": epoch_loss,
                     "finite_steps": int(finite_steps),
                     "nonfinite_steps": int(nonfinite_steps)}
            # Graceful divergence stop: if an entire epoch produced no finite
            # step the model has diverged (typically AMP fp16 overflow). Stop
            # and keep the best checkpoint seen so far rather than crashing.
            if finite_steps == 0:
                entry["diverged"] = True
                history.append(entry)
                break
            if val_tensor is not None:
                model.eval()
                with torch.no_grad():
                    val_loss = _batched_loss(
                        model,
                        val_tensor,
                        val_target,
                        huber,
                        batch_size=self.config.batch_size,
                    )
                if not np.isfinite(val_loss):
                    raise RuntimeError(
                        f"FT-Transformer non-finite validation loss at epoch={epoch}; "
                        "training stopped before saving a polluted checkpoint"
                    )
                entry["val_loss"] = val_loss
                improved = val_loss < best_val - 1e-6
                if improved:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                # Live per-epoch overfit monitor (visible in the tmux log).
                # OVERFIT = val rising above best while train keeps falling.
                if improved:
                    flag = "improved"
                elif epoch_loss < val_loss:
                    flag = "OVERFIT"
                else:
                    flag = "plateau"
                print(
                    f"[ft] epoch {epoch:>3d}  train={epoch_loss:.6f}  val={val_loss:.6f}  "
                    f"best={best_val:.6f}  patience={patience}/{self.config.early_stopping_patience}  "
                    f"nonfinite_steps={nonfinite_steps}  [{flag}]",
                    flush=True,
                )
                if (not improved) and patience >= self.config.early_stopping_patience:
                    print(
                        f"[ft] EARLY STOP at epoch {epoch}: val plateaued {patience} epochs; "
                        f"restoring best checkpoint (val={best_val:.6f})",
                        flush=True,
                    )
                    history.append(entry)
                    break
            history.append(entry)

        if best_state is not None:
            model.load_state_dict(best_state)

        if torch.cuda.is_available() and self.config.log_gpu_memory and str(device).startswith("cuda"):
            try:
                peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                reserved_mb = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
                device_report["peak_gpu_memory_mb"] = peak_mb
                device_report["peak_gpu_memory_reserved_mb"] = reserved_mb
            except Exception:
                pass

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "ft_transformer.pt"
        config_path = output_dir / "ft_transformer_config.json"
        schema_path = output_dir / "ft_transformer_feature_schema.json"
        metrics_path = output_dir / "ft_transformer_metrics.json"
        torch.save(
            {
                "model": model.state_dict(),
                "feature_columns": feature_columns,
                "horizons": horizons,
                "feature_means": means.tolist(),
                "feature_scales": scales.tolist(),
                "config": config.__dict__,
            },
            checkpoint_path,
        )
        config_path.write_text(json.dumps(asdict(self.config), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        schema_path.write_text(
            json.dumps(
                {
                    "feature_columns": feature_columns,
                    "horizons": horizons,
                    "backend": "torch",
                    "architecture": "ft_transformer",
                    "version": "v7",
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


def _auto_feature_columns(dataset: pd.DataFrame) -> list[str]:
    label_columns = {c for c in dataset.columns if c.startswith("forward_return_") or c.startswith("label_end_")}
    forbidden = label_columns | {"symbol", "trade_date", "available_at"}
    return [
        column
        for column in dataset.select_dtypes(include=[np.number, bool]).columns
        if column not in forbidden
    ]


def _prepare(
    frame: pd.DataFrame,
    feature_columns: list[str],
    horizons: list[int],
) -> tuple[np.ndarray, np.ndarray, pd.Series | None]:
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    targets = np.column_stack(
        [pd.to_numeric(frame[f"forward_return_{h}d"], errors="coerce").to_numpy() for h in horizons]
    )
    keep = ~np.isnan(targets).any(axis=1)
    features = frame.loc[keep, feature_columns].to_numpy(dtype=float)
    labels = targets[keep]
    dates = frame.loc[keep, "trade_date"]
    return features, labels, dates


def _resolve_device(device: str, *, require_gpu: bool = False) -> str:  # pragma: no cover - torch path
    import torch  # type: ignore

    if device == "cpu":
        if require_gpu:
            raise RuntimeError("GPU training was required, but device='cpu' was requested")
        return "cpu"
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if require_gpu:
            raise RuntimeError(
                "GPU training was required, but torch.cuda.is_available() is false. "
                + format_cuda_diagnostic(cuda_runtime_probe(torch))
            )
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({device}), but torch.cuda.is_available() is false. "
            + format_cuda_diagnostic(cuda_runtime_probe(torch))
        )
    return device


def _torch_device_report(device: str) -> dict[str, object]:  # pragma: no cover - torch path
    import torch  # type: ignore

    probe = cuda_runtime_probe(torch)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else probe.get("nvidia_smi_gpu_name")
    return {
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "cuda_diagnostic": probe,
    }


def predict_ft_transformer_artifact(
    artifact_dir: str | Path,
    feature_frame: pd.DataFrame,
    *,
    primary_horizon: int | None = None,
    device: str = "cpu",
) -> FTTransformerPredictionResult:
    """Load ``ft_transformer.pt`` and run a deterministic forward pass."""
    if feature_frame is None or feature_frame.empty:
        raise ValueError("FT-Transformer prediction requires a non-empty feature frame")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("FT-Transformer prediction requires PyTorch; install quantagent[training]") from exc
    from quantagent.models.ft_transformer import FTTransformer, FTTransformerConfig

    artifact = Path(artifact_dir)
    if artifact.is_file():
        artifact = artifact.parent
    checkpoint_path = artifact / "ft_transformer.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FT-Transformer checkpoint not found: {checkpoint_path}")
    resolved_device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    feature_columns = tuple(str(c) for c in checkpoint["feature_columns"])
    horizons = tuple(int(h) for h in checkpoint["horizons"])
    missing = [column for column in feature_columns if column not in feature_frame.columns]
    if missing:
        raise ValueError(f"FT-Transformer feature frame missing columns {missing}")
    model_config = FTTransformerConfig(**checkpoint["config"])
    model = FTTransformer(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    means = np.asarray(checkpoint["feature_means"], dtype=float)
    scales = np.asarray(checkpoint["feature_scales"], dtype=float)
    values = feature_frame[list(feature_columns)].to_numpy(dtype=float)
    tensor = torch.tensor((values - means) / scales, dtype=torch.float32, device=resolved_device)
    with torch.no_grad():
        outputs = _batched_predict_tensor(model, tensor, batch_size=4096).detach().cpu().numpy()
    base_columns = [c for c in ("symbol", "trade_date") if c in feature_frame.columns]
    output = feature_frame[base_columns].copy()
    for index, horizon in enumerate(horizons):
        output[f"alpha_{horizon}d"] = outputs[:, index]
    primary = primary_horizon if primary_horizon in horizons else horizons[0]
    output["prediction"] = output[f"alpha_{primary}d"]
    return FTTransformerPredictionResult(
        predictions=output.reset_index(drop=True),
        horizons=horizons,
        feature_columns=feature_columns,
        artifact_dir=str(artifact),
    )


def _softmax_listwise_loss(
    preds: "torch.Tensor",
    targets: "torch.Tensor",
    *,
    temperature: float = 0.5,
) -> "torch.Tensor":
    """Differentiable per-date listwise loss aligned with long-only top-K deployment.

    For each horizon column independently, treat preds[:, h] as scores, build a
    softmax-weighted portfolio over the N tickers in the batch, and return the
    NEGATIVE realized return (so minimising the loss maximises the portfolio).

    Unlike ``argsort``-based rank loss (gradient = 0), this is fully
    differentiable and directly trains the model to put mass on top-return
    names — the same objective the executable backtest uses.
    """
    import torch

    if preds.dim() == 1:
        preds = preds.unsqueeze(-1)
        targets = targets.unsqueeze(-1)
    n, h = preds.shape
    if n < 2:
        return preds.sum() * 0.0
    temp = max(float(temperature), 1e-6)
    # softmax over the batch (N tickers) — each horizon gets its own portfolio
    weights = torch.softmax(preds / temp, dim=0)  # [N, H]
    portfolio_return = (weights * targets).sum(dim=0)  # [H]
    return -portfolio_return.mean()


def _batched_loss(
    model: object,
    features: "torch.Tensor",
    targets: "torch.Tensor",
    loss_fn: object,
    *,
    batch_size: int,
) -> float:  # pragma: no cover - torch path
    import torch

    total_loss = 0.0
    total_rows = 0
    size = max(1, int(batch_size))
    for start in range(0, int(features.shape[0]), size):
        end = min(start + size, int(features.shape[0]))
        preds = model(features[start:end])
        loss = loss_fn(preds, targets[start:end])
        rows = end - start
        total_loss += float(loss.detach().cpu()) * rows
        total_rows += rows
    return total_loss / max(1, total_rows)


def _batched_predict_tensor(
    model: object,
    features: "torch.Tensor",
    *,
    batch_size: int,
) -> "torch.Tensor":  # pragma: no cover - torch path
    import torch

    outputs = []
    size = max(1, int(batch_size))
    for start in range(0, int(features.shape[0]), size):
        end = min(start + size, int(features.shape[0]))
        outputs.append(model(features[start:end]).detach().cpu())
    return torch.cat(outputs, dim=0)


__all__ = [
    "FTTransformerTrainer",
    "FTTransformerTrainerConfig",
    "FTTransformerArtifacts",
    "FTTransformerPredictionResult",
    "predict_ft_transformer_artifact",
]
