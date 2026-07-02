"""V7 deep alpha trainer with optional PyTorch backend.

The trainer takes a wide ``(symbol, trade_date, *features, *labels)``
training frame and fits a small multi-horizon MLP. When PyTorch is
available it runs a real Adam loop with checkpointing and early
stopping; otherwise it falls back to a deterministic numpy ridge head
so the pipeline works on CPU-only research boxes too.

The objective combines:

* Huber return loss per horizon (robust to outliers).
* Cross-sectional rank loss per ``trade_date`` (rank-IC friendly).
* Optional long-short portfolio utility (``--lambda-utility``).

Checkpoints, configs and feature schemas are written under the unified
``quant_paths().models / "v7_alpha"`` tree by default. ``save`` /
``load`` round-trip the full model state.

Live trading is never enabled by this module; it only emits
predictions and ``target_weights_proxy`` columns for the downstream
backtester.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.config.paths import quant_paths


def _default_deep_output_dir() -> str:
    return str(quant_paths().models / "v7_alpha" / "deep")


@dataclass(frozen=True)
class V7DeepAlphaTrainerConfig:
    horizons: tuple[int, ...] = (1, 5, 20, 60, 120, 126)
    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 30
    early_stopping_patience: int = 5
    huber_delta: float = 1.0
    rank_loss_weight: float = 0.5
    utility_loss_weight: float = 0.0
    long_short_topk: int = 20
    device: str = "auto"
    seed: int = 1729
    feature_columns: tuple[str, ...] = ()
    # Pin features to a dataset feature_schema.json contract. When set it takes
    # precedence over auto-derivation: the schema's feature_columns (exact
    # order) are used and feature_version/schema_hash are recorded on the state.
    feature_schema_path: str | None = None
    output_dir: str = field(default_factory=_default_deep_output_dir)
    use_torch: bool = True
    # GPU discipline (matches ft_transformer_trainer): require_gpu fails loud
    # instead of silently training on CPU; log_gpu_memory records the per-fit
    # CUDA peak so sequential walk-forward folds are memory-auditable.
    require_gpu: bool = False
    log_gpu_memory: bool = True
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class V7DeepAlphaState:
    backend: str
    feature_columns: list[str]
    horizons: list[int]
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    output_weights: np.ndarray
    output_biases: np.ndarray
    feature_means: np.ndarray
    feature_scales: np.ndarray
    training_history: list[dict[str, float]]
    # Feature-schema contract provenance (empty when features were auto-derived).
    feature_version: str = ""
    schema_hash: str = ""
    schema_path: str = ""
    gpu_peak_mb: float = 0.0   # CUDA peak allocated during fit (0 on CPU/numpy)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "feature_columns": list(self.feature_columns),
            "horizons": list(self.horizons),
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "output_weights": self.output_weights.tolist(),
            "output_biases": self.output_biases.tolist(),
            "feature_means": self.feature_means.tolist(),
            "feature_scales": self.feature_scales.tolist(),
            "training_history": list(self.training_history),
            "feature_version": self.feature_version,
            "schema_hash": self.schema_hash,
            "schema_path": self.schema_path,
            "gpu_peak_mb": self.gpu_peak_mb,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "V7DeepAlphaState":
        return cls(
            backend=str(payload["backend"]),
            feature_columns=list(payload["feature_columns"]),
            horizons=list(payload["horizons"]),
            weights=[np.asarray(w, dtype=float) for w in payload["weights"]],
            biases=[np.asarray(b, dtype=float) for b in payload["biases"]],
            output_weights=np.asarray(payload["output_weights"], dtype=float),
            output_biases=np.asarray(payload["output_biases"], dtype=float),
            feature_means=np.asarray(payload["feature_means"], dtype=float),
            feature_scales=np.asarray(payload["feature_scales"], dtype=float),
            training_history=list(payload["training_history"]),
            feature_version=str(payload.get("feature_version", "")),
            schema_hash=str(payload.get("schema_hash", "")),
            schema_path=str(payload.get("schema_path", "")),
            gpu_peak_mb=float(payload.get("gpu_peak_mb", 0.0)),
        )


class V7DeepAlphaTrainer:
    """Multi-horizon deep alpha trainer with Torch / numpy backends."""

    def __init__(self, config: V7DeepAlphaTrainerConfig | None = None) -> None:
        self.config = config or V7DeepAlphaTrainerConfig()
        self.state: V7DeepAlphaState | None = None

    def fit(self, dataset: pd.DataFrame, validation_dataset: pd.DataFrame | None = None) -> V7DeepAlphaState:
        if dataset is None or dataset.empty:
            raise ValueError("deep alpha trainer requires a non-empty training dataset")
        feature_columns, schema_version, schema_hash, schema_path = self._resolve_feature_columns(dataset)
        if not feature_columns:
            raise ValueError("deep alpha trainer found no feature columns")
        horizons = [h for h in self.config.horizons if f"forward_return_{h}d" in dataset.columns]
        if not horizons:
            raise ValueError("deep alpha trainer needs at least one forward_return_*d label")
        # Auto-split a validation set out of the training dataset if none was supplied
        # so the early-stopping path always has something to monitor.
        if validation_dataset is None or validation_dataset.empty:
            sorted_dataset = dataset.copy()
            sorted_dataset["trade_date"] = pd.to_datetime(sorted_dataset["trade_date"], errors="coerce")
            sorted_dataset = sorted_dataset.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
            unique_dates = sorted_dataset["trade_date"].dropna().unique()
            if len(unique_dates) >= 5:
                cutoff = unique_dates[int(0.8 * len(unique_dates))]
                train_frame = sorted_dataset[sorted_dataset["trade_date"] < cutoff]
                validation_dataset = sorted_dataset[sorted_dataset["trade_date"] >= cutoff]
                dataset = train_frame
        train_x, train_y, train_dates = self._prepare(dataset, feature_columns, horizons)
        val_x, val_y, val_dates = (
            self._prepare(validation_dataset, feature_columns, horizons) if validation_dataset is not None and not validation_dataset.empty else (None, None, None)
        )

        backend = self._select_backend()
        if backend == "torch":
            state = self._fit_torch(train_x, train_y, val_x, val_y, train_dates, val_dates, feature_columns, horizons)
        else:
            state = self._fit_numpy(train_x, train_y, val_x, val_y, feature_columns, horizons)
        # Stamp the feature-schema contract provenance onto the fitted state so it
        # rides through save()/load() into the checkpoint + manifest.
        state.feature_version = schema_version
        state.schema_hash = schema_hash
        state.schema_path = schema_path
        self.state = state
        return state

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.state is None:
            raise RuntimeError("trainer has no fitted state; call fit() or load()")
        if frame is None or frame.empty:
            return pd.DataFrame()
        feature_columns = self.state.feature_columns
        missing = [c for c in feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"predict frame missing feature columns {missing}")
        x = self._standardise(frame[feature_columns].to_numpy(dtype=float))
        outputs = self._forward(x)
        result = frame[["symbol", "trade_date"]].copy() if {"symbol", "trade_date"}.issubset(frame.columns) else pd.DataFrame()
        for idx, horizon in enumerate(self.state.horizons):
            result[f"alpha_{horizon}d"] = outputs[:, idx]
        return result.reset_index(drop=True)

    def save(self, output_dir: str | Path | None = None) -> Path:
        if self.state is None:
            raise RuntimeError("trainer has no fitted state; cannot save")
        path = Path(output_dir or self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        state_path = path / "deep_alpha_state.json"
        config_path = path / "deep_alpha_config.json"
        schema_path = path / "deep_alpha_feature_schema.json"
        metrics_path = path / "deep_alpha_metrics.json"
        state_path.write_text(json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        config_path.write_text(json.dumps(asdict(self.config), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        schema_path.write_text(
            json.dumps(
                {
                    "feature_columns": list(self.state.feature_columns),
                    "horizons": list(self.state.horizons),
                    "backend": self.state.backend,
                    "feature_count": len(self.state.feature_columns),
                    "version": "v7",
                    # Provenance of the dataset feature-schema contract used.
                    "feature_version": self.state.feature_version,
                    "schema_hash": self.state.schema_hash,
                    "source_schema_path": self.state.schema_path,
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
                    "training_history": list(self.state.training_history),
                    "backend": self.state.backend,
                    "horizons": list(self.state.horizons),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        # Write experiment manifest pointing at every artifact.
        manifest_path = path / "deep_alpha_experiment_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "backend": self.state.backend,
                    "horizons": list(self.state.horizons),
                    "feature_columns_count": len(self.state.feature_columns),
                    "feature_version": self.state.feature_version,
                    "schema_hash": self.state.schema_hash,
                    "source_schema_path": self.state.schema_path,
                    "artifact_paths": {
                        "state": str(state_path),
                        "config": str(config_path),
                        "feature_schema": str(schema_path),
                        "metrics": str(metrics_path),
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return state_path

    def load(self, path: str | Path) -> V7DeepAlphaState:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.state = V7DeepAlphaState.from_dict(payload)
        return self.state

    # ------------------------------------------------------------------ helpers

    def _resolve_feature_columns(
        self, dataset: pd.DataFrame
    ) -> tuple[list[str], str, str, str]:
        """Pick the feature set + return (columns, feature_version, schema_hash, schema_path).

        Precedence:
          1. ``config.feature_schema_path`` — load the pinned dataset contract,
             use its ``feature_columns`` in the schema's exact order, and
             FAIL FAST if any are absent from the dataset (no silent drop).
          2. ``config.feature_columns`` — an explicit caller-supplied list.
          3. ``_auto_feature_columns(dataset)`` — fallback only when no schema
             and no explicit list were supplied.
        """
        if self.config.feature_schema_path:
            schema_path = str(self.config.feature_schema_path)
            feature_columns, version, schema_hash = _load_feature_schema(schema_path)
            missing = [c for c in feature_columns if c not in dataset.columns]
            if missing:
                preview = missing[:20]
                raise ValueError(
                    f"deep alpha trainer: pinned feature schema {schema_path} requires "
                    f"{len(missing)} column(s) absent from the dataset: {preview}"
                    + (" ..." if len(missing) > 20 else "")
                    + " — rebuild the dataset with this schema "
                    "(build-training-dataset-v7 --expected-feature-schema) so the contract matches."
                )
            return list(feature_columns), version, schema_hash, schema_path
        if self.config.feature_columns:
            return list(self.config.feature_columns), "", "", ""
        return self._auto_feature_columns(dataset), "", "", ""

    def _auto_feature_columns(self, dataset: pd.DataFrame) -> list[str]:
        excluded = {"open", "high", "low", "close", "volume", "amount"}
        return [
            column
            for column in dataset.select_dtypes(include=[np.number, bool]).columns
            if not column.startswith("forward_return_")
            and not column.startswith("label_end_")
            and not column.startswith("forward_excess_return_")
            and not column.startswith("forward_rank_")
            and not column.startswith("forward_tradable_return_")
            and column not in excluded
        ]

    def _prepare(self, dataset: pd.DataFrame, feature_columns: list[str], horizons: list[int]) -> tuple[np.ndarray, np.ndarray, pd.Series]:
        missing = [c for c in feature_columns if c not in dataset.columns]
        if missing:
            raise ValueError(f"deep alpha trainer: feature columns missing from dataset: {missing}")
        data = dataset.dropna(subset=[f"forward_return_{h}d" for h in horizons] + feature_columns)
        x = np.nan_to_num(data[feature_columns].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        y = data[[f"forward_return_{h}d" for h in horizons]].to_numpy(dtype=float)
        dates = pd.to_datetime(data["trade_date"], errors="coerce")
        return x, y, dates

    def _select_backend(self) -> str:
        if not self.config.use_torch:
            if self.config.require_gpu:
                raise RuntimeError("require_gpu=True but use_torch=False — enable the torch backend for GPU training")
            return "numpy"
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover - torch optional
            if self.config.require_gpu:
                raise RuntimeError("require_gpu=True but PyTorch is not installed") from exc
            return "numpy"
        return "torch"

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        if self.state is None:
            return x
        means = self.state.feature_means
        scales = np.where(self.state.feature_scales == 0.0, 1.0, self.state.feature_scales)
        return (x - means) / scales

    def _forward(self, x: np.ndarray) -> np.ndarray:
        assert self.state is not None
        activations = self._standardise(x)
        for weight, bias in zip(self.state.weights, self.state.biases):
            activations = np.tanh(activations @ weight + bias)
        return activations @ self.state.output_weights + self.state.output_biases

    # ----------------------------------------------- numpy backend (Ridge head)

    def _fit_numpy(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
        feature_columns: list[str],
        horizons: list[int],
    ) -> V7DeepAlphaState:
        rng = np.random.default_rng(self.config.seed)
        feature_means = train_x.mean(axis=0)
        feature_scales = train_x.std(axis=0) + 1e-9
        standardised = (train_x - feature_means) / feature_scales
        hidden_sizes = self.config.hidden_sizes or (16,)
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        input_dim = standardised.shape[1]
        previous = standardised
        for hidden in hidden_sizes:
            w = rng.standard_normal((input_dim, hidden)) * np.sqrt(1.0 / max(1, input_dim))
            b = np.zeros(hidden)
            previous = np.tanh(previous @ w + b)
            weights.append(w)
            biases.append(b)
            input_dim = hidden
        # ridge-regression closed form for the final layer per horizon
        ridge = 1e-2 * np.eye(input_dim)
        output_weights = np.linalg.pinv(previous.T @ previous + ridge) @ previous.T @ train_y
        output_biases = train_y.mean(axis=0) - previous.mean(axis=0) @ output_weights
        history = [{"epoch": 0, "loss": float(((previous @ output_weights + output_biases - train_y) ** 2).mean())}]
        if val_x is not None and val_y is not None:
            val_std = (val_x - feature_means) / feature_scales
            for w, b in zip(weights, biases):
                val_std = np.tanh(val_std @ w + b)
            val_loss = float(((val_std @ output_weights + output_biases - val_y) ** 2).mean())
            history.append({"epoch": 0, "val_loss": val_loss})
        return V7DeepAlphaState(
            backend="numpy",
            feature_columns=list(feature_columns),
            horizons=list(horizons),
            weights=weights,
            biases=biases,
            output_weights=output_weights,
            output_biases=output_biases,
            feature_means=feature_means,
            feature_scales=feature_scales,
            training_history=history,
        )

    # ------------------------------------------------------- PyTorch backend

    def _fit_torch(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
        train_dates: pd.Series | None,
        val_dates: pd.Series | None,
        feature_columns: list[str],
        horizons: list[int],
    ) -> V7DeepAlphaState:  # pragma: no cover - depends on torch
        import torch
        from torch import nn

        torch.manual_seed(self.config.seed)
        device = self._resolve_device()
        on_cuda = str(device).startswith("cuda")
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        feature_means = train_x.mean(axis=0)
        feature_scales = train_x.std(axis=0) + 1e-9
        train_tensor = torch.tensor((train_x - feature_means) / feature_scales, dtype=torch.float32, device=device)
        target_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)
        if val_x is not None and val_y is not None:
            val_tensor = torch.tensor((val_x - feature_means) / feature_scales, dtype=torch.float32, device=device)
            val_target = torch.tensor(val_y, dtype=torch.float32, device=device)
        else:
            val_tensor = None
            val_target = None

        # Build date-aware batch groups so the rank loss is cross-sectional.
        train_date_codes = (
            torch.tensor(
                pd.Categorical(train_dates).codes if train_dates is not None else np.zeros(train_tensor.shape[0]),
                dtype=torch.long,
                device=device,
            )
            if train_dates is not None
            else None
        )

        layers: list[nn.Module] = []
        input_dim = train_tensor.shape[1]
        for hidden in self.config.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.Tanh())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            input_dim = hidden
        layers.append(nn.Linear(input_dim, len(horizons)))
        model = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        huber = nn.HuberLoss(delta=self.config.huber_delta)

        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        patience = 0
        history: list[dict[str, float]] = []
        for epoch in range(self.config.max_epochs):
            model.train()
            # Sample by date so each batch is a cross-section, not random rows.
            if train_date_codes is not None:
                unique_dates = torch.unique(train_date_codes)
                date_order = unique_dates[torch.randperm(unique_dates.shape[0])]
            else:
                date_order = torch.zeros(1, dtype=torch.long, device=device)
            epoch_loss = 0.0
            total_rows = 0
            batch_dates: list[torch.Tensor] = []
            for date_code in date_order:
                batch_dates.append(date_code.view(1))
                # Group dates into batches that fit batch_size approximately.
                idx = torch.cat(batch_dates)
                if train_date_codes is not None:
                    mask = torch.isin(train_date_codes, idx)
                else:
                    mask = torch.ones(train_tensor.shape[0], dtype=torch.bool, device=device)
                if int(mask.sum()) < self.config.batch_size and date_code is not date_order[-1]:
                    continue
                xb = train_tensor[mask]
                yb = target_tensor[mask]
                date_b = train_date_codes[mask] if train_date_codes is not None else None
                batch_dates = []
                preds = model(xb)
                loss = huber(preds, yb)
                if self.config.rank_loss_weight > 0:
                    loss = loss + self.config.rank_loss_weight * _cross_section_rank_loss_torch(preds, yb, date_b)
                if self.config.utility_loss_weight > 0:
                    loss = loss + self.config.utility_loss_weight * _utility_loss_torch(preds, yb, self.config.long_short_topk)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * xb.shape[0]
                total_rows += int(xb.shape[0])
            epoch_loss /= max(1, total_rows)
            entry = {"epoch": epoch, "loss": epoch_loss}
            if val_tensor is not None:
                model.eval()
                with torch.no_grad():
                    val_preds = model(val_tensor)
                    val_loss = float(huber(val_preds, val_target).item())
                entry["val_loss"] = val_loss
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.config.early_stopping_patience:
                        history.append(entry)
                        break
            history.append(entry)

        if best_state is not None:
            model.load_state_dict(best_state)

        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
        for linear in linear_layers[:-1]:
            weights.append(linear.weight.detach().cpu().numpy().T)
            biases.append(linear.bias.detach().cpu().numpy())
        output_weights = linear_layers[-1].weight.detach().cpu().numpy().T
        output_biases = linear_layers[-1].bias.detach().cpu().numpy()
        gpu_peak_mb = 0.0
        if on_cuda:
            if self.config.log_gpu_memory:
                gpu_peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                history.append({"gpu_peak_mb": gpu_peak_mb})
            # Release GPU tensors/model before returning so sequential folds do
            # not accumulate allocator fragmentation.
            del model, train_tensor, target_tensor
            if val_tensor is not None:
                del val_tensor, val_target
            torch.cuda.empty_cache()
        return V7DeepAlphaState(
            backend="torch",
            feature_columns=list(feature_columns),
            horizons=list(horizons),
            weights=weights,
            biases=biases,
            output_weights=output_weights,
            output_biases=output_biases,
            feature_means=feature_means,
            feature_scales=feature_scales,
            training_history=history,
            gpu_peak_mb=gpu_peak_mb,
        )

    def _resolve_device(self) -> str:
        import torch

        device = self.config.device
        if device == "cpu":
            if self.config.require_gpu:
                raise RuntimeError("GPU training was required, but device='cpu' was requested")
            return "cpu"
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if self.config.require_gpu:
                raise RuntimeError(
                    "GPU training was required, but torch.cuda.is_available() is false"
                )
            return "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested ({device}), but torch.cuda.is_available() is false"
            )
        return device


def _load_feature_schema(schema_path: str | Path) -> tuple[list[str], str, str]:
    """Read (feature_columns, feature_version, schema_hash) from a schema JSON.

    Accepts the gold dataset ``feature_schema.json`` (``feature_version`` +
    ``schema_hash``) and the trainer's own ``deep_alpha_feature_schema.json``
    (``version``). ``feature_columns`` is required, non-empty, and its order is
    preserved exactly. Fails fast on a missing/empty list.
    """
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"feature_schema_path does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = payload.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"feature schema {path} has no usable 'feature_columns' list")
    version = str(payload.get("feature_version", payload.get("version", "")))
    schema_hash = str(payload.get("schema_hash", ""))
    return [str(c) for c in columns], version, schema_hash


def _rank_loss_torch(predictions, targets):  # pragma: no cover - torch path
    import torch

    pred_rank = predictions.argsort(dim=0).argsort(dim=0).float()
    target_rank = targets.argsort(dim=0).argsort(dim=0).float()
    return torch.mean((pred_rank - target_rank) ** 2) / max(1, predictions.shape[0])


def _cross_section_rank_loss_torch(predictions, targets, date_codes):  # pragma: no cover - torch path
    """Rank loss computed per trading-date cross-section.

    Random mini-batch rank loss conflates cross-sectional and time-series
    structure. Here we group rows by their date code and accumulate a
    rank MSE per cross-section so the optimiser shapes per-day rankings
    directly.
    """
    import torch

    if date_codes is None or predictions.shape[0] <= 1:
        return _rank_loss_torch(predictions, targets)
    unique = torch.unique(date_codes)
    total = predictions.new_zeros(())
    count = 0
    for code in unique:
        mask = date_codes == code
        if int(mask.sum()) < 2:
            continue
        pred_g = predictions[mask]
        target_g = targets[mask]
        pred_rank = pred_g.argsort(dim=0).argsort(dim=0).float()
        target_rank = target_g.argsort(dim=0).argsort(dim=0).float()
        total = total + torch.mean((pred_rank - target_rank) ** 2) / max(1, pred_g.shape[0])
        count += 1
    return total / max(1, count)


def _utility_loss_torch(predictions, targets, topk: int):  # pragma: no cover - torch path
    import torch

    k = min(topk, predictions.shape[0])
    if k <= 0:
        return torch.zeros(())
    top_pred = predictions.topk(k, dim=0).indices
    bottom_pred = (-predictions).topk(k, dim=0).indices
    top_returns = torch.gather(targets, 0, top_pred).mean(dim=0)
    bottom_returns = torch.gather(targets, 0, bottom_pred).mean(dim=0)
    return -(top_returns - bottom_returns).mean()


def run_v7_deep_alpha_training(
    dataset: pd.DataFrame,
    config: V7DeepAlphaTrainerConfig | None = None,
    validation_dataset: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> tuple[V7DeepAlphaTrainer, Path]:
    trainer = V7DeepAlphaTrainer(config)
    trainer.fit(dataset, validation_dataset=validation_dataset)
    saved = trainer.save(output_dir)
    return trainer, saved


@dataclass(frozen=True)
class WalkForwardDeepResult:
    """Outcome of a schema-locked walk-forward deep-alpha run."""

    oos_predictions: pd.DataFrame   # symbol, trade_date, fold_id, train/valid window, model_version, alpha_*d, schema_hash, feature_version
    fold_metadata: pd.DataFrame     # fold_id, train/valid dates + counts, schema_hash, feature_count, backend
    schema_hash: str
    feature_version: str
    feature_columns: list[str]
    run_manifest: dict[str, object] = field(default_factory=dict)
    manifest_path: str | None = None


def run_walk_forward_deep_training(
    dataset: pd.DataFrame,
    *,
    feature_schema_path: str,
    base_config: V7DeepAlphaTrainerConfig | None = None,
    split_config: "WalkForwardSplitConfig | None" = None,
    output_dir: str | Path | None = None,
) -> WalkForwardDeepResult:
    """Train one deep-alpha model per walk-forward fold, **schema-locked**.

    Every fold's trainer is pinned to the *same* ``feature_schema_path`` so the
    feature set (columns, order) and its ``schema_hash`` are identical across
    folds — folds are therefore comparable and the OOS predictions live in one
    stable feature space. Each fold trains only on its past window and predicts
    its (embargoed/purged) validation window; the union is the walk-forward OOS.

    The feature schema is the single contract: if it cannot be loaded, or a fold
    is missing a required column, the underlying trainer fails fast (no silent
    column drift). Per-fold models are saved under ``output_dir/fold_{id}`` when
    ``output_dir`` is given.
    """
    from dataclasses import asdict, replace
    from datetime import datetime, timezone

    from quantagent.training.splitters import WalkForwardSplitConfig, split_walk_forward

    if dataset is None or dataset.empty:
        raise ValueError("walk-forward deep training requires a non-empty dataset")
    # Load the locked contract once so we can report it even before any fold runs.
    locked_columns, locked_version, locked_hash = _load_feature_schema(feature_schema_path)
    # Run-level model version: stable for a given (feature_version, schema_hash).
    model_version = f"{locked_version or 'v?'}@{locked_hash[:12]}"

    frame = dataset.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    split_cfg = split_config or WalkForwardSplitConfig()
    folds = split_walk_forward(frame, config=split_cfg)
    if not folds:
        raise ValueError(
            "walk-forward split produced no folds; check split_config vs the date span"
        )

    base = base_config or V7DeepAlphaTrainerConfig()
    out_root = Path(output_dir) if output_dir else None
    pred_parts: list[pd.DataFrame] = []
    meta_rows: list[dict[str, object]] = []
    fold_checkpoints: dict[str, str] = {}
    for fold in folds:
        train_frame = frame.iloc[fold.train_idx]
        valid_frame = frame.iloc[fold.valid_idx]
        fold_dir = (out_root / f"fold_{fold.fold_id}") if out_root else None
        cfg = replace(
            base,
            feature_schema_path=feature_schema_path,
            output_dir=str(fold_dir) if fold_dir else base.output_dir,
        )
        trainer = V7DeepAlphaTrainer(cfg)
        state = trainer.fit(train_frame, validation_dataset=valid_frame)
        # Schema-lock invariant: every fold rides the identical contract.
        if state.schema_hash != locked_hash:
            raise AssertionError(
                f"fold {fold.fold_id} schema_hash {state.schema_hash} != locked {locked_hash}"
            )
        if fold_dir is not None:
            saved = trainer.save(fold_dir)
            fold_checkpoints[str(fold.fold_id)] = str(saved)

        preds = trainer.predict(valid_frame)
        if not preds.empty:
            # Self-describing OOS rows: each carries its fold, train/valid window
            # (== the prediction window), model version, and schema provenance.
            preds = preds.assign(
                fold_id=fold.fold_id,
                train_start=fold.train_dates[0],
                train_end=fold.train_dates[1],
                valid_start=fold.valid_dates[0],
                valid_end=fold.valid_dates[1],
                model_version=model_version,
                schema_hash=locked_hash,
                feature_version=locked_version,
            )
            pred_parts.append(preds)
        meta_rows.append(
            {
                "fold_id": fold.fold_id,
                "train_start": fold.train_dates[0],
                "train_end": fold.train_dates[1],
                "valid_start": fold.valid_dates[0],
                "valid_end": fold.valid_dates[1],
                "n_train": int(fold.train_idx.size),
                "n_valid": int(fold.valid_idx.size),
                "embargo_days": int(fold.embargo_days),
                "backend": state.backend,
                "schema_hash": state.schema_hash,
                "feature_version": state.feature_version,
                "feature_count": len(state.feature_columns),
                "gpu_peak_mb": float(state.gpu_peak_mb),
            }
        )

        # Explicit per-fold CUDA cache clear so sequential folds don't OOM by
        # accumulating allocator fragmentation (belt-and-suspenders alongside
        # the in-fit release).
        if state.backend == "torch":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — cache clearing is best-effort
                pass

    oos = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    fold_plan = pd.DataFrame(meta_rows)

    # Reproducibility contract: everything needed to re-run / verify the run.
    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model_version,
        "schema_hash": locked_hash,
        "feature_version": locked_version,
        "feature_count": len(locked_columns),
        "feature_columns": list(locked_columns),
        "feature_schema_path": str(feature_schema_path),
        "seed": int(base.seed),
        "horizons": list(base.horizons),
        "use_torch": bool(base.use_torch),
        "require_gpu": bool(base.require_gpu),
        "device": base.device,
        "gpu_peak_mb_max": float(max((r.get("gpu_peak_mb", 0.0) for r in meta_rows), default=0.0)),
        "split_config": asdict(split_cfg),
        "n_folds": int(len(folds)),
        "n_oos_predictions": int(len(oos)),
        "dataset_rows": int(len(frame)),
        "dataset_symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
        "dataset_dates": int(frame["trade_date"].nunique()),
        "fold_plan": [
            {k: (str(v) if isinstance(v, (pd.Timestamp,)) else v) for k, v in row.items()}
            for row in meta_rows
        ],
        "fold_checkpoints": fold_checkpoints,
    }

    manifest_path: str | None = None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)
        fold_plan.to_csv(out_root / "fold_plan.csv", index=False)
        if not oos.empty:
            try:
                oos.to_parquet(out_root / "walkforward_predictions.parquet", index=False)
                manifest["predictions_path"] = str(out_root / "walkforward_predictions.parquet")
            except Exception:  # noqa: BLE001 — fall back to csv on a parquet-less box
                oos.to_csv(out_root / "walkforward_predictions.csv", index=False)
                manifest["predictions_path"] = str(out_root / "walkforward_predictions.csv")
        manifest["fold_plan_path"] = str(out_root / "fold_plan.csv")
        manifest_path = str(out_root / "run_manifest.json")
        Path(manifest_path).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )

    return WalkForwardDeepResult(
        oos_predictions=oos,
        fold_metadata=fold_plan,
        schema_hash=locked_hash,
        feature_version=locked_version,
        feature_columns=list(locked_columns),
        run_manifest=manifest,
        manifest_path=manifest_path,
    )


__all__ = [
    "V7DeepAlphaTrainerConfig",
    "V7DeepAlphaState",
    "V7DeepAlphaTrainer",
    "run_v7_deep_alpha_training",
    "WalkForwardDeepResult",
    "run_walk_forward_deep_training",
]


# Helper alias so iterable parameters keep mypy happy when callers pass tuples
_ = Iterable
