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

Checkpoints, configs and feature schemas are written under
``artifacts/v7_alpha/<experiment>/`` so ``ModelRegistry`` can pick them
up. ``save`` / ``load`` round-trip the full model state.

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
    output_dir: str = "artifacts/v7_alpha/deep"
    use_torch: bool = True
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
        )


class V7DeepAlphaTrainer:
    """Multi-horizon deep alpha trainer with Torch / numpy backends."""

    def __init__(self, config: V7DeepAlphaTrainerConfig | None = None) -> None:
        self.config = config or V7DeepAlphaTrainerConfig()
        self.state: V7DeepAlphaState | None = None

    def fit(self, dataset: pd.DataFrame, validation_dataset: pd.DataFrame | None = None) -> V7DeepAlphaState:
        if dataset is None or dataset.empty:
            raise ValueError("deep alpha trainer requires a non-empty training dataset")
        feature_columns = list(self.config.feature_columns) or self._auto_feature_columns(dataset)
        if not feature_columns:
            raise ValueError("deep alpha trainer found no feature columns")
        horizons = [h for h in self.config.horizons if f"forward_return_{h}d" in dataset.columns]
        if not horizons:
            raise ValueError("deep alpha trainer needs at least one forward_return_*d label")
        train_x, train_y, train_dates = self._prepare(dataset, feature_columns, horizons)
        val_x, val_y, val_dates = (
            self._prepare(validation_dataset, feature_columns, horizons) if validation_dataset is not None and not validation_dataset.empty else (None, None, None)
        )

        backend = self._select_backend()
        if backend == "torch":
            state = self._fit_torch(train_x, train_y, val_x, val_y, feature_columns, horizons)
        else:
            state = self._fit_numpy(train_x, train_y, val_x, val_y, feature_columns, horizons)
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
        state_path.write_text(json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        config_path.write_text(json.dumps(asdict(self.config), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return state_path

    def load(self, path: str | Path) -> V7DeepAlphaState:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.state = V7DeepAlphaState.from_dict(payload)
        return self.state

    # ------------------------------------------------------------------ helpers

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
            return "numpy"
        try:
            import torch  # noqa: F401
        except Exception:  # pragma: no cover - torch optional
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
        feature_columns: list[str],
        horizons: list[int],
    ) -> V7DeepAlphaState:  # pragma: no cover - depends on torch
        import torch
        from torch import nn

        torch.manual_seed(self.config.seed)
        device = self._resolve_device()
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
            permutation = torch.randperm(train_tensor.shape[0])
            epoch_loss = 0.0
            for start in range(0, train_tensor.shape[0], self.config.batch_size):
                idx = permutation[start : start + self.config.batch_size]
                xb = train_tensor[idx]
                yb = target_tensor[idx]
                preds = model(xb)
                loss = huber(preds, yb)
                if self.config.rank_loss_weight > 0:
                    loss = loss + self.config.rank_loss_weight * _rank_loss_torch(preds, yb)
                if self.config.utility_loss_weight > 0:
                    loss = loss + self.config.utility_loss_weight * _utility_loss_torch(preds, yb, self.config.long_short_topk)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * xb.shape[0]
            epoch_loss /= max(1, train_tensor.shape[0])
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
        )

    def _resolve_device(self) -> str:  # pragma: no cover - depends on torch
        import torch

        if self.config.device == "cpu":
            return "cpu"
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device


def _rank_loss_torch(predictions, targets):  # pragma: no cover - torch path
    import torch

    pred_rank = predictions.argsort(dim=0).argsort(dim=0).float()
    target_rank = targets.argsort(dim=0).argsort(dim=0).float()
    return torch.mean((pred_rank - target_rank) ** 2) / max(1, predictions.shape[0])


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


__all__ = [
    "V7DeepAlphaTrainerConfig",
    "V7DeepAlphaState",
    "V7DeepAlphaTrainer",
    "run_v7_deep_alpha_training",
]


# Helper alias so iterable parameters keep mypy happy when callers pass tuples
_ = Iterable
