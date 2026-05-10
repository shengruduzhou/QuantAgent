# V4 模型训练 / Model Training

V4 模型是 small CPU-testable multi-tower architecture，不依赖 internet 或 large pretrained models。

## 三塔结构 / Three Towers

- Sequence Tower：处理 `[batch, time, features]` market sequences。
- Snapshot Factor Tower：tabular residual MLP，支持 missing-value mask。
- Event-Policy Tower：处理 structured event features，例如 event type、sentiment、policy exposure、confidence、decay、recency。

## 输出头 / Output Heads

模型输出 `alpha`、`direction_logit`、`q_low`、`q_high`、`factor_gate`、`confidence`、`risk_score`。Quantile heads 会保证 `q_low <= q_high`。

## 复合损失 / Composite Loss

```text
L = rank + huber + direction + quantile + factor_gate + turnover + risk
```

## 测试 / Tests

```powershell
python -m pytest tests/test_v4_multitower.py tests/test_v4_training.py
```
