# 数学优化层 / Math Optimization Layer

V4 的 optimization layer 只负责 target weights，不负责 broker orders。它接收 model alpha、agent posterior、factor gate confidence、risk confidence 和交易成本，输出 constrained portfolio。

## Alpha 融合 / Alpha Blend

`blend_alpha_and_views` 将 model alpha、conformal interval、factor gate、Agent BL posterior、regime multiplier 和 risk confidence 合成 `blended_alpha`。

```text
model_alpha + AgentView posterior + confidence shrink
-> blended_alpha
```

## 组合优化 / Portfolio Optimization

`solve_v4_portfolio` 支持：

- `long_only_enhancement`
- `hedged_alpha`
- `market_neutral_placeholder`

约束包括 max name weight、sector weight、beta range、turnover budget、liquidity/capacity、tradability、cost-aware objective。

## Fallback / 降级

如果 `cvxpy` 不可用，`ContinuousMeanVarianceOptimizer` 使用 deterministic fallback。这样 synthetic flow 不依赖 heavy optimization packages。

## 测试 / Tests

```powershell
python -m pytest tests/test_v4_portfolio.py tests/test_quant_math.py
```
