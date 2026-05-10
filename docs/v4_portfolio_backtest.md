# V4 组合与回测 / Portfolio and Backtest

V4 Portfolio layer 输出 target weights，Backtest layer 用 A-share rule engine 模拟 T+1、涨跌停、停牌、partial fills 和 costs。

## Portfolio Modes / 组合模式

- `long_only_enhancement`：指数增强 long-only。
- `hedged_alpha`：带 ETF/futures hedge placeholder 的 alpha 组合。
- `market_neutral_placeholder`：市场中性占位模式。

## Backtester / 回测器

`EventDrivenBacktester` 输出 nav curve、daily returns、holdings、trades、rejects、diagnostics 和 report。Reject reasons 包括 suspension、limit-up no buy、limit-down no sell、invalid lot、T+1 insufficient shares。

## 测试 / Tests

```powershell
python -m pytest tests/test_v4_portfolio.py tests/test_backtest_v4.py
```
