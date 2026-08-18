# Round 22 — R1 回测专家：撮合语义三修 / Fill-semantics repairs

- 日期 / Date: 2026-08-19
- 基线 commit: `5957011`
- 分支 / Branch: `agent/round22-fill-semantics`
- 角色 / Role: R1 回测专家（可改 `src/quantagent/backtest/**`、`quant_math/ashare.py`、
  `quant_math/transaction_cost.py`、`tests/backtest/**`）
- Python: `AI_quant_venv/bin/python3`
- 修复对象：Round 21 自审报告 `docs/audits/round21/01_backtest.md` 的 **F-03 / F-04 / F-05**

> 纪律：证据缺失记 `unknown`，永不记 pass。每条修复配测试，并用 `git stash`
> 验证该测试对**修复前**的代码会失败。

> 已闭合的前提（本轮不动）：`NAV(t)` 恰好包含 `fill_date <= t` 的成交
> （`docs/interior_bar_nav_defect.md` 已 CLOSED）。
> `tests/backtest/test_interior_bar_nav_clock.py` 与 `tests/domain/test_composite_replay.py`
> 是绑定约束，本轮改动不得削弱。

---

## 0. 基线 / Baseline

```
AI_quant_venv/bin/python3 -m pytest tests/backtest tests/domain \
    tests/test_golden_backtest_scenarios.py tests/test_backtest_engine.py \
    tests/quant_math -q -p no:cacheprovider
-> 380 passed, 26 warnings in 11.30s
```

---

## 1. F-03 —— 涨跌停判据与撮合价不同源

### 1.1 缺陷复述

- `src/quantagent/quant_math/ashare.py` 的 `limit_up_mask` / `limit_down_mask`
  用 `close` 与 `prev_close` 比较判板。
- `src/quantagent/backtest/engine.py` 的 `fill_price_column` 默认 `"open"`，
  成交价取 `fill_date` 那根 bar 的 `open`。

判据看收盘、成交走开盘 ⇒ 两个方向都会错。

### 1.2 实测规模（真实面板，非构造）

面板：`runtime/_VOID_pre_round20_backtests/runtime_walkforward_smallscale/panel_subset.parquet`
（30 个真实 A 股标的，2022-01-04 → 2026-05-18，31,650 行 raw OHLCV）。

| 判据 | limit-up bar 数 | limit-down bar 数 |
|---|---|---|
| 按 `close`（引擎当前用的） | 210 | 56 |
| 按 `open`（引擎实际成交的价） | 31 | 9 |

分歧明细：

| 分歧类型 | 涨停 | 跌停 | 后果 |
|---|---|---|---|
| close 判板、open 未封（**假阻断**） | **189** | **47** | 开盘本可成交，引擎拒单 |
| open 封板、close 未封（**假成交**） | **10** | 0 | 引擎在买不到的涨停开盘价成交，reject 日志看不出 |

即：在这 30 个标的 4 年多的样本里，引擎当前用的涨停判据里
**189/210 = 90.0% 是假阻断**，另有 **10 根 bar 是无法从日志发现的假成交**。

零区间 bar（`high == low` 且 `volume > 0`）共 **10** 根，**全部**落在板上——
即一字板；仓库此前对它没有任何专门判据。

（待补：裁定、修法、stash 对照、数字对比。）
