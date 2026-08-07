# Financial-API visualization mapping

Authoritative references:

- https://github.com/HiThink-Tech/Financial-API/tree/main/examples/inspirations
- https://fuyao.aicubes.cn/llms.txt
- https://fuyao.aicubes.cn/llms-full.txt
- https://fuyao.aicubes.cn/docs/

Before changing market visualization or Fuyao adapters, refresh the references above. Do not infer undocumented fields or paths.

## QuantAgent mapping

| Financial-API inspiration | QuantAgent surface | Data contract |
|---|---|---|
| 01 Stock overview | `/stock-replay` | ticker search + snapshot + forward-adjusted history + valuation; persisted trades are overlays only |
| 02 Financial health | `/market-intelligence` → 财务体检 | annual income/balance/cash-flow statements; `report_date_ms` controls historical availability |
| 03 Concept constituents | `/market-intelligence` → 行业/概念 | THS index catalog + index history + current constituents + constituent snapshot |
| 04 Limit-up ladder | `/market-intelligence` → 市场脉冲 | limit-up ladder capability; do not infer a broken-board pool |
| 05 Watchlist anomalies | market capability boundary | anomaly endpoint may be added only from documented REST schema; no fabricated events |
| 06 MarketDB research | Data Lab | market dump / local panel; show data freshness and adjustment basis |
| 07 Heat radar | `/market-intelligence` → 市场脉冲 | hot-stock and skyrocket are separate semantics; never collapse into one trading score |
| 08 Dragon-tiger watch | `/market-intelligence` → 市场脉冲 | all/org/hot_money boards, preserve signed net values |
| 09 Industry strength | `/market-intelligence` → 行业/概念 | index history plus current constituent breadth; current constituents are not historical constituents |
| 10 Cash-flow quality | `/market-intelligence` → 财务体检 | cash-flow rows aligned to the same period and disclosure date |
| 11 Heat-price relation | `/stock-replay` + future rank-trend overlay | rank and price must share a trading-day axis; no hour/day mixing |
| 12 Limit-up sentiment | `/market-intelligence` → 市场脉冲 | ladder is a finite returned sample; do not call its transition rate a market-wide promotion rate |
| 13 Breakout backtest | `/backtests` | persisted QuantAgent backtest and execution assumptions |
| 14 TSMOM backtest | `/backtests` | persisted QuantAgent backtest and execution assumptions |
| 15 Short-term reversal | `/backtests` | persisted QuantAgent backtest and execution assumptions |
| 16 Dragon-tiger capital flow | `/market-intelligence` → 市场脉冲 | signed dragon-tiger net values; visualization only, not an order instruction |

## UI rules

1. Use one-page financial-product hierarchy: current state first, chart/evidence second, provenance and caveats visible.
2. Use real API or persisted artifact data only. Missing data renders empty/unavailable, never a mock value.
3. API keys remain server-side. Frontend receives normalized data plus provenance only.
4. A-share convention in QuantAgent is up=red and down=green.
5. Charts must support responsive sizing; dense market charts should expose tooltip/crosshair and appropriate zoom/pan.
6. Market observation and QuantAgent decision evidence remain separate layers. Hot lists, limit-up data and dragon-tiger data are not direct buy/sell signals.
