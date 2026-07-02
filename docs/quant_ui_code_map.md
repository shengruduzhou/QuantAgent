# QuantAgent Quant UI 代码地图 / Code Map

> 扫描日期：2026-06-20  
> 范围：仅项目目录内的 `src/`、`scripts/`、`tests/`、`configs/`、`docs/`、`metric/` 与 `runtime/`。  
> 结论：仓库当前没有成熟 Web/API 前端层；量化核心、严格 A 股执行仿真、分时做 T、因子评估、模型训练与大量真实 runtime 产物已经存在，适合新增独立 adapter/service/API/UI 层，不应重写核心逻辑。

## 1. Repository overview

### 1.1 规模与技术栈

- Python package：`src/quantagent/`，Python 3.11+，Pandas / Polars / PyArrow / Pydantic / Typer。
- CLI：`quantagent.cli` 下约 90 个 V7/V8 commands。
- 测试：`tests/` 共 177 个 Python test files。
- 研究脚本：`scripts/` 顶层约 99 个脚本。
- Runtime：当前约 77 GB，主要包含 Parquet、JSON、CSV、model checkpoints、logs。
- Frontend：原始扫描未发现成熟 Web 层；现已新增 `apps/quant-ui/` React + TypeScript + Vite terminal。
- API：原始扫描未发现服务层；现已新增 `services/quant_api/` FastAPI adapter/API。

Quant UI indexer 实现后会跳过 Qlib provider 内部二进制目录与
`feature_cache/outcome_cache` 中间缓存；当前可展示索引约 9,000 个 artifact
（runtime 会随训练任务继续变化），
原始 runtime 文件仍保持不变。

### 1.2 核心模块树

```text
src/quantagent/
├── agents/                 Agent views、LLM orchestrator、specialists
├── backtest/               event backtest、strict V8、A-share execution simulation
├── cli/                    V7/V8 Typer commands
├── config/paths.py         runtime path single source of truth
├── data/                   PIT data、providers、manifests、dataset builders
├── diagnostics/            health、decision report、post-mortem、IC diagnostics
├── domain/schemas.py       research-level domain objects
├── ensemble/               calibration、blend、meta-label、strict search
├── execution/              OrderManager、VirtualBroker、cost/fill、T+1 ledger、QMT dry-run
├── factors/                Alpha101/181、CICC、GTJA、intraday、evaluation、registry
├── fundamental/            due diligence、fraud、valuation、financial features
├── models/                 classical/deep/FT-Transformer models
├── optimization/           GA、Optuna、multi-objective loss
├── paper/                  paper loop
├── portfolio/              target weights、position state、Do-T overlay、decision chain
├── quant_math/             performance、IC、risk、cost、optimizer
├── research/               intraday Do-T EV/factor-combo/walk-forward
├── risk/                   RiskGate、KillSwitch、microstructure guards
├── rl/                     PPO environments and training
├── services/               existing V7 pipeline orchestration
├── strategy/               deterministic signals、sizing、score fusion
├── themes/                 theme discovery、industry chain、stock pool gate
└── training/               splitters、trainers、predictor、Do-T labels/models
```

## 2. Canonical storage and PIT rules

`src/quantagent/config/paths.py` 是 runtime single source of truth。默认指向项目内 `runtime/`，可由环境变量覆盖。Quant UI 必须：

- 通过 `quant_paths()` 解析 runtime，而不是在 service 中散落硬编码路径。
- 对 API 输出只返回项目相对路径。
- 不向浏览器返回 model checkpoint 内容。
- 对 silver/gold artifact 优先读取对应 `runtime/data/v7/manifests/*.json`。
- 保留 `available_at`、`point_in_time_valid` 与 manifest 的 PIT/quality 状态。

当前 `market_panel.parquet` 已确认字段：

```text
symbol, trade_date, open, high, low, close, volume, amount,
available_at, source, source_type, source_reliability,
point_in_time_valid, is_st, is_st_provenance,
is_suspended, is_limit_up, is_limit_down
```

## 3. Entrypoints

### 3.1 Training

主要入口：

- `src/quantagent/cli/v7_train.py`
  - `train-alpha-v7`
  - `train-deep-alpha-v7`
  - `run-real-training-v7`
  - `predict-alpha-v7`
  - `build-target-weights-v7`
  - `run-full-real-training-v7`
  - `synthesize-factors-v7`
- `src/quantagent/cli/v8_deep.py`
  - `train-v8-deep`
- `src/quantagent/training/v8_pipeline.py`
  - `run_v8_training_pipeline`
  - pipeline：data router → labels → factors → GA → target weights → strict backtest → decision report
- `src/quantagent/training/v7_predictor.py`
  - `predict_v7_alpha`
  - 支持 classical、deep、FT-Transformer artifact。
- `src/quantagent/training/model_registry.py`
  - `ModelRegistry`

训练产物主要位于：

```text
runtime/models/
runtime/models/registry/
runtime/reports/v8/deep/<run_id>/<horizon>/
runtime/logs/
```

### 3.2 Backtest

主要入口：

- `src/quantagent/backtest/engine.py`
  - `EventDrivenBacktester`
  - T+1、涨跌停、停牌、lot、cost、slippage。
- `src/quantagent/backtest/ashare_execution_simulator.py`
  - `simulate_ashare_target_weights`
  - 通过 `OrderManager` + `VirtualBroker` 执行 target weights。
- `src/quantagent/backtest/strict_v8.py`
  - `run_strict_backtest_v8`
  - 输出 metrics、NAV、trades、failed orders、risk events、realized PnL。
- `src/quantagent/backtest/full_pipeline_backtester.py`
  - PIT daily callback replay；不是 production-grade execution simulator。
- CLI：
  - `walk-forward-backtest-v7`
  - `run-paper-backtest-v7`
  - `run-strict-a-share-backtest-v8`
  - `run-gated-backtest-v8`

### 3.3 Selection and target weights

- `src/quantagent/themes/stock_pool_selector.py`
  - theme pool：core / strong / satellite / watchlist / exclusion。
- `src/quantagent/themes/stock_pool_gate.py`
  - hard gate 与 drop reasons。
- `src/quantagent/portfolio/v7_target_weights.py`
  - predictions → tradability/liquidity/timing/threshold/top-K/sector/turnover constraints → target weights。
- `src/quantagent/portfolio/decision_chain/chain.py`
  - 15-step decision chain：
    `alpha → liquidity → tradable → limit → ST → sector → market gate → regime → fundamental → policy → broker → drawdown → concentration → risk budget → gross exposure`。
- `src/quantagent/strategy/decision_engine.py`
  - normalized scores → deterministic action / target weights。

### 3.4 Factors

- `src/quantagent/factors/registry.py`
  - `FactorMeta`、`FactorRegistry`。
- `src/quantagent/factors/alpha101.py`
  - Alpha101 implementations + registry descriptions/directions。
- `src/quantagent/factors/alpha181.py`
  - Alpha101 + 80 CICC-inspired factors，支持 synthesized extensions。
- `src/quantagent/factors/cicc_ashare80.py`
- `src/quantagent/factors/gtja191.py`
- `src/quantagent/factors/intraday_volume_price.py`
  - 16 个 minute-derived day factors。
- `src/quantagent/factors/evaluation.py`
  - IC、Rank IC、ICIR、decay、quantile returns、turnover、capacity。
- `src/quantagent/factors/composite.py`
  - cross-sectional standardization、neutralization、ICIR/decay/capacity weights。
- `src/quantagent/factors/factor_synthesis.py`
  - symbolic/rd-agent style factor discovery。

### 3.5 Risk and position lifecycle

- `src/quantagent/risk/risk_gate.py`
  - target-weight 与 order-intent gates。
- `src/quantagent/execution/risk_kill_switch.py`
  - daily loss、drawdown、exposure、reject rate、stale data、manual lock。
- `src/quantagent/portfolio/position_state.py`
  - hard stop、soft stop、trailing stop、breakeven、time/event/liquidity exit。
- `src/quantagent/portfolio/state_machine/machine.py`
  - position lifecycle state machine。
- `src/quantagent/risk/microstructure_guard.py`
  - per-stock defensive risk flags。

### 3.6 Do-T / intraday

- `src/quantagent/execution/intraday_dot_decision.py`
  - `SELL_HIGH`、`BUY_BACK`、`BUY_LOW`、`SELL_RISK`，含 confidence、qty、price、failure control。
- `src/quantagent/execution/intraday_ev_engine.py`
  - EV actions：`NO_TRADE`、`SELL_HIGH`、`BUY_BACK`、`BUY_LOW`、`SELL_AFTER_BUY`。
- `src/quantagent/execution/intraday_ledger.py`
  - T+1-legal pair ledger。
- `src/quantagent/portfolio/do_t_overlay.py`
  - minute bars + yesterday inventory → legal Do-T round trips。
- `src/quantagent/training/do_t_labels.py`
- `src/quantagent/training/do_t_roundtrip_labels.py`
- `src/quantagent/training/do_t_models.py`
- `src/quantagent/research/intraday_dot_ev_backtest.py`
- `src/quantagent/research/intraday_dot_factor_combo.py`
- CLI：`run-do-t-overlay-v8`。

## 4. Core object semantic mapping

| UI concept | Existing code / artifact | Confirmed data | Adapter action |
|---|---|---|---|
| 股票池 | stock pool selector/gate、hybrid stock pool parquet | symbol、sector、scores、action bucket、rank | direct + normalize |
| K 线 | market panel | OHLCV + amount + tradability flags | filter by symbol/date and downsample |
| 因子定义 | FactorRegistry、Alpha101/181/CICC/GTJA source | meta or source-derived formula/description | AST/source catalog |
| 模型训练 | FT metrics/config/schema、registry | training history、features、horizons、GPU/config | direct |
| 模型推理 | predictions parquet、ensemble composite | alpha/composite/horizon scores | direct |
| 买卖信号 | order audit `side` + status | buy/sell, quantity, price, status | map to BUY/SELL |
| 做 T 信号 | Do-T decision/EV/overlay artifacts | actions/pairs/entry/exit/qty/edge | map to T_BUY/T_SELL |
| 仓位 | position history、holdings_daily、target weights | shares or weights depending artifact | direct; derive only when inputs exist |
| 买卖数量 | order audit / realized trades / Do-T overlay | quantity/filled_quantity | direct |
| 风控 | risk events、RiskGate、KillSwitch、position state | event type/reason/violations | direct + categorize |
| 止盈止损 | position state / Do-T `state` | hard/soft/trailing/profit/stop states | direct when logged; no inference from price alone |
| 撮合 | VirtualBroker / FillSimulator | status、fill qty、avg price、message | direct |
| 交易成本 | AShareCostModel / strict realized trades | commission/stamp/transfer/impact or aggregate cost | direct/derive with provenance |
| 收益曲线 | nav.csv / pnl.csv | nav、daily_return | direct |
| 回撤 | NAV | not always stored as a column | derive from running max |

## 5. Runtime taxonomy

### 5.1 Backtests

Strong signature：

```text
<dir>/backtest/metrics.json
<dir>/backtest/nav.csv
<dir>/backtest/pnl.csv
<dir>/backtest/trades.csv
<dir>/backtest/realized_trades.csv
<dir>/backtest/failed_orders.csv
<dir>/backtest/risk_events.json
<dir>/backtest/selected_stocks.csv
```

已验证的 strict V8 trade fields：

```text
trade_date, client_order_id, status, filled_quantity, avg_price,
last_message, symbol, side, quantity, reference_price
```

已验证的 realized trade fields：

```text
symbol, buy_date, sell_date, quantity, buy_price, sell_price,
gross_pnl, cost, net_pnl
```

同名 `trades.csv` 不代表一定是成交回报。当前已确认：

- strict V8 order blotter：可映射 `Trade`。
- board-chase touch table：研究事件，不含 side/quantity，不映射 `Trade`。
- Do-T discovery daily table：研究汇总，不含逐腿成交，不映射 `Trade`。

Adapter 必须先做 schema capability detection，禁止按文件名直接假设成交语义。

### 5.2 Models

Strong signature：

```text
ft_transformer.pt
ft_transformer_config.json
ft_transformer_feature_schema.json
ft_transformer_metrics.json
run_config.json
predictions.parquet
```

`ft_transformer_metrics.json` 已包含 `training_history[{epoch, loss, val_loss, finite_steps, nonfinite_steps}]`。

统一 Model Adapter 还识别：

```text
runtime/models/**/registry/*.json
runtime/models/**/policy.zip
runtime/reports/v8/**/policy.zip
runtime/reports/**/do_t_models.joblib
runtime/{models,reports}/**/*.{pt,pth,joblib,pkl,pickle,zip}
```

Specialized mapping：

- Deep FT：training history、config/schema、predictions、backtest metrics、checkpoint metadata。
- Registered Alpha：registry metadata、quality flags、可用 prediction path。
- RL Policy：training summary、strict verdict/evaluation、`weights_test.parquet`。
- T+1 model bundle：`do_t_models.joblib` + EV backtest report。
- Generic binary：至少进入 catalog 并展示 metadata-only issue，不再完全不可见。

Checkpoint 只作为 artifact metadata 暴露；API 从不 deserialize 或返回 binary content。

注意：`runtime/models/v7_alpha/registry/latest.json` 可能被 tests 写入 test artifact 路径。Indexer 不得无条件把 `latest.json` 视为最新 production model；必须检查 artifact 是否在 runtime 内、metadata、样本数和 production-ready/quality flags。

本次确认 `runtime/models/v7_alpha/registry/` 的 90 个 records 全部指向项目外部
测试目录，已通过 audited cleanup 删除；canonical `runtime/models/registry/` 保持保护。

### 5.3 Factors

```text
runtime/data/v7/silver/factors/
runtime/reports/v7/factor_synthesis*/
runtime/reports/v8/factor_*/
runtime/reports/intraday_dot_factor_combo*/
```

当前 synthesized leaderboard 可以是 empty/null schema；UI 必须显示“本次搜索无 surviving factor”，不能伪造结果。

### 5.4 Selection

Hybrid stock-pool artifacts 已确认包含：

```text
prediction, model_rank,
core_policy_score, core_sentiment_score, fundamental_quality_score,
cicc_* scores, sector_resonance_score, dip_buy_flow_score,
old_dealer_risk_score, trend_strength_score,
do_t_suitability_score, llm_stock_score, llm_confidence,
hybrid_score, action_bucket, hybrid_rank,
research_weight_hint, no_orders_generated
```

### 5.5 Do-T

两类 artifact 不应混淆：

1. `do_t_overlay` round-trip artifacts：明确 `buy_time/sell_time/buy_price/sell_price/quantity/net_pnl`。
2. factor-combo research artifacts：`entry_px/exit_px/requested_qty/filled_qty/fill status/reason/pred_*`。

旧 `paper/replay_2026/dot_overlay_trades.csv` 只有 daily `gross_ret/net_ret/state`，没有真实 minute price/quantity。UI 在该数据源上不得显示伪造的成交价和数量。

## 6. Missing-field policy

### 6.1 可以直接读取

- OHLCV、amount、tradability flags。
- strict backtest side/status/filled quantity/avg price。
- realized buy/sell price and net PnL。
- Do-T overlay entry/exit/quantity/net PnL。
- model score、training history、feature list。
- factor IC/quantile/decay，前提是对应 artifact 存在。

### 6.2 可以推导，但必须记录 provenance

- `amount = price * filled_quantity`。
- fee/tax/slippage：仅在 artifact 有 component 或明确 config 时推导；否则只给 aggregate cost。
- position after：从按时间排序的 filled trades 累积，必须遵守 T+1 ledger semantics。
- cash after：需要 initial cash + complete fill/cost history；缺一项则不推导。
- drawdown：从 NAV running max 推导。
- cumulative PnL：从 realized trades 累加。
- action subtype ADD/REDUCE：需与交易前 position 比较。
- stock name：通过 code-name map 标准化 code 后 join。

### 6.3 不可凭空推导

- 缺少 reason 时，不从盈亏猜测 signal reason。
- 缺少 factor contribution 时，不用 model feature importance 冒充单笔交易贡献。
- 缺少 minute pair 时，不根据日 K 线猜 Do-T 点。
- 缺少 stop event 时，不把亏损卖出自动标成 STOP_LOSS。
- 缺少 cash ledger 时，不显示伪精确 cashAfter。
- 缺少 independent factor trade artifact 时，不声明存在单因子买卖点。

## 7. Adapter seams required

建议新增：

```text
services/quant_api/
├── app.py
├── config.py
├── schemas/
├── adapters/
│   ├── backtests.py
│   ├── factors.py
│   ├── models.py
│   ├── selection.py
│   ├── risk.py
│   ├── market.py
│   └── jobs.py
├── runtime_indexer/
│   ├── indexer.py
│   ├── registry.py
│   └── parsers/
├── services/
└── routes/
```

关键 seam：

- Runtime formats are heterogeneous：需要 signature-based parser registry。
- Wide target weights：API 前转换成长表，不向前端发送数千列 wide frame。
- Large Parquet：只做 projection/filter/limit，禁止全表加载。
- Huge `risk_events.json`：需要流式/分页读取或预建轻量索引。
- Factor metadata 不统一：优先 registry，随后 source catalog，最后 runtime-only factor。
- Job execution：只允许 allowlisted project commands，不接受任意 shell string。
- Job default 必须 research/dry-run，绝不打开 live trading。

## 8. Defensive handling

- Parser 返回 `status: ready | partial | empty | error` 与 `issues[]`。
- 所有 optional 字段返回 `null`，前端统一显示“暂无数据”。
- 文件读取失败不导致整个 runtime index 失败。
- Index cache key 至少包含 path、mtime、size。
- 大文件 metadata scan 与 row data read 分离。
- API path 参数不得逃逸项目根目录。
- 输出 path 一律 repository-relative。
- Production/live flags 不由 UI 自动开启。

## 9. Known uncertainties

- 不同 V7/V8/V8.9 backtest bundle 字段并不完全一致。
- 多数 strict backtest trade audit 不包含逐笔 signal reason、factor contribution、cashAfter。
- 历史 Do-T artifacts 有的只有 daily return，没有 minute fills。
- 单因子 IC/quantile artifacts 分散，且并非每个 factor 都有独立 trade simulation。
- SHAP artifact 未发现统一格式；只在文件实际存在时展示。
- 部分 runtime JSON 保存了旧绝对路径；API 必须转换为项目相对路径或标记 external/stale，不直接暴露。
- `latest.json` 可能指向 test artifact，不能作为唯一 freshness source。

## 10. Implementation decision

- Backend：FastAPI，作为独立 `services/quant_api` package，不污染核心 quant modules。
- Frontend：React + TypeScript + Vite，独立 `apps/quant-ui`。
- Charts：ECharts，K-line 使用 progressive rendering/data zoom。
- State/data：TanStack Query + lightweight local store。
- Realtime：SSE 用于 jobs/log progress；不需要双向控制时不引入 WebSocket。
- First-class pages：Dashboard、Backtest Lab、Stock Replay、T+1 Analysis、Factor Center、Selection Logic、Model Lab、Risk Center、Runtime Explorer、Reports。
