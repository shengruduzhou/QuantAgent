# QuantAgent Quant UI 数据协议 / Data Schema

> 本协议基于实际代码与 `runtime/` artifact 字段设计。  
> 原则：source-backed、nullable by default、provenance explicit、missing data never crashes UI。

## 1. Common response envelope

```ts
type DataStatus = "ready" | "partial" | "empty" | "error"

type DataIssue = {
  code: string
  message: string
  path?: string
  recoverable: boolean
}

type Provenance = {
  sourcePath?: string
  sourceType?: string
  parser?: string
  derivedFields?: string[]
  generatedAt?: string
  indexedAt?: string
}

type ApiResponse<T> = {
  status: DataStatus
  data: T
  issues: DataIssue[]
  provenance?: Provenance
}
```

Rules：

- `data` 不使用 `undefined`；缺失值使用 `null`。
- `partial` 表示主体可展示，但有字段/文件缺失。
- `empty` 表示数据源合法但没有记录。
- `error` 只用于请求主体无法生成；单个 artifact 失败不应拖垮列表。
- `sourcePath` 必须是项目相对路径。

## 2. Runtime artifact

```ts
type ArtifactKind =
  | "backtest"
  | "model"
  | "prediction"
  | "target_weights"
  | "factor"
  | "selection"
  | "risk"
  | "do_t"
  | "report"
  | "log"
  | "manifest"
  | "dataset"
  | "unknown"

type RuntimeArtifact = {
  id: string
  kind: ArtifactKind
  name: string
  path: string
  extension: string
  sizeBytes: number
  modifiedAt: string
  status: DataStatus
  parser?: string
  runId?: string
  strategyVersion?: string
  modelVersion?: string
  horizon?: string
  symbols?: number
  rows?: number
  dateStart?: string
  dateEnd?: string
  tags: string[]
  issues: DataIssue[]
}
```

Indexer identity：

```text
artifact id = stable hash(relative path)
cache       = full metadata snapshot with bounded TTL
```

Runtime Explorer filters：

- `kind` / `extension` / free-text `query`。
- `runId` / `horizon`。
- `modifiedAfter` / `modifiedBefore`。
- `strategy` / `model` / `symbol` 采用 metadata/path matching，不扫描大 Parquet 内容。
- 股票级数据检索使用 backtest/model/selection domain API，避免 Runtime Explorer 做全湖内容搜索。

## 3. Market and K-line

```ts
type KlineBar = {
  datetime: string
  symbol: string
  open: number
  high: number
  low: number
  close: number
  volume: number | null
  amount: number | null
  availableAt: string | null
  isSt: boolean | null
  isSuspended: boolean | null
  isLimitUp: boolean | null
  isLimitDown: boolean | null
  source: string | null
}
```

Field source：

| API field | Source | Handling |
|---|---|---|
| OHLC | market panel | required for a returned bar |
| volume/amount | market panel | nullable |
| availableAt | `available_at` | nullable, never synthesize |
| flags | market panel | nullable |
| source | market panel | nullable |

Large-series rule：

- Default maximum 2,000 bars per response。
- 超限时返回请求区间内最后 N 根真实 bar，不生成合成 OHLC，确保买卖点时间仍能精确对齐。
- API 返回 `sampled: true` 和原始/返回点数；前端需要提示当前为 windowed view。

## 4. Trade

```ts
type TradeAction =
  | "BUY"
  | "SELL"
  | "T_BUY"
  | "T_SELL"
  | "ADD"
  | "REDUCE"
  | "STOP_LOSS"
  | "TAKE_PROFIT"
  | "RISK_EXIT"
  | "UNKNOWN"

type Trade = {
  id: string
  datetime: string
  symbol: string
  name: string | null
  action: TradeAction
  price: number
  quantity: number
  amount: number | null
  fee: number | null
  commission: number | null
  slippage: number | null
  tax: number | null
  transferFee: number | null
  impactCost: number | null
  positionAfter: number | null
  positionWeightAfter: number | null
  cashAfter: number | null
  signalSource: string | null
  signalId: string | null
  modelVersion: string | null
  modelScore: number | null
  factorContributions: Record<string, number> | null
  riskReason: string | null
  pnl: number | null
  cumulativePnl: number | null
  success: boolean | null
  failureReason: string | null
  status: string | null
  tPairId: string | null
  provenance: {
    sourcePath: string
    sourceRow?: number
    derived: string[]
  }
}
```

### 4.1 Strict backtest mapping

| Trade field | Runtime field |
|---|---|
| id | `client_order_id` |
| datetime | `trade_date` |
| symbol | `symbol` |
| action | `side` → BUY/SELL |
| price | `avg_price` when filled, else `reference_price` |
| quantity | `filled_quantity` when > 0, else requested `quantity` |
| amount | derived `price * quantity` |
| success | status is `filled` or `partial` |
| failureReason | `last_message` for rejected/cancelled |
| status | `status` |

Strict order audit 当前没有逐笔 commission/tax/slippage columns。若能通过完整 config + cost model 精确重算，标记为 derived；否则这些字段为 `null`。

### 4.2 Realized trade mapping

`realized_trades.csv` 是 FIFO closed round trip：

```text
symbol, buy_date, sell_date, quantity, buy_price, sell_price,
gross_pnl, cost, net_pnl
```

它用于：

- 单笔 realized PnL。
- buy/sell pair details。
- cumulative realized PnL。

它不包含 order id、signal reason、cash/position after。

### 4.3 Do-T mapping

Do-T overlay：

| API | Runtime |
|---|---|
| T_BUY datetime | `buy_time` or `entry_fill_time` for buy mode |
| T_SELL datetime | `sell_time` or `exit_fill_time` for sell mode |
| price | `buy_price/sell_price` or `entry_px/exit_px` |
| quantity | `quantity` or `filled_qty` |
| pnl | pair-level `net_pnl` or `net_ret` |
| success | positive net PnL plus fill status, when available |
| failureReason | fill reason / stop / restore state |

旧 daily-only Do-T artifact 没有成交价和数量时，adapter 不生成 fake legs。

## 5. Signal

```ts
type SignalType =
  | "BUY"
  | "SELL"
  | "T_BUY"
  | "T_SELL"
  | "HOLD"
  | "RISK_WARNING"
  | "UNKNOWN"

type Signal = {
  id: string
  datetime: string
  symbol: string
  type: SignalType
  price: number | null
  strength: number | null
  confidence: number | null
  source: string | null
  factors: Record<string, number> | null
  reason: string | null
  riskFlags: string[]
  actionRaw: string | null
  tPairId: string | null
}
```

Source priority：

1. Explicit intraday decision/EV action。
2. Explicit strategy signal artifact。
3. Filled order audit as execution signal。
4. Prediction score is not automatically converted into BUY/SELL without a recorded threshold/rule。

## 6. Backtest

```ts
type BacktestSummary = {
  id: string
  name: string | null
  strategyVersion: string | null
  modelVersion: string | null
  factorVersion: string | null
  horizon: string | null
  startDate: string | null
  endDate: string | null
  universeSize: number | null
  initialCash: number | null
  totalReturn: number | null
  annualReturn: number | null
  maxDrawdown: number | null
  sharpe: number | null
  calmar: number | null
  volatility: number | null
  winRate: number | null
  profitFactor: number | null
  turnover: number | null
  tradeCount: number | null
  fillCount: number | null
  tTradeCount: number | null
  tContribution: number | null
  totalCost: number | null
  status: DataStatus
  path: string
  tags: string[]
  capabilities: {
    equity: boolean
    trades: boolean
    researchEvents: boolean
    positions: boolean
    riskEvents: boolean
    doT: boolean
    tradeSchema: "order_blotter" | "research_event_table" | null
  }
}

type EquityPoint = {
  datetime: string
  nav: number
  dailyReturn: number | null
  drawdown: number | null
  benchmarkNav: number | null
  excessNav: number | null
}

type PositionPoint = {
  datetime: string
  symbol: string
  shares: number | null
  availableShares: number | null
  frozenShares: number | null
  weight: number | null
  marketValue: number | null
}
```

Metrics aliases：

```text
annualReturn <- annualized_return | annualised_return | annualized
maxDrawdown  <- max_drawdown | maxDD
tradeCount   <- n_trades | trade_count
fillCount    <- n_fills
```

Normalization 不改变值符号；drawdown 展示层可选择 absolute magnitude，但 API 保留 `drawdownConvention`。

`trades.csv` 只有在包含可验证的 `symbol + side + status + quantity + price` 字段时才映射为 `Trade`。
例如 board-chase touch records、Do-T daily summaries 等研究事件表仅保留实验索引能力，
`trades` API 返回 `unsupported_trade_schema`，不会生成 `UNKNOWN` 假交易。

## 7. Stock replay aggregate

```ts
type StockReplay = {
  backtestId: string
  symbol: string
  name: string | null
  bars: KlineBar[]
  trades: Trade[]
  signals: Signal[]
  positions: PositionPoint[]
  scoreSeries: {
    datetime: string
    modelScore: number | null
    factorScore: number | null
    riskScore: number | null
    doTStrength: number | null
  }[]
  equity: EquityPoint[]
  summary: {
    realizedPnl: number | null
    tradeCount: number
    winRate: number | null
    maxDrawdown: number | null
    firstTrade: string | null
    lastTrade: string | null
  }
  availability: Record<string, boolean>
}
```

`availability` 用于前端明确禁用没有数据的 subplot/column，而不是渲染空折线。

## 8. Do-T analysis

```ts
type DoTTradePair = {
  id: string
  symbol: string
  tradeDate: string
  mode: "SELL_HIGH_BUY_LOW" | "BUY_LOW_SELL_OLD_HIGH" | "UNKNOWN"
  buyTime: string | null
  sellTime: string | null
  buyPrice: number | null
  sellPrice: number | null
  quantity: number | null
  grossPnl: number | null
  cost: number | null
  netPnl: number | null
  edgePct: number | null
  state: string | null
  entryStatus: string | null
  exitStatus: string | null
  highSellFailed: boolean | null
  lowBuyFailed: boolean | null
  missedUpsidePct: number | null
  adverseMovePct: number | null
  drawdownContribution: number | null
  success: boolean | null
  issues: DataIssue[]
}

type DoTAnalysis = {
  sourceId: string
  symbol: string | null
  summary: {
    pairCount: number
    successRate: number | null
    failureRate: number | null
    highSellFailureRate: number | null
    lowBuyFailureRate: number | null
    totalNetPnl: number | null
    returnContribution: number | null
    drawdownContribution: number | null
    qualityScore: number | null
  }
  pairs: DoTTradePair[]
  byRegime: Record<string, Record<string, number>>
}
```

Failure definitions：

- 优先读取 explicit label/metric：
  - `sell_high_fail_new_high_rate`
  - `buy_low_fail_breakdown_rate`
  - `adverse_excursion_after_sell/buy`
  - `state=closed_stop`、restore/fill reasons。
- 只有 minute path 完整时才计算 missed upside/adverse move。
- 没有 future minute path 时返回 `null`。

## 9. Factor

```ts
type FactorDirection =
  | "HIGHER_BETTER"
  | "LOWER_BETTER"
  | "NON_LINEAR"
  | "UNKNOWN"

type Factor = {
  name: string
  displayName: string | null
  category: string | null
  description: string | null
  codeLocation: string | null
  formula: string | null
  direction: FactorDirection
  horizonDays: number | null
  parameters: Record<string, unknown>
  dataSource: string[]
  requiredColumns: string[]
  frequency: string | null
  lookback: number | null
  pitSafe: boolean | null
  missingValuePolicy: string | null
  standardization: string | null
  neutralization: string | null
  usedInTraining: boolean | null
  usedInSelection: boolean | null
  usedInTiming: boolean | null
  usedInRisk: boolean | null
  lifecycle: string | null
  sourceKind: "registry" | "alpha181" | "runtime" | "synthesized"
}

type FactorBacktest = {
  factorName: string
  totalReturn: number | null
  annualReturn: number | null
  maxDrawdown: number | null
  sharpe: number | null
  calmar: number | null
  winRate: number | null
  turnover: number | null
  ic: number | null
  rankIc: number | null
  icir: number | null
  rankIcir: number | null
  coverage: number | null
  stability: number | null
  crowding: number | null
  capacityRmb: number | null
  icSeries: { datetime: string; value: number | null }[]
  rankIcSeries: { datetime: string; value: number | null }[]
  quantileReturns: { datetime: string; quantile: string; value: number }[]
  longShortEquity: EquityPoint[]
  decay: { horizonDays: number; ic: number | null; rankIc: number | null }[]
  trades: Trade[]
  signals: Signal[]
  availability: Record<string, boolean>
}
```

Factor catalog priority：

1. `FactorRegistry` meta。
2. Alpha181 source map + implementation source。
3. synthesized definitions JSON。
4. runtime columns / feature schemas。

单因子 trades 只在 independent factor strategy artifact 存在时返回；不能用 multi-factor portfolio trades 冒充。

## 10. Selection

```ts
type SelectionRun = {
  id: string
  asOfDate: string | null
  candidateCount: number | null
  finalCount: number | null
  usedFallback: boolean | null
  noOrdersGenerated: boolean | null
  path: string
  status: DataStatus
}

type SelectionStock = {
  symbol: string
  name: string | null
  sector: string | null
  modelRank: number | null
  modelScore: number | null
  factorScore: number | null
  llmScore: number | null
  confidence: number | null
  riskScore: number | null
  doTSuitability: number | null
  finalScore: number | null
  finalRank: number | null
  actionBucket: string | null
  included: boolean
  exclusionReason: string | null
  factorContributions: Record<string, number>
}

type DecisionGate = {
  order: number
  name: string
  passed: boolean | null
  reason: string | null
  detail: Record<string, unknown>
}

type DecisionChain = {
  runId: string
  symbol: string
  datetime: string | null
  finalDecision: string | null
  failedGate: string | null
  gates: DecisionGate[]
}
```

若没有 persisted decision trace，可从 hybrid ranking 展示 score stages，但必须标记为 `score_pipeline`，不能伪装成逐 gate execution trace。

## 11. Model

```ts
type ModelSummary = {
  id: string
  modelType: string | null
  version: string | null
  featureVersion: string | null
  createdAt: string | null
  trainStart: string | null
  trainEnd: string | null
  testEnd: string | null
  horizons: number[]
  featureCount: number | null
  sampleCount: number | null
  device: string | null
  gpuName: string | null
  productionReady: boolean | null
  status: DataStatus
  path: string
  issues: DataIssue[]
}

type TrainingMetricPoint = {
  epoch: number
  loss: number | null
  validationLoss: number | null
  metrics: Record<string, number>
}

type FeatureImportance = {
  feature: string
  importance: number
  method: "native" | "permutation" | "shap" | "factor_weight" | "unknown"
}

type PredictionPoint = {
  datetime: string
  symbol: string
  score: number
  horizon: string | null
  actualReturn: number | null
  rank: number | null
}
```

Model discovery：

- Registry JSON。
- Deep run config + FT config/schema/metrics/checkpoint signature。
- Predictions artifact。
- RL summary/verdict。

Checkpoint 仅作为存在性与 size metadata，不通过 API 读取内容。

## 12. Risk

```ts
type RiskEvent = {
  id: string
  datetime: string | null
  symbol: string | null
  type: string
  severity: "info" | "warning" | "critical" | "unknown"
  reason: string | null
  rule: string | null
  blocked: boolean | null
  detail: Record<string, unknown>
  sourcePath: string
}

type RiskOverview = {
  maxDrawdown: number | null
  maxSingleStockLoss: number | null
  maxDailyLoss: number | null
  consecutiveLossDays: number | null
  concentration: number | null
  sectorConcentration: number | null
  volatilityExposure: number | null
  liquidityRisk: number | null
  limitDownRisk: number | null
  suspensionRisk: number | null
  doTFailureRisk: number | null
  eventCounts: Record<string, number>
  rules: RiskRule[]
}

type RiskRule = {
  id: string
  name: string
  description: string
  threshold: number | string | null
  enabled: boolean | null
  codeLocation: string | null
}
```

Risk values are nullable。只有可从 NAV/trades/positions/risk artifacts 可靠计算时才填充。

## 13. Jobs

```ts
type JobType = "backtest" | "train" | "infer"
type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled"

type JobRequest = {
  commandId: string
  parameters: Record<string, string | number | boolean | string[] | null>
}

type Job = {
  id: string
  type: JobType
  status: JobStatus
  commandId: string
  createdAt: string
  startedAt: string | null
  finishedAt: string | null
  progress: number | null
  message: string | null
  outputPaths: string[]
  error: string | null
}
```

Security：

- `commandId` 来自 server allowlist。
- 禁止接收 arbitrary shell command。
- 参数按 command-specific schema validate。
- training/backtest outputs 只能写入 project runtime。
- live trading/QMT enable flags 不在 allowlist。

## 14. Pagination and filtering

```ts
type Page<T> = {
  items: T[]
  total: number
  page: number
  pageSize: number
  hasNext: boolean
}
```

Default：

- tables：`pageSize=100`，maximum 1,000。
- chart points：default 2,000，maximum 10,000。
- logs：cursor pagination，单次 maximum 1,000 lines。
- risk events：cursor pagination，禁止一次返回超大 JSON。

## 15. API availability contract

所有任务要求中的 endpoint 都会注册。尚无真实数据源的 endpoint：

- 返回 HTTP 200 + `status="empty"` 或 `partial`。
- `issues` 说明缺少哪个 artifact。
- 不返回 fabricated example data。
- 不因 optional file 缺失返回 unhandled 500。

## 16. Model observability

```ts
type ModelSummary = {
  id: string
  modelType: string | null
  modelFamily:
    | "deep_alpha"
    | "registered_alpha"
    | "reinforcement_learning"
    | "intraday_t_plus_one"
    | "generic_artifact"
    | null
  version: string | null
  sourceKind: string | null
  verdict: string | null
  status: DataStatus
  path: string
  horizons: number[]
  featureCount: number | null
  sampleCount: number | null
  device: string | null
  productionReady: boolean | null
  capabilities: Record<string, boolean>
  issues: DataIssue[]
}

type ModelObservability = ModelSummary & {
  metrics: Array<{
    key: string
    label: string
    value: number
    source: string
    group: "return" | "risk" | "quality" | "scale" | "other"
    unit: "ratio" | "count" | "bps" | "number"
  }>
  artifacts: Array<{
    role: "checkpoint" | "evaluation" | "config" | "prediction" | "artifact"
    name: string
    path: string
    extension: string
    sizeBytes: number
    modifiedAt: string
    previewable: boolean
  }>
  evaluations: Array<{ name: string; path: string; data: Record<string, unknown> }>
  availability: Record<string, boolean>
  checkpoint: {
    contentExposed: false
    count: number
    sizeBytes: number
  }
}
```

Rules：

- Binary checkpoint 不反序列化、不返回内容，只返回 repository-relative metadata。
- Model comparison 最多 6 个模型，优先排列 total return、annual return、Sharpe、Calmar、win rate、max drawdown、turnover。
- Missing feature importance / SHAP / predictions 不会阻断 metrics、evaluation 或 artifact inventory。

## 17. Runtime cleanup

```ts
type CleanupCandidate = {
  id: string
  category: string
  label: string
  reason: string
  paths: string[]
  sizeBytes: number
  itemCount: number
  modifiedAt: string | null
  safeDefault: boolean
  requiresExplicit: boolean
}

type RuntimeCleanupAnalysis = {
  runtimeSizeBytes: number
  candidateSizeBytes: number
  safeDefaultSizeBytes: number
  candidates: CleanupCandidate[]
  protected: string[]
}
```

Execution rules：

- Client 只能提交 backend 当前 analysis 返回的 candidate ID。
- Confirmation 必须严格等于 `DELETE`。
- 每次执行写入 `runtime/reports/quant_ui/cleanup/cleanup_*.json`。
- raw、silver、manifests、canonical registry 等 protected roots 永远拒绝删除。
