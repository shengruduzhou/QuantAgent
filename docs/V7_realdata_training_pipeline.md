# V7 Real-Data Training Pipeline / 真实数据训练流程

V7 real-data path 以 Qlib CN market data、AkShare/TuShare PIT financial data、valuation snapshots、EvidenceStore 和 factor DSL 构建 gold training dataset，再通过 purged / embargo walk-forward OOS training 产出 metrics、predictions、target_weights、paper/backtest report。默认不启用 live trading，不使用 synthetic fallback。

## Storage / 存储布局

默认 Windows root：

```text
E:\Project\QuantAgent\runtime\
  data\raw\qlib\cn_data\
  data\v7\raw\akshare\fundamentals\
  data\v7\silver\market_panel\
  data\v7\silver\fundamentals\
  data\v7\silver\valuation\
  data\v7\silver\factors\
  data\v7\gold\training_dataset\
  data\v7\manifests\
  models\v7_alpha\
  predictions\
  target_weights\
  reports\v7\
  logs\
```

`QUANTAGENT_HOME` 覆盖全局 root，`QUANTAGENT_DATA_ROOT` 只覆盖 data tier。代码应通过 `quantagent.config.paths.quant_paths()` 解析路径。

## Qlib / AkShare 对齐规则

- Qlib 官方自动流程是 `qrun`：从 dataset、model training、backtest 到 evaluation 一次运行；V7 对应入口是 `auto-train-v7`，但保留 A 股安全边界，只输出 `target_weights` 和 paper/backtest report。
- Qlib Recorder / record templates 对应 V7 的 `experiment_manifest.json`、`metrics.json`、`walk_forward_predictions.csv`、`acceptance_report.json`。
- Qlib OnlineManager / Updater 是持续更新思想来源；V7 的持续训练应反复运行 `auto-train-v7`，但不打开 live trading。
- Qlib CN symbol 形态是 `SH600519` / `SZ000001`；QuantAgent lake 内部统一为 AkShare/TuShare 兼容的 `600519.SH` / `000001.SZ`。CLI 现在会在 Qlib adapter 内自动转换。
- 本机 Qlib CN dump 的 calendar 决定 Qlib 训练覆盖期。当前官方 free dump 常见尾部是 `2020-09-25`，之后的行情补齐走 AkShare market panel。
- AkShare market panel 使用 `stock_zh_a_hist`；valuation 使用 `stock_zh_a_spot_em`；financial statements 使用 `stock_financial_report_sina`、`stock_financial_analysis_indicator`、`stock_history_dividend_detail`；sector 必须使用 per-board membership endpoint 或 local mapping，禁止 cross-join。
- AkShare 官方 FAQ 明确部分接口不提供 `start_date/end_date`，调用后自行过滤；V7 provider 必须保留这个行为，不能伪造缺失区间。

## Auto Date Range / 自动日期范围

`build-akshare-market-panel-v7` 的 `--start-date` 和 `--end-date` 可以省略：

- `--end-date` 缺省时使用 `--as-of-date` 对应的最近 business day；如果 `--as-of-date 2026-05-17` 是周日，则解析为 `2026-05-15`。
- `--start-date` 缺省时优先读取 `provider_uri/calendars/day.txt`，从 Qlib calendar 最后一日的下一个 business day 开始，例如 `2020-09-25 -> 2020-09-28`。
- 如果没有 Qlib calendar，则读取已有 `data\v7\manifests\market_panel.json` 的有效 `end_date` 后续接。
- 如果两者都没有，才使用 five-year fallback，并在 manifest warnings/notes 中记录。

```powershell
.\.venv\Scripts\quantagent.exe build-akshare-market-panel-v7 `
  --symbols 600519.SH,600036.SH,000001.SZ `
  --provider-uri-for-range E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data `
  --as-of-date 2026-05-17 `
  --allow-network
```

## End-to-End Auto Training / 端到端自动训练

`auto-train-v7` 会按已有数据自动选择训练路径：

1. `--symbols auto` 时从 local Qlib `features/` 目录发现股票池，内部统一为 `600519.SH` 风格。
2. 如果已有 `market_panel` manifest 可用，直接使用。
3. 如果 market panel 不可用但 Qlib provider 可用，按 Qlib calendar 全覆盖构建 silver market panel。
4. 如果传入 `--refresh-akshare-market --allow-network`，用 AkShare 补齐 Qlib 后的 recent market panel。
5. 自动生成 labels。
6. 调用 `run-full-real-training-v7`：dataset -> train -> validation-only predictions -> target_weights -> paper/backtest report。

完整真实数据训练示例：

```powershell
.\.venv\Scripts\quantagent.exe auto-train-v7 `
  --symbols auto `
  --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data `
  --model ridge `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5 `
  --min-rows 1000 `
  --min-train-rows 1000 `
  --initial-cash 1000000
```

补齐 Qlib 之后的 AkShare recent market 数据并训练：

```powershell
.\.venv\Scripts\quantagent.exe auto-train-v7 `
  --symbols auto `
  --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data `
  --refresh-akshare-market `
  --allow-network `
  --as-of-date 2026-05-17 `
  --model ridge `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5
```

若要限制初期规模，可加 `--max-symbols 300`；生产级大样本可使用默认 `--max-symbols 0`，表示不限制从 Qlib features 发现的股票数量。

## GPU Training Gate / GPU 训练门槛

`ft_transformer` 是当前 V7 的 PyTorch GPU-capable training path。为了避免“看起来训练了但实际退回 CPU”，CLI 提供硬门槛：

```powershell
.\.venv\Scripts\quantagent.exe auto-train-v7 `
  --symbols auto `
  --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data `
  --model ft_transformer `
  --ft-device cuda `
  --require-gpu `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5
```

- `--ft-device cuda` 明确要求 CUDA device；`--require-gpu` 会在 `torch.cuda.is_available() == False` 时 fail-loud。
- `metrics.json` 会记录 `training_device`、`cuda_available`、`gpu_name`、`gpu_required`，用于审计是否真正进入 GPU training。
- 当前实现不把 `ridge`、`elastic_net`、`lightgbm`、`xgboost` 伪装成 GPU training；如果要用 AMD GPU，需要另接 DirectML / ROCm backend，并在 metrics 中同样记录 device provenance。

## Manual Command Chain / 手工分步命令

```powershell
.\.venv\Scripts\quantagent.exe build-labels-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet

.\.venv\Scripts\quantagent.exe materialize-factors-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --backend polars

.\.venv\Scripts\quantagent.exe build-training-dataset-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --labels E:\Project\QuantAgent\runtime\data\v7\labels.parquet `
  --fundamentals-root E:\Project\QuantAgent\runtime\data\v7\raw\akshare\fundamentals `
  --valuation E:\Project\QuantAgent\runtime\data\v7\silver\valuation\valuation.parquet

.\.venv\Scripts\quantagent.exe run-full-real-training-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --labels E:\Project\QuantAgent\runtime\data\v7\labels.parquet `
  --model ridge `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5 `
  --optimizer-backend auto `
  --objective max_expected_alpha `
  --initial-cash 1000000 `
  --paper-report-output-dir E:\Project\QuantAgent\runtime\reports\v7\paper_report
```

## Paper Report Outputs / 报告输出

`run-full-real-training-v7`、`run-paper-backtest-v7`、`generate-paper-report-v7` 写出：

- `selected_stocks.csv`
- `target_weights.parquet` 或 CSV fallback
- `trades.csv`
- `failed_orders.csv`
- `holdings.csv`
- `pnl.csv`
- `paper_report.json`
- `paper_report.md`
- `paper_report.html`

报告只描述 OOS validation predictions 产生的 `target_weights` 和 paper/backtest 结果，不构成 financial advice，不承诺收益。

## Safety Boundary / 安全边界

- LLM / Agent 不能生成 orders，只能输出 evidence、views、constraints、confidence、risk flags、audit logs。
- Optimizer 只能生成 `target_weights`。
- 只有 `OrderManager` 能把 target weights 转换为 order intents。
- `auto-train-v7` 不启用 live trading；QMT 仍必须通过 risk gate、kill switch、execution simulation、reconciliation、audit replay。
- Production mode 不允许 synthetic fallback；mock data 只允许 tests 和 smoke examples。
