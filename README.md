# QuantAgent V7 / A-share PIT Research System

QuantAgent V7 是面向 A 股散户现实约束的 Point-in-Time quant research system。它不提供 financial advice，不承诺收益，不默认连接真实券商；目标是在真实数据、成本、slippage、T+1、涨跌停、停牌、ST、流动性、换手、回撤和 kill-switch 约束下，研究 out-of-sample net risk-adjusted return。

## 安全边界 / Safety Boundary

- 默认 `live_trading_enabled=false`、`dry_run=true`、`virtual_broker_only=true`。
- Agent 只能输出 structured evidence、views、constraints、confidence、risk flags、audit logs。
- Optimizer / Portfolio Construction 只能输出 `target_weights`。
- 只有 `OrderManager` 可以把 `target_weights` 转成 order intents。
- 任何 QMT submit path 前必须通过 Risk Gate、Kill Switch、execution constraint simulation、reconciliation 和 audit replay。
- Production / realdata 模式不允许 synthetic fallback；mock data 只用于 tests 和 smoke examples。

## 存储布局 / Storage Layout

所有大数据 (raw 行情 dump、silver/gold dataset、模型 checkpoint、predictions、target weights、回测报告、日志) 默认写入 **仓库外** 的统一目录。Windows 默认根目录是 `E:\AI量化\`，POSIX 是 `~/AI_quant`；可通过 `QUANTAGENT_HOME` 环境变量覆盖。

```
E:\AI量化\
  data\raw\{qlib,akshare,tushare,disclosures}\
  data\silver\{market_panel,fundamentals,valuation,disclosures}\
  data\gold\training_dataset\
  models\v7_alpha\
  predictions\
  target_weights\
  reports\v7\
  logs\v7\
  cache\
```

CLI 命令的所有 `--output` / `--output-dir` 参数在未显式传入时都会落到上面的对应子目录。`src/quantagent/config/paths.py:quant_paths` 是单一来源；`src/quantagent/data/lake.py` 在它之上派生 V7 medallion lake 布局。

```powershell
quantagent storage-info-v7 --ensure           # 显示并创建当前 home 下所有目录
$env:QUANTAGENT_HOME = "D:\quant_storage"     # 一行覆盖全局存储根
```

## 真实数据流程 / Real-Data Flow

```text
download / prepare Qlib CN data (setup-qlib-v7)
-> build-market-panel-v7              # qlib → silver/market_panel + manifest
-> build-akshare-v7 or build-fundamentals-v7   # akshare/tushare → silver/fundamentals + manifest
-> build-valuation-v7                 # akshare snapshot or local CSV → silver/valuation + manifest
-> build-labels-v7                    # multi-horizon forward returns
-> build-training-dataset-v7          # PIT as-of join → gold/training_dataset + manifest + feature schema
-> train-alpha-v7 / train-deep-alpha-v7   # purged WF CV + experiment manifest + model registry
-> optimize-alpha-v7                  # grid / random search over hyperparameters (optional)
-> predict-alpha-v7                   # forward-pass against feature dataset → predictions
-> build-target-weights-v7            # alpha → constrained target_weights + diagnostics
-> walk-forward-backtest-v7           # OrderManager + VirtualBroker dry-run
                                      # accepts --target-weights OR --predictions
-> paper-trade-v7                     # 同一条 dry-run 路径
-> v7-live-readiness-report           # 输出 readiness gate 报告（不会开启实盘）
```

`run-full-real-training-v7` 串联了 dataset → train → predict → target_weights → walk-forward backtest 全流程，写一份 `full_pipeline_report.json`。

Qlib 负责 market OHLCV、technical features、labels、backtest base。TuShare / AkShare 负责 financial statements、financial indicators、valuation fields 和 disclosure dates，必须保存 `report_period / ann_date / available_at`。TradingView public pages 仅作为 sentiment / attention context。

## 安装 / Install

```powershell
pip install -e .                 # core
pip install -e ".[data]"         # akshare / tushare / pyarrow
pip install -e ".[research]"     # pyqlib / vectorbt
pip install -e ".[training]"     # torch / transformers / scikit-learn / pyarrow
pip install -e ".[optimization]" # cvxpy
```

Optional extras 缺失时仍可 import 核心包；真实数据 CLI 会报错给出可执行的安装/配置提示。

## CLI / 命令

存储/Qlib 准备：

```powershell
quantagent storage-info-v7 --ensure
quantagent setup-qlib-v7 --region cn                 # 仅打印官方下载命令
quantagent setup-qlib-v7 --region cn --run --allow-community-fallback   # 若 pyqlib 已装则直接下载
quantagent download-qlib-v7 --target-dir E:\AI量化\data\raw\qlib\cn_data --region cn
quantagent check-qlib-v7 --provider-uri E:\AI量化\data\raw\qlib\cn_data --symbols 600519.SH
```

数据建表：

```powershell
quantagent build-market-panel-v7 --provider-uri E:\AI量化\data\raw\qlib\cn_data --symbols 600519.SH --start-date 2020-01-01 --end-date 2026-05-15
quantagent build-akshare-v7 --symbols 600519.SH,000858.SZ --start-date 2020-01-01 --end-date 2026-05-15 --allow-network
quantagent build-fundamentals-v7 --symbols 600519.SH --start-date 2020-01-01 --end-date 2026-05-15 --provider tushare --allow-network
quantagent build-valuation-v7 --as-of-dates 2026-05-15 --symbols 600519.SH --allow-network
quantagent build-labels-v7 --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet
quantagent build-training-dataset-v7 --market-panel ... --labels ... --fundamentals-root ... --valuation ...
```

训练 / 预测 / 组合 / 回测：

```powershell
quantagent train-alpha-v7 --dataset ... --model ridge
quantagent train-alpha-v7 --dataset ... --model lightgbm --allow-model-downgrade   # real LGBM, ridge fallback only with the flag
quantagent train-deep-alpha-v7 --dataset ... --horizons 1,5,20,60,120,126
quantagent optimize-alpha-v7 --dataset ... --search-space configs/example_search_space.json --sampler grid
quantagent predict-alpha-v7 --model-dir ... --feature-dataset ...
quantagent build-target-weights-v7 --predictions ... --market-panel ... --sector-map ...
quantagent run-real-training-v7 --market-panel ... --labels ... --fundamentals-root ...
quantagent run-full-real-training-v7 --market-panel ... --labels ... --sector-map ...   # dataset → train → predict → target_weights → backtest
quantagent walk-forward-backtest-v7 --target-weights ... --market-panel ...
quantagent walk-forward-backtest-v7 --predictions ... --market-panel ... --sector-map ...   # optimiser runs first
quantagent paper-trade-v7 --target-weights ... --market-panel ...
quantagent v7-live-readiness-report --metrics ... --paper-report ...
```

CLI 已经拆分为 `cli/v7_data.py / v7_train.py / v7_backtest.py / v7_readiness.py / v7_research.py / v7_storage.py / v7_optimize.py`，共享 `cli/_utils.py`。

## 新增模块 / New Modules

- `quantagent.config.paths`：统一 `E:\AI量化\` 存储布局，环境变量 `QUANTAGENT_HOME` / `QUANTAGENT_DATA_ROOT` 覆盖。
- `quantagent.training.splitters`：expanding / rolling / purged / chronological 走式 walk-forward 切分。
- `quantagent.training.optimize`：alpha 超参 grid / random search，结果写入 `reports/v7/optimization/`。
- `quantagent.factors.expr`：Alpha101-style 符号化因子 DSL (`Rank(TsMean(Close, 5))`)，零 lookahead 测试覆盖。
- `quantagent.models.ft_transformer`：FT-Transformer 表格架构（PyTorch 可选）。
- `quantagent.training.ft_transformer_trainer`：FT-Transformer trainer，带 AMP / checkpoint resume / 时序 validation 切分。

## 关键配置 / Configs

- `configs/v7.default.yaml`：默认 strict_local research config。
- `configs/v7.mock.yaml`：tests 和 smoke examples 使用，不能作为 production-ready 依据。
- `configs/v7.realdata.yaml`：真实数据 data quality gates 和 execution constraints。
- `configs/v7.train.yaml`：Ridge / ElasticNet training 和 acceptance gates。
- `configs/v7.paper.yaml`：VirtualBroker paper trading 和 live-readiness prerequisites。

## 验证 / Validation

```powershell
python -m pytest tests/
python -m compileall -q src
git diff --check
```

`git diff --check` 在 Windows checkout 中可能报告 CRLF warning；真正 whitespace error 必须修复。

## 文档 / Docs

- [V7 系统架构与 Agent 接口](docs/V7_系统架构与Agent接口.md)
- [V7 证据摄取与交易规则](docs/V7_证据摄取与交易规则.md)
- [V7 Real-Data Training Pipeline](docs/V7_realdata_training_pipeline.md)
- [V7 PIT Data Contract](docs/V7_PIT_data_contract.md)
- [V7 Training Dataset Schema](docs/V7_training_dataset_schema.md)
- [V7 Live Readiness Gates](docs/V7_live_readiness_gates.md)

## 已知限制 / Known Limitations

- 不提供任何收益保证。
- 默认禁止实盘交易；任何 production toggle 必须经独立人工 sign-off。
- Qlib CN 数据需要用户自行准备 provider_uri；TuShare 需要 token；AkShare 网络抓取需要 `--allow-network`。
- LightGBM / XGBoost 在 `train-alpha-v7 --model lightgbm`（或 `xgboost`）下是 **真实** 训练；未安装时默认 fail-loud，需显式 `--allow-model-downgrade` 才退回 ridge。
- `train-deep-alpha-v7` 在 PyTorch 缺失时回退到 numpy ridge head，仍写出可 round-trip 的 state；预测能力会下降。
- `FTTransformerTrainer` 需要 PyTorch；未安装时构造抛 `ImportError`。
- `build-valuation-v7` 默认拉取 AkShare 当日 snapshot；离线环境需用 `--csv-snapshot` 提供历史快照。
- `AkShareSectorProvider` 不再 cross-join 行业到所有 symbol；离线/未提供 `--local-mapping` 时直接 fail-loud。
- `pit_wide_merge_statements` 在做 PIT wide-merge 之前会按 statement 类型加列前缀，并拒绝任何 `(symbol, report_period, available_at)` 重复。
- `evaluate_adverse_regime` 真实计算 bottom-quartile-day rank-IC；不再硬编码 `adverse_regime_passed=True`。
- 业务日历 `TradingCalendar` 从 silver market panel 派生；若未生成 panel，PIT 解析会退回 calendar-day fallback 并写 warning。
