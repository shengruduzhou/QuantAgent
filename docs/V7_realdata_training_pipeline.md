# V7 Real-Data Training Pipeline / V7 真实数据训练流程

## 目标 / Goal

V7 把 Qlib CN + AkShare / TuShare 财务数据组成 PIT-safe training stack，可以从下载到 paper-trading readiness 一条命令链跑完。所有命令默认不开启 live trading，并且禁止 synthetic fallback。

## 数据湖布局 / Data Lake Layout

```
data/v7/
  raw/qlib/            # qlib provider_uri 原始 dump（用户准备）
  raw/akshare/         # AkShare 抓取的 raw 缓存（如需）
  raw/tushare/         # TuShare raw 缓存（如需）
  raw/disclosures/     # 公告/披露原文
  silver/market_panel/ # qlib 归一化后的 PIT market panel
  silver/fundamentals/ # 财务三大表的 PIT cache
  silver/valuation/    # 估值字段
  silver/disclosures/  # 披露日期/公告 metadata
  gold/training_dataset/  # 模型训练数据集
  manifests/           # 每个 dataset 一份 JSON manifest
```

`src/quantagent/data/lake.py:v7_lake_paths` 是这套布局的单一来源；
所有 bootstrap、dataset builder、CLI 都从它读取目录约定。

## 命令链 / Command Chain

```powershell
# 1. 准备 Qlib CN 数据（首次）
quantagent download-qlib-v7 --target-dir ~/.qlib/qlib_data/cn_data --region cn
# Qlib 官方命令：python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

quantagent check-qlib-v7 --provider-uri ~/.qlib/qlib_data/cn_data --symbols 600519.SH --start-date 2020-01-01 --end-date 2026-05-15

# 2. 导出 PIT 市场面板 + close-next-day 技术特征 + manifest
quantagent build-market-panel-v7 --provider-uri ~/.qlib/qlib_data/cn_data \
  --symbols 600519.SH,000858.SZ --start-date 2020-01-01 --end-date 2026-05-15

# 3. 拉 AkShare 财务三大表到 silver fundamentals + manifest
quantagent build-akshare-v7 --symbols 600519.SH,000858.SZ \
  --start-date 2020-01-01 --end-date 2026-05-15 --allow-network

# 4. 生成多 horizon 标签
quantagent build-labels-v7 --market-panel data/v7/silver/market_panel/market_panel.parquet \
  --output data/v7/labels.parquet --horizons 1,5,20,60,120,126

# 5. As-of join 成 gold 训练集（PIT 安全 + manifest + feature schema）
quantagent build-training-dataset-v7 \
  --market-panel data/v7/silver/market_panel/market_panel.parquet \
  --labels data/v7/labels.parquet \
  --fundamentals-root data/v7/silver/fundamentals \
  --output data/v7/gold/training_dataset/training_dataset.parquet \
  --horizons 1,5,20,60,120,126

# 6. 训练 + walk-forward + 输出 metrics / experiment manifest / registry
quantagent train-alpha-v7 \
  --dataset data/v7/gold/training_dataset/training_dataset.parquet \
  --output-dir artifacts/v7_alpha --model ridge

# 7. 模型 inference → 写 wide alpha frame + sidecar JSON summary
quantagent predict-alpha-v7 \
  --model-dir artifacts/v7_alpha \
  --feature-dataset data/v7/gold/training_dataset/training_dataset.parquet \
  --output artifacts/v7_alpha/predictions/predictions.parquet

# 8. 把 alpha 转成受约束的 target weights（ST/停牌/涨跌停过滤 + 行业/单票/换手上限）
quantagent build-target-weights-v7 \
  --predictions artifacts/v7_alpha/predictions/predictions.parquet \
  --market-panel data/v7/silver/market_panel/market_panel.parquet \
  --sector-map data/v7/silver/sector/sector_map.csv \
  --output artifacts/v7_alpha/target_weights/target_weights.parquet

# 9. 走 OrderManager → VirtualBroker dry-run 回测/纸面交易
quantagent walk-forward-backtest-v7 \
  --target-weights artifacts/v7_alpha/target_weights/target_weights.parquet \
  --market-panel data/v7/silver/market_panel/market_panel.parquet
# 或者直接用 predictions（CLI 会先调用 target-weights 优化器）：
quantagent walk-forward-backtest-v7 \
  --predictions artifacts/v7_alpha/predictions/predictions.parquet \
  --market-panel data/v7/silver/market_panel/market_panel.parquet \
  --sector-map data/v7/silver/sector/sector_map.csv
quantagent paper-trade-v7 \
  --target-weights artifacts/v7_alpha/target_weights/target_weights.parquet \
  --market-panel data/v7/silver/market_panel/market_panel.parquet

# 10. live-readiness gate（不会开启实盘，只是报告）
quantagent v7-live-readiness-report \
  --metrics artifacts/v7_alpha/metrics.json \
  --paper-report reports/v7/paper_trade_report.json
```

可选 / 一键串联：

```powershell
quantagent run-real-training-v7 --market-panel ... --labels ... --fundamentals-root ...
quantagent run-full-real-training-v7 --market-panel ... --labels ... --sector-map ...   # dataset → train → predict → target_weights → backtest
quantagent evaluate-alpha-v7 --metrics artifacts/v7_alpha/metrics.json --paper-report reports/v7/paper_trade_report.json
```

## 大规模训练 / Large-Scale Training

- Baseline：Ridge（默认）、ElasticNet。
- 真实 tree 模型：`--model lightgbm` / `--model xgboost` 调用真正的 LightGBM / XGBoost 实现，并把每个 horizon 的 booster 序列化到 `boosters/horizon_<h>.<backend>.txt`。未安装 extras 时默认 **fail-loud**；只有显式 `--allow-model-downgrade` 才会降级到 ridge，manifest 同时写下 `model_requested` / `backend` / `model_downgraded`。
- 深度模型：`quantagent train-deep-alpha-v7` —— 支持 fit / predict / save / load / 检查点 / early stopping / CPU+单卡。Huber 损失 + cross-sectional rank loss + 可选 long-short utility loss。无 PyTorch 时回退 numpy ridge head。
- 全部模型走 purged walk-forward CV（`quantagent.quant_math.purged_cv`）+ embargo + multi-horizon training。
- 训练 artifact 写入 `artifacts/v7_alpha/`：
  - `model_coefficients.json`、`metrics.json`、`feature_schema.json`、`label_schema.json`、`training_config.json`
  - `data_quality_report.json`、`acceptance_report.json`、`walk_forward_predictions.csv`
  - `experiment_manifest.json`（experiment name、horizons、git commit、fold count、production_ready、backend、model_downgraded、adverse_regime_report）
  - `boosters/horizon_<h>.<backend>.txt`（LightGBM/XGBoost 原生模型文件）
  - `predictions/predictions.parquet` + `predictions.summary.json`（`predict-alpha-v7` 写入）
  - `target_weights/target_weights.parquet` + `target_weights.diagnostics.json`（`build-target-weights-v7` 写入）
  - `deep/deep_alpha_state.json` + `deep/deep_alpha_config.json` + `deep/deep_alpha_feature_schema.json` + `deep/deep_alpha_metrics.json` + `deep/deep_alpha_experiment_manifest.json`（深度模型 round-trip 状态）
- `artifacts/v7_alpha/registry/<experiment>.json` + `latest.json`（`ModelRegistry`）。
- 评估指标在 `quantagent.training.metrics` 中统一：IC、rank IC、ICIR、top-minus-bottom spread、Sharpe、Sortino、max drawdown、hit rate、capacity proxy。`compose_alpha_metrics` 给出一组完整结果，可直接写入 `metrics.json`。

## 安全 / Safety

- `live_trading_enabled=false`、`dry_run=true`、`virtual_broker_only=true` 永远是默认。
- AkShare/TuShare network 必须 `--allow-network` 显式开启。
- `build-training-dataset-v7` 拒绝 `allow_synthetic_fallback=true`，PIT 违反会被 quality gate 阻断。
- `AkShareSectorProvider` 离线时直接 `ProviderUnavailable`，绝不 cross-join 行业到所有 symbol。
- `pit_wide_merge_statements` 对每个 statement 做 prefix 化，重复 `(symbol, report_period, available_at)` 会 raise。
- `evaluate_adverse_regime` 真实计算 bottom-quartile 交易日的 rank-IC；不再硬编码 `adverse_regime_passed=True`。
- Production-ready 标记需通过 `evaluate_model_acceptance_gates` 中所有 gate：rank IC、stability、turnover-adjusted net return、drawdown、adverse regime（真实计算）、paper report、非 mock。
