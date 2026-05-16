# V7 Real-Data Training Pipeline / V7 真实数据训练流程

## 目标 / Goal

V7 把 Qlib CN + AkShare / TuShare 财务数据组成 PIT-safe training stack，可以从下载到 paper-trading readiness 一条命令链跑完。所有命令默认不开启 live trading，并且禁止 synthetic fallback。所有大数据 / 模型 / 报告默认写入仓库外的 `E:\AI量化\`（Windows）/ `~/AI_quant`（POSIX），通过 `QUANTAGENT_HOME` 环境变量覆盖。

## 存储布局 / Storage Layout

```
E:\AI量化\
  data\
    raw\{qlib,akshare,tushare,disclosures}\
    silver\{market_panel,fundamentals,valuation,disclosures}\
    gold\training_dataset\
    v7\manifests\
  models\v7_alpha\
  predictions\
  target_weights\
  reports\v7\
  logs\v7\
  cache\
```

`src/quantagent/config/paths.py:quant_paths` 是布局单一来源；`src/quantagent/data/lake.py:v7_lake_paths` 在它上面派生 V7 medallion lake 子层级。所有 bootstrap、dataset builder、CLI 都从它读取目录约定。

环境变量覆盖：

```powershell
$env:QUANTAGENT_HOME = "D:\quant_storage"     # 覆盖所有存储根
$env:QUANTAGENT_DATA_ROOT = "F:\quant_data"   # 单独覆盖数据 root
quantagent storage-info-v7 --ensure           # 显示并创建当前 layout
```

CLI 命令的 `--output` / `--output-dir` 不显式传入时都会落到对应子目录。

## Qlib 准备 / Qlib Setup

```powershell
# 推荐：通过 setup-qlib-v7 在受控目录下下载（要求 pip install -e .[research]）
quantagent setup-qlib-v7 --region cn --run --allow-community-fallback
# 或者完全 manual，等价于原生 Qlib 官方命令
python scripts/get_data.py qlib_data --target_dir E:\AI量化\data\raw\qlib\cn_data --region cn
# 健康检查 + PIT schema 探针
quantagent check-qlib-v7 --provider-uri E:\AI量化\data\raw\qlib\cn_data --symbols 600519.SH --start-date 2024-01-01 --end-date 2026-05-15
```

`setup-qlib-v7` 默认 dry-run（只打印官方命令）；加 `--run` 调用 `qlib.tests.data.GetData`；加 `--allow-community-fallback` 在失败时打印 Qlib release tarball / `scripts/dump_bin.py` 的备用路径。任何失败都 fail-loud 并 `Exit(1)`。

## 命令链 / Command Chain

```powershell
# 2. 导出 PIT 市场面板 + close-next-day 技术特征 + manifest
quantagent build-market-panel-v7 --provider-uri E:\AI量化\data\raw\qlib\cn_data \
  --symbols 600519.SH,000858.SZ --start-date 2020-01-01 --end-date 2026-05-15

# 3. 拉 AkShare 财务三大表到 silver fundamentals + manifest
quantagent build-akshare-v7 --symbols 600519.SH,000858.SZ \
  --start-date 2020-01-01 --end-date 2026-05-15 --allow-network

# 4. 生成多 horizon 标签
quantagent build-labels-v7 --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet \
  --horizons 1,5,20,60,120,126

# 5. As-of join 成 gold 训练集（PIT 安全 + manifest + feature schema）
quantagent build-training-dataset-v7 \
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet \
  --labels E:\AI量化\data\v7\labels.parquet \
  --fundamentals-root E:\AI量化\data\v7\silver\fundamentals \
  --horizons 1,5,20,60,120,126

# 6. 训练 + walk-forward + 输出 metrics / experiment manifest / registry
quantagent train-alpha-v7 \
  --dataset E:\AI量化\data\v7\gold\training_dataset\training_dataset.parquet \
  --model ridge

# 6b. 可选：超参 grid / random search
quantagent optimize-alpha-v7 \
  --dataset E:\AI量化\data\v7\gold\training_dataset\training_dataset.parquet \
  --search-space configs/example_search_space.json \
  --sampler grid --objective rank_ic_mean --mode max

# 7. 模型 inference → 写 wide alpha frame + sidecar JSON summary
quantagent predict-alpha-v7 \
  --model-dir E:\AI量化\models\v7_alpha \
  --feature-dataset E:\AI量化\data\v7\gold\training_dataset\training_dataset.parquet

# 8. 把 alpha 转成受约束的 target weights（ST/停牌/涨跌停过滤 + 行业/单票/换手上限）
quantagent build-target-weights-v7 \
  --predictions E:\AI量化\models\v7_alpha\predictions\predictions.parquet \
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet \
  --sector-map E:\AI量化\data\v7\silver\sector\sector_map.csv

# 9. 走 OrderManager → VirtualBroker dry-run 回测/纸面交易
quantagent walk-forward-backtest-v7 \
  --target-weights E:\AI量化\models\v7_alpha\target_weights\target_weights.parquet \
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet
quantagent walk-forward-backtest-v7 \
  --predictions E:\AI量化\models\v7_alpha\predictions\predictions.parquet \
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet \
  --sector-map E:\AI量化\data\v7\silver\sector\sector_map.csv
quantagent paper-trade-v7 \
  --target-weights E:\AI量化\models\v7_alpha\target_weights\target_weights.parquet \
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet

# 10. live-readiness gate（不会开启实盘，只是报告）
quantagent v7-live-readiness-report \
  --metrics E:\AI量化\models\v7_alpha\metrics.json \
  --paper-report E:\AI量化\reports\v7\paper_trade_report.json
```

可选 / 一键串联：

```powershell
quantagent run-real-training-v7 --market-panel ... --labels ... --fundamentals-root ...
quantagent run-full-real-training-v7 --market-panel ... --labels ... --sector-map ...   # dataset → train → predict → target_weights → backtest
quantagent evaluate-alpha-v7 --metrics ... --paper-report ...
```

## 大规模训练 / Large-Scale Training

- Baseline：Ridge（默认）、ElasticNet。
- 真实 tree 模型：`--model lightgbm` / `--model xgboost` 调用真正的 LightGBM / XGBoost 实现，并把每个 horizon 的 booster 序列化到 `boosters/horizon_<h>.<backend>.txt`。未安装 extras 时默认 **fail-loud**；只有显式 `--allow-model-downgrade` 才会降级到 ridge，manifest 同时写下 `model_requested` / `backend` / `model_downgraded`。
- 深度模型：
  - `quantagent train-deep-alpha-v7` —— 多 horizon MLP，Huber + cross-sectional rank loss + 可选 long-short utility loss，支持 fit / predict / save / load / 检查点 / early stopping / CPU+单卡。无 PyTorch 时回退 numpy ridge head。
  - `quantagent.training.ft_transformer_trainer.FTTransformerTrainer` —— FT-Transformer 表格架构（per-feature 嵌入 + Transformer encoder + 多 horizon 头），带 AMP / checkpoint resume / 时序 validation 切分。当前作为研究 API 暴露。
- 走式 walk-forward：
  - 默认 purged walk-forward CV（`quantagent.quant_math.purged_cv`）。
  - 新增 `quantagent.training.splitters`：expanding / rolling / purged / chronological 四种模式，date-aware 索引，可直接遍历 `(fold, train_frame, valid_frame)`。
- 训练 artifact 写入 `<models_root>/v7_alpha/`：
  - `model_coefficients.json`、`metrics.json`、`feature_schema.json`、`label_schema.json`、`training_config.json`
  - `data_quality_report.json`、`acceptance_report.json`、`walk_forward_predictions.csv`
  - `experiment_manifest.json`（experiment name、horizons、git commit、fold count、production_ready、backend、model_downgraded、adverse_regime_report）
  - `boosters/horizon_<h>.<backend>.txt`（LightGBM/XGBoost 原生模型文件）
  - `predictions/predictions.parquet` + `predictions.summary.json`
  - `target_weights/target_weights.parquet` + `target_weights.diagnostics.json`
  - `deep/deep_alpha_state.json` + `deep/deep_alpha_config.json` + `deep/deep_alpha_feature_schema.json` + `deep/deep_alpha_metrics.json` + `deep/deep_alpha_experiment_manifest.json`
  - `ft_transformer.pt` + `ft_transformer_config.json` + `ft_transformer_feature_schema.json` + `ft_transformer_metrics.json`（独立 FT-Transformer trainer）
- `<models_root>/v7_alpha/registry/<experiment>.json` + `latest.json`（`ModelRegistry`）。
- 评估指标在 `quantagent.training.metrics` 中统一：IC、rank IC、ICIR、top-minus-bottom spread、Sharpe、Sortino、max drawdown、hit rate、capacity proxy。`compose_alpha_metrics` 给出一组完整结果，可直接写入 `metrics.json`。

## 因子表达 DSL / Factor Expression DSL

`quantagent.factors.expr` 提供 Alpha101-style 符号化因子语法：

```python
from quantagent.factors.expr import Rank, TsMean, TsStd, Returns, Close, build_factor_frame, register_factor

register_factor("momentum_5d", Rank(TsMean(Returns(Close, 1), 5)))
register_factor("realized_vol_20d", Rank(TsStd(Returns(Close, 1), 20)))

wide = build_factor_frame(market_panel)            # 宽表 (factor_*)
long = build_factor_frame(market_panel, long_format=True)  # 长表
```

所有时序算子按 `symbol` 分组、按 `trade_date` 升序计算，`Rank` 是 cross-sectional per `trade_date`，单元测试 (`tests/test_v7_factor_expr.py`) 已覆盖无 lookahead 不变量。

## 参数搜索 / Parameter Optimisation

`quantagent.training.optimize.run_alpha_param_search` + `quantagent optimize-alpha-v7` 提供 grid / random search：

```powershell
quantagent optimize-alpha-v7 \
  --dataset E:\AI量化\data\v7\gold\training_dataset\training_dataset.parquet \
  --search-space configs/example_search_space.json \
  --sampler grid --objective rank_ic_mean --mode max
```

- 搜索空间为 JSON `{"model": ["ridge", "elastic_net"], "min_train_rows": [100, 500]}` 等。
- 每个 trial 调用 `run_v7_training_experiment` 并解析 `metrics.json`。
- 输出写入 `<reports_root>/v7/optimization/optimization_report.json` 以及每个 trial 子目录。

## 安全 / Safety

- `live_trading_enabled=false`、`dry_run=true`、`virtual_broker_only=true` 永远是默认。
- AkShare/TuShare network 必须 `--allow-network` 显式开启。
- `build-training-dataset-v7` 拒绝 `allow_synthetic_fallback=true`，PIT 违反会被 quality gate 阻断。
- `AkShareSectorProvider` 离线时直接 `ProviderUnavailable`，绝不 cross-join 行业到所有 symbol。
- `pit_wide_merge_statements` 对每个 statement 做 prefix 化，重复 `(symbol, report_period, available_at)` 会 raise。
- `evaluate_adverse_regime` 真实计算 bottom-quartile 交易日的 rank-IC；不再硬编码 `adverse_regime_passed=True`。
- Production-ready 标记需通过 `evaluate_model_acceptance_gates` 中所有 gate：rank IC、stability、turnover-adjusted net return、drawdown、adverse regime（真实计算）、paper report、非 mock。
- Target-weights 优化器的 ST / 停牌 / 涨停 / 跌停 四个 tradability flag 通过 `_TRADABILITY_CONSTRAINTS` 显式映射到 `block_st` / `block_suspended` / `block_limit_up_buy` / `block_limit_down_sell` 配置；`tests/test_v7_target_weights_constraints.py` 锁住该映射。
- `QlibProvider.daily_ohlcv` 现在为 close-derived 数据计算 `available_at = next trading row`（最后一行 fallback 到 `trade_date + 1 day`），`tests/test_v7_qlib_pit.py` 校验。
