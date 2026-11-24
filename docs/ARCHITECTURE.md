# 系统架构 / Architecture

本文描述**实际运行**的系统（Round 31 用 AST 导入图 + CLI 注册 + JobRunner 白名单 +
crontab + CI 推导，而非沿用旧文档）。治理规则见 [GOVERNANCE.md](GOVERNANCE.md)，
数据见 [DATA.md](DATA.md)，运维见 [OPERATIONS.md](OPERATIONS.md)。

## 1. 入口 / Entry points

| 入口 | 位置 | 说明 |
|---|---|---|
| CLI | `quantagent`（`quantagent.cli:app`，Typer） | 约 115 个命令；`QUANTAGENT_ENABLE_LEGACY_CLI` 只再开放 `v8_intraday` 的做T 复现命令 |
| Web API | `python -m services.quant_api` | FastAPI，同时托管 `apps/quant-ui/dist`（React VNext） |
| JobRunner | `services/quant_api/services/jobs.py` `COMMANDS` | 48 个白名单命令（11 个 CLI + 37 个脚本），统一经 `scripts/job_supervisor.py` |
| 定时任务 | 用户 crontab（8 条） | 见 OPERATIONS.md §定时任务；systemd 单元已删除 |
| CI | `.github/workflows/*.yml` | 全量 pytest（按 `requirements/ci-constraints.txt` 精确安装）+ 依赖审计 + 5 个 smoke 脚本；每周 `Dependency Drift` 用范围内最新版本跑全量 |
| 规范评估器 | `scripts/baseline_protocol.py` variant C | 唯一可信收益数字入口 |

## 2. 研究链路与权威模块 / Live flow and authoritative modules

```
数据   quantagent.data.ashare (adapters/units/contracts/readiness)  ← scripts/u0_*.py
  │    gold: scripts/build_u0_full_universe_gold.py → data.ashare.gold_bridge / execution_panel
因子   factors.alpha101 / gtja191 / cicc_ashare80 / alpha181；DSL factors.expr；
  │    LLM 闭环 factors.factor_synthesis（synthesize-factors-v7，受 expression_safety 约束）
数据集 data.dataset_builder.v7_training_dataset（拒绝 post-decision 列）
训练   training.v7_experiment（run-full-real-training-v7）；training.ft_transformer_trainer（train-v8-deep）
对比   research.model_comparison（audit-nonlinear-factors）；fusion.search（search-factor-fusion）
预测   training.v7_predictor；portfolio.multi_horizon_blender
组合   portfolio.v7_target_weights.build_candidate_weights（只输出 target_weights）
回测   backtest.strict_v8 → backtest.ashare_execution_simulator(_impl)
  │    → execution.order_manager / virtual_broker / fill_simulator / cost_model
评估   backtest.paper_report；quant_math.performance（PSR/DSR/MinTRL）；backtest.quarantine
纸面   paper.continuous_execution / daily_loop → execution.paper_adapter → paper.broker + paper.risk.RiskEngine
```

单一来源（同一概念只在这里实现，其他位置应委托它）：

| 概念 | 权威实现 | 已知残留重复（ROADMAP P1-3） |
|---|---|---|
| A 股交易规则（涨跌停、整手、T+1、印花税日程） | `market_rules/ashare.py` | `quant_math/ashare.py` 已委托；do-T 研究路径有 3 份 `_symbol_limit_band` 拷贝 |
| 交易成本 | `execution/cost_model.AShareCostModel` | `market_rules.trading_costs`（纸面/streaming，佣金默认值不同）、`backtest/fill_model`、`intraday_fill.CostConfig` |
| 绩效指标 | `quant_math/performance` | `strict_v8._compute_metrics`、`paper_report`、`backtest/engine.py` 各有一份 |
| 截面 rank-IC | `factors/evaluation` | 约 10 份私有拷贝，最小截面数不一致 |
| 走式切分 | `training/splitters.py` + `quant_math/purged_cv` | 若干私有 `_walk_forward_splits` |
| 风控 | `paper/risk.RiskEngine`（订单路径上的唯一一个） | `risk/kill_switch`、`risk/risk_gate` 只服务 `production-readiness` 报告 |
| 存储路径 | `config/paths.quant_paths()`（`QUANTAGENT_HOME`） | — |

## 3. 安全边界 / Safety boundary

- 终态 `LIVE_DISABLED`：实盘意图在 agent/routing 之前即被 `safety.operating_mode`
  拒绝；`execution/qmt_gateway.QMTGateway` 默认 `dry_run=True`、
  `live_trading_enabled=False`，仓库内没有任何生产构造点。
- LLM / agent 只产出证据、观点、约束与因子表达式；优化器只产出 `target_weights`；
  只有 `OrderManager` 把权重转成订单意图。
- 纸面账户：订单前 `check_order`、开盘与每笔成交后 `check_portfolio`；日亏 = 开盘权益 5%
  当日停买；回撤 20% 人工解除的只减仓锁；无行业图拒买。
- 研究被否决（退出码 3，`rejected`）≠ 工程故障（`failed`）。

## 4. Web 工作站 / Workstation

- 只维护 `apps/quant-ui/src/vnext` 外壳与注册页面（ADR-002）；设计 token 唯一来源
  `src/vnext/styles/tokens.css`。
- 页面 → API → 产物：`services/quant_api/runtime_indexer` 索引 `runtime/` 产物，
  `services/run_results.py` 从运行自己的产物推导结论（缺失即缺失）。
- 架构决策记录：[ADR-001](architecture/adr-001-institutional-workstation-shell.md)、
  [ADR-002](architecture/adr-002-vnext-only-cutover.md)、
  [ADR-003](architecture/adr-003-atlas-fusion-and-council.md)、
  [ADR-004](architecture/adr-004-council-v2-company-protocol.md)。

## 5. 刻意保留但不在主链上的代码 / Retained off the main path

| 模块 | 为什么保留 |
|---|---|
| `reconciliation/*`、`streaming/*` | Module One/Two 四引擎复合回放证据（`scripts/module1_composite_replay.py`、`phase2_ledger.py`） |
| `backtest/microstructure_simulator`、`data/microstructure/*` | Phase-2 微观结构证据脚本 |
| `rl/*` | 预注册结论 DO NOT ENABLE；保留 harness 以便复现，默认不进入任何生产路径 |
| `backtest/trace_proven_strict_v8`、`execution/live_model_trust*` | 生产证据封装（模型信任证书） |
| `execution/qmt_gateway`、`safety/readiness_tiers` | 实盘接缝与就绪证书契约（安全边界） |
| `services/v7_pipeline_service`、`themes/*`、`fundamental/*` | 只能经由无引用的 v7 CLI 命令到达；ROADMAP P2-2 决定去留 |
