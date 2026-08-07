# QuantAgent V7 Codex 指令 / Codex Guide

## 项目目的 / Project Purpose

QuantAgent V7 是面向 A 股散户现实约束的 PIT quant research system。仓库不是 toy LLM trading demo，不提供 financial advice，不承诺收益。研究输出必须经过测试、回测、风控、paper trading 和 live-readiness reporting，才允许讨论任何 live execution path。

V7 覆盖：

- Point-in-Time financial provider (TuShare / AkShare) + local Parquet/CSV cache。
- Daily Evidence Ingestion Layer (`data/ingestion/*` + `EvidenceStore`)；每条 `EvidenceRecord` 必须带 source、published_at、available_at、raw_hash 和 confidence。
- Qlib CN market panel、technical features、label generation、training slices、backtest base。
- Dynamic theme discovery、industry chain graph、stock pool hard gate。
- Fundamental due diligence、Financial Fraud Risk、News Credibility、Intrinsic Valuation。
- Multi-horizon Alpha: 1 / 5 / 20 / 60 / 120 / 126 days。**生产模型 = FT-Transformer sleeves**（`cli/v8_deep.py train-v8-deep` + `configs/production_blend.json`，单命令物化 `scripts/materialize_production_composite.py`）。Ridge/ElasticNet 为 v7 classical 基线；`models/v7_deep_alpha.py` 等启发式 scorer 非生产（见文件头 STATUS WARNING）。
- Purged walk-forward CV、model artifacts、metrics、acceptance gates。
- A-share execution simulation：T+1、limit-up/down、suspension、ST、lot size、volume cap、slippage、cost、partial fills、failed order audit。
- QMT execution-preparation、Risk Gate、Kill Switch、Reconciliation、Audit Replay。

## A 股安全约束 / A-share Safety Constraints

- No live trading by default：默认禁止实盘交易。
- `QMTGateway` 必须默认 `dry_run=True`，`live_trading_enabled=False`。
- Agents never emit orders：LLM / Agent 只能输出 evidence、views、constraints、confidence、risk flags、audit logs。
- Optimizer never emits orders：Portfolio Construction 只能输出 `target_weights`。
- Only `OrderManager` converts target weights into order intents。
- QMT submit 前必须通过 risk gate、kill switch、execution constraint simulation、reconciliation、audit replay。
- Production mode must not use synthetic fallback；mock data 只允许在 tests 和 smoke examples。
- 不允许新增任何 guaranteed profitability 或收益保证表述。
- **评测窗口纪律（2026-07 起）**：`configs/quarantined_windows.json` 定义禁评窗（被烧 holdout 2025-09-01→2026-05-18；冻结新鲜窗 2026-05-19+，正式首读 ≥120 交易日）。可信评测唯一入口 = `scripts/baseline_protocol.py` variant C（守卫 fail-closed）；任何数字引用须带 trust class（见 `BASELINE_TRUST_CLASSIFICATION.md`）。改进验收规则见 `ACCEPTANCE_RULES.md`。

## Code Style / 代码规范

- Python code、comments、docstrings、function names、class names、variable names、test names、config keys 使用 English。
- Markdown 必须 Chinese-English mixed，以中文说明为主，保留关键 English terms。
- 新增财务字段必须同时定义 `report_period`、`ann_date`、`available_at`，否则不能进入 PIT cache。
- Optional dependencies must degrade gracefully；real-data commands must report actionable install/setup errors。
- 优先做 wrappers、adapters、integration seams，不删除仍被引用的 SOTA components。
- 删除 unused code 或 obsolete `.md` 前，必须证明未被 imports、CLI、tests、README、AGENTS 或 docs 引用。
- 所有 silver / gold artifact 必须伴随一份 `<lake_root>/manifests/<dataset>.json`（`quantagent.data.manifest.DataManifest`）。
- 大数据/模型/报告默认写入 `E:\Project\QuantAgent\runtime\`（Windows）或 `~/AI_quant`（POSIX），通过 `QUANTAGENT_HOME` 环境变量覆盖。`quantagent.config.paths.quant_paths` 是单一来源。

## Institutional Workstation VNext / 一体式工作站

- Web product 只维护 `apps/quant-ui/src/vnext` shell 与其注册页面；禁止恢复 legacy shell、第二套路由或第二套 backend/schema。
- 决策总览与训练实验室是所有 domain page 的结构基线：`WorkbenchHeader`、最多 6 项的 `WorkbenchMetricStrip`、可审计主画布、右侧 evidence/operation inspector、底部全局 Operations Dock。
- Night / dawn / day 必须共享同一 semantic token；图表统一经过 `EChart` theme adapter，禁止页面写死只在深色模式可读的轴线、tooltip 或 dataZoom 配色。
- Empty state 必须解释缺少的 artifact、保持的安全状态和下一步操作；不得留下大面积无说明空白，也不得用 mock/fabricated metric 填空。
- 回测主上下文只能单选；多实验进入独立 Compare，最多 4 项。模型/因子比较最多 4 项，避免视觉拥挤和指标来源混合。
- 因子发现沿用现有 `synthesize-factors-v7` 与统一 JobRunner：证据 → 受限 DSL → schema → PIT/leakage → compute → IC/decay/correlation/regime → portfolio impact → human review → registry gate。LLM 与 network 均需显式、分离确认。
- K 线和时序图遵循人类操作契约：滚轮只缩放、左键拖拽只平移、slider 可直接调整窗口，键盘提供 pan/zoom/latest/all；React rerender 后保留手动窗口。
- Help 永远留在 QuantAgent 内；vn.py / VeighNa 仅作为 version-pinned design and capability audit source，不把产品帮助入口跳到外部网站。
- 大型 TickFlow/runtime 文件保持服务器侧：网页只提交受限 Runtime 路径；coverage/duplicate scan、quarantine transfer、DataRecorder 与 cleanup 都必须内存有界、可取消、可审计。

## Real-Data Commands / 真实数据命令

```powershell
quantagent storage-info-v7 --ensure
quantagent setup-qlib-v7 --region cn                      # 仅打印官方下载命令
quantagent setup-qlib-v7 --region cn --run --allow-community-fallback   # 若 pyqlib 已装则直接下载
quantagent download-qlib-v7 --target-dir E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data --region cn
quantagent check-qlib-v7 --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data --symbols 600519.SH
quantagent build-market-panel-v7 --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data --symbols 600519.SH --start-date 2020-01-01 --end-date 2026-05-15
quantagent build-akshare-v7 --symbols 600519.SH,000858.SZ --start-date 2020-01-01 --end-date 2026-05-15 --allow-network
quantagent build-valuation-v7 --as-of-dates 2026-05-15 --allow-network
quantagent build-labels-v7 --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet
quantagent build-training-dataset-v7 --market-panel ... --labels ... --fundamentals-root ...
quantagent train-alpha-v7 --dataset ...
quantagent train-alpha-v7 --dataset ... --model lightgbm           # real LightGBM, fail-loud if missing
quantagent train-alpha-v7 --dataset ... --model xgboost --allow-model-downgrade   # ridge fallback only with the flag
quantagent train-deep-alpha-v7 --dataset ... --horizons 1,5,20,60,120,126
quantagent optimize-alpha-v7 --dataset ... --search-space search.json --sampler grid
quantagent predict-alpha-v7 --model-dir ... --feature-dataset ...
quantagent build-target-weights-v7 --predictions ... --market-panel ... --sector-map ...
quantagent run-real-training-v7 --market-panel ... --labels ... --fundamentals-root ...
quantagent run-full-real-training-v7 --market-panel ... --labels ... --sector-map ...   # dataset → train → predict → target_weights → backtest
quantagent evaluate-alpha-v7 --metrics ... --paper-report ...
quantagent walk-forward-backtest-v7 --target-weights ... --market-panel ...
quantagent walk-forward-backtest-v7 --predictions ... --market-panel ... --sector-map ...   # optimiser runs first
quantagent paper-trade-v7 --target-weights ... --market-panel ...
quantagent v7-live-readiness-report --metrics ... --paper-report ...
```

Qlib CN official command:

```powershell
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

## Acceptance Gates / 验收门槛

Model 不能标记 production-ready，除非：

- zero PIT violations。
- 每个 horizon 有足够 training rows、symbol coverage、date coverage。
- out-of-sample RankIC stability 为正。
- turnover-adjusted net return after cost 通过阈值。
- max drawdown 低于配置阈值。
- no single factor dominates unrealistically。
- 至少一个 adverse market regime 验证通过（`evaluate_adverse_regime` 真实计算 bottom-quartile-day rank-IC；旧的 `adverse_regime_passed=True` 硬编码已删除）。
- paper trading report exists before live readiness can pass。
- 结果不是只在 mock data 上成立。

## Data 防错铁律 / Data Anti-Footgun

- `AkShareSectorProvider` 必须用 per-board membership endpoint 或 local mapping；**绝不** 把所有 industry 当成 cross-join 应用到每个 symbol。
- 财务报表合并必须走 `pit_wide_merge_statements`：按 statement type 加列前缀（income_revenue / balance_total_assets / cashflow_operating_cash_flow / indicator_*），按 PIT 四键 outer-merge，重复 `(symbol, report_period, available_at)` 必须 raise。
- 真实数据 manifest（`DataManifest`）必须包含 provider、source paths、generated_at、row_count、date_range、symbols、schema report、PIT violations、duplicate rate、warnings 和 content hash。

## A股数据底座 / A-share Data Foundation

- 全宇宙（U0）行情、身份与 PIT 元数据的唯一入口是 `quantagent.data.ashare`；
  新增数据源必须实现 adapter + capability probe，而不是在脚本里直接发请求。
- 每一行落盘数据必须带 provenance（`source` / `source_endpoint` /
  `retrieved_at` / `available_at` / `quality_status`）；同一 symbol 的历史
  **不允许**跨供应商拼接，换源必须写 `SourceBoundary`。
- 单位与复权口径写进 contract 并**实测验证**：`volume` = 股（vendor 的"手"在
  adapter 边界 ×100），`amount` = CNY，面板 = raw 未复权。复权口径用
  `scripts/u0_adjustment_forensics.py` 以除权日因子回放取证。
- 存续期内没有 bar 的交易日不写 NaN 行，进 `session_gaps.parquet` 并分类为
  `SUSPENDED`（命中供应商停牌区间）或 `MISSING_UNEXPLAINED`。
- U0 关卡一律由产物推导（`quantagent.data.ashare.readiness`）；**禁止**把关卡
  写成常量 `True`，NOT_RUN 不得当作 PASS，缺证据即 NOT_READY_INTEGRATION。
- 缺失来源记 `BLOCKED_BY_DATA` 并列出**已实测过的候选来源**；部分覆盖（如仅
  SZSE 有简称变更登记）算 BLOCKED，不算通过。
- 详见 `docs/ashare_data_foundation.md`。

## New Modules / 新增模块

- `quantagent.config.paths` — 统一 `E:\Project\QuantAgent\runtime\` 存储布局，环境变量 `QUANTAGENT_HOME` / `QUANTAGENT_DATA_ROOT` 覆盖。
- `quantagent.training.splitters` — expanding / rolling / purged / chronological 走式 walk-forward 切分。
- `quantagent.training.optimize` — alpha 超参 grid / random search，默认写 `E:\Project\QuantAgent\runtime\reports\v7\optimization\`。
- `quantagent.factors.expr` — Alpha101-style 符号化因子 DSL，`Rank(TsMean(Returns(Close, 1), 5))`，零 lookahead 测试覆盖。
- `quantagent.models.ft_transformer` — FT-Transformer 表格架构（PyTorch 可选）。
- `quantagent.training.ft_transformer_trainer` — FT-Transformer trainer（AMP / checkpoint resume / 时序 validation 切分）。
- `quantagent.cli.v7_storage` — `storage-info-v7` / `setup-qlib-v7`。
- `quantagent.cli.v7_optimize` — `optimize-alpha-v7`。

## Testing Commands / 测试命令

```powershell
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m pytest tests/
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m compileall src
git diff --check
```

## Provider 责任 / Provider Responsibilities

- Qlib：行情、technical factors、label generation、training slices、backtest base。
- TuShare / AkShare：财务报表、财务指标、估值字段、公告披露日期。
- TradingView public pages：sentiment / attention context，不作为基本面或行情真值。
- Policy、announcement、news 原文必须保留 `source / published_at / available_at / raw_hash / confidence` 并进入 `EvidenceStore`。

## ATLAS 设计系统 / Design System

- `apps/quant-ui/src/vnext/styles/tokens.css` 是**唯一** semantic 颜色/字号/间距/动效
  来源，三主题（night / dawn / day）定义同一套 token 名。页面禁止写死 hex；
  缺少的颜色先补进 tokens，再在页面使用。
- 颜色语义分三族，互不重叠：`ui.*`（交互与系统态：azure / violet / cyan / amber）、
  `status.*`（治理裁决：emerald / amber / crimson）、`market.*`（A 股涨跌：**红涨绿跌**）。
  market 色只用于价格与收益数字单元格；status 色必须带文字标签。
- `foundation.css` 提供七个基础原语：`atlas-surface`（可带 `data-rail` 信号轨）、
  `atlas-eyebrow`、`atlas-figure`、`atlas-chip`、`atlas-grid`、`atlas-meter`、
  `atlas-empty`。新页面优先组合这些原语，不要新造一套。
- 图表颜色统一取 `useVNextChartPalette()` 的 `series` / `primary` / `agent` 等，
  与 `--viz-*` token 一一对应；禁止页面内写死只在深色下可读的配色。

## 因子融合工场 / Alpha Foundry（ATLAS L2）

- 融合搜索唯一入口 = `quantagent.fusion` + governed command `search-factor-fusion`。
- **`n_trials` 由枚举的搜索空间决定，任何 API/CLI/UI 都不得提供该参数**；
  它直接进入 Deflated Sharpe 的收缩项，可声明的试验次数等于可伪造的显著性。
- 拟合方案（`ic_weighted` / `ic_ir_weighted` / `inverse_volatility` / `genetic`）
  只能读训练段；对照方案（`equal` / `random_simplex` / `single_factor`）不读训练段，
  但**必须计入试验次数**，事后删除对照来美化 DSR 属于造假。
- 产出的是 **Pareto 前沿**而非单一最优；偏好权重只对前沿内候选排序，
  不改变候选生成、也不改变谁进入前沿。
- 回撤按调仓频率净值序列计算（每 horizon 日一期），UI 必须说明这是日频回撤的下界。
- 基准缺省为 `universe_equal_weight` 时必须标注它包含不可交易标的，会高估超额。

## 决策议事会 / Decision Council（ATLAS L5）

- 七个角色（data_quality / factor_integrity / model_validation / fusion_search /
  portfolio_risk / execution_realism / governance）各自只在 `vetoScope` 内否决。
- 每条裁决必须附带 `evidence`（它实际读取的字段）；**证据缺失记 `unknown`，
  永远不记 `pass`**；`unknown` 不阻塞研究，但也不算放行。
- 人工可推翻任一角色，但推翻必须带 author + 至少 8 字理由，写入
  `runtime/jobs/**/council_overrides.jsonl` append-only 日志；
  原裁决与推翻记录并列保存，代码中没有删除路径。

## 任务控制 / Job control

- 状态机：`queued → starting → running ⇄ paused →
  succeeded|failed|cancelled|rejected`。`starting` 表示进程尚未注册，此时
  pause/cancel 无信号可发；status 只有在 `Popen` 成功并登记进程后才变为 `running`。
- **`rejected` 是终态但不是失败**：运行走完了，是预先声明的研究闸门否决了候选
  （退出码 `3`，见 `quantagent.research.verdict`）。UI 必须把它呈现为**结论**，
  与 `failed`（工程故障）分开；产物完整保留，`research_verdict.json` 记录闸门、
  实测值与补救方向。闸门不得事后放宽来"通过"。
- 每个任务由 `scripts/job_supervisor.py` 包一层：它把 worker PID 和**真实退出码**
  写到 `<job>.status.json`。job 进程 `start_new_session=True` 且 stdout **直接写
  日志文件**（不走管道）——用管道时 API 一重启，训练就会阻塞在写满的 64KB 缓冲区上
  永远"运行中"。API 重启后按序恢复：状态文件里有退出码就据此结束；进程（supervisor
  或 worker 任一）还活着就重新接管并继续跟踪；两者都没有则记
  `exitStatusObserved=false` 并**如实说明退出状态未知**，不得直接断言 failed。
- `pause` 用 SIGSTOP、`resume` 用 SIGCONT，作用于**整棵进程树**：这是**调度**控制，
  不释放 RAM/GPU，UI 必须如实说明暂停的进程仍占用显存。
- 失败必须命名：`services/quant_api/services/job_diagnostics.py` 从任务自己的日志
  归类原因并给出补救动作；无法归类时记 `unclassified` 并附日志尾部，**不得猜测**。
- `retry` 只对 `failed` / `cancelled` 开放，且以**原参数**重放到原 output_dir；
  `succeeded` / `rejected` 重试会覆盖已有证据，应改为从策略发起新的运行。
- JobRecord 持久化 `parameters`，否则任何已完成的任务都无法复现、重试或解释。

## Strategy Workbench / 策略实验室

- 策略 Web 启动统一走 `quantagent.strategy.v1` manifest 与
  `run-full-real-training-v7` allowlist；禁止另建浏览器内训练/回测实现。
- 策略链固定为真实输入校验 → manifest → Human Gate → dataset/factor/train →
  target_weights → A股回测 → risk/paper evidence。
- 用户输入的收益/回撤目标只是研究偏好，UI 不得把它渲染成真实或承诺收益。
- Web API credential 只允许进程内存或服务器环境变量，禁止 URL/argv/log/Runtime/
  localStorage；只向命令声明的 provider 注入。
- 不提供 arbitrary Bash；新执行能力必须登记 allowlisted command、路径边界、
  控制开关、credential provider 和测试。
- **策略是有版本历史的实体，不是一堆文件**：同一 `id` 的多次保存是版本，
  `GET /api/strategies` 每个策略只返回一行（附 `versionCount` / `runCount`）。
  删除走归档（`runtime/archives/strategies/`），运行产物只有显式请求才一并删除。
- 每次启动登记一条 run（`runtime/strategies/<id>/runs.jsonl`），把策略版本、job、
  output_dir 串起来；缺了这条链，完成的运行就只是一个匿名目录。
- 结论由 `services/quant_api/services/run_results.py` 从运行**自己的产物**推导：
  验收闸门、PBO/DSR/SPA、训练证据、成本后净值都带 `sourcePath`。**产物缺失即
  缺失**，不得渲染成 0 或 pass；短窗口年化必须带评估天数警告。
- 发起前必须做**可算的**前置校验，避免训练跑完才失败：
  基准标的是否真在所选面板内；`nSplits × 20` 是否 ≥
  `selectionMinOosDays + selectionMinHoldoutDays`。二者曾是出厂默认值，
  每次运行都在 ~62% 处中止。
- 运行对比上限 4 项，逐指标标注更优方向；跨研究范围/宇宙/评估窗口的运行不可直接比较，
  UI 必须写明这一点。
