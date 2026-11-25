# 数据 / Data

本文件是数据层的唯一说明，合并了原 `docs/ashare_data_foundation.md`（下文 §A 起保留其全文）、
`docs/u0_data_layer_final_state.md`、`docs/V7_PIT_data_contract.md`、
`docs/V7_training_dataset_schema.md`。

## 0. 现状一览 / Status (2026-11-24)

### 数据血缘 / Lineage

```
TickFlow(adjust=none) ─┐   Tencent / Sina / Eastmoney（data.ashare.sources + units，逐响应判定单位）
                       ├─> U0 原始日线 runtime/data/u0/panel/daily_bars_raw.parquet（至 2026-07-24）
U0 PIT：adjust_factors、corporate_actions、suspension、st_intervals（仅深市）、security_master
   └─> scripts/build_u0_full_universe_gold.py ─> runtime/data/gold/full_universe_r29/
          {execution_panel（原始价 + ca_* 分红送转 + 三态涨跌停/ST/停牌）, adjusted_market_panel,
           dataset, labels（delay-1）}
          ├─> 规范评估器 scripts/baseline_protocol.py（默认 execution_panel）
          ├─> Factor Lab r29 筛选、audit-nonlinear-factors
          └─> Strategy Workbench 默认输入（Round 31 起）
v7 银层面板 runtime/data/v7/silver/market_panel（qlib_rescaled + akshare qfq + 东财“手” 拼接）
   **REFUTED**：价格非原始、成交量单位混杂、ST 为今日名单广播（前视）。仍被旧的
   v8.9 生产 blend、fresh_blind 纸面循环与部分 v8 CLI 默认值引用（ROADMAP P1-5）。
```

### 新鲜度 / Freshness

| 数据集 | 数据截至 | 状态 |
|---|---|---|
| U0 原始面板 | 2026-07-24 | 冻结；无定时生产者（22:00 的 `u0_full_universe_backfill.py` 是已被取代的缺失符号补抓器，不是日更，且未带 `--allow-network` 而空转） |
| r29 认证 gold + 执行面板 | 2026-07-24 | 冻结；研究可用窗口 ≤ 2025-08-29（隔离窗见 GOVERNANCE.md） |
| v7 银层面板 | 2026-08-05 | REFUTED 且停更 |
| fresh_blind 前向面板 | 2026-08-05 | **ST 证据已齐**（2026-11-25）：3,273 个种子（沪 1,210 + 深 2,063）在 2026-08-06→ 全部有带日期 ST 证据；下一次 06:00 catchup 起恢复补齐（以其日志为准） |
| 行业图 sector_map | 2026-05-31 快照 | 缺 1,924 个认证符号；当期快照 = 前视，只可用于归因 |
| ST 区间 | 2026-11-24（每日刷新） | `pit/st_pit_evidence.parquet`，每日 06:00 刷新。深市：交易所简称变更登记（2016→）；沪/北：**2026-07-27 起**为 ENG_ST31 前向锚定登记（预注册验证通过）；**2026-07-27 之前仍 BLOCKED_BY_DATA**（ENG_ST30 HOLDOUT 未过），见下 |

### ST / 风险警示证据（ENG_ST30 + ENG_ST31，2026-11-25）

- **契约**：`quantagent.data.providers.st_pit` 区间表（`symbol, start_date, end_date(含), is_st, available_at`），
  ST 与非 ST 区间都必须显式给出；缺行即拒绝（不把"没有记录"读成"不是 ST"）。默认路径
  `runtime/data/u0/pit/st_pit_evidence.parquet`（`QUANTAGENT_HISTORICAL_ST_PATH` 可覆盖）。
- **深市**：交易所简称变更登记（`akshare.stock_info_sz_change_name`），`scripts/u0_st_announcements.py refresh-szse`
  每日重取；认证用的 `pit/st_intervals.parquet`（2026-07-26）不改动。新登记与旧登记的 906 个历史区间完全一致。
- **沪/北**：`scripts/u0_st_announcements.py` 从巨潮公告（类别"特别处理和退市" + 标题关键词）、公告正文生效日、
  公告上的当日简称重建区间。预注册 `docs/research/preregistrations/ENG_ST30_preregistration.md`：DEV（深市偶数尾号）
  7 项全过；HOLDOUT 一次评测 M5b 行精度 0.924 < 0.98 ⇒ **十年历史不写入**，2026-07-27 之前保持 BLOCKED_BY_DATA。
  原始缓存（16,766 条公告、3,973 份正文、名称探针）在 `runtime/data/u0/pit/_st_announcements/`。
- **沪/北前向（ENG_ST31，`scripts/u0_st_forward.py`）**：从带日期的全市场名称快照（TickFlow `get_instruments`，
  A0 = 2026-07-26 19:39 CST；此后每日 05:00 CST 一张）出发，按 ENG_ST30 冻结分类器解析的巨潮公告向前推进状态；
  每张快照和每条巨潮当日简称（活跃公司逐家抓取全部公告）都必须与推进状态一致，否则自上次确认起记 UNKNOWN（无证据行）。
  预注册 `docs/research/preregistrations/ENG_ST31_preregistration.md`：深市 2026-07-27→2026-11-24 一次验证
  V1–V5 全过（235,036 行 0 错、19/19 变更同日、0 UNKNOWN）⇒ 沪/北 2026-07-27 起写入。
  **每日监控**：`publish` 每天用同一方法重算深市并与当天登记对比，V1/V2/V5 低于阈值或 V4 超界即撤下沪/北行（fail-closed）。
- **刷新**：`scripts/u0_st_register_refresh.sh`（由 06:00 `catchup_supervisor.sh` 调用）：重取未收盘月份的公告、
  深市登记、新快照、活跃公司简称，再 `u0_st_forward.py publish`（先发深市，再按决策规则与监控追加沪/北）。
  交易日历 `pit/trading_calendar.parquet` 止于 2026-12-31，年底前必须延长，否则构建拒绝运行。

### r29 认证面板完整性（Round 31 复核，≤ 2025-08-29）

- 文件 sha256 与帧内容哈希均与 manifest 一致；0 重复键；价格为原始价（600519.SH 2023-12-29 收盘 1726.00）。
- 0 日历缺口；退市日之后 0 行；每行 `available_at` 15:00；0 个 OHLC 违例。
- 分红送转入账在 32,378 个除权步中 99.47% 复现后复权收益；12.6% 的 factor-value 入账与 hfq 偏差 > 0.5%。
- 已知缺陷：16 个 B 股在 r29 dataset/labels/面板中（评估器入口已排除；Round 31 起构建器排除，
  需下次重建生效）；北交所两个成交量单位错误簇（2020-07→2021-11 六只 100×；2022-04-12→14 79 只）；
  r29 labels 无 `label_end_{h}d`（Round 31 起构建器写出，需下次重建）。

## 1. PIT 契约 / Point-in-Time contract

- `available_at` 是信息可见时点，不等于 `report_period`。日频收盘派生特征在**其自身交易日收盘**
  可用；一切 join 使用 `available_at <= trade_date`（DEF-026 后的口径）。
- 财务行必须同时带 `report_period`、`ann_date`、`available_at`、`source`；报表合并只走
  `pit_wide_merge_statements`（按报表类型加前缀，重复 `(symbol, report_period, available_at)` 抛错）。
  盘后公告的财报次一交易日可用。
- 政策/公告/新闻原文必须保留 `source / published_at / available_at / raw_hash / confidence`，
  以 `EvidenceRecord`（`quantagent.data.evidence.canonical`）进入证据层。
- 每个 silver/gold 产物附 `DataManifest`（provider、source paths、generated_at、row_count、date_range、
  symbols、schema report、PIT violations、duplicate rate、warnings、content hash）。
- 训练数据集不变量（有测试覆盖）：特征无 `available_at > trade_date`；标签与 post-decision 列
  （`entry_*`、`forward_*`、`fwd_*`、`next_*`、`label_*` 等）不得成为特征（Round 31 起构建器拒绝并记录
  `refused_feature_columns`，钉住的 schema 若要求此类列直接失败）；推理 schema 无原始前向标签。
- 标签口径：认证数据与 `_exec_` 数据集为 **delay-1**（`close(t+1+h)/close(t+1)−1`）；delay-0 会使短周期
  IC 虚增（ROADMAP P1-2）。每个标签产物带口径戳（`data.label_contract`：parquet 元数据 `quantagent.label_convention`，
  CSV 为绑定内容哈希的 `.label_convention.json` sidecar），id：`delay1_close_t1_to_close_t1_plus_h_v1`（gold，按标的**下一行**入场，
  出场跨度不受交易日约束）、`delay1_exact_global_session_t1_to_t1_plus_h_v1`（`_exec_` 与 `build-labels-v7 --label-convention delay1
  --market-calendar <交易日历>`：入场必须恰为下一全局交易日、出场恰为其后 h 个交易日，否则 NaN，标签跨度 ≤ 1+h）、
  `same_close_t_to_close_t_plus_h_v1`（delay0）。两种 delay-1 id 不同，不会被静默混用。`build-labels-v7` 无默认值必须显式选择；数据集构建器等消费者拒绝无戳
  （仅 `--allow-legacy-unstamped-labels` 放行并记 legacy）与混用。旧 `runtime/data/gold/full_universe/labels.parquet` 无戳
  （列推断为 delay-0），默认会被拒绝；r29 认证标签已带戳。

## 2. 训练数据集 schema

`quantagent.data.dataset_builder.v7_training_dataset` 输出 `<dataset>.feature_schema.json`：
`feature_version`、`schema_hash`（sha256）、`feature_columns`、`label_columns`、`forbidden_columns`、
`refused_feature_columns`、`available_at_policy`、`schema_drift`。`--expected-feature-schema` 跨折/重建钉住特征集。

# A. A股数据底座 / A-share Data Foundation（原 ashare_data_foundation.md）

本文档描述 U0 全宇宙数据底座的实现、供应商能力矩阵、运行命令与验收口径。所有
状态均由**真实产物**推导（capability matrix / coverage matrix / validation report /
PIT interval manifests），不存在硬编码通过的关卡。

## 1. 架构 / Architecture

```
scripts/ashare_capability_probe.py   实网能力与授权探测 -> capability matrix
scripts/u0_security_master.py        证券主表（含退市）  -> security_master.parquet
scripts/u0_acquire_bars.py           可续跑日线采集      -> bars/<provider>/sym_*.parquet
scripts/u0_pit_intervals.py          日历/复权因子/分红/停牌/ST -> pit/*.parquet
scripts/u0_acquire_intraday.py       分钟线（分板块抽样）-> intraday/minute_bars.parquet
scripts/u0_assemble_panel.py         装配唯一复权口径面板 -> panel/daily_bars_raw.parquet
scripts/u0_validate.py               校验与对账           -> validation/validation_report.json
scripts/u0_adjustment_forensics.py   复权口径取证         -> validation/adjustment_forensics.json
scripts/u0_audit.py / u0_bar_readiness.py / u0_pit_readiness.py
                                     三份证书（同一 evidence 模块推导）
```

核心库位于 `src/quantagent/data/ashare/`：

| 模块 | 职责 |
|---|---|
| `symbols.py` | 规范化标识：代码 → 交易所 / 板块 / 证券类型；后缀与代码矛盾时**抛错**而非静默改所属交易所 |
| `contracts.py` | 每个数据族的 schema + 单位 / 复权 / 时区语义 + provenance 列 |
| `http.py` | 按 host 限速、有界重试、失败分类（腾讯用 **501** 表示限流，不是 429） |
| `env.py` | 载入仓库 `.env`（后台任务此前因缺 `TICKFLOW_API_KEY` 直接 KeyError 崩溃） |
| `sources.py` | 真实 provider adapter（TickFlow / 腾讯 / 新浪 / 东财） |
| `acquire.py` | 可续跑、分区落盘、逐次尝试写 ledger 的采集器 |
| `readiness.py` | 由产物推导的 U0 关卡与三份证书 |

## 2. 运行环境事实 / Runtime facts

- **出网仅开放 TCP 80/443**。因此 `baostock`（10030）与 `mootdx`/`pytdx`（7709）
  在本运行环境**不可用**，probe 中记为 `BLOCKED_BY_ENVIRONMENT`。
- `xtquant`（QMT/MiniQMT，Level-2 的常规合法来源）为 Windows 客户端，未安装。
- `TUSHARE_TOKEN` 未配置 → tushare 记为 `NO_CREDENTIAL`。
- `QLIB_PROVIDER_URI_1D` / `_1MIN` 指向的目录**不存在** → qlib 记为 `MISSING_DATA_ROOT`。

## 3. 供应商能力矩阵（实测）/ Provider capability

以 `scripts/ashare_capability_probe.py --allow-network` 生成，
产物：`runtime/data/u0/capability/provider_capability_matrix.{csv,json}` 与 `.md`。

| 数据族 | 实测可用供应商 | 说明 |
|---|---|---|
| 日线（raw / qfq / hfq） | TickFlow, 腾讯 | TickFlow 含 `amount`；腾讯**不提供成交额** |
| 证券主表 | TickFlow `exchanges.get_instruments` | 含 name（带 ST/*ST 前缀）、listing_date、股本、涨跌停价 |
| 退市名单 | SSE / SZSE（经 akshare 包装） | 361 条带退市日 |
| 交易日历 | 新浪（akshare） | 8,797 个交易日，1990-12-19 起 |
| 复权因子 / 公司行动 | 新浪 `hfq.js` | 每个除权事件一条，含生效日；覆盖含退市股 |
| 分红送转 | 新浪 F10 | 含公告日 / 登记日 / 除权除息日 |
| 停牌区间 | 东财停复牌报表（按交易日快照） | 快照带停牌起始日，可折叠为区间 |
| 分钟线 | 腾讯（5/15/30/60 分钟，滚动窗口）、东财（1 分钟趋势） | 仅滚动窗口，无深历史 |
| 实时行情 + 五档盘口 | 腾讯 `qt.gtimg.cn`、TickFlow `quotes` | **Level-1 聚合五档，不是 Level-2** |
| 资金流（大单分类） | 东财 | 该 host 按 IP 限流，失败被记录而非吞掉 |

### TickFlow 授权边界（逐方法实测）

| 方法 | 结果 |
|---|---|
| `klines.get`（日线，单票） | SUPPORTED |
| `instruments.get` / `exchanges.list` / `exchanges.get_instruments` / `universes.list` | SUPPORTED |
| `quotes.get_by_symbols` | SUPPORTED |
| `klines.batch` | UNAUTHORIZED（无日/周/月K线批量查询权限） |
| `klines.intraday` / `intraday_batch` | UNAUTHORIZED（无日内分时查询权限） |
| `klines.ex_factors` | UNAUTHORIZED（无除权因子查询权限） |
| `depth.get` | UNAUTHORIZED（无市场深度查询权限，市场 CN） |
| `financials.*` | UNAUTHORIZED（无公司财务数据查询权限） |

实测限速：**10 请求/分钟**（`RateLimitError 请求频率超限 (10/min)`）。

### Level-2

Level-2（逐笔委托 / 逐笔成交 / 十档）在本环境**没有合法可用通路**：TickFlow 的
`depth.get` 明确返回未授权；公开源（腾讯 / 新浪）只提供五档聚合快照，属 Level-1；
QMT/xtquant 为 Windows 客户端且未安装。因此 Level-2 记为**外部授权阻塞**，
capability matrix 中 `l2_depth` 家族的可用供应商为空，不做任何"已支持"的声明。

## 4. 单位与复权口径 / Units and adjustment

- `volume` = **股**。供应商原始单位**不是**每个供应商一个常数（Round 29 实测）：
  东财日线 = 手；新浪日线 = 股；腾讯日线主板/创业板 = 手、**科创板 688/689 = 股**
  （akshare PR #7328）；akshare ≥ 1.18.69 的 `stock_zh_a_hist_tx` 对所有 `sz000*`
  代码跳过自己的 ×100，所以 000001.SZ 返回的是手。因此 scale（×1 / ×100）由
  `quantagent.data.ashare.units.infer_volume_scale` **按每次响应**判定：只有一个候选
  使 `amount / volume` 落在 `[low, high]`（≥95% 行）才算 verified；否则
  `quality_status = UNIT_AMBIGUOUS`、`volume_unit = shares_unverified`，绝不写成 OK。
  无成交额的响应（腾讯 `fqkline`、akshare 1.18.60 的腾讯形状）只能套用板块先验，且保持未验证。
- `amount` = **CNY**；腾讯 `fqkline` 日线**不含**该列，由 provenance 与 `amount_coverage` 如实记录。
- akshare 只允许 `adjust=""`（raw）。供应商 qfq/hfq 在 canonical 路径上直接拒绝：东财/腾讯
  的 qfq 是**减法**、新浪是**乘法**，且每个新除权日都会改写历史（取数日依赖）。
  已有面板用 `scripts/audit_market_panel_units.py` 审计（价格对 U0 raw 的比值是否在
  除权日跳变 ⇒ qfq-as-of-fetch ⇒ `point_in_time_valid` 被判 REFUTED）。
- akshare 版本是数据契约的一部分：安装版本 ≠ `pyproject.toml` 的 pin 时日线 provider
  直接失败（`QUANTAGENT_AKSHARE_VERSION_OVERRIDE=<installed>` 为显式确认）。
- 面板 `daily_bars_raw.parquet` 为 **raw 未复权**，复权因子单独存放于
  `pit/adjust_factors.parquet`，下游按需应用。

口径由 `u0_adjustment_forensics.py` 用除权日因子回放取证，而不是靠字段声明：

| 面板 | 事件数 | 符号一致率 | 判定 |
|---|---|---|---|
| 新 `u0/panel/daily_bars_raw.parquet` | 1,853 | 0.945 | **RAW** |
| 旧 `full_universe_market_panel.parquet`（全量） | 9,611 | 0.540 | ADJUSTED_OR_MIXED |
| 旧面板 · `source_track=frozen_cohort` | 9,407 | 0.532 | ADJUSTED_OR_MIXED |
| 旧面板 · `source_track=u0_backfill` | 204 | 0.946 | RAW |

即：旧面板把 3,872 只**前复权**证券与 235 只**未复权**证券合并，却在 manifest、
coverage summary 与就绪证书里统一声明 `adjustment_method = "none"`。

## 5. 缺口语义 / Session gaps

新面板**只包含实际成交的交易日**，不再写入 NaN 行。存续期内没有 bar 的交易日
落入 `panel/session_gaps.parquet`，并被分类为：

- `SUSPENDED` —— 命中供应商提供的停牌区间（带停牌原因）；
- `MISSING_UNEXPLAINED` —— 没有停牌记录覆盖；`evidence` 列注明是否落在停复牌
  快照窗口之外。

停复牌快照的覆盖窗口在 `pit/suspension_manifest.json` 里显式给出；窗口之外的
交易日**不会**被当作"无停牌"。

## 6. 操作命令 / Operator commands

```bash
# 0) 实网能力与授权探测（先跑，别直接大规模回填）
AI_quant_venv/bin/python3 scripts/ashare_capability_probe.py --allow-network

# 1) 证券主表（TickFlow 实例列表 + 交易所退市名单 + 冻结主表）
AI_quant_venv/bin/python3 scripts/u0_security_master.py --allow-network

# 2) 日线采集（可续跑；TickFlow 为主，腾讯为回退）
AI_quant_venv/bin/python3 scripts/u0_acquire_bars.py --allow-network \
    --providers tickflow --staging-name tickflow_raw --max-minutes 600
# 第二个 worker 从宇宙另一端收敛，避免与第一个重复排队
AI_quant_venv/bin/python3 scripts/u0_acquire_bars.py --allow-network \
    --providers tencent --staging-name tencent_raw --order reverse \
    --skip-if-in tickflow_raw,legacy --max-minutes 420

# 3) PIT 区间表
AI_quant_venv/bin/python3 scripts/u0_pit_intervals.py calendar   --allow-network
AI_quant_venv/bin/python3 scripts/u0_pit_intervals.py factors    --allow-network --max-minutes 120
AI_quant_venv/bin/python3 scripts/u0_pit_intervals.py dividends  --allow-network --max-minutes 150
AI_quant_venv/bin/python3 scripts/u0_pit_intervals.py suspension --allow-network --start 2012-01-01 --max-minutes 240
AI_quant_venv/bin/python3 scripts/u0_pit_intervals.py st

# 4) 分钟线（分板块抽样，公开源只有滚动窗口）
AI_quant_venv/bin/python3 scripts/u0_acquire_intraday.py --allow-network --per-board 40 --frequency 5

# 5) 装配 + 校验 + 取证 + 三份证书
AI_quant_venv/bin/python3 scripts/u0_assemble_panel.py
AI_quant_venv/bin/python3 scripts/u0_validate.py --allow-network
AI_quant_venv/bin/python3 scripts/u0_adjustment_forensics.py
AI_quant_venv/bin/python3 scripts/u0_audit.py
AI_quant_venv/bin/python3 scripts/u0_bar_readiness.py
AI_quant_venv/bin/python3 scripts/u0_pit_readiness.py
```

**中断与续跑**：采集器按 symbol 分区落盘并写 ledger。重跑同一命令会跳过已完成
分区；`PERMANENT` / `ENTITLEMENT` 类失败不再重试，`TRANSIENT` / `RATE_LIMITED`
会重试。在 staging 同级目录放置 `<staging>.cancel` 可让运行中的任务安全停止。

以上命令同时注册为受管 JobRunner 命令（`probe-ashare-capabilities`、
`build-u0-live-security-master`、`acquire-u0-daily-bars`、`build-u0-pit-intervals`、
`acquire-u0-intraday-bars`、`assemble-u0-raw-panel`、`validate-u0-data`、
`audit-u0-adjustment-forensics`），可在工作站 `/governance` 页面查看真实状态。

## 7. 就绪判定 / Readiness

`FULL_UNIVERSE_DATA_READY` 仅在以下**全部**由产物证实时给出：

1. integration — 五类证据文件齐备；
2. provider — 每个强制数据族都有实测可用供应商，回退供应商**被实际调用过**；
3. identity — 主表覆盖全部板块、含退市股、符号规范化校验通过；
4. coverage — 主表中每只证券都有行情；
5. quality — 校验清单无 FAIL 且无 NOT_RUN；
6. pit — 强制 PIT 字段全部有来源。

任一环节缺证据即 `NOT_READY_*`，训练保持禁用。**不允许**为拿到 READY 而放宽关卡。

## 8. 已知外部阻塞 / Known external blockers

| 项 | 状态 | 原因 |
|---|---|---|
| Level-2（逐笔 / 十档） | BLOCKED | TickFlow 未授权；公开源仅五档 L1；QMT 为 Windows 客户端 |
| 深历史分钟线 | BLOCKED | TickFlow 日内未授权；公开源仅滚动窗口 |
| 历史 ST 区间（沪/北，2026-07-27 之前） | BLOCKED_BY_DATA | 无带日期登记；cninfo 公告重建（ENG_ST30）未通过预注册 HOLDOUT（2026-07-27 起由 ENG_ST31 覆盖）；Fuyao/HiThink、Qlib 均无 ST 历史（2026-11-24 复核）；baostock `isST` 需 10030 端口 |
| 财务报表 / 估值（TickFlow） | UNAUTHORIZED | 需另购权限；akshare/新浪为替代路径 |
| baostock / mootdx | BLOCKED_BY_ENVIRONMENT | 运行环境只放行 80/443 |
