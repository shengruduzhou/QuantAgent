# A股数据底座 / A-share Data Foundation

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

- `volume` = **股**（腾讯/东财/新浪的日线以"手"计，adapter 在边界 ×100）。
- `amount` = **CNY**；腾讯日线**不含**该列，由 provenance 与 `amount_coverage` 如实记录。
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
| 历史 ST 区间 | BLOCKED_BY_DATA | 无带日期的简称变更历史来源（cninfo 曾用简称无日期；baostock `isST` 需 10030 端口） |
| 财务报表 / 估值（TickFlow） | UNAUTHORIZED | 需另购权限；akshare/新浪为替代路径 |
| baostock / mootdx | BLOCKED_BY_ENVIRONMENT | 运行环境只放行 80/443 |
