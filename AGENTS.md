# QuantAgent V7 Codex 指令 / Codex Guide

## 项目目的 / Project Purpose

QuantAgent V7 是面向 A 股散户现实约束的 PIT 量化研究系统。覆盖：

- Point-in-Time financial provider (TuShare / AkShare) + local Parquet cache
- Evidence OS（每条证据都带 source / published_at / available_at / hash）
- 政策红头文件解析与主题发现
- 证据驱动的动态产业链图谱（默认禁用静态模板回退）
- 动态主题股票池（core / strong / satellite / watchlist / exclusion）
- 基本面尽调、Financial Fraud Risk、News Credibility、Intrinsic Valuation
- 多周期 Alpha (1 / 5 / 20 / 60 / 120 / 126 天)
- Factor Applicability Hard Gate (walk-forward, 不通过的因子直接退出 Deep Alpha)
- Walk-Forward Sleeve Allocator (long / medium / short / hedge / cash)
- A-share event-driven backtest
- QMT execution-preparation, Risk Gate, Kill Switch, Audit Replay

仓库不是玩具 LLM trading demo，不提供 financial advice。研究输出必须经过测试、回测、风控和 dry-run 执行准备。

## A 股安全约束 / A-share Safety Constraints

- No live trading by default：默认禁止实盘交易。
- QMT dry-run default：`QMTGateway` 必须默认 `dry_run=True`。
- Agents never emit orders：LLM / Agent 只能输出 structured evidence、views、constraints、confidence、risk flags、audit logs。
- Optimizer never emits orders：Optimizer / Portfolio Construction 只能输出 `target_weights`，不允许输出 broker orders。
- OrderManager converts target weights into order intents：订单意图必须在 `OrderManager` 中生成。
- RiskGate and KillSwitch before QMT submit：任何 QMT submit path 前必须通过 risk gate、kill switch、execution constraint simulation 和 reconciliation。
- Optional dependencies degrade gracefully：`xtquant`、`cvxpy`、heavy NLP models、tushare、akshare、torch 不存在时，imports 不应破坏研究流程。
- PIT enforcement is not optional in strict mode：`enforce_pit_fundamentals=true` 时 `available_at > as_of_date` 的财报行必须被丢弃。

## Code Style / 代码规范

- Python code、comments、docstrings、variable names、test names、config keys 必须使用 English。
- 优先做 wrappers、adapters、integration seams，不删除仍被引用的 SOTA components。
- 面向本地有限算力设计：small configs、CPU test mode、deterministic synthetic fixtures。
- 新功能必须配套测试，优先使用 small synthetic panels。
- 任何新增的财务字段必须同时定义 `report_period`、`ann_date`、`available_at`，否则无法进入 PIT 缓存。
- Industry chain edges 必须有证据来源，禁止把共现直接当成产业链因果。

## Markdown Rule / Markdown 中英混写规则

所有 `.md` 文件必须 Chinese-English mixed：

- 每个 section 以中文说明为主，保留关键 English terms。
- Headings should be bilingual where useful，例如 `## 数据层 / Data Layer`。
- 不允许新增纯英文或纯中文 Markdown 文档。
- Code blocks 可以 English-only，因为它们是代码或配置示例。

## Testing Commands / 测试命令

首选命令：

```powershell
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m pytest tests/
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m compileall src
git diff --check
```

## Execution Boundary / 执行边界

默认 V7 offline strict_local 或 mock smoke flow：

```text
build PIT data panel (strict required tables)
-> apply PIT financial filter
-> normalize EvidenceRecord
-> discover policy themes and lifecycle
-> reason industry chain from evidence (no template fallback)
-> build dynamic thematic universe
-> score fundamentals, valuation, fraud risk, news credibility
-> generate 1D / 5D / 20D / 60D / 120D / 126D alpha
-> validate factor applicability and hard-gate non-production factors
-> walk-forward sleeve allocation (when sleeve return history is available)
-> construct portfolio + target_weights
-> decide hedge / cash buffer
-> pass Risk Gate and Kill Switch
-> simulate A-share T+1, limit-up/down, suspension, liquidity and lot constraints
-> OrderManager generates dry-run order intents only after approved target_weights
-> VirtualBroker / audit
```

任何 live order path 必须显式配置 `live_trading_enabled=true`、`dry_run=false`，并通过 kill switch、risk gate、broker reconciliation 和 audit replay。

## Provider 责任 / Provider Responsibilities

- Qlib 只负责：行情、技术因子、label 生成、训练数据切片、回测底座。
- TuShare / AkShare 只负责：财务报表、财务指标、估值字段、公告披露日期。
- TradingView public pages 只负责：sentiment / attention context，不作为基本面或行情真值。
- 政策、公告、新闻原文必须保留 `source`、`published_at`、`available_at`、`hash`、`confidence`。
