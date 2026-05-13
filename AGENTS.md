# QuantAgent V7 Codex 指令 / Codex Guide

## 项目目的 / Project Purpose

QuantAgent V7 的目标是把现有 V6 production scaffold 升级为面向 A 股散户现实约束的 AI Quant OS。系统覆盖 Point-in-Time feature store、Evidence OS、政策与主题发现、产业链图谱、动态主题股票池、基本面尽调、Financial Fraud Risk、News Credibility、多周期 Alpha、Factor Applicability、portfolio optimizer、A-share event-driven backtest、QMT execution-preparation、Risk Gate、Kill Switch 和 Audit Replay。

本仓库不是玩具 LLM trading demo，也不提供 financial advice。所有研究输出必须经过测试、回测、风控和 dry-run 执行准备。

## A 股安全约束 / A-share Safety Constraints

- No live trading by default：默认禁止实盘交易。
- QMT dry-run default：`QMTGateway` 必须默认 `dry_run=True`。
- Agents never emit orders：LLM / Agent 只能输出 structured evidence、views、constraints、confidence、risk flags、audit logs。
- Optimizer never emits orders：Optimizer / Portfolio Construction 只能输出 `target_weights`，不允许输出 broker orders。
- OrderManager converts target weights into order intents：订单意图必须在 `OrderManager` 中生成。
- RiskGate and KillSwitch before QMT submit：任何 QMT submit path 前必须通过 risk gate、kill switch、execution constraint simulation 和 reconciliation。
- Optional dependencies degrade gracefully：`xtquant`、`cvxpy`、heavy NLP models 不存在时，imports 不应破坏研究流程。

## Code Style / 代码规范

- Python code、comments、docstrings、variable names、test names、config keys 必须使用 English。
- 保持 V3/V4/V5/V6/SOTA behavior，除非 V7 明确替换。
- 不删除已有 SOTA components；优先做 wrappers、adapters、integration seams。
- 面向本地有限算力设计：small configs、CPU test mode、deterministic synthetic fixtures。
- 新功能必须配套测试，优先使用 small synthetic panels。

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

默认 V7 offline synthetic / replay-safe flow：

```text
build PIT data panel
-> normalize EvidenceRecord
-> discover policy themes and lifecycle
-> build industry chain graph
-> build dynamic thematic universe
-> score fundamentals, valuation, fraud risk, news credibility
-> generate 1D/5D/20D/60D/120D/126D alpha
-> validate factor applicability
-> construct sleeve allocation and target_weights
-> decide hedge / cash buffer
-> pass Risk Gate and Kill Switch
-> simulate A-share T+1, limit-up/down, suspension, liquidity and lot constraints
-> OrderManager generates dry-run order intents only after approved target_weights
-> VirtualBroker / replay / audit
```

任何 live order path 必须显式配置 `live_trading_enabled=true`、`dry_run=false`，并通过 kill switch、risk gate、broker reconciliation 和 audit replay。
