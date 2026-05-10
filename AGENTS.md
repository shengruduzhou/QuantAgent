# QuantAgent V4 Codex 指令 / Codex Guide

## 项目目的 / Project Purpose

QuantAgent V4 的目标是把现有 V3/SOTA research repo 升级为面向大 A 市场的 AI Quant OS。系统覆盖 point-in-time feature store、三塔预测模型、factor governance、agent evidence views、portfolio optimizer、A-share event-driven backtest，以及 QMT execution-preparation。

本仓库不是玩具 LLM trading demo，也不提供投资建议 / financial advice。所有研究输出必须经过测试、回测、风控和 dry-run 执行准备。

## A 股安全约束 / A-share Safety Constraints

- No live trading by default：默认禁止实盘交易。
- QMT dry-run default：`QMTGateway` 必须默认 `dry_run=True`。
- Agents never emit orders：LLM / Agent 只能输出 structured evidence、views、constraints、confidence、audit logs。
- Optimizer never emits orders：Optimizer 只能输出 target weights，不允许输出 broker orders。
- OrderManager converts target weights into order intents：订单意图必须在 `OrderManager` 中生成。
- RiskGate and KillSwitch before QMT submit：任何 QMT submit path 前必须通过 risk gate、kill switch、reconciliation。
- Optional dependencies degrade gracefully：`xtquant`、`cvxpy`、heavy NLP models 不存在时，imports 不应破坏研究流程。

## Code Style / 代码规范

- Python code、comments、docstrings、variable names、test names、config keys 必须使用 English。
- 保持 V3/SOTA behavior，除非 V4 明确替换。
- 不删除已有 SOTA components；优先做 wrappers、adapters、integration seams。
- 面向本地有限算力设计：small PyTorch configs、CPU test mode、deterministic synthetic fixtures。
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
python -m pytest tests/
python -m pytest tests/test_ashare_rules_v4.py tests/test_v4_services.py
```

如果本地只暴露 Python launcher：

```powershell
py -3.12 -m pytest tests/
```

## Execution Boundary / 执行边界

默认 V4 offline synthetic flow：

```text
build synthetic panel
-> compute features and labels
-> train tiny V4 model
-> infer alpha
-> map agent evidence to AgentView
-> blend alpha and views
-> optimize target weights
-> run A-share backtest
-> generate dry-run order intents
```

任何 live order path 必须显式配置 `live_trading_enabled=true`、`dry_run=false`，并通过 kill switch、risk gate、broker reconciliation。
