# QuantAgent

QuantAgent 是面向 A 股现实约束的 **Point-in-Time 量化研究、回测与纸面交易系统**。
它不提供投资建议、不承诺收益、不连接真实券商（实盘路径终态为 `LIVE_DISABLED`）。
工程目标是用认证数据、严格样本外、交易成本、风控和纸面账户产出**可审计的证据**——
包括诚实的否定结论。

**当前状态（2026-11-24）：RESEARCH / NOT LIVE READY。** 仓库内没有通过成本与统计闸门的
可交易 edge；详见 [docs/RESEARCH.md](docs/RESEARCH.md) 与 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 文档 / Documentation

| 文档 | 内容 |
|---|---|
| [AGENTS.md](AGENTS.md) | 仓库规则、安全约束、代码规范（人与 AI 协作者必读） |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 实际运行的入口、研究链路、每个概念的权威实现、安全边界 |
| [docs/DATA.md](docs/DATA.md) | 数据血缘、认证 r29 面板、PIT 契约、供应商能力与单位口径 |
| [docs/GOVERNANCE.md](docs/GOVERNANCE.md) | 隔离窗、信任分级、验收规则、forward paper 协议 |
| [docs/RESEARCH.md](docs/RESEARCH.md) | 研究结论摘要、统计工具校准、试验台账（`docs/research/ledgers/`） |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | 环境、定时任务现状、任务控制、纸面账户、Web 工作站运维 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 优先级路线图 |
| [docs/audits/HISTORY.md](docs/audits/HISTORY.md) | 历轮审计的发现与修复 |
| [docs/UI.md](docs/UI.md) | 前端代码地图 |

## 快速开始 / Quick start（Linux）

```bash
AI_quant_venv/bin/python3 -m pip install -e ".[all,test]"
AI_quant_venv/bin/python3 -m pytest tests/ -q          # 约 8 分钟
./scripts/run_quant_ui.sh --runtime "$PWD/runtime" --host 127.0.0.1 --port 8000
```

运行时数据（数据、模型、报告、日志）位于 `runtime/`（`QUANTAGENT_HOME` 可覆盖；
`quantagent.config.paths.quant_paths()` 是单一来源），不进入 git。

## 可信评测 / Trusted evaluation

唯一可信的收益数字入口是规范评估器（默认读取认证 r29 执行面板）：

```bash
AI_quant_venv/bin/python3 scripts/baseline_protocol.py --help
```

任何数字的引用都必须带 {窗口, 试验数 N, 信任分级, artifact 路径}；隔离窗由
`configs/quarantined_windows.json` 强制，见 [docs/GOVERNANCE.md](docs/GOVERNANCE.md)。

## 主要命令 / Main commands

CLI 入口为 `quantagent`（`quantagent --help` 列出全部命令）。研究闭环：

```bash
quantagent run-full-real-training-v7 --market-panel <panel> --labels <labels> ...   # 数据集 → 训练 → 预测 → 目标权重 → 回测
quantagent audit-nonlinear-factors ...      # 治理化线性 vs 非线性对比（隔离窗、PBO、DSR、SPA）
quantagent search-factor-fusion ...         # 因子融合 Pareto 前沿（试验数由搜索空间决定）
quantagent synthesize-factors-v7 ...        # 受限 DSL 因子发现（含显著性门）
quantagent paper-clear-kill-switch --ledger <ops.jsonl> --confirm --author <name> --reason "<理由>"
```

Web 工作站（策略实验室、研究运行、Factor Lab、回测、风控、纸面账户）通过 JobRunner
白名单启动上述命令，不开放任意 shell。

## 安全边界 / Safety boundary

- Agent / LLM 只输出证据、观点、约束与因子表达式；组合构建只输出 `target_weights`；
  只有 `OrderManager` 生成订单意图。
- `QMTGateway` 默认 `dry_run=True`、`live_trading_enabled=False`；实盘意图在路由前被拒绝。
- 生产/真实数据路径禁止合成回退；mock 数据只用于测试。
