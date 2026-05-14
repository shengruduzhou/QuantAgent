# QuantAgent V7：A 股多 Agent 研究与风控系统 / A-share Multi-Agent Research OS

QuantAgent V7 是面向 A 股散户现实约束的主题研究、基本面筛选、多周期 Alpha、组合风控、回测和模拟执行系统。它不是投资建议系统，也不是真实券商自动下单系统；默认只做 research、backtest、paper / virtual trading、audit replay 和风险提示。

## 目标 / Goal

V7 主链是 `Data Providers -> Point-in-Time Evidence OS -> Policy & Theme Agents -> Industry Chain Graph -> Thematic Universe -> Fundamental / Fraud / News Credibility Agents -> Multi-Horizon Alpha -> Factor Applicability -> Market Regime -> Portfolio Construction -> Hedge Decision -> Risk Gate -> A-share Execution Simulation -> OrderManager -> VirtualBroker -> Audit`。

核心升级是把 V6 的价格、因子、Agent evidence 和安全执行主线，扩展为政策红头文件、产业链、动态主题股票池、公司基本面可信度、新闻可信度、多周期 Alpha、A 股 T+1 / 涨跌停 / 停牌 / 流动性约束和可审计归因的闭环。

## 安全边界 / Safety Boundary

- 默认不连接真实券商，默认使用 `VirtualBroker`。
- `LLM / agent` 只能输出 `EvidenceRecord`、score、view、constraint、risk flag、audit log，不能输出 order。
- 模型只能输出 alpha、confidence、prediction interval、risk penalty 和 factor contribution。
- Optimizer / Portfolio Construction 只能输出 `target_weights`。
- 只有 `OrderManager` 可以把 `target_weights` 转成 order intents。
- `Risk Gate`、`Kill Switch`、execution constraint simulation 和 reconciliation 必须在任何 QMT submit path 前完成。
- 真实券商接入必须显式配置 `live_trading_enabled=true` 且 `dry_run=false`，并保持默认关闭。

## 文档 / Docs

- V7 架构与 Agent 接口：`docs/V7_系统架构与Agent接口.md`
- V7 算法、风控、回测与验收：`docs/V7_算法风控回测与验收.md`
- V6 历史设计文档仍保留在 `docs/V6_*.md`，用于兼容现有测试和回溯旧架构决策，不作为新开发主入口。

## 快速验证 / Quick Validation

```powershell
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m pytest tests/
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m compileall src
git diff --check
```

## V7 CLI / 命令入口

```powershell
quantagent validate-v7 --config configs/v7.default.yaml
quantagent run-daily-v7 --config configs/v7.default.yaml --date 2026-05-14 --output-dir reports/v7
quantagent run-daily-v7 --config configs/v7.mock.yaml --date 2026-05-14 --output-dir reports/v7
```

`configs/v7.default.yaml` 默认是 `strict_local`，缺少 PIT policies、base_universe、market_state 或 company-theme 数据时会拒绝 synthetic fallback，避免把 mock research 伪装成真实研究。`configs/v7.mock.yaml` 才使用 deterministic synthetic inputs，用于离线 smoke test、Theme Discovery、industry chain graph、dynamic thematic universe、multi-horizon alpha、A-share execution constraints、Risk Gate 和 Audit Log 验证；两种模式都不会生成真实交易订单。

## 数据与远程抽取 / Data And Remote Extraction

V7 支持显式 `online` 数据模式接入 policy/news/disclosure/TradingView public pages、Qlib、AkShare 和 TuShare provider。网络调用默认关闭，必须配置 `allow_network=true`；TradingView public pages 只作为 sentiment context，不作为官方行情或基本面真值。

`policy_extraction` 是 OpenAI-compatible remote schema extraction seam，默认 `enabled=false`。启用后它只把红头文件、公告和新闻抽取成 `EvidenceRecord`、theme、sub-theme、chain nodes、confidence 和 risk flags，不允许输出 order 或 trade advice。

## 配置 / Config

V7 默认配置入口是 `configs/v7.default.yaml`。V6 的 `configs/v6.default.yaml`、provider、risk limits、replay scenarios 和 model configs 仍可作为现有实现基础，但新功能应优先向 V7 schema、DAG 和 Agent contract 对齐。
