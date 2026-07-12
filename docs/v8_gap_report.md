# V8 Gap Report — Superseded

原始报告生成于 **2026-05-29**，针对 `v8.2` working tree。报告中的大部分 Gap、测试数量、模块状态和生产路径已经失效，继续保留 400 多行旧结论会误导当前维护者。

## 当前状态

- 本路径仅作为旧引用的稳定跳转点保留。
- 生产与研究边界：见根目录 `AGENTS.md`、`ARCHITECTURE_AUDIT.md` 和 `ACCEPTANCE_RULES.md`。
- 可信评测窗口：见 `configs/quarantined_windows.json`、`BASELINE_TRUST_CLASSIFICATION.md` 和 `EVALUATION_PROTOCOL_V2.md`。
- 当前生产配置及其 trust class：见 `configs/production_blend.json`。
- 数据、因子、模型、组合、执行和治理的新结构：见 `docs/quantagent_governance_architecture.md`。

## 历史结论处理

原报告不是当前能力清单，也不能作为 production-readiness 证据。需要追溯 2026-05-29 时点时，应通过 Git 历史读取本文件旧版本，而不是把旧内容继续暴露在默认文档面。
