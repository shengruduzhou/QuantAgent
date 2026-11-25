# 路线图 / Roadmap（Round 31，2026-11-24）

依据：Round 31 六岗审计与复现（证据见 `runtime/archives/audits/round31/`）、
[RESEARCH.md](RESEARCH.md)、[GOVERNANCE.md](GOVERNANCE.md)、[DATA.md](DATA.md)。
原则：**先让结论可信，再追求结论更好；不为增加功能而扩张系统。**

## 现在能可靠做到的 / What works today

- 认证 r29 全宇宙数据（原始价、分红送转、三态涨跌停/停牌、退市、隔离窗干净）+ 规范评估器
  （variant C：T+1、法定印花税日程、整手、参与率、滑点、红利税、PSR/DSR/自助法 CI）。
- 统计工具中 DSR / PSR / SPA 校准良好；隔离窗守卫覆盖评估器、模型对比、融合搜索、RD-Agent。
- 纸面账户：事件溯源、规范账本、风控接线（日亏停买、回撤只减仓锁可审计解除）、分红送转入账。
- 安全边界：`LIVE_DISABLED` 终态，订单只经 `OrderManager`。
- 工作站：研究运行的启动/监控/裁决/对比，缺失证据显示为缺失。

## 不能做到的 / What it cannot do

- 没有任何成本后可交易 edge（透明量价组合 −11.73%/年 vs 等权 +6.89%）。
- FRESH 窗没有在积累数据（管道自 2026-08-05 停摆）⇒ 不存在可用的最终 holdout。
- "生产 FT-Transformer" 只是历史产物（前视宇宙 + 被证伪面板 + likely_overfit）。

## P0 — 解除阻塞 / Unblock（需要用户决定）

1. **FRESH 窗 ST 证据：已解除（2026-11-25）。** ENG_ST30（十年公告重建）HOLDOUT 未过；ENG_ST31（以 2026-07-26 带日期
   全市场快照为锚的前向登记）预注册验证一次通过（深市 235,036 行 0 错、19/19 变更同日）⇒ 沪/北 2026-07-27 起有证据，
   3,273 个种子全覆盖。**剩余动作**：确认 06:00 catchup 实际补齐 2026-08-06→ 的 K 线（耗时数小时，看当日日志）；
   年底前延长交易日历；每日监控若撤下沪/北行需重新预注册。
2. **让旧生产 blend 退出决策路径。** `configs/production_blend.json` 已标注
   `historical_unvalidated`；fresh_blind 仍用它打分。候选模型只在 r29 认证数据上重训。

## P1 — 研究有效性 / Research validity

1. **E1 类组合的成本后经济检验（决定模型工作是否有意义）。** 预注册：截面 rank 目标的 ridge
   （158 个认证量价特征）、long-only top-50 周频、variant C、2019-01 → 2025-08-29 年度扩窗重训。
   通过：净超额 CAGR 的平稳自助法 95% CI 下界 > 0，DSR（累计 N ≥ 135）≥ 0.95，6 年中 ≥ 4 年为正。
   试验数 1（加 E1 设定 GBM 为 2）。不通过 ⇒ 日频量价 long-only 方向停止，转向数据（基本面/一致预期）。
2. **统一标签口径为 delay-1。** `build-labels-v7` 的 delay-0 使短周期 IC 虚增最多 53%；
   给标签产物加口径戳并拒绝混用。**Cycle 32 工程部分已完成（待独立审计）**：所有标签产出者写口径戳
   （parquet schema metadata / CSV sidecar，`data.label_contract`）；`build-labels-v7` 必须显式
   `--label-convention delay1|delay0`（delay1 = 全局交易日钟，须 `--market-calendar`；**不改默认值**：原隐式 delay-0 若静默改为 delay-1，会改变重跑旧命令
   得到的数字，故改为拒绝猜测；工作台"重建 Labels"显式传 delay1）；数据集构建器、`synthesize-factors-v7`
   及两个评估脚本拒绝无戳标签（`--allow-legacy-unstamped-labels` 显式放行并记为 legacy）并拒绝混用
   （钉住 schema 的口径不同 / 行情面板自带标签列）。剩余：用 delay-1 重建尚在使用的 v7 delay-0 产物。
3. **收敛重复实现并做一次预注册重基线。** 交易成本 5 份（佣金 3.0 vs 2.5 bps；冲击系数相差 100×）、
   绩效指标 8 份、rank-IC 约 10 份（最小截面数不一致）、做T 路径的 3 份涨跌停拷贝。
   以 `execution/cost_model`、`quant_math/performance`、`factors/evaluation`、`market_rules/ashare`
   为唯一实现。冲击定律改为波动率缩放会改变规范数字，需预注册并经用户批准。
4. **持久化多重检验台账。** 接到 `audit-nonlinear-factors`、`search-factor-fusion`、RD-Agent；
   DSR 一律用同族累计 N。**Cycle 32 已完成（待独立审计）**：`research.trial_ledger`
   （`$QUANTAGENT_HOME/state/trial_ledger.jsonl`，append-only、哈希链、跨进程锁、校验失败即拒绝计数）；
   三条命令在计算**之前**于同一把锁内读取前序 N 并预留本次试验（崩溃的运行照样计数，并发运行不会互相漏计），
   逐试验结果入账（含对照、失败与拒绝），DSR 用 `历史基线 + 已入账 + 本次`；`prior_family_trials` 为必填参数。
   **仍未完成（open）**：GP 符号回归 / LLM 公式脚本 / 非 RD 的 `synthesize_factors`（`--legacy-ga`）各族尚未入账；
   Stage-4 `ExperimentLedger`（无 CLI 调用方）与 `TrialLedger` 待统一为一个台账；台账被截断/整体删除后仍能校验通过
   （计数可下降）——需要外部高水位标记；RD-Agent `begin_run(planned_trials=0)` 在并发时不预留，存在竞态。
5. **退役 v7 银层面板。** 约 89 处引用改为经 `quant_paths` 的单一常量指向 U0/r29，随后归档。
6. **重叠标签 IC 的 t 值**改用非重叠 cohort（`factors/evaluation`）；筛选用 |t| ≥ 3。
7. **纸面 ↔ 回测摩擦一致。** 纸面目前按收盘价零滑点成交；需要统一成本/冲击模型，并让纸面风控
   对可成交限价的名义金额留容差（直接改限价会让恰好 10% 权重的订单被单票上限拒绝）。
8. **E2（非线性设定消融）**在 P1-1 之后、按修订后的预注册（加入 rank 目标线性对照、选择折延长到 2019）。

## P2 — 工程 / Engineering

1. **62 个无引用的 CLI 命令**及只由它们到达的 51 个模块（themes/*、fundamental/*、models/v7_*、
   services/v7_pipeline_service、backtest/engine.py，约 13K 行）：先隐藏 + 弃用警告，确认无人手工使用后删除。
2. v7/v8 后缀的 CLI 模块改为领域名并保留别名（JobRunner 命令 ID 不变）。
3. 因子计算改为直接输出宽表 float32（长表 object 列占用约 33×内存）。
4. 单一证券主表加载器（现有 4 份，8 个 U0 脚本读的是没有退市日期的那份）；北交所单位错误簇隔离；
   行业图改为 PIT。
5. UI：选择运行（selection runs）没有生产者，相关页面要么接上要么删除；模型 → 回测 → 纸面的跳转；
   约 7.9K 行遗留全局 CSS 逐类清理；market_playbooks v1/v2/v3 合并。
6. cron：删除从未运行的做T 资金流行与空转的回填行（用户 crontab）；决定 stage10 去留。
7. `runtime/` 约 40 GB 可再生/过期数据的清理清单（需用户批准，见归档目录）。
8. 清理 76 个非祖先远程分支（需用户执行）。

## 冻结或放弃 / Freeze or abandon

| 组件 | 证据 | 决定 |
|---|---|---|
| RL（PPO） | 预注册 44 次运行全部为负 | 冻结：保留 harness，默认不启用 |
| LLM / 多 agent 交易推理 | 无任何可测增益 | 只保留证据摄取与因子表达式提案；不进入决策 |
| RD-Agent LLM 因子循环 | 接受门曾接受 48% 噪声 | 门已收紧；在有独立确认窗之前其产物不得进训练 |
| 做T、打板 | 无可实现 edge / 逆向选择 | 放弃 |
| v8.9 FT sleeves | 前视宇宙 + 被证伪面板 | 历史产物，不再使用 |
