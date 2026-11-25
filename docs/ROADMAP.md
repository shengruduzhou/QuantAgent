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
   3,273 个种子全覆盖。**K 线已补齐（Cycle 32，2026-11-25 独立复核）**：2026-08-06→11-24 共 73 个交易日，
   每日 3,264–3,273 只。**但 FRESH 管道仍未恢复**：同日 `orders=FAILED`。已删除函数的调用已修（c4517df）；
   剩余阻塞是 v7 面板以缺行表示停牌，严格模拟器 `missing_execution_bar` 失败关闭（数据工程 wave 2：
   有日期停牌证据的缺行补为显式停牌行，无法解释的不填）。**更大的阻塞**：约 383 只在 FRESH 窗内无证据地
   停止出现（07-24 有 3,447 只、07-27 起 3,275 只；疑为旧 catch-up 以 panel_max 现存代码为种子），需从各自最后
   一根 K 线回补到当前；在此之前面板完整性判 INCOMPLETE。
   另：2026-05-19→08-05 行的 ST 出处为当期快照广播（沪/北 07-27 前无法 PIT 重标）；08-11 起无盲测下单记录。
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
   **Cycle 32 状态（不改数字的部分已完成）**：rank-IC 13 处委托 `cross_sectional_rank_ic`、绩效 6 处委托
   `sharpe_ratio`/`max_drawdown`（冻结拷贝等价测试 ≤1e-12）。**仍需预注册重基线才能统一的差异**（数值例）：
   - `paper_report` Sharpe 用 ddof=0：n=20 时比 ddof=1 高 2.6%；回撤不含初始资金点：初始 100 万、NAV 90/95/96 万
     → 报 0% 而真实 −10%（`strict_v8` 报 10%）。
   - `backtest/engine.py` 年化指数 252/len(nav)（应为收益数）：11 点 NAV 1.00→1.10 报 +788% 而非 +1004%。
   - `v7_experiment`/`model_comparison` 年化 `(1+日均)^252`：同一 252 日序列 −22.0% vs 几何 −25.2%。
   - `beta_decomposition` 用 244 日、其余 252：Sharpe 差 1.6%。`v7_experiment` Sharpe 分母 `std+1e-12`、退化时 0.0。
   - `v7_experiment._rank_ic` 先排名后丢 NaN 标签（例：0.832 vs 0.800）且退化日记 0.0 计入均值（例：0.5 vs 1.0）。
   - `fusion.factor_ic_series` 常数因子截面：n=7 时得 5.6e-17（计入均值），n=8 时得 NaN。
   - `information_coefficient` 的 HAC t 按输入行顺序而非日期排序计算自协方差（非重叠 t 已按日期排序）。
   - 成本：`backtest/engine.py`/`CostModelConfig` 印花税固定 5 bps（2023-08-28 前少收 5 bps）；佣金 2.5/3.0 bps 并存；
     冲击三处默认一致（10·√p bps），未见 100× 差异。做T 3 份涨跌停拷贝与 `market_rules` 不等价（无日期、920xxx）。
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
   **Cycle 32 已完成**：`quant_math.performance.nonoverlapping_cohort_t_stat`（h 个错位 cohort t 的均值，未定义日以 NaN
   占位保持对齐；Richardson & Smith 1991）；`ICSummary.(rank_)nonoverlap_t_stat`；`governance_metrics` 新闸门
   `min_nonoverlap_rank_t_stat=3.0`、`lifecycle` 的 validated 需 ≥3（NaN 不通过）；RD-Agent 门委托且数值不变。
   零假设校准（AR(1) φ=0.98 因子、600 面板×750 日×50 名，固定种子）：|t|>1.96 拒绝率 h=5 HAC 7.7% vs 非重叠 3.2%，
   h=20 HAC 11.8% vs 3.3%；非重叠 |t|≥3 为 0.17%。**未决（需预注册 + 重基线后才能改）**：RD-Agent 门
   （`factor_synthesis._nonoverlapping_ic_tstat`）的 IC 序列先删未定义日再分 cohort（缺日使后续日串入相邻 cohort），
   且无合格 cohort 时以 0.0 代替 NaN；改正会改变门的数字，本分支未改。
   注意：非重叠 t 只在"仅重叠"零假设下方差 ≤1；因子/收益本身有持续性时任何非重叠 t 都会过大，不得当作 N(0,1)
   喂给 p 值或 DSR。
   - `information_coefficient` 的 `_nonoverlap_t` 按输入中出现的日期分 cohort：某交易日整日无行仍会串组；有交易所日历时应先 reindex 到日历。
   - `multi_horizon_blender` 现要求 ≥3 个完整配对（代码审查指出与旧实现非逐位一致，但更正确）。
   - 标签期限无法推断时 `information_coefficient` 的非重叠 t 为 NaN，`lifecycle` 的 validated 不可达（设计上 fail-closed）。
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
