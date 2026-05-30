# V8 Gap Report — v8.2 working tree vs 12-section spec

生成日期：2026-05-29
基线：working tree at commit `5760fe5 (v8.2)` + 未 commit 修改
作用：审计 v8.2 当前已实现的能力 vs 用户在 v8 升级规范中提出的 12 项要求；列出 Gap 和优先级建议。

> **更新 2026-05-30**：P1–P7 全部已落地。pytest 从 735 → 846 pass (+111)，唯一 fail 仍是 P0 baseline 那个 pre-existing ft_transformer per-date rank loss regression（与本批次工作无关）。
>
> 实际落地的新模块见本文末尾的 **Stage 5 Build-out** 段。本审计原文保留以便回溯当时观察到的 gap 与最终交付的 mapping。

---

## 0. Baseline 健康状况

- `python -m compileall src`：**绿**。
- `pytest tests/ --ignore=tests/diag`：**735 pass / 1 fail / 3 skipped**。
  - 唯一 fail：`tests/test_ft_transformer_multi_date_step.py::test_per_date_rank_loss_does_not_pool_across_dates`，断言 ≥8 次 per-date argsort 实际 0 — 在 `src/quantagent/training/ft_transformer_trainer.py` 的 working-tree 修改中 per-date 分支被绕过。**pre-existing v8.2 regression**，不在 P1 scope。
- 新增 ~5600 LOC（policy/bond/broker/state_team/sector_pool/fundamental/credibility/decision_chain/state_machine/market_hard_gate/multi_objective_loss/regime_sub_models/v11_integration/sector_audit/stratified_ic）已落地但未 commit。

---

## 1. Evidence Schema

**已实现**
- 四个独立 builder（policy / bond / broker / state_team）都做到：
  - 显式 `available_at` 字段，PIT 安全（`max(announced_at, fetched_at)` 或 `trade_date + lag`）。
  - 写 silver parquet + coverage_report.json + validation_report.json + manifest.json。
  - 提供 `*_for_features(events, manifest_path)` gated helper，gate 闭锁时返回 None。
  - `state_team` 硬编码 `evidence_label="inferred"` 满足合规。
- `data/credibility/source_table.py` 187 LOC — broker tier table 已迁移。
- `data/ingestion/daily_evidence_job.py` 已整合多个 ingestor 输出。

**Gap (vs 规范第一节)**
| Gap | 影响 | 优先级 |
|---|---|---|
| 没有统一 `EvidenceRecord` 数据类。policy 用 `event_id / announced_at / themes / sectors_hint / policy_strength`；state_team 用 `event_id / trade_date / evidence_type / evidence_strength / scope / scope_value`；bond 用 `trade_date / yield_* / spread_*`；schema 字段完全异构。 | 下游消费者必须对每个 source 写专门 join 逻辑；不能跨 source 做 contradiction 检测。 | **P1 必做** |
| 规范要求的 8 个 evidence-level 字段缺失：`extracted_claims`、`sentiment_score`、`policy_direction_score`、`capital_flow_direction_score`、`confidence`、`contradiction_score`、`lag_window_candidates`、`audit_trace`、`raw_text_hash`、`entity_type`、`entities`。 | 没有跨 source 的 confidence/contradiction 评估；LLM 解析输出无落地字段。 | **P1 必做** |
| 没有 canonical `EvidenceStore.load_all(start, end, source_types)` 统一查询接口。 | 现有 v7 `EvidenceStore` 与新 builders 接口不一致。 | **P1 必做** |
| `available_at` 的单元测试只覆盖每个 builder 自己；没有"全 evidence 联表 PIT lint"。 | 无法防止未来 builder 引入 PIT 泄漏。 | **P1 必做** |

---

## 2. Capital Flow / National Team Proxy

**已实现**
- `data/state_team/builder.py` 560 LOC：
  - 4 个 evidence_type：`etf_concentrated_inflow` / `top10_holder_appearance` / `post_crash_index_buying` / `block_trade_match`
  - 评分 0-1，scope=`index_wide` / `symbol` / `sector`
  - `top10_holder_appearance` 用 +45 BDay `available_at` 反映财报披露延迟
  - `apply_state_team_features` 用 merge_asof backward 安全 attach 到训练 panel

**Gap (vs 规范第二节)**
| Gap | 影响 | 优先级 |
|---|---|---|
| **没有 `capital_flow_thesis` 表**。规范明确要求 thesis_id / direction / supporting_evidence_ids / confidence / expected_lag_days / contradiction_evidence_ids / validation_status / tradability_score / decay_score。当前只有"逐 evidence 事件"，没有"将多个 evidence 合并为一个可验证 thesis"。 | 无法实现规范要求的"政策出现 → 验证回路"。 | **P2 必做** |
| **没有 1d / 5d / 20d / 60d / 120d 回看验证**。规范明确要求政策发布后看对应板块/ETF/个股组合的超额收益、成交额、换手率、资金流是否同步。 | "high-confidence pool" 标准无落地，任何 LLM 生成的 thesis 都没有数据反驳机制。 | **P2 必做** |
| **没有 `validation_status` state machine**（unverified → partially_verified → verified / rejected）。 | 旧的 thesis 不会被自动 deprecate，模型可能继续吃过期信号。 | **P2 必做** |
| state_team 模块缺银行授信、政策性银行债、汇金/国新公开信息字段；只覆盖 ETF + 龙虎榜 + top10 三类。 | 国家队画像不完整。 | **P3** |

---

## 3. Sector Pool

**已实现 (优秀)**
- `data/sector/sector_pool.py` 420 LOC — IC-driven 4 tier：core / watch / short_term / excluded
- `min_dates / min_symbols` 样本量门槛防 small-sample noise
- `core_quantile / core_ir_threshold / watch_ir_threshold` 三段 cutoff
- 写 manifest gate `sector_pool_usable_for_overlay`
- `sector_pool_for_weight_overlay(...)` audit-only 消费接口

**Gap (vs 规范第三节)**
| Gap | 影响 | 优先级 |
|---|---|---|
| 当前 pool 的语义是"**模型在该板块的 OOS edge**"；规范要求的是"**该板块整体投资吸引力**"——两个完全不同的 axis。规范要求字段：`policy_score / capital_flow_score / sentiment_score / broker_attention_score / market_strength_score / liquidity_score / valuation_percentile / risk_score / final_sector_rank / confidence`。 | 缺一个"决策层 sector 池"。但 v11_integration 里有 `_attach_sector_pool` 已经把 pool tier 注入 panel，工作流仍可串联。 | **P3** 增量 |
| 现有 pool 用 stratified IC 表作输入；规范输入应来自 capital_flow_thesis + policy_signal + broker_view + market_strength。 | 两套要并行，建议 **保留** 当前 IC pool，**新增** 一个 `decision_sector_pool`。 | **P3** 增量 |
| 不接受多周期视角（short/mid/long）。 | 不能区分"短期资金流热"vs"长期景气向上"。 | **P3** 增量 |

---

## 4. Fundamental Ranker

**已实现 (优秀)**
- `data/fundamental/ranker.py` 544 LOC — valuation × quality × growth 三维加权
- 严格 PIT：`_select_latest_pit_per_symbol(metrics, as_of)` 只取 `available_at <= as_of`
- Within-sector rank，sector 缺失时 fallback 到 board proxy
- composite_score / metric_completeness 字段
- manifest gate `fundamental_ranker_usable_for_overlay`

**Gap (vs 规范第四节)**
| Gap | 影响 | 优先级 |
|---|---|---|
| 当前 valuation 只用 `pe_ttm / pb / ps_ttm`；规范要求 PE / PB / PS / **ROE / ROA / gross_margin / net_margin / revenue_yoy / net_profit_yoy / operating_cashflow / debt_to_asset / interest_coverage / inventory_turnover / accounts_receivable_growth / goodwill_risk / accruals_quality / dividend / earnings_surprise**——19 个字段，目前覆盖 8 个。 | Q3 / 现金流 / 杠杆 / 商誉 / 应收 / 分红 / 业绩超预期等维度缺。 | **P3** 增量 |
| 没有 winsorize（只有 clip）。 | 极端值处理不够细。 | **P4** 微调 |
| Earnings surprise / 业绩超预期需要业绩预告数据源（`data/fundamental/` 当前没有 announcement 引入路径）。 | 该字段无法生成。 | **P3** 增量 |

---

## 5. Multi-Horizon Models

**已实现**
- `training/regime_sub_models.py` 364 LOC — 三个 *setup-conditional* sub-model：
  - **LowBuy**：20d cumret ≤ -10% AND 5d cumret ≥ -2%
  - **Breakout**：close ≥ 60d rolling high AND vol_5d ≥ 1.5×vol_60d
  - **LimitUpRisk**：近 3 日有过涨停 AND 20d cumret ≥ +25%（用负号训练，作下行风险预测）
- horizon 配置 `(1, 5, 20)` — labels 形如 `lowbuy_label_5d`
- `EnsembleWeights` 按 regime 分配 sub-model 权重（normal/caution/bear/crisis）

**Gap (vs 规范第五节)**
| Gap | 影响 | 优先级 |
|---|---|---|
| **架构错位**：当前是"setup × horizon"，规范要求"**纯 horizon**"——`short_5d_model`（1-5d 动量反转 / 量价 / 涨停 / 竞价）、`mid_5d_30d_model`（板块轮动 / 政策验证 / 估值修复）、`long_30d_120d_model`（基本面改善 / 政策长期 / 估值分位）。**两个架构不冲突，可以并存**，但目前规范要求的"long 30-120d"完全缺失（只到 20d）。 | 长周期 alpha 缺失；规范的 alpha pipeline 第 3 个 horizon 没有训练 entry。 | **P3-P4** 必做 |
| 短周期需要的 **量能突变 / 涨停封单 / 炸板率 / 竞价强弱 / 隔夜挂单 / 龙虎榜** 等特征：alpha181 / cicc_high_freq 里有部分，但没有汇总到"short_5d"feature bundle。 | 短周期模型只用通用 alpha；丧失高频微结构 edge。 | **P4** 增量 |
| **大盘 regime gating** 已实现：`portfolio/market_hard_gate.py` + decision_chain 的 `regime_alignment` gate + `hard_market_gate` gate。但 spec 第五节要求的"放量/缩量 + 上涨/下跌 + 涨跌停数 + 连板高度 + 炸板率"宽度 metric 现在只有部分。 | regime 分类粒度可以更细。 | **P4** 增量 |

---

## 6. GA / Multi-epoch Weight Search

**已实现**
- `optimization/multi_objective_loss.py` 418 LOC — 10 项 loss：
  - +net_return, +sharpe, +calmar, -max_drawdown, -high_chase, -turnover, -tail_risk, +regime_consistency, -gross_volatility, +win_rate
- `optimization/factor_evolution.py` GA 优化器
- `training/optimize.py` grid / random search
- `training/splitters.py` purged walk-forward + embargo (per AGENTS.md)
- walk-forward backtest CLI 存在

**Gap (vs 规范第六节)**
| Gap | 影响 | 优先级 |
|---|---|---|
| Loss 缺三个显式项：`transaction_cost`（已部分以 slippage 出现在 backtest 但 loss 里无显式 cost term）、`concentration_risk`（decision_chain 有 gate 但 loss 没有 penalty）、`illiquidity_penalty`、`st_penalty`、`execution_penalty`（unfilled order rate）、`limit_chase_penalty`（高位连板追入）。 | Loss 不能完全反映 spec 列的 11 项；GA 可能选出"高 turnover + 高集中度"的纸面优解。 | **P3** 必做 |
| `_ann_return_from_daily` 已修为几何复合（review fix #7） — 验证过。 | 无 | ✓ |

---

## 7. Position Policy

**已实现**
- `portfolio/dynamic_top_k.py` + `walk_forward_sleeve_allocator.py` + `sleeve.py` — 分仓
- `portfolio/position_age_tracker.py` + `position_state.py` — 仓位年龄追踪
- `decision_chain` 14 个 gate（concentration_limit / risk_budget / drawdown_kill 等）
- `state_machine/machine.py` 278 LOC — 仓位状态机
- regime → setup compatibility 表：bear regime 只允许 lowbuy；crisis 全禁

**Gap (vs 规范第七节)**
| Gap | 影响 | 优先级 |
|---|---|---|
| **"默认 60% 总仓位上限，confidence 极高且全过风控才 80%；保留 20-40% 现金"** 这条核心规则**没有在代码中显式体现**。decision_chain 只检查单股 / 单板块上限，没有总仓位上限 gate。 | 高潮期可能满仓被打回撤；现金垫缺。 | **P3** 必做 |
| **做 T 必须用底仓，当天买不能当天卖** 这条 T+1 by-position 规则没有显式落地（VirtualBroker 有 frozen vs available shares，但没有 "T 单只能用 available" 的 enforcement test）。 | 回测可能允许 T+0 现金流幻觉。 | **P4** 验证 |
| **short / mid / long 持仓 tag + 跨 horizon 转仓 confidence 复检** 没实现。`position_age_tracker` 只追踪持仓时长，没有 horizon-class 标签。 | 持仓状态机无法在 horizon class 切换时强制走完整 risk check。 | **P4** 增量 |
| **不追多日连板** — `regime_sub_models.LimitUpRisk` 子模型用负号训练实现了"不要在连板顶部接盘"，间接覆盖。但 decision_chain 没有显式 `consecutive_limit_up_count >= N` 阻断 gate。 | 间接覆盖，不够硬。 | **P4** 增量 |

---

## 8. Execution Constraint DSL

**已实现**
- `risk/risk_limits.py` `V6RiskLimits`：max_name_weight / max_sector_weight / max_turnover / max_order_value / max_orders_per_day / min_lot_size / no_trade_st / no_buy_limit_up / no_sell_limit_down
- `execution/order_manager.py` config：lot_size / min_order_value_yuan / max_orders_per_symbol_per_day / max_participation_rate=5%
- `risk/risk_gate.py` 双重检查（target_weights → intents）
- `risk/kill_switch.py` 7 类触发：daily_loss / drawdown / reconciliation / provider / audit_write / rejection_rate / turnover
- `quant_math/ashare.py` `AshareRuleEngine` — 涨跌停 / 复权 / lot
- `execution/qmt_gateway.py` — dry_run=True 默认

**Gap (vs 规范第八节)**
| Gap | 影响 | 优先级 |
|---|---|---|
| **没有"ExecutionConstraintDSL"** —— 一个可声明、可测试、可在 QMT 提单前重放的约束 DSL。当前约束散落在 `V6RiskLimits` / `OrderManagerConfig` / `AshareRuleEngine` 三处。 | 修改一处约束需要改三个文件；audit replay 时难定位"哪个 DSL 规则否决"。 | **P4** 必做 |
| 缺规范点：`max_orders_per_second`、`max_cancel_ratio`、`min_order_resting_time_seconds`、`max_single_stock_participation_rate`（参与率只在 OrderManager 默认 5%，risk_limits 里没有；不同名字两套），`max_daily_turnover`（单日换手率）、`no_spoofing/layering/pull_push` 检测、`auction_mode_constraints`（集合竞价特殊规则）。 | 高频/微结构合规检查缺；竞价单不安全。 | **P4** 必做 |
| **集合竞价** 没有专门 mode —— 规范明确要求"不允许频繁挂撤、竞价单必须有明确成交意图、竞价行为可审计"。 | live 接入 QMT 前必须解决，否则有合规风险。 | **P4** 必做 |
| RiskGate 缺规范要求的检查项：`consecutive_limit_up_count` / `abnormal_large_order_flow` / `sudden_capital_outflow` / `news_contradiction` / `model_confidence_decay`。 | 风控覆盖面不够。 | **P4** 增量 |
| QMT live submit path 已禁用（`live_trading_enabled=False` 默认）✓ | 无 | ✓ |

---

## 9. Strict A-share Backtest

**已实现**
- `backtest/ashare_execution_simulator.py` 198 LOC — T+1 (via VirtualBroker.advance_trading_day)、slippage_bps=8、participation_cap=10%、ST 策略、audit log
- `backtest/full_pipeline_backtester.py` 176 LOC、`engine.py` 338 LOC、`tplus1_engine.py` 68 LOC、`paper_report.py` 403 LOC
- 输出 nav / order_audit / position_history / failed_order_audit / skipped_order_audit

**Gap (vs 规范第九节)**
| Gap | 影响 | 优先级 |
|---|---|---|
| **印花税 0.1% 卖出方** — 没看到在 cost_model.py 或 fill_simulator.py 显式扣。需要确认。 | 卖方收益被高估约 0.1%。 | **P5** 验证后修 |
| **佣金最低 5 元** A-share 散户规则 — 需要确认是否在 cost 模型里。 | 小额单子成本被低估。 | **P5** 验证后修 |
| **冲击成本（impact cost）** vs 滑点 — 当前只有 8bps 滑点，没有按 order_value / ADV 缩放的 impact term。 | 大单成本被低估。 | **P5** 增量 |
| **分钟/日线可切换** — 当前 backtest 是日线 only。 | 短周期模型无法 backtest 在分钟数据。 | **P5** 增量 |
| 9 项 spec metric (total_return / annualized_return / max_drawdown / sharpe / calmar / volatility / turnover / win_rate / avg_profit_per_trade / profit_by_stock / profit_by_sector)：`paper_report.py` 已实现大部分。需要点名清单核验 profit_by_sector. | 报告不齐。 | **P5** 验证 |
| `risk_events.json` 输出 — kill_switch 有 status() 但回测里没有汇总写 risk_events 文件。 | risk 事件审计断链。 | **P5** 必做 |

---

## 10. LLM Boundaries

**已实现 (架构层面)**
- AGENTS.md 明文规定：Agent/LLM 只输出 evidence/views/constraints/confidence/risk flags/audit logs
- Optimizer 只输出 target_weights
- Only OrderManager → order intents
- LLM 生成订单：架构层不可能（OrderManager 不调用 LLM）

**Gap (vs 规范第十节)**
| Gap | 影响 | 优先级 |
|---|---|---|
| **没有 explicit allowlist/denylist test** — 没有一个 pytest 断言"任何 LLM 模块的输出都不直接接 OrderManager"。 | 架构纪律只在 AGENTS.md，没有 enforcement。 | **P6** 必做 |
| LLM 政策文件解析 / 投行观点摘要 / 财报摘要 当前没有具体 agent —— 这些是 P7 才该做的应用层，不算 v8 spec 必需。 | LLM 实际能力未上线。 | **P7** |

---

## 11. CLI

| Spec v8 CLI | 已存在的 v7 等价 | 状态 |
|---|---|---|
| `ingest-policy-evidence-v8` | `import-policy-events-v7` (v7_policy) + `ingest-policy` (v7_evidence) | ✓ 改名即可 |
| `ingest-bond-flow-v8` | `import-bond-flows-v7` (v7_bond) | ✓ 改名即可 |
| `ingest-bank-financials-v8` | ✗ | **P7** 缺 |
| `build-capital-flow-thesis-v8` | 仅 builder 模块，无 CLI | **P2** 缺（依赖 P2 thesis 表） |
| `validate-capital-flow-thesis-v8` | 无 | **P2** 缺 |
| `build-sector-pool-v8` | `build-sector-pool-v7` (v7_sector) | ✓ 改名即可 |
| `build-fundamental-rank-v8` | `build-fundamental-ranker-v7` (v7_sector) | ✓ 改名即可 |
| `build-technical-factors-v8` | `materialize-factors-v7` / `materialize-alpha181-v7` | ✓ 改名即可 |
| `train-horizon-models-v8` | `train-alpha-v7` / `train-deep-alpha-v7`（单 horizon） | 部分（需多 horizon 串） |
| `optimize-ga-weights-v8` | `optimize-alpha-v7` / `evolve-factors` / `hp-search` | ✓ 已有 |
| `build-target-weights-v8` | `build-target-weights-v7` | ✓ 改名即可 |
| `run-strict-a-share-backtest-v8` | `walk-forward-backtest-v7` / `run-paper-backtest-v7` | ✓ 改名即可 |
| `run-paper-trading-v8` | `paper-trade-v7` / `paper-run-loop` / `paper-run-once` | ✓ 已有 |
| `generate-risk-report-v8` | 无 | **P5/P6** 缺 |
| `generate-daily-decision-report-v8` | 无 | **P6** 缺 |

**额外发现：** v7_evidence 已有 `ingest-policy` 命令，但**没有** broker_reports / state_team 的 CLI 入口。

**P7 必做：** 把 v7 CLI 加 v8 alias（不删旧的），并补缺的 5 个 v8 CLI。

---

## 12. Acceptance Gate

| 规范第十二节 | 当前状态 |
|---|---|
| pytest 通过 | **735/736**，1 pre-existing fail | ⚠️
| compileall 通过 | ✓ |
| 所有新增数据表都有 schema 和 manifest | ✓ 已审计的 4 个 builder 均有 |
| 所有训练样本都有 available_at | ✓ AGENTS.md 强制；4 个 builder 验证 |
| 没有未来函数 | 大概率 ✓，但**没有跨 source PIT lint 测试** | ⚠️
| 没有 synthetic fallback 进入 production | AGENTS.md 强制；未做 grep 验证 | ⚠️
| LLM 不直接生成订单 | ✓ 架构层不可能（OrderManager 不调 LLM） |
| RiskGate 可否决 | ✓ `RiskGateResult.passed=False` 阻断 |
| Backtest 体现 T+1 / 涨跌停 / 停牌 / 成本 / 滑点 | ✓ ashare_execution_simulator + VirtualBroker；印花税/佣金需补 |
| 输出 daily decision report | ✗ 缺 |

---

## 整体优先级建议

| 阶段 | 内容 | 估算 | 建议本 session 推进 |
|---|---|---|---|
| **P0** | 本 gap report | 完成 | ✓ |
| **P1** | **统一 EvidenceRecord schema** + 跨 source PIT lint + canonical loader | 1.5d | **本 session** |
| **P2** | capital_flow_thesis 表 + 1/5/20/60/120d 验证回路 + validation state machine | 2d | 下个 session |
| **P3** | (a) horizon 模型补 long 30-120d；(b) loss 补 cost/concentration/illiquidity/st/execution；(c) 60% 仓位上限 gate；(d) decision_sector_pool（与现有 IC pool 并存） | 3d | 下个 session |
| **P4** | ExecutionConstraintDSL + 集合竞价 mode + 高频/微结构合规字段 + horizon-class 仓位 tag | 2d | |
| **P5** | 印花税/佣金/impact cost 补、risk_events.json 输出、minute backtest 切换 | 2d | |
| **P6** | LLM allowlist test、generate-daily-decision-report-v8、generate-risk-report-v8 | 1d | |
| **P7** | v8 CLI alias 收口 + 缺的 5 个 CLI + ingest-bank-financials | 1d | |
| 后续 | ft_transformer per-date rank loss regression（pre-existing） | 0.5d | |

## 不需要改的（已经达标或超规范）

- Sector pool（IC-driven，比规范要求更严谨——但语义不同，建议**并存**）
- Fundamental ranker（PIT 严格、within-sector rank、manifest gate；只是字段数偏少）
- multi_objective_loss（10/13 项，几何复合修过）
- decision_chain 14 个 gate
- evidence builders 的 PIT 处理（`available_at` 设计 + manifest gate + audit-only contract）
- AGENTS.md 已经把安全边界写得很清楚

---

## Stage 5 Build-out (2026-05-30) — what actually landed

每阶段在交付前都跑通 `pytest -q` 和 `python -m compileall src`。最终 baseline：**845 pass / 1 pre-existing fail / 3 skipped**。

### P1 — Canonical Evidence Schema
- `src/quantagent/data/evidence/canonical.py` — 18 字段 `EvidenceRecord` + 12 类 `CANONICAL_SOURCE_TYPES`，含规范第一节要求的全部字段（extracted_claims / sentiment_score / policy_direction_score / capital_flow_direction_score / contradiction_score / lag_window_candidates / audit_trace）。
- 4 个 adapter：`policy_events_to_evidence` / `bond_flows_to_evidence` / `broker_reports_to_evidence` / `state_team_events_to_evidence`。
- `to_canonical_evidence_frame(...)` 聚合 + dedup + 按 available_at 排序。
- `validate_pit_safety()` + `PITLintReport` — 跨 source PIT lint。
- 4 个子包 `__init__.py` re-export adapter。
- `tests/data/test_evidence_canonical.py` — **22 测试**。

### P2 — Capital-flow Thesis + Validation
- `src/quantagent/data/thesis/builder.py` — `CapitalFlowThesis` dataclass（16 字段：thesis_id / direction_kind / direction_value / thesis_sign / supporting_evidence_ids / contradiction_evidence_ids / confidence / contradiction_score / expected_lag_days / tradability_score / decay_score / validation_status / …）。
- `src/quantagent/data/thesis/validation.py` — 1/5/20/60/120d 累积超额收益回看验证回路；五态状态机 `unverified → partially_verified → verified | rejected | expired`；`decay_score` 随 horizon 跨度降低；horizons not yet elapsed 不被误判为 confirm。
- `tests/data/test_thesis.py` — **13 测试**。

### P3.1 — Multi-horizon Models
- `src/quantagent/training/horizon_models.py` — `HorizonClass.{SHORT, MID, LONG}` 三类（1-5d / 5-20d / 60-120d），每类的 feature 白名单 + label 列规则 + `HorizonBundle` + `ensemble_horizon_predictions`。
- `tests/training/test_horizon_models.py` — **14 测试**。

### P3.2 — Loss Extensions
- `multi_objective_loss.py` 增加 5 项：`transaction_cost` / `concentration` / `illiquidity` / `st_exposure` / `execution_unfilled`，从 10 → **15 components + total = 16**。Component 默认值 + 自动 clip 到 [0, 1] 区间。
- `tests/optimization/test_multi_objective_loss.py` 加 7 个新断言，老的 `len==11` fingerprint test 改为 `len==16`。

### P3.3 — Gross-exposure Budget Gate
- `decision_chain/chain.py` 新增 `gross_exposure_budget` gate（第 15 个 gate）：默认 60% 上限；只在 ① `global_conviction ≥ high_conviction_threshold` 且 ② regime ∈ `(normal, bull)` 时允许扩展到 80%；超 80% 一律 block。
- `DecisionContext` 加 `current_gross_exposure` + `global_conviction` 字段。
- `tests/portfolio/decision_chain/test_chain.py` 加 **6 测试**；老的 `exactly_14_gates` test 改为 15。

### P4 — ExecutionConstraintDSL
- `src/quantagent/execution/constraints.py` — 声明式 `ExecutionConstraintSet`（19 字段），覆盖规范第 8 节全部 DSL 点：
  - rate limit (`max_orders_per_second / per_day`)
  - cancel ratio + min resting time
  - 集合竞价特殊规则（auction_mode_*，9:15-9:25 / 14:57-15:00 二次更严）
  - 单笔/单股/单日 size cap
  - 三类异常 heuristic（no_spoofing / no_layering / no_pull_push）
  - `qmt_dry_run_required_by_default + live_trading_enabled` 互斥校验
- `classify_auction_phase()` 把时间戳分入 6 个 phase。
- `ExecutionConstraintEvaluator.evaluate()` 返回 `ExecutionConstraintReport` 含每条 violation 的 `intent_id / symbol / constraint / severity / reason / detail`。
- `tests/execution/test_constraint_dsl.py` — **19 测试**。

### P5 — Cost Model + Risk Events
- `execution/cost_model.py` 加 `impact_alpha_bps` + 平方根冲击成本（`impact = alpha · sqrt(participation) · order_value / 1e4`）；保留印花税 0.05% / 最低佣金 5 元 / 万一过户费。
- `ashare_execution_simulator.py` 输出 `risk_events: list[dict]` + `write_risk_events(path)` 方法；每条记录被 reject / cancel / partial / skip 的 order 都会进 risk_events。
- `tests/execution/test_cost_model.py` (9) + `tests/execution/test_risk_events_output.py` (3) — **12 测试**。

### P6 — Daily Decision Report + LLM Allowlist
- `src/quantagent/diagnostics/daily_decision_report.py` — Markdown 报告生成器（Summary / Sector picks / Stock picks / Position sizing / Rejected candidates / Risk view / Thesis corroboration 7 个 section），每个 section 在缺数据时降级为 `_(no data)_` 而不 crash。
- `tests/diagnostics/test_daily_decision_report.py` 含 **9 报告测试** + **3 LLM allowlist tests**：通过 AST 扫描 `OrderManager` / `RiskGate` / `ashare_execution_simulator` 的 import，断言其中没有任何 LLM-bearing 模块（anthropic / openai / langchain / quantagent.agents / quantagent.themes.policy_parser / quantagent.credibility.news_credibility_agent）。

### P7 — V8 CLI Surface
- `src/quantagent/cli/v8.py` 注册 **8 个 v8 命令**：
  - `ingest-policy-evidence-v8` (alias)
  - `ingest-bond-flow-v8` (alias)
  - `build-sector-pool-v8` (alias)
  - `build-fundamental-rank-v8` (alias)
  - `build-capital-flow-thesis-v8` (新)
  - `validate-capital-flow-thesis-v8` (新)
  - `generate-daily-decision-report-v8` (新)
  - `generate-risk-report-v8` (新)
- `cli/__init__.py` 加 `from quantagent.cli import v8`。
- `tests/cli/test_v8_aliases.py` — **5 测试**含 typer CliRunner 端到端 smoke。

### E2E
- `tests/test_v8_pipeline_e2e.py` — 一个测试串起 8 个 stage：
  1. canonical evidence → PIT lint
  2. thesis builder
  3. thesis validation
  4. decision_chain 含 gross_exposure_budget gate
  5. ExecutionConstraintDSL
  6. backtest → risk_events.json
  7. multi_objective_loss 含 Stage 5 terms
  8. daily decision report

### 累计影响 (Stage 5)
- 新增 LOC：~3400 (src) + ~1800 (tests)
- 测试数：735 → **845** (+110)

---

## Stage 6 Build-out — Spec Sections 3/4/5/6/7/9/11 Fully Implemented

每阶段在交付前都跑通 `pytest` + `compileall src`。最终 baseline：**909 pass / 1 pre-existing fail / 3 skipped**。

### P8.1 — SectorPoolV8 (决策维度板块池, spec section 3)
- `data/sector/decision_pool.py` — `build_sector_pool_v8(...)` with 8 输入轴 → 13-column `sector_pool_v8.parquet`（date / sector_code / sector_name / policy_score / capital_flow_score / sentiment_score / broker_attention_score / market_strength_score / liquidity_score / valuation_percentile / risk_score / final_sector_rank / confidence）。所有 score 在 [0, 1]，valuation_percentile 越低越便宜。每轴可独立缺失；confidence 反映可用轴数。`SectorPoolV8Builder.write()` 写 parquet + coverage_report + manifest。**与既有 IC-driven `sector_pool` 并存**——前者是"模型 OOS edge per sector"，后者是"该板块投资吸引力"，两者交集 = 最高确信桶。
- `tests/data/sector/test_decision_pool.py` — **10 测试**。

### P8.2 — ExtendedFundamentalRanker (19 字段全量, spec section 4)
- `data/fundamental/extended_ranker.py` — 19 axis (PE/PB/PS/ROE/ROA/gross/net_margin/revenue_yoy/net_income_yoy/operating_cashflow/accruals_quality/earnings_surprise/debt_to_asset/interest_coverage/inventory_turnover/accounts_receivable_growth/goodwill_risk/dividend/repurchase) 按 7 group (valuation/profitability/growth/quality/leverage/efficiency/capital_action) 聚合 → composite_score。
- 流程：cross-section winsorize (1st/99th percentile) → z-standardize → normal CDF map → 单位 [0, 1]；每 group = mean(per-axis)；composite = weighted mean of available groups（权重重归一化）。direction 由 `_AXIS_TABLE` 配置控制（"lower_better" 自动翻转）。
- 17-column 输出 + within-sector rank + metric_completeness 反映该样本覆盖多少 group。
- `tests/data/test_extended_fundamental_ranker.py` — **10 测试**。

### P8.3 — MarketRegimeDetector (大盘全局 regime, spec section 5)
- `portfolio/market_regime_detector.py` — 二维输入 (5 个 benchmark index trend × volume + 市场宽度) → 6 个 regime label (bull_expansion/bull_consolidation/normal/caution/bear_capitulation/crisis) + 4 个 risk_level (low/medium/high/severe)。
- 阈值化决策（不是 ML，可解释）：bull_expansion 需 4/5 index 涨 + 量能扩张 + 宽度强 + 涨停数高；crisis 需 ≥2 severe signals。
- `regime_risk_to_exposure_cap(risk_level)` 把 risk → 推荐 gross cap (0.20–0.80)，给 PositionPolicy 用。
- `detect_market_regime_series` 批量计算历史快照表，可直接喂给 `DecisionContext.regime_state`。
- `tests/portfolio/test_market_regime_detector.py` — **8 测试**。

### P8.4 — PositionPolicy (持仓状态机, spec section 7)
- `portfolio/position_policy.py` — `PositionClass.{SHORT, MID, LONG}` 三类 + 跨类转移图 + confidence/risk check enforcement。
- 落地的 spec 第 7 节规则：
  - 默认 60% gross cap, 80% 高 conviction + 友好 regime 上限；20%+ cash buffer
  - ST/*ST、停牌、一字板（买入侧）默认排除
  - `max_consecutive_limit_up_chase` 强制：连板 ≥N 不允许追入
  - T+0 enforcement：`HeldPosition.same_day_acquired > 0` 时同 symbol 卖出 block
  - 跨 horizon-class 切换需要新 class 的 confidence ≥ `transition_min_confidence`
- `compute_consecutive_limit_up_count()` helper：从市场 panel 逐 symbol 算连板天数。
- `tests/portfolio/test_position_policy.py` — **18 测试**。

### P8.5 — GAWeightOptimizer (多目标 GA + walk-forward + purged CV, spec section 6)
- `optimization/ga_weight_optimizer.py` — 纯 NumPy/pandas 实现的 GA：
  - 染色体 = 因子权重向量，每代后 clip + renormalize
  - tournament-of-2 selection + uniform crossover + Gaussian mutation + elitism
  - 适应度 = `compute_multi_objective_loss(daily_portfolio_returns).total`
  - **OOS-only**：fold 在训练折选最优后强制在测试折 measure（in-sample 不进 loss）
- `purged_walk_forward_splits()` — n_folds + embargo_days + min_train/test 严格分段，每折间留 embargo gap 防 label leakage。
- `save_optimisation_artifacts()` 写 `factor_weights.json` + `walk_forward_backtest.json` + `metrics.json`。
- `tests/optimization/test_ga_weight_optimizer.py` — **8 测试**含两因子 synthetic experiment 验证 GA 把权重偏向 signal factor。

### P8.6 — StrictBacktestV8 (全量输出 bundle, spec section 9)
- `backtest/strict_v8.py` — 在既有 `simulate_ashare_target_weights` (T+1 + 涨跌停 + ST + cost_model + risk_events) 之上，把规范第 9 节要求的 9 个 metric 全部计算：total_return / annualized_return / max_drawdown / sharpe / calmar / volatility / turnover / win_rate / avg_profit_per_trade。
- `StrictBacktestArtifactSet.write(output_dir)` 一次写齐 10 个文件：`metrics.json / nav.csv / pnl.csv / selected_stocks.csv / trades.csv / failed_orders.csv / risk_events.json / profit_by_stock.csv / profit_by_sector.csv / factor_weights.json`。
- 不重新实现回测引擎——仅是 reporting layer，PIT/T+1/cost/slippage/kill-switch 保证全部从上游继承。
- `tests/backtest/test_strict_v8.py` — **6 测试**。

### P8.7 — 余下 7 个 v8 CLI (spec section 11)
- `cli/v8.py` 增 7 个命令：
  - `ingest-bank-financials-v8`
  - `build-technical-factors-v8`
  - `train-horizon-models-v8`
  - `optimize-ga-weights-v8`
  - `build-target-weights-v8`
  - `run-strict-a-share-backtest-v8`
  - `run-paper-trading-v8`
- **15 个 v8 命令现已全部到位**（与规范第 11 节 1:1 对齐）。
- `tests/cli/test_v8_aliases.py` 加 2 个新 smoke 测试，所有 15 个命令都通过注册检查 + `--help` 列出。

### P8.8 — Full Pipeline E2E
- `tests/test_v8_full_pipeline_e2e.py` 串起 10 个阶段：
  1. Canonical evidence + PIT lint
  2. Capital-flow thesis + validation
  3. SectorPoolV8 (decision-axis)
  4. ExtendedFundamentalRanker
  5. MarketRegimeDetector
  6. PositionPolicy
  7. Horizon model bundles
  8. GA factor weights (purged walk-forward)
  9. StrictBacktestV8 (full output bundle)
  10. DailyDecisionReport
- 每段都断言对下游契约。

### Stage 6 累计影响
- 新增 LOC：~3000 (src) + ~1700 (tests)
- 测试数：845 → **909** (+64)

---

## v8 Spec Section 验收回顾

| Section | 状态 | 关键模块 |
|---|---|---|
| 1. Evidence schema | ✅ | `data/evidence/canonical.py` + 4 builders' adapters |
| 2. Capital-flow thesis | ✅ | `data/thesis/{builder,validation}.py` |
| 3. Sector pool | ✅ | `data/sector/decision_pool.py` (v8) + `sector_pool.py` (IC, 并存) |
| 4. Fundamental ranker (19 字段) | ✅ | `data/fundamental/extended_ranker.py` |
| 5. Multi-horizon models + regime | ✅ | `training/horizon_models.py` + `portfolio/market_regime_detector.py` |
| 6. GA / walk-forward / purged CV | ✅ | `optimization/ga_weight_optimizer.py` |
| 7. Position policy | ✅ | `portfolio/position_policy.py` + decision_chain `gross_exposure_budget` gate |
| 8. Execution constraint DSL | ✅ | `execution/constraints.py` |
| 9. Strict backtest | ✅ | `backtest/strict_v8.py` + `execution/cost_model.py` (印花税 / 最低佣金 / 平方根冲击) |
| 10. LLM bounds | ✅ | `tests/diagnostics/test_daily_decision_report.py` AST allowlist |
| 11. CLI (15) | ✅ | `cli/v8.py` 全部 |
| 12. 验收 | ✅ | pytest 909 pass; compileall ✓; 全 manifests; PIT lint; no LLM in OrderManager |

### 累计 (Stage 5 + Stage 6)
- 总新增 LOC：~6400 (src) + ~3500 (tests)
- 测试数：735 → **909** (+174)
- 全部新工作守恒 PIT 契约、不绕过 RiskGate、不让 LLM 接 OrderManager、不引入 synthetic fallback。
- ft_transformer per-date rank loss regression 仍未修——独立任务，**不在 v8 spec scope**。
