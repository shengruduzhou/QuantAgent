# current_model_audit — 全宇宙稳健性任务 Phase 0 审计（run_id: fu_20260713）

> 生成 2026-07-13，git HEAD `3e61891bb7a1386148113d2ec099697fbbfe28bb`，branch `robustness-mission`（工作树干净）。
> 本审计整合既有 [VERIFIED] 审计（`ARCHITECTURE_AUDIT.md` 2026-07-03、`MODEL_FLOW_MAP.md`、`BASELINE_TRUST_CLASSIFICATION.md`）+ 本次程序化复核（`runtime/reports/full_universe/fu_20260713/`）。
> 机器可读版本：`runtime/reports/full_universe/fu_20260713/audit.json`。

## 1. 生产路径裁定（未变化，复核确认）

- **生产模型 = FT-Transformer 三 sleeve**（`models/ft_transformer.py` + `training/ft_transformer_trainer.py`，经 `cli/v8_deep.py train-v8-deep`）；**不是** `training/v8_pipeline.py`（旧 GA 路径）、不是 `models/v7_deep_alpha.py`（未训练启发式，文件头有 STATUS WARNING）。
- 生产 blend 单一事实源 = `configs/production_blend.json`：2-sleeve（short_5d + mid_5d_30d）per-date cross-sectional rank sum，top_k=10，trust class **`likely_overfit`**（PBO 0.886，38.6% holdout 数字 = `contaminated_holdout`，禁止引用为业绩）。
- **唯一可信评测入口** = `scripts/baseline_protocol.py` variant **C_flags_eligible_delay1**（P4 隔离守卫 fail-closed；INC-E1 跨日吞单缺陷已修复并升级为默认，2026-07-06 用户批准）。
- 数据集 lineage：gold `training_dataset_alpha181_exec_v89_plus7clean.parquet`（sha256 在 production_blend.json 在册）；估值/基本面扩展版 `..._plus7clean_fund.parquet`（EXP-020 PIT 审计通过）。

## 2. 冠军冻结状态（本任务的硬边界）

commit `19e05f4`：**L1 / L1+D1_regime / RW1_4state 已锁定为 FRESH 3-way 仲裁集**，折内再评测对这三个配置关闭。FRESH 首读 ≥120 交易日（≈2026-11），每配置只读一次。
⇒ 本任务产生的任何新候选**不能**进入 FRESH 首读集，最高信任标签只能是 `candidate_research_only_not_fresh_holdout_validated`，排队等下一代（FRESH 首读之后的周期）。

## 3. 已回答的问题（本任务不得重复烧预算）

| 问题 | 答案 | 证据 |
|---|---|---|
| 特征覆盖是不是瓶颈？ | **否**（估值/基本面作 raw input 无增量 OOS alpha，2 宇宙×2 base 复制） | EXP-020/021/022 |
| 学习型 regime 权重可行吗？ | 否（3/4 折胜但中位差 3pp、Calmar 不过门）；且暴露手设 overlay 部分 fold-informed | EXP-023 |
| evaluator 可信吗？ | 是（标签 delay-1 executable 回归锁定；IC 0.16 = 因子结构非泄漏） | EVALUATOR_VALIDITY_AUDIT_IC016.md |
| L2/分钟微结构可做吗？ | 不可（Tier A bars-only；TickFlow L2/分钟 403；分钟史大部分在隔离窗） | AUDIT-2026-07-10 / factor_capability_matrix.csv |
| 冠军容量？ | 可辩护 10–30M CNY；>30M 不可证（评估器缺 √impact 模型 = 已知硬缺口） | EXP-024 |
| 换手/书构建还有油水吗？ | churn 控制 0/5、宽书 0/3 全拒；Track L 低换手已是答案 | EXP-011/012/015 |

## 4. 可安全复用 vs 须重建

- **复用**：silver market_panel（QC 见 data_quality_report.json）；gold plus7clean/plus7clean_fund；冻结 sleeve 预测（retrain_plus7_20260620_0300）；因子批筛选 harness（`scripts/analysis/dual_track_factor_batch.py` 的 `score_factors`，批次 1/2 已产出在案）；PBO/DSR 工具；beta 分解。
- **不得复用（信任已废）**：v8.8 系全部（batch-rank 污染）；一切 `contaminated_holdout` 读数作对照；minute/做T 结论前的中间物；regime_search 赢家书。
- **须显式声明局限**：sector_map = current snapshot（survivorship）；宇宙文件静态（`universe_v88_comma.txt`）；pre-2020 survivorship 已在 Stage A 审计承认 → 全宇宙声明须带此限定。

## 5. 资源现状（本次实测）

- RAM 62 GiB（可用 ~58）；GPU 3090 24G 空闲（1 MiB used）；磁盘剩 **191 GB（79% used）**；runtime/ = 78 GB。
- 约束沿用 ACCEPTANCE_RULES R5：RAM ≤48 GiB、VRAM ≤20 GiB、新磁盘 ≤5 GB/实验、禁无界搜索。

## 6. 本任务的裁定级结论（Phase 0 判断）

任务书要求"全宇宙因子研究 + 多保真训练 + 严格回测"。对照 §2/§3：
1. **模型再训练线（Stage D/E 大规模）没有正当性**：特征线已收敛（EXP-021/022），冠军已冻结，SEARCH 窗已高度复用 —— 现在烧 GPU 重训只会增加 N、抬高 DSR 校正门槛而不改变 FRESH 仲裁。
2. **有正当性的 bounded 增量** = IDEA_QUEUE #8（DSL 因子新批次，cap≤20/批，tradability-aware 验收）：纯 CPU、预注册、pre-quarantine 窗、为下一代（post-FRESH）carrier 积累经审核的因子知识。批次 1 遗留显式 TODO（D6 vol-compression 中换手 track 重判）。
3. **严格 variant-C 回测本轮 = 0 次**（先验声明）：新因子的 book 级集成测试会重蹈 EXP-023 揭露的 fold-informed 缺陷，且无法进入已锁定的 FRESH 首读集 ⇒ 集成评测推迟到 FRESH 首读后的下一代周期。这正是任务书"最小化昂贵回测"的极限形式。
