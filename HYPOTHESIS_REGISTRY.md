# HYPOTHESIS_REGISTRY — 假设预注册台账（Stage D/E）

> **先注册后跑**：候选网格、试验数 N、选择指标必须先写在这里。同族 N 历史累加（DSR 用累计 N）。
> 选择数据分区：TRAIN ≤2024-06-30；SEARCH/VAL 2024-08-28→2025-08-31；QUARANTINE 2025-09-01→2026-05-18（guard 强制）；FRESH 2026-05-19+（未入库，冻结制）。

## 累计试验数台账（同族累加）

| 族 | 历史 N | 出处 |
|---|---|---|
| sleeve-blend × top-k 选择 | 27（ensemble_search）+ 10（topk_sweep/fine 含窗）≈ **37** | census |
| regime policy | 8 | regime_search |
| 因子组合（combo greedy） | ~36 greedy 步 | factor_combo_search |
| **H-001 新增** | +4（预注册，无搜索） | 本表 |
| **H-002 新增** | +3 | 本表 |

---

## H-001 族稳健 blend 优于单点赢家（状态：**TESTED — NOT ACCEPTED**，EXP-001，2026-07-03）

> 结果：无聚合配置全门通过；C3 rank-median 在 maxDD（12.6%）与最差季度（−8.4%）上最佳但换手超标（0.336>0.25）→ 转入 H-002 修复。C2 在本窗占优的结构性偏向（本窗=其选择窗）已记录。详见 EXPERIMENT_LEDGER EXP-001。

- **来源**：PBO_DSR_ANALYSIS.md（PBO 0.886）+ RESEARCH_LOG #R3（组合优于选择）。
- **假设**：在 sleeve 预测族上做**先验聚合**（不选择），其 SEARCH 窗子窗稳定性与净化后风险指标 ≥ 单点赢家 (1,1,0)k10，且无选择噪声暴露。
- **机制**：候选相关 0.805、排名反转 ⇒ 选择方差 > 选择收益；聚合消除选择方差。
- **候选（预注册 N=4，全部先验定义，禁止事后加）**：
  C1 = 3-sleeve 平均（HorizonEnsembleWeights 0.30/0.45/0.25，代码历史默认 = reference）；
  C2 = 2-sleeve rank sum（现生产候选，作对照）；
  C3 = 3-sleeve **rank 中位数**（family-median）；
  C4 = 3-sleeve 等权 rank sum（1,1,1）。
- **数据**：冻结 sleeve 预测（retrain_plus7_20260620_0300）+ SEARCH 窗 panel。无新训练。
- **泄漏风险**：低（全窗 ≤2025-08-31，guard 强制；配置先验）。residual：SEARCH 窗历史复用 → 绝对量级不可信，只比较相对形态。
- **实现文件**：复用 `materialize_production_composite.py`（权重参数化即可支持 C1/C2/C4；C3 需 ~15 行 median 模式）+ `scripts/analysis/` 评测驱动。
- **测试计划**：4 配置 × variant-C on SEARCH 窗 → 季度子窗（4 折）CAGR/DD/换手 → 子窗排名稳定性 + 4 候选 CSCV-PBO（N=4 时 PBO 粒度粗，主要看子窗一致性）。
- **资源**：~4×15s 回测 + 子窗切分 ≈ 5 分钟，RSS <4G。
- **接受**：某聚合配置子窗最差表现 ≥ C2 的子窗最差，且换手 ≤0.25/日、子窗方向一致 ⇒ 立为新参考配置（trust=searched_validation, N=4 记账）。
- **拒绝**：所有聚合在 ≥3/4 子窗劣于 C2 → 保留 C2 为候选但维持 likely_overfit 标签等 FRESH 窗裁决。

## H-002 换手感知平滑（状态：**TESTED — ACCEPTED（带窗口注记）**，EXP-002，2026-07-03）

> 结果：α∈{0.3,0.5,0.7} 三档**全部**通过预注册门（换手 0.022–0.077 ≤0.10；CAGR/最差季度不劣化，实测反而改善：ema0.5/0.7 四季全正）。机制接受；α 加冕与绝对量级确认**推迟至 walk-forward / FRESH 窗**（SEARCH 窗复用注记）。候选组合 **C3+EMA** 立为"待 WF 确认的 challenger 配置"；生产配置不变更。累计 N：blend 族 44。

- **来源**：RESEARCH_LOG #R4（Gârleanu–Pedersen 部分调仓）。
- **假设**：对参考 blend 的 composite_score 做 EMA 平滑可把换手从 ~0.16–0.21/日 压到 ≤0.10/日，SEARCH 窗净 CAGR 损失 ≤3pp（成本节约部分补偿信号迟滞）。
- **候选（预注册 N=3）**：EMA α ∈ {0.3, 0.5, 0.7}（α=新信号权重）。
- **泄漏风险**：低；EMA 仅用过去分数。
- **实现**：materializer 加 `--score-ema-alpha`（~10 行）。
- **测试**：3 α × variant-C SEARCH 窗 + 季度子窗 vs 参考配置；报告换手/成本敏感（8→15bps）。
- **资源**：~10 分钟，RSS <4G。
- **接受**：存在 α 使换手 ≤0.10/日 且子窗最差 CAGR 不劣于参考 −3pp。
- **拒绝**：全部 α 子窗劣化 >3pp 或稳定性变差。

## H-003 数据新鲜化（状态：**BLOCKED — 待用户批准**，属"touching fresh data"红线）

- 用 `update_market_panel_daily.py` 把 silver panel 从 2026-05-18 补到当前（约 +30 交易日），随即按 EVALUATION_PROTOCOL_V2 §2 冻结（零评测、零选择，仅积累）。附带：panel 备份 manifest + 更新后 FRESH 窗登记。**不批准则 FRESH 窗永远无法成熟。**

## H-004 长 sleeve 诊断（状态：REGISTERED，EXP-004，诊断类 N+0）

- **来源**：长 sleeve weight=0 来自 likely_overfit 的搜索，从未独立验证（IDEA #5，用户批准 2026-07-03）。
- **问题**：长 sleeve 是否携带 regime 条件性信息（尤其回撤期），被 weight=0 丢弃？
- **设计（诊断，不产生选择 ⇒ 试验数 +0）**：SEARCH 窗；每 horizon 的 IC 评估仅用 `label_end_{h}d < 2025-09-01` 的日期（**label 前视不跨隔离窗**）；指标 = 分 regime（bull/sideways/bear）与分季度的 per-date rank-IC、回撤期贡献、与 short/mid 的截面秩相关、shrinkage 稳定性（复用 EXP-000 已有 27 候选日收益矩阵中 w1_1_0.5 vs w1_1_0，**零新回测**）。
- **纪律**：结果不直接调生产权重；只决定是否立后续假设（如 H-005 regime 条件性 blend）。
- **资源**：CPU ≤10min，RSS <6G。

## H-008 走式验证（状态：**TESTED — C3+EMA REJECTED for adoption**，EXP-008，2026-07-06）

> 9/9 重训 + 24 评测完成。C3_ema0.7 在 4/5 稳健性轴上胜过现生产候选 C2（中位折 CAGR +33.0% vs +23.8%、最差折 −29.9% vs −33.0%、worstDD 25.0% vs 31.5%、15bps 敏感性中位 −8.8% vs −22%、DSR 0.736 vs 0.651），但换手门（max 0.259>0.10 承诺）与统计门（PBO 0.833/DSR<0.95）未过 ⇒ 预注册规则下**不采纳**，维持 challenger。C2 自身 worstDD 31.5%、压力折换手 0.70/日 —— **现生产候选在自家走式上也过不了验收门**。族级发现：F2 崩塌折全员 −30~−55% ⇒ 下一优先假设 = 回撤/regime 暴露控制层（见 H-009 预告）。

## H-009 回撤/regime 暴露控制（状态：**REGISTERED 2026-07-06，EXP-009 待跑**）

- 动机：EXP-008 族级 F2 失败（bench −33.1%，全候选 −29.7~−55.2%）；IDEA #7。
- **候选规则（N=3，先验冻结，跑后不改）**——触发器只用等权全A bench 序列（panel 收盘价），t−1 日观测 → t 日执行（与书的 delay-1 一致，零前视）；gross ∈ (0,1] 恒不加杠杆，缩掉的部分=现金：
  - **R1 回撤分档**：bench 60 日滚动峰值回撤 DD(t−1)：DD<8% → gross 1.0；8%≤DD<15% → 0.5；DD≥15% → 0.3
  - **R2 趋势过滤**：bench 收盘(t−1) ≥ 60 日均线 → 1.0；否则 → 0.5
  - **R3 波动分档**：bench 20 日实现波动年化 σ(t−1)：σ<25% → 1.0；25%≤σ<40% → 0.5；σ≥40% → 0.3
- 载体：冻结的 C3_ema0.7 目标权重（EXP-008 各折原样重建，零重训）；对照 = 无 overlay 的 C3_ema0.7 与 C2。
- 评测：H-008 同 4 折、variant-C 全约束、8bps；崩塌折 F2 单列。
- **验收（先验）**：① 最差折 maxDD < 25.0%（基线 worstDD）② F2 CAGR > −29.9%（基线）③ 换手 ≤ 基线+0.05/日 且 ≤0.35 ④ 中位折 CAGR ≥ 基线−5pp（≥+28.0%）⑤ 无杠杆（构造保证）⑥ 新鲜窗零接触 ⑦ PBO/DSR 更新入账。
- **拒绝**：任一规则族全违 ①/② 或全部候选违 ④。禁止跑后调档位/加规则（累计 N：blend+overlay 族 50+3=53）。
- 预算：CPU-only，12 次 variant-C ≈10min，RSS <4G，磁盘 <50MB。

## H-010 R2 趋势过滤的滞回修复（状态：**REGISTERED 2026-07-06，EXP-010 待跑；本周期 overlay 线最终迭代——跑后无论结果本线停止，等 FRESH 窗裁决**）

- 动机：EXP-009 中 R2 唯一败于横盘 whipsaw 换手（F1 0.360）；机制修复 = 滞回/平滑，非盲调。
- **候选（N=2，先验冻结）**，载体与评测同 H-009：
  - **R2a 确认滞回**：连续 5 个交易日 bench(t−1)<MA60 才降至 0.5；连续 5 日 ≥MA60 才回 1.0（双向确认）
  - **R2b 平滑 gross**：g_t = EMA(α=0.2) of R2 原始二元 gross（渐进调仓替代硬切换）
- **验收（与 H-009 相同四门）**：worstDD<25.0% ∧ F2>−29.9% ∧ maxTurn≤min(0.309,0.35) ∧ 中位≥+28.0%。
- 累计 N：53+2=**55**。预算 8 次 variant-C ≈7min CPU。

## H-011 书构建层 churn 控制（状态：**REGISTERED 2026-07-06，EXP-011 待跑；Track A 第一批**）

- 动机：EXP-008 换手门失败（C3_ema0.7 max 0.259 > 0.10 承诺）+ EXP-009/010 结构性结论（churn 在书构建层解，overlay 线已关闭）。
- **候选（N=5，先验冻结，跑后不改）**——载体=C3_ema0.7 分数（EXP-008 原样重建，零重训），k=10 等权，long-only，gross≤1，eligibility/delay-1 与 variant-C `_target_weights` 完全一致：
  - **B1_buffer30** 排名保留区：进入 top-10，持有至跌出 top-30
  - **B2_minhold10** 最短持有 10 交易日（锁仓名额制，新入者 age=1）
  - **B3_partial30** 部分调仓 w_t=0.7·w_{t−1}+0.3·target_t（<0.5% 剪除后归一）
  - **B4_reb5d** 每 5 交易日重构，期间目标权重不变
  - **B5_buffer_r2a_ramp** B1 书 × R2a confirm-5 MA60 gross{1.0,0.5}，gross 变动 ≤0.1/日（渐进切换，t−1 观测 t 执行）
- 评测：H-008 同 4 折 variant-C 8bps + 全员 15bps 敏感性（仅报告）；PBO（6 书 CSCV）+ DSR@N=60。
- **验收门（先验，全过才接受机制）**：G1 maxTurn≤0.10 ∧ G2 worstDD≤0.2503 ∧ G3 F2≥−24.9%（基线+5pp）∧ G4 中位≥+28.02% ∧ G5 sector max≤0.33 ∧ G6 无杠杆 ∧ G7 新鲜窗零接触；生产采纳另需 DSR≥0.95（不自动改生产）。G3 单独不过=churn-solved/crash-unsolved 记录在案。
- 累计 N：55+5=**60**。预算：CPU 40 次 variant-C ≈10min，RSS<4G，磁盘<50MB。
- 详细定义/命令/基线冻结：BOOK_CHURN_CONTROL_EXPERIMENT.md。

## 队列中未立项（见 IDEA_QUEUE.md）

H-004 sector 集中度约束收紧；H-005 长 sleeve 诊断价值（何时该有非零权重）；H-006 DSL 因子新批次（capped）；H-007 offline RL turnover-controller；H-008 walk-forward 重训协议（模型层，需 GPU 授权）。
