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

## H-011 书构建层 churn 控制（状态：**TESTED — 全部 REJECTED（0/5），EXP-011，2026-07-06**）

> 结果：换手门被 B2/B3/B4 以 4–17× 裕度通过（0.014–0.041 vs 0.10）且中位 CAGR 反升、F1 翻正——**churn 机制被证明**；但全员 G2/G3 失败：慢书在 F2 崩塌折死得更惨（−31.6~−43.0% vs 载体 −29.9%，worstDD 30.8–37.4% vs 25.0%）。结构性结论=每日重选是隐性崩盘防御，churn 控制与崩盘生存在本族**直接冲突**。附带发现：k=10 折级 CAGR 有 ±3pp/bps 级执行路径噪声（偶发 >20pp）⇒ 4 折 k=10 继续挖掘收益递减，下一步=k=30 宽书结构变化（H-012 待预注册）。详见 BOOK_CHURN_CONTROL_EXPERIMENT.md / EXP-011。

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

## H-012 k=30 宽书结构稳健性（状态：**TESTED — 全部 REJECTED（0/3），EXP-012，2026-07-06**）

> 结果：素 k30 换手反升（0.30–0.51，边界穿越随 k 增长）且崩塌不解（F2 −31.8%）；W2 k30+部分调仓=**基础设施级发现**（换手 0.008–0.015、中位 +33.9% 全场最佳、bps 噪声带 0.007 vs k10 的 0.088 = 12× 收窄——全周期唯一折级差异可信的形态）但崩塌惩罚不随书宽消失（F2 −39.4%）；W3 C2@k30 全面崩（中位 +16.4%）。**F2 暴露=信号级**，书层任何变换不可解。PBO 0.667；DSR 全<0.95；N=63。详见 EXP-012 / wf_h008/exp012_widebook/。

- 动机：EXP-011 发现③（k=10 折级 CAGR ±3pp/bps 路径噪声、>20pp 盆地跳变 → 参数挖掘到收益递减点）+ 发现②（k=10 集中书的崩盘冲击）+ EXP-004（k30 掺长改善最差季度）+ 容量/流动性（宽书聚合流动性更高）。结构性变化，非参数变体。
- **候选（N=3，先验冻结，跑后不改）**——同 H-008 4 折冻结 sleeve 预测，零重训：
  - **W1** = C3_ema0.7 分数 @ k=30 等权（前沿配置加宽）
  - **W2** = C3_ema0.7 分数 @ k=30 × B3 部分调仓（w_t=0.7w+0.3target，<0.005/3 比例剪除后归一）——宽书是否化解慢书崩盘冲突
  - **W3** = C2_prod_rank110 @ k=30（现生产候选加宽，对照）
- **评测**：variant-C 全约束；每候选 × 每折 × bps∈{8,9,10}（噪声带=3 bps 点位折 CAGR 极差，**报告用非选择用**）；k10 载体同样跑 3 bps 点位作参照带。判定用 8bps 点（与历届一致）。
- **验收门（与 H-011 同套，冻结）**：G1 maxTurn≤0.10 ∧ G2 worstDD≤0.2503 ∧ G3 F2≥−24.9% ∧ G4 中位≥+28.02% ∧ G5 sector≤0.33 ∧ G6 无杠杆 ∧ G7 隔离窗零接触；DSR≥0.95 仍为生产采纳门。部分通过按"哪些门"如实记录（如 crash-solved/upside-diluted）。
- 累计 N：60+3=**63**。预算：CPU 48 次 variant-C ≈10min，RSS<4G，磁盘<50MB。
- 纪律：k=30 是结构参数非搜索维度（不试 k∈{20,25,...}——只测预登记的 30，取自 EXP-004 已有先验）。

## H-013 低 churn 书 × 快速 regime de-risk 合成（状态：**INVALID — 处理从未被施加，EXP-013，2026-07-06；触发 INC-E1**）

> 结果：S1/S2 与 W2 基线逐点雷同触发执行取证 → 发现**执行模拟器跨日 (symbol,side) 静默吞单缺陷（INC-E1）**：每票整个回测最多买/卖各一次，F1 素书意图订单值 81.6% 被静默丢弃，R2a flip 日 invested 0.998 不动（overlay 从未表达）。EXP-008..013 全部结论戳 pre-INC-E1；修复补丁已拟未应用（红线：trusted evaluator 语义变更先问）。硬停止条款照常生效。详见 EVALUATOR_ORDER_DEDUP_BUG.md / 台账 INC-E1。原注册文本如下。

- 动机：周期结构图完成（EXP-011/012）：换手已解（partial-adjust）、路径噪声已解（W2 形态）、崩塌未解且为信号级（仅 R2a 型 regime 控制碰到过：F2 −16.8%）。唯一未测组合 = 三个已各自验证的机制的合成。
- **诚实注记（多重性）**：R2a 的 F2 表现是已花费的自由度（EXP-009/010 预注册选出后封存）；本批只测**组合方式**，N=2，台账 63→65。F2 折已被 ~63 次候选评估看过——本批结果无论多好，绝对量级不可信，只作 FRESH 窗前的机制排序依据。
- **候选（N=2，先验冻结，跑后不改，无第三次迭代）**——书=W2（C3_ema0.7 分数 @k30 部分调仓 0.7/0.3，prune 0.05/30，EXP-012 原样），gross=R2a confirm-5 MA60 状态机（EXP-010 原参数，{1.0,0.5}，t−1 观测 t 执行）：
  - **S1_instant**：gross 瞬时切换（原 R2a 语义）
  - **S2_ramp25**：gross 变动 ≤0.25/日（2 日过渡——比 B5 的 0.1/日快 2.5×，比瞬时便宜一半峰值交易）
- **评测**：同 H-012（4 折 × bps∈{8,9,10}，判定 8bps 点，噪声带报告）；PBO/DSR@N=65（与 W2 无 overlay + k10 载体同池）。
- **验收门（G1–G7 冻结不变）**：maxTurn≤0.10 ∧ worstDD≤0.2503 ∧ F2≥−24.9% ∧ 中位≥+28.02% ∧ sector≤0.33 ∧ 无杠杆 ∧ 隔离零接触。全过 ⇒ 写 PRODUCTION_CANDIDATE_PROPOSAL（明确"待 FRESH 裁决，不改生产"）；DSR≥0.95 仍为生产采纳统计门（预期不过）。
- 预算：24 次 variant-C ≈7min CPU，RSS<4G。累计 N=**65**。
- **硬停止条款**：本批后 H-008 4 折冻结（任何配置不得再评测），直到 FRESH 窗首读（≥120 交易日 ≈2026-11）或用户显式重开。

## H-017 基本面质量/成长 tilt 提升低换手核心（状态：**已登记 · 跑**，2026-07-07）

- **瓶颈诊断**：当前最佳 L1_c3ema07_minhold10 崩塌折 worstDD 36.6% 为信号级（载体=纯技术动量 sleeve 混合，无基本面锚）；D1 低波 tilt 修 DD 但收益减半。**假设：PIT-safe 基本面质量/成长（tickflow_fin_features：roe/net_margin/gross_margin/revenue_yoy/net_income_yoy——已存在但生产数据集未并入、sleeve 从未训练=未开发正交信号）作为 tilt 能以更小收益代价改善崩塌/正交加 alpha。**
- **复用**：扩展现有 `dual_track_factor_batch.py`（`--batch fundamental`，抽出 `score_factors` 共享评分）+ `augment_training_dataset.py` 的 merge 模式；不新建模块。PIT 安全：tickflow 特征按公告日 step，额外 +1 交易日 per-symbol lag。
- **候选（7，先验冻结）**：QF_roe / QF_net_margin / QF_gross_margin / QF_revenue_yoy / QF_net_income_yoy / QF_quality(rank-mean roe+net_margin+gross_margin) / QF_growth(rank-mean revenue_yoy+net_income_yoy)。每因子按日截面 rank（对 ROE 极端值稳健）。
- **评测**：窗口 2023-07-03..2025-08-29（pre-quarantine，断言）；rank_IC/ICIR@h10/h20、top-q 换手、LS cost-adj@8/15/25bps、F2 崩塌 IC、去相关（含 vs 技术 refs mom20/liq/lowvol20——测正交性）、capacity。验收：oriented-positive IC≥0.015 ∧ ICIR≥0.2 ∧ 换手≤0.15 ∧ 成本生还 ∧ 去相关簇保最优。
- **阶段 2（若有 survivor）**：复用 EXP-016 模式，best 基本面因子 rank 以 0.3 tilt L1 min-hold-10 书，对比 F2/DD/中位 vs L1 baseline vs D1 tilt。
- **预算**：CPU-only ≈2min 因子批 + ≈3min 集成；RSS<4G；零重训；零 fresh 接触。N 74→75（集成新配置计 1）。FRESH 为仲裁。

- **动机**：INC-E1 修正后 EXP-011 重跑（EXP011_CORRECTED_INC_E1.md）**推翻** pre-INC-E1 结论——真实换手 0.19–0.78（非 0.014–0.041 伪影），且 min-hold **改善**崩塌（非加深）。**B2_minhold10 在修正载体上 4/4 核心轴碾压**：中位 +36.4% vs +1.3%、换手 0.202 vs 1.035（5×↓）、F2 −40.2% vs −56.7%（+16pp）、DSR 0.427 vs 0.025（17×）；仅差绝对换手门（0.202>0.10）与绝对 worstDD（36.6%）。B5（buffer+R2a ramp）是唯一压住 worstDD 的（25.2%）。
- **设计（先验冻结 ≤4，跑后不改）**——载体 C3_ema0.7@k10 修正 sim：
  - **M1_minhold15**：min-hold 15 日（B2 的 10→15，进一步压换手）
  - **M2_minhold10_partial**：min-hold 10 + partial-adjust 0.5（B2×B3 复合，压增量再平衡换手）
  - **M3_minhold10_r2aramp**：B2 + R2a ramp 0.1/日（B2 收益 × B5 的 DD 控制——最可能同过 G1/G2/G3）
  - **M4_minhold20**：min-hold 20 日（换手–收益前沿慢端锚点）
- **验收门（vs 修正载体 + 绝对承诺）**：G1 maxTurn≤0.10（绝对）∧ G2 worstDD≤33.9%（修正载体）∧ G3 F2≥−51.7%（载体+5pp）∧ G4 中位≥+1.3%（载体）∧ G5 sector≤0.33 ∧ 无杠杆 ∧ 隔离零接触。全过且 DSR≥0.95 ⇒ PRODUCTION_CANDIDATE_PROPOSAL（待 FRESH 裁决，不改生产）。
- **纪律（关键）**：H-008 4 折已被 ~65 次候选看过；INC-E1 修正重跑经用户批准（不加 N）。**H-014 是新候选 = 新 fold-mining**，违反 H-013 硬停止条款。故**登记但不跑**——直到 (a) FRESH 窗首读（≈2026-11）提供真 OOS 仲裁，或 (b) 用户显式批准再花 4 折自由度。符合任务"不优化脆弱参数赢家"红线。预算（若跑）：≈16 次 variant-C ≈5min CPU，N 65→69。

## H-015 双轨换手比较（Track L 低换手 vs Track H 高换手，state：**DONE — Track L 验证 / Track H 拒绝，2026-07-07；见 DUAL_TRACK_RESULT_H015.md / 台账 EXP-015**）

> 结果：Track L 碾压 Track H。低换手 min-hold/reb 书（turn 0.19–0.20）中位 CAGR +27~+36%、med@25bps 仍 +17~+24%；Track H 快书全部不生还 25bps（最佳 H4 med@25 −10.7%）→ **Track H 因成本被拒**。换手控制（非周期）是杠杆。最佳稳健 = L1_c3ema07_minhold10（中位 +36.4%、首个正超额 +14.4%、turn 0.202）；最佳防御 = L3_midlong_minhold10（F2 −33.0%）。均非生产就绪（worstDD ~36% 崩塌折 + DSR<0.06，N=73，PBO 0.0）。原注册文本如下。


- **授权**：用户 dual-track 指令显式要求"同 date folds、同 evaluator、同 costs"跑 3–5 L + 3–5 H 候选 → 显式重开 H-008 4 折（覆盖 H-013 停止条款）。诚实记 N，FRESH 仍为真仲裁。
- **协议**：DUAL_TURNOVER_STRATEGY_PROTOCOL.md；harness `scripts/analysis/dual_track_eval.py`；corrected sim；8/15/25 bps；净指标裁定。
- **Track L 候选（4，先验冻结，跑后不改）**——载体从冻结 sleeve 预测构建：
  - **L1_c3ema07_minhold10**：C3_ema0.7 载体 + min-hold-10（= EXP-011 B2 复锚，已知最佳机制）
  - **L2_midlong_ema07**：mid_5d_30d + long_30d_120d rank-mean，EMA0.7，plain top-10（中期低换手信号）
  - **L3_midlong_minhold10**：L2 载体 + min-hold-10（中期 + 持有约束）
  - **L4_c3ema07_reb10**：C3_ema0.7 载体 + 每 10 交易日再平衡（节流）
- **Track H 候选（4，先验冻结，跑后不改）**——短周期快信号：
  - **H1_short_fast**：short_5d-only rank，plain top-10（快、高换手参照）
  - **H2_short_hyst**：short_5d + score hysteresis keep-zone（held 名 rank<2k 保留，抑噪 churn）
  - **H3_c2_fast**：C2_prod_rank110（short+mid rank）plain top-10（现生产族快书对照）
  - **H4_short_minhold3**：short_5d + min-hold-3（轻持有，切快信号噪声）
- **验收门**：见协议 §5（Track L / Track H 各一套；净指标；绝对生产承诺另计）。
- **参照**：corrected C3_ema0.7 载体（中位 +1.3%、worstDD 33.9%、F2 −56.7%、maxTurn 1.035、DSR 0.026）。
- **统计**：fold-block CSCV PBO + DSR across {8 候选 + 载体}；判定 8bps。累计 N 65→**73**。
- **预算**：CPU-only，8 候选 × 4 折 × 3 bps ≈96 次 variant-C ≈15min，RSS<4G，磁盘<50MB，零重训，零 fresh 接触。
- **纪律**：a-priori 可解释候选，无微参数扫；净指标裁定；折结果只排序机制不封生产赢家。

## 队列中未立项（见 IDEA_QUEUE.md）

H-004 sector 集中度约束收紧；H-005 长 sleeve 诊断价值（何时该有非零权重）；H-006 DSL 因子新批次（capped）；H-007 offline RL turnover-controller；H-008 walk-forward 重训协议（模型层，需 GPU 授权）。

## H-018 板块轮动 tilt（状态：**TESTED — REJECT（顺周期，崩塌 −25pp），EXP-018，2026-07-07**）

> 结果：板块动量 tilt 助弱/牛折（F1 +1.5→+16.3%、F3 +97→+112%）但崩塌灾难（F2 −41→−66.4%，worstDD 42.5%）。坐实 naive 板块动量 whipsaws。L1 baseline 仍收益冠军。原注册如下。
- 瓶颈/动机：任务显式列 板块轮动 为 critical；复用 `factors/sector_rotation.sector_relative_strength`（20d 板块相对市场强度）作为 tilt，测热板块倾斜是否加 return 或改善崩塌。**PIT 注意：sector_map=current_snapshot(2026-05-31) 成员=轻度非 PIT 泄漏（成员稳定，收益 PIT 安全）——诚实标注。**
- 候选 1（sector_rs tilt w=0.3 on L1 min-hold-10）；复用 `dual_track_d1_integration.py --factor sector_rs`；N 75→76。验收：改善中位 return 或崩塌/DD 且不破执行。Stage 8 先验=naive 板块动量 whipsaws，预期谨慎。FRESH 仲裁。

## H-019 regime-conditional D1 低波 tilt（状态：**TESTED — ACCEPT 机制（最佳 Calmar 1.14），EXP-019，2026-07-07**）

> 结果：仅崩塌 regime 施加 D1 → 中位 +25.3%（>静态 D1 +18.6%）、F2 −32.3%、**worstDD 22.1% 全场最低、Calmar 1.14 全场最佳**（唯一改善风险调整收益的 overlay）。残余成本=R2a 牛市回调误触发。收益冠军仍 L1 baseline +36.4%；风险调整冠军 L1+D1_regime。原注册如下。
- 动机：EXP-016 D1 静态 tilt 修崩塌但收益减半（始终去风险含牛折）。假设：**仅在崩塌 regime（R2a bench<MA60 confirm-5，t−1观测t执行）施加 D1 tilt(w=0.5)、其余全动量**，可保 +36% 牛折收益同时获崩塌保护——组合两已验证机制（R2a EXP-010 + D1 EXP-016），先验非扫参。
- 候选 1（d1_regime w=0.5 crash-only on L1 min-hold-10）；复用 `dual_track_d1_integration.py --factor d1_regime --weight 0.5`（reuse gross_series + D1，零新模块）；N 76→77。验收：F2/worstDD 改善 且 中位 return 不塌（接近 L1 baseline +36.4%，远好于静态 D1 +18.6%）。FRESH 仲裁。

## H-020 PIT 估值+基本面训练集集成（状态：**REGISTERED 2026-07-08，数据工程票，候选特征集已冻结；见 VALUATION_FUNDAMENTAL_INTEGRATION_PLAN.md**）

> 动机：诊断确认生产训练集 plus7clean（327列）**零 firm-level 估值/基本面值**，仅有 `missing_*` 占位旗标；但 ①模型 `horizon_models.select_features` 的 LONG 腿已按名模式 whitelist `roe/net_margin/gross_margin/revenue_yoy/debt_to_asset/valuation_percentile/…`（架构缺口=数据缺口）②`silver/fundamentals/metrics_panel.parquet` 已是 PIT-safe 面板（3654 syms，`announce_date`+`available_at`，eps/bps/ocfps/roe/margins/growth/debt/turnover）③估值输入齐备（per-share + 日 close），仅 `valuation` silver 目录为空。
- 复用（不造重复引擎）：metrics_panel 原样、`enrich_panel_fundamentals.py` 的 merge_asof PIT 模式、`financial_features.py` 作估值函数落点、`augment_training_dataset.py` 合并 harness、trainer 现有 name-pattern、修正后 strict_v8。新增≈1函数（TTM 去累计+比率+分位）+1薄物化脚本。
- 冻结特征集：直接基本面（roe/roe_diluted/net_margin/gross_margin/revenue_yoy/net_income_yoy/debt_to_asset/inventory_turnover/operating_cash_to_revenue）+ per-share 估值（pb=close/bps、eps_ttm 去累计、pe_ttm、ocfps_ttm、pcf、earnings_yield、ocf_yield）+ 截面分位（valuation_percentile、pb_own_pctile_2y）+ 复合（quality_composite、growth_composite）。**显式排除不可无造假构建者**：PS、EV/EBITDA、股息率、分析师预期、turnover_rate/market_cap（无股本）。
- 数据事实（已验证）：eps/ocfps 为 YTD 累计→需去累计 TTM；bps 为 PIT 时点→pb 直接安全；无股本数据。
- PIT 泄漏门（训练前必过）：G-PIT-1 每行 available_at≤trade_date；G-2 只用 announce_date<t；G-3 TTM 手算抽验（000001.SZ 2025Q3）；G-4 截面分位仅用当日；G-5 合并行数不变+不越下个 available_at 前填；G-6 隔离窗零接触。
- 验收（本票=纯数据工程，不动模型/生产、不作 CAGR 声明）：block 构建成功 + 全 PIT 门过 + 核心块（pb/roe/margins）覆盖≥90% 训练行且诚实 missingness + 合并行数不变 + schema 新版发出。
- 解锁（另行预注册）：**H-021** GPU 重训 + WF 消融（technical-only vs +fund vs +val vs full，修正模拟器下按 OOS 成本调整 CAGR 排序，nested WF，PBO/DSR，25/50bps，容量）——需 GPU 预算/门预注册；**H-022** 复用 `tplus1_engine.py` 的 T+1 做T。
- 累计 N：数据工程票不计 selection 试验（N 维持 77）。预算 CPU ≈15min，RSS<8G，磁盘 +~0.5G。

## H-021 GBM 消融：估值/基本面是否为模型增量 alpha（状态：**TESTED — GPU NO-GO，EXP-021，2026-07-08**）

> 结果：无一特征组达 +0.005 增量 IC 门 ⇒ **GPU (H-022) NO-GO**。base(301) OOS meanIC 0.182；+val 0.186（Δ+0.0038，ICIR 1.083→1.174，pb/book_yield/earnings_yield 进 top15）；+fund 0.180（伤）；full 0.178（伤）。诚实注记：base top15 由逐日常量 idx/macro 门控主导+全不可交易宇宙 ⇒ 0.18 夸大可交易 edge（phantom breadth）；估值 −0.09 标准 IC 大半被 size/技术轴吸收=冗余非无用。派生 H-022（可交易宇宙+截面 base 重测）。详见 EXP-021。原注册如下。

- 动机：H-020 证实估值有强截面 IC（pb 60d IC −0.091）但那是**单因子** IC；关键问题=在已含 alpha101/181+gtja+macro 的**非线性模型**里，估值/基本面是否提供**增量** OOS 预测力。[[full-universe-deep-mlp-no-edge]] 记载纯技术全宇宙 MLP OOS rank-IC≈0 无 edge（无估值特征）——本实验直接检验估值是否救活全宇宙模型。且 `--feature-policy judgment` 只从 horizon_factor_assignment_plus7.json（旧数据集产出）选特征→天真 GPU 重训会忽略新列；故先用 GBM 廉价确证增量价值再决定是否花 GPU（"不盲跑 GPU"）。
- 设计（N=4 预注册，先验冻结）：LightGBM 截面 ranker，label=forward_return_60d（逐日截面 rank 归一），特征 4 组消融：
  - A base（原 plus7clean 数值特征：alpha/gtja/macro/idx/flow，排除 label/key/flag）
  - B base + 估值（pb/pe_ttm/earnings_yield/valuation_percentile/pb_own_pctile_2y/pcf/ocf_yield/book_yield）
  - C base + 基本面（roe/roe_diluted/net_margin/gross_margin/revenue_yoy/net_income_yoy/debt_to_asset/inventory_turnover/operating_cash_to_revenue/quality_composite/growth_composite）
  - D base + 估值 + 基本面（full）
- 数据/切分（严格时序，隔离前）：train 2018-01-02..2022-12-31，embargo 60d，OOS test 2023-03..2025-08-29（隔离窗 2025-09-01 前，新鲜窗零接触）。相同超参跨 4 组（先验冻结：n_estimators 600、lr 0.03、num_leaves 63、subsample 0.8、colsample 0.7、min_child_samples 200、早停 valid=2023-01..2023-02）。
- 指标：OOS 每日截面 Spearman rank-IC（vs forward_return_60d）均值、ICIR、t；top-decile 组合日均超额（快速 proxy，不过执行模拟器，仅信号层）；特征重要度（新列是否进 top-30）。
- **验收/判定门（先验）**：B 或 D 的 OOS mean rank-IC 较 A **提升 ≥ +0.005 绝对**（≈ +实质 ICIR）且新估值列进入重要度 top-15 ⇒ 估值有增量 alpha → 立 H-022（GPU 深训：regen judgment 含新列 + 重训 LONG，另行 GPU 预注册/预算）；若 B/D 不优于 A（Δ<+0.002）⇒ 估值在非线性技术模型里无增量（模型不缺估值，缺别的）→ 不花 GPU，记录并转向。
- 复用：现有 gold 数据集 plus7clean_fund、forward_return_60d 标签、baseline_protocol 常量、修正评估器（若做 top-decile 需要则用 strict_v8，本票默认信号层 IC 不过模拟器）。新增 1 脚本 scripts/gbm_val_fund_ablation.py。
- 预算：CPU-only ≈10-20min，RSS<16G，磁盘<50MB。累计 N：+4（GBM 消融）→ 81。

## H-022 估值增量：可交易宇宙 + 截面-only base（状态：**TESTED — 确认冗余，GPU NO-GO，EXP-022，2026-07-08**）

> 结果：控制两混淆后估值仍无增量——A base_xs IC 0.15787（top-decile 多头 +4.58% t+9.2）；B +val IC 0.15911（ΔIC +0.0012），top-decile 多头 +4.44%（微降）。8/8 估值列进 top15 但净增量≈0 ⇒ **GPU NO-GO**。跨 2 宇宙×2 base 复制：估值作原始模型输入冗余于 alpha101/181+gtja191（价量库已张成 value/size 空间）；PIT 面板保留作 regime 条件/防御用途。详见 EXP-022。原注册如下。

- 动机：H-021 判 GPU no-go，但两处混淆使"估值无增量"结论不完整：①base top15 由**逐日常量** idx/macro/flow 特征主导（无截面信息，仅作 regime 门）②全宇宙含不可交易微盘 → size 效应吸收估值 + phantom breadth 夸大 IC。H-022 同时控制两者：**去常量特征（截面-only base）+ 限可交易宇宙**，直接检验估值增量是否是全宇宙冗余的伪影。服务"capacity-adjusted return"目标。
- 设计（N=2 预注册，先验冻结）：LightGBM 截面 ranker，label=forward_return_60d 逐日 rank：
  - base_xs = 仅**截面**特征（alpha*/gtja*/per-stock return_1d/momentum*/volatility*/amount*/volume*），**剔除逐日常量**（idx_*/macro_*/flow_*）
  - base_xs + val（+8 估值列）
- 可交易宇宙（先验）：每日 eligible（~is_st & ~is_suspended & ~is_limit_up）**且** amount_mean_20d ≥ 当日截面中位数（流动上半）。去掉不可交易微盘尾。
- 切分/超参：同 H-021（train 2018..2022-12-31，OOS 2023-04..2025-08-29，隔离前；LGBM 参数冻结同 H-021）。
- 指标：OOS mean rank-IC/ICIR/t；**top-decile 多头 60d 均收益 + 多空 decile spread（可交易 proxy，容量感知）**；估值列重要度。
- **判定门（先验）**：+val 较 base_xs OOS mean rank-IC 提升 ≥ +0.005 **或** top-decile 多头 60d 收益提升 ≥ 显著（t>2）⇒ 估值在可交易宇宙有增量 → 重开 GPU H-023 评估（另行预注册）；否则确认估值冗余（在此建模/宇宙下），停止估值线，记录。
- 复用：plus7clean_fund、gbm 脚本骨架（加 universe 过滤 + 常量特征剔除 + decile 收益）；**RAM 修复**：流动过滤减半行 + 剔除常量列 + 顺序训练显式 del，目标峰 <30 GiB。
- 预算：CPU ≈5-10min，磁盘<10MB。累计 N：+2 → 83。

## H-023 学习型 regime→tilt 权重元模型（状态：**TESTED — 两轴皆不过 ⇒ REJECT（先验门），EXP-023，2026-07-10**）

> 结果：RW1_4state 中位 +33.4%/worstDD 35.3%/Calmar 0.947/med@25 +21.3%——3/4 折胜 L1 baseline 但 A 轴差中位（33.4<36.4）、B 轴差 Calmar（0.947<1.14）；RW2_2state 全面弱（+24.2%/0.663）。**关键诚实发现：因果学习器无法复现 EXP-019 的崩塌保护（trailing 2018→2023 中 crash-highvol 态 D1 IC<0.01 → F2 只部分防护），手设规则的知识来自与折重叠的 2023-25 因子批窗口 ⇒ EXP-019 Calmar 1.14 部分 fold-informed，可信度下调**；vol-split 增值（RW1≫RW2）。全部硬门过（成本生还/换手/因果/零隔离）。FRESH 首读预登记三方对比：L1 / L1+D1_regime / RW1_4state。详见 EXP-023。原注册如下。

> 编号注记：H-022 判定门中提到的"GPU H-023"从未开启（门未过）；H-023 重新指派给本票（regime-aware factor-weight meta-model，CPU）。

- **瓶颈诊断**：EXP-021/022 双重复制判定特征线收敛（估值/基本面作原始输入无无条件增量 alpha）→ 瓶颈在 regime/book/执行。EXP-016..019 的 tilt 全是**手设** (factor, w, regime) 且设计序列经过看折迭代（fold-informed：D1 选中、w=0.3/0.5、crash-only 结构均在看到前次折结果后登记）；EXP-019（crash→D1 w=0.5）是目前唯一改善风险调整收益的 overlay（Calmar 1.14）。用户指令（2026-07-10）显式要求 regime-aware factor weights 全流程模型。
- **假设**：从每折 OOS 前的 trailing 数据（2018→t−embargo）按 regime 状态**学习** tilt 组件构成与权重 τ_s（纯因果、月度 refit、零折内知识、零手设结构），能 (A) 提升原始 CAGR 超过 L1 baseline，或 (B) ≥ 手设 EXP-019 的风险调整收益。若学习规则连手设都不及，反证手设 overlay 的成绩部分来自 fold-informed 设计序列（诚实记录）。
- **复用（零新引擎）**：dual_track_d1_integration harness 模式（carrier=C3_ema0.7 / book=min-hold-10 / strict_v8 修正 sim / H-008 4 折 / 8/15/25bps 全一致）、D1/quality/sector_rs tilt 组件函数、R2a gross_series（含 t−1→t shift）、R3 vol 阈值 0.25（repo 既有规则，非新搜）、bench_series。新增 1 脚本 scripts/analysis/regime_weight_meta.py。
- **设计（N=2 预注册，先验冻结，跑后不改）**：blend_t = (1−τ_s(t))·carrier_rank + τ_s(t)·tilt_rank_s(t)；book/sim 不变：
  - 组件宇宙（冻结 3）：D1 低波 / quality（roe+net_margin+gross_margin rank-mean，+1d lag）/ sector_rs（PIT 注记同 EXP-018：sector_map=current_snapshot，成员稳定、收益 PIT-safe，诚实标注）；动量代理 = rank-mean(ret5, ret20, ret60)（镜像 short/mid/long sleeve 结构）**仅用于 τ 的相对刻度，不进 tilt**
  - regime 状态（bench=eqw-all-A，t−1 观测 t 施加，与 gross_series 同 shift 语义）：**RW1_4state** = R2a trend(2 态) × bench 20d 年化 vol≥0.25(2 态)；**RW2_2state** = R2a trend 2 态（消融：vol 分割是否增值）
  - **学习规则（冻结，无搜参）**：对每 regime s，trailing 窗（2018-01-02 → t−11，11 日 embargo ≥ 标签视界）内 regime-s 日的每组件日截面 Spearman IC（vs 前向 10d 收益，h=10 对齐 min-hold-10 书）；tilt_s = 正 IC（≥0.01）组件按 IC 比例 rank-加权（全部 <0.01 → τ_s=0 纯 carrier）；**τ_s = 0.5 · IC_tilt⁺/(IC_tilt⁺ + IC_mom⁺ + 0.01)**，其中 IC_tilt = tilt_s 组合的加权平均组件 IC、IC_mom = 动量代理 IC、⁺=max(0,·)；cap 0.5 继承 EXP-019 已花 dof。月度 refit（每 21 交易日），折内 trailing 累积更新=因果
  - regime 最小样本 60 日，不足 → 回退该折无条件 trailing IC 权重
- **评测**：与 EXP-016..019 完全同门（variant-C、修正 sim、4 折、8/15/25bps、k=10 min-hold-10）。对照（已花 dof，不再重跑评价）：L1 baseline 中位 +36.4%/worstDD 36.6%/Calmar 0.99；EXP-019 d1_regime 中位 +25.3%/worstDD 22.1%/Calmar 1.14
- **验收门（先验，跑后不改）**：**A 轴（收益冠军挑战）**：中位 CAGR8 > +36.4% ∧ worstDD ≤ 36.6%；**B 轴（风险调整冠军挑战）**：Calmar(中位 CAGR8/worstDD) > 1.14 ∧ 中位 CAGR8 ≥ +25.3%。任一候选任一轴过 ⇒ ACCEPT 该轴机制；两候选两轴皆不过 ⇒ REJECT 学习型 regime 权重线（本建模下），记录并停线——**不得看折后修改学习规则再跑**。硬门：med@25bps > 0 ∧ maxTurn ≤ 0.25 ∧ 隔离窗（≥2025-09-01）零接触 ∧ regime/标签全因果（embargo 断言）
- **纪律**：H-008 折已被 ~83 次候选看过——本批 +2，N 83→**85**；用户 2026-07-10 全流程指令 = 显式授权继续用折（同 H-015 先例）；折结果只作机制排序，FRESH（≈2026-11）为真仲裁。**GPU 注记：本模型自由参数 ~几十（组件×regime），GPU 无增值；GPU 深训线维持 H-021/022 预注册 NO-GO。若本票证明 regime 条件化在 book 级加 alpha，GPU 级 regime 模型（MoE/深度 regime 门控）另行预注册。**
- **预算**：CPU ≈10-20min（trailing IC 面板 2018→2025 ~3-5min + 2 候选×4 折×3 bps sim ~8min），RSS<8G，磁盘<20MB，零重训，零 fresh 接触。

## H-024 T+1 做T overlay 于冠军书（状态：**DISPOSED 不跑——数据不可行 + 先前已有定论，2026-07-10**）

- 用户指令要求"full model combined with T+1 intraday trading"。核查结果（不烧试验数）：
  1. **数据不可行**：minute_bars silver 覆盖 **2025-06-16..2026-06-12**（675 syms=旧持仓宇宙）。H-008 折窗（2023-07..2025-08-29）与之重叠仅 ~2.5 个月；其余分钟数据全在隔离/FRESH 窗（≥2025-09-01）内不可触。折上做T回测无数据基础。
  2. **问题已被回答（两层独立证据）**：①深历史日频代理做T = DO_NOT_ENABLE（[[dot-t-board-rl-fixes]]）；②2026-06-17 成本敏感 EV 做T闭环（1分钟/OHLCV/持仓宇宙）**定论无可实现 edge**——edge-frontier rank-IC≈0，top-predicted 分钟在 maker 10bps 成本下仍 −19~−26bps 净（[[dot-t-ev-engine-rebuild]]）。引擎保留为 NO_TRADE-默认 do-no-harm 覆盖层。
- **裁定**：在现有数据层（日频 + 1分钟 OHLCV）做T不增加真实 CAGR，只增加 churn——按任务要求如实上报。重开此线的唯一先决条件 = 真 Level-2 订单流数据（买卖队列/逐笔），届时另行预注册。不新增代码、不计 N。

## H-025 有据可查的价量因子批次 3（低换手 + 中换手 re-gate，状态：**REGISTERED 2026-07-13，EXP-025，跑前登记**）

- **来源**：全宇宙稳健性任务 fu_20260713（Phase 0 审计裁定：模型重训线无正当性，bounded 增量 = IDEA_QUEUE #8 因子批次）；batch-1 显式遗留 TODO（D6 vol-compression 中换手 track）；外部来源治理见 docs/research/external_factor_source_registry.md（全部学术一手公式或显式标注 approximation，零虚构专有公式）。
- **假设**：现库 alpha101/181+gtja191 未显式张成的**有据**价量族（彩票 MAX / 已实现偏度 / 下行半波动 / vol-of-vol / 隔夜分解 / FIP / 量稳 / 量价背离 / CGO 日线近似 / 缩量）中，存在通过 tradability-aware 验收门的低换手候选，可为**下一代（post-FRESH）** carrier/模型积累经审核因子。冠军三元组已冻结（19e05f4），本批产出**不做**书级集成、**不进** FRESH 首读集。
- **候选空间（N=13，冻结，无参数扫描；公式=DSL，方向先验声明，判错即 reject 不得翻符号）**：
  M1_max_ret_neg_20 = −TsMax(r1,20)［defensive］；M2_skew_neg_20 = −TsMean((r−TsMean(r,20))³,20)/(TsStd(r,20)³+ε)［defensive］；M3_pv_corr_neg_20 = −TsCorr(Close,Volume,20)；M4_volume_quiet_5_60 = −Log(TsMean(V,5)/(TsMean(V,60)+ε)+ε)；M5_clv_20 = TsMean((2C−H−L)/(H−L+ε),20)；M6_overnight_neg_20 = −TsMean(Open/Delay(Close,1)−1,20)；M7_vov_neg_20 = −TsStd(TsStd(r,5),20)［defensive］；M8_semivol_neg_20 = −TsStd(r·(1−Sign(r))/2,20)［defensive］；M9_liq_shock_neg_20 = −(Log(V+1)−TsMean(Log(V+1),20))/(TsStd(Log(V+1),20)+ε)；M10_vol_cv_neg_20 = −TsStd(V,20)/(TsMean(V,20)+ε)；M11_fip_20 = Sign(Returns(C,20))·TsMean(Sign(r1),20)；M12_cgo_vwap60_neg = −(C/(TsSum(Amt,60)/(TsSum(V,60)+ε))−1)［approximation 标注］；D6R_vol_compression_regate = −TsStd(r,5)/(TsStd(r,60)+ε)［defensive_medium，batch-1 原样重判于中换手门］。
- **窗口**：评测 2023-07-03..2025-08-29（pre-quarantine，harness 断言）；F2 crash 子窗 2024-01-02..2024-06-28。**禁区**：2025-09-01..2026-05-18（burned）与 2026-05-19+（FRESH）零接触。
- **选择指标与门（先验，跑后不改，同 batch-1 语义 + 两处显式收紧）**：
  1. g_ic：oriented rank-IC(h10) ≥ +0.015 ∧ (ICIR(h10) ≥ 0.20 ∨ ICIR(h20) ≥ 0.20)；
  2. g_turn：低换手类 top-quintile 日换手 ≤ 0.15；**defensive_medium 类 ≤ 0.35 且 LS@25bps ≥ 0.5×LS@8bps（成本衰减 ≤50%）**［新，中换手 track 的先验成本门］；
  3. g_cost：LS@8bps > 0 ∧ LS@25bps > 0；
  4. g_crash（defensive/defensive_medium 类）：F2-crash rank-IC ≥ 0；
  5. **g_novel［新，本批显式门］：max |Spearman| vs REF ≤ 0.85**，REF = {mom20, liq, lowvol20, rev60=−Returns(C,60), pv_ret_corr=TsCorr(r1,V,20)}（lowvol20 代理已物化 survivor D1；rev60/pv_ret_corr 代理已知强因子/既有 survivor 族）；
  6. g_decorr：solo-passer 间 |corr|>0.90 集群 keep-best（按 |ICIR|）。
- **试验数**：+13。因子筛选族累计 N = 7(batch1) + 7(batch2) + 13 = **27**；全库累计 N 85 → **98**（DSR 校正按族累计口径）。
- **资源预算**：CPU-only ≤15min，RSS ≤8 GiB，新磁盘 ≤50 MB（ledger CSV + 报告），零 GPU，零重训，零 variant-C 回测。
- **产出路径**：FACTOR_CANDIDATE_LEDGER_batch3.csv（根目录，续 batch1/2 惯例）+ runtime/reports/full_universe/fu_20260713/factor_screen_leaderboard.csv + 台账 EXP-025。
- **接受后动作（先验声明）**：survivor 物化为 reviewed synth_*（不入生产、不入冻结冠军、不做折内书集成——EXP-023 已证手设集成序列 fold-informed，重蹈无意义）；排队至 FRESH 首读后的下一代周期。0 survivor = 有效负结论，照实入账。
- **标签口径注记（诚实）**：筛选用 forward_return_labels = 同日 close→close(t+h)（与 batch-1/2 完全一致，跨批可比）；此口径较生产 delay-1 executable 标签轻度乐观 ⇒ 筛选 IC 只作排序/门槛用，不作收益声明（任务书 Stage-B 语义）。

## H-026 统一因子整合 + 增量模型消融 + 条件严格回测推进（状态：**TESTED — GPU_NO_GO / REJECT_INCREMENTAL_FACTOR_POOL（GBM 特征空间），EXP-026，2026-07-13；ridge 诊断揭示线性层出路（IDEA_QUEUE #13，post-FRESH）**）

- **来源**：H-025 后续任务指令。H-025 只完成静态验证+单因子筛选，未证明 6 survivor 对模型/组合有**增量**价值。本票在不触任何禁窗的前提下走完 C1-C3（整合/共同标签重评/全局去冗余）→ Phase D（CPU 增量消融）→ 条件 GPU/严格回测。
- **因子池（精确成员，冻结）**：
  1. 生产 = FT 三 sleeve judgment schema 联合 **127 特征**（short 22/mid 22/long 90，含 18 synth/llm；快照 runtime/reports/full_universe/fu_20260713/production_feature_union.json）。消融 M0 基线用**更严的宽基**（见下）。
  2. 历史 survivor（不在生产数据集列内）= **{D1_low_vol_20}**（batch-1）。
  3. H-025 survivor = **{M3_pv_corr_neg_20, M4_volume_quiet_5_60, M7_vov_neg_20, M10_vol_cv_neg_20, M12_cgo_vwap60_neg, D6R_vol_compression_regate}**。
  4. conditional-only（登记在册、**不进**模型组）：QF_roe/QF_net_margin/QF_quality（crash 条件，EXP-017）；PIT 估值 8 列（regime 条件保留，EXP-020/022）；D1_regime overlay（冻结冠军成员）。
  5. **复议历史 reject = 0**：逐条对照四类失效条件审计（无评估器缺陷影响过因子筛——INC-E1 修复早于全部批次；无新数据；冗余源 D1/lowvol 仍在池；换手/成本/方向拒绝对 tilt 与模型输入两用途均保守有效）——审计结论逐行写入 registry `reconsideration_reason`。
- **标签（共同协议）**：gold `_exec_` 数据集标签已是 **delay-1 executable**（IC016 审计回归锁定）：主门 `forward_return_20d`（匹配 survivor 8-12 天持有；诚实声明：H-021/022 先例用 60d，本票 20d 为 survivor-horizon 匹配选择，**跑前声明**，60d 全程并报为诊断，不得跑后换轴）。C2 面板重评用同构自建 exec 标签（close(t+1+h)/close(t+1)−1，t+1 涨停/停牌=入场不可行剔除），原 H-025 同日 close 指标全保留不覆盖。
- **宇宙**：EXP-022 tradable（eligible ∧ amount_mean_20d ≥ 当日中位）。**Track A 标签 = `fixed_cohort_searched_validation`**（3,872 固定 cohort，无 2020 后新股——不得称"full universe"）。Track B = 修复审计+缺口报告（本票不静默替代）。
- **模型组（冻结）**：M0 = EXP-022 `base_xs`（截面-only，剔 idx_/macro_/flow_ 与估值/基本面 NEW_ALL，~301 列）；M1 = M0+{D1}；M2 = M0+{6 H-025}；M3 = M0+{7 survivor 经 C3 剪枝（互 |ρ|>0.90 集群按稳健分留一）}；M4 = M0+{RC7 = 7 survivor 逐日等权 rank-mean 单列复合（先验定义，零拟合）}。去重后组空/同 = 如实记录不造假试验。
- **模型**：门控模型 = LightGBM regressor，**参数与 EXP-022 逐项相同**（600 trees/lr 0.03/leaves 63/subsample 0.8/colsample 0.7/min_child 200/λ1.0/early-stop 40）；诊断模型 = Ridge α=10（逐日 rank-pct 特征、折内中位数插补、隔日训练抽样），只报告不参与门判定。特征处理全部折内拟合。
- **外折（冻结，purged expanding）**：train_end ∈ {2021-12-31, 2022-12-31, 2023-12-31}；test_start = train_end 后**第 62 个交易日**之后的首个交易日（embargo ≥ 60d 标签视界+1 delay）；test_end ∈ {2023-03-31, 2024-03-29, 2025-08-29}（互不重叠，全部 pre-quarantine，断言）；valid = train 末 40 日（early-stop 用）。纯折外预测。**不用 H-008 折结果设计权重**。
- **增量门（先验，跑后不改；沿用 H-021/022 门值）**：某组通过 iff（h20 主轴）：
  A) 3 折配对 Δmean-daily-rankIC 的中位数 ≥ **+0.005** ∧ ≥2/3 折 Δ>0 ∧ ≥2/3 折有新列进 top-15 importance；或
  B) 经济替代门：Δtop-decile 多头（逐日 h20，成本前）≥2/3 折 >0 ∧ 合并 t≥2 ∧ 中位 Δ ≥ +0.001 ∧ top-decile 换手不劣化 >10%。
  **GPU go/no-go**：任一组过门 ⇒ 允许 FT 重训（架构冻结、只动因子集、另行 GO 记录）；全不过 ⇒ **GPU_NO_GO**，Phase E 不进入。严格 variant-C 终评上限 2 个 finalist（仅在过门后）。
- **试验数**：+4（M1..M4；M0=基线不计）。增量消融族累计 4(H-021)+2(H-022)+4 = **10**；全库 98 → **102**。C3 代表选择为确定性先验稳健分（rank(exec_ICIR)+rank(exec_LS25)+rank(crash_IC)−rank(turnover)），不计 N。
- **资源**：Phase D CPU-only，RAM ≤48 GiB（预期 ~35，EXP-022 实测 31.7）、时长 ≤2.5h、新磁盘 ≤1 GB、GPU=0（除非过门后另行记录）。禁窗：2025-09-01..2026-05-18 与 2026-05-19+ 零接触（断言）。
- **产出**：runtime/reports/full_universe/fu_20260713_h026/{factor_master_registry.{parquet,csv}, factor_correlation_matrix.parquet, factor_clusters.csv, factor_cluster_report.md, ablation_results.json, fold_metrics.csv, preregistration.json, …}；全部结果标 `candidate_research_only_not_fresh_holdout_validated`。

## H-027 线性残差 alpha + learning-to-rank + 替代表格模型基准（状态：**TESTED — 13/13 全不过门 ⇒ REJECT_ALL_ALTERNATIVE_MODELS，EXP-027，2026-07-13；H-026 线性增量被交叉拟合证伪（残差反转）；LambdaMart 换手发现 → IDEA_QUEUE #14**）

- **来源**：EXP-026 ridge 诊断（线性 ΔIC +0.0058，3/3 折正；GBM 冗余）。核心问题：该残差信息应经 ①交叉拟合线性残差 sleeve ②ranking 对齐 boosted-tree ③现代表格神经网络 ④受约束 OOF ensemble 中哪条路径表达——或在正确的交叉拟合下根本不存在（H-026 的 ridge 增量是"线性 vs 线性"，对 GBM 基线的残差增量未经检验）。
- **不变量（与 EXP-026 逐项相同，直接可比）**：3 expanding purged 折（train_end 2021/2022/2023-12-31，embargo 62 交易日，test end 2023-03-31/2024-03-29/2025-08-29）；label = gold `_exec_` delay-1 executable forward_return_20d 逐日 rank；宇宙 = tradable 固定 cohort（Track A，`fixed_cohort_searched_validation`）；冻结 FRESH 三候选零变更；禁窗零接触（断言）。Track B 修复不与本票混合。
- **基线 B0** = EXP-026 LightGBM M0（参数逐项同）。EXP-026 未持久化预测 ⇒ 同代码路径/参数/折重生成并本票起持久化 OOF（如实注记，非新试验）。
- **候选矩阵（冻结，13 主 + 2 条件 = max 15 trials）**：
  - **残差 sleeve**（特征 = 7 池因子逐日 rank-pct + fillna 0.5，折内处理）：外折 train 内 3 块时序 block 交叉拟合基线（各块由另两块训练的 LGBM 预测；残差模型永不见 outer-test）；残差目标 = rank_pct(y) − rank_pct(pred_cf)；**L1 = Ridge α=10**，**L2 = ElasticNet α=1e-3, l1_ratio=0.5**（固定）；最终分 = rank_pct(B0_test) + λ·resid_pred，**λ ∈ {0.10, 0.20, 0.30} 固定网格，每 λ 计 1 trial（共 6）**；多 λ 过门时取最小 λ（先验确定性规则，禁看折选 λ）。
  - **Ranking 树**：**X0** = XGBRanker M0 / **X1** = XGBRanker M3（objective=rank:ndcg, tree_method=hist, eta 0.05, max_depth 8, min_child_weight 200, subsample 0.8, colsample 0.7, n_estimators 600, early-stop 40 on valid NDCG@100, lambdarank_pair_method=topk, lambdarank_num_pair_per_sample=8, qid=trade_date）；**C1** = CatBoostRanker LambdaMart M3（iterations 600, lr 0.05, depth 8, early-stop 40, NDCG@100；已探测可构造）；**LG1** = LightGBM rank_xendcg M0（H-026 基线是 regression 目标 ⇒ 非同一，成立为 trial；其余参数同 B0）。梯度等级 = 逐日 h20 五分位 {0..4}（固定分箱）。树模型特征 = 原始值（NaN 原生，同 H-026 惯例）。
  - **深度表格（GPU，本票任务指令即授权；VRAM 记录，OOM fail-closed，禁静默降 CPU）**：**T0** = TabM M0 / **T1** = TabM M3（tabm 0.0.3，TabM.make 默认 arch：n_blocks 3, d_block 512, dropout 0.1, k=32, 无数值嵌入；AdamW lr 1e-3 wd 3e-4, batch 4096, AMP, ≤25 epochs, patience 5 on valid 日 RankIC；目标 = 逐日 rank-pct，MSE over k 头均值；特征 rank-pct+0.5 填充）；**R1** = pytabkit RealMLP_TD_Regressor M3（TD=调优默认值即其设计，零搜索；device cuda；同 rank-pct 特征）。禁跑 TabR/ModernNCA/TabPFN/MASTER/HIST/AutoML。
  - **条件 blend**（仅当组件独立过门）：**E2** = 最佳过门 ranking 模型 + L1 残差；**E3** = 最佳树 + 最佳深度 + L1。权重固定先验（E2 = 0.7/0.3，E3 = 0.6/0.2/0.2 rank 融合；新 sleeve ≤30% 帽），零折内拟合。
- **预测门（vs B0，全 AND，先验）**：①3 折配对 ΔRankIC 中位 ≥ +0.005 ②≥2/3 折 Δ>0 ③合并逐日配对 t 检验 p<0.10 ④崩塌折护栏：F1（2022 熊）ΔIC ≥ −0.002 ⑤贡献跨折稳定（blend/λ 符号一致，报告）⑥成本调整 top-decile 代理改善 >0（25bps，中位）⑦25bps 代理保持正号 ⑧top-decile 换手增幅 ≤10%。**跑后不降门。**
- **严格回测推进**：仅过门候选，S1 + 结构不同的 S2（≤2），variant-C 8/15/25bps × 1M/10M CNY，book 参数全同冠军协议；全不过 ⇒ 0 次严格回测。
- **试验数**：+13 主（L1×3λ, L2×3λ, X0, X1, C1, LG1, T0, T1, R1）+ E2/E3 条件 ⇒ max +15。全库累计 102 → **117**（模型基准新族 15）。
- **资源**：总墙钟 ≤4h（stage1 CPU ≤90min / stage2 GPU ≤90min）；RAM ≤48 GiB；VRAM ≤20 GiB（torch.cuda.max_memory_allocated 记录）；新磁盘 ≤2 GB（OOF 预测持久化）；分阶段脚本，单候选失败分类记录不拖垮全局。
- **产出**：runtime/reports/full_universe/fu_20260713_h027/{preregistration.json, oof/*.parquet, candidate_metrics.csv, gate_verdicts.json, stage逐日IC存档, final_summary.md}；全部标 `searched_validation` + `candidate_research_only_not_fresh_holdout_validated`。
