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

## H-001 族稳健 blend 优于单点赢家（状态：REGISTERED，待跑 EXP-001）

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

## H-002 换手感知平滑（状态：REGISTERED，待跑 EXP-002，依赖 H-001 出参考配置）

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

## 队列中未立项（见 IDEA_QUEUE.md）

H-004 sector 集中度约束收紧；H-005 长 sleeve 诊断价值（何时该有非零权重）；H-006 DSL 因子新批次（capped）；H-007 offline RL turnover-controller；H-008 walk-forward 重训协议（模型层，需 GPU 授权）。
