# EVALUATION_PROTOCOL_V2 — 评测协议 v2（Phase 2 提案，替代"单一 holdout"制）

> 前提事实：silver panel 最后交易日 = **2026-05-18**（parquet footer 元数据实测）；今天 2026-07-03。
> 旧 holdout（2025-09-01→2026-05-15/18）已烧毁（≥35 次直接评测，见 `HOLDOUT_CONTAMINATION_AUDIT.md`）。
> **结论：盘上当前不存在任何干净的 final holdout。** 现实世界存在 2026-05-19→今 的约 6 周新数据，尚未入库 —— 这是唯一能"长出"新 holdout 的来源。

## 1. 时间轴分区（即刻生效）

| 分区 | 窗口 | 用途 | 规则 |
|---|---|---|---|
| TRAIN | 2018-01-02 → 2024-06-30 | 模型拟合 | 不变；embargo 30d |
| SEARCH/VAL | 2024-08-28 → 2025-08-31 | 一切搜索/选择/调参 | 允许，但每次搜索必须登记试验数 N（见 §4） |
| **QUARANTINE** | **2025-09-01 → 2026-05-18** | 禁区 | 禁止一切新评测与选择；评测器守卫强制（patch P4）；已有读数只可引用带 `contaminated_holdout` 标签 |
| **FRESH FORWARD** | **2026-05-19 → 未来** | 新 final holdout | 只进不出：入库后**冻结**，只允许预注册配置各评测一次 |

## 2. 新鲜 holdout 的建立（唯一途径 = 时间前进）

1. 用 `scripts/update_market_panel_daily.py` 将 panel 补到最新（≈2026-07-02，+~30 个交易日）。数据入库≠可评测：先积累。
2. **预注册**（在评测前写入 `configs/preregistered_evals.json`，含 git hash）：待测配置最多 **3 个**（建议：① plus7 3-sleeve 先验 blend k10；② 生产 2-sleeve rank blend k10；③ walk-forward 重选出的新候选），每配置在 FRESH 窗**只评一次**。
3. **最短窗长门槛**：≥ **120 个交易日**（≈6 个月，约 2026-11 中）才做首次正式读数；之前只积累 forward paper 日志，不出数字结论。20–30 个交易日的年化噪声足以再造一个 38.6% 幻觉。
4. 每次 FRESH 窗访问经 P4 守卫留痕（`holdout_access_log.jsonl`）。

## 3. 隔离窗（QUARANTINE）规则

- 允许：引用**已存在**的 census 内读数（带分类标签）；负结论（overlay REJECT 等）沿用。
- 禁止：任何新回测、任何用该窗做的比较/选择/汇报，包括"只是看一眼"。
- 例外流程：确需诊断（如复现 bug）时 `--allow-quarantined "<原因>"`，留痕并在实验台账登记，产出数字不得用于选择。

## 4. 无新数据期间的选择协议（现在→2026-11）

一切改进只能用 TRAIN+SEARCH 分区，按以下三层验证：

### 4a. Purged/embargoed walk-forward（主力）
- 基建已有：`training/splitters.py`（purged/rolling/expanding）+ `run_walk_forward_deep_training`（schema-locked 折叠）+ variant-C。
- 折叠设计（模型层改动）：expanding，折验证窗=季度，embargo=30d（≥最长 label 126d 时用 126d——**修正点：embargo 必须 ≥ 最长 horizon**，现行 30d 对 120/126d label 不足，这是本审计新发现的次级泄漏风险，列入 LEAKAGE 跟进）。
- 报告口径：**逐折 variant-C 年化的分布**（min/median/max + 亏损折占比），禁止只报均值。
- 便宜层（blend/policy/book 改动）不重训模型：在冻结的 sleeve 预测上做折内选择+折外评测。

### 4b. CPCV / PBO（过拟合概率，量化 multiple testing）
- 对每次"搜索"（N 候选）：保留全部候选在 SEARCH 窗的**逐日净值**（`ensemble_search_plus7/_tmp` 已具备此形态），跑 CSCV/PBO（组合分块 S=8→70 个 train/test 划分，纯 pandas，分钟级、内存小）。
- **验收线：PBO ≤ 0.25**；同时报告赢家的 **Deflated Sharpe Ratio**（用试验数 N、偏度、峰度校正；DSR>0.95 置信才算显著）。
- **第一个立即可做的动作（零新数据、零 holdout 接触）：对已存在的 27 候选 ensemble search 做回溯 PBO/DSR** —— 直接量化 38.6% 赢家的过拟合概率。这应是 Phase 6 实验 #1。
- 试验数 N 必须按**历史累计**计（同族搜索多轮要累加），登记于 `HYPOTHESIS_REGISTRY.md`。

### 4c. 稳健性横切（对任何候选赢家）
- 邻域稳定：权重/k 在 ±1 格邻域的 val 表现不塌方（赢家孤峰 = 拒绝）。
- 子窗一致：SEARCH 窗二分/三分后各段方向一致。
- 成本敏感：slippage 8→15→25bps 斜率报告。
- 换手/容量门（数值门槛见 `ACCEPTANCE_RULES.md`）。

## 5. Forward paper 协议（终极裁判）

1. 前置：P6 完成（forward 推理对齐生产、特征保真 >0.99）——当前 forward 路径指向 v8.8 旧模型，**修复前所有 forward 数字无效**。
2. 每日收盘后：冻结模型打分 → 决策链 → 记录 target_weights 至 append-only `runtime/reports/forward_paper/`（含配置 hash、git hash）。
3. 不回填、不重算历史；月度汇总 variant-C 口径 vs 等权全A。
4. 模型/配置变更 = 新 track_id 另起序列，旧序列不删。
5. live-readiness 讨论的必要条件：FRESH 窗（§2）+ forward paper ≥3 个月方向一致。

## 6. 汇报纪律

任何数字出现在报告/UI/对话中必须携带四元组：**{窗口, 试验数 N, 信任分类, artifact 路径}**。违反即视为 marketing number。

## 7. 本协议的已知局限（诚实声明）

- SEARCH 窗（2024-08→2025-08）自身已高度复用（数十轮搜索），其上的 val 数字系统性偏乐观 —— 这正是必须等 FRESH 窗的原因。walk-forward 逐折分布 + PBO 是当前能做到的最强纠偏，不是完美解。
- 2018→2024 训练窗内的市场结构变化（注册制、量化监管、微盘风格）意味着逐折分布可能宽——宽就是真相，别用均值抹平。
