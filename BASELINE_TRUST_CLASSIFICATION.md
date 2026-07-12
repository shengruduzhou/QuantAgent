# BASELINE_TRUST_CLASSIFICATION — 生产指标信任分级（Phase 2）

> 分类日期 2026-07-03。类别定义（从严）：
> - `clean_oos`：配置先验固定（pre-registered / 代码默认），在样本外窗口**首次**评测，评测结果未反哺任何选择。
> - `walk_forward_oos`：多折 purged/embargoed 走式样本外的折间聚合。
> - `searched_validation`：在 validation 窗上多候选搜索后选出（数字本身是搜索赢家的 val 表现）。
> - `contaminated_holdout`：数字的产生**或采纳**使用了 holdout 窗信息（含"先看多个 holdout 读数再挑"）。
> - `in_sample`：训练窗内。
> - `unknown`：lineage 不足以判定。
>
> 原则：**分类取最坏适用类**。一个 val 上干净搜索出的赢家，如果其 holdout 读数是在多次窥视语境中被采纳的，就是 `contaminated_holdout`。

## 逐项分类

| # | 指标 | 数值 | artifact | 分类 | 理由 |
|---|---|---|---|---|---|
| 1 | **生产宣称：2-sleeve rank blend, k10, holdout CAGR / Calmar** | **+38.6% / 3.26** | `ensemble_search_plus7/winner_heldout/backtest/metrics.json` + `PRODUCTION_CONFIG.json` | **`contaminated_holdout`** | 搜索程序 val-only（干净），但：(a) 搜索因 8.3% holdout 读数而启动；(b) k-grid 由前日 holdout k-sweep 收窄；(c) 采纳决策发生在同窗 ≥9 个替代读数已知之后 = max-over-looks；(d) 无 PBO/DSR 校正。 |
| 2 | 3-sleeve 0.30/0.45/0.25 blend, k10, holdout | +8.3% | `realtest_plus7_holdout_top10` | **`clean_oos`**（带¹） | blend 权重为代码历史默认（先验），plus7 预测在该窗的第一批读数，结果未被用于产生它自己。¹窗口本身已被前代模型消费（topk_fine 等），故其"holdout"属性已降级为"一次干净的 OOS 单窗读数"。 |
| 3 | 3-sleeve blend, k20, holdout | 见 census | `realtest_plus7_holdout_top20` | `clean_oos`（带¹） | 同上。 |
| 4 | 2-sleeve **平均** blend, k10/k20, holdout | +14.5% (k10) | `retrain_plus7_.../realtest_2sleeve_holdout_*` | **`contaminated_holdout`** | "2-sleeve"这一配置因 long sleeve 重训失败而临时构造并直接在 holdout 上评测；该读数进入了后续 blend 决策语境。非先验、非 val 选出。 |
| 5 | 等权全A benchmark（holdout 窗） | +19.9% | 各 metrics.json `benchmark_annualized_return` | **`clean_oos`** | 机械计算、零参数、零选择。当前最可信的单一数字。 |
| 6 | v8.9 rankfix composite, k50, 2024-08-28→2026-05-15 | **+17.25% / maxDD 10.9% / Sharpe 1.34** | `v89_closed_loop/baseline/baseline_v89_current.json` | **`clean_oos`**（带²） | 单次评测，k50 为 baseline_protocol 默认，模型训练≤2024-06-30。²窗口混合了 validation 期(2024-08→2025-08)与 holdout 期，且此窗其后被大量搜索复用——数字本身产生时干净，作为对照 baseline 使用时须注明"全 OOS 混窗、单看"。 |
| 7 | v8.9 rankfix, k10, holdout | +17.9% | `realtest_current_holdout_top10` | `clean_oos`（带¹） | k=10 的选择时点存疑（前日 topk_fine holdout sweep 已看过 k10），保守可降为 `contaminated_holdout`；按证据链记 clean 带重大脚注。 |
| 8 | topk_sweep / topk_fine 全部 k 网格数 | 各值 | `topk_sweep/*.json`, `topk_fine/*.json` | **`contaminated_holdout`** | 显式 k 探索，窗口含/等于 holdout。 |
| 9 | factor combo 赢家（heldout 读数） | 见 census | `factor_combo_search/combo_heldout` | **`contaminated_holdout`** | val 上贪心搜索（且用 fwd1d 简化代理，非 variant-C）+ post-hoc holdout 看一眼；未采纳进生产。 |
| 10 | regime_search 全部候选 + 两个 winner 书 | y2026 ann +68.0% / +57.4% | `regime_search/leaderboard.json`, `winner_w111_k5_maxret`, `winner_w210_k10_robust` | **`contaminated_holdout`（最重度）** | 选择函数 finalscore **直接以 0.55–0.7 权重使用 y2026（holdout 内）表现**；且 4.4 个月子窗年化夸大 + 日换手 0.81–0.86 的容量/成本现实性存疑。 |
| 11 | v8.8 各 sleeve/blend sweep 数 | 各值 | `runtime/reports/v8/deep/v88_*` | **`contaminated_holdout` + 数据污染** | test 窗含 holdout 用于代际选择；且 v8.8 数据集本身 batch-rank 污染（rankfix 前）。双重不可用。 |
| 12 | +7 因子的验收 IC/ICIR | 表内各值 | `pooled_eval_clean/factor_eval_table.csv` | **`searched_validation`** | 多候选(45)搜索后验收，OOS 窗≈2024-08→2025-08-31（oos_days=263，未触 holdout）。作为因子筛选证据合格；不是收益声明。 |
| 13 | ensemble search 27 候选的 val CAGR（含 71.7% 赢家 val 读数） | 0.72 等 | `ensemble_search_plus7/summary.json` | **`searched_validation`** | 27 试验的 max，val 窗内。71.7%→38.6% 的衰减本身就是过拟合信号。 |
| 14 | RL 2026 结果 | — | `runtime/models/v88_rl_pit` 等 | **`contaminated_holdout`** | 训练窗(→2025-12-31)侵入 holdout。 |
| 15 | stage1/3a/3b/4/5 各 gate 读数 | 全 REJECT | `v89_closed_loop/stage*` | `contaminated_holdout`（窗口意义上） | 负结论（拒绝 overlay）方向上可采信——在污染窗上都证明不了自己，更不会在干净窗上成立。 |

## 当前最可信 baseline（裁定）

**不存在完全干净的 final-holdout 数字。** 从严排序：

1. **benchmark（等权全A，holdout 窗）= +19.9%**：唯一零选择数字。任何生产声明首先要和它比。
2. **v8.9+7 3-sleeve 先验 blend, holdout 窗：k10 = +8.3%**（class `clean_oos`¹）：当前生产模型族在近似-holdout 窗上最诚实的读数 —— **低于 benchmark**。
3. **v8.9 rankfix, k50, 全 OOS 混窗 = +17.25%**（class `clean_oos`²）：v8.9 家族最稳的单看数字，可作为改进对照的**临时 reference**，注明混窗属性。

**因此当前诚实的结论是：生产候选未被证明在 holdout 窗跑赢被动等权基准；38.6% 为未证实宣称（unproven claim），在 walk-forward + 新鲜 forward 窗验证前不得引用为业绩。**

## 引用规则（即刻生效）

任何对内/对 UI 引用生产业绩，必须同时给出：数值 + 分类标签 + artifact 路径 + 该窗被看次数（census 可查）。禁止单独引用 38.6%、68%、57.4%、71.7%。
