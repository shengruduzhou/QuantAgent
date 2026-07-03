# EXPERIMENT_HISTORY_SUMMARY — 实验史梳理（Phase 1）

> git 历史只有 42 个粗粒度 commit（v3→v8.9，squash 式），真正的实验史在
> `runtime/reports/`（46 个实验族目录）与历史会话记录中。
> 标注：**[A]** = artifact 在盘可查；**[R]** = 历史记录，Phase 2 复核关键项。

## 1. 版本主线（git + artifact 对齐）

| 阶段 | 关键事件 | 证据 |
|---|---|---|
| v3–v6 | 原型期 | git log |
| v7.x (21 commits) | PIT lake、providers、gold builder、MLP trainer、执行仿真、CLI v7 族成型 | src 结构 + AGENTS.md |
| v8–v8.7 | FT-Transformer sleeves、strict_v8、decision chain、sweep 体系 | cli/v8_deep.py 等 |
| **v8.8 数据事故** | `--batch-symbols 300` 令 22 个 alpha 列被**局部截面 rank 污染**，全 sleeve 受染 | [R]; 修复脚本 `scripts/rankfix_*.py` 4 件在盘 [A] |
| **v8.9** | rankfix 数据集重建 + 全量重训（当前代基线） | `training_dataset_alpha181_exec_v88_rankfix.parquet` [A] |
| v8.9 closed loop | LLM 因子闭环 iter1–3：19 个候选中 **5 个**过洁净 OOS（含 2 个新颖 LLM 因子） | `v89_closed_loop/iter{1,2,3}/` [A][R] |
| **v8.9+7 (现生产)** | +7 洁净 synth 因子物化 → plus7clean 数据集 → 3-sleeve GPU 重训 → ensemble weight search 选出 2-sleeve rank blend | `retrain_plus7_20260620_0300/`, `ensemble_search_plus7/`, `PRODUCTION_CONFIG.json` [A] |
| plus8 | v89_plus8 数据集已建，状态待查 | gold 目录 [A]，[TO-VERIFY] |

## 2. 已定论的负结果（不要重做）

1. **+20%/yr 超额是幻影**：来自涨停/ST 不可交易名单的纸面填充；由此诞生
   `baseline_protocol.py` 四 variant 分解，variant C 为唯一可信口径。[R]
2. **全宇宙 deep MLP 无 edge**：242 特征、6 折 GPU walk-forward，OOS rank-IC h1/h5≈0，
   top-100 年化 −0.9% vs 宇宙等权 +18% ⇒ 模型跑输自己的宇宙。管线诚实性反而由此验证。[R]
   （edge 来自 book construction / ensemble，不是原始截面 ranker。）
3. **做T（1 分钟 OHLCV）无可实现 edge**：edge-frontier rank-IC≈0，maker 10bps 下
   top-predicted 分钟净 −19~−26bps；引擎保留为 NO_TRADE 默认覆盖层。~20 个
   `intraday_dot_*` 报告目录属于此死结论。[R][A]
4. **RL +39pp 可疑**：env-flat（book 无离散度时 env 无法选择），已加
   `env_can_select` 守卫；strict 复评为准。[R]
5. **regime 因子子集搜索死路**；**naive sector momentum 追涨被 whipsaw**。[R]
6. **打板逆向选择 −2%/板**。[R]

## 3. 已定论的正结果 / 有效资产

1. **v8.9 干净基线**（rankfix 后，train≤2024-07 洁净截断）：variant-C top-50
   **+17.3% CAGR / maxDD 10.9% / Calmar 1.58**（2024-08→2026-05）。[R]
2. **beta 分解（stage 11）**：v8.9 size30 CAGR +56.8%，对全A beta 0.91，
   **Jensen alpha +12.9%/yr vs 全A**（+35% vs 沪深300）⇒ 有真 alpha 非纯 beta。
   `scripts/stage11_beta_decompose.py` + `backtest/beta_decomposition.py` [A][R]
3. **LLM 因子闭环可信化**：tradability-aware IC 验收、ICIR 下限、记忆 JSONL、
   coverage 引导；5/19 幸存率是健康水平。[A][R]
4. **alpha101 向量化**：全 panel 1.5–2hr → 3.8min（workers=16），位精确等价测试在。[R]
5. **feature schema 契约**：schema_hash 锁列集，walk-forward 折叠间禁漂移。[A]

## 4. 当前生产声明 vs 盘面证据（Phase 2 第一案）

同一 holdout（2025-09-01..2026-05-15）、同一次 plus7 重训的预测：

| blend 方案 | holdout CAGR | 出处 |
|---|---|---|
| 3-sleeve 0.30/0.45/0.25 平均 | **+8.3%** | `realtest_plus7_holdout_top10` [A] |
| 2-sleeve 平均 | **+14.5%** | `realtest_2sleeve_holdout_top10` [A] |
| 2-sleeve rank blend（生产宣称） | **+38.6%** | `ensemble_search_plus7/winner_heldout` [A] |
| 等权全A benchmark | **+19.9%** | 同上 metrics [A] |

疑点（全部进 `LEAKAGE_AUDIT.md`）：
- holdout 被 `topk_fine/holdout_{10,20,30,40}.json`、`factor_combo_search/combo_heldout`、
  多个 `realtest_*_holdout_*` 反复评测 ⇒ "untouched" 名不副实，38.6% 未经
  multiple-testing / PBO 校正。
- validation CAGR 71.7% → holdout 38.6% 的大衰减 + blend 方式敏感度（8%↔38%）
  提示脆弱性。
- `run_v89_closed_loop.sh` 复跑用默认 3-sleeve blend，**不能复现**生产 2-sleeve 配置
  （流程漂移）。

## 5. 遗留可用但未接线的方向（机会清单，供 Phase 5/6 排队）

- globalpercent 宏观概率面板：未接线。[R]
- stage 10 概念硬度（純度+订单+业绩+资金）：live screen 层，PIT 化验证未完成。[R]
- 概念链训练需要 forward PIT 数据积累。[R]
- `scripts/stage12_risk_control_suite.py` / `stage13_book_optimizer.py`：存在，
  是否已进生产 book 层 [TO-VERIFY]。
- fundamental hardness = 防御性 beta-reducer（低 DD）但窗口间 alpha 不稳。[R]
