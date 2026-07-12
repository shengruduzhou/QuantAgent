# HOLDOUT_CONTAMINATION_AUDIT — 最终 holdout 污染审计（Phase 2）

> 审计日期 2026-07-03。逐文件证据见 `HOLDOUT_ARTIFACT_CENSUS.csv`（56 行，含 mtime、窗口、指标、分类）。
> 审计对象：holdout 窗口 **2025-09-01 → 2026-05-15**（`run_v89_closed_loop.sh` 声明的 "final-test window: unseen by selection+model"）。

## 结论（一句话）

**"untouched holdout" 声明不成立。** 该窗口自 2026-06-11 起被模型代际选择间接消费，自 2026-06-19 起被直接反复评测（**≥35 次 direct variant-C 评测 + ≥13 次含窗评测**），且至少 **5 类选择决策**使用了 holdout 信息。生产宣称的 38.6% CAGR 是多次窥视后的幸存读数，未经 multiple-testing 校正，**不得作为 clean holdout 证据引用**。

## 1. 时间线重构（全部来自文件 mtime + 内容）

| 时间 | 事件 | holdout 消费 |
|---|---|---|
| 06-11 20:07 / 21:35 | v8.7-exec / v8.8-judgment 三 sleeve sweep（test→2026-05-15） | 间接：代际/feature-policy 选择的 test 窗含 holdout |
| 06-13 12:03 | v8.9 rankfix sweep（同窗） | 间接 |
| 06-19 08:22 | `baseline_v89_current.json`（2024-08-28→2026-05-15, k50, **+17.25%** vs bench +51.2%） | 含窗单次读 |
| 06-19 18:29 | `topk_sweep/` k∈{30,50,100,200,300,500}（含窗） | 6 次含窗，k 探索 |
| 06-19 18:35 | `topk_fine/` full k∈{5..40} + **holdout_k∈{10,20,30,40}** | **4 次 DIRECT holdout**（rankfix 预测） |
| 06-20 03:06 | `realtest_current_holdout_top10`（rankfix, **+17.9%**） | DIRECT #5 |
| 06-20 03:59–13:32 | plus7 重训（short/mid 完成 → long 两次失败 → 13:32 补齐） | 训练本身干净（train≤2024-06-30, embargo 30d） |
| 06-20 05:10 | `realtest_2sleeve_holdout_top{10,20}`（long 未完成时跑的 2-sleeve 平均，**+14.5%**/k10） | DIRECT #6-7 |
| 06-20 14:31–14:33 | `realtest_plus7_holdout_top{10,20}`（3-sleeve 先验 blend，**+8.3%**/k10）+ full 窗×2 | DIRECT #8-9 + 含窗×2 |
| 06-20 15:38 | `ensemble_search_plus7`：27 候选**仅在 validation** 评测（干净），胜者 (1,1,0)×k10 上 holdout 复跑一次 → **+38.6%** | DIRECT #10 |
| **06-20 15:44** | **`PRODUCTION_CONFIG.json` 手写**，采纳 38.6% | **选择事件**（见 §2） |
| 06-20 20:14/20:18 | `factor_combo_search(+v2)/combo_heldout` | DIRECT #11-12 |
| 06-21 07:07–07:09 | `regime_search/leaderboard.json`：8 候选 × (non2026 + y2026) 全部评测，**finalscore = 0.55–0.7 权重 × y2026_cagr** → 导出 winner_w111_k5（y2026 ann **+68.0%**, turnover 0.86/日）、winner_w210_k10（**+57.4%**, 0.81/日） | **DIRECT #13-28 且选择函数直接用 holdout** |
| 06-21 21:02 → 06-23 | stage1 / stage3a / stage3b / stage4 / stage5（non2026 + y2026 子窗上的执行/做T/成本/长验证） | DIRECT ≥8 次（全部 gate=REJECT，未采纳） |

## 2. 使用了 holdout 信息的选择决策（污染核心）

1. **`regime_search` 的选择函数显式使用 holdout**：`finalscore{w2026=0.55/0.6/0.7}` 直接加权 y2026_cagr（2026-01-02→05-13 在 holdout 内）。这是教科书式 holdout-tuning。
2. **PRODUCTION_CONFIG 的采纳本身**：写于 15:44，此前当日已知同窗至少 5 个替代读数（8.3% / 14.5% / 17.9% / topk_fine 4 个）。在已知全部备选 holdout 表现的情况下把 38.6% 立为生产 ⇒ 事实上的 max-over-looks 选择。
3. **blend search 的启动是对 holdout 读数的反应**：时序证明 8.3%（14:31）在前、搜索（15:38）在后 —— garden of forking paths：分析路径本身由 holdout 结果驱动。
4. **k-grid 的收窄**：ensemble search 的 k∈{10,20,30} 在前一日 topk_fine **holdout** k-grid（10..40）之后确定。
5. **模型代际选择**（v8.7-exec vs v8.8-judgment vs v8.9 rankfix、feature-policy=judgment 等）由 06-11..13 sweep 决定，其 test 窗覆盖 holdout 期。

## 3. 未被污染的部分（同样重要，避免过度否定）

- **模型训练**：train 2018-01-02→2024-06-30，embargo 30d，early-stop 用时间序验证 —— 与 holdout 无接触。[VERIFIED run_config.json]
- **+7 因子验收**：pooled_eval_clean 的 factor 表 `oos_days=263` ≈ 2024-08→2025-08-31，未触及 holdout（尽管 summary 缺 `oos_end` 字段，属 provenance 缺陷而非污染）。[VERIFIED factor_eval_table.csv]
- **ensemble_weight_search 程序本身**：27 候选只看 validation，胜者只跑一次 holdout —— 程序设计正确，被上述 §2.2/2.3 的**流程**污染。
- **3-sleeve 先验 blend 的 +8.3%（k10）/ realtest_plus7_holdout_top20**：blend 权重 0.30/0.45/0.25 是历史默认值（`HorizonEnsembleWeights` 代码默认），属 plus7 模型在 holdout 上的**第一批先验读数**。
- benchmark +19.9%：机械计算，无参数。

## 4. 侧链污染（与生产无关但需记录）

- **RL**：`rl_pit_train_eval.py --train-end 2025-12-31` ⇒ RL 策略**在 holdout 内部训练**（runtime/models/v88_rl_pit）。任何 RL 在 2026 窗口的结论不可作为 holdout 证据。
- **intraday/做T 家族**：`tickflow_intraday_factor_combo_train.py` 等默认 `--start 2025-09-01`，在 holdout 期内训练/评测（结论均为 REJECT，未进生产）。
- **forward_daily_inference.py**：钉死在 v8.8 corrupted 代际 + 11 列因子不可复现（fidelity spearman≈0.71），其 forward 输出不可作为任何证据。

## 5. 量化总结

| 项 | 计数 |
|---|---|
| DIRECT holdout-窗 strict 评测（variant C 或子窗等价） | **≥35** |
| 含 holdout 的全窗（2024-08→2026-05）strict 评测 | ≥13 |
| 使用 holdout 信息的选择决策 | ≥5 类 |
| 评测过的 distinct 配置族（模型代际×blend×k×regime-policy×overlay） | ≥60 |
| 经 multiple-testing 校正后 38.6% 的置信度 | **未量化 —— 必须先做 PBO/DSR（见 EVALUATION_PROTOCOL_V2 §4）** |

## 6. 裁定

- 窗口 **2025-09-01→2026-05-18 就地 QUARANTINE**（隔离规则见 `EVALUATION_PROTOCOL_V2.md` §3）。
- 38.6% 归类 `contaminated_holdout`（详见 `BASELINE_TRUST_CLASSIFICATION.md`）。
- 任何后续改进**不得**以 38.6% 为对照 baseline，也不得再在该窗上做任何选择性评测。
