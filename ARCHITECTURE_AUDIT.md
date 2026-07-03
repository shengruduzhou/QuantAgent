# ARCHITECTURE_AUDIT — 架构审计（Phase 1）

> 生成日期 2026-07-03。全部结论基于代码与 artifact 直接检查（引用路径），不依赖口头描述。
> 标注规则：**[VERIFIED]** = 本次直接读代码/artifact 确认；**[RECORDED]** = 来自历史实验记录，Phase 2 需复核；**[TO-VERIFY]** = 待 Phase 2/3 深查。

## 0. 硬件与资源约束（Phase 0 实测）

- CPU 20 cores；RAM 62 GiB（可用 ~58 GiB）；GPU RTX 3090 24 GB（空闲）；磁盘 **仅剩 186 GB（已用 80%）**。
- 仓库代码 ~11 MB / 87k 行 Python；`runtime/` 共 **83 GB**，其中 gold 数据集 **56 GB**、silver 14 GB。
- 结论：**禁止**整表载入多年度 panel（单个 gold parquet 数 GB 级）；**禁止**随手复制数据集（磁盘会满）；所有新实验 artifact 必须小型化并定期清理。

## 1. 真实生产路径（true production path）[VERIFIED]

生产链只有一条，入口是 `scripts/run_v89_closed_loop.sh` + `scripts/run_v89_plus7_retrain.sh`：

```
silver market_panel (runtime/data/v7/silver/market_panel/market_panel.parquet, PIT flags)
  → alpha181 因子库 (src/quantagent/factors/alpha181.py = Alpha101 近似 101 个 + CICC 80 个)
  → + synth_* LLM/DSL 因子 (factors/factor_synthesis.py + llm_factor_proposer.py, DSL 受限)
  → gold training dataset (data/dataset_builder/v7_training_dataset.py,
      schema_hash 锁定, forward_return_{1,5,20,60,120,126}d + forward_tradable_return_*d)
  → 3 条 FT-Transformer horizon sleeves (cli/v8_deep.py `train-v8-deep`,
      models/ft_transformer.py + training/ft_transformer_trainer.py;
      d_token=256, n_blocks=6, n_heads=8, dropout=0.25, rank+label norm, embargo 30d)
  → sleeve blend → composite_score
  → 组合层: risk/decision_chain.py (15-gate) / ensemble/strict_policy_search.py
  → 唯一可信评测: scripts/baseline_protocol.py variant C
      (backtest/strict_v8.py → backtest/ashare_execution_simulator.py:
       T+1, 涨跌停, 停牌, ST, lot=100, 成交量参与上限 10%, slippage 8bps, 成本模型)
```

当前生产配置 `runtime/reports/v89_closed_loop/PRODUCTION_CONFIG.json` [VERIFIED]：

- model = **v8.9+7**（`retrain_plus7_20260620_0300`），blend = short_5d 与 mid_5d_30d 两 sleeve 的 **per-date cross-sectional rank 相加**，**丢弃 long_30d_120d**；top_k=10；variant C。
- 宣称 held-out (2025-09-01..2026-05-15)：CAGR 38.6%，maxDD 11.85%，Calmar 3.26。
- 数字来源 = `ensemble_search_plus7/winner_heldout/backtest/metrics.json`（确为 variant-C 严格产物），
  但见 §5 风险：holdout 已被多次消费。

## 2. 误导性 / 过时模型路径（obsolete or misleading paths）

| 路径 | 实际身份 | 风险 | 状态建议 |
|---|---|---|---|
| `models/v7_deep_alpha.py` (`V7DeepAlphaModel`) | **未训练**的 numpy 启发式 tower + 固定 horizon gate，随机种子初始化 [VERIFIED] | 名字含 "DeepAlpha"，极易被误当生产模型 | Phase 7 标记 deprecated/heuristic |
| `models/v7_multi_horizon.py` (`V7MultiHorizonBaselineModel`) | 手写加权启发式 baseline [VERIFIED] | 同上 | 同上 |
| `training/v7_deep_trainer.py` (`V7DeepAlphaTrainer` MLP) | 64→32 MLP + numpy ridge fallback；全宇宙 walk-forward 已证明**无独立 edge**（OOS rank-IC≈0，top-100 跑输宇宙均值）[RECORDED] | docstring 自称 "THE deep trainer"；含 walk-forward runner（此部分仍有用） | 保留 walk-forward 基建，模型头标注 no-edge |
| `training/v8_pipeline.py`（GA factor-weight 路径） | 旧 v8 spec GA 权重优化管线 | 与 FT-Transformer 生产路径并行存在，命名撞车 | [TO-VERIFY] 是否仍被 CLI/脚本调用 |
| `AGENTS.md` 模型描述 | 写的是 "默认 Ridge，Deep Alpha disabled" | **文档已过时**，与生产事实（FT-Transformer）矛盾 [VERIFIED] | Phase 7 更新 |
| CLI `v7_train.py` 的 `train-alpha-v7`/`train-deep-alpha-v7` 系 | 训练 MLP/classical 路径 | 与 `train-v8-deep` 并存，新人会走错门 | Phase 3 依赖图后处理 |

## 3. 脚本重复与蔓延（157 个 scripts/）[VERIFIED 计数]

明显的重复/家族（Phase 3 将出依赖图后给 PRUNE_PLAN）：

- **回测入口重复**：`baseline_protocol.py`（可信）vs `executable_benchmark.py`、`factor_regime_backtest.py`、`overlay_backtest_ab.py`、`board_chase_eval.py`、各 `stage*_*.py`（1–13 阶段共 30+ 个一次性研究脚本）——大多写自己的简化回测，产出"pretty but untrusted numbers"。
- **sweep 重复**：`run_v8_sweep.py` vs `run_v8_deep_sweep.py` vs `run_v89_rankfix_sweep.sh` vs `ensemble_weight_search.py` vs `factor_combo_search.py` vs `regime_strategy_search.py`。
- **intraday panel 构建三份**：`build_intraday_minute_panel.py` / `build_intraday_panel_2026.py` / `build_intraday_panel_full.py`。
- **做T 家族**（~15 个 `intraday_dot_*` / `dot_*`）：结论已定（1 分钟 OHLCV 无 edge [RECORDED]），仅 NO_TRADE 覆盖层有用。
- **rankfix 家族**（4 个）：v8.8 数据修复的一次性取证脚本，已完成使命。

## 4. 唯一可信评测路径（trusted evaluation path）[VERIFIED]

- `scripts/baseline_protocol.py`，variant **C_flags_eligible_delay1**（flags ON + eligible ranking + t+1 fill）。
- 底层 `run_strict_backtest_v8` → `simulate_ashare_target_weights`（OrderManager/FillSimulator/VirtualBroker，审计到单笔 fill）。
- benchmark = 无摩擦等权全A（close-to-close mean），保守 bar。
- 输出 UI 可发现 artifact（`--save-backtest-dir` → `backtest/{metrics.json,nav.csv}`）。
- **Contract 细节（PIT、成本、状态处理逐条列举）→ 见 Phase 2 的 `STRICT_EVALUATOR_CONTRACT.md`。**

## 5. 风险代码路径 / 隐藏泄漏风险（Phase 2 优先清单）

1. **最终 holdout 已被反复消费** [VERIFIED]：2025-09-01..2026-05-15 这个"untouched"窗口，
   在 `v89_closed_loop/` 下至少被这些 artifact 直接评测过：
   `topk_fine/holdout_{10,20,30,40}.json`、`realtest_plus7_holdout_top{10,20}`、
   `realtest_current_holdout_top10`、`retrain_plus7_.../realtest_2sleeve_holdout_*`、
   `ensemble_search_plus7/winner_heldout`、`factor_combo_search/combo_heldout`。
   ⇒ 38.6% 实质上是"多次窥视 holdout 后幸存的最好读数"，选择偏差量级必须在 Phase 2 量化（multiple-testing 校正 / 新 holdout）。
2. **同一窗口内部读数离散度极大** [VERIFIED]：同一 holdout、同一预测源，3-sleeve 平均 blend = +8.3%、2-sleeve 平均 = +14.5%、2-sleeve rank blend = +38.6%，而 benchmark = +19.9%。对 blend 方式如此敏感 ⇒ 疑似脆弱/被选择出来的结果。
3. **closed-loop 脚本与生产 blend 不一致** [VERIFIED]：`run_v89_closed_loop.sh` 第4步用 `run_v8_deep_sweep.blend()`（固定 0.30/0.45/0.25 三 sleeve 平均），而 PRODUCTION_CONFIG 是 2-sleeve rank blend ⇒ 复跑 closed loop **不会**复现生产模型。
4. **universe 文件静态** [VERIFIED 存在]：`runtime/data/v7/universe_v88_comma.txt` 为固定 symbol 列表（单行 comma 文件）⇒ 宇宙如何生成、是否引入 survivorship，Phase 2 必查。
5. **sector_map 为 current snapshot**（survivorship）[RECORDED]，stage 8 已知问题。
6. `_cross_sectional_normalize` / `label_norm` 在 `cli/v8_deep.py` 数据准备处执行 [VERIFIED 位置]——是否严格 per-date、是否用了全样本统计量，Phase 2 逐行审。
7. 涨跌停 flat-10% 旧旗标残留于 silver panel，gold builder 强制 board-aware 重推 [VERIFIED]——但**直接吃 silver panel 的旁路脚本**（各 stage 脚本）可能仍在用错误旗标。

## 6. 死输出 / 冗余 artifact（初查，Phase 3 出完整清单）

- `runtime/reports/` 46 个实验目录，做T 家族约 20 个已属死结论。
- gold 目录存 8+ 个多 GB 训练集版本（v87/v88/v88_rankfix/v89/plus7clean/plus8/nosynth/core30...）[VERIFIED]，其中 v8.8（未 rankfix）已被判污染 [RECORDED] ⇒ 候选归档/删除（先建依赖图，磁盘压力大）。
- `runtime/models/` 下 8 个模型 family，多数为已废弃实验（full_universe_nosynth 系列 = 已证 no-edge）。

## 7. CLI 面（22 个模块）[VERIFIED 列表]

- 生产实际只用：`v8_deep.py`（train-v8-deep）、`v7_data.py`（build-training-dataset-v7 等）、部分 `v7_backtest.py`。
- `v8_gated / v8_intraday / v8_portfolio / v8_verify / v7_optimize / v7_bond / v7_policy ...` 使用状态 [TO-VERIFY]（Phase 3 依赖/调用扫描）。
- 未用 CLI flag 清单 → Phase 3（需逐命令 `--help` + 调用点扫描，纯静态可完成）。
