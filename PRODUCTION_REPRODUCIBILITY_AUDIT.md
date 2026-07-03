# PRODUCTION_REPRODUCIBILITY_AUDIT — 生产可复现性审计（Phase 2）

> 审计日期 2026-07-03。核心问题：**"当前生产配置能否由一条命令复现？" —— 答案：不能。** 共发现 7 处断裂。

## Q1. `run_v89_closed_loop.sh` 能否复现 `PRODUCTION_CONFIG.json`？ → **否**

- closed loop 第 4 步 blend 调用 `run_v8_deep_sweep.blend()`，硬编码 `HorizonEnsembleWeights()`（**3-sleeve 0.30/0.45/0.25 加权平均**，per-sleeve 原始 score）。[scripts/run_v8_deep_sweep.py:46-85]
- 生产配置是 **2-sleeve（short+mid）per-date cross-sectional rank 相加，long 丢弃**，由 `ensemble_weight_search.py`（**不在 closed loop 的 6 步之内**）事后搜出。
- ⇒ 复跑 closed loop 得到的 composite_score 与生产不同：**流程与生产已分叉**。

## Q2. 代码里的生产 blend 与配置一致吗？ → **不一致，且配置无消费者**

- `grep -rn PRODUCTION_CONFIG scripts/ src/ services/` → **零命中**：该文件是手写备忘，没有任何代码读取它。
- 生产 blend 的唯一可执行定义散落在 `ensemble_weight_search.py` 的搜索过程里（`_ranked_sleeves` + 权重 (1,1,0)），产物 `winner_predictions.parquet` 是**一次性物化结果**，没有"从配置再生"的路径。

## Q3. long sleeve 是真被丢弃，还是只在下游被忽略？ → **只在下游被忽略**

- long sleeve 照常训练（closed loop / retrain 脚本三 sleeve 全训，long 还带 `--train-micro-batch 1024` 特殊处理），预测列仍写入 `ensemble_composite.parquet`。
- 丢弃只发生在赢家权重 (1,1,0) 的 blend 一步。⇒ 每次重训继续支付 long sleeve 的 GPU 小时（约数小时/次），产出被扔掉。
- 附加发现：06-20 当天 long sleeve 重训**失败两次**（`_RETRAIN_COMPLETE_fail1`、`_LONG_REDONE_fail` 标记文件），2-sleeve 想法正是在 long 缺席时产生的 —— "丢 long"这一结构性决定从未在干净协议下独立验证过。

## Q4. top-K 是在看 holdout 之前还是之后选的？ → **之后（部分）**

- k=10 最早作为 holdout 直接评测出现于 06-19 `topk_fine/holdout_10.json`（前代模型）；06-20 ensemble search 的 k-grid {10,20,30} 才做 val 内选择。
- ⇒ k 的**候选范围**由 holdout 观测塑形，val 选择只是在被塑形的网格内挑。分类上属于"部分 after"。详见 `HOLDOUT_CONTAMINATION_AUDIT.md` §2.4。

## Q5. 生产 artifact 是否有完整 command/config/schema-hash lineage？ → **不完整**

逐环节：

| 环节 | 有 | 缺 |
|---|---|---|
| sleeve 训练 | `run_config.json`（数据集路径+全部超参）、`ft/ft_transformer_feature_schema.json`（特征列） | **git commit hash、完整命令行、数据集 schema_hash**（gold 的 feature_version/schema_hash 未 stamp 进 sleeve artifact） |
| blend/winner | `ensemble_search_plus7/summary.json`（权重、k、val/heldout 读数、全部 27 候选） | 命令行、git hash、输入 predictions 的 hash；`winner_predictions.parquet` 无 manifest |
| 因子验收 | `factor_eval_table.csv`（逐因子 IC/ICIR/相关/容量） | summary 缺 `oos_end`（现行脚本 469 行会写，说明当时版本旧/未传参记录）；无 git hash |
| 生产声明 | `PRODUCTION_CONFIG.json`（人话描述） | 机器可读性=0、无生成者、无校验 |
| 数据集 | `training_dataset_alpha181_exec_v89_plus7clean.parquet` + 同名 schema json（部分版本有） | plus7clean 的 build 命令未见 manifest 记录 [TO-VERIFY Phase 3] |

## Q6. 生产模型有 forward/serving 路径吗？ → **有，但指向错误模型**

`scripts/forward_daily_inference.py`（唯一日频 forward 打分器）：

- `RUN_DIR` 硬编码 = `runtime/reports/v8/deep/v88_judgment_20260611_2015` —— **v8.8 corrupted 代际**，非生产 plus7；
- blend 用该 run 的 3-sleeve 权重，非生产 2-sleeve rank blend；
- 自述 KNOWN FIDELITY LIMIT：11 个 alpha 列因 v8.2 向量化重构不可复现，overlap spearman≈0.71；
- ⇒ **生产模型目前没有可用的前向推理路径**。若明天要实盘/paper，跑的将是旧模型+错 blend+走样特征。

## Q7. 汇总判定

| 维度 | 状态 |
|---|---|
| 一条命令复现生产 composite | ❌ |
| 配置↔代码一致 | ❌（配置无消费者） |
| holdout 前选 k | ❌（部分之后） |
| 完整 lineage（git+cmd+schema hash 贯穿） | ❌ |
| forward 路径与生产一致 | ❌ |
| 训练环节自身可复现（数据集+run_config+种子） | ✅（近似；缺 git hash） |
| 严格评测器可复现（baseline_protocol 确定性） | ✅ |

**修复方案（最小 patch，先计划不实施）→ `PRODUCTION_RECONCILIATION_PATCH_PLAN.md`。**
