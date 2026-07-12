# PRODUCTION_RECONCILIATION_PATCH_PLAN — 生产一致化最小补丁计划（Phase 2，仅计划）

> 目标：让"生产配置"变成**一个机器可读文件 + 一条再生命令**，并堵住 holdout 再消费。
> 原则：diff 最小、不动可信路径（PIT / schema lock / variant-C / 审计尾迹）、每个 patch 独立可验证、小 commit。
> **本阶段不实施** —— 实施排在 Phase 6 第 1 优先级（evaluation trust fixes），逐条过 smoke 后合入。

## P1. 机器可读生产 blend 配置（新文件，无代码风险）

- 新增 `configs/production_blend.json`：
  ```json
  {
    "model_run": "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300",
    "blend": {"method": "cs_rank_sum", "weights": {"short_5d": 1, "mid_5d_30d": 1, "long_30d_120d": 0}},
    "top_k": 10, "variant": "C_flags_eligible_delay1",
    "selection": {"window": "2024-08-28..2025-08-31", "n_trials": 27,
                   "script": "scripts/ensemble_weight_search.py"},
    "provenance": {"git_hash": "<fill>", "dataset": ".../training_dataset_alpha181_exec_v89_plus7clean.parquet",
                    "dataset_schema_hash": "<fill>", "created": "2026-06-20"},
    "trust": {"class": "contaminated_holdout", "audit": "HOLDOUT_CONTAMINATION_AUDIT.md"}
  }
  ```
- `PRODUCTION_CONFIG.json` 保留为人读备忘，顶部加一行指向 configs 文件（deprecation note）。

## P2. 单命令物化生产 composite（~60 行新脚本）

- 新增 `scripts/materialize_production_composite.py --config configs/production_blend.json --out <path>`：
  读 sleeve `predictions.parquet` → 按 config 的 method/weights 计算 composite（`cs_rank_sum` 与 `ensemble_weight_search._ranked_sleeves` 逐位一致，加等价性测试）→ 写 parquet + manifest（git hash、输入文件 sha256、命令行）。
- 验证：对 `winner_predictions.parquet` 做逐行等价 diff（容差 0）。

## P3. closed loop 第 4 步接配置（1 行改动级）

- `run_v89_closed_loop.sh` blend 步骤改为调用 P2 脚本（默认 `--config configs/production_blend.json`），
  旧 `run_v8_deep_sweep.blend()` 路径保留但打印 DEPRECATED 警告。
- 效果：复跑 closed loop = 复现生产 blend。

## P4. holdout 隔离守卫（评测器 ~15 行）

- 新增 `configs/quarantined_windows.json`：`[{"start": "2025-09-01", "end": "2026-05-18", "reason": "burned holdout, see HOLDOUT_CONTAMINATION_AUDIT.md"}]`。
- `scripts/baseline_protocol.py`：评测窗与隔离窗相交时 **默认拒绝**，需显式 `--allow-quarantined "<原因>"` 才放行，且把访问记录追加到 `runtime/state/holdout_access_log.jsonl`（时间、命令、原因、git hash）。
- 这是防再犯的关键闸门，成本极低。

## P5. lineage stamp（3 个小 patch）

- `baseline_protocol.py` 输出 json 增加：git hash、predictions 文件 sha256、完整 argv。
- `evaluate_discovered_factors.py`：summary 必写 `oos_start/oos_end/oos_days`（缺参也写 null + 实际推断窗口）。
- `cli/v8_deep.py` 训练收尾：把 gold 数据集的 `feature_version/schema_hash` + git hash 写进 sleeve `run_config.json`。

## P6. forward 推理对齐生产（较大，单独排期）

- `forward_daily_inference.py`：`RUN_DIR`/blend 改为读 `configs/production_blend.json`；
- 前置阻塞项：11 个不可复现 alpha 列（v8.2 向量化行为漂移）——需先用 `rebuild_alpha_columns_v89.py` 思路对 plus7 特征做一次 fidelity 校验（overlap spearman 目标 >0.99）再切换。
- 在修好前，在脚本头部与输出 parquet 里显式标注 `model_generation=v8.8-DEPRECATED`。

## P7. 搜索脚本默认窗改为安全值（1 行×N）

- `ensemble_weight_search.py` / `factor_combo_search.py` / `regime_strategy_search.py` / `stage*` 的
  `--test-start 2025-09-01` 默认值改为**无默认 + 必填**，并在帮助文本注明隔离窗规则（防止无意识再烧）。

## 实施顺序与验收

| 顺序 | patch | 风险 | 验收 |
|---|---|---|---|
| 1 | P4 隔离守卫 | 低 | 单测：隔离窗内调用被拒；`--allow-quarantined` 放行且留痕 |
| 2 | P1+P2 配置+物化 | 低 | winner_predictions 逐行等价；manifest 字段齐 |
| 3 | P5 lineage | 低 | 输出 json 含 git/sha256/argv |
| 4 | P3 closed loop 接线 | 中 | `--dry-run` 打印新命令；小样本跑通 |
| 5 | P7 默认窗 | 低 | argparse 必填生效 |
| 6 | P6 forward 对齐 | 高（特征保真前置） | overlap spearman >0.99 后方可切换 |

不做的事：不重写 closed loop、不删旧 blend、不改 strict 引擎、不动 gold builder。
