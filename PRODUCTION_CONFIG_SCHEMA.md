# PRODUCTION_CONFIG_SCHEMA — `configs/production_blend.json` 字段规范（Stage B）

> 唯一事实源。消费方：`scripts/materialize_production_composite.py`（物化）、`scripts/run_v89_closed_loop.sh` step 4（复跑）。修改此文件 = 修改生产定义，必须走 `ACCEPTANCE_RULES.md` 流程并更新 `lineage.git_hash_at_declaration`。

| 字段 | 类型 | 语义 | 必填 |
|---|---|---|---|
| `version` | int | schema 版本（当前 1） | ✅ |
| `model_run` | path | 冻结的 retrain 目录（含各 sleeve 与 ensemble_composite） | ✅ |
| `composite_source` | filename | model_run 内的合成分数文件（默认 `ensemble_composite.parquet`；from_composite 模式输入） | ✅ |
| `sleeve_predictions` | map sleeve→relpath | 各 sleeve 预测文件（from_sleeves 模式输入） | ✅ |
| `blend.method` | enum | 目前仅 `cs_rank_pct_weighted_sum`：per trade_date pct-rank 各 sleeve 分数后按权重求和（与 `ensemble_weight_search._ranked_sleeves` 位一致；单测锁定） | ✅ |
| `blend.weights` | map sleeve→float | 权重；**0.0 = 该 sleeve 被排除**（rank 仍计算、贡献为零，语义与原搜索一致） | ✅ |
| `top_k` | int | 组合层等权持仓数 | ✅ |
| `evaluator.script / variant / slippage_bps` | — | 可信评测器绑定（variant C 不得降级） | ✅ |
| `long_sleeve.status` | enum | `included` / `excluded_from_blend_retained_for_diagnostics` / `excluded` | ✅ |
| `long_sleeve.reason` | str | 人话理由（含证据引用）——"为什么不训/不用" | ✅ |
| `selection.*` | — | 选择过程档案：window、script、**n_trials**、pbo_s8/pbo_s16/dsr_n27、analysis 文档 | ✅ |
| `trust.class` | enum | `clean_oos` / `walk_forward_oos` / `searched_validation` / `likely_overfit` / `contaminated_holdout` / `unknown`（定义见 `BASELINE_TRUST_CLASSIFICATION.md`） | ✅ |
| `trust.holdout_claim` | str | 对历史宣称数字的显式定性（防再引用） | ✅ |
| `trust.reference_config` | str | 当前对照基线配置 | ✅ |
| `lineage.dataset` / `dataset_sha256` | path/hex | 训练数据集及全量 sha256 | ✅ |
| `lineage.dataset_schema_json` | path\|null | 数据集 feature schema（plus7clean 历史缺失 → null + note） | ✅ |
| `lineage.feature_schema_sha256` | map | 各 sleeve `ft_transformer_feature_schema.json` 的 sha256 | ✅ |
| `lineage.train_window` / `prediction_window` | str | 训练/预测日期窗 | ✅ |
| `lineage.git_hash_at_declaration` / `created` | — | 声明时 git hash 与日期 | ✅ |

## 物化输出的 manifest（`<out>.manifest.json`）

`created / git_hash / argv / config_path / config_echo / model_run / mode(from_composite|from_sleeves) / inputs_sha256(全输入) / output + output_sha256 / rows / trust_class / verification{reference, keys_identical, max_abs_score_diff, identical_values, note}`。

## 不变式（invariants）

1. 配置里没有的东西不存在于"生产"——任何口头/备忘定义无效。
2. `trust.class` 随证据升降级，物化器原样透传，**不会**因为可复现就变 clean。
3. 权重/k/方法变更 ⇒ 新的 selection 档案（n_trials 累计）+ 走验收门，禁止就地改数。
4. 被烧 holdout（`configs/quarantined_windows.json`）不得作为任何字段的证据来源。
