# PRODUCTION_REPRODUCIBLE — 生产可复现性修复报告（Stage B，已实施）

> 日期 2026-07-03，分支 `robustness-mission`。修复对象 = `PRODUCTION_REPRODUCIBILITY_AUDIT.md` 的 Q1/Q2/Q3/Q5 断裂。

## 1. 现在的单命令复现

```bash
AI_quant_venv/bin/python3 scripts/materialize_production_composite.py \
  --verify-against runtime/reports/v89_closed_loop/ensemble_search_plus7/winner_predictions.parquet
```

**实测结果（2026-07-03）**：`identical_values=True, max_abs_diff=0.0`，1,410,354 行 —— 与冻结的生产 artifact **数值完全一致**（parquet 字节可因压缩元数据不同，manifest 里已注明）。`--from-sleeves` 模式（从各 sleeve `predictions.parquet` 重建，用于全新 retrain）同样验证 `max_abs_diff=0.0`。

## 2. 机器可读配置 = 唯一事实源

`configs/production_blend.json`（被代码真实消费，不再是手写备忘）：

- **blend**：`cs_rank_pct_weighted_sum`，weights short=1.0 / mid=1.0 / **long=0.0**（与 `ensemble_weight_search._ranked_sleeves` 位一致，等价性单测覆盖）；
- **top_k=10**，evaluator = `baseline_protocol.py` variant `C_flags_eligible_delay1`，slippage 8bps；
- **long sleeve**：`excluded_from_blend_retained_for_diagnostics`，理由字段写明（weight=0 来自 validation 搜索，该选择 `likely_overfit` PBO 0.886，从未独立验证）；训练可选（见 §3）；
- **lineage**：dataset sha256（8.1GB plus7clean 全量哈希 `272e4736…`）、ensemble_composite sha256、三个 sleeve feature-schema sha256、train/prediction 窗、声明时 git hash；
- **trust**：`class=likely_overfit`；38.6% holdout 声明显式标注 `contaminated_holdout — never cite as clean OOS`；参考配置 = 先验 3-sleeve 平均。
- 已知缺口如实入档：plus7clean 数据集没有同名 schema json（历史遗留，Q5），以 sleeve-level feature schema 哈希代偿。

字段规范 → `PRODUCTION_CONFIG_SCHEMA.md`。旧 `runtime/reports/v89_closed_loop/PRODUCTION_CONFIG.json` 已盖 `_NOTICE`（superseded）+ `_trust_class` 戳，保留为历史记录。

## 3. closed loop 与生产对齐（流程分叉修复）

`scripts/run_v89_closed_loop.sh` 两处改动：

1. **step 4 retrain**：默认只训练生产 blend 实际使用的 sleeve（short+mid）。long sleeve 不再被静默训练后丢弃（每次省数 GPU 时）；需要诊断时 `QUANTAGENT_TRAIN_SLEEVES="short_5d mid_5d_30d long_30d_120d"` 显式开启。
2. **blend 步**：`run_v8_deep_sweep.blend()`（3-sleeve 平均，与生产不符）替换为 `materialize_production_composite.py --config configs/production_blend.json --from-sleeves` —— 复跑 closed loop 现在产出**生产同款** blend，且自动带 manifest。

`bash -n` 语法检查通过。旧 blend() 保留于 `run_v8_deep_sweep.py`（sweep 场景仍用），不再被 closed loop 调用。

## 4. provenance stamp

每次物化写 `<out>.manifest.json`：created / git_hash / argv / config echo / 模式（from_composite|from_sleeves）/ 全部输入 sha256 / 输出 sha256 / 行数 / **trust_class**（从配置继承 —— 物化不洗白信任等级）。

## 5. 测试证据

| 检查 | 结果 |
|---|---|
| `tests/test_production_blend.py`（blend 与原搜索 `_ranked_sleeves` 位一致、缺列语义、per-date 秩性质） | 3 passed |
| 物化 vs 冻结 winner artifact（from_composite 模式） | identical_values=True, diff=0.0 |
| 物化 vs 冻结 winner artifact（**from_sleeves** 模式，closed-loop 路径） | identical_values=True, diff=0.0 |
| `bash -n run_v89_closed_loop.sh` | OK |
| P4 守卫回归（`tests/test_quarantine_guard.py`） | 19 passed |

## 6. 仍未修复（如实声明，排期中）

- **forward 推理**（P6）：`forward_daily_inference.py` 仍钉在 v8.8 旧代际 + 11 列特征不可复现 —— 生产模型的日频 forward 路径依旧无效，属独立高风险项（特征保真校验前置）。
- 搜索脚本 `--test-start 2025-09-01` 默认值（P7）：guard 已在 `bp.evaluate` 层拦截，默认值改必填排 Stage C。
- plus7clean 数据集 schema json 缺失：重建数据集时（未来）必须带 `--expected-feature-schema`。
