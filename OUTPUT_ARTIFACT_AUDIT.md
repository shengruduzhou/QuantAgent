# OUTPUT_ARTIFACT_AUDIT — 输出物审计（Stage C / Phase 3）

> 2026-07-03。磁盘现状：**总 936G，已用 80%，剩 186G**；`runtime/` 83G（gold 56G + silver 14G + reports ~4G + 其他）。

## 1. gold 训练集（58G，最大头）

| 文件 | 大小 | 状态 | 处置建议 |
|---|---|---|---|
| `training_dataset_alpha181_exec_v89_plus7clean.parquet` | 8G | **生产**（sha256 已入 configs/production_blend.json） | **保留（锁定）** |
| `training_dataset_alpha181_exec_v88_rankfix.parquet` | 8G | v8.9 基线数据集；closed loop 的 LABELS/BASE | **保留** |
| `training_dataset_alpha181_exec_v89_plus8.parquet` | 8G | plus8（8 因子版），未投产，Phase 6 可能用 | 保留（观察） |
| `training_dataset_alpha181_exec_v88.parquet` | 8G | **已证污染**（batch-rank，v88 corruption） | **删除候选 #1**（保留 rankfix 取证脚本与 22 列 diff 即可）→ 释放 8G |
| `training_dataset_alpha181_exec_v87.parquet` | 7G | 上上代 | 删除候选 #2 → 7G |
| `training_dataset_alpha181_exec_v89.parquet` | 8G | 被 plus7clean 取代；v89 rankfix 基线的复训重现需要它 | **归档候选**（若保留 v8.9 基线可再现性则留；二选一，见 PRUNE_PLAN） |
| `training_dataset_alpha181_full_nosynth.parquet` | 7G | 全宇宙 no-edge 探针（结论已定） | 删除候选 #3 → 7G |
| `training_dataset.parquet` | 2G | 早期 legacy | 删除候选 #4 → 2G |
| `training_dataset_core30.parquet` / `tickflow_fin_features` / `alpha101_rankfix_22cols` | 1G×3 | 小件：core30 实验 / tickflow 特征 / rankfix 取证 diff | core30 删除候选；后两者保留 |

**潜在回收 ≈ 24–32G（把 80% 占用压回 ~77%）。全部删除需用户批准（Operating mode 红线），本阶段仅提案。**

## 2. runtime/reports（~4G）

- `v8/`（2.7G）：v8.7/v8.8/v8.9 sweep 全档 —— v8.8 属污染代际但其 sleeve 预测被 forward_daily_inference 引用（P6 修复前**不可删**）；v8.9 rankfix 是 topk/baseline 证据链（census 引用）→ 保留。
- `v89_closed_loop/`（1.2G）：审计证据链主体（census 60+ 引用）→ **保留（冻结为证据）**。
- `intraday_dot_*` 约 20 目录（~600M）：死结论家族 → 归档/删除候选（tar.zst 后 ~100M）。
- `pbo_dsr_retro/`（新，<1M）：保留。

## 3. 无消费者/误导性输出（重点）

| 输出 | 问题 | 处置 |
|---|---|---|
| `PRODUCTION_CONFIG.json`（旧） | 曾是唯一"生产定义"但无代码消费 | ✅ 已盖 `_NOTICE` superseded + `_trust_class` 戳 |
| `runtime/reports/v8/forward/ensemble_forward.parquet` | v8.8 钉死 + 特征漂移的 forward 分数流，**每天还在追加** | ✅ 脚本已加运行警告；P6 修复前其数据不可作证据；是否停 systemd forward 由用户定（PRUNE_PLAN §P-F） |
| `winner_w111_k5_maxret` / `winner_w210_k10_robust` | y2026（隔离窗）上选择+评测的"漂亮数字" | 保留为污染证据，metrics 已在 census 标 `contaminated_holdout`；不迁移、不引用 |
| stage 系列各自的简化回测 csv/json | 与 variant-C 口径不可比 | 文档标注（BACKTEST 口径唯一性写入 ACCEPTANCE_RULES R3） |
| `topk_sweep`/`topk_fine` 全档 | holdout k 探索证据 | 保留（证据），禁止用于 k 选择 |

## 4. UI 可见面

`services/quant_api` indexer 扫描 `**/backtest/metrics.json` → 上述污染 artifact 会**继续出现在 UI**。
提议（PRUNE_PLAN §P-H）：indexer/adapters 读取 `trust_class` 字段并在 UI 打标（P4 已让 forensic 输出自带该字段；历史文件无字段 → UI 显示 "unclassified"）。涉及 services 代码，小 patch，待批。

## 5. 新增输出的纪律（即刻生效，已在执行）

- 本 mission 新产物均 <1M 且带 manifest（materializer、pbo_dsr_retro）。
- 实验产物预算：单实验 ≤5G、结束即清中间物（ACCEPTANCE_RULES R5）。
