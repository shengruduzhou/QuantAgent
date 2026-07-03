# CODEBASE_CLEANUP_REPORT — 清理执行日志（滚动更新）

> 每批次一节：证据 → 干跑 → 执行 → 验证（targeted tests + materializer verify + quarantine smoke）→ commit。
> 依据：`DEAD_CODE_AUDIT.md` / `OUTPUT_ARTIFACT_AUDIT.md` / `PRUNE_PLAN.md` / `DELETION_CANDIDATE_MANIFEST.csv`。

## 批次记录

### B-0 · 2026-07-03 · 安全加固（PRUNE P-A/P-B，commit 07d5d4f）
- STATUS WARNING：`models/v7_deep_alpha.py`、`models/v7_multi_horizon.py`（启发式≠生产）；`training/v8_pipeline.py`（legacy GA 管线）。
- 运行时警告：`forward_daily_inference.py`（v8.8 corrupted 钉死）。
- 三个污染源搜索脚本 OOS 窗口必填化 + `regime_strategy_search.py` 显式 quarantine guard。
- 验证：26 tests green。**零删除。**

### B-1 · 2026-07-03 · P-E 数据集删除批次 1（commit 85abfe7）
- **删除**（先写 sha256+parquet 元数据 manifest 至 `runtime/archives/deletion_manifests/`）：
  - `training_dataset_alpha181_exec_v88.parquet` **7.80G**（batch-rank 污染，rankfix 取证链保留）
  - `training_dataset_alpha181_exec_v87.parquet` **6.06G**（上上代，唯一引用=已完结的 build_v88_dataset.py）
- **归档后删除**：`runtime/reports/{intraday_dot_*,dot_selective*}` 840M → `runtime/archives/intraday_dot_reports_20260703.tar.zst`（442M，5,133 文件，tar 列表校验）
- **改判 keep**：`training_dataset_alpha181_full_nosynth.parquet`（governed no-synth 基线，v8_gated/v8_verify/evaluate_discovered_factors 共 6 处默认引用——语义角色而非陈旧物）
- 附带修复：**UI `runtime_cleanup.py` 过期 keep-list 曾把生产数据集列为删除候选** → keep-list 更新为 {plus7clean, rankfix, v89, plus8, full_nosynth}；`train-v8-deep` 默认数据集 → plus7clean。
- 验证：26 tests ✅ / materializer `max_abs_diff=0.0` ✅ / quarantine smoke exit 3 ✅
- **磁盘：186G → 200G 可用（80%→78%）**

### B-2 · 2026-07-04 · 孤儿一次性脚本 deprecation（commit 4199745）
- 32 个脚本加 DEPRECATED 头（纯注释，零行为变化）：做T 家族 12、intraday panel builders 3、stage1/3a/3b/4/5/7/9 一次性 12、rankfix 取证 4（保留为证据链）、board_chase/paper_replay/overlay_regime_split 3。
- 依赖证据：逐文件 0 引用复扫（scripts/src/tests/docs/services/systemd/README/AGENTS）；4 个仍被引用者自动跳过（dot_overlay_backtest 等）。
- 移除窗口：2026-10-01 后仍无人使用则进入删除批次。
- 验证：compileall ✅ / 26 tests ✅。

### B-3 · 2026-07-04 · H-003 数据修复配套（commit 8521c83）
- `update_market_panel_daily.py`：SDK epoch-ms + `adjust="none"` + volume 手→股 ×100 + **start 3 天缓冲**（TickFlow 掉首日根因修复）。
- 新增一次性 `scripts/repair_fresh_window_20260704.py`（用后列入删除窗口）。

## 阻塞项（等待批准/前置）

| 项 | 大小 | 阻塞原因 |
|---|---|---|
| `training_dataset.parquet` | 1.94G | `paper/daily_loop.py` + cli v7 默认路径需先 repoint（Phase 7） |
| `training_dataset_core30.parquet` | 0.79G | `cli/v8.py` deterministic_core30_fallback 默认 |
| `training_dataset_alpha181_exec_v89.parquet` | 7.27G | v8.9 基线复训重现性 —— 默认保留 |
| 32 个已 deprecate 脚本的物理删除 | ~0 | 观察期至 2026-10-01 |

## 累计成效

- 磁盘回收 **~14.3G**（+ 归档净省 ~0.4G）；80% → 78%。
- 命名误导消除：启发式模型/legacy 管线/损坏 forward 路径全部带警告或 fail-fast。
- 危险默认消除：3 个搜索脚本窗口必填；train-v8-deep 默认=生产数据集；UI keep-list 保护生产数据。
- 技术债台账：见"阻塞项"+ DEAD_CODE_AUDIT §5（27 个 strict 直调脚本的绕过面已被 P-G stamp 覆盖）。
