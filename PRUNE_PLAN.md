# PRUNE_PLAN — 清理执行计划（Stage C / Phase 3）

> 顺序执行，每步独立 commit + 冒烟。图例：✅=本轮已做｜🟡=可自主执行（小、可逆）｜🔴=需用户批准（大数据删除 / 触及可信评测器 / 停服务）。

## P-A ✅ 误导模型路径标注（已 commit）
`v7_deep_alpha.py` / `v7_multi_horizon.py` STATUS WARNING；`v8_pipeline.py` legacy note；`forward_daily_inference.py` 运行时警告。

## P-B ✅ 污染源搜索脚本加固（已 commit）
三脚本 OOS 窗口必填化；`regime_strategy_search.py` 显式 quarantine guard。

## P-C 🟡 orphan 研究脚本批量 deprecate 标注
95 个 orphan 中的做T/stage/一次性家族（~80 个）：文件头插入统一标注块
`# DEPRECATED(2026-07-03): one-shot research script; conclusion recorded in <doc>; scheduled for removal after 2026-10-01 if unused.`
不改逻辑、不删除。批量脚本化 + compileall 冒烟。**观察期后**再进入 P-E。

## P-D 🟡 重复 build 脚本合并提案
`build_intraday_panel_{2026,full}` + `build_intraday_minute_panel` 三合一；`enrich_market_panel_boardfix` 并入 `enrich_market_panel`（flag 化）。做T 家族若 P-E 删除则前者一并处置 —— 排 Phase 7，先不动。

## P-E 🔴 大数据删除（需批准，预计回收 24–32G）
按风险从低到高逐个：
1. `training_dataset_alpha181_exec_v88.parquet`（8G，**已证污染**；rankfix 取证脚本+22 列 diff 保留）
2. `training_dataset_alpha181_full_nosynth.parquet`（7G，no-edge 探针结论已档案化）
3. `training_dataset.parquet`（2G legacy）+ `training_dataset_core30.parquet`（1G）
4. `training_dataset_alpha181_exec_v87.parquet`（7G，上上代）
5. `runtime/reports/intraday_dot_*` 打包归档（600M → ~100M tar.zst）
6. `training_dataset_alpha181_exec_v89.parquet`（8G）—— **除非**决定放弃 v8.9-rankfix 基线复训重现性；默认不删
每步删除前：`grep -r` 路径引用复查 + 写入 CODEBASE_CLEANUP_REPORT.md。

## P-F 🔴 forward 管线处置（需决策）
`quantagent-forward.service` 每日用 v8.8 corrupted 代际打分（已加警告）。选项：
(a) 停 timer 直到 P6（特征保真+切 plus7）完成；(b) 保持运行仅作管道健康监测，数据标 untrusted。
建议 (a) —— 省算力且防误用。**等待用户选择。**

## P-G 🔴 strict_v8 软警告（触及可信评测器文件，需批准）
在 `run_strict_backtest_v8` 入口加一行：target_weights 日期与 quarantine 相交时 stderr 警告（不阻断、不改返回值）。堵住 27 个直调脚本的绕过面。diff ~6 行 + 单测。

## P-H 🟡 UI trust_class 透传
`services/quant_api` adapters 读取 metrics.json 的 `trust_class` 并透传 API（缺省 "unclassified"）；UI 端显示徽标。~20 行。排在 Stage D 之前的空隙执行。

## P-I 🟡 AGENTS.md 事实修正
"默认 Ridge / Deep Alpha disabled" 段落更新为 FT-Transformer 生产事实 + quarantine 规则一行 + configs/ 指针。排 Phase 7 文档统一批次。

## 冒烟基线（每步后跑）
`python -m compileall scripts src` · `pytest tests/test_quarantine_guard.py tests/test_production_blend.py -q` · `bash -n scripts/run_v89_closed_loop.sh` · （若动 services）`scripts/quant_ui_api_smoke.py`
