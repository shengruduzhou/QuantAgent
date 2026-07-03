# EXPERIMENT_LEDGER — 实验台账（append-only，含失败）

> 每条必填：git hash / 数据 schema 或 sha256 / 窗口 / 完整命令 / 改动文件 / 假设 / 指标 / RSS 峰值 / 时长 / 结论。

---

## EXP-000 · 2026-07-03 · 回溯 PBO/DSR（Phase 2.5）— **DONE / 结论 likely_overfit**

- git: `d8e065c`（分析脚本入库 commit）；输入 sha：ensemble_search_plus7/_tmp 27 文件（census 记录）
- 窗口：SEARCH 2024-08-30→2025-08-29（242d，quarantine 零接触，断言强制）
- 命令：`AI_quant_venv/bin/python3 scripts/analysis/pbo_dsr_ensemble_search_plus7.py`
- 改动：新增分析脚本（analysis-only）
- 假设：生产 blend 选择可能是噪声选择
- 指标：PBO 0.886(S=8)/0.882(S=16)；IS 赢家 OOS 均秩 6.5/27；DSR 0.919@N27→0.875@N200；半窗 Spearman −0.12；重放保真 Spearman 0.9922
- 资源：RSS 峰 2.23 GiB / 435 s
- 结论：**REJECT 单点赢家选择**；族信号存续（27/27 正，中位 +54%）→ 派生 H-001

## EXP-P4 · 2026-07-03 · 隔离守卫实施（Stage A）— **DONE / 19 tests + 3 smokes 通过**

- git: `e06ca86`；改动：`configs/quarantined_windows.json`、`src/quantagent/backtest/quarantine.py`、`scripts/baseline_protocol.py`、`tests/test_quarantine_guard.py`
- 验证：单测 19 passed；CLI 冒烟（quarantined→exit 3；clean 2025-06→08 运行正常；override 运行+trust_class 戳+日志 1 行）
- 资源：秒级/`<2 GiB`

## EXP-PB · 2026-07-03 · 生产可复现物化（Stage B）— **DONE / 数值全等**

- git:（Stage B commit，见 git log）；改动：`configs/production_blend.json`、`scripts/materialize_production_composite.py`、`tests/test_production_blend.py`、`run_v89_closed_loop.sh`（sleeve 默认 short+mid + blend 步切换）、旧 PRODUCTION_CONFIG 盖戳
- 验证：blend 等价单测 3 passed；from_composite 与 from_sleeves 两模式 vs 冻结 winner artifact `identical_values=True, max_abs_diff=0.0`（1,410,354 行）
- lineage：plus7clean sha256 `272e4736…`（8.1G 全量，28s）

---

## EXP-001 · **PROPOSED**（对应 H-001，等待执行窗）

- 命令（拟）：materializer 4 配置物化 → `baseline_protocol` variant-C @ SEARCH 窗 + 季度子窗分析脚本
- 预算：CPU ~5min / RSS <4G / 新增磁盘 <100MB
- 预注册 N=4（HYPOTHESIS_REGISTRY 累计台账已记）

## EXP-002 · **PROPOSED**（对应 H-002，依赖 EXP-001 出参考配置）

- 预算：CPU ~10min / RSS <4G；预注册 N=3

## EXP-003 · **BLOCKED-ON-USER**（H-003 新鲜数据入库+冻结；touching-fresh-data 红线）
