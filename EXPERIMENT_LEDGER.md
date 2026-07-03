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

## EXP-001 · 2026-07-03 · 族稳健 blend（H-001）— **DONE / H-001 未获接受（信息量高）**

- git: aa5b322（执行时 HEAD）；输入：ensemble_composite.parquet（sha 见 production_blend.json）
- 窗口：SEARCH 2024-08-28→2025-08-31（quarantine 断言）；k=10 固定；variant C；N=4 预注册
- 命令：`AI_quant_venv/bin/python3 scripts/analysis/exp001_family_robust_blend.py`
- 结果（CAGR / maxDD / 换手 / 最差季度）：
  C1 先验平均 +38.7% / 20.2% / **0.051** / −18.1%；C2 生产 rank(1,1,0) +73.2% / 14.5% / 0.159 / −10.3%；
  **C3 rank 中位 +54.0% / 12.6%(最佳) / 0.336(超标) / −8.4%(最佳)**；C4 rank(1,1,1) +67.8% / 21.2% / 0.155 / −28.4%
- 判定：无聚合配置全门通过（C3 仅败于换手门）⇒ H-001 按预注册标准**不接受**；C2 保持 likely_overfit 标签。
  诚实注记：本窗=C2 自己的选择窗，比较结构性偏向 C2；全员 2024-12→2025-02 亏损（族级 regime）。
  C1（曾用作 holdout 信任锚）在窗内明显弱 + 换手 0.05 过低（raw-score 平均导致排名僵化）。
- 资源：63.5s / RSS 2.46 GiB；产物 runtime/reports/v89_closed_loop/exp001_family_blend/（<1MB）

## EXP-002 · 2026-07-03 · C3 换手 EMA（H-002）— **DONE / H-002 全门通过（带窗口注记）**

- git: aa5b322；窗口/评测同上；N=3 预注册（α∈{0.3,0.5,0.7}，per-symbol ewm(adjust=False)）
- 命令：`AI_quant_venv/bin/python3 scripts/analysis/exp002_turnover_ema.py`
- 结果：C3_raw +54.0% / DD 12.6% / turn 0.336 / worstQ −8.4% →
  ema0.3 **+82.5%** / 14.1% / **0.022** / −3.0%；ema0.5 +71.4% / 18.0% / 0.041 / **+27.6%（四季全正）**；
  ema0.7 +74.0% / 14.5% / 0.077 / +5.4%（四季全正）
- 判定：三个 α 全部通过（换手 ≤0.10 且 CAGR/最差季度未劣化 −3pp）⇒ **H-002 接受**：EMA 平滑把换手压 4–15 倍且不弃收益。
  ⚠ 注记：绝对量级属被复用的 SEARCH 窗，仅机制可信（降换手成本 + 牛腿持有效应）；窗内只有一个回撤事件，慢书的"慢出场"风险未被检验。α 加冕**推迟**至 walk-forward/FRESH 窗（不做在本窗挑 max 的事）。
- 累计试验台账：blend 族 37+4+3=**44**
- 资源：59.5s / RSS 2.1 GiB；产物 runtime/reports/v89_closed_loop/exp002_turnover_ema/（<1MB）

## EXP-003 · **BLOCKED-ON-USER**（H-003 新鲜数据入库+冻结；touching-fresh-data 红线）
