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

## EXP-004 · 2026-07-03 · 长 sleeve 诊断（H-004）— **DONE / 诊断结论已档**

- git:（P-E 后 HEAD）；窗口 SEARCH，PIT 纪律 = 每 horizon 限 `label_end < 2025-09-01`（h120 评估止于 2025-03-06）
- 命令：`AI_quant_venv/bin/python3 scripts/analysis/exp004_long_sleeve_diagnostic.py`
- 结果：长 sleeve h120 ICIR 2.24（全族最高）且唯一 bear-regime 非负 IC（h60 +0.069）；但与 mid 秩相关 0.783（高冗余）；k10 静态掺长伤最差季度（−2.6%→−16%/−26%），k30 掺 0.5 微升（+12.6%→+15.5%）
- 判定：weight=0 在 k10 生产书**当前可辩护**；熊市保险价值 + 宽书交互 → 派生 H-005 两个候选设计（未立项）。试验数 +0（诊断）
- 资源：2.6s / RSS 0.91 GiB；详见 LONG_SLEEVE_DIAGNOSTIC.md

## EXP-008 · 2026-07-04 · H-008 走式验证 — **RUNNING（GPU 已授权）**

- 协议：WALK_FORWARD_PROTOCOL_H008.md（4 折 expanding、embargo 126 交易日、候选 N=6 预注册、选择指标先验=中位折 CAGR、全 gate；累计 N=50）
- git: `4ac8e8b`（runner+evaluator 入库）；启动 2026-07-04；runner=`scripts/analysis/run_wf_h008.py`（RAM 48G/OOM 重试一次/磁盘 6G 守卫，逐 sleeve ledger → `wf_h008/runner_ledger.jsonl`）
- F1/short_5d 起跑实测：GPU util 99%；评测驱动 `exp008_walkforward_eval.py` 待重训完成后执行（F4 复用生产 retrain，零 GPU）
- 纪律：新鲜窗零接触（guard 强制）；候选定义冻结；无中途调参

## EXP-003 · 2026-07-03 · 新鲜数据入库+冻结（H-003）— **RUNNING（用户已批准）**

- 发现①：TickFlow SDK 2026-06 破坏性变更（`start_date/end_date`→epoch-ms `start_time/end_time`）导致日更脚本静默失效 = **panel 冻在 2026-05-18 的根因**
- 发现②：SDK 现默认返回**今日向前复权**历史（600519 05-18 close 1292.31 ≠ panel 1323.00）；`adjust="none"` 与 panel as-of-day 基准精确相等；volume 单位 = 手（panel = 股，×100 换算，实测 49,661×100≈4,966,097）；amount/OHLC 完全一致
- 修复：`update_market_panel_daily.py` 改 epoch-ms + `adjust="none"` + volume×100（含注释审计线索）
- 冒烟纪律：连通性冒烟**不写 panel**（`--max-symbols` 部分追加会因 `> last` 过滤永久毒化后续全量追加——已识别并规避）
- 冻结守卫先行：2026-05-19→2027-12-31 已入 quarantined_windows.json（frozen_future_holdout），数据未落地前评测已 fail-closed
- **首轮摄取 QC（2026-07-03 夜）**：appended 73,284 行 / new_max 2026-07-02，但 ①**2026-05-19 整日缺失**（TickFlow 掉首日，两次实测复现；根因修复=start 提前 3 天缓冲）②**1,281/3,653 symbols 全窗失败**（限流；每日覆盖 3,637→2,361）③05-20 的涨跌停旗标错用 05-18 prev-close
- 修复：`scripts/repair_fresh_window_20260704.py`（count=40 全量重取 + 3 次退避重试 + 仅插缺失行不覆盖 + **全窗旗标重算** + tail 备份）后台运行中；完成后 QC → FRESH_HOLDOUT_FREEZE_MANIFEST.md
