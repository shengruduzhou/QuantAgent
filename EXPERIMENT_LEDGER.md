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
- **执行事故链（全部按协议中止+诊断，候选/超参零改动）**：
  ①abort#1（git 3b9b23d 前）：runner 未设 `QUANTAGENT_HORIZON_ASSIGNMENT` → F1 short/mid 训在 17/20 特征（prod=22）→ 作废重训；同时发现 8192/1024 长 sleeve 配置在生产史上从未成功；新增**机械 schema-parity 门**
  ②abort#2（5d57906）：长 sleeve 在生产同款 2048/128 下仍 OOM → 诊断出 `--train-micro-batch` 在 dates_per_step=1 下是 **no-op**（按日分组无从再切）+ 碎片化贴顶；expandable_segments 单独不够
  ③abort#3（c0939b1→29e4c63）：加 activation checkpointing（梯度精确，等价冒烟 6 位小数一致）后长 sleeve 训完 5.3h，**被 schema 门拒绝：fold=178 特征 vs prod=90** → 挖出未记录的生产血统 `QUANTAGENT_JUDGMENT_MAX_FACTORS=64`（finish_long_plus7.sh；已补录 production_blend.json lineage）
- 副产品：trainer 获得 `--activation-checkpointing`（Phase 8 资产）；生产 lineage 补全两个 env 依赖；浪费 GPU ≈6.5h（全程有账）

### EXP-008 终局（2026-07-06）— **完成 / C3+EMA 按预注册门 REJECTED（不采纳），相对优势记录在案**

- 重训 9/9 成功（schema 门全过）：short 25–33min、mid 27–40min、long 176–255min；RSS 峰 18.7G；GPU 有效 ≈13h
- 评测：24 次 variant-C（+8 次 15bps 敏感性）；RSS 峰 1.9G/197s；产物 `wf_h008/{wf_summary.json,candidate_fold_metrics.csv,stitched_daily_returns.csv,cost_sensitivity_15bps.json}`
- **折表（CAGR）**：F1(bench−2.6%)/F2(**bench−33.1%**)/F3(+69.4%)/F4(+46.5%)：
  C1 −5.2/−55.2/+29.1/+78.2（中位+12.0，最差−55.2）｜C2 −8.9/−33.0/+82.2/+56.6（+23.8，−33.0，**换手 0.28–0.70/日超 R3**）｜C3中位数 −7.9/−29.7/+39.4/+71.1｜**C3_ema0.7 −6.9/−29.9/+73.0/+77.8（中位+33.0，最差−29.9，worstDD 25.0%，DSR 0.736 全场最高）**
- 15bps 敏感性：C3_ema0.7 中位衰减 −8.8%（F2 几乎不动 −29.9→−30.2）；C2 −22%（F2 −33→**−52.7**）
- 门判定（C3_ema0.7）：胜过 C2 门 ✓、DD 门 ✓（C2 自己 worstDD 31.5% ✗）、成本门 ✓、行业门 ✓、新鲜窗零接触 ✓；**换手门 ✗（max 0.259 > 0.10 承诺；SEARCH 窗的 0.02–0.08 不外推）**、**统计门 ✗（fold-block PBO 0.833 粗粒度、DSR 0.736<0.95）** ⇒ 预注册规则下不采纳
- **科学结论**：①C1 信任锚被走式否定（全场最差）②C2（现生产候选）不稳健：worstDD 超门 + 换手在压力折爆表 ③C3_ema0.7 = 当前证据下最优配置（4/5 轴占优）但族级共同失败模式 = **F2 型崩塌折（2024H1 微盘股崩）无一幸免** → 缺的是回撤/regime 暴露控制层，不是 blend 选择 ④换手本身窗口依赖
- 试验数台账：blend 族 50（含本次 6 复评）；生产配置**不变更**（无 proposal，按指令）

## EXP-009 · 2026-07-06 · 暴露控制 overlay（H-009，N=3 先验）— **DONE / 全部 REJECTED**

- git f52a0f2；12 次 variant-C；102s / RSS 1.8G；产物 wf_h008/exp009_overlay/
- R1 回撤分档 ✗(worstDD 27.0%>25.0%)；R2 MA60 ✗(仅换手 F1 0.360)；R3 波动分档 ✗(中位 26.0%<28.0%)
- 机制：R2 方向正确（F2 +7.1pp、worstDD 23.1%）败于横盘 whipsaw churn → H-010

## EXP-010 · 2026-07-06 · R2 滞回修复（H-010，N=2 先验，**本线最终迭代**）— **DONE / 全部 REJECTED，线关闭**

- git（见 HEAD）；8 次 variant-C；70s / RSS 1.83G；产物 wf_h008/exp010_hysteresis/
- R2a 确认滞回：**风险端全周期最佳**（F2 −16.8%、worstDD 19.9%、中位 +33.2%、F3 +117.9%）但 F1 换手 0.362 ✗（whipsaw >5 日，每切换交易半书）；R2b EMA gross：换手 ✓ 但 F2 −38.0% ✗（双向迟钝）
- 累计 blend+overlay N=**55**；无生产提案；**结构性结论：churn 应在书构建层解（Track A 持有期/节流），不在 overlay 层**；R2a 封存待 FRESH 窗

## EXP-003 · 2026-07-03 · 新鲜数据入库+冻结（H-003）— **RUNNING（用户已批准）**

- 发现①：TickFlow SDK 2026-06 破坏性变更（`start_date/end_date`→epoch-ms `start_time/end_time`）导致日更脚本静默失效 = **panel 冻在 2026-05-18 的根因**
- 发现②：SDK 现默认返回**今日向前复权**历史（600519 05-18 close 1292.31 ≠ panel 1323.00）；`adjust="none"` 与 panel as-of-day 基准精确相等；volume 单位 = 手（panel = 股，×100 换算，实测 49,661×100≈4,966,097）；amount/OHLC 完全一致
- 修复：`update_market_panel_daily.py` 改 epoch-ms + `adjust="none"` + volume×100（含注释审计线索）
- 冒烟纪律：连通性冒烟**不写 panel**（`--max-symbols` 部分追加会因 `> last` 过滤永久毒化后续全量追加——已识别并规避）
- 冻结守卫先行：2026-05-19→2027-12-31 已入 quarantined_windows.json（frozen_future_holdout），数据未落地前评测已 fail-closed
- **首轮摄取 QC（2026-07-03 夜）**：appended 73,284 行 / new_max 2026-07-02，但 ①**2026-05-19 整日缺失**（TickFlow 掉首日，两次实测复现；根因修复=start 提前 3 天缓冲）②**1,281/3,653 symbols 全窗失败**（限流；每日覆盖 3,637→2,361）③05-20 的涨跌停旗标错用 05-18 prev-close
- 修复：`scripts/repair_fresh_window_20260704.py`（count=40 全量重取 + 3 次退避重试 + 仅插缺失行不覆盖 + **全窗旗标重算** + tail 备份）后台运行中；完成后 QC → FRESH_HOLDOUT_FREEZE_MANIFEST.md

## EXP-011 · 2026-07-06 · 书构建层 churn 控制（H-011，N=5 先验，Track A 第一批）— **DONE / 全部 REJECTED（0/5），churn 机制本身被证明**

- git 注册 `1994cd4`（先注册后跑）；40 次 variant-C（20×8bps + 20×15bps）；327.6s / RSS 2.02G；产物 wf_h008/exp011_book_churn/
- 中途一次协议中性修复：实现层防失控 assert（书≤40）与注册定义（B3 无上限）矛盾 → 放宽到 500，候选零改动
- **折表（8bps CAGR）**：B1 −6.2/−43.0/+81.2/+91.5；B2 +14.5/−40.4/+62.4/+79.8；B3 +6.5/−31.6/+66.4/+86.2；B4 +15.0/−40.1/+74.2/+67.1；B5 +7.5/−33.5/+77.0/+60.6（载体基线 −6.9/−29.9/+73.0/+77.8）
- **门判定**：G1 换手 B2 0.041/B3 0.015/B4 0.034 **✓**（4–17× 优于 0.10 承诺；B1 0.153/B5 0.140 ✗）；**G2 worstDD 全员 ✗（30.8–37.4% vs 25.0%）；G3 F2 全员 ✗（全部劣于基线 −29.9%）**；G4 中位全员 ✓（+34.0~+41.1% vs +33.0% 基线）；G5/G6/G7 全过
- **发现①**：churn 已解——换手门在书层用一行规则即可过，且中位 CAGR 反升、F1 由 −6.9% 翻正至 +15.0%
- **发现②（拒绝主因）**：**慢书在崩塌折死得更惨**——载体的每日重选=隐性崩盘防御（分数衰减→快速换出崩塌名字）；keep-zone/锁仓/节流的每一天延迟都直接变成更深回撤；B5 的 confirm-5+0.1/日 ramp 太慢无法抵消。churn 控制与崩盘生存在本信号族上**结构性冲突**
- **发现③（评测器稳健性）**：k=10 折级 CAGR 有 **±3pp/1–2bps 扰动噪声 + 偶发 >20pp 盆地跳变**（B1/F1 bps 扫描 −6.2/−4.2/−8.4/−6.1/+17.2% @8/9/10/12/15bps；已验证确定性+无输入变异）⇒ EXP-008..011 所有 <5pp 折级差异在执行路径噪声内；F2 失败非噪声（5/5 同向 2–13pp）
- PBO 0.833（粗粒度不变）；DSR@N=60：B3 0.885 最高、载体 0.872，无一 ≥0.95；累计 N=**60**
- 结论：无生产提案；B2/B3/B4 机制封存为构件；**下一步=结构性变化而非参数挖掘**（k=30 宽书 + 路径噪声带测量 → H-012 另行预注册；4 折 k=10 挖掘已到收益递减点）

## EXP-012 · 2026-07-06 · k=30 宽书结构稳健性（H-012，N=3 先验）— **DONE / 全部 REJECTED（0/3），基础设施级发现**

- git 注册 `66560e2`（先注册后跑）；48 次 variant-C（4 书 × 4 折 × bps∈{8,9,10}）；407.7s / RSS 1.90G；产物 wf_h008/exp012_widebook/
- **折表（8bps）**：W1 素 k30 −4.6/−31.8/+70.1/+95.8（maxTurn 0.397）；W2 k30+部分调仓 +9.1/−39.4/+58.6/**+96.4**（maxTurn **0.015**）；W3 C2@k30 −8.9/−37.9/+58.3/+41.8（maxTurn 0.508）；REF k10 载体 −6.9/−29.9/+73.0/+77.8
- **门判定**：W1 ✗G1/G2/G3；W2 ✓G1（6× 裕度）✓G4（中位 +33.9% 全场最佳）✗G2（32.3%）✗G3（F2 −39.4%）；W3 ✗G1/G3/G4（中位 +16.4%——现生产候选信号加宽即崩）
- **发现①**：素加宽反而更抖——k30 每日重选换手 0.30–0.51 > k10 的 0.11–0.26（排名边界穿越随 k 增长）；宽书必须配平滑机制
- **发现②（基础设施级）**：**W2 的 bps 噪声带 0.007 vs k10 载体 0.088 = 12× 收窄**（F4 三点位完全一致 +0.964）——全周期唯一折级差异可信的配置形态；W3 噪声带 0.195（C2 信号+集中调仓=最脆）
- **发现③**：崩塌折与书宽无关（W1 −31.8/W2 −39.4/W3 −37.9 vs k10 −29.9）——**F2 暴露是信号级**（模型族在崩塌期持续把崩塌段排前），任何书变换不解，只有 regime 暴露控制（R2a 型）碰到过它（F2 −16.8%）
- PBO 0.667（↓自 0.833）；DSR@N=63 全 <0.95（W1 0.786 最高）；累计 N=**63**
- 结论：无生产提案；**周期结构图完成**：换手已解（partial-adjust 任意 k）+ 路径噪声已解（W2 形态）+ 崩塌未解（书层任何宽度均不可解）⇒ 唯一未测组合 = 低 churn 书 × 快速 R2a de-risk → H-013（本周期最终折接触批次，硬停止条款）
