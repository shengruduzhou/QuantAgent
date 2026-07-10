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

## EXP-013 · 2026-07-06 · 低 churn 书 × 快速 de-risk 合成（H-013，N=2 先验）— **INVALID（处理从未被施加）→ 触发 INC-E1 重大发现**

- git 注册 `4fd2b20`（先注册后跑）；24 次 variant-C；产物 wf_h008/exp013_synthesis/
- 表面结果（S1/S2 与 W2 基线折表几乎逐点相同、换手 0.008–0.018、mean_gross 0.68–0.88）触发执行取证：**R2a flip 日 invested fraction 0.998 纹丝不动 → overlay 从未表达**
- 判定：**INVALID-AS-DESIGNED**（非 REJECTED——处理未施加，无信息量）；硬停止条款照常生效：H-008 4 折冻结至 FRESH 首读或用户重开
- 取证链（3 探针）→ 根因 = **INC-E1 执行模拟器跨日订单去重缺陷**（见下一条与 EVALUATOR_ORDER_DEDUP_BUG.md）

## INC-E1 · 2026-07-06 · 执行模拟器跨日 (symbol,side) 静默吞单 — **全评估栈级缺陷，已证实+量化，修复方案待用户批准**

- **机制**：`OrderManager._make_id` 确定性 sha1（signal_id="manual" 恒定）+ `_submit_all` 对 history 已有 id 静默 `continue` + 模拟器全期共用一个 manager 且从不 `reset_daily_counters` ⇒ **每只股票整个回测最多买一次、卖一次**，重复同向订单无审计地消失
- **4 行复现**：tw=[0.50,0.25,0.50,0.50] 单票 → d3 回补单消失，仓位永停 25%
- **量化**：①F1 C3_ema0.7 k10 素书：意图订单值 **81.6% 被静默丢弃**（1,796 单/141.6M vs 成交 31.9M，当日意图口径）②W2×恒定 0.5 gross：首日 48.5% 正确、随后每日+4~5% NAV 爬回满仓 ③W2×R2a flip 日 invested 0.998 不动
- **影响面**：全部 variant-C（含 +17.3% 信任锚、EXP-000..013、PBO/DSR 重放）；失真分层=素书中度（重入场+drift 再平衡被丢）/平滑书重度（增量全丢，"低换手"是伪影）/overlay 首卖可执行但 re-risk 回补全丢；**EXP-011 发现③的路径噪声大部分即本 bug**（毫厘成交差决定谁先占 history 槽→级联）；基准线不受影响（不经模拟器）
- **纪律**：EXP-008..013 全部结论戳 `pre-INC-E1`，修复重跑前不得引用；红线"trusted evaluator 语义变更先问"→ **补丁只提案未应用**（两行修复：日循环 reset_daily_counters + signal_id 带日期）；回归契约测试入库 tests/test_order_dedup_regression.py（xfail strict=True，修复落地即强制摘标）
- **为什么以前没炸**：单测用逐日新 manager/唯一 signal_id；重放保真=坏模拟器 vs 同一坏模拟器（Spearman 0.9922）；只有"同名多次同向交易"书型显形——本周期书构建实验第一次系统性触碰
- 再验证顺序提案（修复批准后）：单测→信任锚（v8.9 rankfix +17.25%、plus7 holdout 族）→EXP-008 折表→EXP-009..013→PBO/DSR 全量；≈2-3h CPU
- 累计 N=**65**（EXP-013 两候选照记，虽 INVALID）

## INC-E1 修复 PROMOTED + 再验证级联 · 2026-07-06 · **用户批准 → trusted-evaluator 默认修正，全部 pre-INC-E1 数字重跑**

- **批准**：用户 2026-07-06 明确批准 "Promote fix + full re-validation"（红线"trusted evaluator 语义变更先问"满足）
- **修复**：`fix_cross_day_order_dedup` 默认 False→**True**（commit `7f09453`）；日循环 `reset_daily_counters()` + 按日 `signal_id=bt-YYYYMMDD`；置 False 可复现 pre-INC-E1 旧模拟器（取证用）；OrderManager.reconcile 新增 signal_id 参数（默认 "manual"，实盘幂等路径不受影响，模拟器是唯一调用者）
- **验证**：buy→cut→rebuy 默认 3/3 成交（旧 2/3）；`tests/test_order_dedup_regression.py` 摘除 xfail 标记（现断言修正行为）+ 新增旧路径复现测试（flag=False→2 成交）；95 execution 测试全绿；生产 materializer 字节等价**不受影响**（只 blend sleeve 分数，不经模拟器）

### 再验证① EXP-008 折表（corrected，commit `2825c97`）— **重大反转**
- 命令：`AI_quant_venv/bin/python3 scripts/analysis/exp008_walkforward_eval.py`（默认拾取修正后 sim）；产物 wf_h008/{wf_summary,candidate_fold_metrics,stitched_daily_returns}，pre-INC-E1 副本存 wf_h008/pre_inc_e1/；报告 EXP008_CORRECTED_INC_E1.md
- RSS 峰值 **2.0 GiB**；时长 **210s**；CPU-only；零重训；零 fresh-holdout 接触（4 折全 OOS<2025-09-01，guard 武装）
- **换手 3–13× 上修（核心伪影）**：C3_ema0.7 maxTurn 0.259→**1.035**；C3_ema0.3 0.070→**0.643**；全候选 0.57–1.35/日 → **EXP-011"换手已在书层解决"被推翻**（是被丢的增量单，非低 churn）
- **DSR 崩塌**：C3_ema0.7 0.736→**0.026**；全候选 DSR<0.05 → N=50 多重检验校正后**无任一 blend 有显著换手调整后 Sharpe**
- **中位折 CAGR 崩塌**：C3_ema0.7 +33.0%→**+1.3%**；C3_ema0.3→+7.8%（现最佳）；C3_ema0.5→+3.9%；incumbent C2 +23.8%→**−24.6%**（现全场最差）；全候选中位超额 vs 基准为负（−24%..−34%）
- **F2 崩塌更深**：−53.7%..−70.9%（vs bench −33.1%）；C3_ema0.7 最不坏 −56.7%
- **15bps 敏感性（重生成 corrected）**：C2 每折 −47%..−78%；C3_ema0.7 −31%/−65%/+8%/+1% → 真实换手下极端成本敏感
- **fold-block PBO 0.833→0.167**（一致平庸，非利好信号）
- **方向守恒**：EMA 平滑仍碾压快速逐日重选书（C1/C2/median）→ H-008 定性结论（平滑有益、C2 非强锚）存活，但经济性摧毁、换手门普遍严重违反

### 再验证② 信任锚 v8.9 rankfix k50 clean-OOS（corrected）
- 命令：`baseline_protocol.py --predictions .../v89_rankfix_20260613_1044/ensemble_composite.parquet --top-k 50 --start 2024-08-09 --end 2025-08-29 --score-column composite_score --variants C_flags_eligible_delay1`
- **corrected ann +30.09%**（vs pre-INC-E1 +17.25%——但后者窗口更长含 2026 回撤，不可直接比）；**excess −35.23%**（bench +65.32%，clean bull 窗口大幅跑输被动等权）
- **关键洞察**：INC-E1 对收益方向**不定**——k10 快书隐藏换手成本（修正后收益↓）；k50 牛市窗口冻结回补（修正后收益↑）。诚实结论不变且被强化：strat 在 clean 窗口大幅跑输等权基准

### 再验证③ EXP-011 book-churn（corrected）— 进行中，见下批次
- pre-INC-E1 副本存各 exp0{09..13}/pre_inc_e1/；EXP-011/012/013 硬编码 BASE 门槛为 pre-INC-E1 载体值，需按 corrected 载体重判（在报告中重裁，非改脚本）

### 再验证④ EXP-010 R2a hysteresis overlay（corrected，EXP010_CORRECTED_INC_E1.md）— **R2a 崩塌优势坐实为伪影**
- 命令：`AI_quant_venv/bin/python3 scripts/analysis/exp010_hysteresis_overlay.py`；77s，RSS 1.9 GiB；pre 副本 exp010_hysteresis/pre_inc_e1/
- pre-INC-E1 R2a F2 **−16.8%**（周期最佳风险画像）= 伪影：de-risk 卖出后 bug 丢弃全部 re-risk 回补单 → 书冻结防御。修正后 R2a 正常回补付真实往返
- **corrected R2a**：F1 −20.3%/DD14.5% · **F2 −48.5%/DD28.2%**（vs 载体 −56.7%/33.9% = +8pp CAGR/+6pp DD）· F3 +16.7%（vs +33.4% 上行被稀释 17pp）· F4 +14.7%（−7pp）；换手 0.74–0.86，mean_gross 0.68–0.88
- **结论**：R2a 是真实但温和且有成本的崩塌对冲（非免费午餐）。**书层 B2_minhold10 全面碾压 overlay 层 R2a**：B2 崩塌改善更多（−40.2% vs −48.5%）且中位收益↑（+36.4%）而 R2a 崩塌改善少且稀释收益 → **churn/崩塌控制归属书构建（min-hold）而非 gross 切换 overlay**，修正数据更强支持 Track A 优于 overlay 线
- **Track D 再定范围**：gross-exposure RL overlay 的基线机制（R2a）已被书层压制 → RL 价值（若有）应为 min-hold 书之上的 turnover-aware 控制器，非独立 gross scaler
- N 不变（冻结候选重跑）。EXP-009 raw overlay (R1/R2/R3) 与 EXP-012 wide-book 修正重跑待做（同属 user-authorized 再验证，无新 trial）

## H-015 双轨换手比较（EXP-015，Track L vs Track H）· 2026-07-07 · **Track L 验证 / Track H 拒绝（成本不生还）**
- track: dual；候选 8（4L+4H，先验冻结 registry H-015）；N 65→**73**；命令 `AI_quant_venv/bin/python3 scripts/analysis/dual_track_eval.py`；产物 exp015_dual_track/{results.json,dual_track_metrics.csv}；报告 DUAL_TRACK_RESULT_H015.md
- 数据：wf_h008 冻结 sleeve 预测 + silver panel（sha 见 H-003 manifest）；窗口 H-008 F1–F4（全 OOS<2025-09-01，guard 武装）；bench eqw-all-A；corrected sim；8/15/25bps
- RSS 峰值 **2.06 GiB**；13m25s；CPU-only；零重训；零 fresh 接触
- **净指标（8bps median CAGR / median excess / maxTurn / F2 / med@25bps / DSR）**：
  - L1_c3ema07_minhold10 **+36.4% / +14.4% / 0.202 / −41.0% / +24.1% / 0.055**（最佳整体）
  - L4_c3ema07_reb10 +31.8% / +9.9% / 0.190 / −46.5% / +21.8% / 0.041
  - L3_midlong_minhold10 +27.4% / +5.5% / 0.200 / **−33.0%（最佳崩塌）** / +16.9% / 0.040
  - H4_short_minhold3 +16.8% / −12.0% / 0.627 / −51.8% / **−10.7%（25bps 死）** / 0.004
  - L2_midlong_ema07 +12.0%（plain 无 hold→churn 1.01 弱）；H1/H2/H3 全负（churn 1.2–1.44，25bps −48~−57%）
- **门**：Track L L1/L3/L4 各 **4/5**（仅差 worstDD：崩塌折 35–37% vs 载体 33.9%）；Track H H4 4/5 但**差关键 med@25bps 门**（成本生还=H 轨定义门），其余 1–2/5
- **PBO 0.0**（L1/L4 一致占优→IS 最佳从不 OOS 低于中位）；**DSR 全 <0.06**（N=73 生产统计门不过）
- **结论**：**Track L 验证为稳健路径；Track H 因不生还真实成本被拒绝**。换手控制（min-hold/reb-throttle）而非周期是杠杆（L2 plain churn 1.01 弱，加 min-hold→L3 +27.4%；快信号 H4 min-hold-3 亦被救 +16.8%）。最佳稳健 = **L1_c3ema07_minhold10**（首个中位超额为正 +14.4%，vs 载体负）；最佳防御 = L3（F2 −33.0%）。**均非生产就绪**（worstDD ~36% 崩塌折 + DSR<0.06）；折已重挖，FRESH 为仲裁。无生产提案。

## Track C 因子批次 1（defensive/low-turnover，DUAL_TRACK_FACTOR_BATCH_PLAN.md）· 2026-07-07 · **1 survivor: D1_low_vol_20**
- track: C（因子生成）；候选 7（先验冻结）；命令 `AI_quant_venv/bin/python3 scripts/analysis/dual_track_factor_batch.py`；产物 FACTOR_CANDIDATE_LEDGER.csv
- 窗口 2023-07-03..2025-08-29（pre-quarantine，断言）；PIT-safe DSL（quantagent.factors.expr）；116s，RSS 2.70 GiB，CPU-only，零 fresh 接触
- **验收逻辑修正**（第一版 bug：sign-agnostic accept 收了负 IC capacity-trap + 去相关把 D1/D7 双杀）→ 现要求 oriented-positive IC + 低换手 + 成本生还 + (defensive)崩塌 IC≥0 + 去相关簇保最优
- **survivor: D1_low_vol_20 = −TsStd(Returns(Close,1),20)**：rank_IC10 +0.080 / ICIR +0.35 / 换手 0.074（≈13.5 日持有）/ **F2 崩塌 IC +0.080（崩塌期仍有效=防御）** / LS@25bps +0.0045（成本生还）→ 正是 H-015 残余崩塌（信号级）所需杠杆
- D7_downside_range redundant（0.91 corr D1）；D6_vol_compression reject（换手 0.329>0.15，但 ICIR 0.48 最高→留待中换手批次）；D2/D3/D4/D5 reject（负 oriented IC=反转/流动性溢价，long 侧=capacity trap）
- **物化计划**：注册 synth_low_vol_20（不入生产）；下一步集成测试 = D1 rank 以 0.3 权重 tilt corrected C3_ema0.7 载体 × L1 min-hold-10 书，看是否改善 F2 崩塌/worstDD；数据集重建延后至集成测试证成。FRESH 仍为仲裁

## EXP-016 · D1_low_vol_20 集成测试（H-015 物化计划）· 2026-07-07 · **防御因子 tilt 修复崩塌/DD，代价=中位收益减半（真实 trade-off）**
- track: dual（Track C 因子 → Track L 书集成）；候选 1 新（D1 tilt w=0.3；w=0 复现 L1 已计）；N 73→**74**；命令 `AI_quant_venv/bin/python3 scripts/analysis/dual_track_d1_integration.py`；产物 exp016_d1_integration/results.json
- 载体 = corrected C3_ema0.7 rank ⊕ 0.3×D1 rank；书 = min-hold-10（L1 赢家）；corrected sim；8/15/25bps；4 折；200s，RSS 2.14 GiB，零重训，零 fresh 接触
- **D1 tilt vs L1 baseline**：中位 CAGR8 +36.4%→**+18.6%**（减半）；worst fold/F2 崩塌 −41.0%→**−27.1%（+14pp，全测最佳崩塌）**；**worstDD 36.6%→24.8%（−12pp，现过门<33.9%）**；median@25bps +24.1%→+9.2%（仍生还）；换手 0.198（不变，D1 低换手不加 churn）；F4 +71.2%→+35.9%（牛市上行被稀释=防御代价）
- **结论**：D1 防御 tilt 精确修复 H-015 残余崩塌/DD 失败（唯一未过的 worstDD 门现过），代价=中位收益减半（低波 tilt 让出高动量牛市上行）。**真实风险/收益权衡非免费午餐**：Calmar 0.99→0.75（中位降幅>DD 降幅），但绝对 worstDD 与崩塌生还大幅改善。最佳 drawdown-adjusted/崩塌生还候选 = **L1_d1tilt_w30**（worstDD 24.8%、F2 −27.1%、中位 +18.6%、25bps 生还 +9.2%）；最佳原始 CAGR = L1 baseline（+36.4%）。w 不扫（避免 fold 调参）；FRESH 为仲裁。均非生产就绪（DSR 未测新配置，折已重挖）。无生产提案。

## EXP-017 · 基本面质量 tilt 集成（H-017 阶段2）· 2026-07-07 · **REJECT 崩塌防御——长短 IC 不等于长多书改善**
- track: dual（Track C 基本面→Track L 书）；候选 1 新（quality tilt w=0.3）；N 74→**75**；命令 `dual_track_d1_integration.py --factor quality`（复用泛化后的集成 harness）；产物 exp017_quality_integration/results.json
- 载体 = corrected C3_ema0.7 rank ⊕ 0.3×QF_quality rank（roe+net_margin+gross_margin 按日 rank-mean，+1 日 lag）；书 = min-hold-10；206s，RSS 2.64 GiB
- **quality tilt vs L1 baseline**：中位 +36.4%→**+34.2%**（几乎不减，远好于 D1 的减半）；**F2 崩塌 −41.0%→−45.1%（更差 4pp！）**；worstDD 36.6%→37.8%（略差）；**F1 弱折 +1.5%→+7.8%（改善！DD 15.6→12.4%）**；F3/F4 稀释；median@25bps +24.1%→+20.9%（仍强）；换手不变 0.196
- **关键方法论发现**：因子的**截面长短 crash-IC（+0.08）不等于集中长多 top-10 书的崩塌改善**——quality 在全宇宙 long-short 崩塌有效，但并入动量 top-10 书后崩塌反而略差。D1 低波有效因其直接剔除高波崩塌名（对长多书有效），quality 是弥散基本面特征。
- **判定 REJECT 崩塌防御**（未改善崩塌/DD——实际瓶颈）；但 quality **改善 F1 弱/平折**且几乎零收益代价 → 潜在 regime-conditional（弱/震荡市）用途，非崩塌防御。三方对比：L1 baseline（+36.4%/F2 −41%/DD 36.6%）vs L1+D1（+18.6%/−27.1%/24.8%）vs L1+quality（+34.2%/−45.1%/37.8%）。**D1 低波仍是唯一验证的崩塌杠杆。** 均非生产就绪，FRESH 仲裁。无生产提案。

## EXP-018 · 板块轮动 tilt（sector relative strength）· 2026-07-07 · **REJECT——顺周期，崩塌灾难性恶化**
- track: dual（Track C 板块→Track L 书）；候选 1（sector_rs tilt w=0.3）；N 75→**76**；命令 `dual_track_d1_integration.py --factor sector_rs`（复用泛化 harness + `factors/sector_rotation.sector_relative_strength`，零新模块）；产物 exp018_sector_integration/
- 载体 = corrected C3_ema0.7 rank ⊕ 0.3×板块20d相对强度 rank；书 = min-hold-10；208s，RSS 2.17 GiB。**PIT 注记：sector_map=current_snapshot 成员=轻度泄漏（成员稳定）**
- **sector_rs tilt vs L1 baseline**：中位 +36.4%→+34.8%（略降）；**F1 弱折 +1.5%→+16.3%（大改善）**；**F3 牛 +97.2%→+111.8%（改善）**；**F2 崩塌 −41.0%→−66.4%（灾难 −25pp！）** worstDD 36.6%→42.5%；F4 +71.2%→+53.4%；median@25bps +24.1%→+22.8%；换手不变
- **判定 REJECT**：板块动量顺周期——弱/牛市助力但崩塌灾难性放大（领涨板块崩得最狠，2024H1 题材崩盘）。坐实 Stage 8"naive 板块动量 whipsaws"。
- **三 tilt 综合（D1/quality/sector）统一图景**：F2 崩塌由顺周期/高波暴露驱动——仅**去风险**（D1 低波）能救；**加动量**（板块）放大；**弥散基本面**（quality）够不到集中长多书。**L1 baseline 仍是收益冠军（中位 +36.4%、超额 +14.4%），无 overlay 能在不炸崩塌的前提下提收益。** 均非生产就绪，FRESH 仲裁。

## EXP-019 · regime-conditional D1 低波 tilt（仅崩塌 regime）· 2026-07-07 · **ACCEPT（最佳 Calmar/风险调整收益）——首个改善 drawdown-adjusted return 的 overlay**
- track: dual（Track C 因子 × 崩塌 regime overlay）；候选 1（d1_regime w=0.5 crash-only）；N 76→**77**；命令 `dual_track_d1_integration.py --factor d1_regime --weight 0.5`（复用 gross_series R2a + D1，零新模块）；产物 exp019_d1_regime_integration/；198s，RSS 2.22 GiB
- 载体 = corrected C3_ema0.7；tilt = D1 rank，**权重 0.5 仅当 R2a 崩塌 regime（bench<MA60 confirm-5，t−1观测t执行）激活，其余=纯动量**；书 = min-hold-10
- **d1_regime vs L1 baseline**：中位 +36.4%→**+25.3%**（远好于静态 D1 +18.6%）；**F2 崩塌 −41.0%→−32.3%（+9pp）**；**worstDD 36.6%→22.1%（−14.5pp，全场最低）**；F1 +1.5→+2.6%(DD 15.6→11.5%)；F3/F4 稀释（+97→+53、+71→+48：R2a 在牛市回调期误触发施加防御 tilt）；median@25bps +24.1%→+16.5%；换手不变 0.197
- **5-variant Calmar 综合（中位CAGR/worstDD）**：baseline 0.99；D1 静态 0.75；**D1 regime 1.14（最佳）**；quality 0.91；sector 0.82。**regime-D1 是唯一改善风险调整收益（Calmar 1.14>0.99）的 overlay，非单纯拿收益换 DD。**
- **判定 ACCEPT 机制**（过"更低 DD + 更好崩塌生还，收益代价可接受"验收门）：最佳 drawdown-adjusted 候选。残余成本=R2a 触发器在牛市回调误触（不扫参调触发器=避免 fold-mining）。**收益冠军仍是 L1 baseline（+36.4%，用户容忍高 DD）；风险调整冠军 = L1+D1_regime（Calmar 1.14）。** 非生产就绪（DSR 未测新配置、折已重挖、需 FRESH）。无生产提案。

## EXP-020 · 2026-07-08 · PIT 估值+基本面训练集集成（H-020，数据工程票）— **DONE / ACCEPTED（PIT 全过，估值信号强）**

- git 注册：1a39ddd（先注册后建）；VALUATION_FUNDAMENTAL_INTEGRATION_PLAN.md
- 诊断：生产 plus7clean（327列）零 firm-level 估值/基本面值，仅 missing_* 占位；但 LONG 腿 select_features 已按名 whitelist（架构缺口=数据缺口）；silver/fundamentals/metrics_panel.parquet 已是 PIT 面板（3654 syms，announce_date+available_at，eps/bps/ocfps/roe/margins/growth/debt）；valuation silver 目录空
- 复用：metrics_panel 原样、enrich merge_asof PIT 模式、trainer name-pattern、修正 strict_v8；新增 build_valuation_fundamental_features.py（TTM 去累计+比率+分位，向量化 rolling-rank）+ merge_valuation_fundamental_into_training.py（分块 RAM 安全）+ audit_val_fund_pit.py + ic_precheck_val_fund.py
- 产物：val_fund_quarterly.parquet（257k）+ val_fund_features.parquet（6.78M，57s/8.6G）+ training_dataset_alpha181_exec_v89_plus7clean_fund.parquet（6,781,038 行，行数不变，335 特征，feature_version=plus7clean_fund，schema_hash e815e492，55s/11G）
- TTM 自检：000001.SZ 2025Q3 eps_ttm=2.08 精确（去累计+滚动4Q）
- PIT 审计全过：G-PIT-3 as-of roe 4000/4000（修正同 available_at 多报表 tie-break：19.2% 组>1 报表，latest period_end 定序胜出）；G-PIT-4 当日截面分位 max|diff|=0.0；负 pb/pe=0%（亏损→NaN 设计）；max date 2026-05-13 隔离前，新鲜窗零接触；近年 pb/roe≈99.7% 覆盖
- IC 预检（隔离前，原始截面 rank-IC）：**pb vs 60d IC −0.091（t −28.9，ICIR −0.67）估值是强信号且此前缺失**；valuation_percentile +0.061（t +17.6）；pb_own_pctile_2y −0.063（t −25.6）；pe_ttm −0.035（t −10）。原始质量/成长 60d 弱负（roe −0.024、growth −0.012）=regime 混淆（小盘牛），作模型输入非 tilt
- 诚实排除（无 PIT 数据不造假）：PS/EV-EBITDA/股息率/分析师预期/turnover_rate/market_cap（无股本）
- 验收：本票纯数据工程，PIT 全过+覆盖达标+行数不变+schema 发出 ⇒ ACCEPTED；不动模型/生产；解锁 H-021 GPU 重训消融
- 累计 N：不计（数据工程票），维持 77

## EXP-021 · 2026-07-08 · GBM 消融：估值/基本面增量 alpha（H-021，N=4 先验）— **DONE / GPU (H-022) NO-GO，估值弱正/基本面负**

- git 注册：见 H-021 commit；LightGBM 截面 ranker，label=forward_return_60d 逐日 rank，train 2018..2022-12-31 / embargo / OOS test 2023-04..2025-08-29（隔离前，新鲜零接触）；494.6s；**峰 RSS 51.5 GiB（超 <16G 估计，未 OOM；后续须减特征/加流动性过滤降内存）**
- 折表（OOS mean rank-IC / ICIR）：A base(301) **0.18208 / 1.083**；B base+val(309) 0.18588 / **1.174**（Δ IC +0.0038，ICIR +0.09，pb/book_yield/earnings_yield 进 top15）；C base+fund(312) 0.18024 / 1.013（Δ −0.0018，**基本面伤 IC**）；D full(320) 0.17833 / 1.061（Δ −0.0038）
- **判定（先验门：B 或 D ΔIC≥+0.005 且新列进 top15 → GPU go）：无一达 +0.005 ⇒ GPU (H-022) NO-GO**。估值小幅正增量+改善一致性（ICIR）但 <门；原始基本面为噪声（regime 混淆，与 [[full-universe-deep-mlp-no-edge]] 及 EXP-017 一致）
- **诚实注记（方法学，非交付但重要）**：A base top15 由**逐日常量**特征主导（flow_margin_sh、idx_*_close、macro_shibor_1y）——这些无截面信息，作 regime/time 门；叠加全（含不可交易）宇宙 ⇒ 0.18 IC 夸大可交易 edge（phantom breadth，[[honest-baseline-truth]]）。估值标准 IC −0.09 大半已被 base 的 size/技术轴吸收 ⇒ 增量小=冗余，非"估值无用"
- **科学结论**：把原始估值/基本面塞进同一全宇宙模型不移动指针；提取估值价值的路径 = ①**可交易/容量约束宇宙**（去掉不可交易微盘 size 效应，估值冗余可能消失）②size 中性化后加估值 ③regime 条件化。⇒ 派生 H-022（CPU：cross-sectional-only base + 流动性可交易宇宙的估值增量重测；先于任何 GPU）
- 累计 N：+4 → 81

## EXP-022 · 2026-07-08 · 可交易宇宙估值增量（H-022，N=2 先验）— **DONE / 确认估值冗余（GPU NO-GO），跨宇宙复制**

- 控制 H-021 两混淆：截面-only base（250 特征，剔除逐日常量 idx/macro/flow）+ 可交易宇宙（eligible ∧ amount_mean_20d≥当日中位）；train 2018..2022 / OOS 2023-04..2025-08；73s；**峰 RSS 31.7 GiB（RAM 修复生效，vs H-021 51.5G）**
- 结果：A base_xs IC **0.15787** ICIR 1.067，top-decile 多头 60d **+4.58%（t+9.2）**，多空 decile +5.52%；B +val IC 0.15911（**ΔIC +0.0012**）top-decile 多头 +4.44%（**Δ −0.14pp**）多空 +5.36%。**8/8 估值列进 top15 重要度（模型用了它们）但净增量≈0，top-decile 多头反微降（轻微过拟）**
- **判定（先验门：ΔIC≥+0.005 或 top-decile 多头显著提升）：均不达 ⇒ GPU (H-023) NO-GO**
- **稳健结论（跨 2 宇宙×2 base 设计复制）**：估值/基本面作**原始输入**加进已含 alpha101/181+gtja191 的非线性截面模型，**无增量 OOS alpha**——现有价量因子库已张成估值信号空间（value/reversal/size 已被技术因子捕获），故估值虽有强 −0.09 **单因子** IC 但作模型输入冗余。**非"估值无用"，而是"此建模下冗余"**
- 保留价值：①PIT 估值面板已建可复用（regime 条件/防御 overlay/特定 regime 估值 tilt——EXP-017/019 已证条件价值）②诚实旗标：base 绝对 IC 0.16 偏高（先验因子库属性，非本实验引入；delta 结论不受影响，绝对值可疑留待未来泄漏审计）
- **对总任务的含义**：raw-CAGR 瓶颈**不在特征覆盖**（估值/基本面无增量预测力），在容量/book/执行（signal 多在不可交易名上=phantom breadth）+ 修正模拟器下的稳健性。特征线在无条件 alpha 上收敛
- 累计 N：+2 → 83

## EXP-023 · 2026-07-10 · 学习型 regime→tilt 权重元模型（H-023，N=2 先验）— **DONE / 两轴皆不过 ⇒ REJECT（先验门）；关键诚实发现：手设 overlay 的崩塌保护不可由因果学习复现**

- git 注册 e0a0ad1（先注册后建）；命令 `AI_quant_venv/bin/python3 scripts/analysis/regime_weight_meta.py`；产物 exp023_regime_weight_meta/results.json（含每次 refit 的全部学到权重 trace，可审计）；263s，峰 RSS 4.09 GiB，CPU-only，零重训，零 fresh 接触（IC/label 面板硬顶 2025-08-29，label 前向 10d 永不读隔离窗）
- 设计回顾：blend=(1−τ_s)·carrier+τ_s·tilt_s；组件 D1/quality/sector_rs（与 EXP-016..019 同函数同 PIT lag）；τ_s 与组件构成由 trailing（2018→t−11 embargo）regime 条件日截面 IC（h=10）**纯因果**学得，月度 refit；动量代理只作 τ 刻度。RW1=trend×vol 4 态；RW2=trend 2 态。carrier/book/sim/折全同 EXP-016..019
- **折表（CAGR8 / DD8 / CAGR25 / mean_τ）**：
  - RW1_4state：F1 **+16.4%**/7.6%/+8.7%/0.44 · F2 −32.7%/35.3%/−38.0%/0.19 · F3 +50.5%/11.0%/+35.9%/0.20 · F4 +54.9%/10.3%/+43.5%/0.34
  - RW2_2state：F1 +4.4%/12.7% · F2 −45.8%/36.6% · F3 +90.1%/9.4% · F4 +44.1%/12.4%
- **聚合**：RW1 中位 **+33.4%** / worstDD 35.3% / Calmar 0.947 / F2 −32.7% / med@25bps **+21.3%** / maxTurn 0.202；RW2 +24.2% / 36.6% / 0.663 / −45.8% / +16.1%
- **门判定（先验，跑后未改）**：A 轴（中位>36.4% ∧ DD≤36.6%）：RW1 差中位 3pp ✗；B 轴（Calmar>1.14 ∧ 中位≥25.3%）：RW1 Calmar 0.947 ✗；RW2 两轴皆远 ✗ ⇒ **REJECT 学习型 regime 权重线（本建模下），停线，不看折改规则**。硬门全过：med@25 +21.3% 生还、turn 0.202≤0.25、全因果（embargo 断言）、零隔离接触
- **科学发现（比门判定更重要）**：
  1. **因果学习器 3/4 折胜 L1 baseline**（F1 +16.4 vs +1.5、F2 −32.7 vs −41.0、F4 DD 10.3% vs 基线折 DD 更深）且 3/4 折 DD≤11%；输在 F3 牛折稀释（+50.5 vs +97.2，τ≈0.2 让出动量上行）→ 中位差 3pp
  2. **无法复现 EXP-019 的 worstDD 22.1%**：trailing 2018→2023-06 数据里 crash-highvol 态 D1 IC<0.01（学习器学到 τ=0 不防护），而"D1 在崩塌有效"的手设知识来自 Track C 因子批的 2023-07..2025-08 评估窗——**与折重叠** ⇒ EXP-019 Calmar 1.14 部分是 fold-informed 设计的产物，非纯 OOS 知识。手设 overlay 系列（EXP-016..019）的可信度整体下调一级，FRESH 仲裁地位升级为必要非充分
  3. **vol-split 是真实信息**（RW1 中位 33.4% vs RW2 24.2%，F2 −32.7 vs −45.8）：2 态 pooling 稀释条件 IC 至阈下（RW2 mean_τ 0.04-0.23），4 态才可检测。regime 条件化方向本身有效
  4. 学到的结构可解释且随时间演化：早期（2023-06 trailing）up-lowvol 态 D1 主导（w 0.77）、crash-lowvol 纯 D1、crash-highvol 空；F4 期 mean_τ 升至 0.34（trailing 已含 2024 崩塌证据）——学习器"迟到地"学会了防护，正是因果性的代价
- **FRESH 首读预登记对比集（零新折使用）**：L1 baseline（收益冠军）/ L1+D1_regime（手设风险冠军，fold-informed 嫌疑）/ RW1_4state（纯因果学习者）——三者在 FRESH 窗（≈2026-11 首读）的相对表现直接检验"手设 overlay 是否过拟合折"
- 累计 N：+2 → **85**

## AUDIT-2026-07-10 · evaluator 有效性审计（IC 0.16）+ 数据能力分级 — **evaluator TRUSTED；IC 非泄漏；phantom breadth 定量；Tier A 裁定**

- 报告：EVALUATOR_VALIDITY_AUDIT_IC016.md + reports/tickflow/data_capability_tier.md；回归锁 tests/test_executable_label_convention.py（2 tests 过）；非选择性诊断，N 不变
- **标签审计**：生产标签实为 **delay-1 executable**（close(t+1+h)/close(t+1)−1，入场不可行行已剔除，scripts/build_executable_labels_dataset.py 有意设计）——1d 匹配 100%、60d 94.7%；同日泄漏通道不存在（corr(ret1,fwd60)=−0.0003）；两个保守向缺陷记录：①~5% 60d 标签建于 2026-07-04 panel 修复前（陈旧日历噪声，非前视）②未复权价标签低估股息（反价值偏置）。v7_label_builder docstring 文档漂移已修
- **IC 0.158 量级校准**（同宇宙/窗/标签单因子）：lowvol20 +0.110 / rev60 +0.089 / size +0.091 / low_price +0.075 ⇒ 250 特征 GBM 0.158 = 常规因子结构非泄漏；**GBM top-decile +4.58% ≈ rev60 单因子 +4.51%（模型 decile 级增量≈0）**
- **phantom breadth 定量**（eligible eqw 年化，2023-04..2025-08）：非流动下半 **+25.4%**/流动上半 +11.7%/top20% 流动 **+7.4%** —— breadth 溢价单调消失于流动性
- **数据能力 Tier A（仅 bar）**：TickFlow L2/分钟全 403（live 探针）；磁盘无任何逐笔/委托/深度数据；qlib 1min 仅 2020-09..2021-06；minute_bars 675 syms 大部分在隔离窗；fundflow_minute 仅 1 日×17 syms。微结构线（Phase 5D/7 高级做T）数据不可行；Tier C 最低采购规格已写入报告

## EXP-024 · 2026-07-10 · 冻结冠军容量研究（诊断，零选择，N 不变）— **机械可执行至 ~100M CNY；经济可信容量 ~10–30M；瓶颈=评估器无非线性冲击模型**

- 命令 `scripts/analysis/exp024_capacity_study.py`；产物 exp024_capacity_study/results.json；469s，RSS 1.94 GiB；书=冻结 L1 + L1+D1_regime（RW1 同 carrier 池/换手，结论可迁移）；AUM 格 {1/10/30/100/300M}（依实测持仓 ADV 23–39M、流动性 rank ~0.10 选定）
- **持仓流动性画像（关键）**：冠军书 97% 书日在宇宙流动性下半，中位 rank **0.10（底 decile）**；10% 参与帽下单名日可交易中位仅 2.7M CNY ⇒ **+36.4% 冠军是微盘非流动策略，恰在 phantom-breadth 溢价段选股**
- 8bps 线（L1 中位）：1M +36.4% / 10M +34.0% / 30M +34.8% / 100M +34.8% / 300M +28.3%（300M 帽强制慢入场反降 worstDD 30.8%——附带平滑）；25bps 线：30M +23.9% / 100M +25.1%。d1_regime 全格稳定 +24~28%、DD 19–22%
- **诚实判读**：微小退化是**线性 8bps + 10%/日参与**假设的产物——底 decile 流动性名以 10% 参与率成交的真实冲击 ≫25bps。故：①参与帽机械可行性证实（订单可跨日建仓，~10 日持有期吸收 2–3 日建仓）②25bps 敏感线 −10pp 中位 ③**可辩护容量声明 = 10–30M CNY（≈$1.5–4M）近基线收益；>30M 不可证（评估器缺 √participation 冲击模型 = 容量声明的硬缺口）**。failed 计数随 AUM 降（530→82）= 小 AUM 的最小手数舍入伪影，非容量信号
- 对 Phase 4 问题"edge 是否经济真实"：**小规模真实**（25bps 生还、入场过滤、参与帽下成立）；**机构规模未证且当前不可证**。下一评估器能力票（若立项）：√冲击模型 + 成交量分布执行
