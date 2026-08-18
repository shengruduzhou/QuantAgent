# Round 21b — R3 因子专家 + R4 选股专家：FIND-R3-01 闭合报告

- 角色 / Role: R3（因子）+ R4（选股）合并席位
- 日期 / Date: 2026-08-18
- 基线 / Baseline: `main` @ `057f8cf`（clean）
- 性质 / Nature: **只读审计**。未修改 `src/`、`apps/`、`services/`，未 commit/push。
- Python: `AI_quant_venv/bin/python3`
- 唯一任务：闭合 **FIND-R3-01（P0）** —— 认证全宇宙面板只有 15 个特征。

> 裁决词典：`PASS`（实测通过）/ `FAIL`（实测失败）/ `unknown`（无证据，**不等于通过**）/
> `BLOCKED_BY_DATA`（缺数据能力，fail-closed）。
> 本文件**增量写入**：每确认一条立刻落盘。

---

## 目录 / Skeleton

- §1 两个面板的关系（Q1）—— 待填
- §2 `FULL_UNIVERSE_GOLD_READY` 认证了什么（Q2）—— 待填
- §3 能否把丰富因子接到全宇宙：阻塞项 + 成本 + 方案（Q3）—— 待填
- §4 PIT 正确性抽查（Q4）—— 待填
- §5 事件类因子 = 0 的成因与补实现依赖（Q5）—— 待填
- §6 选股链路：打分 → 排序 → top-k，不可交易剔除时点（Q6 / R4）—— 待填
- §7 Findings 汇总
- §8 复现命令与实测数字清单

---

## §2 `FULL_UNIVERSE_GOLD_READY` 到底认证了什么（Q2）— **实测结论：不含任何"特征广度"检查**

有**两条**独立的授予路径，两条都读过全文，**都没有任何一项检查特征数量、特征族或因子覆盖**。

### 2.1 授予路径 A：构建脚本自己签发的 `quality_certificate.json`

`scripts/build_u0_full_universe_gold.py:203` `run_quality_checks()`，
`scripts/build_u0_full_universe_gold.py:450` `granted = quality["structurally_valid"]`。

当前代码里的 10 项检查（落盘证书里只有 9 项，见 §2.3）：

| # | check | 它问的问题 | 与特征广度有关？ |
|---|---|---|---|
| 1 | `no_duplicate_security_dates` | `(symbol, trade_date)` 唯一 | 否 |
| 2 | `no_pre_listing_rows` | 无早于上市日的行 | 否 |
| 3 | `no_post_delisting_rows` | 无晚于退市日的行 | 否 |
| 4 | `universe_includes_delisted_names` | 面板含已死名字（三态，Round 18/19 新增） | 否 |
| 5 | `adjustment_mode_declared` | 复权口径唯一 | 否 |
| 6 | `volume_non_negative` | volume ≥ 0 | 否 |
| 7 | `close_positive` | close > 0 | 否 |
| 8 | `labels_present` | **存在**任一 `forward_return_*` 列（`bool(label_columns)`） | 否 |
| 9 | `masks_present` | **存在**任一 `mask_*` 列（`bool(mask_columns)`） | 否 |
| 10 | `no_infeasible_entries` | 每行有可行的 t+1 入场 | 否 |

注意 #8/#9 是**存在性**判据（`bool(list)`），不是**充分性**判据。
`FEATURE_COLUMNS`（`build_u0_full_universe_gold.py:157-162`）是一个**硬编码的 15 元组**，
证书**从不读它**，也从不比对它与任何因子注册表。

### 2.2 授予路径 B：`readiness_tiers.py` 的 tier 判定

`src/quantagent/safety/readiness_tiers.py:169-199` `full_universe_gold()` 的 8 条 requirement：
`gold_manifest_present` / `dataset_hash_recorded` / `adjustment_mode_declared` /
`no_duplicate_security_dates` / `no_out_of_life_rows` / `missingness_masks_present` /
`labels_present`（**文件存在即可**）/ `lineage_present`（**文件存在即可**）。

⇒ **6/8 是"文件存在"或"哈希被记录"，2/8 是行级唯一性/生命期。零特征检查。**

### 2.3 落盘证书比当前代码少一项 —— 证书是旧版脚本产物

`runtime/data/gold/full_universe/quality_certificate.json` 的 `checks` 数组只有 **9 项**，
**没有** `universe_includes_delisted_names`（Round 18/19 才加），也没有 `unknown_checks` 键。
`source_commit = 5ad870b49c9e171b469b18c8bcdc016509408170`，`generated = 2026-07-29T05:16:52Z`。
⇒ 现役证书是 **2026-07-29 的旧版脚本**签发的，不是当前 `main` 的检查集签发的。
（这不改变 §2.1 结论 —— 新增那一项也与特征广度无关。）

### 2.4 精确表述：这张证书认证了什么、没认证什么

证书自己的 `scope_note` 已经写得很准确（`quality_certificate.json` 末尾）：

> "Structural readiness only. It permits full-universe training; it does NOT permit
> formal research claims, which require FULL_UNIVERSE_RESEARCH_READY."

`readiness_tiers.py:63-66` 的权限表同样明确：
`FULL_UNIVERSE_GOLD_READY.allows = ("full_universe_training", "feature_analysis")`，
`forbids = ("performance_claims", "model_promotion", "paper_portfolio_operation")`。

**精确表述 / Precise statement**：
`FULL_UNIVERSE_GOLD_READY` 认证的是**"这个 parquet 的行是干净且可复现的"** ——
行唯一、不越生命期、复权口径唯一、价量取值合法、标签与掩码列**存在**、
血缘与哈希被记录。它**完全没有**认证：
(a) 面板里有多少特征、属于哪些因子族；
(b) 这些特征是否有 alpha；
(c) 训练出来的模型是否可用（这被 `forbids: performance_claims / model_promotion` 显式排除）。

**裁决**：证书本身**没有说谎**（`PASS`，措辞与权限表都准确）。
问题在于**这是全仓唯一被称作"认证/certified"的数据产物**，
而"certified"这个词在操作员眼里天然读作"这就是可以用来训练的那份数据"。
`allows = full_universe_training` 这一句更强化了这个读法。
⇒ **FIND-R3-01 的准确形状不是"证书造假"，而是"唯一被认证的训练面板在构造上就只有 15 个基础价量特征"**。

### 2.5 15 个特征是**故意**的，不是遗漏 —— 有代码内的书面意图

`scripts/build_u0_full_universe_gold.py:99-107` `build_features()` 的 docstring 原文：

> "Deliberately a small, transparent set built only from prices and volume the panel
> already carries. **This mission validates the frozen model surface on the real
> universe; it is not a factor search, so no new signal families are introduced here.**"

⇒ 这份面板的**设计目的是"在真实全宇宙上验证已冻结的模型接口"**，不是"生产研究面板"。
它被当成后者使用（因为它是唯一持证的），是**用途漂移**而非实现缺陷。

---

## §1 两个面板的关系（Q1）— **实测：不是子集关系，是两条互不相干的数据底座**

### 1.1 实测口径（复现命令见 §8-A / §8-B）

| 维度 | 认证 gold `runtime/data/gold/full_universe/dataset.parquet` | 丰富面板 `…_v89_plus7clean_fund.parquet` |
|---|---|---|
| 行数 | **10,917,401** | **6,781,038** |
| 符号数 | **5,790** | **3,638** |
| 交易日数 | **2,563** | **2,023** |
| 日期范围 | 2016-01-04 → 2026-07-23 | **2018-01-02 → 2026-05-13** |
| 板块构成 | SH_Main 3,868,049 / SZ_Main 3,592,368 / ChiNext 2,493,817 / **STAR 706,743** / **BSE 256,424** | SH_Main 1,480 符号 / SZ_Main 1,338 / ChiNext 820 —— **STAR 0、BSE 0、B 股 0** |
| 数据底座 | U0 A 股底座：`runtime/data/u0/panel/daily_bars_raw.parquet` + `security_master.parquet` + `pit/{adjust_factors,suspension_intervals,st_intervals}` | v7 silver：`runtime/data/v7/silver/market_panel/market_panel.parquet`（`vendor="qlib+akshare"`，`source="akshare_live_provider:multi_source"`） |
| 复权 | **hfq**（后复权，`manifest.json:adjustment_method`） | **qfq**（前复权，`market_panel.json:extra.adjust`，`as_of_date=2026-05-18`） |
| 有 `DataManifest`？ | 是（`manifest.json` + `lineage.json` + `quality_certificate.json`） | **否**（只有 `*.feature_schema.json`；`runtime/data/v7/manifests/training_dataset.json` 指向的是另一个产物 `full_universe.batch_000.parquet`，2,808,837 行 / 197 列） |

### 1.2 产出流水线（实测溯源）

丰富面板的链条（每一步都在仓库里有对应脚本/文档）：

```
qlib cn_data (calendar 到 2020-09-25) + akshare qfq 回补 2020-09-28→2026-05-18
   → runtime/data/v7/silver/market_panel/market_panel.parquet
      15,105,783 行 / 13 列 / 3,872 符号 / 1999-11-10→2026-05-18
      manifest: runtime/data/v7/manifests/market_panel.json
      quality_status = "warning"（warnings 全是 akshare_east_money_failed:*，见 §1.4）
   → alpha181 因子（runtime/data/v7/silver/factors/alpha181_full.parquet，101+80）
   → training_dataset_alpha181_exec_v87 → v88 → v88_rankfix → v89
   → v89_plus7clean（+7 个 LLM 合成因子，configs/production_blend.json 锁 sha256 272e4736…）
   → v89_plus7clean_fund（+23 估值/基本面列）
      by scripts/merge_valuation_fundamental_into_training.py
      block = runtime/data/v7/silver/valuation/val_fund_features.parquet
      该脚本**逐 row-group 断言行数不变**（"row count invariant is asserted per chunk
      and in aggregate"，脚本 docstring），所以 335→348 列这一步**没有 dropna**。
```

认证 gold 的链条：

```
runtime/data/u0/panel/daily_bars_raw.parquet (raw, U0 ashare adapter)
   → scripts/build_u0_full_universe_gold.py --start-date 2016-01-01
   → hfq 复权一次 → build_features() 的硬编码 15 列 → 3 个 forward_return 标签
   → 5 个 mask → 10 折 purged walk-forward → quality_certificate.json
```

⇒ **两条链的第一块砖就不同**（U0 raw vs qlib+akshare qfq），
**没有任何代码把一条链的因子接到另一条链的面板上**。

### 1.3 行数差 4,136,363 的**实测**分解（不是猜测）

把认证 gold 依次施加 v7 的日期窗与 v7 的符号集：

| 步骤 | 行数 | 相对上一步减少 |
|---|---|---|
| 认证 gold 全量 | 10,917,401 | — |
| 只限 v7 日期窗 `[2018-01-02, 2026-05-13]` | **9,240,580** | −1,676,821（−15.4%，**日期更短**） |
| 再限 v7 符号集（3,638 个） | **7,000,574** | −2,240,006（−20.5%，**宇宙更小：STAR/BSE/B 股/新股全无**） |
| v7 实际行数 | 6,781,038 | −219,536（**−3.1% 残差**） |

**裁决**：行数差的 **94.7% 来自"日期更短 + 宇宙更小"**（1,676,821 + 2,240,006 = 3,916,827 / 4,136,363），
**不是** dropna 掉基本面缺失行。剩下 3.1%（219,536 行）是两条链上游
（供应商覆盖、`failed_symbols`、停牌/标签可行性口径）的差异，**不是** `_fund` 那一步造成的
——`merge_valuation_fundamental_into_training.py` 显式断言行数不变。

符号集近乎单向包含：**v7 的 3,638 个符号里只有 1 个（`300114.SZ`）不在认证 gold 里**；
认证 gold 多出 2,153 个（ChiNext 619、**STAR 613**、**BSE 328**、SH_Main 316、SZ_Main 259、
B 股 200xxx 11 + 900xxx 5、`302xxx` 1、`689xxx` 1）。

### 1.4 顺带确认的两个质量事实

1. **丰富面板没有 `DataManifest`** —— 违反 `AGENTS.md`「所有 silver / gold artifact 必须
   伴随一份 `<lake_root>/manifests/<dataset>.json`」。`PRODUCTION_REPRODUCIBILITY_AUDIT.md:37`
   已记录「plus7clean 的 build 命令未见 manifest 记录 [TO-VERIFY Phase 3]」——**本轮实测确认该缺口仍在**。
   → **FIND-R3b-04 / P1**。
2. **丰富面板是 qfq（前复权，as-of 2026-05-18）**。前复权价位序列会随未来除权事件被整体重述，
   **价格水平量（如 `close`、`pb` 的分母、任何以绝对价位为输入的因子）在历史点上不是 PIT 的**；
   收益率类不受影响。认证 gold 用 hfq（后复权），历史点位不被未来重述。
   **两个面板的复权口径不同 ⇒ 因子值不可直接互换**，这是接入方案必须处理的第一个真问题（见 §3）。

---

## §3 能否把丰富因子接到全宇宙（Q3）— **能，且计算成本很小；阻塞项只有一条是真的**

### 3.1 阻塞项逐条实测

| # | 候选阻塞项 | 实测 | 裁决 |
|---|---|---|---|
| B1 | 认证 gold 缺 Alpha101/GTJA191 需要的输入列 | gold schema 含 `open/high/low/close/volume/amount` 全部 6 列（`vwap = amount/volume` 可导出） | **不是阻塞** |
| B2 | 计算成本太高 | **实测全panel 一次算完 Alpha101 = 165.3 s**（见 §3.3） | **不是阻塞** |
| B3 | 内存不够 | 实测 peak RSS **20.79 GB**（本机 62 GB / 可用 55 GB） | **不是阻塞** |
| B4 | 基本面数据覆盖不到全宇宙 | 3,654 / 5,790 = **63.1%** 符号；行级 79.38% | **真阻塞（部分）**，见 §3.2 |
| B5 | 复权口径不同（gold=hfq / v7=qfq） | 因子必须**在 gold 自己的 hfq 面板上重算**，不能从 v7 面板搬列 | **真阻塞（方法论）**，见 §3.4 |
| B6 | 事件类因子无 builder | 见 §5 | **真阻塞（要写代码）** |
| B7 | L2 / 逐笔订单流 | A 股零公开供应商（Round 21 已裁） | **BLOCKED_BY_DATA**（与本题无关，不影响 165 个价量因子接入） |

### 3.2 基本面覆盖：**实测 63.1% 符号 / 79.38% 行，且是"近年更差"不是"早年更差"**

来源：`runtime/data/v7/silver/fundamentals/metrics_panel.parquet`
（257,128 行、17 列、3,654 符号，`manifest.json` 记 `source="tickflow.financials"`，
`available_at_rule = "announce_date + 1d"`）
与 `runtime/data/v7/silver/valuation/val_fund_quarterly.parquet`（257,128 行、3,654 符号）。

**符号级覆盖**：认证 gold 的 5,790 个符号中 **3,654 个有基本面（63.1%）**，
**2,136 个没有**，按板块分解：

| 板块 | 无基本面符号数 |
|---|---|
| **STAR（科创板 688）** | **613**（= gold 里全部 STAR） |
| ChiNext | 609 |
| **BSE（北交所）** | **328**（= gold 里全部 BSE） |
| SH_Main | 322 |
| SZ_Main | 246 |
| B 股 200xxx / 900xxx | 16 |
| 其他（302/689 各 1） | 2 |

**行级覆盖随时间的实测曲线**（判据 = 该 symbol 至少有一条 `available_at <= trade_date` 的财报）：

| 年 | gold 行数 | 有 PIT 基本面的比例 | 当年活跃符号数 |
|---|---|---|---|
| 2016 | 649,235 | **91.71%** | 3,045 |
| 2017 | 752,693 | 92.77% | 3,480 |
| 2018 | 825,708 | 93.34% | 3,585 |
| 2019 | 888,899 | **93.41%（峰值）** | 3,780 |
| 2020 | 953,177 | 90.71% | 4,205 |
| 2021 | 1,075,408 | 82.40% | 4,706 |
| 2022 | 1,173,460 | 75.23% | 5,115 |
| 2023 | 1,260,221 | 70.10% | 5,385 |
| 2024 | 1,294,831 | 68.22% | 5,436 |
| 2025 | 1,314,304 | 67.43% | 5,499 |
| 2026 | 729,465 | **66.43%（谷底）** | 5,548 |
| **合计** | 10,917,401 | **79.38%** | 5,790 |

**这与直觉相反，必须写清楚原因**：基本面覆盖**不是早年更差，而是近年更差**。
原因是 `metrics_panel` 是 **2026-06-01 一次性抓取的、宇宙被冻结在 3,658 个符号**
（`manifest.json:universe_size=3658`）的快照 —— 那个宇宙正是 v7 的 SH_Main+SZ_Main+ChiNext，
**从未包含科创板和北交所**。随着 2019 年后 STAR、2021 年后 BSE 大量上市，
分母涨而分子不涨，覆盖率就一路下滑。
⇒ **补覆盖的动作不是"回补历史"，而是"把抓取宇宙从 3,658 扩到 5,790"**。

### 3.3 计算成本：**实测，不是外推**

机器：20 核 / 62 GB RAM / `AI_quant_venv`。

小样本标定（2024-01-01→2024-06-30，626,269 行 / 5,380 符号 / 117 交易日）：

| 库 | 形态 | wall | us/row | peak RSS |
|---|---|---|---|---|
| Alpha101 | `wide=True, workers=16` | 17.3 s | 27.6 | 2.10 GB |
| GTJA-191 | 长表（默认，输出 40,081,216 行） | 44.5 s | 71.1 | 4.79 GB |

**全量实测（不是外推）—— 认证 gold 全部 10,917,401 行 × 5,790 符号一次算完**：

| 库 | 形态 | wall | us/row | peak RSS | 有限值比例 |
|---|---|---|---|---|---|
| **Alpha101（101 列）** | `wide=True, workers=16` | **165.3 s = 2.76 min** | **15.14** | **20.79 GB** | **0.7833** |
| GTJA-191（64 列） | `wide=True`（串行） | 见 §3.3.1 回填 | — | — | — |

⇒ **"把 Alpha101 接到全宇宙"的计算成本是 3 分钟和 21 GB 内存。**
这条不能再被当作阻塞项。

### 3.4 一个必须写进方案的实测陷阱：**分块计算的 warmup 必须 ≥ 250 交易日**

如果因内存顾虑而按日期分块计算（很多人的第一反应），warmup 不足会**静默产出错值**。
实测：对同一批 2024-01-01→2024-06-30 的 626,269 行，
用 ~145 个交易日 warmup（从 2023-06-01 起算）与 ~390 个交易日 warmup（从 2022-06-01 起算）
分别计算 Alpha101，比较同一 `(symbol, trade_date)` 上的值：

- **101 个因子中 58 个的最大绝对差 > 1e-9**；
- `alpha072` 最大差 **396.8**、`alpha024` **20.1**、`alpha023` **16.9**；
- `alpha052` 有 **488,007 / 626,269 = 78% 的行 NaN 模式不一致**。

根因（代码级）：Alpha101 的最长回看窗是 **250**（`alpha101.py:490` `sum(returns,250)`）、
**240**（`alpha101.py:551` `sum(returns,240)`）、**230**（`alpha101.py:458` `corr(...,230)`）、
**200**（`alpha101.py:477` `mean(close,200)`、`alpha101.py:482` `corr(...,200)`）、
**180**（`alpha101.py:581` `adv180`）。GTJA-191 同类：`gtja191.py:226` `_ts_sum(d, RET, 244)`。

**更强的一条**：GTJA-191 的 `SMA(X,n,m)`（`gtja191.py:65-67`）实现为
`ewm(alpha=m/n, adjust=False)` —— 这是**无限冲激响应**递归，**任何有限 warmup 都只是近似**。
⇒ GTJA 的 SMA 系因子应当**按符号从上市首日一次性算到底**（本机可行，见 §3.3），
不得按日期分块。

**结论**：既然全量一次过只要 2.76 min / 21 GB，**接入方案就不应该分块**。
如果未来面板增大到必须分块，warmup 下限 = **250 交易日**，且 GTJA-SMA 族必须走整段路径。

---

## §4 PIT 正确性抽查（Q4）— **PASS，有独立第三方公告日交叉验证**

### 4.1 抽查对象与设计

在 348 列面板里挑 3 个基本面字段：**`roe`**、**`revenue_yoy`**、**`net_margin`**
（`pe_ttm` 与 `eps_ttm` 一并观察，见 4.4），选定一个**具体财报公告日**，
验证公告日**之前**的因子值用的是**上一期**报表、公告日**之后**才切到本期。

标的：**`600519.SH`（贵州茅台）**，事件：**2023 年第一季度报告**。

### 4.2 三条日期的来源与一致性

| 字段 | 值 | 来源 |
|---|---|---|
| `period_end` | 2023-03-31 | `runtime/data/v7/silver/fundamentals/metrics_panel.parquet` |
| `announce_date` | **2023-04-26** | 同上（vendor = `tickflow.financials`） |
| `available_at` | **2023-04-27** | 同上（规则 `announce_date + 1d`，`fundamentals/manifest.json`） |
| **独立第三方公告日** | **2023-04-26** | **巨潮资讯 CNINFO**（`ak.stock_zh_a_disclosure_report_cninfo`，实跑成功）：<br>「贵州茅台2023年第一季度报告」公告时间 **2023-04-26**，<br>`announcementId=1216595064` |

⇒ **`announce_date` 经官方披露平台独立核对无误**（不是只信 vendor 自报）。

### 4.3 面板实测：值在 `available_at` 当天、而非公告日当天切换

复现命令见 §8-D。`600519.SH` 在 2023-04-18 → 2023-05-08 的实际面板取值：

| trade_date | `roe` | `revenue_yoy` | `net_margin` | `pb` | close |
|---|---|---|---|---|---|
| 2023-04-18 | 32.4077 | 16.5256 | 52.6795 | 10.1273 | 1592.27 |
| 2023-04-24 | 32.4077 | 16.5256 | 52.6795 | 9.8219 | 1544.26 |
| 2023-04-25 | 32.4077 | 16.5256 | 52.6795 | 9.9681 | 1567.25 |
| **2023-04-26（公告日当天）** | **32.4077** | **16.5256** | **52.6795** | 10.0461 | 1579.50 |
| **2023-04-27（available_at）** | **10.0028** | **18.6582** | **55.5393** | 9.1632 | 1592.19 |
| 2023-04-28 | 10.0028 | 18.6582 | 55.5393 | 9.1768 | 1594.55 |
| 2023-05-08 | 10.0028 | 18.6582 | 55.5393 | 8.9683 | 1558.32 |

对照 `metrics_panel` 的两期真值：

- `period_end=2022-12-31`（年报，`announce_date=2023-03-31`，`available_at=2023-04-01`）：
  `roe=32.4077`、`revenue_yoy=16.5256`、`net_margin=52.6795` ✓ 与 04-26 及之前逐位相同
- `period_end=2023-03-31`（一季报，`available_at=2023-04-27`）：
  `roe=10.0028`、`revenue_yoy=18.6582`、`net_margin=55.5393` ✓ 与 04-27 及之后逐位相同

**裁决：PASS。** 公告日之前 **0 处**用到该期财报；
更强的一点是**公告日当天也没有切换**（A 股季报常在盘后披露，用 `announce_date+1d`
是保守且正确的），切换严格发生在 `available_at`。

`pb` 的联动可算术验证：04-26 `1579.50 / bps(2022-12-31)=157.2258 = 10.0461` ✓；
04-27 `1592.19 / bps(2023-03-31)=173.7590 = 9.1632` ✓ ——
说明 `pb` 的分母也是 PIT 切换的，不是用当期 bps 回填历史。

### 4.4 面板级 PIT 复验（不是只看一个点）

随机抽 150 个符号（seed=7，实得 144 个在面板内、280,542 行），
用 `metrics/val_fund` 季度块以 `merge_asof(direction="backward", allow_exact_matches=True)`
在 `available_at` 上重建期望值，与面板实际值逐行比对：

| 字段 | 最大绝对差 | NaN 模式不一致行数 | 有限值行数 |
|---|---|---|---|
| `roe` | **0.000e+00** | **0** | 279,717 |
| `revenue_yoy` | **0.000e+00** | **0** | 280,452 |
| `net_margin` | **0.000e+00** | **0** | 280,542 |
| `debt_to_asset` | **0.000e+00** | **0** | 280,542 |
| `eps_ttm` | **0.000e+00** | **0** | 161,057 |

**`available_at > trade_date` 的行数 = 0**（280,542 行全检）。
构建脚本 `scripts/build_valuation_fundamental_features.py:171-172` 里也有同一断言
（`assert (merged["available_at"].isna() | (merged["available_at"] <= merged["trade_date"])).all()`）。

### 4.5 顺带发现的两个非泄漏问题（都朝保守方向，记为 P2）

1. **重述（restatement）被正确地按第二个 `available_at` 处理**：
   `600519.SH` 的 `period_end=2022-12-31` 在 `metrics_panel` 里出现**两次**
   （`announce_date` 2023-03-31 与 **2024-04-03**，`roe` 32.4077 → 32.4105）。
   PIT 语义正确 —— 但 `eps_ttm` 的 TTM 去累计要求**四个连续单季**，
   重述插入的重复期打断了链条 ⇒ `600519.SH` 在 2023 全年 `eps_ttm = NaN`、
   连带 **`pe_ttm = NaN`**（见 §4.3 表，04-27 之后 `pe_ttm` 为 NaN）。
   **全面板 `eps_ttm` 有限值只占 57.4%**（161,057 / 280,542，抽样口径）。
   这是 fail-closed（宁可 NaN 不编造），**不是缺陷**，但 `pe_ttm` 的实际可用覆盖
   远低于"有基本面的 79.38% 行"这个数字，接入方案必须按字段分别声明覆盖率。
2. **`missing_fundamentals` / `missing_valuation` 在有值的行上是 0** ✓
   （`merge_valuation_fundamental_into_training.py:33` 显式覆盖旧占位符），
   本轮抽查未发现陈旧占位符。

### 3.3.1 GTJA-191 全量实测回填

| 库 | 形态 | wall | us/row | peak RSS | 有限值比例 | 因子列数 |
|---|---|---|---|---|---|---|
| **GTJA-191** | `wide=True`（串行，无 workers 参数） | **220.3 s = 3.67 min** | **20.18** | **19.57 GB** | **0.9892** | **64** |

（`compute_gtja191_factors` 的默认长表形态在同一输入上会产出 **698,713,664 行**、
按 626k 行样本外推峰值 ≈ 83 GB ⇒ **必须传 `wide=True`**，这是接入脚本的硬性要求。）

### 3.3.2 两库合计的实测总成本

| 项 | 实测值 |
|---|---|
| Alpha101（101 列，16 进程） | 165.3 s |
| GTJA-191（64 列，串行） | 220.3 s |
| **合计 CPU wall** | **385.6 s ≈ 6.4 分钟** |
| 峰值内存（分两次跑，取大者） | **20.79 GB** |
| 新增因子列 | **165** |
| 新增数据量（float64 未压缩） | 165 × 10,917,401 × 8 B = **14.4 GB** |
| 落盘估计（parquet + snappy，按现有 348 列 / 6.78M 行 = 8.9 GB 的字节密度折算） | **≈ 9–12 GB** |

**⇒ 结论：把 Alpha101 + GTJA-191 接到认证全宇宙面板的边际成本是
"6.4 分钟 CPU + 21 GB 内存 + 约 10 GB 磁盘"。** 这不是阻塞。

### 3.5 可执行接入方案 / Executable ingestion plan

分四阶段，每阶段有明确产物与验收判据。**不改动现有认证 gold**（保持 hash
`10b63ba024dd7428` 可复现），而是产出一个**带自己证书的派生面板**。

#### 阶段 A（半天，纯计算，零新数据依赖）—— 165 个价量因子

- 新增 `scripts/build_u0_full_universe_factors.py`：
  读 `runtime/data/gold/full_universe/dataset.parquet` 的
  `[symbol, trade_date, open, high, low, close, volume, amount]`
  （已是 **hfq**，与 `manifest.json:adjustment_method` 一致，**不从 v7 的 qfq 面板搬列**），
  一次性调 `compute_alpha101(wide=True, workers=16)` 与
  `compute_gtja191_factors(wide=True)`，按 `(symbol, trade_date)` 内连接回原面板。
- 产物：`runtime/data/gold/full_universe_factors/dataset.parquet`
  （41 + 165 = **206 列**，10,917,401 行）+ `manifest.json` + `lineage.json`
  （必须记 `parent_dataset_hash=10b63ba024dd7428`）。
- 验收：行数**必须**仍为 10,917,401（断言）；
  `alpha101` 有限值比 ≥ 0.78、`gtja` ≥ 0.98（本轮实测基线）；
  重跑两次 `feature_hash` 逐位一致。
- **成本：6.4 分钟 CPU + 21 GB RAM + ~10 GB 磁盘（实测）**。

#### 阶段 B（1–2 天，需网络）—— 把基本面宇宙从 3,658 扩到 5,790

- 真正的阻塞不是代码而是**抓取宇宙**：`scripts/fetch_fundamentals_tickflow.py`
  当初以 3,658 个符号的 v7 宇宙运行（`fundamentals/manifest.json:universe_size=3658`），
  **STAR 613 + BSE 328 从未被抓过**。
- 动作：以 `runtime/data/u0/security_master.parquet` 的 5,790 个符号（含已退市名字）
  重跑该脚本 → 新 `metrics_panel`；再跑
  `scripts/build_valuation_fundamental_features.py`（quarterly → daily）
  以**认证 gold 的 keys** 作为 `--keys`。
- **必须 fail-closed**：Round 21 主报告 §3.2 已实测东财 `*_em` 端点从本机 100% 不可达；
  TickFlow 走另一条路，但任何抓不到的符号必须落 `BLOCKED_BY_DATA` 名单，
  **不得用行业均值/前值填充**。缺就是缺，`missing_fundamentals=1`。
- 验收：覆盖率**按板块、按年**分别发布（§3.2 的表就是基线），
  不得只报一个总数；2016–2019 已有 ~92%，目标是把 2021+ 从 66–82% 拉回同一水平。
- **成本估计**：以现有 3,654 符号耗时 3,066 s（`fundamentals/manifest.json:elapsed_seconds`）
  线性外推到 5,790 符号 ≈ **4,857 s ≈ 81 分钟**（TickFlow 侧限流可能更久）。

#### 阶段 C（半天）—— 证书语义修补，这一条**不能省**

现状的危险不在数字而在**措辞**：`FULL_UNIVERSE_GOLD_READY` 是全仓唯一
被叫作"认证"的数据产物，且其权限表写着 `allows: full_universe_training`。
在阶段 A/B 完成前，**至少要让"这份面板只有 15 个价量统计量"这件事在产物里可见**：

- 在 `quality_certificate.json` 增加一条**三态**检查
  `feature_families_declared`，产出结构如
  `{"declared": ["price_volume_basic"], "absent": ["alpha101","gtja191","fundamental","event","alternative","microstructure_l2"]}`；
  它**不阻塞**授予（证书的语义本就是结构性就绪），但让读者不必靠推断。
- 与 `fusion/search.py:172-173` 已经落地的
  `"modelClass": "rank_weighted_additive"` / `"representsInteraction": false`
  是同一条诚实性原则（Round 21 DEF-037 已按此修）。

#### 阶段 D（1 天）—— 事件类因子（见 §5 的数据依赖清单）

---

### 3.6 一条必须先答的前置问题（否则阶段 A 的产物没人用）

**认证 gold 面板当前没有任何生产训练命令在读它。** 现役生产模型
（`configs/production_blend.json`）读的是 v7 的 `plus7clean` 面板。
因此阶段 A 完成后还需要把训练入口指过去，否则只是多了一个没人消费的 parquet。
本轮**未**审计训练命令的数据集绑定关系 ⇒ 记 `unknown`，列为下一轮 R3 首要事项。

---

## §5 事件类因子 = 0（Q5）— **"声明了忘了实现"，不是"实现被删了"**；且对上一轮的一处裁决作更正

### 5.1 `earnings_revision_score`：全仓 **4 处引用，0 个 builder，0 个测试**

`grep -rn "earnings_revision" .`（排除 `runtime/`、`.git/`、本轮审计文档）**全部命中**：

| file:line | 性质 |
|---|---|
| `src/quantagent/data/v7_feature_groups.py:52` | **声明**（`MEDIUM_TERM_FEATURES` 元组成员） |
| `src/quantagent/models/v7_deep_alpha.py:50` | **消费者**（`MEDIUM_FEATURES` 元组） |
| `src/quantagent/models/v7_multi_horizon.py:115` | **消费者**（`_score_columns(...)` 参数） |

**没有第四处。** 没有任何函数计算它，没有任何测试引用它。

**git 取证**（`git log -S"earnings_revision_score" --all`，3 个历史提交）：

| commit | 加了什么 |
|---|---|
| `ae8a703` (v7.3) | `v7_multi_horizon.py` 的**消费者**那一行 |
| `c4c41e9` (v7.6) | 一处**声明** |
| `106095e` (v7.13) | 另一处**声明** |

⇒ **裁决：`earnings_revision_score` 从进入仓库的第一天起就只有消费者和声明，
从未存在过 builder。** 这是「声明了忘了实现」，**不是**「实现被删了」。

**它为什么没有炸**：`v7_multi_horizon.py:131-135` 的 `_score_columns()` 是
`[... for column in columns if column in row.index and not pd.isna(...)]` —— **缺列静默跳过**，
再对剩下的取均值。于是"5 列均值"在运行时悄悄变成"2–3 列均值"，权重被重新分配到
其余因子上，**没有任何 warning**。这是本仓反复出现的缺陷形状（缺失测量被合理默认静默替换）。
缓解因素：这两个模型文件**都带 `.. warning:: STATUS` 头**，明确写着
"NOT the production model and NOT trained"（`v7_deep_alpha.py:3-9`、`v7_multi_horizon.py:3-6`），
生产模型是 FT-Transformer sleeves ⇒ **降级为 P2**。

### 5.2 对上一轮 `03_factor.md:138-141` 的**更正**：`sector_rotation_score` **有** builder

上一轮我写「这两个声明列没有任何 builder 会生成它们」——**对 `sector_rotation_score` 这半句是错的**。
实测存在两个真实实现：

- `src/quantagent/factors/sector_rotation.py:72` `def sector_rotation_score(frame, sector_column="sector", window=20)`
  —— 对行业面板做 z-score 平均（`:89`）。
- `src/quantagent/portfolio/sector_rotation.py:83`
  —— 把行业热度 `heat["rotation_score"]` 映射回个股。

它没有出现在两个训练面板里的真正原因是**它是行业级产物、从未被并进个股训练面板**，
不是"没实现"。**更正记录在案。**

**但顺带查出一条真缺陷**：`src/quantagent/services/v7_pipeline_service.py:1097-1098`

```python
if "sector_rotation_score" not in data.columns:
    data["sector_rotation_score"] = data.get("market_attention_score", pd.Series(50.0, index=data.index)).fillna(50.0) / 100.0
```

⇒ 列缺失时**凭空造出常数 0.5**（`market_attention_score` 也缺时）。
这正是 `MEMORY` 里那条启发式点名的形状：**报出数字的路径上的 `fillna(...)` / `.get(x, 默认)` 默认有罪**。
一个恒为 0.5 的"行业轮动分数"在截面上零方差 ⇒ 对排序无贡献，但**在证据链里看起来像一个已测量的因子**。
→ **FIND-R3b-05 / P1**。

### 5.3 补实现 `earnings_revision_score` 的数据依赖清单

按 **PIT 可行性**从高到低排三档。**建议先做第 1 档**——它零新数据依赖。

#### 档 1（推荐先做）：SUE / PEAD —— **零新数据依赖，PIT 完全可行**

盈利"预期修正"的经典无分析师版本 = **标准化未预期盈利（SUE）**：
用**同比季度盈利的时间序列模型**（seasonal random walk + drift）作为"预期"，
实际公布值减去它再标准化。

| 需要的字段 | 现在有吗 | 来源 |
|---|---|---|
| 季度 `eps_basic`（YTD 累计） | **有** | `runtime/data/v7/silver/fundamentals/metrics_panel.parquet` |
| 单季去累计逻辑 | **有，已实现** | `scripts/build_valuation_fundamental_features.py:52` `ttm_from_ytd()` |
| `announce_date` | **有，且经巨潮交叉验证**（§4.2） | 同上 |
| `available_at`（= `announce_date+1d`） | **有** | 同上 |
| 事件后 N 日漂移窗口 | **有** | 认证 gold 的 `forward_return_{1,5,20}d` |

PIT 判据：SUE 在 `available_at` 当天首次可见，**与 §4 已验证的路径完全同构** ⇒ 可行。
覆盖率上限 = 基本面覆盖率（§3.2：符号 63.1%、行 79.38%，扩宇宙后可提升）。

#### 档 2：业绩预告 / 业绩快报（earnings pre-announcement）—— PIT 可行，需抓取

| 需要 | 来源 | 状态 |
|---|---|---|
| 预告类型（预增/预减/扭亏…）、预告净利润区间、**预告公告日** | 巨潮资讯 CNINFO（本轮实测 `ak.stock_zh_a_disclosure_report_cninfo` **可达**） | **可行** |
| 同上（东财 `stock_yjyg_em` 等） | 东方财富 | **BLOCKED_BY_NETWORK**：主报告 §3.2 实测 `*_em` 端点从本机 100% 不可达 |

预告本身自带公告日 ⇒ PIT 干净。**这是本项目实际可拿到的最"事件"的数据。**

#### 档 3：分析师一致预期修正（真正的 "earnings revision"）—— **PIT 上有实质困难**

| 需要 | 来源 | 状态 |
|---|---|---|
| 机构一致预期 EPS / 净利润的**历史时间序列**（每次修正带日期） | 东财一致预期 / 研报 | **BLOCKED_BY_NETWORK**（`*_em` 不可达） |
| 同上 | 同花顺 iwencai / 研报 | `unknown` —— 本轮未实测 |
| 同上 | TickFlow | `unknown`，见下方风险 |

**PIT 上的实质困难**：绝大多数公开一致预期接口返回的是**当期快照**（今天的一致预期），
没有"某历史日的一致预期是多少"的带日期登记。用当期快照回填历史 = 前视，
与 Stage 10 的概念成分、Stage 8 的 `sector_map` 是同一个坑（见 `MEMORY.md`）。
⇒ **除非拿到带修正日期的历史登记，档 3 应记 `BLOCKED_BY_DATA` 而不是实现它。**

#### 阶段 B 的一条未闭合风险（记 `unknown`，不记 pass）

`runtime/data/v7/silver/fundamentals/manifest.json` 记 `source="tickflow.financials"`，
2026-06-01 成功抓了 3,654 个符号；但 `MEMORY.md` 的 TickFlow 权限复核（2026-07-12）
记「订阅**仅日线**…FORBIDDEN=…**全部财务**」。两条证据互相矛盾，
可能是权限在 6 月后被收回。本轮**未**发起实际探测（避免消耗配额、且不读 `.env`）
⇒ **阶段 B 开工前必须先跑一次 TickFlow 财务能力探测**，否则整个阶段可能一开始就 BLOCKED。

---

## §6 选股链路（Q6 / R4 席位）— 顺序**是对的**，但在认证面板上**结构性失效**

### 6.1 打分 → 排序 → top-k 的完整链路（`file:line`）

| 阶段 | 位置 | 说明 |
|---|---|---|
| ① 打分（模型预测） | `src/quantagent/cli/v7_train.py:693`、`src/quantagent/cli/v7_backtest.py:66`、`src/quantagent/paper/daily_loop.py:201` `predict_v7_alpha(...)` | 产出 `predictions[symbol, trade_date, prediction]` |
| ② 多 horizon 融合 | `src/quantagent/portfolio/multi_horizon_blender.py:163-169` | 截面 rank 后混合 |
| ③ **不可交易过滤** | **`src/quantagent/portfolio/v7_target_weights.py:314-328`** | 见 §6.2 |
| ④ 流动性上限 | `v7_target_weights.py:334-360` | 按 `amount × participation` 定单票上限 |
| ⑤ **排序 + top-k 选择** | **`v7_target_weights.py:399-418`**（`ai_threshold` 或 `nlargest`） | `selection_mode` 默认 `ai_threshold` |
| ⑥ 动态 top-k | `v7_target_weights.py:432-440` + `portfolio/dynamic_top_k.py:99` | 默认关闭（`dynamic_top_k_enabled=False`） |
| ⑦ softmax / 权重化 → 单票+行业上限投影 → 换手上限 → 归一 | `v7_target_weights.py:420-700` | 模块 docstring `:16-18` 记录该顺序 |
| ⑧ 持仓带 / 时序门 | `portfolio/hold_band.py:140`、`portfolio/timing_gate.py` | 减少无谓换手 |

### 6.2 剔除是在**打分前**还是**打分后** —— **在打分后、排序前**，顺序正确

`_TRADABILITY_CONSTRAINTS`（`v7_target_weights.py:89-94`）是唯一真值表：

```python
("is_suspended", "block_suspended",   "suspended"),
("is_st",        "block_st",          "st"),
("is_limit_up",  "block_limit_up_buy","limit_up_buy_block"),
("is_limit_down","block_limit_down_sell","limit_down_sell_block"),
```

过滤在 `:314-328` 执行 → `eligible = merged[keep_mask]`（`:329`）→
top-k 在 `:399-418` 只在 `eligible` 上做。

**⇒ 不可交易标的是在 top-k 选择之前被剔除的，不会占用持仓名额，
也不会造成"持仓数不足"的偏差。这一条 PASS**（符合 akquant 横截面清单"评分前剔除"的口径）。
新股 seasoning 由认证 gold 的 `mask_seasoning`（60 个交易日）在数据层面处理。

### 6.3 【FIND-R3b-01 / P0】但在认证面板上，这四条过滤器**一条都不会执行**

`v7_target_weights.py:316-317`：

```python
for column, config_attr, reason in _TRADABILITY_CONSTRAINTS:
    if column not in merged.columns:
        continue                 # <-- 列不存在 ⇒ 静默跳过，不是拒绝、不是告警
```

而 **认证全宇宙的行情面板没有这四列中的任何一列**：

| 面板 | 列数 | `is_st` | `is_suspended` | `is_limit_up` | `is_limit_down` |
|---|---|---|---|---|---|
| **`runtime/data/gold/full_universe/adjusted_market_panel.parquet`** | **10** | ✗ | ✗ | ✗ | ✗ |
| `runtime/data/v7/silver/market_panel/market_panel.parquet` | 18 | ✓ | ✓ | ✓ | ✓ |

（认证 gold 的 `dataset.parquet` 里有 `mask_is_st` / `mask_is_suspended`，
但**名字不同**，`_TRADABILITY_CONSTRAINTS` 找的是 `is_st` / `is_suspended`；
且它**根本没有涨跌停两列**。）

**而这个没有旗标的面板正是 UI 的默认值**：
- `services/quant_api/services/strategies.py:481-486` —— `marketPanelPath` 候选列表的**第一项**（默认）就是它；
- `apps/quant-ui/src/vnext/strategy/StrategyStudioPage.tsx:52` —— `DEFAULT_DRAFT.marketPanelPath` 就是它。

**四个调用点无一验证列存在**：`cli/v7_backtest.py:72`（`read_frame(market_panel_path)` 直通）、
`cli/v7_train.py:720`、`cli/v7_train.py:1500`、`paper/daily_loop.py:217`
—— `grep "is_st\|is_suspended\|is_limit_up"` 在这三个文件里 **0 命中**。

**实测复现（本轮实跑，见 §8-F）**：40 个标的、top_k=10，让 ST / 停牌 / 涨停三只票拿到最高预测值。

| 场景 | 持仓数 | ST 入选 | 停牌入选 | 涨停入选 | `diagnostics["rejected"]` |
|---|---|---|---|---|---|
| **A：无旗标面板**（= 认证 `adjusted_market_panel` 的 schema） | 10 | **True** | **True** | **True** | **0 条，reasons = NONE** |
| B：有旗标面板（= v7 silver 的 schema） | 10 | False | False | False | 6 条，reasons = `['limit_up_buy_block','st','suspended']` |

**为什么这条比"静默"更糟**：`diagnostics["config"]`（`v7_target_weights.py:705` `asdict(config)`）
仍然会写出 `block_st=True`、`block_suspended=True`、`block_limit_up_buy=True`，
而 `diagnostics["rejected"]` 是空列表。**读报告的人看到的是"开了 ST 拦截且当期没有一只 ST 被拦"**，
真相是"从来没有检查过"。`constraint_surface`（`:695-703`）只列 sector/single-name/liquidity/
cash_floor/long_short/turnover/weighting，**完全不提可交易性** ⇒ 无处可看出这件事。

**这与 Round 21 的 DEF-023 / DEF-033 是同一个缺陷形状**：缺失的测量被静默跳过，
且因为内部一致性检查照样通过而长期存活。

**修法（主角色实施，三选一，(a) 是最小充分修复）**：
- (a) `_TRADABILITY_CONSTRAINTS` 循环把 `continue` 改为记录一条
  `UnmeasuredConstraint`（沿用 DEF-033 已建立的三态机制），并在
  `diagnostics` 增加 `tradability_fully_measured: bool`；
  `paper/daily_loop.py` 这条**生产路径**要求 `fully_measured` 才允许下单，
  研究路径允许但必须把 `unmeasured` 写进产物。
- (b) `build_u0_full_universe_gold.py` 把 `adjusted_market_panel.parquet`
  的 `mask_is_st` / `mask_is_suspended` 同时以 `is_st` / `is_suspended` 落列，
  并补 `is_limit_up` / `is_limit_down`（认证 gold 有 OHLC + 板块，可按板块涨跌幅规则推导；
  注意 `MEMORY` 记录的"涨跌停板宽近似 10%"精度问题需一并处理）。
- (c) `constraint_surface` 增补 `tradability: {checked: [...], unmeasured: [...]}`。
  **(c) 是诚实性下限，不可省略。**

### 6.4 R4 其余项（本轮时间所限，如实记 `unknown`）

- 行业分类是否 PIT / 版本化：`unknown`（`MEMORY` 记 Stage 8/10 的 `sector_map` 是当期快照，
  但本轮未复验 `sector_map_for_optimization` 的当前实现）。
- `hold_band` / `dynamic_top_k` / `timing_gate` 的换手与集中度实测：`unknown`。
- 流动性上限在大资金下是否真的绑定：`unknown`（`capital_yuan` 默认值未审）。

---

## §7 Findings 汇总 / Findings

### 【FIND-R3b-01 / P0】认证行情面板缺可交易性列 ⇒ 四条 ST/停牌/涨跌停过滤器静默全部不执行

- `file:line`：`src/quantagent/portfolio/v7_target_weights.py:316-317`（`if column not in merged.columns: continue`）；
  面板：`runtime/data/gold/full_universe/adjusted_market_panel.parquet`（10 列，四个旗标全无）；
  默认值来源：`services/quant_api/services/strategies.py:483`、`apps/quant-ui/src/vnext/strategy/StrategyStudioPage.tsx:52`；
  未校验的调用点：`src/quantagent/cli/v7_backtest.py:72`、`src/quantagent/cli/v7_train.py:720`、
  `src/quantagent/cli/v7_train.py:1500`、`src/quantagent/paper/daily_loop.py:217`。
- 场景：操作员在 Strategy Studio 用默认 `marketPanelPath` 跑组合构建 →
  ST / 停牌 / 涨停股全部进入 top-k，`rejected` 为空，而 `config` 里写着 `block_st=True`。
- 复现：§8-F（实跑输出已贴，A 场景 ST/停牌/涨停三只全部入选、rejected=0）。
- 修法：§6.3 (a)+(c)。
- 严重性：**P0** —— 直接影响可实现收益，且报告主动误导（"拦截已开启且无命中"）。

### 【FIND-R3b-02 / P0】工作站两个默认入口都指向 15 特征面板，因子广度在产品层被钉死

- `file:line`：`apps/quant-ui/src/domain/jobTemplates.ts:20`
  （**训练**模板 `dataset_path = runtime/data/gold/full_universe/dataset.parquet`）、
  `apps/quant-ui/src/vnext/fusion/FusionSearchForm.tsx:32`
  （**因子融合**默认 `factorPanelPath` 同一文件）、
  `FusionSearchForm.tsx:35+178-180`（`factorNames` 默认空且是**自由文本输入、无候选列表**）。
- 场景：`jobTemplates.ts:15-19` 的注释只说明了这次切换是为了**宇宙从 3,872 扩到 5,790**
  （"a silently narrower universe is worse than a job that fails"），
  **完全没有记录同一次切换把特征从 348 列砍到 15 列**（156 个 alpha101、58 个 gtja191、
  21 个基本面、22 个宏观全部丢失）。操作员点"训练"即在 15 个基础价量统计量上训练；
  点"因子融合"则在同 15 列里搜权重，其中 3 列是均线 TA。
  ⇒ **这就是用户"系统只有 TA"的产品级机制**。
- 复现：§8-A（面板列构成）+ 上述 `file:line` 直读。
- 修法：优先按 §3.5 阶段 A 产出 206 列派生面板并把两个默认值指过去；
  在此之前，至少按阶段 C 让"只有 15 个价量统计量"在证书与 UI 上可见。
- 严重性：**P0**。

### 【FIND-R3b-03 / P1】基本面抓取宇宙冻结在 3,658，STAR/BSE 从未被抓过；覆盖率**逐年下降**

- 证据：`runtime/data/v7/silver/fundamentals/manifest.json`（`universe_size=3658`、
  `n_symbols_written=3654`、`elapsed_seconds=3066`）；
  §3.2 实测：符号覆盖 3,654/5,790 = **63.1%**，行覆盖 **79.38%**，
  2019 年 93.41% → 2026 年 **66.43%**；未覆盖里 **STAR 613 全部 + BSE 328 全部**。
- 场景：任何在认证全宇宙上做的"基本面 + 量价"研究，其基本面在科创板/北交所上恒为缺失，
  且缺失率随年份单调上升 —— 若模型对缺失做了任何均值/零填充，
  等于在近年样本上系统性地把一整类板块推向"基本面中性"。
- 修法：§3.5 阶段 B（以 `security_master.parquet` 的 5,790 符号重跑抓取，
  按板块×年份发布覆盖率，缺失 fail-closed）。
- 前置风险：TickFlow 财务权限状态 `unknown`（§5.3 末）。
- 严重性：**P1**。

### 【FIND-R3b-04 / P1】生产训练面板没有 `DataManifest`

- 证据：`runtime/data/v7/gold/training_dataset/` 下 `…_plus7clean_fund.parquet`
  只有 `*.feature_schema.json`；`runtime/data/v7/manifests/` 里唯一的
  `training_dataset.json` 指向的是 `full_universe.batch_000.parquet`（2,808,837 行 / 197 列），
  **不是**这个 6,781,038 行 / 348 列的产物。
- 违反：`AGENTS.md`「所有 silver / gold artifact 必须伴随一份
  `<lake_root>/manifests/<dataset>.json`」。`PRODUCTION_REPRODUCIBILITY_AUDIT.md:37`
  已记为 `[TO-VERIFY Phase 3]`，**本轮实测确认缺口仍在**。
- 后果：该面板的 provider、PIT violation 计数、duplicate rate、warnings 无处可查；
  §1 的宇宙/日期/复权口径全部是本轮**反推**出来的，不是产物自述的。
- 严重性：**P1**。

### 【FIND-R3b-05 / P1】`sector_rotation_score` 缺列时被填成常数 0.5

- `file:line`：`src/quantagent/services/v7_pipeline_service.py:1097-1098`。
- 场景：列缺失 → `market_attention_score` 也缺失 → 全列 = 50.0/100 = **0.5**。
  截面零方差因子对排序无贡献，但在证据链里呈现为一个"已测量的行业轮动分数"。
- 修法：改为不产出该列 + 记 `unmeasured`，由消费者显式处理缺失（禁止默认值）。
- 严重性：**P1**（命中 `MEMORY` 的既有启发式：报数路径上的 `.get(x, 默认)` 默认有罪）。

### 【FIND-R3b-06 / P2】`earnings_revision_score` 从未有过 builder

- `file:line`：声明 `src/quantagent/data/v7_feature_groups.py:52`；
  消费者 `src/quantagent/models/v7_deep_alpha.py:50`、`src/quantagent/models/v7_multi_horizon.py:115`；
  静默降级点 `src/quantagent/models/v7_multi_horizon.py:131-135`。
- git 取证：`ae8a703`(v7.3) 加消费者、`c4c41e9`(v7.6)/`106095e`(v7.13) 加声明 ⇒
  **「声明了忘了实现」，非「实现被删了」**。
- 降级理由：两个消费者文件都带 `STATUS: NOT the production model and NOT trained` 头。
- 补实现路径：§5.3 档 1（SUE / PEAD，零新数据依赖，PIT 可行）。

### 【FIND-R3b-07 / P2】按日期分块计算 Alpha101 会静默产出错值（warmup 陷阱）

- 实测：~145 交易日 warmup vs ~390 交易日 warmup，同一批 626,269 行上
  **101 个因子中 58 个最大绝对差 > 1e-9**，`alpha072` 差 396.8，
  `alpha052` **78% 的行 NaN 模式不一致**。
- 根因：`alpha101.py:490`(250)、`:551`(240)、`:458`(230)、`:477`/`:482`(200)、`:581`(180)；
  `gtja191.py:226`(244)；且 `gtja191.py:65-67` 的 `SMA` 是 IIR 递归，任何有限 warmup 都是近似。
- 缓解：全量一次过只要 6.4 分钟 / 21 GB（§3.3），**接入方案不应分块**；
  若将来必须分块，warmup 下限 = 250 交易日且 GTJA-SMA 族走整段路径。

### 本轮明确记为 `unknown`（**不记 pass**）

| 项 | 为什么 unknown |
|---|---|
| TickFlow 财务端点当前是否仍被授权 | 两条证据矛盾（`manifest.json` 2026-06-01 成功 vs `MEMORY` 2026-07-12 记 FORBIDDEN），本轮未探测 |
| 分析师一致预期历史修正是否可 PIT 获取 | 同花顺/iwencai 路径未实测；东财 `*_em` 已知不可达 |
| 行业分类是否 PIT / 版本化 | 未复验 `sector_map_for_optimization` |
| `hold_band` / `dynamic_top_k` / `timing_gate` 的实测换手与集中度 | 未跑 |
| 训练命令与数据集的绑定关系（阶段 A 产物谁来消费） | 未审 |
| 认证 gold 与 v7 面板之间残差 219,536 行（3.1%）的逐条归因 | 只做到"非 `_fund` 步骤造成"，未逐符号归因 |

---

## §8 复现命令与实测数字 / Reproduction

所有命令在 `/home/shanhefu/QuantAgent`、`main @ 057f8cf`、`AI_quant_venv/bin/python3` 下实跑。

**§8-A 两个面板的形状**
```bash
AI_quant_venv/bin/python3 -c "
import pyarrow.parquet as pq
for p in ['runtime/data/gold/full_universe/dataset.parquet',
          'runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet']:
    f=pq.ParquetFile(p); c=f.schema_arrow.names
    print(p, f.metadata.num_rows, len(c),
          'alpha:',len([x for x in c if x.startswith('alpha')]),
          'gtja:',len([x for x in c if x.startswith('gtja')]))"
# 10917401 41 alpha:0 gtja:0   /   6781038 348 alpha:156 gtja:58
```

**§8-B 行数差分解**（输出见 §1.3）：读两份 parquet 的 `[symbol, trade_date]`，
对 gold 依次施加 v7 的日期窗与符号集。
`10,917,401 → 9,240,580 → 7,000,574` vs v7 实际 `6,781,038`。

**§8-C 基本面覆盖**（输出见 §3.2）：`metrics_panel.parquet` 的
`groupby('symbol')['available_at'].min()` 映射到 gold 行，按年统计
`trade_date >= first_available_at` 的比例。总计 **8,665,828 / 10,917,401 = 79.38%**。

**§8-D PIT 抽查**
```bash
AI_quant_venv/bin/python3 -c "
import pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc
d=ds.dataset('runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet')
t=d.to_table(columns=['symbol','trade_date','close','roe','revenue_yoy','pe_ttm','eps_ttm','pb','net_margin'],
  filter=(pc.field('symbol')=='600519.SH')&(pc.field('trade_date')>=pd.Timestamp('2023-04-18'))&(pc.field('trade_date')<=pd.Timestamp('2023-05-08')))
print(t.to_pandas().sort_values('trade_date').to_string(index=False))"
# roe 32.4077 直到 2023-04-26（公告日当天），2023-04-27（available_at）才切到 10.0028
```
独立公告日核对：
```bash
AI_quant_venv/bin/python3 -c "
import akshare as ak
print(ak.stock_zh_a_disclosure_report_cninfo(symbol='600519', market='沪深京',
      keyword='第一季度报告', start_date='20230401', end_date='20230510').to_string())"
# 贵州茅台2023年第一季度报告  公告时间 2023-04-26  （CNINFO 巨潮资讯）
```

**§8-E 全量因子计算成本**（脚本 `fullbench.py`，读 gold 的 8 列 OHLCVA）
```
ALPHA101(workers=16,wide) FULL PANEL rows=10,917,401 cols=103 wall=165.3s (2.76 min) 15.14 us/row peakRSS=20.79GB  finite=0.7833
GTJA191(serial,wide)      FULL PANEL rows=10,917,401 cols=66  wall=220.3s (3.67 min) 20.18 us/row peakRSS=19.57GB  finite=0.9892
```

**§8-F 可交易性静默失效复现**（`build_v7_target_weights`，40 标的 / top_k=10）
```
A no-flag panel (gold adjusted_market_panel schema, 10 cols): held=10; S000(ST) in=True, S001(SUSP) in=True, S002(LU) in=True; rejected_entries=0; reasons: NONE
B with-flag panel (v7 silver market_panel schema, 18 cols):   held=10; S000(ST) in=False, S001(SUSP) in=False, S002(LU) in=False; rejected_entries=6; reasons: ['limit_up_buy_block','st','suspended']
```

**§8-G warmup 陷阱**（`warmup.py`）
```
warm~145d: rows=1,384,634 wall=26.1s ; warm~390d: rows=2,604,125 wall=39.1s
rows compared 626,269 factors 101 -> 58 个因子差 >1e-9；alpha072 396.8；alpha052 NaN 模式不一致 488,007 行
```

---

## §9 对 FIND-R3-01 的最终裁决 / Verdict

**认证面板能不能补上因子？——能，而且计算侧几乎没有成本。**

- 阻塞**不是**算力（实测 6.4 分钟 / 21 GB 就能算出 165 个 Alpha101+GTJA191 因子）；
- 阻塞**不是**输入列（认证 gold 的 OHLCVA 六列齐备）；
- **真阻塞只有两条**：
  1. **基本面抓取宇宙冻结在 3,658**，STAR 613 + BSE 328 从未被抓 ⇒ 覆盖 63.1% 符号 / 79.38% 行，且逐年恶化；
  2. **复权口径不同**（gold=hfq / v7=qfq）⇒ 因子必须在 gold 自己的面板上**重算**，不能搬列；
- 还有一条**不是技术阻塞而是产品事实**：工作站的训练与融合两个默认入口
  （`jobTemplates.ts:20`、`FusionSearchForm.tsx:32`）都指向这份 15 特征面板，
  **所以用户在他能看到的一切产物上得到的结论"系统只有 TA"是对的**。

**证书本身没有说谎**：`FULL_UNIVERSE_GOLD_READY` 明确写着
"Structural readiness only … does NOT permit formal research claims"，
其检查集（两条授予路径共 18 项）里**没有一项**声称检查过特征广度。
问题在于它是全仓唯一被称为"认证"的数据产物，
而 `build_features()` 的 docstring 里那句
"it is not a factor search, so no new signal families are introduced here"
——**这份面板的设计目的是验证模型接口，不是做研究**——
从未出现在证书、UI 或任何操作员看得到的地方。

---

## §10 审计环境声明 / Audit provenance

- 本报告全部实测在工作树 `main` 上进行。会话开始时 `HEAD = 057f8cf`；
  收尾时另一会话（主角色）已把 `HEAD` 推进到 **`03ed77a`**
  （"Merge: close the interior-bar NAV misalignment (P0)"），
  工作树另有 4 个 `paper/` 相关文件处于已修改未提交状态。
- **已核验本报告引用的全部文件在 `057f8cf..03ed77a` 之间零变更**：
  `git diff --stat 057f8cf 03ed77a -- src/quantagent/{portfolio,factors,data,services,models}/
  scripts/build_u0_full_universe_gold.py src/quantagent/safety/readiness_tiers.py
  apps/quant-ui/src/domain/jobTemplates.ts apps/quant-ui/src/vnext/fusion/FusionSearchForm.tsx
  apps/quant-ui/src/vnext/strategy/StrategyStudioPage.tsx services/quant_api/services/strategies.py`
  → **输出为空**。⇒ 所有 `file:line` 在两个 commit 上都成立。
- 本席位**未**修改 `src/`、`apps/`、`services/`，**未** commit / push。
  唯一写入的文件是本报告 `docs/audits/round21/03b_certified_panel.md`。
- **未**读取或打印 `.env`。唯一的网络调用是 `ak.stock_zh_a_disclosure_report_cninfo`
  （巨潮资讯公开接口，无凭证）。
- **未**使用任何 mock / synthetic 数据支撑结论；§8-F 的复现用的是**合成的最小场景**，
  且已在该节明确标注为"40 个标的的构造场景"，不作为任何覆盖率或收益结论的依据。
