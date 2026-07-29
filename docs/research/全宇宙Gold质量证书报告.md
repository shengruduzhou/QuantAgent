# 全宇宙 Gold 质量证书报告

- **生成时间**：2026-07-29（UTC）
- **源提交**：`c23d20bf3e7fe6f8b7e9c8f36927935368245129`
- **证书**：`runtime/data/gold/full_universe/quality_certificate.json`

## 1. 判定

```
certificate : FULL_UNIVERSE_GOLD_READY
granted     : true
dataset_hash: 10b63ba024dd7428
schema_hash : 658e926fd0b42d0f
rows        : 10,917,401
symbols     : 5,790
date_range  : 2016-01-04 .. 2026-07-23
```

**最高已授予层级由 `ENGINEERING_PIPELINE_READY` 上升至 `FULL_UNIVERSE_GOLD_READY`。**

## 2. 结构检查（全部 PASS）

| 检查 | 结果 | 实测 |
| --- | --- | --- |
| `no_duplicate_security_dates` | PASS | 0 |
| `no_pre_listing_rows` | PASS | 0 |
| `no_post_delisting_rows` | PASS | 0 |
| `adjustment_mode_declared` | PASS | 单一 `hfq` |
| `volume_non_negative` | PASS | 0 负值 |
| `close_positive` | PASS | 0 非正 |
| `labels_present` | PASS | forward_return_{1,5,20}d |
| `masks_present` | PASS | 5 个掩码列 |
| `no_infeasible_entries` | PASS | 0 |

`failed_checks = []`

## 3. 本层允许与禁止（关键）

**允许**：全宇宙训练、特征分析。

**明确禁止**：**业绩声明**、模型晋级、Paper 组合运行。

> 结构就绪 ≠ 研究就绪。Gold 证书只说明数据集**结构完整可训练**，
> 不说明基于它得出的任何数字可以用来选模型或做结论。

## 4. 随附警告（未被平滑掉）

> ST intervals are not a complete dated register (SZSE only); mask_is_st is
> UNKNOWN for exchanges without one. This dataset is therefore NOT
> point-in-time complete for ST, and FULL_UNIVERSE_RESEARCH_READY must stay
> withheld.

因此 `FULL_UNIVERSE_RESEARCH_READY` 未授予，未met 项为
`st_intervals_available`、`no_blocked_pit_fields`。

## 5. 授予过程中修正的一处真实接口错配

首次评估时 Gold 层仍报 UNMET，原因不是数据有问题，而是**接口不一致**：

- 构建器写 `dataset_hash`，而 `ReadinessEvaluator` 读 `content_hash`；
- 构建器把计数嵌在 `quality.*` 下，而评估器读证书**根层**。

处置：**让构建器满足评估器既有的契约**（评估器已合入 main 并有测试），
而不是给层级再教一套形状。否则同一个数据集会因为读取路径不同而得出不同结论。

## 6. 四层就绪现状

| 证书 | 状态 |
| --- | --- |
| `ENGINEERING_PIPELINE_READY` | ✅ 已授予 |
| `FULL_UNIVERSE_GOLD_READY` | ✅ **本次授予** |
| `FULL_UNIVERSE_RESEARCH_READY` | ❌ 阻塞于 `st_intervals` |
| `LOCAL_PAPER_READY` | ❌ 前置未满足（无经批准的研究模型） |
| `LIVE_TRADING_READY` | **刻意不实现**（`NOT_IMPLEMENTED_BY_POLICY`） |
