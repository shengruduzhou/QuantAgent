# U0 到 Gold 全链路报告

- **生成时间**：2026-07-29（UTC）
- **源提交**：`c23d20bf3e7fe6f8b7e9c8f36927935368245129`
- **构建脚本**：`scripts/build_u0_full_universe_gold.py`
- **产物目录**：`runtime/data/gold/full_universe/`

## 1. 结果

| 项 | 值 |
| --- | --- |
| 行数 | **10917401** |
| 证券数 | **5790**（旧 V7 冻结队列为 3,872） |
| 日期范围 | 2016-01-04 .. 2026-07-23 |
| dataset_hash | `10b63ba024dd7428` |
| schema_hash | `658e926fd0b42d0f` |
| 特征数 | 15 |
| 标签 | forward_return_{1,5,20}d |
| 折数 | 6（purged expanding walk-forward） |
| embargo | 20 个交易日（= 最长标签期限） |

板块分布：SH_Main 3,868,049 / SZ_Main 3,592,368 / ChiNext 2,493,817 /
STAR 706,743 / BSE 256,424 —— **五个板块齐全**。

## 2. 链路

```
U0 原始日线面板 (10,924,350 行 / 5,792 只，2016 起)
+ 证券主表（上市/退市日期）
+ 复权因子（hfq）
+ 停牌区间（PIT）
+ ST 区间（PIT，深交所有据；其余 UNKNOWN）
+ 板块与日期相关的涨跌幅规则
+ IPO 60 交易日锁定（预登记规则）
+ 可执行资格
+ 15 个特征
+ delay-1 可执行标签
→ 全宇宙 Gold
```

## 3. 十项产物（全部生成）

| 文件 | 大小 |
| --- | --- |
| `manifest.json` | 参数、哈希、源提交、重建命令 |
| `dataset.parquet` | 2.27 GB |
| `adjusted_market_panel.parquet` | 453 MB |
| `eligibility.parquet` | 10.0 MB |
| `labels.parquet` | 338 MB |
| `feature_coverage.parquet` | 240 KB |
| `missingness_masks.parquet` | 11.4 MB |
| `folds.json` | 6 折 + embargo 说明 |
| `lineage.json` | 输入清单、上游判定、哈希 |
| `quality_certificate.json` | 结构检查与授予结论 |

## 4. 强制约束（代码层面，非文档承诺）

| 约束 | 实现 |
| --- | --- |
| 原始价与复权价**绝不混用** | API 无逐列复权参数；一次性对全部价格列应用 |
| 复权口径显式记录 | `adjustment_method = hfq`，写入 manifest 与每行 |
| 成交量保持**股** | 复权只作用于价格列，volume/amount 不缩放 |
| 成交额保持 **CNY** | 同上 |
| 无重复 `symbol, trade_date` | 实测 **0** |
| 无上市前行 | 实测 **0** |
| 无退市后行 | 实测 **0** |
| ST 为 PIT | 三态掩码；无据处 **UNKNOWN**，绝不填 FALSE |
| 停牌为 PIT | 来自带日期区间 |
| 涨跌幅按板块与日期 | 复用 `ashare_rules`（含 2023-04-10 注册制切换） |
| IPO 锁定 60 交易日 | **按观测到的交易日计数**，非自然日 |
| delay-1 可执行标签 | `close(t+1+h)/close(t+1)-1` |
| t+1 涨停/停牌禁止入场 | 不可行行**丢弃**，非保留 |
| 缺失值显式掩码 | `missingness_masks.parquet` 逐特征布尔 |
| 确定性重建 | 同参数两次构建 hash 一致（实测 `4ee514912f2e6d31`） |
| **绝不回落到旧队列** | 无 fallback 分支；失败即失败 |

## 5. 丢弃统计（全部有据）

```
suspended_at_t        : 1,027
suspended_at_t1       :   471
st_at_t               :     0   ← ST 未知，故无法据此排除
entry_price_missing   : 5,792   ← 每只标的序列尾部各一行
rows_dropped_total    : 6,949
```

## 6. 确定性验证

同参数（300 只 / 2022 起）两次独立构建：

```
第一次 dataset_hash = 4ee514912f2e6d31
第二次 dataset_hash = 4ee514912f2e6d31   ← 一致
```

第二次是在把 Amihud 特征从 `groupby.apply` 改为 `transform` **之后**跑的，
哈希不变即证明该性能优化**未改变行为**。

## 7. 重建命令

```bash
python scripts/build_u0_full_universe_gold.py --start-date 2016-01-01 --output runtime/data/gold/full_universe
```

退出码 0 表示 `FULL_UNIVERSE_GOLD_READY` 授予；2 表示结构检查未通过。

## 8. 诚实限制

- ST 非 PIT 完整（见 `历史ST区间闭环或区间证书报告.md`），故本数据集
  **不足以支撑正式研究结论**；
- 特征集刻意保持精简透明（15 个价量特征），**本任务不做因子挖掘**，目标是在
  真实全宇宙上验证冻结的模型面。
