# QuantAgent V7 PIT 数据与财务特征 / PIT Data Loop and Financial Features

## 目标 / Goal

让 V7 的财务数据真正闭环：

```text
原始来源 (TuShare Pro / AkShare)
  -> PIT provider (TuShareFinancialProvider / AkShareFinancialProvider)
  -> 本地 Parquet 缓存 (FinancialStatementCache)
  -> PIT 过滤 (available_at <= as_of_date)
  -> 财务特征工程 (build_financial_features)
  -> 投影到 V7 EvidenceRecord schema (derive_v7_financial_columns)
  -> V7DataHub.fundamentals
  -> score_fraud_risk + score_financial_statements + intrinsic valuation + long horizon factors + Deep Alpha
```

Qlib 只负责行情、技术因子、label 生成、训练数据切片、回测底座；**不**负责财务事实。

## Point-in-Time 三键 / The Three PIT Keys

每条财务记录必须同时携带：

- `report_period` — 报告期，例如 `2024-12-31`。
- `ann_date` — 公告日期，例如 `2025-03-29`。
- `available_at` — 模型/回测最早可以读到该记录的日期。默认是 `ann_date + 1 个交易日`（`available_lag_days` 可配置），用来防止盘后公告被当天的回测看见。

`apply_point_in_time_filter` 严格使用 `available_at <= as_of_date`。缺失 `available_at` 的行会被视为 *不可见*，直接丢弃。

## Provider / Cache 接口 / Provider and Cache Surface

### TuShareFinancialProvider

路径：`src/quantagent/data/providers/tushare_financial_provider.py`

提供：

- `income(request)` — 利润表
- `balance_sheet(request)` — 资产负债表
- `cashflow(request)` — 现金流量表
- `financial_indicator(request)` — 财务指标
- `disclosure_dates(request)` — 公告披露日期
- `all_statements(request)` — 一次返回上述五张表
- `merge_statements(...)` — 把多个 `ProviderResult` 合并成一张宽表

默认 `allow_network=False`，必须显式开启网络并且环境变量 `TUSHARE_TOKEN` 存在时才会实际下载。`tushare` Python 包缺失会抛 `ProviderUnavailable`。

### AkShareFinancialProvider

路径：`src/quantagent/data/providers/akshare_financial_provider.py`

只提供利润表 / 资产负债表 / 现金流量表，定位为 TuShare 的兜底通道。`source_reliability=0.72`，低于 TuShare 的 `0.85`。

### FinancialStatementCache

路径：`src/quantagent/data/providers/financial_cache.py`

- 默认根目录：`data/v7/fundamentals`
- 默认存储格式：Parquet（缺 `pyarrow` 时自动回退到 CSV）
- 支持的表：`income / balance_sheet / cashflow / financial_indicator / disclosure_dates`
- `upsert(statement, frame)` 会按 `(symbol, report_period, ann_date)` 去重，新值覆盖旧值
- `load_pit_frame(statement, as_of_date, symbols)` 会严格执行 PIT 过滤

## 财务特征工程 / Financial Feature Engineering

路径：`src/quantagent/fundamental/financial_features.py`

```python
features = build_financial_features(
    income=tushare_provider.income(request).frame,
    balance_sheet=tushare_provider.balance_sheet(request).frame,
    cashflow=tushare_provider.cashflow(request).frame,
    financial_indicator=tushare_provider.financial_indicator(request).frame,
)
features = apply_point_in_time_filter(features, trade_date="2026-05-15")
projected = derive_v7_financial_columns(features)
```

输出包含但不限于：

```text
revenue / net_income / operating_cash_flow / cogs
gross_margin / net_margin / debt_to_asset
receivables_to_revenue / inventory_to_revenue
goodwill_ratio / rd_intensity
ocf_to_profit / fcf_yield
revenue_growth / profit_growth / ocf_growth / gross_margin_change
pe_ttm / pb / ps_ttm / ev_ebitda / peg
market_cap / free_float_market_cap
```

复合分数（Beneish / Piotroski / Altman / 综合 fraud risk score）由下游 `score_fraud_risk` / `score_financial_statements` 计算，特征层只做纯数据变换。

## V7DataHub 集成 / V7DataHub Integration

`configs/v7.default.yaml`：

```yaml
data:
  provider_mode: strict_local
  required_tables: [policies, base_universe, market_state, market_panel, fundamentals]
  enforce_pit_fundamentals: true
  use_financial_cache: true
  fundamentals_root: data/v7/fundamentals

fundamentals_pipeline:
  primary_provider: tushare
  fallback_provider: akshare
  available_lag_days: 1
  cache_format: parquet
  growth_lookback_periods: 1
  cap_extreme_quantile: 0.99
```

`V7DataHub.load` 行为：

1. 从 `LocalV7ResearchProvider` 读 CSV bundle。
2. 如果启用 `use_financial_cache`，再从 `FinancialStatementCache` 读 PIT 财报，跑 `build_financial_features` + `derive_v7_financial_columns`，把结果合并/覆盖到 `bundle.fundamentals`。
3. `strict_local` 模式下任何 required table 为空都会抛 `V7DataQualityError`，禁止 synthetic fallback。
4. `enforce_pit_fundamentals=true` 时把 `available_at > as_of_date` 的行丢弃，并在 warnings 里报告丢弃条数。

## CLI / 命令行

```powershell
# 拉取财务报表到本地 PIT 缓存
quantagent build-fundamentals-v7 \
  --symbols 600519.SH,000858.SZ \
  --start-date 2018-01-01 \
  --end-date 2026-05-15 \
  --provider tushare \
  --allow-network

# 切换到 AkShare 兜底
quantagent build-fundamentals-v7 \
  --symbols 600519.SH \
  --start-date 2018-01-01 \
  --end-date 2026-05-15 \
  --provider akshare \
  --allow-network
```

## 设计约束 / Design Constraints

- 永远不要把 `report_period` 当成 `available_at`，那会让 Q4 财报在 12-31 当天就被回测看到。
- 永远不要从 Qlib 拉财务数据：Qlib `feature` 的命名空间被预留给行情和派生技术特征。
- Provider 缺失时（无 token、无 akshare 包、无网络）必须显式抛 `ProviderUnavailable`，不可静默回退到 mock。
- Cache 的 `upsert` 不会自动补全缺失字段；只会写入 provider 实际返回的列。

## 后续工作 / Follow-Ups

- 接入 `disclosure_date` 后构造真正的 `available_at`（基于交易日历），替换当前的 `ann_date + N` 简化近似。
- 增加业绩快报 (`express`)、业绩预告 (`forecast`)、主营业务构成 (`fina_mainbz`) provider。
- 把 fraud-risk 字段（审计意见、监管处罚、关联交易、商誉减值、股权质押）作为独立 evidence stream 接入 EvidenceRecord。
