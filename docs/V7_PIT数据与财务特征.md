# V7 PIT 数据与财务特征 / PIT Data and Financial Features

## 目标 / Goal

财务数据必须严格 Point-in-Time。任何新增财务字段都必须同时带：

- `report_period`
- `ann_date`
- `available_at`

缺少 `available_at` 的财务事实不能进入 strict PIT cache。

## Provider 分工 / Provider Split

- Qlib：market OHLCV、technical factors、labels、training slices、backtest base。
- TuShare / AkShare：financial statements、financial indicators、valuation fields、disclosure dates。
- Local cache：`FinancialStatementCache` 负责落盘与 PIT read。

Qlib 不负责财务事实，避免把 feature namespace 当成基本面真值。

## Qlib / 本地行情 Provider

`QlibProvider` 提供：

- `daily_ohlcv(request)`
- `health_check(request=None)`
- `validate_qlib_market_schema(frame, as_of_date=None)`

必需字段：

```text
symbol, trade_date, open, high, low, close, volume, amount, available_at
```

CLI check：

```powershell
quantagent check-qlib-v7 --provider-uri D:\qlib_data\cn_data --symbols 600519.SH --start-date 2026-05-01 --end-date 2026-05-15
```

本地 CN 数据通常需要先按 Qlib 官方流程下载或构建 provider_uri，然后在 `configs/v7.default.yaml` 中设置：

```yaml
data:
  qlib_provider_uri: D:\qlib_data\cn_data
  qlib_region: cn
```

测试中 `tests/test_v7_pit_financial.py` 的 Qlib integration 会在缺少 `qlib` 包或 `QUANTAGENT_TEST_QLIB_PROVIDER_URI` 时 skip。

## AkShare / 免费兜底 Provider

AkShare 是 free-first fallback。它现在提供：

- `AkShareFinancialProvider.health_check()`
- `akshare_financial_schema_report(frame)`
- `AkShareLiveProvider.health_check()`
- `akshare_market_schema_report(frame)`

Provider 会对 empty response、missing required columns、schema drift 发出 warning。测试覆盖 normalized schema 和 canonical column snapshot。

## Financial Cache / 财务缓存

`FinancialStatementCache` 默认根目录：

```text
data/v7/fundamentals
```

它按 statement name 存储 income、balance_sheet、cashflow、financial_indicator、disclosure_dates。读取时使用：

```python
cache.load_pit_frame("income", as_of_date="2026-05-15", symbols=("600519.SH",))
```

PIT 过滤规则是 `available_at <= as_of_date`。`available_at > as_of_date` 必须丢弃。

## Data Quality / 数据质量报告

`V7DataHub` 在返回值中加入：

```text
data_mode.quality_report
```

每张表报告：

- row count；
- missing columns；
- source；
- source reliability mean；
- duplicate rate；
- PIT violation count；
- provider warnings。

`EvidenceStore.quality_report(as_of_date)` 对 evidence partitions 也提供同样检查。
