# V7 PIT 数据与财务特征 / PIT Data and Financial Features

## 目标 / Goal

V7 的真实数据层必须 Point-in-Time safe。任何 financial statement row 必须带 `symbol / report_period / ann_date / available_at / source / source_reliability / raw_hash / point_in_time_valid`。缺少 `available_at` 的财务事实不能进入 strict PIT cache。

## Qlib Market Data / Qlib 行情层

Qlib 只负责 market data、technical features、labels、training slices 和 backtest base。官方 CN download command：

```powershell
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

本仓库提供 wrapper：

```powershell
quantagent download-qlib-v7 --target-dir ~/.qlib/qlib_data/cn_data --region cn
quantagent build-market-panel-v7 --provider-uri ~/.qlib/qlib_data/cn_data --symbols 600519.SH --start-date 2020-01-01 --end-date 2026-05-15
```

Market schema hard requirements：

```text
symbol, trade_date, open, high, low, close, volume, amount, available_at
```

Optional tradability columns are reported when present：`is_suspended / is_st / is_limit_up / is_limit_down`。Close-derived features 在 `v7_dataset_builder.build_market_features` 中按 next trading row 设置 `available_at`，避免 same-day close lookahead。

## AkShare Financials / AkShare 财务层

AkShare financial adapter 现在使用 A-share symbol conversion：

```text
600000.SH -> sh600000
000001.SZ -> sz000001
300750.SZ -> sz300750
688981.SH -> sh688981
```

`stock_financial_report_sina` 支持 `公告日期`，当缺失时使用 `更新日期` 作为 announcement/update date，并按 `available_lag_days` 生成 `available_at`。

```powershell
quantagent build-akshare-v7 --symbols 600519.SH,000858.SZ --start-date 2020-01-01 --end-date 2026-05-15 --allow-network
```

Provider 会 batch symbols、retry、rate limit，并为 normalized rows 写入 `raw_hash`。Raw source snapshots 后续可以接入专用 snapshot store；当前 PIT cache 保留 normalized source identity 和 row hash。

## Dataset and Labels / 训练数据与标签

`src/quantagent/data/v7_dataset_builder.py` 负责把 market features、PIT fundamentals、evidence scores、theme exposure、risk features 合成 trainable dataset。

`src/quantagent/data/v7_label_builder.py` 生成 horizons：

```text
1, 5, 20, 60, 120, 126
```

Labels 是 future outcomes，只能用于 training / validation，不能 join 到 inference frame。

## Quality Gates / 质量硬门槛

`src/quantagent/data/v7_quality_gates.py` 提供 hard gates：

- zero PIT violations。
- minimum rows per horizon。
- minimum symbol coverage。
- minimum date coverage。
- reject mock / synthetic data for production readiness。
- no unrealistic single-factor dominance。

这些 gates 是 blocking checks，不只是 report。
