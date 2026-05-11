# V6 数据与 PIT 特征 / Data and PIT Feature

## 目标 / Goal
建立统一 provider adapter，支持真实数据 runtime、local CSV、mock fixture，并保证 point-in-time 正确性。

## 架构 / Architecture
`data/providers/base.py` 定义 MarketDataProvider、NewsDataProvider、FundamentalsProvider、MacroDataProvider、FundFlowProvider、TradingCalendarProvider、CommodityDataProvider 和 IndexDataProvider。

## 数据流 / Data Flow
provider 输出 `ProviderResult`，FeatureStore 按 `event_cutoff` 做 PIT join，生成 feature_version 指纹和 data_quality metadata。

## 关键模块 / Key Modules
`mock_provider.py` 用于单元测试；`local_csv_provider.py` 用于本地缓存；`akshare_provider.py` 和 `tushare_provider.py` 是外部 adapter skeleton。

## CLI 使用方式 / CLI
```powershell
quantagent build-features-v6 --config configs/v6.default.yaml --output-dir data/processed
```

## 配置方式 / Config
`configs/data_providers.v6.yaml` 记录 provider contract；`configs/v6.default.yaml` 中的 `data.provider` 控制使用 mock、local_csv、akshare 或 tushare。

## 安全边界 / Safety
API key 不写入代码；外部 provider 不可用时记录 warning，并 fallback 到 mock / cache，不让核心测试失败。

## 测试方式 / Testing
`tests/test_v6_data_providers_mock.py` 验证 mock provider 输出 OHLCV、news、fundamentals 和 trading calendar。

## 验收标准 / Acceptance
无网络也能构建 V6 features；真实数据缺失时 report 中必须显示 data_quality warning。

