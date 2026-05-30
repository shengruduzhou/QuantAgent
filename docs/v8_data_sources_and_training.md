# V8 数据源接入 + 训练 (P9 build-out)

实现 5 月 31 日讨论的四个 China 数据源（Qlib / AkShare / TuShare / BaoStock）的统一接入 + 端到端训练。

## 1. 四源覆盖矩阵 (实现状态)

| 来源 | 日线 | 分钟 | 全 A | 实现位置 | 适用场景 |
|---|---|---|---|---|---|
| **Qlib CN** | ✓ | 1m (官方下载入口) | 是 | `data/providers/qlib_provider.py` | baseline / 训练 / 快速搭建 |
| **AkShare** | ✓ | 1/5/15/30/60 | 按 sym 循环 | `data/providers/akshare_provider.py` + flow/index/macro/financial 子模块 | raw / silver 摄取层 |
| **TuShare** | ✓ | 部分 | 较完整 | `data/providers/tushare_provider.py` + tushare_financial | 财报 / 公告 / 基础面 |
| **BaoStock** | ✓ | 5/15/30/60 | 是 | `data/providers/baostock_provider.py` ← **P9.1 新增** | 免费 fallback |

> **本地 CSV 兜底**：`data/providers/local_csv_provider.py` 给 smoke / 离线场景使用，**不算 production source**。

> **QMT 本地缓存**：受券商授权限制，不在 router 直接调用范围内；通过 `execution/qmt_gateway.py` 在 live 链路单独处理。

## 2. 统一路由 `MultiSourceDataRouter`

`src/quantagent/data/router.py` — 优先级 + 失败大声 + 部分覆盖 backfill：

```python
from quantagent.data.providers.qlib_provider import QlibProvider
from quantagent.data.providers.akshare_provider import AkShareProvider
from quantagent.data.providers.baostock_provider import BaoStockProvider
from quantagent.data.providers.tushare_provider import TuShareProvider
from quantagent.data.router import build_default_router, RouterConfig

router = build_default_router(
    qlib_provider=QlibProvider(provider_uri="/path/to/qlib_cn"),
    akshare_provider=AkShareProvider(),
    baostock_provider=BaoStockProvider(),
    tushare_provider=TuShareProvider(),
    config=RouterConfig(
        daily_priority=("qlib", "akshare", "baostock", "tushare"),
        allow_mock_fallback=False,   # 生产线默认禁止合成数据
    ),
)

from quantagent.data.providers.base import ProviderRequest
res = router.daily_ohlcv(ProviderRequest(
    start_date="2024-01-01", end_date="2024-06-30",
    symbols=("600519.SH", "000001.SZ"),
))
print(res.primary_source, "rows:", len(res.frame))
```

### 路由 spec
- 全部 source 全 fail 时，默认抛 `RouterAllSourcesUnavailable`，**不静默返回 mock**。
- 主源返回部分覆盖时，按 `merge_partial_results` 在后续源里 backfill 缺失 symbols。
- 每行附 `source_name`，下游 audit 直接看出哪个源提供了那一行。
- minute 路由优先级独立：`minute_priority=("akshare", "baostock", "qlib")`。

## 3. 端到端训练命令

新 CLI **`train-v8-pipeline`** 串起规范全部 12 节：

```bash
# 完整四源 (生产)
quantagent train-v8-pipeline \
  --symbols 600519.SH,000001.SZ,600036.SH,000651.SZ,002594.SZ \
  --start-date 2022-01-01 --end-date 2024-06-30 \
  --use-qlib --qlib-uri ~/.qlib/qlib_data/cn_data \
  --use-akshare \
  --use-baostock \
  --use-tushare \
  --horizon-class short_5d \
  --top-k 10 \
  --ga-population 24 --ga-generations 10 \
  --output-dir runtime/reports/v8/pipeline/full_run

# 仅 BaoStock 免费栈
quantagent train-v8-pipeline \
  --symbols 600519.SH,000001.SZ \
  --start-date 2024-01-01 --end-date 2024-06-30 \
  --use-baostock \
  --output-dir runtime/reports/v8/pipeline/free_run

# 离线 smoke (LocalCsvProvider)
quantagent train-v8-pipeline \
  --symbols 600519.SH \
  --start-date 2024-01-01 --end-date 2024-06-30 \
  --local-csv /path/to/csv/root \
  --output-dir runtime/reports/v8/pipeline/smoke
```

### 管线 stages (`training/v8_pipeline.py`)
1. **Router fetch** — daily_ohlcv 通过路由器拉数据，写 `market_panel.parquet` + `router_diagnostics.json`
2. **Forward labels** — 按 horizon_class 选择 horizons (short = (1, 5), mid = (5, 20), long = (60, 120))，PIT 安全
3. **Factor panel** — `build_default_factor_panel`（mr_5d + mom_20d 兜底；生产可换 alpha101）
4. **Horizon bundles** — short / mid / long 三类 feature 白名单切片
5. **GA optimisation** — purged walk-forward + embargo + OOS-only validation，输出 `ga/factor_weights.json` + `walk_forward_backtest.json` + `metrics.json`
6. **Target weights** — top-K 等权 wide pivot → `target_weights.parquet`
7. **Strict backtest** — `strict_v8.py` (T+1 + 涨跌停 + 印花税 + 平方根冲击 + 滑点 + risk_events)，emit 10 个文件
8. **Daily report** — `daily_decision_report.py` Markdown

## 4. 失败模式 + 合规

- **任何环节失败大声**：路由层不静默回退到 mock；`run_v8_training_pipeline` 在 `_ensure_market_panel` 处抛 `RouterAllSourcesUnavailable` 把上游空响应暴露到 CLI。
- **PIT 守恒**：每个 provider 都为最后一根 K 线设 `available_at = trade_date + 1d`，并把这一约束传到下游 forward 标签。
- **无 LLM 接 OrderManager**：P6 的 AST 静态测试仍然 enforce。`train-v8-pipeline` 不引入新的 LLM 路径。
- **QMTGateway 仍 dry_run**：`run-paper-trading-v8` / `train-v8-pipeline` 仅写本地 audit；live submit 入口未启用。

## 5. 离线训练演示

`tests/training/test_v8_pipeline.py::test_full_pipeline_produces_backtest_and_report` 用一个内存中确定性的 `_SyntheticProvider`（注册成 router 的真实源 stub），跑通完整管线，断言：

- market_panel / forward_returns / factor_panel / target_weights 全部写入
- backtest dir 含 metrics / trades / factor_weights
- router_diagnostics 记录 primary_source 与 fallback_chain
- daily_report.md 生成成功

## 6. 真实数据接入清单

| 阶段 | 准备项 | 命令 |
|---|---|---|
| Qlib | 下载 cn_data | `python scripts/get_data.py qlib_data --region cn` |
| TuShare | TuShare token | `export TUSHARE_TOKEN=xxx` |
| AkShare | pip install akshare | (无 token) |
| BaoStock | pip install baostock | (无 token) |
| 板块 / 行业 | 构建 sector_map | `quantagent build-sector-map-v7` |
| 财报 PIT | 构建 fundamentals | `quantagent build-fundamentals-v7` |
| 训练 | 跑管线 | `quantagent train-v8-pipeline --use-qlib --use-akshare ...` |

## 7. P9 新增测试统计 (vs P8 baseline)

- P9.1 BaoStockProvider — **10 测试**（含 symbol normalisation / 空响应 / login 失败 / 5min timestamp / ST flag）
- P9.2 MultiSourceDataRouter — **14 测试**（含优先级、partial coverage merge、fail-loud、mock-fallback 开关）
- P9.3 V8TrainingPipeline — **8 测试**（含 forward returns、factor blend、top-K wide、full e2e、router diagnostics）

**最终 pytest baseline: 909 → 941 pass (+32 P9)；1 pre-existing fail 仍是 ft_transformer per-date rank loss regression（独立任务，不在 v8 spec scope）。**
