# Fuyao / 同花顺金融数据全能力覆盖

> 核验基准：2026-08-07 官方文档。权威入口：
> - https://fuyao.aicubes.cn/llms.txt
> - https://fuyao.aicubes.cn/llms-full.txt
> - https://fuyao.aicubes.cn/docs/
> - https://fuyao.aicubes.cn/docs/api-reference/overview/
> - https://fuyao.aicubes.cn/docs/mcp/overview/

QuantAgent 不把“能调用一个通用 URL”视为完整接入。本仓库维护机器可审计的 Fuyao capability registry，并要求每一个已发布 REST 能力都有明确的数据获取策略；官方新增/删除接口后，覆盖计数或策略映射漂移会直接使测试失败。

## 当前官方契约快照

- **31 个已开放 REST `/api/*` 数据端点**。
- **30 个 MCP tools**。MCP 是 REST 的薄包装，与 REST 共用后端 capability 和字段语义。
- **1 个 REST-only 例外**：`/api/a-share/special-data/anomaly-analysis-list` 当前没有 MCP tool。
- **3 个全市场 Parquet dump 下载端点**，路径位于 `/dump/market-dumps/*`，不是 `/api/dump/*`。
- **3 组仍为“敬请期待”的能力**：股票基础信息、指数概况/历史成分/权重、个股反查同花顺指数。

机器清单：`src/quantagent/data/fuyao_catalog.py`  
全量同步策略：`src/quantagent/data/fuyao_full_sync.py`

## 31 个 REST 能力

| 域 | 能力 | REST |
|---|---|---|
| Meta | 标的检索 | `/api/meta/tickers/search` |
| Meta | 标的列表 | `/api/meta/tickers/list` |
| A股行情 | 行情快照 | `/api/a-share/prices/snapshot` |
| A股行情 | 历史日K | `/api/a-share/prices/historical` |
| A股估值 | 估值快照 | `/api/a-share/valuations/snapshot` |
| A股除复权 | 复权因子事件 | `/api/a-share/corporate-actions/adjustment-factors` |
| A股财务 | 利润表 | `/api/a-share/financials/income-statements` |
| A股财务 | 资产负债表 | `/api/a-share/financials/balance-sheets` |
| A股财务 | 现金流量表 | `/api/a-share/financials/cash-flow-statements` |
| A股财务 | 财务指标 | `/api/a-share/financials/indicators` |
| A股基础 | 交易日历 | `/api/a-share/calendar/trading-days` |
| 同花顺功能 | 涨停股票池 | `/api/a-share/special-data/limit-up-pool` |
| 同花顺功能 | 连板天梯 | `/api/a-share/special-data/limit-up-ladder` |
| 同花顺功能 | 飙升榜 | `/api/a-share/special-data/skyrocket-list` |
| 同花顺功能 | 热股榜单 | `/api/a-share/special-data/hot-stock-list` |
| 同花顺功能 | 历史热股排行 | `/api/a-share/special-data/hot-stock-list-history` |
| 同花顺功能 | 个股热榜排名走势 | `/api/a-share/special-data/hot-stock-rank-trend` |
| 同花顺功能 | 异动原因列表 | `/api/a-share/special-data/anomaly-analysis-list` |
| 同花顺功能 | 指定个股异动原因 | `/api/a-share/special-data/anomaly-analysis-stock` |
| 同花顺功能 | 龙虎榜 | `/api/a-share/special-data/dragon-tiger-list` |
| 指数 | 同花顺指数目录 | `/api/a-share-index/catalog/ths-index-list` |
| 指数 | 当前成分股 | `/api/a-share-index/constituents/ths-stock-list` |
| 指数 | 行情快照 | `/api/a-share-index/prices/snapshot` |
| 指数 | 历史日K | `/api/a-share-index/prices/historical` |
| 基金 | 基本资料 | `/api/fund/profile/detail` |
| 基金 | 重仓股 | `/api/fund/portfolio/holdings` |
| 基金 | 净值序列 | `/api/fund/performance/nav` |
| 基金 | 区间收益 | `/api/fund/performance/returns` |
| 基金 | 持有人结构 | `/api/fund/holders/detail` |
| 基金 | ETF 行情快照 | `/api/fund/market/snapshot` |
| 基金 | ETF 历史日线 | `/api/fund/market/historical` |

## 3 个全市场 dump

当前官方文档路径为：

- `/dump/market-dumps/daily-k/download-url`：A 股全市场约 10 年未复权日 K。
- `/dump/market-dumps/daily-k-10d/download-url`：A 股全市场最近 10 个交易日未复权日 K。
- `/dump/market-dumps/adjustment-factors/download-url`：A 股全市场复权因子事件。

下载链接是短时 S3 预签名 URL，不能持久化。QuantAgent 只保存下载后的 Parquet、hash、schema/manifest，不保存预签名 URL。

## 全量同步语义

`sync-fuyao-all` 的“all”定义为：**在官方当前公开范围、保留期和账号权限内，所有可枚举的数据类均有获取策略，任何失败都进入 manifest/report，不存在静默遗漏或 mock 数据。**

```bash
quantagent audit-fuyao-coverage \
  --output data/fuyao/coverage.json

quantagent sync-fuyao-all \
  --output-dir data/fuyao/full \
  --deep \
  --include-dumps \
  --resume \
  --allow-network
```

如果需要覆盖公募 REITs，在官方 Meta API 提供 REIT 全量枚举之前，需显式追加代码：

```bash
quantagent sync-fuyao-all \
  --output-dir data/fuyao/full \
  --extra-reits '180101.SZ,508000.SH' \
  --deep --include-dumps --resume --allow-network
```

> 上面的代码仅演示参数形态；真实 universe 必须来自用户确认的标准 thscode 或未来官方 REIT 枚举能力，不能用示例列表冒充完整 REIT universe。

## 深度模式覆盖方式

- **标的宇宙**：分页拉尽 `a-share`、`a-share-index`、`fund-otc`、`fund-etf`、`fund-lof`。
- **A股价格**：10 年全市场 raw dump + 10 日增量 dump + 全市场复权因子；当前快照另行全市场分批归档。
- **财务报表**：每只 A 股同时归档 annual / quarterly，使用官方最大 10 年日期窗口。
- **财务指标**：每只 A 股、每个 `yyyy-1..4` 报告期逐一归档；由于官方 schema 未给出披露时间，原始指标明确标记为非 PIT，不能直接进入历史训练。
- **估值**：全 A 股批量保存最新 snapshot；官方当前没有历史估值序列。
- **热榜**：day/hour 当前榜，最近一年每天历史热股榜，所有 A 股最近一年 rank trend。
- **异动**：当前全市场 REST-only anomaly list + 每 50 只股票分批 anomaly-stock。
- **龙虎榜**：交易日历覆盖范围内，每个交易日分别归档 all / org / hot_money。
- **涨停**：当前与最近一年交易日逐日分页拉取 limit-up pool；ladder 为官方固定近 30 个交易日矩阵。
- **指数**：4 类目录（industry/cn_concept/region/tszs）、所有当前指数 snapshot、当前 constituents、最多 10 年历史日 K。
- **基金**：所有可枚举 OTC / ETF / LOF 的 profile、holdings、5 年 NAV、returns、holders；ETF 额外归档当前行情与最多 5 年日线。
- **Parquet dump**：3 类均下载并校验主 schema、行数和 SHA-256。

## 不能被代码“保证”的上游边界

以下不是 QuantAgent 缺接口，而是当前官方契约本身的边界；同步报告必须显式暴露：

1. A 股历史 K 当前只支持 `1d`，单请求窗口最多 10 年。
2. A 股交易日历固定仅最近一年。
3. 热股历史、个股排名走势、龙虎榜历史最多一年。
4. 连板天梯固定近 30 个交易日。
5. 估值只有最新 snapshot，没有官方历史估值 API。
6. 基金 NAV 可请求的最长 range 是 `fyear`（5 年）。
7. 基金场内行情目前只支持 ETF；LOF、场外基金、REITs 不可用该行情接口。
8. Meta ticker-list 当前只公布 `fund-otc` / `fund-etf` / `fund-lof`，而基金业务 API 又接受 `reits`。因此官方当前缺少可证明“全 REIT universe”的枚举路径。
9. 财务指标接口没有文档化 `report_date_ms`，不能在历史训练中假定报告期末即可知。
10. 股票基础信息、历史指数成分/权重、个股反查同花顺指数仍处于“敬请期待”。
11. API Key 权限、限流、上游超时/不可用可能造成单次同步缺口；这些缺口必须出现在 `fuyao_full_sync_report.json`，不得自动填假数据。

## PIT / 量化使用规则

- 财务三表：`report_date_ms` 是历史可用性时间；`period_end_ms` 仅是会计报告期。
- 当前指数成分股绝不能回填到历史日期；官方历史成分/权重尚未发布。
- 财务指标没有披露时间时 fail closed；先与具有披露时间的正式来源 join，再进入历史训练。
- 热榜、异动、龙虎榜、涨停等是 observation data；不要在数据接入层将其改造成交易信号或胜率结论。
- raw daily dump 为未复权；复权必须使用独立 adjustment-factor 事件按 as-of 规则构造。

## API Key

只允许服务端环境变量：

```text
HITHINK_FINANCE_API_KEY=...
```

不得写入前端 bundle、日志、测试 fixture、仓库配置、同步报告或数据 manifest。
