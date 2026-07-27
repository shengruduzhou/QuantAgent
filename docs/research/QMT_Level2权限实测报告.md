# QMT Level-2 权限实测报告

- **生成时间**：2026-07-28（UTC）
- **源提交**：`9787de0727d2caa8c3720891f1af8b4af9b4017d`
- **产物**：`runtime/data/capabilities/qmt/level2_probe.json`

## 1. 结论

**未取得任何真实 Level-2 记录。判定 = `PLATFORM_UNAVAILABLE`，取得记录数 = 0。**

本机为 Linux，无 Wine／xtquant／MiniQMT；QMT 为 Windows 客户端。

**本仓库在没有真实记录的情况下，绝不声称支持 Level-2。**

## 2. 官方文档的权限要求（事实）

来源：<https://dict.thinktrader.net/nativeApi/xtdata.html>（2026-07-28 访问）

官方原文：

> **「获取lv2数据时需要数据终端有lv2数据权限」**

即 Level-2 需要**数据终端持有 lv2 权限**。这是文档陈述，不是本账户的权限证明。

## 3. 待探测的六项能力

| 能力 | period 字面量 | 若可用将产生的规范类 |
| --- | --- | --- |
| `l2quote` | `l2quote` | `LEVEL2_SNAPSHOT`（按价位聚合） |
| `l2quoteaux` | `l2quoteaux` | 辅助快照 |
| `l2order` | `l2order` | `EXCHANGE_ORDER_EVENT`（逐笔委托） |
| `l2transaction` | `l2transaction` | `EXCHANGE_TRADE_EVENT`（逐笔成交） |
| `l2transactioncount` | `l2transactioncount` | 成交笔数统计 |
| `l2orderqueue` | `l2orderqueue` | `LEVEL2_ORDER_BOOK`（委托队列） |

**当前六项全部 `PLATFORM_UNAVAILABLE`，records_retrieved = 0。**

## 4. 权限拒绝与空数据的区分

`looks_like_permission_error()` 识别权限类报错关键词
（`权限`/`未授权`/`authoriz`/`permission`/`VIP`/`lv2`/`level2` 等），
使得：

- 权限拒绝 → `PERMISSION_DENIED`；
- 空且权限未确认 → `EMPTY_UNVERIFIED`（**不可当作"无 Level-2 数据"**）。

有专门测试覆盖两种情形。

## 5. 保真度后果

在取得真实 Level-2 之前：

- 排队位置、订单老化、前方撤单、价格-时间优先级 → **不可宣称**；
- 模拟器 Level B 的 `queue_position_shares` 恒为 `None`；
- 在快照数据上请求排队成交会**抛异常**，不做静默降级。

## 6. 未决

Level-2 六项能力**全部未测量**。需 Windows 主机 + 券商 lv2 数据权限。
